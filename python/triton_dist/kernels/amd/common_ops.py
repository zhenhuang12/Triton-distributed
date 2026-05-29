################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import triton
import torch
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from typing import Optional

from triton_dist.language.extra.hip.language_extra import (
    load,
    atomic_add,
    sync_grid,
    atomic_cas,
    tid,
    __syncthreads,
    ld,
    st,
)
from hip import hip
from triton_dist.utils import (
    HIP_CHECK,
    get_shmem_backend,
    mori_shmem_barrier_all_on_stream,
    rocshmem_barrier_all_on_stream,
    MORI_SHMEM_SIGNAL_DTYPE,
    supports_p2p_native_atomic,
    get_triton_dist_world,
    get_triton_dist_local_world_size,
)

# AMD MoE/Fused EP code is structurally derived from NVIDIA. Keep the
# NVSHMEM_SIGNAL_DTYPE name as an alias to MORI_SHMEM_SIGNAL_DTYPE so that
# downstream layer code that imports NVSHMEM_SIGNAL_DTYPE keeps working without
# rename and the diff against the NVIDIA file stays minimal.
NVSHMEM_SIGNAL_DTYPE = MORI_SHMEM_SIGNAL_DTYPE


@triton.jit
def _is_cta_master():
    thread_idx_x = tid(0)
    thread_idx_y = tid(1)
    thread_idx_z = tid(2)
    return (thread_idx_x + thread_idx_y + thread_idx_z) == 0


@triton.jit
def _is_gpu_master():
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)
    return (pid_x + pid_y + pid_z) == 0


@triton.jit
def unsafe_barrier_on_this_grid(ptr):
    """ triton implementation of cooperative_group::thid_grid().sync()
    WARNING: use with care. better launch triton with launch_cooperative_grid=True to throw an explicit error instead of hang without notice.
    """
    __syncthreads()
    pid_size_x = tl.num_programs(axis=0)
    pid_size_y = tl.num_programs(axis=1)
    pid_size_z = tl.num_programs(axis=2)
    expected = pid_size_x * pid_size_y * pid_size_z
    if _is_cta_master():
        nb = tl.where(
            _is_gpu_master(),
            tl.cast(0x80000000, tl.uint32, bitcast=True) - (expected - 1),
            1,
        )
        old_arrive = atomic_add(ptr.to(tl.pointer_type(tl.uint32)), nb, scope="agent", semantic="release")
    else:
        old_arrive = tl.cast(0, tl.uint32)

    if _is_cta_master():
        current_arrive = load(ptr, semantic="acquire", scope="agent")
        while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
            current_arrive = load(ptr, semantic="acquire", scope="agent")

    __syncthreads()


@triton.jit
def load_envreg(val: tl.constexpr):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %envreg{val};",
        constraints=("=r"),
        args=[],
        dtype=(tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def s_sleep(cycles: tl.constexpr):
    """Emit AMDGCN ``s_sleep`` as a back-off inside a busy-wait spin loop.

    ``s_sleep N`` idles the wavefront for roughly ``N * 64`` clocks (``N`` in
    0..127), yielding issue slots to sibling waves instead of hammering the
    L2/XGMI path with back-to-back polling ``ld_acquire``\\ s.  The operand must
    be a compile-time immediate, so it is baked into the asm string;
    ``is_pure=False`` keeps it from being hoisted out of the loop or removed by
    DCE (the ``$0`` output is a discarded dummy).
    """
    tl.inline_asm_elementwise(
        asm=f"s_sleep {cycles}\nv_mov_b32 $0, 0",
        constraints="=v",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def load_grid_ws_abi_address():
    envreg1 = load_envreg(tl.constexpr(1))
    envreg2 = load_envreg(tl.constexpr(2))
    grid_ws_abi_address = (tl.cast(envreg1, tl.uint64) << 32) | tl.cast(envreg2, tl.uint64)
    return tl.cast(grid_ws_abi_address, tl.pointer_type(tl.uint32), bitcast=True)


@triton.jit
def cooperative_barrier_on_this_grid():
    """ triton implementation of cooperative_group::this_grid().sync()
    WARNING: use with care. better launch triton with launch_cooperative_grid=True to throw an explicit error instead of hang without notice.
    """
    sync_grid()  # use __ockl_grid_sync


@triton.jit
def barrier_on_this_grid(ptr, use_cooperative: tl.constexpr):
    if use_cooperative:
        cooperative_barrier_on_this_grid()
    else:
        unsafe_barrier_on_this_grid(ptr)


@triton.jit(do_not_specialize=["rank"])
def barrier_all_ipc_kernel_v2(rank, num_ranks, comm_buf_base_ptrs):
    if tid(0) < num_ranks:
        remote_base_ptr = tl.load(comm_buf_base_ptrs + tid(0)).to(tl.pointer_type(tl.int32))
        while atomic_cas(remote_base_ptr + rank, 0, 1, scope="system", semantic="release") != 0:
            pass

    if tid(0) < num_ranks:
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while atomic_cas(local_base_ptr + tid(0), 1, 0, scope="system", semantic="acquire") != 1:
            pass

    __syncthreads()


@triton.jit(do_not_specialize=["rank"])
def barrier_all_ipc_kernel(rank, num_ranks, comm_buf_base_ptrs):
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    __syncthreads()


@triton_dist.jit(do_not_specialize=["rank", "num_ranks"])
def barrier_all_kernel_v2(rank, num_ranks, comm_buf_ptr):
    if tid(0) < num_ranks:
        remote_base_ptr = dl.symm_at(comm_buf_ptr, tid(0))
        while atomic_cas(remote_base_ptr + rank, 0, 1, scope="system", semantic="release") != 0:
            pass

    if tid(0) < num_ranks:
        local_base_ptr = dl.symm_at(comm_buf_ptr, rank)
        while atomic_cas(local_base_ptr + tid(0), 1, 0, scope="system", semantic="acquire") != 1:
            pass

    __syncthreads()


@triton_dist.jit(do_not_specialize=["rank", "num_ranks"])
def barrier_all_kernel(rank, num_ranks, comm_buf_ptr):
    for i in range(num_ranks):
        remote_base_ptr = dl.symm_at(comm_buf_ptr, i)
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = dl.symm_at(comm_buf_ptr, rank)
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    __syncthreads()


def nvshmem_barrier_all_on_stream(stream: Optional[torch.cuda.Stream] = None):
    """Call shmem barrier on stream: mori_shmem when backend is mori_shmem, else rocshmem.

    Named ``nvshmem_*`` to mirror NVIDIA helpers so that copied EP MoE / Fused
    layer code on AMD can import this symbol with no rename.
    """
    if get_shmem_backend() == "mori_shmem":
        return mori_shmem_barrier_all_on_stream(stream)
    return rocshmem_barrier_all_on_stream(stream)


@triton_dist.jit(do_not_specialize=["local_rank", "rank", "local_world_size"])
def barrier_all_intra_node_atomic_cas_block(local_rank, rank, local_world_size, symm_flag_ptr):
    """Intra-node barrier across `local_world_size` ranks using atomic CAS over
    symmetric memory.

    NOTE: requires a P2P fabric that supports native atomics. On AMD this
    requires MI300X / MI355X with XGMI. Caller should gate on
    :func:`supports_p2p_native_atomic`.
    """
    with dl.simt_exec_region() as (thread_idx, block_size):
        local_rank_offset = rank - local_rank
        if thread_idx < local_world_size:  # thread_idx => local_rank
            remote_ptr = dl.symm_at(symm_flag_ptr + local_rank, thread_idx + local_rank_offset)
            while atomic_cas(remote_ptr, 0, 1, scope="system", semantic="release") != 0:
                pass

        if thread_idx < local_world_size:  # thread_idx => local_rank
            while (atomic_cas(symm_flag_ptr + thread_idx, 1, 0, scope="system", semantic="acquire") != 1):
                pass
        __syncthreads()


@triton_dist.jit
def _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, local_world_size, symm_flags, target_value):
    with dl.simt_exec_region() as (thread_idx, block_size):
        if thread_idx < local_world_size:  # thread_idx => local_rank
            local_rank_offset = rank - local_rank
            remote_ptr = dl.symm_at(symm_flags + local_rank, thread_idx + local_rank_offset)
            st(remote_ptr, target_value, scope="system", semantic="release")
            while ld(symm_flags + thread_idx, scope="system", semantic="acquire") != target_value:
                pass
        __syncthreads()


@triton_dist.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "target_value"])
def barrier_all_intra_node_non_atomic_block(local_rank, rank, num_ranks, symm_flags, target_value):
    """
        symm_flags is expected to:
        1. of int32 or int64 dtype
        2. has at least num_ranks * 2 elements
        3. of symmetric pointer

        symm_flags [0, num_ranks * 2) is used to sync all ranks.
    """
    tl.static_assert(symm_flags.dtype.element_ty == tl.int32 or symm_flags.dtype.element_ty == tl.int64)
    _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flags, target_value)
    # next iter
    _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flags + num_ranks, target_value)


class BarrierAllContext:
    """
    You may use this to barrier all ranks in global, or just in intra-node team.

    NOTE: shmem barrier_all is slower for intra-node only.
    """

    def __init__(self, is_intra_node):
        import os as _os
        from triton_dist.utils import shmem_create_tensor, shmem_free_tensor_sync
        self.is_intra_node = is_intra_node
        self.target_value = 1
        self._shmem_free_tensor_sync = shmem_free_tensor_sync
        if self.is_intra_node:
            # On AMD we don't have nvshmem.team_my_pe(TEAM_NODE); we derive
            # ``rank`` from the triton_dist world (set up by
            # init_rocshmem/mori/nvshmem_by_torch_process_group), and
            # ``local_world_size`` from torch's standard env var. Intra-node
            # implies world_size == local_world_size for the callers we care
            # about, but ``local_rank = rank % local_world_size`` keeps us
            # correct under non-uniform rank ordering.
            self.rank = get_triton_dist_world().rank()
            self.local_world_size = (get_triton_dist_local_world_size() or int(_os.environ.get("LOCAL_WORLD_SIZE", "0"))
                                     or int(_os.environ.get("WORLD_SIZE", "1")))
            self.local_rank = self.rank % self.local_world_size
            self.num_local_ranks = self.local_world_size
            self.symm_barrier = shmem_create_tensor((self.num_local_ranks, ), torch.int32)
            self.symm_barrier.fill_(0)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def finalize(self):
        if self.is_intra_node:
            self._shmem_free_tensor_sync(self.symm_barrier)


def barrier_all_on_stream(stream: Optional[torch.cuda.Stream] = None, *, ctx: Optional["BarrierAllContext"] = None):
    """
    NVIDIA-style barrier on stream.

    - When ``ctx`` is ``None`` or not intra-node, falls back to the global
      shmem barrier (``mori_shmem`` or ``rocshmem``).
    - When ``ctx`` is intra-node, uses a fast intra-node CAS barrier (or
      non-atomic flag barrier on fabrics without native atomics).

    ``ctx`` is keyword-only so the historical positional
    ``barrier_all_on_stream(stream)`` call sites keep working unchanged.

    barrier_all_on_stream does not support CUDAGraph.
    """
    if ctx is None or not ctx.is_intra_node:
        return nvshmem_barrier_all_on_stream(stream)

    if supports_p2p_native_atomic():
        barrier_all_intra_node_atomic_cas_block[(1, )](ctx.local_rank, ctx.rank, ctx.num_local_ranks, ctx.symm_barrier)
    else:
        barrier_all_intra_node_non_atomic_block[(1, )](ctx.local_rank, ctx.rank, ctx.num_local_ranks, ctx.symm_barrier,
                                                       ctx.target_value)
        ctx.target_value += 1


def _wait_eq_hip(signal_tensor: torch.Tensor, val: int, stream: Optional[torch.cuda.Stream] = None):
    # This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    # please refer to: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_m.html#ga9ef06d564d19ef9afc11d60d20c9c541
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        err = hip.hipStreamWaitValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            hip.hipStreamWaitValueEq,
            0xFFFFFFFF,
        )
    else:
        err = hip.hipStreamWaitValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            hip.hipStreamWaitValueEq,
            0xFFFFFFFFFFFFFFFF,
        )
    HIP_CHECK(err)


def _set_signal_hip(signal_tensor: torch.Tensor, val: int, stream: Optional[torch.cuda.Stream] = None):
    # This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    # https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_m.html#ga2520d4e1e57697edff2a85a3c03d652b
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        err = hip.hipStreamWriteValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            0,
        )
    else:
        err = hip.hipStreamWriteValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            0,
        )
    HIP_CHECK(err)
