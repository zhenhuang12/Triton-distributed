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

import torch
import triton
import triton.language as tl
from triton_dist.language.extra.hip.language_extra import tid, st, ld
from typing import Any


# ---------------------------------------------------------------------------
# AMD vectorized scalar loads / stores.
#
# The NVIDIA file uses PTX inline-asm to emit ``ld.global.v2/v4`` and
# ``st.global.v2/v4`` so that one warp can issue a single 128-bit transaction
# returning four 32-bit values. On AMD/HIP we can rely on the Triton AMD
# backend to coalesce four consecutive ``tl.load`` / ``tl.store`` calls over
# uint32 lanes into a single ``global_load_dwordx4`` / ``global_store_dwordx4``
# instruction. We therefore expose load_v2/load_v4/store_v2/store_v4 as
# ``@triton.jit`` helpers that issue 2 or 4 scalar loads/stores. The "suffix"
# argument is preserved for API parity but ignored: ``b32`` and ``u32`` are
# the only suffixes actually used by callers, and we always operate on uint32
# lanes (the same width the PTX path used internally).
# ---------------------------------------------------------------------------


@triton.jit
def load_v2(ptr, suffix: tl.constexpr):
    p = ptr.to(tl.pointer_type(tl.uint32))
    return tl.load(p + 0), tl.load(p + 1)


@triton.jit
def load_v4(ptr, suffix: tl.constexpr):
    p = ptr.to(tl.pointer_type(tl.uint32))
    return tl.load(p + 0), tl.load(p + 1), tl.load(p + 2), tl.load(p + 3)


@triton.jit
def store_v2(ptr, val0, val1, suffix: tl.constexpr):
    p = ptr.to(tl.pointer_type(tl.uint32))
    tl.store(p + 0, tl.cast(val0, tl.uint32, bitcast=True))
    tl.store(p + 1, tl.cast(val1, tl.uint32, bitcast=True))


@triton.jit
def store_v4(ptr, val0, val1, val2, val3, suffix: tl.constexpr):
    p = ptr.to(tl.pointer_type(tl.uint32))
    tl.store(p + 0, tl.cast(val0, tl.uint32, bitcast=True))
    tl.store(p + 1, tl.cast(val1, tl.uint32, bitcast=True))
    tl.store(p + 2, tl.cast(val2, tl.uint32, bitcast=True))
    tl.store(p + 3, tl.cast(val3, tl.uint32, bitcast=True))


# ---------------------------------------------------------------------------
# Pure-Triton replacements for the PTX ``zero_vec_f32`` / ``cvt.rn.bf16x2.f32``
# / ``cvt.f32.bf16`` paths used by the EP MoE fused kernels.
# ---------------------------------------------------------------------------


@triton.jit
def zero_vec_f32(vec_size: tl.constexpr):
    """Return ``vec_size`` zeroed f32 scalars as a tuple, matching the
    NVIDIA PTX ``mov.b32 $i, 0`` flavor of ``zero_vec_f32``.

    NOTE: callers in ``ep_all2all_fused.py`` always unpack 8 scalars
    (``acc1 .. acc8``) regardless of ``VEC_SIZE`` because the NVIDIA inline
    asm version returned a fixed-width tuple.  Triton's tuple-return arity
    is fixed per kernel-instantiation, so we mirror the NVIDIA behavior and
    always return 8 scalars; the extra ones simply remain unused when
    ``VEC_SIZE < 8`` (this matches the PTX ``mov.b32 $i,0;`` lanes that
    NVIDIA also emits unconditionally for the 8-vector case).
    """
    z: tl.constexpr = tl.cast(0, tl.float32)
    return z, z, z, z, z, z, z, z


@triton.jit
def unpack_bf16x2_f32(v1, v2, v3, v4):
    """Unpack four ``int32``s (each containing two ``bf16`` lanes) into eight
    ``float32`` scalars.

    NVIDIA emits this with ``mov.b32 {b0, b1}, $i`` followed by ``cvt.f32.bf16``.
    On AMD we bitcast the int32 to a 2-wide ``bfloat16`` tensor, then convert
    to fp32; both operations are supported natively by the Triton AMD backend.
    """
    a = tl.cast(v1, tl.uint32, bitcast=True)
    b = tl.cast(v2, tl.uint32, bitcast=True)
    c = tl.cast(v3, tl.uint32, bitcast=True)
    d = tl.cast(v4, tl.uint32, bitcast=True)
    # Use 16-bit extracts then bitcast each half to bf16, finally widen to f32.
    a0 = tl.cast(a & 0xFFFF, tl.uint16).to(tl.uint16, bitcast=True)
    a1 = tl.cast(a >> 16, tl.uint16).to(tl.uint16, bitcast=True)
    b0 = tl.cast(b & 0xFFFF, tl.uint16).to(tl.uint16, bitcast=True)
    b1 = tl.cast(b >> 16, tl.uint16).to(tl.uint16, bitcast=True)
    c0 = tl.cast(c & 0xFFFF, tl.uint16).to(tl.uint16, bitcast=True)
    c1 = tl.cast(c >> 16, tl.uint16).to(tl.uint16, bitcast=True)
    d0 = tl.cast(d & 0xFFFF, tl.uint16).to(tl.uint16, bitcast=True)
    d1 = tl.cast(d >> 16, tl.uint16).to(tl.uint16, bitcast=True)
    f0 = tl.cast(a0, tl.bfloat16, bitcast=True).to(tl.float32)
    f1 = tl.cast(a1, tl.bfloat16, bitcast=True).to(tl.float32)
    f2 = tl.cast(b0, tl.bfloat16, bitcast=True).to(tl.float32)
    f3 = tl.cast(b1, tl.bfloat16, bitcast=True).to(tl.float32)
    f4 = tl.cast(c0, tl.bfloat16, bitcast=True).to(tl.float32)
    f5 = tl.cast(c1, tl.bfloat16, bitcast=True).to(tl.float32)
    f6 = tl.cast(d0, tl.bfloat16, bitcast=True).to(tl.float32)
    f7 = tl.cast(d1, tl.bfloat16, bitcast=True).to(tl.float32)
    return f0, f1, f2, f3, f4, f5, f6, f7


@triton.jit
def pack_f32_bf16x2(vec):
    """Inverse of :func:`unpack_bf16x2_f32`: pack 8 fp32 scalars into 4 int32
    values each holding two bf16 lanes.

    The NVIDIA path uses ``cvt.rn.bf16x2.f32`` which rounds-to-nearest. AMD
    relies on the standard ``to(tl.bfloat16)`` cast which also rounds-to-nearest.
    """
    v1, v2, v3, v4, v5, v6, v7, v8 = vec
    h0 = tl.cast(v1, tl.bfloat16).to(tl.uint16, bitcast=True)
    h1 = tl.cast(v2, tl.bfloat16).to(tl.uint16, bitcast=True)
    h2 = tl.cast(v3, tl.bfloat16).to(tl.uint16, bitcast=True)
    h3 = tl.cast(v4, tl.bfloat16).to(tl.uint16, bitcast=True)
    h4 = tl.cast(v5, tl.bfloat16).to(tl.uint16, bitcast=True)
    h5 = tl.cast(v6, tl.bfloat16).to(tl.uint16, bitcast=True)
    h6 = tl.cast(v7, tl.bfloat16).to(tl.uint16, bitcast=True)
    h7 = tl.cast(v8, tl.bfloat16).to(tl.uint16, bitcast=True)
    p0 = (tl.cast(h1, tl.uint32) << 16) | tl.cast(h0, tl.uint32)
    p1 = (tl.cast(h3, tl.uint32) << 16) | tl.cast(h2, tl.uint32)
    p2 = (tl.cast(h5, tl.uint32) << 16) | tl.cast(h4, tl.uint32)
    p3 = (tl.cast(h7, tl.uint32) << 16) | tl.cast(h6, tl.uint32)
    return p0, p1, p2, p3


@triton.jit
def copy_warp(
    dst_ptr,
    src_ptr,
    nbytes,
):
    """Wavefront-cooperative byte copy.

    AMD wavefronts are 64 lanes wide, so we use ``WARP_SIZE = 64`` instead
    of NVIDIA's 32. The 128-bit / 64-bit / 32-bit / 16-bit / 8-bit chunked
    structure of the NVIDIA helper is preserved one-for-one.
    """
    WARP_SIZE: tl.constexpr = tl.constexpr(64)
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE

    src_ptr = tl.cast(src_ptr, dtype=tl.pointer_type(tl.uint8), bitcast=True)
    dst_ptr = tl.cast(dst_ptr, dtype=tl.pointer_type(tl.uint8), bitcast=True)

    for vec_idx in range(lane_idx, nbytes // 16, WARP_SIZE):
        t1, t2, t3, t4 = load_v4(src_ptr + vec_idx * 16, tl.constexpr("b32"))
        store_v4(dst_ptr + vec_idx * 16, t1, t2, t3, t4, tl.constexpr("b32"))

    src_ptr = src_ptr + nbytes // 16 * 16
    dst_ptr = dst_ptr + nbytes // 16 * 16
    nbytes = nbytes % 16

    if nbytes != 0:
        for vec_idx in range(lane_idx, nbytes // 8, WARP_SIZE):
            t1, t2 = load_v2(src_ptr + vec_idx * 8, tl.constexpr("b32"))
            store_v2(dst_ptr + vec_idx * 8, t1, t2, tl.constexpr("b32"))
        src_ptr = src_ptr + nbytes // 8 * 8
        dst_ptr = dst_ptr + nbytes // 8 * 8
        nbytes = nbytes % 8

    if nbytes != 0:
        for vec_idx in range(lane_idx, nbytes // 4, WARP_SIZE):
            t = ld(src_ptr.to(tl.pointer_type(tl.uint32)) + vec_idx)
            st(dst_ptr.to(tl.pointer_type(tl.uint32)) + vec_idx, t)
        src_ptr = src_ptr + nbytes // 4 * 4
        dst_ptr = dst_ptr + nbytes // 4 * 4
        nbytes = nbytes % 4

    if nbytes != 0:
        for vec_idx in range(lane_idx, nbytes // 2, WARP_SIZE):
            t = ld(src_ptr.to(tl.pointer_type(tl.uint16)) + vec_idx)
            st(dst_ptr.to(tl.pointer_type(tl.uint16)) + vec_idx, t)
        src_ptr = src_ptr + nbytes // 2 * 2
        dst_ptr = dst_ptr + nbytes // 2 * 2
        nbytes = nbytes % 2

    if nbytes != 0:
        t = ld(src_ptr.to(tl.pointer_type(tl.uint8)))
        st(dst_ptr.to(tl.pointer_type(tl.uint8)), t)


@triton.jit(do_not_specialize=["nelems"])
def copy_1d_tilewise_kernel(dst_ptr, src_ptr,  #
                            nelems,  #
                            BLOCK_SIZE: tl.constexpr,  #
                            ):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = nelems // BLOCK_SIZE

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        data = tl.load(src_ptr + offs, volatile=True)
        tl.store(dst_ptr + offs, data)

    if nelems % BLOCK_SIZE:
        if pid == NUM_COPY_SMS - 1:
            offs = num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < nelems
            data = tl.load(src_ptr + offs, mask=mask, volatile=True)
            tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit(do_not_specialize=["nelems"])
def copy_1d_persistent_kernel(dst_ptr, src_ptr,  #
                              nelems,  #
                              BLOCK_SIZE: tl.constexpr,  #
                              ):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = nelems // BLOCK_SIZE
    elem_size: tl.constexpr = tl.constexpr(dst_ptr.dtype.element_ty.primitive_bitwidth) // 8
    vec_size: tl.constexpr = tl.constexpr(16 // elem_size)

    if BLOCK_SIZE >= vec_size:
        tl.static_assert(BLOCK_SIZE % vec_size == 0, "BLOCK_SIZE must be divisible by vec_size")
        BLOCK_VEC: tl.constexpr = tl.constexpr(BLOCK_SIZE // vec_size)

        for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
            offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_VEC) * vec_size
            v0, v1, v2, v3 = load_v4((src_ptr + offs).to(tl.pointer_type(tl.uint32)), suffix=tl.constexpr("u32"))
            store_v4((dst_ptr + offs).to(tl.pointer_type(tl.uint32)), v0, v1, v2, v3, suffix=tl.constexpr("u32"))

    else:
        for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
            offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            data = tl.load(src_ptr + offs)
            tl.store(dst_ptr + offs, data)

    if nelems % BLOCK_SIZE:
        if pid == 0:
            offs = num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < nelems
            data = tl.load(src_ptr + offs, mask=mask)
            tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit(do_not_specialize=["M"])
def copy_2d_persistent_kernel(
    dst_ptr,
    src_ptr,  #
    M,  #
    N,
    stride_m,
    stride_n,
    stride_dst_m,
    stride_dst_n,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        data = tl.load(src_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask)
        tl.store(dst_ptr + offs_m[:, None] * stride_dst_m + offs_n[None, :] * stride_dst_n, data, mask=mask)


@triton.jit(do_not_specialize=["M"])
def copy_2d_kernel(
    dst_ptr,
    src_ptr,  #
    M,  #
    N: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    stride_dst_m: tl.constexpr,
    stride_dst_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_tiles_n
    pid_n = pid % num_tiles_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    data = tl.load(src_ptr + offs_m[:, None].to(tl.int64) * stride_m + offs_n[None, :] * stride_n, mask=mask)
    tl.store(dst_ptr + offs_m[:, None].to(tl.int64) * stride_dst_m + offs_n[None, :] * stride_dst_n, data, mask=mask)


# NOTE: ``copy_2d_tma_kernel`` is intentionally not ported.
#
# The NVIDIA path uses ``tl.make_tensor_descriptor`` which lowers to Hopper
# TMA (``cp.async.bulk.tensor``). AMD has no TMA-equivalent instruction; we
# rely on ``copy_2d_persistent_kernel`` / ``copy_2d_kernel`` for the 2D case
# instead. The :func:`copy_tensor` helper below already dispatches between
# the persistent and tilewise kernels and never reaches the TMA branch.


def copy_tensor(dst_tensor: torch.Tensor, src_tensor: torch.Tensor, num_sms: int = -1, eager=False, persistent=True):
    if dst_tensor.numel() == 0 or src_tensor.numel() == 0:
        return
    if eager:
        dst_tensor.copy_(src_tensor)
        return

    blocksizes = []
    choices = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    total_elems = 256 * 128

    assert dst_tensor.shape == src_tensor.shape, f"dst_tensor.shape: {dst_tensor.shape}, src_tensor.shape: {src_tensor.shape}"

    for dim in reversed(dst_tensor.shape):
        for i, choice in enumerate(choices):
            if total_elems % choice == 0 and dim >= choice:
                blocksizes.append(choice)
                total_elems //= choice
                break
    blocksizes = blocksizes[::-1]
    assert len(blocksizes) == dst_tensor.ndim

    if dst_tensor.ndim == 1:
        assert src_tensor.is_contiguous()
        assert dst_tensor.is_contiguous()
        BLOCK_SIZE_M = blocksizes[0]
        if persistent:
            assert num_sms > 0, "num_sms must be provided for persistent copy"
            copy_1d_persistent_kernel[(num_sms, )](dst_tensor, src_tensor, dst_tensor.shape[0], BLOCK_SIZE_M)
        else:
            grid = (triton.cdiv(dst_tensor.shape[0], BLOCK_SIZE_M), )
            copy_1d_tilewise_kernel[grid](dst_tensor, src_tensor, dst_tensor.shape[0], BLOCK_SIZE_M)
    elif dst_tensor.ndim == 2:
        BLOCK_SIZE_M = blocksizes[0]
        BLOCK_SIZE_N = blocksizes[1]
        if persistent:
            assert num_sms > 0, "num_sms must be provided for persistent copy"
            copy_2d_persistent_kernel[(num_sms, )](dst_tensor, src_tensor, dst_tensor.shape[0], dst_tensor.shape[1],
                                                   src_tensor.stride(0), src_tensor.stride(1), dst_tensor.stride(0),
                                                   dst_tensor.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)
        else:
            grid = (triton.cdiv(dst_tensor.shape[0], BLOCK_SIZE_M) * triton.cdiv(dst_tensor.shape[1], BLOCK_SIZE_N), )
            copy_2d_kernel[grid](dst_tensor, src_tensor, dst_tensor.shape[0], dst_tensor.shape[1], src_tensor.stride(0),
                                 src_tensor.stride(1), dst_tensor.stride(0), dst_tensor.stride(1), BLOCK_SIZE_M,
                                 BLOCK_SIZE_N)
    else:
        raise ValueError(f"Unsupported tensor dimension: {dst_tensor.ndim}")


@triton.jit
def fill_1d_persistent_kernel(
    dst_ptr,
    value,
    M,
    stride_m,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M)

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        offs = tile_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offs < M
        data = tl.full([BLOCK_SIZE_M], value, dtype=dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit
def fill_2d_persistent_kernel(
    dst_ptr,
    value,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        data = tl.full([BLOCK_SIZE_M, BLOCK_SIZE_N], value, dtype=dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, data, mask=mask)


@triton.jit
def fill_3d_persistent_kernel(
    dst_ptr,
    value,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for batch in range(B):
        for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
            pid_m = tile_id // num_tiles_n
            pid_n = tile_id % num_tiles_n
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask = mask_m[:, None] & mask_n[None, :]
            data = tl.full([BLOCK_SIZE_M, BLOCK_SIZE_N], value, dtype=dst_ptr.dtype.element_ty)
            tl.store(dst_ptr + batch * stride_b + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, data,
                     mask=mask)


def fill_tensor(tensor: torch.Tensor, value: Any, num_sms: int = -1, eager=False):
    if tensor.numel() == 0:
        return
    if eager:
        tensor.fill_(value)
        return

    if num_sms == -1:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    blocksizes = []
    choices = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    total_elems = 256 * 128

    for dim in reversed(tensor.shape):
        for i, choice in enumerate(choices):
            if total_elems % choice == 0 and dim >= choice:
                blocksizes.append(choice)
                total_elems //= choice
                break
    blocksizes = blocksizes[::-1]
    assert len(blocksizes) == tensor.ndim, f"blocksizes: {blocksizes}, tensor.shape: {tensor.shape}"

    if tensor.ndim == 1:
        BLOCK_SIZE = blocksizes[0]
        fill_1d_persistent_kernel[(num_sms, )](tensor, value, tensor.shape[0], tensor.stride(0), BLOCK_SIZE)
    elif tensor.ndim == 2:
        BLOCK_SIZE_M = blocksizes[0]
        BLOCK_SIZE_N = blocksizes[1]
        fill_2d_persistent_kernel[(num_sms, )](tensor, value, tensor.shape[0], tensor.shape[1], tensor.stride(0),
                                               tensor.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)
    elif tensor.ndim == 3:
        BLOCK_SIZE_M = blocksizes[1]
        BLOCK_SIZE_N = blocksizes[2]
        fill_3d_persistent_kernel[(num_sms, )](tensor, value, tensor.shape[0], tensor.shape[1], tensor.shape[2],
                                               tensor.stride(0), tensor.stride(1), tensor.stride(2), BLOCK_SIZE_M,
                                               BLOCK_SIZE_N)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")


@triton.jit
def reduce_1d_persistent_kernel(
    src_ptr,
    dst_ptr,
    reduce_dim,
    stride_reduce,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_SMS = tl.num_programs(0)
    num_tiles = tl.cdiv(reduce_dim, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, NUM_SMS):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < reduce_dim
        data = tl.load(src_ptr + offs * stride_reduce, mask=mask).to(tl.float32)
        accum = tl.sum(data).to(dst_ptr.dtype.element_ty)
        tl.atomic_add(dst_ptr, accum)


@triton.jit
def reduce_2d_persistent_kernel(
    src_ptr,
    dst_ptr,
    reduce_dim,
    M,
    stride_reduce,
    stride_m,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_SMS = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M)

    for tile_id in range(pid, num_tiles, NUM_SMS):
        offs = tile_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offs < M
        accum = tl.zeros((BLOCK_SIZE_M), dtype=tl.float32)
        for i in range(reduce_dim):
            data = tl.load(src_ptr + i * stride_reduce + offs * stride_m, mask=mask).to(tl.float32)
            accum += data
        accum = accum.to(dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs, accum, mask=mask)


@triton.jit
def reduce_3d_persistent_kernel(
    src_ptr,
    dst_ptr,
    reduce_dim,
    M,
    N,
    stride_reduce,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for tile_id in range(pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(reduce_dim):
            data = tl.load(src_ptr + i * stride_reduce + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n,
                           mask=mask).to(tl.float32)
            accum += data
        accum = accum.to(dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs_m[:, None] * N + offs_n[None, :], accum, mask=mask)


def reduce_tensor(tensor: torch.Tensor, num_sms: int, dim=0, acc_dtype=torch.float32, eager=False):
    if tensor.numel() == 0:
        return torch.empty([1], dtype=tensor.dtype, device=tensor.device).fill_(0)
    assert acc_dtype == torch.float32
    if eager:
        return tensor.to(acc_dtype).sum(dim=dim).to(tensor.dtype)

    ndim = tensor.ndim
    dim = dim % ndim
    blocksizes = []
    choices = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    total_elems = 256 * 128

    for d in reversed(tensor.shape):
        if d == dim:
            blocksizes.append(1)
            continue
        for i, choice in enumerate(choices):
            if total_elems % choice == 0 and d >= choice:
                blocksizes.append(choice)
                total_elems //= choice
                break
    blocksizes = blocksizes[::-1]
    assert len(blocksizes) == tensor.ndim

    if tensor.ndim == 1:
        reduce_dim = tensor.shape[dim]
        stride_reduce = tensor.stride(dim)
        output = torch.empty([1], dtype=tensor.dtype, device=tensor.device)
        BLOCK_SIZE = 1024
        reduce_1d_persistent_kernel[(num_sms, )](tensor, output, reduce_dim, stride_reduce, BLOCK_SIZE)
    elif tensor.ndim == 2:
        reduce_dim = tensor.shape[dim]
        stride_reduce = tensor.stride(dim)
        M = tensor.shape[1 - dim]
        stride_m = tensor.stride(1 - dim)
        output = torch.empty([M], dtype=tensor.dtype, device=tensor.device)
        BLOCK_SIZE_M = blocksizes[1 - dim]
        reduce_2d_persistent_kernel[(num_sms, )](tensor, output, reduce_dim, M, stride_reduce, stride_m, BLOCK_SIZE_M)
    elif tensor.ndim == 3:
        reduce_dim = tensor.shape[dim]
        stride_reduce = tensor.stride(dim)
        shapes = []
        strides = []
        blocks = []
        for i in range(ndim):
            if i != dim:
                shapes.append(tensor.shape[i])
                strides.append(tensor.stride(i))
                blocks.append(blocksizes[i])
        M, N = shapes
        stride_m, stride_n = strides
        output = torch.empty([M, N], dtype=tensor.dtype, device=tensor.device)
        BLOCK_SIZE_M, BLOCK_SIZE_N = blocks
        reduce_3d_persistent_kernel[(num_sms, )](tensor, output, reduce_dim, M, N, stride_reduce, stride_m, stride_n,
                                                 BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")
    return output
