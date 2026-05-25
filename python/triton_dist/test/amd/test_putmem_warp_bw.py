################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the MIT License (see other files in this tree).
#
################################################################################
r"""
putmem_warp vs putmem_nbi_warp bandwidth — backend-agnostic micro-benchmark.

Measures intra-node warp-level put bandwidth using *exactly* the two
``libshmem_device`` calls used by ``kernels/amd/ep_all2all_fused.py``:

  * ``libshmem_device.putmem_warp``      — blocking (source-buffer reusable
                                            on return)
  * ``libshmem_device.putmem_nbi_warp``  — non-blocking; kernel ends with
                                            ``libshmem_device.fence()`` so we
                                            time issue + drain to wire, not
                                            just issue.

The kernel structure mirrors ``tile_kernel_dispatch_token_intra_node``:
one warp owns one "token" chunk (``chunk_bytes`` = ``2 * hidden`` in the
real dispatch path) and rolls the loop over (warp_id, total_warps).

Backends:
  * ``TRITON_DIST_SHMEM_BACKEND=rocshmem`` (default)
  * ``TRITON_DIST_SHMEM_BACKEND=mori_shmem``

Example (8-rank intra-node):

  # rocshmem
  ulimit -l unlimited
  ROCSHMEM_HEAP_SIZE=2147483648 TRITON_DIST_SHMEM_BACKEND=rocshmem \
      bash ./scripts/launch_amd.sh \
           ./python/triton_dist/test/amd/test_putmem_warp_bw.py --size-sweep

  # mori_shmem
  TRITON_DIST_SHMEM_BACKEND=mori_shmem \
      bash ./scripts/launch_amd.sh \
           ./python/triton_dist/test/amd/test_putmem_warp_bw.py --size-sweep
"""

import argparse
import gc
import os
import sys

# Make sure rocshmem heap default is big enough for the largest sweep config
# below (~64 MB per chunk × N warps).
os.environ.setdefault("ROCSHMEM_HEAP_SIZE", str(2 * 1024 * 1024 * 1024))
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "2G")

_test_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_test_dir, "../../../.."))
_triton_dist_python_path = os.path.join(_workspace_root, "python")
if _triton_dist_python_path not in sys.path:
    sys.path.insert(0, _triton_dist_python_path)
_triton_python_path = os.path.join(_workspace_root, "3rdparty/triton/python")
if os.path.exists(_triton_python_path) and _triton_python_path not in sys.path:
    sys.path.insert(0, _triton_python_path)

import torch  # noqa: E402
import torch.distributed  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton_dist  # noqa: E402
from triton_dist.language.extra import libshmem_device  # noqa: E402
from triton_dist.language.extra.hip.language_extra import tid  # noqa: E402
from triton_dist.profiler_utils import perf_func  # noqa: E402
from triton_dist.utils import (  # noqa: E402
    finalize_distributed,
    get_shmem_backend,
    initialize_distributed,
    shmem_create_tensor,
)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@triton_dist.jit
def putmem_warp_bw_kernel(
    src_ptr,            # symmetric source buf (shmem heap), 1-byte element dtype
    dst_ptr,            # symmetric dest buf (shmem heap),   1-byte element dtype
    target_rank,
    n_chunks,           # number of warp-sized chunks to issue
    chunk_bytes,        # bytes per put op (== element offset stride, since dtype is 1B)
    NBI: tl.constexpr,  # 0 = putmem_warp (blocking), 1 = putmem_nbi_warp
    num_warps: tl.constexpr,
):
    WARP_SIZE: tl.constexpr = 64
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    thread_idx = tid(0)
    warp_id = thread_idx // WARP_SIZE
    global_warp_id = pid * num_warps + warp_id
    total_warps = num_pid * num_warps

    for chunk_id in range(global_warp_id, n_chunks, total_warps):
        # src_ptr / dst_ptr point to a uint8 tensor, so pointer arithmetic is
        # in bytes — chunk_bytes is both the element-offset stride and the
        # byte count passed to putmem_*_warp.
        offset = chunk_id.to(tl.int64) * chunk_bytes
        if NBI:
            libshmem_device.putmem_nbi_warp(dst_ptr + offset, src_ptr + offset, chunk_bytes, target_rank)
        else:
            libshmem_device.putmem_warp(dst_ptr + offset, src_ptr + offset, chunk_bytes, target_rank)

    # Drain in-flight NBI puts so the kernel-time we measure includes
    # delivery-to-wire, matching the blocking-put semantics. fence() on
    # rocshmem maps to rocshmem_fence_wave_wrapper (s_waitcnt vmcnt(0));
    # on mori_shmem it maps to mori_shmem_fence_thread.
    if NBI:
        libshmem_device.fence()


# ---------------------------------------------------------------------------
# Per-pair benchmark
# ---------------------------------------------------------------------------
def bench_pair(
    src_rank,
    dst_rank,
    src_tensor,
    dst_tensor,
    chunk_bytes,
    num_sms,
    num_warps,
    variant,         # "warp" or "nbi_warp"
    warmup,
    iters,
):
    """Return (bandwidth_GBs, latency_ms_per_kernel)."""
    # src_tensor / dst_tensor are uint8, so numel() == bytes.
    nbytes_total = src_tensor.numel()
    assert nbytes_total % chunk_bytes == 0, \
        f"buffer size {nbytes_total} must be a multiple of chunk_bytes={chunk_bytes}"
    n_chunks = nbytes_total // chunk_bytes
    nbi = 1 if variant == "nbi_warp" else 0
    grid = (num_sms, )

    def run():
        putmem_warp_bw_kernel[grid](
            src_tensor,
            dst_tensor,
            dst_rank,
            n_chunks,
            chunk_bytes,
            NBI=nbi,
            num_warps=num_warps,
        )

    # functional warmup + clear receiver
    dst_tensor.zero_()
    torch.cuda.synchronize()
    run()
    torch.cuda.synchronize()

    _, latency_ms = perf_func(run, iters=iters, warmup_iters=warmup)
    bw_gbs = (nbytes_total / (latency_ms * 1e-3)) / (1024**3) if latency_ms > 0 else 0.0
    return bw_gbs, latency_ms, nbytes_total, n_chunks


def print_matrix(title, mat, world_size):
    print(f"\n{title} (GB/s):")
    header = f"{'Src\\Dst':<9}|"
    for j in range(world_size):
        header += f" {j:^6} |"
    print(header)
    print("-" * len(header))
    for i in range(world_size):
        row = f"  {i:<7}|"
        for j in range(world_size):
            row += f" {'  -   ' if i == j else f'{mat[i][j].item():6.2f}'} |"
        print(row)
    print("-" * len(header))


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
def run_one_size(rank, world_size, tp_group, chunk_bytes, n_chunks,
                 num_sms, num_warps, variants, warmup, iters, dtype):
    # Both buffers are allocated as uint8 (1 byte/elem) so pointer arithmetic
    # in the kernel is byte-correct without a per-dtype divide. ``dtype`` is
    # kept as an arg so future runs can switch to bf16 etc., but for the BW
    # itself only ``chunk_bytes`` matters.
    nbytes_total = chunk_bytes * n_chunks

    # Both endpoints need symmetric heap residency:
    #   * dst: target of remote put (mandatory)
    #   * src: mori_shmem's IBGDA / put paths require source to be registered
    #     with the NIC; on rocshmem IPC backend it works either way, so the
    #     stricter requirement wins.
    dst_tensor = shmem_create_tensor((nbytes_total, ), torch.uint8)
    src_tensor = shmem_create_tensor((nbytes_total, ), torch.uint8)
    torch.manual_seed(42 + rank)
    # uint8 normal_() unsupported; fill with rank-tagged pattern.
    src_tensor.fill_((rank * 13 + 1) & 0xFF)
    del dtype  # silence linter

    if rank == 0:
        print(f"\n--- chunk_bytes={chunk_bytes:>7}  n_chunks={n_chunks:>5}"
              f"  total={nbytes_total/(1024**2):>7.2f} MiB"
              f"  num_sms={num_sms} num_warps={num_warps} ---")

    for variant in variants:
        bw_mat = torch.zeros(world_size, device="cuda", dtype=torch.float32)
        lat_mat = torch.zeros(world_size, device="cuda", dtype=torch.float32)

        for src_rank in range(world_size):
            for dst_rank in range(world_size):
                if src_rank == dst_rank:
                    continue
                if rank == src_rank:
                    bw, lat, _, _ = bench_pair(
                        src_rank, dst_rank, src_tensor, dst_tensor,
                        chunk_bytes=chunk_bytes,
                        num_sms=num_sms,
                        num_warps=num_warps,
                        variant=variant,
                        warmup=warmup,
                        iters=iters,
                    )
                    bw_mat[dst_rank] = bw
                    lat_mat[dst_rank] = lat
                torch.distributed.barrier(tp_group)

        all_bw = torch.zeros(world_size, world_size, device="cuda", dtype=torch.float32)
        all_lat = torch.zeros(world_size, world_size, device="cuda", dtype=torch.float32)
        torch.distributed.all_gather_into_tensor(all_bw, bw_mat.view(1, world_size))
        torch.distributed.all_gather_into_tensor(all_lat, lat_mat.view(1, world_size))
        torch.distributed.barrier(tp_group)

        if rank == 0:
            backend = get_shmem_backend()
            tag = f"[{backend}] variant=putmem_{variant}  chunk={chunk_bytes}B  total={nbytes_total/(1024**2):.2f}MiB"
            print_matrix(tag, all_bw, world_size)
            # also print a flat min/avg/max
            mask = ~torch.eye(world_size, dtype=torch.bool, device="cuda")
            vals = all_bw[mask]
            print(f"  -> bw min/avg/max = {vals.min().item():.2f} / {vals.mean().item():.2f} "
                  f"/ {vals.max().item():.2f}  GB/s   latency avg = {all_lat[mask].mean().item()*1000:.2f} us")

    del src_tensor, dst_tensor
    gc.collect()
    torch.cuda.synchronize()


def default_size_configs():
    """(chunk_bytes, n_chunks) tuples.

    chunk_bytes mirrors ``2 * hidden`` in the real fused-dispatch path:
        256  -> hidden=128   (toy)
       1024  -> hidden=512
       3072  -> hidden=1536  (current AMD ep_moe_fused default)
       8192  -> hidden=4096
      14336  -> hidden=7168  (DeepSeek-V3)
      28672  -> hidden=14336 (large)
    n_chunks=4096 keeps the timed kernel ~ms-scale across sizes.
    """
    chunks = [256, 1024, 3072, 8192, 14336, 28672]
    return [(c, 4096) for c in chunks]


def parse_args():
    p = argparse.ArgumentParser(description="putmem_warp vs putmem_nbi_warp bandwidth")
    p.add_argument("--chunk-bytes", type=int, default=14336,
                   help="Bytes per put op (per warp). Default = 2*hidden for hidden=7168.")
    p.add_argument("--n-chunks", type=int, default=4096, help="Number of warp-sized puts per kernel.")
    p.add_argument("--num-sms", type=int, default=8, help="Grid size.")
    p.add_argument("--num-warps", type=int, default=4, help="Warps per CTA (= put-issuers per CTA).")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "uint8"])
    p.add_argument("--variants", type=str, default="warp,nbi_warp",
                   help="Comma list: warp / nbi_warp")
    p.add_argument("--size-sweep", action="store_true",
                   help="Run the chunk_bytes sweep table.")
    return p.parse_args()


def main():
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    tp_group = initialize_distributed()
    torch.distributed.barrier(tp_group)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "uint8": torch.uint8}
    dtype = dtype_map[args.dtype]

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in ("warp", "nbi_warp"):
            raise ValueError(f"unknown variant: {v}")

    if RANK == 0:
        print("=" * 80)
        print(f"putmem_warp bandwidth test")
        print(f"  backend     : {get_shmem_backend()}")
        print(f"  world_size  : {WORLD_SIZE}")
        print(f"  variants    : {variants}")
        print(f"  dtype       : {args.dtype}")
        print(f"  num_sms     : {args.num_sms}")
        print(f"  num_warps   : {args.num_warps}  (= put-issuers per CTA)")
        print("=" * 80)

    if args.size_sweep:
        configs = default_size_configs()
    else:
        configs = [(args.chunk_bytes, args.n_chunks)]

    for chunk_bytes, n_chunks in configs:
        run_one_size(
            rank=RANK,
            world_size=WORLD_SIZE,
            tp_group=tp_group,
            chunk_bytes=chunk_bytes,
            n_chunks=n_chunks,
            num_sms=args.num_sms,
            num_warps=args.num_warps,
            variants=variants,
            warmup=args.warmup,
            iters=args.iters,
            dtype=dtype,
        )

    torch.distributed.barrier(tp_group)
    if RANK == 0:
        print("\n" + "=" * 80)
        print("Done.")
        print("=" * 80)

    gc.collect()
    torch.cuda.synchronize()
    finalize_distributed()


if __name__ == "__main__":
    main()
