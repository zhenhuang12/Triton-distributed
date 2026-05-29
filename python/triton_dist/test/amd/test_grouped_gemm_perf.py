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
"""Standalone grouped-GEMM benchmark (AMD).

Extracts the per-tile MoE grouped-GEMM kernel that lives inside
``ep_all2all_fused.py``'s mega-kernel
(``tile_kernel_moe_grouped_gemm_nk_const``) and runs it on its own,
without the dispatch / combine pipeline around it. The outer
``mega_kernel_grouped_gemm`` mirrors the persistent atomic_add
scheduling loop in ``mega_kernel_dispatch_token_moe_grouped_gemm``
(see ``ep_all2all_fused.py:840``) so the per-tile work issued to the
GPU is byte-for-byte identical to the production path -- only the
producer/consumer barriers are removed (``NEED_WAIT=False``,
``NEED_NOTIFY=False``) since there is no dispatcher writing the
inputs and no combiner reading the outputs.

Reported TFLOPS use the *real* arithmetic count
``2 * sum(split_size) * N * K`` (i.e. ``2 * (num_tokens * topk) * N * K``)
-- the kernel masks out tail rows in the last tile of every expert so
useful FLOPs equal this exactly. Wasted tile-tail work is reported
separately as ``padded_tflops`` so you can also see what the
hardware-issued workload looks like.

Usage::

    python3 -m torch.distributed.run --nproc_per_node=1 \\
        python/triton_dist/test/amd/test_grouped_gemm_perf.py \\
        --num-tokens 8192 --hidden 4096 --intermediate 2048 \\
        --num-experts 256 --topk 6 --warmup 5 --iters 20

Single-process (no torchrun) also works -- the kernel does not need
SHMEM because the NEED_WAIT / NEED_NOTIFY paths are constant-folded
out.
"""

import argparse

import torch
import triton
import triton.language as tl

from triton_dist.kernels.amd.group_gemm import (
    GROUP_GEMM_BLOCK_SIZE_M,
    build_block_row_idx_info_kernel,
)
from triton_dist.language.extra.hip.language_extra import (
    __syncthreads,
    ld_acquire,
    st_release,
    tid,
)

try:
    from primus_turbo.pytorch.ops import grouped_gemm as turbo_grouped_gemm
    _HAS_TURBO = True
    _TURBO_IMPORT_ERR = None
except ImportError as e:
    turbo_grouped_gemm = None
    _HAS_TURBO = False
    _TURBO_IMPORT_ERR = e

# ---------------------------------------------------------------------------
# Extracted device kernels (verbatim from ep_all2all_fused.py modulo the
# dropped dl.num_ranks() call, which is dead code under NEED_WAIT=False).
# ---------------------------------------------------------------------------


@triton.jit
def dot_k_const(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    M,
    N,
    K: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    need_mask: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if need_mask:
            a = tl.load(
                a_ptrs, mask=((tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
                              (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[None, :]))
        else:
            a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[None, :])
        b = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[:, None])

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(a_ptrs.dtype.element_ty)
    if need_mask:
        c_mask = (tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        c_mask = (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def tile_kernel_moe_grouped_gemm_nk_const(
    pid,
    num_pid,
    counter_ptr,
    barriers_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NEED_WAIT: tl.constexpr,
    NEED_NOTIFY: tl.constexpr,
    USE_BLOCK_WISE_BARRIER: tl.constexpr,
    IS_DISPATCH_TWO_STAGET: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_block_n
    pid_n = pid % num_block_n

    expert_id = tl.load(expert_ids_ptr + pid_m)
    split_size = tl.load(split_size_ptr + expert_id)
    split_size_cum = tl.load(split_size_cum_ptr + pid_m)
    row_begin = split_size_cum
    tile_num = tl.load(tile_num_ptr + pid_m)
    tile_num_cum = tl.load(tile_num_cum_ptr + pid_m)
    tile_begin = tile_num_cum - tile_num
    local_pid_m = pid_m - tile_begin

    thread_idx = tid(0)

    local_pid_m, pid_n = tl.swizzle2d(local_pid_m, pid_n, tile_num, num_block_n, GROUP_SIZE_M)

    if NEED_WAIT:
        if IS_DISPATCH_TWO_STAGET:
            if USE_BLOCK_WISE_BARRIER:
                barrier_idx = local_pid_m + tile_begin
                if thread_idx == 0:
                    while ld_acquire(barriers_ptr + barrier_idx, scope=tl.constexpr("gpu")) != 1:
                        pass
                __syncthreads()
            else:
                barrier_idx = expert_id
                while ld_acquire(barriers_ptr + barrier_idx, scope=tl.constexpr("gpu")) != 1:
                    pass
        else:
            if thread_idx < WORLD_SIZE:
                barrier_idx = expert_id * WORLD_SIZE + thread_idx
                while ld_acquire(barriers_ptr + barrier_idx, scope=tl.constexpr("gpu")) != 1:
                    pass
            __syncthreads()

    row_remain = split_size - local_pid_m * BLOCK_SIZE_M

    offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = (b_ptr + expert_id.to(tl.int64) * stride_be + offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    offs_token = row_begin.to(tl.int64) + local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = (a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    if row_remain >= BLOCK_SIZE_M:
        dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                    stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False)
    elif row_remain > 0:
        dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                    stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, True)

    if NEED_NOTIFY:
        __syncthreads()
        token_begin = row_begin + local_pid_m * BLOCK_SIZE_M
        valid_tokens = min(row_remain, BLOCK_SIZE_M)
        if thread_idx < valid_tokens:
            st_release(barriers_ptr + (token_begin + thread_idx) * num_block_n + pid_n, 1, scope=tl.constexpr("gpu"))


@triton.jit(do_not_specialize=["M"])
def mega_kernel_grouped_gemm(
    task_counter_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,
    counter_ptr,
    barriers_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Persistent grouped-GEMM mega kernel.

    Same atomic_add scheduling loop as
    ``mega_kernel_dispatch_token_moe_grouped_gemm`` in
    ``ep_all2all_fused.py``, but with only the grouped-GEMM branch -- no
    dispatch task ids in the counter, NEED_WAIT / NEED_NOTIFY off.
    """
    task_id = tl.atomic_add(task_counter_ptr, 1)
    group_gemm_total_tiles_m = tl.load(num_total_tiles_ptr)
    group_gemm_total_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    group_gemm_tasks = group_gemm_total_tiles_m * group_gemm_total_tiles_n

    while task_id < group_gemm_tasks:
        tile_kernel_moe_grouped_gemm_nk_const(
            task_id,
            group_gemm_tasks,
            counter_ptr,
            barriers_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            expert_ids_ptr,
            split_size_ptr,
            split_size_cum_ptr,
            tile_num_ptr,
            tile_num_cum_ptr,
            num_total_tiles_ptr,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_be,
            stride_bn,
            stride_bk,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M,
            NEED_WAIT=False,
            NEED_NOTIFY=False,
            USE_BLOCK_WISE_BARRIER=False,
            IS_DISPATCH_TWO_STAGET=False,
            WORLD_SIZE=1,
        )
        task_id = tl.atomic_add(task_counter_ptr, 1)


# ---------------------------------------------------------------------------
# Metadata + launcher (mirrors the host side of ep_all2all_fused.py).
# ---------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return max(p, 1)


def build_routing_metadata(split_size: torch.Tensor, num_experts: int, block_size_m: int, num_sms: int):
    """Run the existing build_block_row_idx_info_kernel to materialise the
    six metadata tensors the grouped-GEMM tile kernel consumes.

    Returns (expert_ids, split_size_cum, tile_num, tile_num_cum,
    num_total_tiles, split_size_cum_per_expert).  All allocated to the
    upper-bound length used in production (cdiv(M, BLOCK_SIZE_M) + G).
    """
    device = split_size.device
    M_total = int(split_size.sum().item())
    M_grid_max = triton.cdiv(M_total, block_size_m) + num_experts

    expert_ids = torch.zeros(M_grid_max, dtype=torch.int32, device=device)
    split_size_cum = torch.zeros(M_grid_max, dtype=torch.int32, device=device)
    tile_num = torch.zeros(M_grid_max, dtype=torch.int32, device=device)
    tile_num_cum = torch.zeros(M_grid_max, dtype=torch.int32, device=device)
    expert_tile_offset = torch.zeros(num_experts, dtype=torch.int32, device=device)
    split_size_cum_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    num_total_tiles = torch.zeros(1, dtype=torch.int32, device=device)

    E_PAD = _next_pow2(num_experts)
    build_block_row_idx_info_kernel[(num_sms, )](
        split_size,
        split_size_cum_per_expert,
        expert_ids,
        split_size_cum,
        tile_num,
        tile_num_cum,
        expert_tile_offset,
        num_total_tiles,
        E=num_experts,
        E_PAD=E_PAD,
        BLOCK_SIZE_M=block_size_m,
        NUM_SMS=num_sms,
    )
    return expert_ids, split_size_cum, tile_num, tile_num_cum, num_total_tiles, split_size_cum_per_expert


def run_grouped_gemm(
    a: torch.Tensor,  # [M_total, K]
    weights: torch.Tensor,  # [G, N, K]
    split_size: torch.Tensor,  # [G] int32
    out: torch.Tensor,  # [M_total, N]
    num_sms: int,
    block_size_n: int = 256,
    block_size_k: int = 64,
    group_size_m: int = 3,
    num_warps: int = 8,
    num_stages: int = 3,
):
    G, N, K = weights.shape
    M_total, K_a = a.shape
    assert K_a == K
    M_grid_max = triton.cdiv(M_total, GROUP_GEMM_BLOCK_SIZE_M) + G

    (expert_ids, split_size_cum, tile_num, tile_num_cum, num_total_tiles,
     _per_expert) = build_routing_metadata(split_size, G, GROUP_GEMM_BLOCK_SIZE_M, num_sms)

    task_counter = torch.zeros(1, dtype=torch.int32, device=a.device)
    # NEED_WAIT/NEED_NOTIFY are off so these are unused, but the kernel still
    # takes pointers -- pass single-element dummies sized to the worst case.
    dummy_counter = torch.zeros(max(G, M_grid_max), dtype=torch.int32, device=a.device)
    dummy_barriers = torch.zeros(max(G, M_grid_max), dtype=torch.int32, device=a.device)

    mega_kernel_grouped_gemm[(num_sms, )](
        task_counter,
        a,
        weights,
        out,
        expert_ids,
        split_size,
        split_size_cum,
        tile_num,
        tile_num_cum,
        num_total_tiles,
        dummy_counter,
        dummy_barriers,
        M_total,
        N,
        K,
        a.stride(0),
        a.stride(1),
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        out.stride(0),
        out.stride(1),
        BLOCK_SIZE_M=GROUP_GEMM_BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def run_turbo_grouped_gemm(a: torch.Tensor,  # [M_total, K]
                           weights: torch.Tensor,  # [G, N, K]
                           split_size: torch.Tensor,  # [G] int32
                           ):
    """Primus-Turbo baseline: same DeepEPTokenDispatcher-paired grouped_gemm
    used by ``turbo_ep_moe`` in test_ep_moe_fused.py.

    Matches the triton path's contract: A is [M_total, K], weights are
    [G, N, K] (trans_b=True does the .T per expert), output is [M_total, N].
    """
    group_lens = split_size.to(device=a.device, dtype=torch.int64)
    return turbo_grouped_gemm(a, weights, group_lens, trans_b=True)


def torch_reference_grouped_gemm(a, weights, split_size):
    G, N, K = weights.shape
    out = torch.empty(a.shape[0], N, dtype=a.dtype, device=a.device)
    offs = 0
    for e in range(G):
        s = int(split_size[e].item())
        if s == 0:
            continue
        out[offs:offs + s] = a[offs:offs + s].to(torch.float32) @ weights[e].to(torch.float32).T
        offs += s
    return out


# ---------------------------------------------------------------------------
# Driver: shapes mirror test_ep_moe_fused.py, timing matches its style.
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Standalone grouped-GEMM benchmark extracted from ep_all2all_fused.py")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    p.add_argument("--num-tokens", type=int, default=8192, help="Number of routed tokens before topk expansion")
    p.add_argument("--topk", type=int, default=8, help="Top-K experts -- M_routed = num_tokens * topk")
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--hidden", type=int, default=1536, help="Output dim N for the grouped GEMM (fc1 path)")
    p.add_argument("--intermediate", type=int, default=480, help="Reduction dim K for the grouped GEMM (fc1 path)")
    p.add_argument("--block-n", type=int, default=256)
    p.add_argument("--block-k", type=int, default=64)
    p.add_argument("--group-m", type=int, default=3)
    p.add_argument("--num-warps", type=int, default=8)
    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--num-sms", type=int, default=-1, help="Persistent grid size; -1 = device CU count")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--skip-turbo", action="store_true", help="Skip the Primus-Turbo grouped_gemm baseline comparison")
    return p.parse_args()


def _uniform_split(M_total: int, G: int, device, generator=None) -> torch.Tensor:
    """Random per-expert token counts that sum to M_total.

    Multinomial draw -> matches the routing-imbalance profile of a real
    MoE step (some experts get more tokens than others).
    """
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    # Dirichlet-like: sample uniform probs, normalise, then multinomial.
    probs = torch.rand(G, generator=generator)
    probs = probs / probs.sum()
    counts = torch.multinomial(probs, M_total, replacement=True, generator=generator)
    split = torch.bincount(counts, minlength=G).to(torch.int32)
    assert int(split.sum().item()) == M_total
    return split.to(device)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    dtype = getattr(torch, args.dtype)

    G = args.num_experts
    N = args.hidden  # output dim
    K = args.intermediate  # reduction dim
    M_total = args.num_tokens * args.topk

    if args.num_sms == -1:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    else:
        num_sms = args.num_sms

    print(f"Grouped-GEMM benchmark | dtype={args.dtype} num_tokens={args.num_tokens} "
          f"topk={args.topk} -> M_routed={M_total} experts={G} N={N} K={K} "
          f"BLOCK_M={GROUP_GEMM_BLOCK_SIZE_M} BLOCK_N={args.block_n} "
          f"BLOCK_K={args.block_k} num_sms={num_sms}")

    # Per-expert token counts via multinomial draw -> uneven routing.
    split_size = _uniform_split(M_total, G, device)

    # Inputs: random routed-token matrix + per-expert weight stack [G, N, K].
    a = (torch.randn(M_total, K, dtype=dtype, device=device) * 0.1)
    weights = (torch.randn(G, N, K, dtype=dtype, device=device) * 0.1)
    out = torch.empty(M_total, N, dtype=dtype, device=device)

    run_turbo = (not args.skip_turbo) and _HAS_TURBO
    if args.skip_turbo:
        print("[turbo] skipped via --skip-turbo")
    elif not _HAS_TURBO:
        print(f"[turbo] primus_turbo not importable ({_TURBO_IMPORT_ERR}) — "
              f"skipping turbo baseline. Install with `pip install primus_turbo` "
              f"or pass --skip-turbo to silence this.")

    # ---- Correctness ----------------------------------------------------
    if not args.skip_correctness:
        run_grouped_gemm(a, weights, split_size, out, num_sms=num_sms, block_size_n=args.block_n,
                         block_size_k=args.block_k, group_size_m=args.group_m, num_warps=args.num_warps,
                         num_stages=args.num_stages)
        ref = torch_reference_grouped_gemm(a, weights, split_size)
        diff = (out.float() - ref.float()).abs()
        max_err = diff.max().item()
        rel_err = (diff / (ref.float().abs() + 1e-6)).max().item()
        cos_sim = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
        ok = (cos_sim > 0.999) and (max_err < 1.0)
        tag = "PASS" if ok else "FAIL"
        print(f"[correctness] triton {tag} max|diff|={max_err:.4e} max|rel|={rel_err:.4e} "
              f"cos_sim={cos_sim:.6f}")
        if not ok:
            raise SystemExit(1)

        if run_turbo:
            t_out = run_turbo_grouped_gemm(a, weights, split_size)
            t_diff = (t_out.float() - ref.float()).abs()
            t_max = t_diff.max().item()
            t_rel = (t_diff / (ref.float().abs() + 1e-6)).max().item()
            t_cos = torch.nn.functional.cosine_similarity(t_out.float().flatten(), ref.float().flatten(), dim=0).item()
            t_ok = (t_cos > 0.999) and (t_max < 1.0)
            t_tag = "PASS" if t_ok else "FAIL"
            print(f"[correctness] turbo  {t_tag} max|diff|={t_max:.4e} max|rel|={t_rel:.4e} "
                  f"cos_sim={t_cos:.6f}")
            if not t_ok:
                raise SystemExit(1)

    # ---- Perf -----------------------------------------------------------
    # Warmup
    for _ in range(args.warmup):
        run_grouped_gemm(a, weights, split_size, out, num_sms=num_sms, block_size_n=args.block_n,
                         block_size_k=args.block_k, group_size_m=args.group_m, num_warps=args.num_warps,
                         num_stages=args.num_stages)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        run_grouped_gemm(a, weights, split_size, out, num_sms=num_sms, block_size_n=args.block_n,
                         block_size_k=args.block_k, group_size_m=args.group_m, num_warps=args.num_warps,
                         num_stages=args.num_stages)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / args.iters

    # Real arithmetic count: 2 * sum(split_size) * N * K
    # = 2 * M_routed * N * K. The kernel masks tail rows in the last tile
    # of every expert, so this is exactly the useful FLOPs.
    real_flops = 2.0 * M_total * N * K
    real_tflops = real_flops / (avg_ms * 1e-3) / 1e12

    # What the GPU actually issues: every expert rounds its M up to
    # BLOCK_SIZE_M before tiling. Wasted work = (padded - real) FLOPs.
    padded_M = (((split_size.to(torch.int64) + GROUP_GEMM_BLOCK_SIZE_M - 1) // GROUP_GEMM_BLOCK_SIZE_M) *
                GROUP_GEMM_BLOCK_SIZE_M).sum().item()
    padded_flops = 2.0 * padded_M * N * K
    padded_tflops = padded_flops / (avg_ms * 1e-3) / 1e12

    # I/O bandwidth (BF16/FP16 = 2 bytes):
    #   reads:  A = M_routed*K, B = G*N*K (whole weight stack is touched
    #           across experts -- pessimistic if some experts are empty)
    #   writes: C = M_routed*N
    elem_bytes = a.element_size()
    touched_experts = int((split_size > 0).sum().item())
    bytes_read = elem_bytes * (M_total * K + touched_experts * N * K)
    bytes_write = elem_bytes * (M_total * N)
    gbps = (bytes_read + bytes_write) / (avg_ms * 1e-3) / 1e9

    print(f"[triton] avg latency = {avg_ms:8.4f} ms over {args.iters} iters")
    print(f"[triton] real   FLOPs = 2*M_routed*N*K = {real_flops:.3e}  "
          f"-> {real_tflops:8.3f} TFLOPS")
    print(f"[triton] padded FLOPs = 2*sum(cdiv(s,BM)*BM)*N*K = {padded_flops:.3e}  "
          f"-> {padded_tflops:8.3f} TFLOPS (HW-issued)")
    print(f"[triton] effective BW (A+B read + C write) = {gbps:8.2f} GB/s")
    print(f"[perf]   routing: {touched_experts}/{G} experts touched, "
          f"max={int(split_size.max().item())} min={int(split_size.min().item())} "
          f"mean={M_total/G:.1f}")

    if run_turbo:
        # Warmup
        t_out_buf = run_turbo_grouped_gemm(a, weights, split_size)
        for _ in range(args.warmup):
            t_out_buf = run_turbo_grouped_gemm(a, weights, split_size)
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(args.iters):
            t_out_buf = run_turbo_grouped_gemm(a, weights, split_size)
        t_end.record()
        torch.cuda.synchronize()
        del t_out_buf
        turbo_avg_ms = t_start.elapsed_time(t_end) / args.iters

        turbo_tflops = real_flops / (turbo_avg_ms * 1e-3) / 1e12
        speedup = turbo_avg_ms / avg_ms  # >1.0x => triton faster than turbo

        print(f"[turbo]  avg latency = {turbo_avg_ms:8.4f} ms over {args.iters} iters")
        print(f"[turbo]  real   FLOPs = 2*M_routed*N*K = {real_flops:.3e}  "
              f"-> {turbo_tflops:8.3f} TFLOPS")
        print(f"[compare] triton vs turbo: triton {avg_ms:8.4f} ms vs "
              f"turbo {turbo_avg_ms:8.4f} ms  =>  triton speedup = {speedup:6.3f}x "
              f"({'triton faster' if speedup > 1.0 else 'turbo faster'})")


if __name__ == "__main__":
    main()
