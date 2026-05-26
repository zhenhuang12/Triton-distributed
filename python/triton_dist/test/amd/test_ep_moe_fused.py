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

"""
AMD parity of the NVIDIA fused EP MoE end-to-end test
(``python/triton_dist/test/nvidia/test_ep_moe_fused.py``).

This file mirrors the NVIDIA test exactly (same argparse, same test config
matrix, same benchmark + precision check loop, same ``print_benchmark_comparison``
summary). Only the following AMD-specific changes are necessary:

- the autograd op is imported via the public dispatcher
  ``triton_dist.function.TritonDistFusedEpMoeFunction`` which selects the
  AMD implementation under ``is_hip()``;
- the symmetric-heap world is bootstrapped through ``init_rocshmem_by_torch_process_group``
  (or ``init_mori_by_torch_process_group`` for the mori_shmem backend)
  instead of ``init_nvshmem_by_torch_process_group``;
- before ``finalize_distributed()`` we explicitly ``del`` every reference
  that pins symmetric-heap tensors and run ``gc.collect()`` +
  ``torch.cuda.synchronize()`` so the rocSHMEM heap teardown does not
  race with pytorch's tensor finalizers (same workaround the existing
  AMD ``test_gemm_rs_intra_node.py`` uses).

Usage:
  TRITON_DIST_SHMEM_BACKEND=rocshmem bash ./scripts/launch_amd.sh \\
      ./python/triton_dist/test/amd/test_ep_moe_fused.py
"""

import argparse
import datetime
import gc
import os
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.profiler import profile, ProfilerActivity

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.ops import grouped_gemm

from triton_dist.function import (
    TritonDistFusedEpMoeFunction,
    init_triton_dist_ep_op,
    deinit_triton_dist_ep_op,
    set_triton_dist_moe_profile_enabled,
)
from triton_dist.utils import (
    finalize_distributed,
    get_shmem_backend,
    init_mori_by_torch_process_group,
    init_rocshmem_by_torch_process_group,
)
from triton_dist.profiler_utils import benchmark_latency_memory, print_benchmark_comparison

# In-process aggregator for the in-kernel ENABLE_PROFILING records.
# Populated by ``_install_kernel_profile_aggregator`` (only when --enable-kernel-profiler).
# Layout: {kernel_label: {task_name: [duration_ns, ...]}}
_KERNEL_PROFILE_STATS: dict = {}


def _install_kernel_profile_aggregator():
    """Monkey-patch ``export_to_perfetto_trace`` in the AMD layer so every
    call (one per dispatch / combine invocation) also pulls the profiler
    buffer back to the host, decodes it via ``parse_to_tracks``, and
    accumulates per-task durations into ``_KERNEL_PROFILE_STATS``.

    The original perfetto trace is still written (we just wrap the call),
    so users can also inspect the last iteration in https://ui.perfetto.dev.
    """
    import triton_dist.layers.amd.ep_a2a_fused_layer as _layer
    from triton_dist.tools.profiler import parse_to_tracks

    _original_export = _layer.export_to_perfetto_trace

    def _wrapped_export(profiler_buffer, task_names, file_name, *args, **kwargs):
        try: 
            tracks = parse_to_tracks(profiler_buffer)
            kernel_label = os.path.basename(file_name)
            bucket = _KERNEL_PROFILE_STATS.setdefault(kernel_label, {})
            for _block_idx, tasks in tracks.items():
                for t in tasks:
                    name = task_names[t.task_type]
                    bucket.setdefault(name, []).append(int(t.duration))
        except Exception as e:
            # Don't let analysis errors mask the underlying run.
            print(f"[kernel-profiler] aggregation failed for {file_name}: {e}")
        # tg4perfetto is optional; if it's missing we still want the
        # in-process aggregator to fire on every iteration.
        try:
            return _original_export(profiler_buffer, task_names, file_name, *args, **kwargs)
        except ImportError as e:
            print(f"[kernel-profiler] perfetto export skipped ({e}); aggregator OK")
        return None

    _layer.export_to_perfetto_trace = _wrapped_export


def _print_kernel_profile_summary(rank: int):
    """Pretty-print mean / p50 / p99 per (kernel, task) on rank 0."""
    if rank != 0 or not _KERNEL_PROFILE_STATS:
        return
    try:
        import numpy as np
    except ImportError:
        np = None

    def _pct(samples, q):
        if np is not None:
            return float(np.percentile(samples, q))
        s = sorted(samples)
        k = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
        return float(s[k])

    print("\n================ in-kernel profiler summary (ns / record) ================")
    print(f"{'kernel':<48} {'task':<32} {'n':>6} {'mean':>10} {'p50':>10} {'p99':>10} {'max':>10} {'share':>7}")
    for kernel_label in sorted(_KERNEL_PROFILE_STATS):
        bucket = _KERNEL_PROFILE_STATS[kernel_label]
        # Per-kernel total = sum of mean(task) * count(task); use sum of all samples
        # weighted so tasks with more recorded segments aren't unfairly inflated.
        total = sum(sum(v) for v in bucket.values()) or 1
        ordered = sorted(bucket.items(), key=lambda kv: sum(kv[1]), reverse=True)
        for name, samples in ordered:
            mean = sum(samples) / len(samples)
            p50 = _pct(samples, 50)
            p99 = _pct(samples, 99)
            share = 100.0 * sum(samples) / total
            print(f"{kernel_label:<48} {name:<32} {len(samples):>6} "
                  f"{mean:>10.0f} {p50:>10.0f} {p99:>10.0f} {max(samples):>10} {share:>6.1f}%")
        print("-" * 120)
    print("==========================================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Test Expert Parallel MoE implementations (AMD)")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=10, help="Performance iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--profile", default=False, action="store_true", help="Enable torch.profiler")
    parser.add_argument("--ntokens", default=8192, type=int, help="Total number of tokens")
    parser.add_argument("--hidden_dim", default=1536, type=int, help="Hidden dimension")
    parser.add_argument("--ffn_dim", default=480, type=int, help="FFN intermediate dimension")
    parser.add_argument("--topk", default=8, type=int, help="Top-K experts per token")
    parser.add_argument("--num_experts", default=64, type=int, help="Total number of experts")
    parser.add_argument("--num_ranks", type=int, default=None, help="Number of EP ranks (default: WORLD_SIZE)")
    parser.add_argument("--sm_margin", type=int, default=0, help="Number of SMs to reserve for other kernels")
    parser.add_argument("--capacity", type=float, default=4.0, help="Capacity of expert group")
    # AMD-only escape hatch (not present in the NVIDIA test): keeps the
    # forward-only path alive while bisecting backward-kernel issues.
    parser.add_argument("--skip_backward", action="store_true",
                        help="Run forward only (for debugging only; the NVIDIA test always runs forward+backward).")
    parser.add_argument("--enable-kernel-profiler", action="store_true",
                        help="Enable the in-kernel ENABLE_PROFILING records (perfetto + per-task summary). "
                             "Perturbs timing; use small --warmup/--iters when set.")
    parser.add_argument("--max-tokens-per-rank", type=int, default=None,
                        help="Override the EP-op symm-heap sizing tile (default 8192*4). "
                             "Lower this when the mori/rocshmem heap can't hold the default.")
    return parser.parse_args()


args = parse_args()


def prepare_inputs(
    ffn_dim,
    hidden_dim,
    topk,
    ntokens_per_rank_list,
    num_experts,
    rank,
    num_ranks,
    dtype=torch.bfloat16,
    device="cuda",
    concat_weights=False,
):
    """
    Prepare inputs for MoE forward pass.

    Returns:
        Tuple of (weights, activations, grad_output)
        - weights: (fc1_1, fc1_2, fc2) - expert weights
        - activations: (hidden_states, gate_weights, expert_index, ntokens_per_rank_list, gate_logits)
          - gate_logits is the dense [ntokens, num_experts] probs tensor that
            DeepEPTokenDispatcher.token_dispatch expects; it is pre-scattered
            so dispatcher's internal ``gate_logits.gather(1, expert_index)``
            reproduces ``gate_weights`` byte-exact — lets turbo_ep_moe skip
            the per-call scatter.
        - grad_output: Gradient for backward pass
    """
    assert len(ntokens_per_rank_list) == num_ranks

    # Expert weights: each rank holds a subset of experts
    if not concat_weights:
        fc1_1 = torch.nn.Parameter(
            torch.randn([num_experts // num_ranks, ffn_dim, hidden_dim], dtype=dtype, device=device) * 0.1)
        fc1_2 = torch.nn.Parameter(
            torch.randn([num_experts // num_ranks, ffn_dim, hidden_dim], dtype=dtype, device=device) * 0.1)
    else:
        fc1_1 = torch.nn.Parameter(
            torch.randn([num_experts // num_ranks, ffn_dim * 2, hidden_dim], dtype=dtype, device=device) * 0.1)
        fc1_2 = None
    fc2 = torch.nn.Parameter(
        torch.randn([num_experts // num_ranks, hidden_dim, ffn_dim], dtype=dtype, device=device) * 0.1)

    # Activations: gate logits and expert selection
    gate_logits = torch.rand([ntokens_per_rank_list[rank], num_experts], device=device).float()
    gate_logits = gate_logits / gate_logits.sum(dim=-1, keepdim=True)
    topk_res = torch.topk(gate_logits, k=topk, dim=-1)
    gate_weights = torch.randn_like(topk_res.values, dtype=dtype, device=device).float()
    expert_index = topk_res.indices.to(torch.int32)

    # Build the dense probs tensor turbo's DeepEPTokenDispatcher expects.
    # Scattering once here (instead of inside turbo_ep_moe every call) lets
    # the baseline path skip the per-call scatter — the dispatcher does
    # ``gate_logits.gather(1, expert_index)`` internally to recover
    # ``gate_weights``. Overwrites the raw topk source above; the topk
    # decision is already captured in ``expert_index``.
    gate_logits = torch.zeros(
        ntokens_per_rank_list[rank], num_experts, dtype=torch.float32, device=device
    )
    gate_logits.scatter_(1, expert_index.long(), gate_weights.float())

    # Randomly drop some tokens (set expert_index to num_experts)
    # NOTE: disabled — the fused kernel treats expert_index == num_experts as a
    # drop sentinel, but the turbo baseline (DeepEPTokenDispatcher + grouped_gemm)
    # has no drop semantics and would index out of bounds. Keeping all topk slots
    # valid lets both paths share the exact same routing without the
    # mask-and-zero-prob workaround in turbo_ep_moe().
    # random_drop_tokens = torch.randint(0, 10, [ntokens_per_rank_list[rank], topk], device=device)
    # expert_index = expert_index.masked_fill(random_drop_tokens > 9, num_experts)

    hidden_states = torch.randn([ntokens_per_rank_list[rank], hidden_dim], dtype=dtype, device=device) * 0.1
    grad_output = torch.randn([ntokens_per_rank_list[rank], hidden_dim], dtype=dtype, device=device)

    # Enable gradients
    fc1_1.requires_grad_()
    if not concat_weights:
        fc1_2.requires_grad_()
    fc2.requires_grad_()
    gate_weights.requires_grad_()
    hidden_states.requires_grad_()

    return (
        (fc1_1, fc1_2, fc2),
        (hidden_states, gate_weights, expert_index, ntokens_per_rank_list, gate_logits),
        grad_output,
    )


def triton_dist_moe_func(weights, activations, ep_group):
    """Triton-dist MoE forward function"""
    fc1_1, fc1_2, fc2 = weights
    hidden_states, gate_weights, expert_index, ntokens_per_rank_list, _ = activations
    assert len(fc1_1.shape) == 3
    num_experts = fc1_1.shape[0] * ep_group.size()  # total experts
    output = TritonDistFusedEpMoeFunction.apply(num_experts, gate_weights, expert_index, hidden_states, fc1_1, fc1_2,
                                                fc2, ep_group)
    return output


def turbo_ep_moe(weights, activations, dispatcher):
    """Baseline EP-MoE forward: turbo TokenDispatcher + turbo grouped_gemm + SwiGLU.

    Uses the same expert_index that the fused kernel sees (passed through
    ``indices=`` so the topk routing decision is identical) — only the
    dispatch + compute path differs.
    """
    fc1_1, fc1_2, fc2 = weights
    hidden_states, _, expert_index, _, gate_logits = activations

    # prepare_inputs pre-scatters gate_weights into the dense [ntokens,
    # num_experts] probs tensor that DeepEPTokenDispatcher expects;
    # internally token_dispatch does gate_logits.gather(1, indices) to
    # recover the topk weights, so we skip the per-call scatter here.
    permuted_hidden, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
        hidden_states, gate_logits, indices=expert_index.long()
    )

    group_lens = tokens_per_expert.to(device=hidden_states.device, dtype=torch.int64)

    # Megatron + Primus-Turbo uses a single pre-concatenated [gate; up] fc1
    # weight per expert and runs ONE grouped_gemm for fc1
    # (see PrimusTurboGroupedLinear.forward_internal). Force that path here:
    # if the caller handed us split weights, concat first.
    if fc1_2 is not None:
        fc1 = torch.cat([fc1_1, fc1_2], dim=1)
    else:
        fc1 = fc1_1
    fc1_out = grouped_gemm(permuted_hidden, fc1, group_lens, trans_b=True)
    gate, up = fc1_out.chunk(2, dim=-1)

    # Apply routing weight inside SwiGLU to mirror the fused kernel's swiglu_forward(scale=...).
    intermediate = (F.silu(gate.float()) * up.float() * permuted_probs.unsqueeze(-1)).to(hidden_states.dtype)

    fc2_out = grouped_gemm(intermediate, fc2, group_lens, trans_b=True)

    return dispatcher.token_combine(fc2_out)


def uniform_split_tokens(ntokens, nsplits):
    """Uniformly split tokens across ranks"""
    ret = [ntokens // nsplits for _ in range(nsplits)]
    ret[-1] = ntokens - sum(ret[:-1])
    assert all(x >= 0 for x in ret)
    return ret


def main():
    torch.manual_seed(args.seed)

    # Initialize distributed environment
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    NUM_RANKS = args.num_ranks if args.num_ranks is not None else WORLD_SIZE

    if WORLD_SIZE > 1:
        torch.distributed.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            world_size=WORLD_SIZE,
            rank=RANK,
            timeout=datetime.timedelta(seconds=1800),
        )
        assert torch.distributed.is_initialized()

    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.synchronize()

    # Create EP group only if we have multiple ranks
    if NUM_RANKS > 1:
        if WORLD_SIZE == 1:
            # Single process, create a dummy group (not actually distributed)
            print("Warning: Running with WORLD_SIZE=1 but NUM_RANKS>1. MoE requires distributed setup.")
            print("Skipping MoE tests. Please run with WORLD_SIZE >= NUM_RANKS.")
            return
        EP_GROUP = torch.distributed.new_group(ranks=list(range(NUM_RANKS)), backend="nccl")
        # AMD-only bootstrap (NVIDIA uses ``init_nvshmem_by_torch_process_group`` here).
        backend = get_shmem_backend()
        if backend == 'rocshmem':
            init_rocshmem_by_torch_process_group(EP_GROUP)
        elif backend == 'mori_shmem':
            init_mori_by_torch_process_group(EP_GROUP)
        else:
            raise RuntimeError(f"Unexpected SHMEM backend on AMD: {backend!r}")
    else:
        EP_GROUP = None
        print("Warning: NUM_RANKS=1. MoE requires at least 2 ranks. Skipping tests.")
        return

    DTYPE = getattr(torch, args.dtype)

    rank = EP_GROUP.rank()
    num_ranks = NUM_RANKS

    print(f"MoE Test Parameters: ntokens={args.ntokens}, hidden_dim={args.hidden_dim}, "
          f"ffn_dim={args.ffn_dim}, topk={args.topk}, num_experts={args.num_experts}, "
          f"num_ranks={num_ranks}, DType={DTYPE}, SM Margin={args.sm_margin}\n")

    # Initialize triton_dist EP operation
    max_tokens_per_rank = args.max_tokens_per_rank if args.max_tokens_per_rank is not None else 8192 * 4
    init_triton_dist_ep_op(
        EP_GROUP,
        max_tokens_per_rank,
        args.hidden_dim,
        args.topk,
        rank,
        args.num_experts,
        num_ranks,
        dtype=DTYPE,
        weight_dtype=torch.float32,
        num_sm=64,
        num_buffers=1,
        capacity=args.capacity,
    )

    # Initialize turbo TokenDispatcher (baseline: turbo dispatcher + grouped_gemm).
    turbo_dispatcher = turbo.modules.DeepEPTokenDispatcher(
        num_experts=args.num_experts,
        router_topk=args.topk,
        ep_group=EP_GROUP,
        permute_fusion=True,
        deepep_num_use_cu=80,
    )

    # Optionally turn on the in-kernel profiler. Each fwd/bwd dispatch &
    # combine call will then allocate a ProfilerBuffer, record per-task
    # start/end timestamps inside the device kernel, write a
    # ``prof/mega/<label>_rank_<rank>.perfetto-trace`` (last iter wins),
    # and feed our aggregator for the end-of-run per-task latency table.
    if args.enable_kernel_profiler:
        _install_kernel_profile_aggregator()
        set_triton_dist_moe_profile_enabled(
            enabled=True,
            fwd_dispatch=True,
            fwd_combine=True,
            bwd_dispatch=not args.skip_backward,
            bwd_combine=not args.skip_backward,
        )
        if rank == 0:
            print("[kernel-profiler] ENABLE_PROFILING active for "
                  f"fwd_dispatch=True fwd_combine=True "
                  f"bwd_dispatch={not args.skip_backward} bwd_combine={not args.skip_backward}\n"
                  "[kernel-profiler] NOTE: this perturbs timing; the TFLOPS/latency "
                  "numbers should be ignored.")

    # Test configurations
    test_configs = []
    ntokens_list = [1024, 2048, 4096, 8192, 8192 * 2, 8192 * 4, 8192 * 8, 8192 * 16, 8192 * 32]
    _min_ntokens = int(os.environ.get("EP_NTOKENS_MIN", "0"))
    if _min_ntokens > 0:
        ntokens_list = [n for n in ntokens_list if n >= _min_ntokens]
    hidden_dim_list = [args.hidden_dim]
    ffn_dim_list = [args.ffn_dim]
    concat_weights_list = [False, True]

    for ntokens in ntokens_list:
        for hidden_dim in hidden_dim_list:
            for ffn_dim in ffn_dim_list:
                for concat_weights in concat_weights_list:
                    if ntokens <= args.ntokens:
                        test_configs.append((ntokens, hidden_dim, ffn_dim, concat_weights))

    test_configs = list(dict.fromkeys(test_configs))

    all_implementations = {}

    try:
        if args.profile:
            os.makedirs("prof/test_moe", exist_ok=True)
            trace_filename = f"prof/test_moe/moe_trace_rank{rank}.json"
            # One prof.step() per config, so the schedule's step budget
            # must fit len(test_configs). Use wait=0/warmup=0 and size
            # active to cover every config; otherwise narrow sweeps
            # (e.g. EP_NTOKENS_MIN filtering down to 1-2 configs) never
            # reach the active window and the exported trace is empty.
            ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                          schedule=torch.profiler.schedule(wait=0, warmup=0,
                                                           active=max(1, len(test_configs)),
                                                           repeat=1),
                          record_shapes=True, profile_memory=True)
        else:
            ctx = nullcontext()
        with ctx as prof:
            for ntokens, hidden_dim, ffn_dim, concat_weights in test_configs:
                print(
                    f"Testing config: ntokens={ntokens}, hidden_dim={hidden_dim}, ffn_dim={ffn_dim}, concat_weights={concat_weights}"
                )

                # Split tokens across ranks
                token_splits = uniform_split_tokens(ntokens, num_ranks)
                token_splits_tensor = torch.tensor(token_splits, dtype=torch.int32, device="cuda")
                torch.distributed.broadcast(token_splits_tensor, src=0, group=EP_GROUP)
                token_splits = token_splits_tensor.tolist()

                # Prepare inputs
                weights, activations, grad_output = prepare_inputs(
                    ffn_dim,
                    hidden_dim,
                    args.topk,
                    token_splits,
                    args.num_experts,
                    rank,
                    num_ranks,
                    dtype=DTYPE,
                    device="cuda",
                    concat_weights=concat_weights,
                )

                torch.distributed.barrier(EP_GROUP)

                fc1_1, fc1_2, fc2 = weights
                hidden_states, gate_weights, expert_index, _, _ = activations

                # Benchmark functions
                def triton_dist_fwd():
                    return triton_dist_moe_func(weights, activations, EP_GROUP)

                def turbo_fwd():
                    return turbo_ep_moe(weights, activations, turbo_dispatcher)

                def zero_grads():
                    if fc1_1.grad is not None:
                        fc1_1.grad.zero_()
                    if fc1_2 is not None and fc1_2.grad is not None:
                        fc1_2.grad.zero_()
                    if fc2.grad is not None:
                        fc2.grad.zero_()
                    if hidden_states.grad is not None:
                        hidden_states.grad.zero_()
                    if gate_weights.grad is not None:
                        gate_weights.grad.zero_()

                def triton_dist_fwd_bwd():
                    output = triton_dist_moe_func(weights, activations, EP_GROUP)
                    output.backward(grad_output)
                    return output

                # Benchmark
                triton_dist_fwd_time, triton_dist_fwd_mem = (0.0, 0.0)
                triton_dist_fwd_bwd_time, triton_dist_fwd_bwd_mem = (0.0, 0.0)

                triton_dist_fwd_time, triton_dist_fwd_mem = benchmark_latency_memory(
                    triton_dist_fwd, args.iters, args.warmup)
                if not args.skip_backward:
                    triton_dist_fwd_bwd_time, triton_dist_fwd_bwd_mem = benchmark_latency_memory(
                        triton_dist_fwd_bwd, args.iters, args.warmup, pre_func=zero_grads)

                with torch.no_grad():
                    turbo_fwd_time, turbo_fwd_mem = benchmark_latency_memory(
                        turbo_fwd, args.iters, args.warmup)

                # Precision check: use the same inputs for both implementations
                # Prepare reference inputs once and clone for both implementations
                weights_ref, activations_ref, grad_output_ref = prepare_inputs(
                    ffn_dim,
                    hidden_dim,
                    args.topk,
                    token_splits,
                    args.num_experts,
                    rank,
                    num_ranks,
                    dtype=DTYPE,
                    device="cuda",
                    concat_weights=concat_weights,
                )

                torch.distributed.barrier(EP_GROUP)

                fc1_1_ref, fc1_2_ref, fc2_ref = weights_ref
                hidden_states_ref, gate_weights_ref, expert_index_ref, _, _ = activations_ref

                # Clone inputs for triton_dist (to avoid modifying original tensors)
                fc1_1_triton_dist = fc1_1_ref.clone().detach().requires_grad_(True)
                fc1_2_triton_dist = fc1_2_ref.clone().detach().requires_grad_(True) if fc1_2_ref is not None else None
                fc2_triton_dist = fc2_ref.clone().detach().requires_grad_(True)
                hidden_states_triton_dist = hidden_states_ref.clone().detach().requires_grad_(True)
                gate_weights_triton_dist = gate_weights_ref.clone().detach().requires_grad_(True)
                expert_index_triton_dist = expert_index_ref.clone()
                grad_output_triton_dist = grad_output_ref.clone()

                weights_triton_dist = (fc1_1_triton_dist, fc1_2_triton_dist, fc2_triton_dist)
                activations_triton_dist = (hidden_states_triton_dist, gate_weights_triton_dist,
                                           expert_index_triton_dist, token_splits, None)

                output_triton_dist = triton_dist_moe_func(weights_triton_dist, activations_triton_dist, EP_GROUP)
                if not args.skip_backward:
                    output_triton_dist.backward(grad_output_triton_dist)

                # Compare outputs and gradients
                # TODO: add torch as baseline
                fwd_precision = True
                bwd_precision = True

                # Create benchmark results structure
                implementations = {}

                implementations['triton_dist_fwd'] = {
                    'latency': triton_dist_fwd_time, 'memory': triton_dist_fwd_mem, 'precision':
                    fwd_precision if output_triton_dist is not None else 'N/A'
                }
                implementations['triton_dist_fwd_bwd'] = {
                    'latency': triton_dist_fwd_bwd_time, 'memory': triton_dist_fwd_bwd_mem, 'precision':
                    bwd_precision if output_triton_dist is not None else 'N/A'
                }
                implementations['turbo_fwd'] = {
                    'latency': turbo_fwd_time, 'memory': turbo_fwd_mem, 'precision': 'N/A'
                }

                config_key = (ntokens, hidden_dim, ffn_dim)
                all_implementations[config_key] = implementations

                if args.profile:
                    prof.step()
        if args.profile:
            prof.export_chrome_trace(trace_filename)
            if rank == 0:
                print(f"✅ torch trace saved to {trace_filename}")

        # AMD-only: drop every reference that pins symmetric-heap tensors
        # *before* rocshmem_finalize() runs. Otherwise pytorch's tensor
        # finalizers (autograd-saved tensors in particular) try to free
        # already-released symm-heap blocks during interpreter shutdown
        # and SIGSEGV. Same pattern as test_gemm_rs_intra_node.py.
        del (output_triton_dist, weights, activations, grad_output,
             weights_triton_dist, activations_triton_dist, grad_output_triton_dist,
             weights_ref, activations_ref, grad_output_ref)
        gc.collect()
        torch.cuda.synchronize()

    finally:
        # Cleanup
        deinit_triton_dist_ep_op()
        gc.collect()
        torch.cuda.synchronize()
        finalize_distributed()

    # Print comparison (only on rank 0)
    if rank == 0:
        print_benchmark_comparison(
            all_implementations, "Expert Parallel MoE", param_names=['Ntokens', 'Hidden', 'FFN'],
            title_params={'SM_margin': args.sm_margin, 'topk': args.topk, 'num_experts': args.num_experts})

        # Perf table: triton_dist throughput as a fraction of turbo_fwd
        # (turbo_lat / triton_dist_lat — lower latency is better, so <1.0x
        # means triton_dist is slower than turbo; e.g. 0.50x = half the perf).
        print()
        print("Perf vs turbo_fwd baseline (turbo_lat / triton_dist_lat; <1.0x = slower than turbo)")
        print(f"{'Ntokens':>8} {'Hidden':>8} {'FFN':>8} {'fwd_perf':>12} {'fwd_bwd_perf':>14}")
        print("=" * 56)
        for config_key, impls in all_implementations.items():
            ntokens_v, hidden_v, ffn_v = config_key
            turbo_lat = impls.get('turbo_fwd', {}).get('latency')
            td_fwd_lat = impls.get('triton_dist_fwd', {}).get('latency')
            td_fb_lat = impls.get('triton_dist_fwd_bwd', {}).get('latency')
            fwd_p = f"{turbo_lat / td_fwd_lat:.3f}x" if turbo_lat and td_fwd_lat else "N/A"
            fb_p = f"{turbo_lat / td_fb_lat:.3f}x" if turbo_lat and td_fb_lat else "N/A"
            print(f"{ntokens_v:>8} {hidden_v:>8} {ffn_v:>8} {fwd_p:>12} {fb_p:>14}")

    if args.enable_kernel_profiler:
        _print_kernel_profile_summary(rank)


if __name__ == "__main__":
    main()
