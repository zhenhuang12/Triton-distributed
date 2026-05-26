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

import argparse
import datetime
import os
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.profiler import profile, ProfilerActivity

from triton_dist.function.nvidia.common import init_triton_dist_ep_op, deinit_triton_dist_ep_op
from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction
from triton_dist.utils import finalize_distributed, init_nvshmem_by_torch_process_group
from triton_dist.profiler_utils import benchmark_latency_memory, print_benchmark_comparison

import primus_turbo.pytorch as turbo
from primus_turbo.pytorch.ops import grouped_gemm


def parse_args():
    parser = argparse.ArgumentParser(description="Test Expert Parallel MoE implementations")
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
    
    Args:
        ffn_dim: FFN intermediate dimension
        hidden_dim: Hidden dimension
        topk: Top-K experts per token
        ntokens_per_rank_list: List of token counts per rank
        num_experts: Total number of experts
        rank: Current rank
        num_ranks: Total number of ranks
        dtype: Data type
        device: Device
    
    Returns:
        Tuple of (weights, activations, grad_output)
        - weights: (fc1_1, fc1_2, fc2) - expert weights
        - activations: (hidden_states, gate_weights, expert_index, ntokens_per_rank_list)
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

    # Randomly drop some tokens (set expert_index to num_experts)
    random_drop_tokens = torch.randint(0, 10, [ntokens_per_rank_list[rank], topk], device=device)
    expert_index = expert_index.masked_fill(random_drop_tokens > 9, num_experts)

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
        (hidden_states, gate_weights, expert_index, ntokens_per_rank_list),
        grad_output,
    )


def triton_dist_moe_func(weights, activations, ep_group):
    """Ditron MoE forward function"""
    fc1_1, fc1_2, fc2 = weights
    hidden_states, gate_weights, expert_index, ntokens_per_rank_list = activations
    assert len(fc1_1.shape) == 3
    num_experts = fc1_1.shape[0] * ep_group.size()  # total experts
    output = TritonDistFusedEpMoeFunction.apply(num_experts, gate_weights, expert_index, hidden_states, fc1_1, fc1_2,
                                                fc2, ep_group)
    return output


def turbo_ep_moe(weights, activations, dispatcher):
    """Baseline EP-MoE forward: turbo TokenDispatcher + turbo grouped_gemm + SwiGLU.

    expert_index from activations is forwarded to dispatcher.token_dispatch via
    `indices=`, so the topk routing decision is byte-identical to what the fused
    kernel sees — only the dispatch / compute path differs.
    """
    fc1_1, fc1_2, fc2 = weights
    hidden_states, gate_weights, expert_index, _ = activations
    num_experts = dispatcher.num_experts
    ntokens = hidden_states.shape[0]

    # token_dispatch expects probs shaped [ntokens, num_experts]; scatter the
    # per-topk gate_weights back into a dense tensor at the expert_index slots.
    # Mask out drops (expert_index == num_experts) by zeroing both index and weight.
    valid = expert_index < num_experts
    safe_idx = expert_index.long().masked_fill(~valid, 0)
    probs = torch.zeros(ntokens, num_experts, dtype=torch.float32, device=hidden_states.device)
    probs.scatter_(1, safe_idx, gate_weights.float() * valid.float())

    permuted_hidden, tokens_per_expert, permuted_probs = dispatcher.token_dispatch(
        hidden_states, probs, indices=safe_idx
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
        init_nvshmem_by_torch_process_group(EP_GROUP)
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
    max_tokens_per_rank = 8192 * 4
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
    )

    # Test configurations
    test_configs = []
    ntokens_list = [1024, 2048, 4096, 8192, 8192 * 2, 8192 * 4, 8192 * 8, 8192 * 16, 8192 * 32]
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
            ctx = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                          schedule=torch.profiler.schedule(wait=1, warmup=2, active=5,
                                                           repeat=1), record_shapes=True, profile_memory=True)
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
                hidden_states, gate_weights, expert_index, _ = activations

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
                hidden_states_ref, gate_weights_ref, expert_index_ref, _ = activations_ref

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
                                           expert_index_triton_dist, token_splits)

                output_triton_dist = triton_dist_moe_func(weights_triton_dist, activations_triton_dist, EP_GROUP)
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

    finally:
        # Cleanup
        deinit_triton_dist_ep_op()
        finalize_distributed()

    # Print comparison (only on rank 0)
    if rank == 0:
        print_benchmark_comparison(
            all_implementations, "Expert Parallel MoE", param_names=['Ntokens', 'Hidden', 'FFN'],
            title_params={'SM_margin': args.sm_margin, 'topk': args.topk, 'num_experts': args.num_experts})

        # Ratio table: triton_dist vs turbo_fwd baseline (denominator = turbo_fwd)
        print()
        print(f"{'Ntokens':>8} {'Hidden':>8} {'FFN':>8} {'fwd_ratio':>12} {'fwd_bwd_ratio':>14}")
        print("=" * 56)
        for config_key, impls in all_implementations.items():
            ntokens_v, hidden_v, ffn_v = config_key
            turbo_lat = impls.get('turbo_fwd', {}).get('latency')
            td_fwd_lat = impls.get('triton_dist_fwd', {}).get('latency')
            td_fb_lat = impls.get('triton_dist_fwd_bwd', {}).get('latency')
            fwd_r = f"{td_fwd_lat / turbo_lat:.3f}x" if turbo_lat and td_fwd_lat else "N/A"
            fb_r = f"{td_fb_lat / turbo_lat:.3f}x" if turbo_lat and td_fb_lat else "N/A"
            print(f"{ntokens_v:>8} {hidden_v:>8} {ffn_v:>8} {fwd_r:>12} {fb_r:>14}")


if __name__ == "__main__":
    main()
