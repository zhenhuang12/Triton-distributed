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
AMD EP A2A kernels and helpers.
Provides bincount and re-exports intra-node kernels from ep_a2a_intra_node.
"""

import torch
import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra.hip.language_extra import tid, ld, st, atomic_add, __syncthreads
from triton_dist.language.extra.language_extra import threads_per_warp


@triton_dist.jit(do_not_specialize=["n", "length", "num_sm"])
def kernel_bincount(n, input, output, length, num_sm, num_warps: tl.constexpr):
    """
    GPU bincount: count occurrences of each index in [0, length). AMD version using tid(0)
    and fixed threads_per_block (no simt_exec_region). Same semantics as nvidia/ep_a2a.py.
    """
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    thread_idx = tid(0)
    threads_per_block = num_warps * threads_per_warp()
    for i in range(pid * threads_per_block + thread_idx, n, num_pid * threads_per_block):
        val = ld(input + i)
        if val < length:
            atomic_add(output + val, 1, scope="agent", semantic="relaxed")


def bincount(input_tensor, length, output=None, output_dtype=torch.int32, num_sm=16, num_warps=8):
    """GPU bincount for AMD (no AOT). input_tensor: 1D int32 on device; output: length elements."""
    if output is None:
        output = torch.zeros(length, dtype=output_dtype, device=input_tensor.device)
    assert input_tensor.dim() == 1 and input_tensor.is_contiguous()
    assert output.size(0) >= length and output.dtype == output_dtype
    n = input_tensor.size(0)
    grid = (num_sm, )
    kernel_bincount[grid](n, input_tensor, output, length, num_sm, num_warps=num_warps)
    return output


@triton_dist.jit(do_not_specialize=["num_tokens", "max_num_tokens", "num_sms"])
def kernel_get_dispatch_send_reqs(
    workspace,  # [NUM_SM * BLOCK_SIZE]
    exp_indices,  # [num_tokens, topk]
    send_reqs_for_nodes,  # [nnodes, 2, max_num_tokens], init with -1
    num_tokens: int,
    max_num_tokens: int,
    experts_per_rank: int,
    num_sms: int,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    local_world_size: tl.constexpr,
):
    """AMD port of :func:`kernel_get_dispatch_send_reqs` from kernels/nvidia/ep_a2a.py.

    Same algorithm; only differs in the language-extra import set (HIP ``ld``/``st``
    instead of CUDA PTX). No AOT entry point.
    """
    expert_per_node = experts_per_rank * local_world_size

    world_size = dl.num_ranks()
    rank = dl.rank()
    nnodes = world_size // local_world_size
    cur_node_id = rank // local_world_size
    total_experts = experts_per_rank * world_size
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    tiles_per_node = tl.cdiv(num_tokens, BLOCK_SIZE)
    num_tiles = tiles_per_node * nnodes
    PADDED_TOPK: tl.constexpr = triton.next_power_of_2(topk)
    offs_token = tl.arange(0, BLOCK_SIZE)
    offs_topk = tl.arange(0, PADDED_TOPK)
    thread_idx = tid(0)
    for tile_id in range(pid, num_tiles, num_pid):
        target_node_id = tile_id // tiles_per_node
        tile_id_token = tile_id % tiles_per_node
        # skip current node (no rdma request)
        if target_node_id != cur_node_id:
            offs = (offs_token[:, None] + tile_id_token * BLOCK_SIZE) * topk + offs_topk[None, :]
            mask = (offs_token[:, None] + tile_id_token * BLOCK_SIZE < num_tokens) & (offs_topk[None, :] < topk)
            # the `other` should be equal to or greater than total_experts
            expert_idx = tl.load(exp_indices + offs, mask=mask, other=total_experts)
            node_idx = expert_idx // expert_per_node
            send_token_mask = tl.where(node_idx == target_node_id, 1,
                                       0).to(exp_indices.dtype.element_ty)  # [BLOCK_SIZE, PADDED_TOPK]
            send_token_mask = tl.sum(send_token_mask, axis=1)  # [BLOCK_SIZE, ]
            send_token_mask = tl.where(send_token_mask > 0, 1, 0).to(exp_indices.dtype.element_ty)  # [BLOCK_SIZE, ]
            tl.store(workspace + pid * BLOCK_SIZE + offs_token, send_token_mask)
            __syncthreads()

            if thread_idx == 0:
                num_tokens_cur_tile = min(num_tokens - tile_id_token * BLOCK_SIZE, BLOCK_SIZE)
                token_start = -1
                token_end = 0
                has_start = False
                token_mask_base_ptr = workspace + pid * BLOCK_SIZE
                send_reqs_base_ptr = send_reqs_for_nodes + target_node_id * 2 * max_num_tokens
                cnt = 0
                for i in range(num_tokens_cur_tile):
                    token_mask = ld(token_mask_base_ptr + i)
                    if token_mask == 0:
                        if has_start:
                            has_start = False
                            token_end = i + tile_id_token * BLOCK_SIZE
                            st(send_reqs_base_ptr + tile_id_token * BLOCK_SIZE + cnt + max_num_tokens, token_end)
                            cnt += 1
                    else:
                        if not has_start:
                            has_start = True
                            token_start = i + tile_id_token * BLOCK_SIZE
                            st(send_reqs_base_ptr + tile_id_token * BLOCK_SIZE + cnt, token_start)
                if has_start:
                    st(send_reqs_base_ptr + tile_id_token * BLOCK_SIZE + cnt + max_num_tokens,
                       num_tokens_cur_tile + tile_id_token * BLOCK_SIZE)
            __syncthreads()


def get_dispatch_send_reqs(exp_indices, send_reqs_for_nodes, experts_per_rank, local_world_size, num_sms):
    """AMD port of :func:`get_dispatch_send_reqs` (no AOT branch)."""
    BLOCK_SIZE = 256
    workspace = torch.empty((num_sms * BLOCK_SIZE), dtype=exp_indices.dtype, device=exp_indices.device)
    max_num_tokens = send_reqs_for_nodes.shape[-1]
    assert send_reqs_for_nodes.dtype == exp_indices.dtype
    num_tokens, topk = exp_indices.shape
    num_warps = 8
    kernel_get_dispatch_send_reqs[(num_sms, )](
        workspace,
        exp_indices,
        send_reqs_for_nodes,
        num_tokens,
        max_num_tokens,
        experts_per_rank,
        num_sms,
        topk=topk,
        BLOCK_SIZE=BLOCK_SIZE,
        local_world_size=local_world_size,
        num_warps=num_warps,
    )


# Re-export intra-node kernels and helpers so layer can import from this module only.
from triton_dist.kernels.amd.ep_a2a_intra_node import (
    kernel_combine_token_intra_node,
    kernel_dispatch_token_intra_node,
    get_ag_splits_and_recv_offset_for_dispatch_intra_node,
    kernel_skipped_token_local_dispatch_intra_node,
    kernel_skipped_token_inplace_local_combine_intra_node,
)

__all__ = [
    "kernel_bincount",
    "bincount",
    "kernel_get_dispatch_send_reqs",
    "get_dispatch_send_reqs",
    "kernel_combine_token_intra_node",
    "kernel_dispatch_token_intra_node",
    "get_ag_splits_and_recv_offset_for_dispatch_intra_node",
    "kernel_skipped_token_local_dispatch_intra_node",
    "kernel_skipped_token_inplace_local_combine_intra_node",
]
