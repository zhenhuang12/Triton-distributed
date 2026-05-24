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
"""AMD stub for the fused EP-MoE autograd function.

The NVIDIA implementation in
:mod:`triton_dist.function.nvidia.ep_moe_fused` chains
``mega_dispatch_group_gemm`` / SwiGLU / ``mega_group_gemm_combine`` from
``EpAll2AllFusedOp``. The AMD ports of those kernels have not landed yet
(see ``triton_dist.layers.amd.ep_a2a_layer`` for the available non-fused
path); calling :meth:`TritonDistFusedEpMoeFunction.forward` here raises a
clear error so users do not silently fall through to an unrelated backend.
"""

import torch

from .common import custom_fwd, custom_bwd


_NOT_IMPLEMENTED_MSG = (
    "TritonDistFusedEpMoeFunction is not yet implemented on AMD. The fused "
    "dispatch+GEMM+combine pipeline (``mega_dispatch_group_gemm`` / "
    "``mega_group_gemm_combine``) has not been ported to HIP. Use the "
    "non-fused ``triton_dist.layers.amd.ep_a2a_layer.EPAll2AllLayer`` for "
    "now."
)


class TritonDistFusedEpMoeFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        num_experts: int,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        hidden_states: torch.Tensor,
        fc1_1: torch.Tensor,
        fc1_2: torch.Tensor,
        fc2: torch.Tensor,
        ep_group,
    ):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
