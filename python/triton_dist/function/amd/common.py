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
"""AMD-side mirror of :mod:`triton_dist.function.nvidia.common`.

The pure-Python state (profile flags, stream/op getters, optim configs) is
ported verbatim so that callers written against the NVIDIA module can rely
on the same surface on HIP. The kernel-backed init/deinit/context helpers
are stubbed with :class:`NotImplementedError` until the AMD fused EP+MoE
backend (counterpart of ``layers.nvidia.ep_a2a_fused_layer.EpAll2AllFusedOp``
and ``kernels.nvidia.group_gemm``) lands.
"""

import torch
from dataclasses import dataclass

try:
    from torch.amp import custom_bwd as torch_custom_bwd
    from torch.amp import custom_fwd as torch_custom_fwd

    CUSTOM_FWD_BWD_EXTRA_KWARGS = {"device_type": "cuda"}
except ImportError:
    from torch.cuda.amp import custom_bwd as torch_custom_bwd
    from torch.cuda.amp import custom_fwd as torch_custom_fwd

    CUSTOM_FWD_BWD_EXTRA_KWARGS = {}


def custom_fwd(*args, **kwargs):
    return torch_custom_fwd(*args, **kwargs, **CUSTOM_FWD_BWD_EXTRA_KWARGS)


def custom_bwd(*args, **kwargs):
    return torch_custom_bwd(*args, **kwargs, **CUSTOM_FWD_BWD_EXTRA_KWARGS)


# Global context (mirrors the NVIDIA module). The fused EP-MoE op is not yet
# wired on AMD; ``triton_dist_ep_op*`` stay ``None`` and the init helpers
# raise.

MAX_TOKENS_PER_RANK = None
DITRON_EP_STREAM = None

triton_dist_ep_op = None
triton_dist_ep_op1 = None
triton_dist_ep_op2 = None
triton_dist_ep_op_bwd = None

DITRON_EP_STREAM_1 = None
DITRON_EP_STREAM_2 = None
fwd_dispatch1_event = None
bwd_dispatch_event = None
bwd_combine_event = None

PROFILE_DITRONT_MOE_FWD_DISPATCH = False
PROFILE_DITRONT_MOE_FWD_COMBINE = False
PROFILE_DITRONT_MOE_BWD_DISPATCH = False
PROFILE_DITRONT_MOE_BWD_COMBINE = False

DITRON_PROFILE_OUTPUT_DIR = "prof/mega"


def set_triton_dist_moe_profile_enabled(
    enabled: bool,
    fwd_dispatch: bool = True,
    fwd_combine: bool = True,
    bwd_dispatch: bool = True,
    bwd_combine: bool = True,
    output_dir: str = None,
) -> None:
    global PROFILE_DITRONT_MOE_FWD_DISPATCH, PROFILE_DITRONT_MOE_FWD_COMBINE
    global PROFILE_DITRONT_MOE_BWD_DISPATCH, PROFILE_DITRONT_MOE_BWD_COMBINE
    global DITRON_PROFILE_OUTPUT_DIR

    PROFILE_DITRONT_MOE_FWD_DISPATCH = enabled and fwd_dispatch
    PROFILE_DITRONT_MOE_FWD_COMBINE = enabled and fwd_combine
    PROFILE_DITRONT_MOE_BWD_DISPATCH = enabled and bwd_dispatch
    PROFILE_DITRONT_MOE_BWD_COMBINE = enabled and bwd_combine

    if output_dir is not None:
        DITRON_PROFILE_OUTPUT_DIR = output_dir


def get_triton_dist_moe_profile_enabled() -> dict:
    return {
        "fwd_dispatch": PROFILE_DITRONT_MOE_FWD_DISPATCH,
        "fwd_combine": PROFILE_DITRONT_MOE_FWD_COMBINE,
        "bwd_dispatch": PROFILE_DITRONT_MOE_BWD_DISPATCH,
        "bwd_combine": PROFILE_DITRONT_MOE_BWD_COMBINE,
        "output_dir": DITRON_PROFILE_OUTPUT_DIR,
    }


def get_triton_dist_profile_output_dir() -> str:
    return DITRON_PROFILE_OUTPUT_DIR


_NOT_IMPLEMENTED_MSG = (
    "Fused EP-MoE op is not yet implemented on AMD. The AMD counterpart of "
    "``triton_dist.layers.nvidia.ep_a2a_fused_layer.EpAll2AllFusedOp`` and "
    "``triton_dist.kernels.nvidia.group_gemm`` have not been ported. Use the "
    "non-fused dispatch/combine path in ``triton_dist.layers.amd.ep_a2a_layer`` "
    "instead, or contribute the fused backend."
)


class TritonDistEpContext:
    """Stub. Real context is created by :func:`init_triton_dist_ep_ctx` once
    the AMD fused backend lands."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def triton_dist_ep_op_initialized(ep_implementation: str = "mega"):
    if ep_implementation in ["mega", "mega_recomp"]:
        return triton_dist_ep_op is not None
    if ep_implementation == "split_mbs":
        return triton_dist_ep_op1 is not None and triton_dist_ep_op2 is not None
    raise ValueError(
        f"Invalid ep_implementation: {ep_implementation}, expected: ['mega', 'mega_recomp', 'split_mbs']")


def init_triton_dist_ep_op(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def deinit_triton_dist_ep_op(ep_implementation: str = "mega"):
    """No-op on AMD: nothing was allocated."""
    global triton_dist_ep_op, triton_dist_ep_op1, triton_dist_ep_op2
    global MAX_TOKENS_PER_RANK
    if ep_implementation in ["mega", "mega_recomp"]:
        MAX_TOKENS_PER_RANK = None
        triton_dist_ep_op = None
    elif ep_implementation == "split_mbs":
        MAX_TOKENS_PER_RANK = None
        triton_dist_ep_op1 = None
        triton_dist_ep_op2 = None
    else:
        raise ValueError(
            f"Invalid ep_implementation: {ep_implementation}, expected: ['mega', 'mega_recomp', 'split_mbs']")


def init_triton_dist_ep_ctx(*args, **kwargs):
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def get_ep_capacity(ep_implementation: str = "mega"):
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def get_triton_dist_ep_stream(idx=0):
    if idx == 0:
        if DITRON_EP_STREAM is None:
            print("Warning: DITRON_EP_STREAM is not initialized.")
        return DITRON_EP_STREAM
    if idx == 1:
        if DITRON_EP_STREAM_1 is None:
            print("Warning: DITRON_EP_STREAM_1 is not initialized.")
        return DITRON_EP_STREAM_1
    if idx == 2:
        if DITRON_EP_STREAM_2 is None:
            print("Warning: DITRON_EP_STREAM_2 is not initialized.")
        return DITRON_EP_STREAM_2
    raise ValueError(f"Invalid idx: {idx}, expected: [0, 1, 2]")


def get_triton_dist_ep_op(idx=0):
    if idx == 0:
        return triton_dist_ep_op
    if idx == 1:
        return triton_dist_ep_op1
    if idx == 2:
        return triton_dist_ep_op2
    raise ValueError(f"Invalid idx: {idx}, expected: [0, 1, 2]")


@dataclass
class MoEOptimConfig:
    num_build_sms: int
    num_copy_sms: int
    num_group_gemm_warps: int
    num_dispatch_warps: int
    num_combine_warps: int
    num_dispatch_sms: int
    num_tail_sms_in_dispatch: int
    num_combine_sms: int
    num_reduce_sms_in_combine: int
    dispatch_use_block_wise_barrier: bool


def get_moe_optim_config(use_mega: bool = False, is_forward: bool = True):
    """Conservative defaults for MI300/MI355.

    Mirrors the NVIDIA shape of returning a populated :class:`MoEOptimConfig`
    so callers can read the fields, but the numbers have not been tuned for
    CDNA3/4. Treat as a placeholder until the AMD fused backend lands and we
    can sweep the configuration space.
    """
    max_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if is_forward:
        return MoEOptimConfig(
            num_build_sms=8,
            num_copy_sms=max_sms,
            num_group_gemm_warps=8,
            num_dispatch_warps=16 if use_mega else 32,
            num_combine_warps=32,
            num_dispatch_sms=min(80, max_sms),
            num_tail_sms_in_dispatch=32 if use_mega else 0,
            num_combine_sms=min(80, max_sms),
            num_reduce_sms_in_combine=min(80, max_sms) if use_mega else 0,
            dispatch_use_block_wise_barrier=use_mega,
        )
    return MoEOptimConfig(
        num_build_sms=8,
        num_copy_sms=max_sms,
        num_group_gemm_warps=8,
        num_dispatch_warps=32,
        num_combine_warps=16 if use_mega else 32,
        num_dispatch_sms=min(64, max_sms),
        num_tail_sms_in_dispatch=16 if use_mega else 0,
        num_combine_sms=min(64, max_sms),
        num_reduce_sms_in_combine=100 if use_mega else 0,
        dispatch_use_block_wise_barrier=use_mega,
    )
