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
"""Public re-exports for triton_dist.function.

Dispatches between the NVIDIA and AMD implementations of the fused EP MoE
op based on the current backend (``is_cuda()`` vs ``is_hip()``).
"""

from triton_dist.utils import is_cuda, is_hip

if is_cuda():
    from .nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction  # noqa: F401
    from .nvidia.common import (  # noqa: F401
        init_triton_dist_ep_op,
        deinit_triton_dist_ep_op,
        set_triton_dist_moe_profile_enabled,
        get_triton_dist_ep_stream,
    )
elif is_hip():
    from .amd.ep_moe_fused import TritonDistFusedEpMoeFunction  # noqa: F401
    from .amd.common import (  # noqa: F401
        init_triton_dist_ep_op,
        deinit_triton_dist_ep_op,
        set_triton_dist_moe_profile_enabled,
        get_triton_dist_ep_stream,
    )

