"""Spellcast3D's own CUDA kernels for the Pixal3D fork (README.md beside setup.py: the build, the contract, the figures).

    int8_gemm_group(xq, xs, wq, ws, bias=None, gelu=False, out_dtype=torch.float16)  -> y [M, N]

The flows' W8A8 linear on the cards Triton's int8 GEMM does not serve: y = (sum over k of xq * wq) * xs * ws + bias,
then GELU(tanh) when asked, exactly int8mm.reference()'s semantics. xq int8 [M, K], xs fp32 [M] (one scale per row)
or [M, K // 128] (per row and 128-K group), wq int8 [N, K], ws fp32 [N] (per output channel) or [N, K // 128] (per
channel and group, column-major as int8mm.quantize_weight(group=128) lays it out; another layout is copied), bias fp32
[N] or None; out_dtype float16 / float32 / bfloat16, or int32 for the raw integer sum (per-row scales, no bias / GELU:
the checks). K % 128 == 0, N % 16 == 0, any M >= 0. The integer products are exact (int32
per group, or over the whole K with per-row scales); a group's sum is scaled into an fp32 accumulator.

The kernel by the card: sm_75 and newer the int8 tensor cores (mma.sync m8n8k16), sm_61 .. sm_72 dp4a; below sm_61 an
error. `torch.ops.pixal3d_kernels.int8_gemm_group` takes two more arguments for developers: raster (the tile order, 0
= the default) and kernel (0 = by the card, 1 = tensor cores, 2 = dp4a).
"""

from __future__ import annotations

from typing import Optional

import torch

from . import _C  # noqa: F401  (registers torch.ops.pixal3d_kernels)

GROUP = 128     # input channels per group scale


@torch.library.register_fake("pixal3d_kernels::int8_gemm_group")
def _int8_gemm_group_fake(xq, xs, wq, ws, bias, gelu, out_dtype, raster=0, kernel=0):
    # the output's shape and dtype without running anything (meta tensors, torch.compile's tracing)
    return xq.new_empty((xq.shape[0], wq.shape[0]), dtype=out_dtype)


def int8_gemm_group(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
                    bias: Optional[torch.Tensor] = None, gelu: bool = False,
                    out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return torch.ops.pixal3d_kernels.int8_gemm_group(xq, xs, wq, ws, bias, gelu, out_dtype)
