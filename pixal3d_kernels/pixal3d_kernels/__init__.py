"""Spellcast3D's own CUDA kernels for the Pixal3D fork (README.md beside setup.py: the build, the contract, the figures).

    int8_gemm_group(xq, xs, wq, ws, bias=None, gelu=False, out_dtype=torch.float16)  -> y [M, N]
    quantize_rows(x, group=0, inv_smooth=None)                                         -> (q [M, K], s)
    modulate(h, scale1, shift, out_dtype)                                              -> (h * scale1 + shift) cast
    residual_gated(x, h, gate)                                                         -> x + float(h) * gate
    qk_norm_rope(x, gamma, scale, eps=1e-12, phases=None, out_dtype=None)              -> the q / k RMS norm (+ RoPE)
    torch.ops.pixal3d_kernels.int8_gemm_group80(...)  (sm_80+)  -> the Triton group GEMM's exact twin (see below)

The flows' W8A8 linear on the cards Triton's int8 GEMM does not serve: y = (sum over k of xq * wq) * xs * ws + bias,
then GELU(tanh) when asked, exactly int8mm.reference()'s semantics. xq int8 [M, K], xs fp32 [M] (one scale per row)
or [M, K // 128] (per row and 128-K group), wq int8 [N, K], ws fp32 [N] (per output channel) or [N, K // 128] (per
channel and group, column-major as int8mm.quantize_weight(group=128) lays it out; another layout is copied), bias fp32
[N] or None; out_dtype float16 / float32 / bfloat16, or int32 for the raw integer sum (per-row scales, no bias / GELU:
the checks). K % 128 == 0, N % 16 == 0, any M >= 0. The integer products are exact (int32
per group, or over the whole K with per-row scales); a group's sum is scaled into an fp32 accumulator.

The element-wise kernels give the bits of the eager torch chains they stand for (README.md, "The element-wise
chains"); they run on every card from sm_61.

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


def has(name: str) -> bool:
    """Whether this build has the operator `name` (an older wheel has the GEMM alone)."""
    try:
        getattr(torch.ops.pixal3d_kernels, name)
        return True
    except (AttributeError, RuntimeError):
        return False


def int8_gemm_group(xq: torch.Tensor, xs: torch.Tensor, wq: torch.Tensor, ws: torch.Tensor,
                    bias: Optional[torch.Tensor] = None, gelu: bool = False,
                    out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    return torch.ops.pixal3d_kernels.int8_gemm_group(xq, xs, wq, ws, bias, gelu, out_dtype)


# sm_80+: int8_gemm_group80(xq, xs, wq, ws, bias, gelu, out, rx, rg, r2, tile, group_m) -> (y, qs), the flows'
# group-scaled GEMM with modules/int8mm.py's Triton kernel's bits (csrc/int8_gemm_sm80.cu; int8mm.group_cuda calls it)
if has("int8_gemm_group80"):
    @torch.library.register_fake("pixal3d_kernels::int8_gemm_group80")
    def _int8_gemm_group80_fake(xq, xs, wq, ws, bias, gelu, out, rx, rg, r2, tile, group_m):
        y = xq.new_empty((xq.shape[0], wq.shape[0]), dtype={2: torch.int8, 3: torch.float16}.get(out, torch.bfloat16))
        qs = xq.new_empty((xq.shape[0], wq.shape[0] // 128) if out == 2 else (0,), dtype=torch.float32)
        return y, qs


def quantize_rows(x: torch.Tensor, group: int = 0,
                  inv_smooth: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., K] (fp16 / fp32 / bf16) -> (int8 [M, K], fp32 scales [M] or [M, K // group])."""
    return torch.ops.pixal3d_kernels.quantize_rows(x.reshape(-1, x.shape[-1]), int(group), inv_smooth)


def modulate(h: torch.Tensor, scale1: torch.Tensor, shift: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """(h * scale1 + shift).to(out_dtype), h fp32 [..., C], scale1 / shift one row [C] (or [1, C]) of one dtype."""
    return torch.ops.pixal3d_kernels.modulate(h, scale1, shift, out_dtype)


def residual_gated(x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """x + h.float() * gate, x fp32 [..., C], h [..., C], gate one row [C] (or [1, C])."""
    return torch.ops.pixal3d_kernels.residual_gated(x, h, gate)


def qk_norm_rope(x: torch.Tensor, gamma: torch.Tensor, scale: float, eps: float = 1e-12,
                 phases: Optional[torch.Tensor] = None, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """x [T, H, 128] (the last dimension contiguous) -> (F.normalize(x.float(), dim=-1) * gamma * scale) in out_dtype
    (x's by default), then, with phases (complex64 [T, 64]), the rotary embedding of that result in out_dtype."""
    return torch.ops.pixal3d_kernels.qk_norm_rope(x, gamma, float(scale), float(eps), phases,
                                                  out_dtype or x.dtype)
