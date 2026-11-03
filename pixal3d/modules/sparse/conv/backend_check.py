"""Which sparse-convolution backend a card below sm_80 runs: flex_gemm's Triton kernels where they give spconv's
answers on this card, spconv otherwise - decided once, at the install's calibration (pixal3d.calibrate), and kept
in the calibration record's "conv_backend" section, which modules/precision.py reads at start.

flex_gemm's masked implicit GEMM is a Triton kernel: fp16 tl.dot runs on sm_75 (Turing) under Linux, but Triton's
Windows builds are untested on those cards, so nothing may depend on it without this check. spconv's CUDA kernels
run everywhere from sm_61; they are the fallback and the reference here. Where flex_gemm serves, the decoders
decode in boxes (models/sc_vaes/chunked.py: a flex_gemm-only path) and run 3-30x faster (an RTX 3060 on the
turing profile, boy at 1024: upsample 11.7 -> 0.4 s).
"""

from __future__ import annotations

import time
from typing import Any

import torch

TOLERANCE = 1e-2        # relative L2 between the two backends' fp16 answers (their rounding differs; ~1e-3 measured)


def compare(device: Any, voxels: int = 30_000, grid: int = 64, channels: int = 64,
            dtype: torch.dtype = torch.float16) -> dict:
    """One submanifold 3x3x3 convolution of random voxels under spconv and under flex_gemm with the same weights:
    {"backend": the one to use, "rel": their relative L2 distance, "<backend>_ms": the time of a call, "why"}."""
    from .. import SparseTensor, config
    from .conv import SparseConv3d

    before = config.CONV
    g = torch.Generator(device=device).manual_seed(0)
    ijk = torch.randint(0, grid, (voxels, 3), device=device, dtype=torch.int32, generator=g)
    coords = torch.cat([torch.zeros(voxels, 1, device=device, dtype=torch.int32), ijk], 1).unique(dim=0)
    feats = torch.randn(coords.shape[0], channels, device=device, dtype=dtype, generator=g)
    found: dict = {}
    outs: dict = {}
    weight = bias = None
    try:
        for backend in ("spconv", "flex_gemm"):
            try:
                config.set_conv_backend(backend)
                conv = SparseConv3d(channels, channels, 3).to(device=device, dtype=dtype)
                with torch.no_grad():
                    if weight is None:
                        weight, bias = conv.weight.detach().clone(), conv.bias.detach().clone()
                    else:
                        conv.weight.copy_(weight)
                        conv.bias.copy_(bias)
                    x = SparseTensor(feats, coords)
                    y = conv(x).feats
                    torch.cuda.synchronize(device)
                    times = []
                    for _ in range(3):
                        began = time.perf_counter()
                        y = conv(SparseTensor(feats, coords)).feats
                        torch.cuda.synchronize(device)
                        times.append(time.perf_counter() - began)
                outs[backend] = y.float()
                found[f"{backend}_ms"] = round(1000 * sorted(times)[1], 2)
            except Exception as exc:        # noqa: BLE001 - a backend that cannot run here is simply not chosen
                found[f"{backend}_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        config.set_conv_backend(before)
    if "spconv" not in outs:
        found.update(backend="flex_gemm" if "flex_gemm" in outs else "spconv",
                     why="spconv could not run the check" + ("" if "flex_gemm" in outs else "; nor flex_gemm"))
        return found
    if "flex_gemm" not in outs:
        found.update(backend="spconv", why="flex_gemm's kernels do not run on this card")
        return found
    rel = float((outs["flex_gemm"] - outs["spconv"]).norm() / outs["spconv"].norm().clamp_min(1e-30))
    found["rel"] = rel
    if not torch.isfinite(outs["flex_gemm"]).all() or rel > TOLERANCE:
        found.update(backend="spconv", why=f"flex_gemm's answers differ from spconv's (relative L2 {rel:.3g})")
    else:
        found.update(backend="flex_gemm", why=f"flex_gemm's kernels give spconv's answers (relative L2 {rel:.3g})")
    return found
