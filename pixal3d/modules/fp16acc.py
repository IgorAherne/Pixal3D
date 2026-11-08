"""The ss and lr flows' token-sized linears in fp16 with fp16 accumulation, behind the switch PIXAL3D_FP16ACC: OFF by
default (an opt-in precision change, not a default).

GeForce Ampere and newer cards run fp16 x fp16 tensor-core products that accumulate in fp16 at twice the rate of
the ones that accumulate in fp32, which is how every bf16 product runs (data-centre and workstation cards run both at
one rate: there the switch costs precision and gains nothing). Measured on an RTX 3060 at the ss flow's shapes: the
products at 42-47.6 TFLOPS against 26-27.7 for bf16 (1.6-1.8x rather than 2x: the card's power cap holds the clock at
1.75 GHz); with the casts and the pass after the product a layer is 1.2-1.6x faster at the ss flow's 4,096 tokens,
1.1-1.4x at the boy's 1,579 of lr, a whole ss pass 16 % and an lr pass 13 %. The cost is precision: on the flows' real
inputs the fp16-accumulated products are 0.4-1.0x bf16's own output rounding away from an fp64 reference where the
layer's inner dimension is 2048 or less (1.1-1.4x once rounded to bf16 as well), 1.8-2.9x on the MLP's second layer
(fc2: 8192 products per sum; 2.7-12.8x in an earlier measurement), which therefore stays in bf16. The flows' values
fit fp16's range with ~18x to spare (the largest input of a linear seen 430, output 3,696). The ss and lr samplers
amplify any rounding change into another sample (the sparse structure's occupancy flips first), so the switch changes
the props: it is judged end to end, as every precision change was.

Converted, per block of the ss and lr flows: the self-attention's to_qkv and to_out, the cross-attention's to_q and
to_out, the MLP's first layer (its GELU done in the same pass), proj_linear. Not converted: fc2, the cross-attention's
to_kv (five context tokens), the modulation and the input / output layers (a few rows each), the hr and tex flows
(their int8 GEMMs, modules/int8.py). The switch is read when `convert` runs (the staged engine's convert hook, after
SageAttention's fold); a network already converted stays as it is:

    "on", "1", "true", "yes"     the ss and lr flows on the ampere profile (modules/precision.py); on the turing and
                                 pascal profiles nothing is converted (one line says so)
    unset, "", "off", ...        nothing changes: the bf16 layers, bit for bit

How a layer runs: its weight cast to fp16 once, at conversion, each output channel first times a power of two 2^k
(k >= 0, the smallest that keeps the row's values exact in fp16 to 2^-20 of the row's norm: SageAttention's fold
puts to_qkv's v rows 2^-8 down, where most of them would be fp16 subnormals and lose bits; every other row needs
k = 0); the activations cast to fp16 at the call; torch.mm with torch.backends.cuda.matmul.allow_fp16_accumulation
set for the call (cuBLAS's 16-bit compute type; F.linear's fused bias picks slower kernels at some shapes); then one
Triton pass: (y * 2^-k + bias) [-> GELU(tanh)] in fp32, rounded to the layer's dtype, bf16. The pass is checked once
per process against the same arithmetic in PyTorch; where Triton is missing or disagrees the PyTorch ops serve.
Off the card, under autograd or autocast a layer runs F.linear on its weight cast back (the bf16 weight to 2^-20 of
each row: fp16 keeps neither the tiniest values, below 2^-17 after the shift, nor the bf16 layer's bits).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import precision

KINDS = ("ss", "lr")                                   # the flows the switch converts
LAYERS = ("qkv", "out", "xq", "xout", "fc1", "proj")   # fc2 (inner dimension 8192) stays in the torso's dtype
MAX_SHIFT = 12                  # the largest 2^k a channel may take; a row that needs more keeps its layer in bf16
EXACT = 2.0 ** -20              # a row's relative L2 error from the fp16 cast that counts as exact
_ON = {"1", "on", "true", "yes"}
_OFF = {"", "0", "off", "false", "no"}
_ATTENTION = ("SparseMultiHeadAttention", "MultiHeadAttention")
_FFN = ("SparseFeedForwardNet", "FeedForwardNet")
_PROJECT = ("SparseProjectAttention", "ProjectAttention")

_warned: set = set()
_epilogue_state: Optional[bool] = None     # None: not tried; True: the Triton pass agreed with PyTorch; False: torch
_kernel = None


def requested() -> bool:
    """Whether PIXAL3D_FP16ACC asks for the switch (anything but on / off is off, with one warning)."""
    text = os.environ.get("PIXAL3D_FP16ACC", "").strip().lower()
    if text in _ON:
        return True
    if text not in _OFF and text not in _warned:
        _warned.add(text)
        print(f"[FP16ACC] PIXAL3D_FP16ACC={text!r} is neither on nor off: the switch stays off", flush=True)
    return False


def allowed() -> tuple[bool, str]:
    """(whether this process converts, why not): the switch, the ampere profile, a torch that has the switch."""
    if not requested():
        return False, "PIXAL3D_FP16ACC is off"
    profile = precision.profile()
    if profile.name != "ampere":
        return False, (f"the {profile.name} profile's card has no fast fp16 accumulation (GeForce Ampere and newer "
                       f"only)")
    if not hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
        return False, f"torch {torch.__version__} has no allow_fp16_accumulation"
    return True, ""


def describe() -> str:
    """One line for a start-up message: what the switch does in this process."""
    if not requested():
        return "fp16 accumulation off"
    ok, why = allowed()
    if not ok:
        return f"PIXAL3D_FP16ACC=on asked for, not used: {why}"
    return "PIXAL3D_FP16ACC=on: the ss and lr flows' linears (fc2 excepted) in fp16 with fp16 accumulation"


# =====================================================================================================================
# The weights
# =====================================================================================================================

def channel_exponents(w: torch.Tensor) -> torch.Tensor:
    """k per output channel of w [N, K] (int32 [N]): the smallest k in 0..MAX_SHIFT for which w * 2^k cast to fp16 is
    within EXACT of itself (relative L2 over the row), -1 for a row no k serves (its values span more than fp16 has
    room for, or reach past 2^14 when scaled)."""
    w = w.detach().float()
    norm = w.norm(dim=1).clamp_min(1e-30)
    out = torch.full((w.shape[0],), -1, dtype=torch.int32, device=w.device)
    for k in range(MAX_SHIFT + 1):
        scaled = w * 2.0 ** k
        err = (scaled.half().float() - scaled).norm(dim=1) / (norm * 2.0 ** k)
        fits = (err <= EXACT) & (scaled.abs().amax(dim=1) < 2.0 ** 14) & (out < 0)
        out[fits] = k
        if bool((out >= 0).all()):
            break
    return out


class Fp16AccLinear(nn.Module):
    """An nn.Linear / SparseLinear's work as fp16 x fp16 with fp16 accumulation (module docstring): `gelu` does the
    MLP's GELU(tanh) in the same pass. Takes a tensor [..., K] or a VarLenTensor / SparseTensor, answers in the same
    form and in the source layer's dtype."""

    def __init__(self, source: nn.Linear, gelu: bool = False):
        super().__init__()
        w = source.weight.detach()
        N, K = w.shape
        k = channel_exponents(w)
        if bool((k < 0).any()):
            raise ValueError(f"{int((k < 0).sum())} of the {N} output channels do not fit fp16")
        shifted = bool((k > 0).any())
        scaled = w.float() * torch.pow(2.0, k.float())[:, None]
        self.register_buffer("weight16", scaled.half().contiguous())                    # fp16 [N, K]
        self.register_buffer("unscale", torch.pow(2.0, -k.float()).contiguous() if shifted else None)   # fp32 [N]
        self.register_buffer("bias", source.bias.detach().float().contiguous() if source.bias is not None else None)
        self.in_features, self.out_features, self.dtype, self.gelu = K, N, w.dtype, gelu
        self.shift_max = int(k.max())
        self.shifted_rows = int((k > 0).sum())

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, "
                f"fp16 accumulation{', GELU(tanh)' if self.gelu else ''}"
                + (f", {self.shifted_rows} channels shifted (2^{self.shift_max} at most)" if self.shifted_rows else ""))

    def weight_as(self, dtype: torch.dtype) -> torch.Tensor:
        """The source layer's weight in `dtype` (the scaled fp16 copy times 2^-k)."""
        w = self.weight16.float()
        if self.unscale is not None:
            w = w * self.unscale[:, None]
        return w.to(dtype)

    def forward(self, x: Any) -> Any:
        if hasattr(x, "feats"):
            return x.replace(self._forward(x.feats))
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not x.is_cuda or x.shape[-1] != self.in_features or x.numel() == 0 or torch.is_grad_enabled()
                or torch.is_autocast_enabled()):
            y = F.linear(x, self.weight_as(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
            return F.gelu(y, approximate="tanh") if self.gelu else y
        flat = x.reshape(-1, self.in_features).to(torch.float16)
        matmul = torch.backends.cuda.matmul
        before = matmul.allow_fp16_accumulation
        matmul.allow_fp16_accumulation = True
        try:
            y = torch.mm(flat, self.weight16.t())
        finally:
            matmul.allow_fp16_accumulation = before
        del flat
        out = epilogue(y, self.unscale, self.bias, self.gelu, self.dtype)
        return out.view(*x.shape[:-1], self.out_features)


# =====================================================================================================================
# The pass after the product: y * 2^-k + bias [GELU] -> bf16
# =====================================================================================================================

def _torch_epilogue(y: torch.Tensor, unscale: Optional[torch.Tensor], bias: Optional[torch.Tensor], gelu: bool,
                    dtype: torch.dtype) -> torch.Tensor:
    out = y.float()
    if unscale is not None:
        out = out * unscale
    if bias is not None:
        out = out + bias
    if gelu:
        out = F.gelu(out, approximate="tanh")
    return out.to(dtype)


def _build():
    global _kernel
    if _kernel is not None:
        return _kernel
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def epilogue_kernel(Y, OUT, UNSCALE, BIAS, M, N, HAS_UNSCALE: tl.constexpr, HAS_BIAS: tl.constexpr,
                        GELU: tl.constexpr, BF16: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        rm = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        cols = rn < N
        mask = (rm[:, None] < M) & cols[None, :]
        offs = rm[:, None] * N + rn[None, :]
        y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
        if HAS_UNSCALE:
            y = y * tl.load(UNSCALE + rn, mask=cols, other=1.0)[None, :]
        if HAS_BIAS:
            y = y + tl.load(BIAS + rn, mask=cols, other=0.0)[None, :]
        if GELU:
            inner = 0.7978845608028654 * (y + 0.044715 * y * y * y)
            y = 0.5 * y * (1.0 + libdevice.tanh(inner))
        if BF16:
            tl.store(OUT + offs, y.to(tl.bfloat16), mask=mask)
        else:
            tl.store(OUT + offs, y.to(tl.float16), mask=mask)

    _kernel = epilogue_kernel
    return _kernel


def _triton_epilogue(y: torch.Tensor, unscale: Optional[torch.Tensor], bias: Optional[torch.Tensor], gelu: bool,
                     dtype: torch.dtype) -> torch.Tensor:
    M, N = y.shape
    out = torch.empty((M, N), dtype=dtype, device=y.device)
    BLOCK_M, BLOCK_N = 32, 256
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    _build()[grid](y, out, unscale if unscale is not None else y, bias if bias is not None else y, M, N,
                   HAS_UNSCALE=unscale is not None, HAS_BIAS=bias is not None, GELU=gelu,
                   BF16=dtype == torch.bfloat16, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4)
    return out


def _check_epilogue(device: torch.device) -> bool:
    """The Triton pass once on small tables against the PyTorch ops (one bf16 ulp of leeway for GELU's tanh); a
    Triton that cannot build or run it, or gives other values, leaves the PyTorch ops in its place."""
    global _epilogue_state
    try:
        g = torch.Generator(device=device).manual_seed(0)
        y = (torch.randn((40, 300), device=device, generator=g) * 4).half()
        unscale = torch.pow(2.0, -torch.randint(0, 9, (300,), device=device, generator=g).float())
        bias = torch.randn((300,), device=device, generator=g)
        for gelu in (False, True):
            for us in (None, unscale):
                got = _triton_epilogue(y, us, bias, gelu, torch.bfloat16).float()
                want = _torch_epilogue(y, us, bias, gelu, torch.bfloat16).float()
                if not bool(((got - want).abs() <= 2.0 ** -7 * want.abs() + 2.0 ** -12).all()):
                    raise RuntimeError(f"the pass gives other values (gelu {gelu}; largest difference "
                                       f"{float((got - want).abs().max()):.4g})")
        torch.cuda.synchronize(device)
    except Exception as exc:        # noqa: BLE001 - whatever Triton or the card raise: the PyTorch ops serve
        _epilogue_state = False
        print(f"[FP16ACC] the bias / GELU pass runs as PyTorch ops ({type(exc).__name__}: {str(exc)[:200]})",
              flush=True)
        return False
    _epilogue_state = True
    return True


def epilogue(y: torch.Tensor, unscale: Optional[torch.Tensor], bias: Optional[torch.Tensor], gelu: bool,
             dtype: torch.dtype) -> torch.Tensor:
    """y (fp16 [M, N], the product) * unscale [N] + bias [N] [-> GELU(tanh)], in fp32, rounded to `dtype`."""
    if _epilogue_state is None:
        _check_epilogue(y.device)
    if _epilogue_state and y.is_contiguous() and dtype in (torch.bfloat16, torch.float16):
        return _triton_epilogue(y, unscale, bias, gelu, dtype)
    return _torch_epilogue(y, unscale, bias, gelu, dtype)


# =====================================================================================================================
# The conversion
# =====================================================================================================================

def _targets(module: nn.Module) -> list[tuple[str, Any, Any, bool]]:
    """(layer type, container, attribute, GELU fused) of the linears directly inside `module` that the switch
    converts; the MLP's GELU module, when fused, is replaced too (by convert)."""
    name = type(module).__name__
    found: list[tuple[str, Any, Any, bool]] = []
    if name in _ATTENTION:
        cross = getattr(module, "_type", "self") == "cross"
        for attr, kind in (("to_qkv", "qkv"), ("to_q", "xq"), ("to_out", "xout" if cross else "out")):
            if isinstance(getattr(module, attr, None), nn.Linear):
                found.append((kind, module, attr, False))
    elif name in _PROJECT:
        if isinstance(getattr(module, "proj_linear", None), nn.Linear):
            found.append(("proj", module, "proj_linear", False))
    elif name in _FFN:
        mlp = getattr(module, "mlp", None)
        if isinstance(mlp, nn.Sequential) and len(mlp) == 3 and isinstance(mlp[0], nn.Linear):
            act = mlp[1]
            fuse = type(act).__name__ in ("GELU", "SparseGELU") and getattr(act, "approximate", "") == "tanh"
            found.append(("fc1", mlp, "0", fuse))
    return [hit for hit in found if hit[0] in LAYERS]


def convert(model: nn.Module, kind: str = "") -> nn.Module:
    """The model (a flow of `kind`) with its ss / lr token-sized linears as Fp16AccLinear where the switch is on and
    the profile allows it; the model unchanged otherwise (and off the card, and for the hr / tex flows). A layer
    whose weights do not fit fp16 (channel_exponents) keeps its bf16 layer, and the line says so."""
    if kind not in KINDS or not requested():
        return model
    ok, why = allowed()
    if not ok:
        if why not in _warned:
            _warned.add(why)
            print(f"[FP16ACC] PIXAL3D_FP16ACC=on: nothing converted ({why})", flush=True)
        return model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return model
    if device.type != "cuda":
        return model
    counts: dict[str, int] = {}
    kept: list[str] = []
    shift = 0
    with torch.no_grad():
        # the containers, not the layers, are collected first: a list holding the old layers would keep every bf16
        # weight on the card beside its fp16 copy until the end (1.8 GB more at the peak); this way each one is
        # freed as it is replaced
        found = [(name, *hit) for name, module in model.named_modules() for hit in _targets(module)]
        for mod_name, layer, container, attr, fuse in found:
            old = container[int(attr)] if attr.isdigit() else getattr(container, attr)
            if old.weight.dtype == torch.bfloat16:
                try:
                    new = Fp16AccLinear(old, gelu=fuse)
                except ValueError as exc:
                    kept.append(f"{mod_name}.{layer} ({exc})")
                    del old
                    continue
                if attr.isdigit():
                    container[int(attr)] = new
                    if fuse:
                        container[int(attr) + 1] = nn.Identity()
                else:
                    setattr(container, attr, new)
                shift = max(shift, new.shift_max)
                counts[layer] = counts.get(layer, 0) + 1
                del new
            del old
    if counts or kept:
        print(f"[FP16ACC] {type(model).__name__} ({kind}): "
              + ", ".join(f"{k} x{v}" for k, v in counts.items())
              + " in fp16 with fp16 accumulation (PIXAL3D_FP16ACC; fc2 stays bf16"
              + (f"; channels shifted by up to 2^{shift}" if shift else "") + ")"
              + (f"; kept in bf16: {', '.join(kept[:4])}" if kept else ""), flush=True)
    return model
