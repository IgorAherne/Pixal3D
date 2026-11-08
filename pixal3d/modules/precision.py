"""The card's precision profile: what the flows and decoders run in on this card, decided once per process.

Pixal3D was made for Ampere and newer: the flows in bf16, Triton kernels, natten, flash_attn and SageAttention (all
sm_80+). Older cards run the same networks with what they have (Spellcast3D round L, measured on an RTX 2060 Super
and a GTX 1080):

    ampere   sm_80 and newer: as before - the flows' torso in bf16, the Triton kernels, the int8 GEMMs, SageAttention,
             flex_gemm's convolutions, natten.
    turing   sm_70 - sm_79 (RTX 20xx, GTX 16xx, Volta): no bf16 arithmetic (torch emulates it at a seventh of the
             fp16 rate, and no bf16 attention kernel exists) - the torso in fp16 on the tensor cores (27 TFLOPS on an
             RTX 2060 Super); the int8 GEMMs through pixal3d_kernels' tensor-core kernel with group scales (Triton
             emits no int8 MMA for sm_75; torch._int_mm's per-row layers where that package is missing);
             attention through torch SDPA or the SageAttention wheel's Turing kernel where the install's calibration
             ran it, xformers on request (flash_attn is sm_80+); the sparse
             decoders' convolutions on spconv (flex_gemm's are Triton kernels, which Windows cannot be trusted to
             compile for Turing - Triton elementwise kernels stay optional, behind their self-checks); NAF's
             neighborhood attention in plain torch (natten's wheel is sm_80+).
    pascal   below sm_70 (GTX 10xx): fp16 arithmetic runs at a 64th of the fp32 rate on consumer Pascal and bf16 is
             emulated, so the torso is fp32 (its weights 0.9 GB: the token-sized linears are int8), the int8 GEMMs go
             through torch._int_mm (cuBLASLt's dp4a kernels: exact, 12-22 TOPS on a GTX 1080 against 6-7 TFLOPS of
             fp32), attention through torch SDPA in fp32 (its fp16 kernel takes 20 times longer there), the decoders
             on spconv (in fp16, as their checkpoints say: fp32 decoders ran the biggest prop out of an 8 GB card), NAF in
             plain torch, and no Triton at all (Triton does not support Pascal).

On both old profiles the flows' eager element-wise chains (norm modulations, gated residuals, the q / k RMS norm and
rotary embedding, the int8 quantisers) run as pixal3d_kernels' CUDA kernels with the chains' own bits
(modules/old_cards.py), and the decoders' spconv kernels are the ones the install's calibration chose
(sparse/conv/spconv_choices.py), so that every process makes the same meshes.

The profile follows the card (torch.cuda.get_device_capability) unless the environment says otherwise:
PIXAL3D_PRECISION = ampere | turing | pascal forces one (developers, tests); PIXAL3D_CARD_SM = 75 (and the like)
names the card's compute capability for a process that cannot see the card (Spellcast3D's installer makes the
flows' faster-loading copies with the card hidden: the copies hold the torso's dtype, so they must know it). Without
a card and without either variable the profile is ampere, as the code always was (the CPU tests). PIXAL3D_TRITON =
off keeps every Triton kernel off on the turing profile (each is optional there, behind a self-check that catches
what the compiler raises - not a compiler that takes the process down, which Windows builds for Turing could do:
nobody has run one; this is the way out), as the pascal profile always is.

Every network's constructor arguments from its checkpoint's json pass through `model_args` (the staged loader and
the copies): a flow's "dtype" becomes the torso's, a decoder's "use_fp16" is turned off where fp16 is slow. The
attention and convolution backends a configuration asks for are checked with `attention_backend` / `conv_backend`
(modules/attention/config.py, modules/sparse/config.py): one that has no kernels for the card is replaced, with
one line printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

NAMES = ("ampere", "turing", "pascal")
TURING_FROM = (7, 0)
AMPERE_FROM = (8, 0)
ASSUMED = {"ampere": (8, 6), "turing": (7, 5), "pascal": (6, 1)}     # the card a forced profile stands for
# The attention backends the modules know and the lowest architecture whose kernels each one has in Pixal3D's
# wheels: flash_attn 2.8.3 (cu128) sm_80; xformers 0.0.32 sm_75; torch's SDPA (its memory-efficient kernel, the
# same CUTLASS lineage as xformers') every card. flash_attn_3 / flash_attn_4 (Hopper and newer) are left to the
# configuration that asks for them. Below sm_80 the profile takes SDPA: xformers 0.0.32 refuses to import beside
# the flash_attn 2.8.3 wheel the venv holds ("Requires Flash-Attention version <= 2.8.2") unless told to ignore
# the version, and SDPA needs nothing; ATTN_BACKEND=xformers still asks for it.
ATTENTION_FROM = {"flash_attn": (8, 0), "xformers": (7, 5), "sdpa": (0, 0)}
ATTENTION_ORDER = ("flash_attn", "sdpa")


@dataclass(frozen=True)
class Profile:
    name: str
    capability: Optional[tuple[int, int]]
    torso: torch.dtype          # the flows' torso (their blocks): bf16, fp16 or fp32
    attention: str              # the attention backend the card runs where the configured one cannot
    conv: str                   # the sparse convolutions' backend: flex_gemm (Triton) or spconv
    triton: bool                # whether Triton kernels may be tried at all (each still has its self-check)
    decoders_fp16: bool         # the sparse decoders' torso in fp16 (as their checkpoints say), else fp32
    naf_torch: bool             # NAF's neighborhood attention in plain torch (natten has no kernels for the card)

    @property
    def torso_name(self) -> str:
        return {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[self.torso]


_profile: Optional[Profile] = None


def capability() -> Optional[tuple[int, int]]:
    """The card's compute capability: PIXAL3D_CARD_SM when set (two digits, "75" = (7, 5); "100" = (10, 0)), else the
    current CUDA device's, else None (no card visible)."""
    text = os.environ.get("PIXAL3D_CARD_SM", "").strip()
    if text:
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 2:
            return int(digits[:-1]), int(digits[-1])
    if torch.cuda.is_available():
        try:
            return tuple(torch.cuda.get_device_capability())
        except Exception:       # noqa: BLE001 - a card torch cannot ask behaves as none
            return None
    return None


def name_for(cc: Optional[tuple[int, int]]) -> str:
    if cc is None or cc >= AMPERE_FROM:
        return "ampere"
    return "turing" if cc >= TURING_FROM else "pascal"


def _make(name: str, cc: Optional[tuple[int, int]]) -> Profile:
    if name == "ampere":
        return Profile(name, cc, torch.bfloat16, "flash_attn", "flex_gemm", True, True, False)
    if name == "turing":
        return Profile(name, cc, torch.float16, _attention_for(cc or (7, 5)), _recorded_conv(), _triton_allowed(),
                       True, True)
    # the decoders in fp16 as on Turing (their checkpoints' dtype; Pascal computes fp16 in fp32 inside): in fp32 the
    # biggest prop's texture decode at 1024 ran a GTX 1080 out of memory (NVML 7.46 of 8 GB), in fp16 it fits (5.9 GB)
    return Profile(name, cc, torch.float32, "sdpa", "spconv", False, True, True)


def _triton_allowed() -> bool:
    """Whether the turing profile may try Triton kernels at all: PIXAL3D_TRITON=off says no."""
    return os.environ.get("PIXAL3D_TRITON", "").strip().lower() not in ("off", "0", "false", "no")


def _recorded_conv() -> str:
    """The sparse convolutions' backend below sm_80: flex_gemm where the install's calibration found its Triton
    kernels giving spconv's answers on this card (the record's "conv_backend" section, made by
    modules/sparse/conv/backend_check.py), spconv otherwise - and always for a process that cannot see the card."""
    try:
        from .. import calibration

        found = calibration.section("conv_backend")
    except Exception:       # noqa: BLE001 - no record, no card: spconv
        found = None
    if found and found.get("ok", True) and found.get("backend") == "flex_gemm":
        return "flex_gemm"
    return "spconv"


def _attention_for(cc: tuple[int, int]) -> str:
    return next(name for name in ATTENTION_ORDER if cc >= ATTENTION_FROM[name])


def profile() -> Profile:
    """This process's profile (made once; one line printed when it is not ampere's)."""
    global _profile
    if _profile is None:
        forced = os.environ.get("PIXAL3D_PRECISION", "").strip().lower()
        cc = capability()
        if forced and forced not in NAMES:
            print(f"[PRECISION] PIXAL3D_PRECISION={forced!r} is none of {', '.join(NAMES)}: the card decides", flush=True)
            forced = ""
        if forced and name_for(cc) != forced:
            cc = ASSUMED[forced]        # a forced profile on another card (or none): the backends follow the profile
        _profile = _make(forced or name_for(cc), cc)
        if _profile.name == "pascal":
            # torch's fused linear-with-bias (gemm_and_bias) goes through cuBLASLt, whose fp32 SGEMM reads out of
            # bounds on a GTX 1080 (compute-sanitizer: maxwell_sgemm_64x64_tn; cuBLAS 12.6 and 12.8): an illegal
            # memory access a few seconds into the ss flow. torch reads this at its first addmm; the hub sets it for
            # the card's processes too (gpu.pixal3d_card_env)
            os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")
        if _profile.name != "ampere" or forced:
            sm = f"sm_{cc[0]}{cc[1]}" if cc else "no card"
            print(f"[PRECISION] {_profile.name} profile ({sm}{', PIXAL3D_PRECISION' if forced else ''}): the flows' "
                  f"torso in {_profile.torso_name}, attention {_profile.attention}, convolutions {_profile.conv}, "
                  f"Triton {'allowed' if _profile.triton else 'off'}, the decoders in "
                  f"{'fp16' if _profile.decoders_fp16 else 'fp32'}", flush=True)
    return _profile


def reset() -> None:
    """Forget the profile (tests that change the environment)."""
    global _profile
    _profile = None


def torso() -> torch.dtype:
    return profile().torso


def model_args(args: dict) -> dict:
    """A checkpoint's constructor arguments for this card: a flow's "dtype" (bfloat16 in every shipped json) as the
    torso's, a decoder's "use_fp16" off where the card has no fast fp16. The dict itself when nothing changes."""
    p = profile()
    out = args
    if isinstance(args.get("dtype"), str) and args["dtype"] in ("bfloat16", "bf16") and p.torso != torch.bfloat16:
        out = {**out, "dtype": p.torso_name}
    if args.get("use_fp16") is True and not p.decoders_fp16:
        out = {**out, "use_fp16": False}
    return out


def attention_backend(wanted: str) -> str:
    """`wanted` where its kernels run on this card, else the profile's backend (one line printed)."""
    cc = profile().capability
    lowest = ATTENTION_FROM.get(wanted)
    if lowest is None or cc is None or cc >= lowest:
        return wanted
    chosen = profile().attention
    print(f"[PRECISION] the attention backend {wanted} has no kernels for sm_{cc[0]}{cc[1]}: {chosen} runs instead",
          flush=True)
    return chosen


def conv_backend(wanted: str, explicit: bool = False) -> str:
    """`wanted` where it runs on this card (flex_gemm's kernels are Triton's: sm_80+ here), else the profile's.
    `explicit`: the environment named it (SPARSE_CONV_BACKEND), which is kept whatever the card."""
    p = profile()
    if explicit or wanted != "flex_gemm" or p.conv == "flex_gemm" or p.capability is None:
        return wanted
    cc = p.capability
    print(f"[PRECISION] the sparse convolutions' backend flex_gemm (Triton) is not used on sm_{cc[0]}{cc[1]}: "
          f"{p.conv} runs instead (SPARSE_CONV_BACKEND=flex_gemm asks for it all the same)", flush=True)
    return p.conv
