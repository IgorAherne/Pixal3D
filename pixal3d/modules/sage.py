"""SageAttention (int8 Q.K^T, fp16 P.V) for the flows' full self-attention, behind the switch PIXAL3D_SAGE (on by default).

The flows' self-attention over all tokens of one prop (sparse: lr, hr, tex - one sequence of N tokens; dense: ss -
4096 tokens) is ~60 % of a detail pass at 1536, and flash_attn already runs it at the card's bf16 tensor peak.
SageAttention (thu-ml, Windows wheels by woct0rdho) quantises q and k to int8 (one scale per block of tokens and
head, k's per-channel mean over the tokens subtracted first) and multiplies P by v in fp16: about twice flash_attn's
speed on an RTX 3060 (an hr / tex pass at 1536: -26 %; ss -7 %, lr -3 %), at a cost in accuracy that is measured,
not free (code/devtools/pixal3d-lowmem/round-g/sage in Spellcast3D: on the flows' real q, k, v at 1536 its error
against an fp32 reference is 1.15-10.4 times flash_attn's own - which is the bf16 rounding of the output - and up to
~30 times in the heads whose k has a few dominant channels; a whole hr sampling ends ~1.2 times as far from the
flash_attn run as the fused-vs-eager change does).

The switch is the environment variable PIXAL3D_SAGE, read when the attention config modules are imported
(modules/sparse/config.py and modules/attention/config.py hold it as `SAGE`; their set_sage() changes it in a
running process):

    "on", "1", "true", "yes"     SageAttention where it applies (below), after its self-check
    "off", "0", "false", "no"    flash_attn everywhere, as before: the same numbers bit for bit
    unset, ""                    DEFAULT (below)
    "cuda", "triton", "wheel"    on, with the sm_8x kernel forced (developers): the CUDA kernel fed by the fused
                                 preprocessing, the Triton path, the wheel's own CUDA call
    "cuda-shipped", "cuda-tuned" on, with one of the wheel's two CUDA kernels forced (developers, below): the
                                 shipped one (= "wheel"), the round J tuned one (no timing at first use)
    anything else                off, with a warning

Where it applies: CUDA tensors in bf16, no autograd, no autocast, head size 128, the full self-attention of ONE
sequence (sparse: q, k, v sparse tensors of one prop with the same length, the call with three arguments; dense:
q, k, v [B, L, H, 128] with as many keys as queries). Cross-attention (5 context tokens), several sequences and
every other call keep their backend.

The kernel: on sm_8x (RTX 30xx / 40xx, A100) sageattn_qk_int8_pv_fp16_cuda with qk_quant_gran "per_warp" and
pv_accum_dtype "fp16+fp32" (woct0rdho's post3 carries SASS for sm_80 only, which then runs on every 8.x card;
Spellcast3D's own builds from post4 on carry SASS for sm_80, sm_86 and sm_89 - the driver runs the card's own -
and PTX for compute_89), except at the token counts TRITON_TOKENS lists for the installed wheel and the card's
architecture: there sage_triton.triton runs - bit for bit the answer
of the wheel's sageattn_qk_int8_pv_fp16_triton (proven with torch.equal on the flows' real tensors), with its
preprocessing in one Triton pass and its kernel compiled without Triton's length specialisation (which made it
7-10 % slower at lengths divisible by 16). Both kernels quantise q and k to int8 (the CUDA kernel q per 16 tokens,
the Triton one per 128) and sum P.V over 64 keys in fp16 before adding to fp32: they differ in the last bits, not
in quality (round I: the attention error 1.01-1.17 times the CUDA kernel's; round J: whole hr / tex samplings at
1536 end 1.03 / 0.84 times as far from the CUDA kernel's as the largest accepted yardstick does). The table is a
measurement (code/devtools/pixal3d-lowmem/round-j/sagefast/sj_rule.py in Spellcast3D prints it for a card and a
wheel): a fixed rule rather than a timing at first use, so that a prop gives the same numbers in every run (the
two CUDA kernels below give the same bits, so between those a timing at first use decides).
On sm_90 and newer, whose cards cannot run that SASS, sageattn_qk_int8_pv_fp16_triton. Both cast v to fp16: v
is therefore scaled by 2^-V_SHIFT before the call and the result by 2^V_SHIFT after it - exact in bf16, the same
values to the last digit (measured) - so that neither overflows below |v| ~ 262,000 (the flows' largest |v|
measured: 860; without the scaling the fp16 partial sums could overflow from |v| ~ 1,023, and the pure-fp16
accumulator, "fp16", gives Inf on the flows' real tensors); the engine folds that scaling into the weights (`fold`).
Sequences of any length are handled by the kernels (masked last block; lengths of 64 tokens or fewer are not
taken: the wheel's sm_80 kernel reads unset shared memory there, fixed upstream after the build we use).

Two CUDA kernels, the same bits (sm_8x; Spellcast3D's wheel from post5 on): the wheel holds the kernel as shipped
(the op qk_int8_sv_f16_accum_f16_attn_inst_buf, which sageattn_qk_int8_pv_fp16_cuda calls) and round J's tuned
build of the same kernel (TUNED_OP, the same arguments: two blocks an SM, the P.V step one output tile at a time;
-13 % a call on an RTX 3060, unmeasured on other cards - it may be slower on some, which must never happen to a
user). Both run the same arithmetic in the same order: the same answer bit for bit (torch.equal on the flows'
captured q / k / v and on random tables, round J sagecuda), so the choice between them can never change a result,
only the time. The choice is made per process on the card itself: at the self-check, after both have given the
same bits on its table, `_race` times the two ops on the same random int8 tables (RACE_TOKENS: 21,504 and 8,192
tokens, 12 heads x 128, interleaved ABBA, CUDA events, about half a second on an RTX 3060, the tables freed after)
and the tuned one serves only where it is at least RACE_MARGIN (5 %) faster at every size timed - otherwise, and
wherever the timing cannot run (an older wheel without the op, no memory for the tables, an error), the shipped
one, as before.
One `[SAGE]` line says which won and by how much; PIXAL3D_SAGE=cuda-shipped / cuda-tuned skips the timing.
An installed Spellcast3D does that timing once, at install, on a quiet card (`calibrate`, called by
`python -m pixal3d.calibrate`, with more calls than the first-use race: CALIBRATION_RACE) and keeps the answer in
the install's calibration record (pixal3d/calibration.py); a process then takes the recorded choice and times
nothing. The record serves only for the card and the wheel it was made with: another card or wheel races at first
use as an uncalibrated install does.

The int8 P.V twin of that kernel (wheel post6 on, Spellcast3D round K: INT8PV_OP; the same q / k quantisation,
Q.K^T and online softmax, then P as 8-bit integers per row and 64-key block relative to the block's maximum times
v as int8 per 64-key block and channel with its mean subtracted, on the int8 tensor cores: -18 to -20 % a call on
an RTX 3060, a different rounding) serves the detail flows' blocks that `mark_int8pv` marks (INT8PV_FLOWS,
INT8PV_SKIP: the hr flow's block 0 keeps the fp16 kernel, where its attention is nearly uniform and the integer
code's one step loses the small weights), judged end to end on the five test props (meshes 0.03-0.06 % of the
bounding-box diagonal from the fp16 kernel's, the int8 GEMMs' class). It is chosen by a timing of the whole call
against the fp16 one (the calibration record, else the first use: `_race_int8pv`, INT8PV_MARGIN), so that no card
gets slower; PIXAL3D_INT8PV=off keeps the fp16 kernel for every block.

Before the first SageAttention call of a process each kernel the card may use runs once on a small random table
against an fp32 reference (`_check`; the Triton kernels compile there); a kernel that fails or answers wrongly is
left out (the CUDA kernel then serves every length), and a card older than sm_80, a missing or broken wheel or no
kernel left turns SageAttention off with a `[SAGE]` warning, and the flows keep flash_attn.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from . import precision

DEFAULT = True      # what an unset PIXAL3D_SAGE means: on, after the self-check (round G: the five test props at 1536
#                     made with it are of the same quality as with flash_attn - Chamfer 0.17-0.24 % of the bounding-box
#                     diagonal, the class of the fused kernels' own change - for -20 % of a batch)
V_SHIFT = 8         # v * 2^-8 into the kernel, its answer * 2^8: exact in bf16
HEAD_DIM = 128      # the only head size checked
MIN_TOKENS = 65     # one block of 64 keys or fewer: see the docstring

_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}
_KERNELS = {"cuda", "triton", "wheel", "cuda-shipped", "cuda-tuned"}     # on, with the kernel forced (docstring)
_ALIASES = {"cuda-shipped": "wheel", "cuda-tuned": "tuned"}             # PIXAL3D_SAGE spelling -> candidate name
TURING_OP = "qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d"    # torch.ops.sageattention_qattn_sm75.<this>
#                     (Spellcast3D round L, the wheel from post7): the tuned sm80 kernel's twin for sm_75 (RTX 20xx) on
#                     Turing's mma shapes without cp.async, the same arguments and quantisers (the wheel's per_warp_int8,
#                     whose kernels that wheel carries for sm_75 too), fp16 in and out (the turing profile's torso)
PASCAL_OP = "sageattn_pascal"   # torch.ops.sageattention_qattn_sm61.<this> (Spellcast3D round L, the wheel from post7):
#                     int8 Q.K^T by dp4a with fp32 or int8 P.V for sm_61 (GTX 10xx), the per_warp scheme, fp32 in
#                     and out (the pascal profile's torso); its int8 P.V is the round K scheme with 32-key blocks
TUNED_OP = "qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d"    # torch.ops.sageattention_qattn_sm80.<this>: the tuned
#                     kernel (Spellcast3D's wheel post5 on; the shipped op's arguments, the same bits)
RACE_TOKENS = ((21504, 4), (8192, 6))   # the shipped and the tuned op timed at first use: (tokens, calls of each)
#                     - an hr / tex pass's size and a smaller one; 12 heads x 128, interleaved ABBA (an even count:
#                     drift weighs on both alike), the median; ~0.5 s on an RTX 3060
RACE_MARGIN = 0.05  # the tuned op serves only where it is at least this much faster at every size timed. Round J on
#                     an RTX 3060: the call-by-call race reads the tuned op ~1 point better than the flows' passes do
#                     (-12.8 % against -11.8 % a call at 21.5k tokens; blocks of one kernel: -10.5 %), and a median of
#                     a few calls varies by about +-3 points - 5 % keeps the shipped op on a card where the tuned one
#                     is not clearly faster, and loses nothing on the 3060 (the race there: -9 to -15 %)
CALIBRATION_RACE = ((21504, 12), (8192, 16))  # the same race at install (calibrate): more calls, the card quiet
# sm_8x, the automatic choice (PIXAL3D_SAGE on or unset), measured per card architecture (round J, sagefast):
FUSED_PREP = False  # the CUDA kernel fed by sage_triton.ship_prep (one Triton pass) instead of the wheel's own
#                     preprocessing; taken only where the self-check finds the wheel's bits on its table
TRITON_TOKENS: dict = {     # (SageAttention version, major, minor) -> ((lo, hi), ...): token counts lo <= N < hi
    #                          where sage_triton.triton (the wheel's Triton kernel, unspecialised, its preprocessing
    #                          fused) runs instead of the CUDA kernel. Measured per wheel and card architecture
    #                          (round J, sagefast: sj_micro.py / sj_rule.py); a wheel or architecture not listed
    #                          keeps the CUDA kernel at every N until it is measured.
    # post3 (woct0rdho's build) on sm_86 (RTX 3060): the Triton path 7-9 % faster a call from 13,307 to 49,152
    # tokens (-8 % at 21,547), an hr / tex pass at 1536 -141 / -170 ms, whole samplings within the yardsticks'
    # class; from 8,192 tokens only: the detail flows (ss and lr, 2-4k tokens, keep the CUDA kernel - their whole
    # samplings were not re-checked with it). Round J's own build post4 (a faster CUDA kernel, the same bits):
    # the CUDA kernel 4-6 % faster than the Triton path at every N from 1,216 to 52,965 - no entry. post5 (post4's
    # tuned kernel beside the shipped one, chosen by `_race`): no entry either.
    ("2.2.0+cu128torch2.8.0.post3", 8, 6): ((8192, 1 << 30),),
}


def from_env() -> bool:
    """PIXAL3D_SAGE as a bool (the config modules call this once, when they are imported)."""
    text = os.environ.get("PIXAL3D_SAGE", "").strip().lower()
    if text == "":
        return DEFAULT
    if text in _ON or text in _KERNELS:
        return True
    if text not in _OFF and text not in _warned:
        _warned.add(text)
        print(f"[SAGE] PIXAL3D_SAGE={text!r} is none of on, off, cuda, triton, wheel, cuda-shipped, cuda-tuned: "
              f"SageAttention stays off", flush=True)
    return False


def preference() -> str:
    """The kernel PIXAL3D_SAGE asks for: "cuda", "triton", "wheel" (also for cuda-shipped), "tuned" (cuda-tuned),
    or "auto" (on / unset)."""
    text = os.environ.get("PIXAL3D_SAGE", "").strip().lower()
    return _ALIASES.get(text, text) if text in _KERNELS else "auto"


_warned: set = set()


_state: Optional[bool] = None       # None: not tried yet; True: passed its self-check on this card; False: off
_kernel: Optional[Callable] = None
KERNEL = ""                         # the kernel in use, for reports
_kernel_int8pv: Optional[Callable] = None   # the int8 P.V kernel where the self-check accepted it (INT8PV below)
KERNEL_INT8PV = ""
# The int8 P.V kernel (int8 Q.K^T as before, P as 8-bit integers per row and 64-key block, v as int8 per 64-key
# block and channel with its mean subtracted; round K) serves the detail flows' blocks only, and not the blocks
# whose attention it measurably hurts: block 0 of the hr flow attends nearly uniformly to value tokens with
# outliers, where an integer code with one step loses the small weights (its error there is ~10x the fp16
# kernel's against ~1.1x elsewhere, measured on captured passes). ss and lr keep the fp16 P.V kernel: their
# attention is a small share of a pass and their samples flip the most.
INT8PV_FLOWS = ("hr", "tex")
INT8PV_SKIP = {"hr": (0,), "tex": ()}
INT8PV_OP = "qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d"   # torch.ops.sageattention_qattn_sm80.<this>, wheel post6 on
INT8PV_MARGIN = 0.05        # the int8 P.V kernel serves only where it is at least this much faster than the fp16 P.V
#                             kernel at every size timed (the user's rule: no card may get slower; the two round
#                             differently, so a card that gains nothing keeps the fp16 kernel's numbers)


def usable(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Whether this self-attention may run SageAttention (the caller has checked the switch and that it is the full
    self-attention of one sequence). The first time, the kernel is tried on the card (`_check`)."""
    if (not q.is_cuda or q.dtype != precision.torso() or k.dtype != q.dtype or v.dtype != q.dtype
            or torch.is_grad_enabled() or torch.is_autocast_enabled()):
        return False
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        return False
    if q.shape[-3] < MIN_TOKENS or k.shape[-3] < MIN_TOKENS:
        return False
    if _state is None:
        _check(q.device)
    return bool(_state)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaled: bool = False,
              int8pv: bool = False) -> torch.Tensor:
    """Softmax(q k^T / sqrt(D)) v for q, k, v [B, N, H, D] (last dimension contiguous), not causal: [B, N, H, D] in
    q's dtype. Call only where usable() said True. `scaled`: v already carries the 2^-V_SHIFT (the weights were
    folded, `fold`): the answer keeps it, and the folded to_out undoes it. `int8pv`: the block was marked for the
    int8 P.V kernel (mark_int8pv), which serves where the self-check accepted it; the fp16 P.V kernel otherwise."""
    kernel = _kernel_int8pv if int8pv and _kernel_int8pv is not None else _kernel
    if scaled:
        return kernel(q, k, v)
    out = kernel(q, k, v * 2.0 ** -V_SHIFT)
    return out.mul_(2.0 ** V_SHIFT)


def mark_int8pv(model: torch.nn.Module, kind: str) -> int:
    """Mark the full self-attention modules of a flow network of `kind` (ss, lr, hr, tex) that may run the int8 P.V
    kernel: every block's of the flows INT8PV_FLOWS names, except the blocks INT8PV_SKIP lists for that kind.
    Done once per loaded network by the engine; harmless without the kernel (attention() then runs the fp16 P.V
    kernel for every block). Returns how many modules were marked."""
    count = 0
    skip = set(INT8PV_SKIP.get(kind, ()))
    allowed = kind in INT8PV_FLOWS
    for index, block in enumerate(getattr(model, "blocks", ())):
        for module in block.modules():
            if (type(module).__name__ not in ("SparseMultiHeadAttention", "MultiHeadAttention")
                    or getattr(module, "_type", None) != "self" or getattr(module, "attn_mode", None) != "full"):
                continue
            module.sage_int8pv = allowed and index not in skip
            count += int(module.sage_int8pv)
    return count


def fold(model: torch.nn.Module) -> int:
    """The 2^-V_SHIFT of v and the 2^V_SHIFT of the answer folded into the weights of every full self-attention of
    `model`: to_qkv's v rows (weight and bias) times 2^-V_SHIFT, to_out's weight times 2^V_SHIFT. Powers of two
    scale a floating-point value exactly (no value of the flows comes near the exponent's ends), and every product
    and sum of the attention and of the two GEMMs scales with them, so every path - SageAttention without its two
    scaling passes (0.8 ms a block at 21.5k tokens), flash_attn, SDPA, the int8 layers (their row and channel
    scales carry the powers) - gives the same bits as before. Done once per loaded network (the module's
    `sage_folded` says so), by the engine when the switch is on. Returns how many modules were folded."""
    down, up = 2.0 ** -V_SHIFT, 2.0 ** V_SHIFT
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if (type(module).__name__ not in ("SparseMultiHeadAttention", "MultiHeadAttention")
                    or getattr(module, "_type", None) != "self" or getattr(module, "attn_mode", None) != "full"
                    or getattr(module, "sage_folded", False)):
                continue
            C = module.channels
            _scale_rows(module.to_qkv, slice(2 * C, 3 * C), down)
            _scale_rows(module.to_out, slice(0, C), up, bias=False)
            module.sage_folded = True
            count += 1
    return count


def _scale_rows(linear: torch.nn.Module, rows: slice, factor: float, bias: bool = True) -> None:
    """The output rows `rows` of a linear layer (nn.Linear, or an int8 layer with `weight_q` and `scale`) times
    `factor`: the weight rows, or the int8 layer's channel scales, and the bias rows when `bias`."""
    if hasattr(linear, "weight_q"):
        linear.scale[rows].mul_(factor)
    else:
        linear.weight[rows].mul_(factor)
    if bias and getattr(linear, "bias", None) is not None:
        linear.bias[rows].mul_(factor)


def _candidates(device: torch.device, want: str) -> dict[str, tuple[Callable, str]]:
    """The kernels to try on this card for the preference `want` (preference()): {name: (kernel(q, k, v), label)},
    each [B, N, H, 128] NHD bf16 -> the same. "wheel" always; on sm_8x "tuned" where the wheel has TUNED_OP and
    the choice is automatic or asks for it, "cuda" where FUSED_PREP or asked for, and "triton" where TRITON_TOKENS
    lists this architecture or it is asked for (a kernel not needed is not compiled)."""
    import sageattention as sa

    major, minor = precision.profile().capability or torch.cuda.get_device_capability(device)
    if major == 6:
        op = _pascal_op()
        if op is None:
            raise RuntimeError(f"the card is sm_{major}{minor}; the installed wheel has no Pascal kernel ({PASCAL_OP})")
        return {"pascal": (_pascal_call(op, False), f"{PASCAL_OP} (int8 Q.K^T by dp4a, fp32 P.V)")}
    if major == 7:
        op = _turing_op()
        if op is None:
            raise RuntimeError(f"the card is sm_{major}{minor}; the installed wheel has no Turing kernel (_qattn_sm75)")
        return {"turing": (_wheel_call(op), f"{TURING_OP} for sm_75 (per_warp, fp16+fp32)")}
    if major < 8:
        raise RuntimeError(f"the card is sm_{major}{minor}; SageAttention's kernels serve sm_61, sm_75 and sm_80 and newer")

    def wheel_cuda(q, k, v):
        return sa.sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="NHD", is_causal=False,
                                                qk_quant_gran="per_warp", pv_accum_dtype="fp16+fp32")

    def wheel_triton(q, k, v):
        return sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout="NHD", is_causal=False,
                                                  quantization_backend="triton")

    if major != 8:
        return {"wheel": (wheel_triton, "sageattn_qk_int8_pv_fp16_triton")}
    found = {"wheel": (wheel_cuda, "sageattn_qk_int8_pv_fp16_cuda (per_warp, fp16+fp32)")}
    op = _tuned_op() if want in ("auto", "tuned") else None
    if op is not None:
        found["tuned"] = (_wheel_call(op), f"sageattn_qk_int8_pv_fp16_cuda (per_warp, fp16+fp32), the tuned kernel "
                                           f"({TUNED_OP})")
    if want in ("wheel", "tuned"):
        return found
    from . import sage_triton

    if want == "cuda" or (want == "auto" and FUSED_PREP):
        found["cuda"] = (sage_triton.ship, "the wheel's sm80 kernel (per_warp, fp16+fp32), preprocessing fused")
    if want == "triton" or (want == "auto" and _triton_ranges(device)):
        found["triton"] = (sage_triton.triton, "the wheel's Triton kernel, unspecialised, preprocessing fused")
    return found


def _tuned_op() -> Optional[Callable]:
    """The wheel's tuned CUDA op (TUNED_OP), or None for a wheel without it (woct0rdho's post3, Spellcast3D's post4)."""
    import sageattention.core  # noqa: F401  (loads _qattn_sm80, which registers torch.ops.sageattention_qattn_sm80)

    try:
        return getattr(torch.ops.sageattention_qattn_sm80, TUNED_OP)
    except (AttributeError, RuntimeError):
        return None


def _turing_op() -> Optional[Callable]:
    """The wheel's Turing op (TURING_OP, the extension _qattn_sm75), or None for a wheel without it."""
    try:
        import sageattention._qattn_sm75  # noqa: F401  (registers torch.ops.sageattention_qattn_sm75)
    except Exception:       # noqa: BLE001 - a wheel without the extension: the op may still be registered by a build
        pass
    try:
        return getattr(torch.ops.sageattention_qattn_sm75, TURING_OP)
    except (AttributeError, RuntimeError):
        return None


def _pascal_op() -> Optional[Callable]:
    """The wheel's Pascal op (PASCAL_OP, the extension _qattn_sm61), or None for a wheel without it."""
    try:
        import sageattention._qattn_sm61  # noqa: F401  (registers torch.ops.sageattention_qattn_sm61)
    except Exception:       # noqa: BLE001 - a wheel without the extension: the op may still be registered by a build
        pass
    try:
        return getattr(torch.ops.sageattention_qattn_sm61, PASCAL_OP)
    except (AttributeError, RuntimeError):
        return None


def _pascal_call(op: Callable, int8pv: bool) -> Callable:
    """The Pascal op on q, k, v [B, N, H, 128] NHD (fp32, fp16 or bf16; v may be a strided view): the kernel quantises
    q and k itself (and v, for the int8 P.V); the answer in q's dtype. Its accumulators are fp32: the 2^-V_SHIFT
    fold of v is exact and needs no care here."""
    def run(q, k, v):
        torch.cuda.set_device(v.device)
        o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
        op(q, k, v, o, 0, q.size(-1) ** -0.5, 1 if int8pv else 0)
        return o

    return run


def _wheel_call(op: Callable) -> Callable:
    """sageattn_qk_int8_pv_fp16_cuda(q, k, v, "NHD", is_causal=False, qk_quant_gran="per_warp", pv_accum_dtype=
    "fp16+fp32") step for step as the wheel's core.py takes it at head size 128 (no padding, k's mean subtracted,
    no LSE) - the same functions on the same tensors in the same order - with `op` as its CUDA kernel."""
    from sageattention.quant import per_warp_int8

    def run(q, k, v):
        torch.cuda.set_device(v.device)
        km = k.mean(dim=1, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=16, BLKK=64)
        o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
        op(q_int8, k_int8, v.to(torch.float16), o, q_scale, k_scale, 0, 0, 2, q.size(-1) ** -0.5, 0)
        return o

    return run


def _race(device: torch.device, plan: tuple = RACE_TOKENS) -> tuple[bool, str, list]:
    """The shipped and the tuned CUDA op timed on this card (see the docstring): the same random int8 q (also as k)
    and fp16 v at each of RACE_TOKENS, 12 heads x 128, the calls interleaved (shipped, tuned / tuned, shipped), one
    CUDA event pair each, the medians compared (round J: the tuned / shipped ratio on these tables is the one on
    the bench's quantised tables and on the flows' captured q / k / v to ~1 point, race_check.py). Both ops have run on the self-check's table just before (loaded,
    the same bits). `plan`: (tokens, calls of each) per size. Returns (the tuned op serves, the [SAGE] line, the
    rows (tokens, shipped ms, tuned ms)). Raises on a CUDA error; a size whose tables (5 bytes a token and channel)
    do not fit in what the card and PyTorch's cache have free is left out."""
    ops = torch.ops.sageattention_qattn_sm80
    fns = {"shipped": ops.qk_int8_sv_f16_accum_f16_attn_inst_buf, "tuned": getattr(ops, TUNED_OP)}
    heads, began = 12, time.perf_counter()
    rows, skipped = [], []
    torch.cuda.synchronize(device)
    for n, calls in plan:
        need = n * heads * HEAD_DIM * 5                  # int8 q (= k), fp16 v, bf16 output
        free = torch.cuda.mem_get_info(device)[0]
        spare = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if need * 1.25 > free + spare:
            skipped.append(n)
            continue
        g = torch.Generator(device=device).manual_seed(n)
        q8 = torch.randint(-127, 128, (1, n, heads, HEAD_DIM), dtype=torch.int8, device=device, generator=g)
        v16 = torch.randn((1, n, heads, HEAD_DIM), dtype=torch.float16, device=device, generator=g)
        o = torch.empty((1, n, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)
        q_scale = torch.full((1, heads, (n + 127) // 128 * 8), 0.02, dtype=torch.float32, device=device)
        k_scale = torch.full((1, heads, (n + 63) // 64), 0.02, dtype=torch.float32, device=device)
        events: dict = {name: [] for name in fns}
        stream = torch.cuda.default_stream(device)       # the ops launch on the legacy default stream
        for i in range(calls):
            for name in (("shipped", "tuned") if i % 2 == 0 else ("tuned", "shipped")):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(stream)
                fns[name](q8, q8, v16, o, q_scale, k_scale, 0, 0, 2, HEAD_DIM ** -0.5, 0)
                end.record(stream)
                events[name].append((start, end))
        torch.cuda.synchronize(device)
        ms = {name: statistics.median(a.elapsed_time(b) for a, b in pairs) for name, pairs in events.items()}
        rows.append((n, ms["shipped"], ms["tuned"]))
        del q8, v16, o, q_scale, k_scale, events
    seconds = time.perf_counter() - began
    if not rows:
        return False, (f"[SAGE] the shipped CUDA kernel serves: the tuned one was not timed (no room for the "
                       f"tables of {', '.join(f'{n:,}' for n in skipped)} tokens)"), rows
    wins = all(t <= s * (1 - RACE_MARGIN) for _, s, t in rows)
    text = ", ".join(f"{n:,} tokens {s:.2f} -> {t:.2f} ms ({100 * (t / s - 1):+.1f} %)" for n, s, t in rows)
    if skipped:
        text += f"; {', '.join(f'{n:,}' for n in skipped)} not timed (no room)"
    if wins:
        return True, (f"[SAGE] the tuned CUDA kernel (round J) serves, timed against the shipped one: {text} "
                      f"({seconds:.2f} s)"), rows
    return False, (f"[SAGE] the shipped CUDA kernel serves: the tuned one (round J) is not {100 * RACE_MARGIN:.0f} % "
                   f"faster at every size timed: {text} ({seconds:.2f} s)"), rows


def _wheel_version() -> Optional[str]:
    from importlib import metadata

    try:
        return metadata.version("sageattention")
    except Exception:        # noqa: BLE001 - no distribution metadata
        return None


def _stored(device: torch.device) -> Optional[dict]:
    """The calibration record's choice between the two CUDA ops (`calibrate`), when it was made on this card with
    the installed wheel; None otherwise (the first-use race then decides)."""
    from .. import calibration

    entry = calibration.section("sage", device)
    if not entry or "tuned" not in entry or entry.get("wheel") != _wheel_version():
        return None
    return entry


def calibrate(device: torch.device) -> dict:
    """The install's calibration of the choice between the wheel's two CUDA kernels (`python -m pixal3d.calibrate`
    calls it once, the card quiet): both ops on the self-check's table (they must give the same bits, else the
    shipped one is recorded), then the race with CALIBRATION_RACE. Returns the record's "sage" section: {"wheel",
    "capability", "tuned": bool, "summary", "timings": [{"tokens", "shipped_ms", "tuned_ms"}], "why"}."""
    major, minor = precision.profile().capability or torch.cuda.get_device_capability(device)
    found = {"wheel": _wheel_version(), "capability": [major, minor], "tuned": False, "timings": []}
    if major != 8:
        found.update(why=f"sm_{major}{minor}: the CUDA kernels are used on sm_8x only", summary="not sm_8x")
        return found
    op = _tuned_op()
    if op is None:
        found.update(why="the installed wheel has no tuned kernel", summary="no tuned kernel in the wheel")
        return found
    q, k, vs, _want = _check_table(device)
    shipped = _candidates(device, "wheel")["wheel"][0](q, k, vs)
    tuned = _wheel_call(op)(q, k, vs)
    torch.cuda.synchronize(device)
    if not torch.equal(shipped, tuned):
        found.update(why="the tuned kernel did not give the shipped kernel's bits on this card",
                     summary="the tuned kernel's bits differ: the shipped one serves")
        return found
    del q, k, vs, _want, shipped, tuned
    wins, line, rows = _race(device, CALIBRATION_RACE)
    found["tuned"] = wins
    found["timings"] = [{"tokens": n, "shipped_ms": round(a, 3), "tuned_ms": round(b, 3)} for n, a, b in rows]
    found["summary"] = ", ".join(f"{n:,} tokens {a:.2f} -> {b:.2f} ms ({100 * (b / a - 1):+.1f} %)"
                                 for n, a, b in rows) or "not timed"
    found["why"] = line
    # the int8 P.V kernel against the fp16 one chosen above (the whole calls), the same way
    op8 = _int8pv_op() if int8pv_switch() != "off" else None
    if op8 is not None:
        fp16 = _wheel_call(op) if wins else _candidates(device, "wheel")["wheel"][0]
        wins8, line8, rows8 = _race_int8pv(device, fp16, _int8pv_call(op8), CALIBRATION_RACE)
        found["int8pv"] = wins8
        found["int8pv_timings"] = [{"tokens": n, "fp16_ms": round(a, 3), "int8_ms": round(b, 3)} for n, a, b in rows8]
        found["int8pv_summary"] = ", ".join(f"{n:,} tokens {a:.2f} -> {b:.2f} ms ({100 * (b / a - 1):+.1f} %)"
                                            for n, a, b in rows8) or "not timed"
        found["int8pv_why"] = line8
    return found


def _triton_ranges(device: torch.device) -> tuple:
    """TRITON_TOKENS' ranges for the installed wheel and this card's architecture (none: the CUDA kernel serves)."""
    version = _wheel_version()
    if version is None:      # no distribution metadata: nothing was measured for it
        return ()
    return tuple(TRITON_TOKENS.get((version, *torch.cuda.get_device_capability(device)), ()))


def _by_length(cuda: Callable, triton: Callable, ranges: tuple) -> Callable:
    def run(q, k, v):
        n = q.shape[1]
        for lo, hi in ranges:
            if lo <= n < hi:
                return triton(q, k, v)
        return cuda(q, k, v)
    return run


def _choose(device: torch.device, cands: dict, tried: dict[str, Optional[torch.Tensor]]) -> tuple[Callable, str]:
    """The kernel for this card from the candidates that passed the self-check (`tried`: name -> answer on the
    check table, None for one that failed). sm_8x: the CUDA kernel - the tuned op with the wheel's preprocessing
    ("tuned") where it gives the shipped op's bits on the table and wins `_race`, else the fused preprocessing
    ("cuda") where it gives the wheel's own bits on the table, else the wheel's own call - and the Triton path at
    the token counts TRITON_TOKENS lists for this architecture; PIXAL3D_SAGE=cuda / triton / wheel / cuda-shipped /
    cuda-tuned forces one. sm_90 and newer: the wheel's Triton call."""
    ok = {name: cands[name] for name, got in tried.items() if got is not None}
    want = preference()
    if "cuda" in ok and "wheel" in ok and not torch.equal(tried["cuda"], tried["wheel"]):
        print("[SAGE] the fused preprocessing does not give the wheel's bits on this card: the wheel's own call "
              "serves", flush=True)
        del ok["cuda"]
    if "tuned" in ok and ("wheel" not in ok or not torch.equal(tried["tuned"], tried["wheel"])):
        print("[SAGE] the tuned CUDA kernel does not give the shipped kernel's bits on this card: the shipped one "
              "serves", flush=True)
        del ok["tuned"]
    if want in ok:
        return ok[want]
    if want != "auto":
        print(f"[SAGE] PIXAL3D_SAGE={os.environ.get('PIXAL3D_SAGE', want).strip().lower()}: that kernel is not "
              f"available on this card or failed its self-check; the automatic choice serves", flush=True)
    cuda = ok.get("cuda") or ok.get("wheel") or ok.get("turing") or ok.get("pascal")
    if "tuned" in ok:
        stored = _stored(device)
        if stored is not None:
            wins = bool(stored.get("tuned"))
            line = (f"[SAGE] the {'tuned' if wins else 'shipped'} CUDA kernel serves, as timed on this card when "
                    f"Spellcast3D was installed ({stored.get('summary', 'calibration record')})")
        else:
            try:
                wins, line, _rows = _race(device)
            except Exception as exc:    # noqa: BLE001 - a timing that fails changes nothing: the shipped op serves
                wins, line = False, (f"[SAGE] the shipped CUDA kernel serves: timing the tuned one failed "
                                     f"({type(exc).__name__}: {str(exc)[:200]})")
        print(line, flush=True)
        if wins:
            cuda = ok["tuned"]
    triton = ok.get("triton")
    if cuda is None:
        raise RuntimeError("the CUDA kernel did not pass its self-check")
    ranges = _triton_ranges(device)
    if triton is None or not ranges:
        return cuda
    text = ", ".join(f"{lo}+" if hi >= 1 << 30 else f"{lo}-{hi - 1}" for lo, hi in ranges)
    return _by_length(cuda[0], triton[0], ranges), f"{cuda[1]}; at {text} tokens {triton[1]}"


def int8pv_switch() -> str:
    """PIXAL3D_INT8PV: "on" / unset (the int8 P.V kernel where the wheel has it, after its self-check and the
    race), "off" (never), "force" (developers: without the race)."""
    text = os.environ.get("PIXAL3D_INT8PV", "").strip().lower()
    if text in _OFF:
        return "off"
    return "force" if text == "force" else "on"


def _int8pv_op() -> Optional[Callable]:
    """The wheel's int8 P.V op (INT8PV_OP), or None for a wheel without it (post5 and older)."""
    import sageattention.core  # noqa: F401  (loads _qattn_sm80, which registers torch.ops.sageattention_qattn_sm80)

    try:
        return getattr(torch.ops.sageattention_qattn_sm80, INT8PV_OP)
    except (AttributeError, RuntimeError):
        return None


def _int8pv_call(op: Callable) -> Callable:
    """The int8 P.V attention as the fork calls it ([B, N, H, 128] NHD bf16 -> the same): the wheel's per_warp_int8
    for q and k (as the fp16 kernel's call, _wheel_call), per_block_int8_v for v (its mean per channel subtracted,
    int8 per 64-token block and channel, transposed), then `op`."""
    from sageattention.quant import per_block_int8_v, per_warp_int8

    def run(q, k, v):
        torch.cuda.set_device(v.device)
        km = k.mean(dim=1, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=16, BLKK=64)
        v_int8, v_scale, v_mean = per_block_int8_v(v, tensor_layout="NHD", smooth_v=True)
        o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
        op(q_int8, k_int8, v_int8, o, q_scale, k_scale, v_scale, v_mean, 0, 0, 2, q.size(-1) ** -0.5, 0)
        return o

    return run


def _race_int8pv(device: torch.device, fp16: Callable, int8pv: Callable,
                 plan: tuple = RACE_TOKENS) -> tuple[bool, str, list]:
    """The fp16 P.V call in use and the int8 P.V call timed on this card, whole calls (their quantisers included)
    on the same random bf16 q, k, v at each of `plan`'s sizes, 12 heads x 128, interleaved ABBA, CUDA events, the
    medians compared: the int8 one serves only where it is at least INT8PV_MARGIN faster at every size. Returns
    (the int8 call serves, the [SAGE] line, the rows (tokens, fp16 ms, int8 ms)). A size whose tables do not fit
    is left out."""
    fns = {"fp16": fp16, "int8": int8pv}
    heads, began = 12, time.perf_counter()
    rows, skipped = [], []
    torch.cuda.synchronize(device)
    for n, calls in plan:
        need = n * heads * HEAD_DIM * 2 * 6                # q, k, v, their int8 / fp16 copies, the output
        free = torch.cuda.mem_get_info(device)[0]
        spare = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if need * 1.25 > free + spare:
            skipped.append(n)
            continue
        g = torch.Generator(device=device).manual_seed(n)
        dt = precision.torso()
        q = torch.randn((1, n, heads, HEAD_DIM), dtype=dt, device=device, generator=g)
        k = torch.randn((1, n, heads, HEAD_DIM), dtype=dt, device=device, generator=g)
        v = torch.randn((1, n, heads, HEAD_DIM), dtype=dt, device=device, generator=g)
        events: dict = {name: [] for name in fns}
        for i in range(calls):
            for name in (("fp16", "int8") if i % 2 == 0 else ("int8", "fp16")):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                fns[name](q, k, v)
                end.record()
                events[name].append((start, end))
        torch.cuda.synchronize(device)
        ms = {name: statistics.median(a.elapsed_time(b) for a, b in pairs) for name, pairs in events.items()}
        rows.append((n, ms["fp16"], ms["int8"]))
        del q, k, v, events
    seconds = time.perf_counter() - began
    if not rows:
        return False, (f"[SAGE] the fp16 P.V kernel serves every block: the int8 one was not timed (no room for "
                       f"the tables of {', '.join(f'{n:,}' for n in skipped)} tokens)"), rows
    wins = all(b <= a * (1 - INT8PV_MARGIN) for _, a, b in rows)
    text = ", ".join(f"{n:,} tokens {a:.2f} -> {b:.2f} ms ({100 * (b / a - 1):+.1f} %)" for n, a, b in rows)
    if wins:
        return True, (f"[SAGE] the int8 P.V kernel serves the detail flows' blocks (hr's block 0 excepted), timed "
                      f"against the fp16 P.V one: {text} ({seconds:.2f} s)"), rows
    return False, (f"[SAGE] the fp16 P.V kernel serves every block: the int8 one is not {100 * INT8PV_MARGIN:.0f} % "
                   f"faster at every size timed: {text} ({seconds:.2f} s)"), rows


def _setup_int8pv(device: torch.device, q: torch.Tensor, k: torch.Tensor, vs: torch.Tensor,
                  want_out: torch.Tensor) -> None:
    """The int8 P.V kernel beside the chosen fp16 one (`_kernel_int8pv`): where the wheel has the op, the switch
    allows it and the card is sm_8x, it runs once on the self-check's table against fp32 attention (the 3 % bar
    of _check; it measures ~1.5 % there, the fp16 kernel ~1.2 %), then serves where the calibration record says
    so, or where it wins the race against the fp16 call (PIXAL3D_INT8PV=force: without the race). Off, or any
    failure, keeps the fp16 kernel for every block (the numbers as before)."""
    global _kernel_int8pv, KERNEL_INT8PV
    _kernel_int8pv, KERNEL_INT8PV = None, ""
    switch = int8pv_switch()
    if switch == "off" or _kernel is None:
        return
    try:
        major, _minor = precision.profile().capability or torch.cuda.get_device_capability(device)
        if major == 8:
            op = _int8pv_op()
            call = _int8pv_call(op) if op is not None else None
            what = f"{INT8PV_OP} (int8 P.V: u8 P per 64-key block, v per block and channel)"
        elif major == 6:
            op = _pascal_op()
            call = _pascal_call(op, True) if op is not None else None
            what = f"{PASCAL_OP} (int8 P.V by dp4a: u8 P per 32-key block, v per block and channel)"
        else:
            call = None
        if call is None:
            return
        got = call(q, k, vs)
        torch.cuda.synchronize(device)
        rel = float((got.float() * 2.0 ** V_SHIFT - want_out).norm() / want_out.norm())
        if got.shape != want_out.shape or not bool(torch.isfinite(got).all()) or not rel <= 0.03:
            print(f"[SAGE] the int8 P.V kernel is left out: its answer differs from fp32 attention (relative L2 "
                  f"{rel:.3g}); the fp16 P.V kernel serves every block", flush=True)
            return
        del got
        if switch == "force":
            wins, line = True, "[SAGE] the int8 P.V kernel serves the detail flows' blocks (PIXAL3D_INT8PV=force)"
        else:
            stored = _stored(device)
            if stored is not None and "int8pv" in stored:
                wins = bool(stored["int8pv"])
                line = (f"[SAGE] the {'int8' if wins else 'fp16'} P.V kernel serves the detail flows' blocks, as "
                        f"timed on this card when Spellcast3D was installed ({stored.get('int8pv_summary', 'record')})")
            else:
                wins, line, _rows = _race_int8pv(device, _kernel, call)
        print(line, flush=True)
        if wins:
            _kernel_int8pv = call
            KERNEL_INT8PV = what
    except Exception as exc:        # noqa: BLE001 - whatever the wheel or the card raises: the fp16 kernel serves
        _kernel_int8pv, KERNEL_INT8PV = None, ""
        print(f"[SAGE] the int8 P.V kernel is left out ({type(exc).__name__}: {str(exc)[:200]}); the fp16 P.V "
              f"kernel serves every block", flush=True)


def _check_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The self-check's table (see _check): q, k, v * 2^-V_SHIFT [1, 1000, 12, 128] bf16 and the fp32 answer."""
    g = torch.Generator(device=device).manual_seed(0)
    n, heads = 1000, 12
    qkv = torch.randn((1, n, 3, heads, HEAD_DIM), device=device, dtype=precision.torso(), generator=g)
    qkv[:, :, 2] *= 1000
    q, k, v = qkv[:, :, 0].contiguous(), qkv[:, :, 1].contiguous(), qkv[:, :, 2]
    want_out = F.scaled_dot_product_attention(q.float().transpose(1, 2), k.float().transpose(1, 2),
                                              v.float().transpose(1, 2)).transpose(1, 2)
    return q, k, v * 2.0 ** -V_SHIFT, want_out


def _check(device: torch.device) -> bool:
    """Every candidate kernel once on a random table (1000 tokens: more than one block of keys and not a multiple
    of it; v as a strided view and 1000 times larger than q and k, as the flows' v is), against fp32 attention:
    finite, and within 3 % relative L2 (they measure ~1.3 % there, flash_attn ~0.23 %); a candidate that fails or
    raises is left out, and _choose picks among the others (on sm_8x timing the two CUDA kernels, `_race`, when
    both are there). None left turns SageAttention off."""
    global _state, _kernel, KERNEL
    try:
        cands = _candidates(device, preference())
        q, k, vs, want_out = _check_table(device)
        tried: dict[str, Optional[torch.Tensor]] = {}
        notes = []
        for name in cands:
            try:
                got = cands[name][0](q, k, vs)
                torch.cuda.synchronize(device)
                rel = float((got.float() * 2.0 ** V_SHIFT - want_out).norm() / want_out.norm())
                if got.shape != want_out.shape or not bool(torch.isfinite(got).all()) or not rel <= 0.03:
                    raise RuntimeError(f"its answer differs from fp32 attention (relative L2 {rel:.3g}, shape "
                                       f"{tuple(got.shape)})")
                tried[name] = got
            except Exception as exc:        # noqa: BLE001 - whatever the wheel, the card or Triton raises
                tried[name] = None
                notes.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
        if notes:
            print(f"[SAGE] left out after the self-check: {'; '.join(notes)}", flush=True)
        _kernel, KERNEL = _choose(device, cands, tried)
        _setup_int8pv(device, q, k, vs, want_out)
    except Exception as exc:        # noqa: BLE001 - whatever the wheel, the card or Triton raises: flash_attn serves
        _state, _kernel = False, None
        print(f"[SAGE] SageAttention does not run on this card ({type(exc).__name__}: {str(exc)[:300]}); the "
              f"flows keep their attention backend", flush=True)
        return False
    _state = True
    print(f"[SAGE] the flows' full self-attention runs SageAttention: {KERNEL} (PIXAL3D_SAGE)", flush=True)
    return True
