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
left out (the CUDA kernel then serves every length), and a missing or broken wheel or no kernel left turns
SageAttention off with a `[SAGE]` warning, and the flows keep their attention backend.

Cards below sm_80 (Spellcast3D round L; the wheel from post7): one candidate each - the Turing op TURING_OP on sm_7x
(fp16, the tuned sm80 kernel's twin) and the Pascal op PASCAL_OP on sm_6x (fp32 in and out, int8 Q.K^T by dp4a,
with an int8 P.V twin raced as on sm_8x). Neither runs in a process until the install's calibration has run it on
this card at the flows' sizes and recorded so (`calibrate` -> `_old_card_trial`, in a child process: a kernel
that faults takes its process's CUDA context with it, which must never happen to the engine; the record's
"candidate_ok"); without that record the card's SDPA serves, except under a forced profile (PIXAL3D_PRECISION: a
developer's test of another card's path). The Turing extension carries the int8 P.V twin too (INT8PV_OP on Turing's
mma shapes, with its own v quantiser): the calibration's trial times it against the fp16 op as on sm_8x and records
the race, but it serves only where PIXAL3D_INT8PV asks for it (on / force) while INT8PV_TURING_DEFAULT is False -
the same marked blocks, the same self-check and race as on sm_8x.
The Pascal extension carries twins of its op from the wheel post8 on (Spellcast3D round O: PASCAL_TWINS, each the
shipped op's arithmetic in its order - the same bits, proven by torch.equal - with another code shape, which only a
GTX 10xx can time): the calibration's trial races them against the shipped op per P.V mode (`_pascal_twins_trial`,
the record's "pascal_twins"), and a process takes the recorded twin once it gave the shipped op's bits on the
self-check's table (`_pascal_twin`); the shipped op serves without a record, with an older wheel, or with
PIXAL3D_PASCAL_TWIN=shipped (developers: =<twin> or "fp32=<twin>,int8=<twin>" forces one).
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
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
PASCAL_TWINS = ("u28", "u44", "s28", "s24")     # torch.ops.sageattention_qattn_sm61.sageattn_pascal_<twin>
#                     (Spellcast3D round O, the wheel from post8): PASCAL_OP's arithmetic in its order - the same bits
#                     for every input - with another code shape: the main loop under Pascal's ~8 KB instruction cache
#                     (the shipped one is 8,216 bytes; u28 its score loop unrolled by 2, u44 its P.V loop by 4) and
#                     the P.V by channel slices (s28, s24: half the shared-memory traffic a key). No card of the round
#                     could time them: the calibration's trial races them against the shipped op on the user's card
#                     (`_pascal_twins_trial`) and the record's "pascal_twins" names the one that serves, per P.V mode
PASCAL_TWINS_FP32 = ("u28", "u44")  # the twins raced for the fp32 P.V (ss, lr, hr's block 0): the slice twins' fp32
#                     form holds 48.9 KB of shared memory, one block an SM against the shipped kernel's two
PASCAL_TWIN_MARGIN = 0.05   # a twin serves only where it is at least this much faster than the shipped op at every
#                     size timed (they give the same bits, so the margin only guards against the timing's noise:
#                     interleaved medians of 12-16 calls; 5 %, as the project's other races)
OLD_CARD = {6: "pascal", 7: "turing"}   # the candidate kernel of a card below sm_80, by the capability's major
TRIAL_SECONDS = 600                     # the calibration's child process trying that kernel: its time limit
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
INT8PV_TURING_DEFAULT = False   # what an unset PIXAL3D_INT8PV means on sm_7x (RTX 20xx): the fp16 P.V kernel for every
#                                 block; the user decides after the twin's end-to-end judgement on an RTX 2070


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
    scale a floating-point value exactly in bf16 and fp32 (no value of the flows comes near their exponent's ends;
    in an fp16 torso v's weights below 2^-6 become subnormals and lose bits, so there the fold must come before the
    int8 quantiser, never after it: staged/weights.py makes no int8 copy for fp16), and every product and sum of the
    attention and of the two GEMMs scales with them, so every path - SageAttention without its two scaling passes
    (0.8 ms a block at 21.5k tokens), flash_attn, SDPA, the int8 layers (their row and channel scales carry the
    powers) - gives the same bits as before. Done once per loaded network (the module's `sage_folded` says so), by
    the engine when the switch is on. Returns how many modules were folded."""
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


def _candidates(device: torch.device, want: str, trial: bool = False) -> dict[str, tuple[Callable, str]]:
    """The kernels to try on this card for the preference `want` (preference()): {name: (kernel(q, k, v), label)},
    each [B, N, H, 128] NHD in the torso's dtype -> the same. "wheel" always; on sm_8x "tuned" where the wheel has
    TUNED_OP and the choice is automatic or asks for it, "cuda" where FUSED_PREP or asked for, and "triton" where
    TRITON_TOKENS lists this architecture or it is asked for (a kernel not needed is not compiled). Below sm_80 the
    card's one candidate (OLD_CARD), where the install's calibration has run it on this card (`trial`: the
    calibration's own run of it)."""
    import sageattention as sa

    major, minor = precision.profile().capability or torch.cuda.get_device_capability(device)
    if major < 8:
        if major not in OLD_CARD:
            raise RuntimeError(f"the card is sm_{major}{minor}; SageAttention's kernels serve sm_61, sm_75 and sm_80 "
                               f"and newer")
        name, call, label = _old_card_candidate(major)
        if call is None:
            raise RuntimeError(f"the card is sm_{major}{minor}; the installed wheel has no {name} kernel")
        why = None if trial else _old_card_unchecked(device, name)
        if why is not None:
            raise RuntimeError(f"the {name} kernel is not used: {why}")
        if not trial:       # the twin the record chose (it runs on the self-check's table: only once allowed above)
            name, call, label = _old_card_candidate(major, device)
        return {name: (call, label)}

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


def _old_card_candidate(major: int, device: Optional[torch.device] = None) -> tuple[str, Optional[Callable], str]:
    """(name, the call - None where the wheel lacks the extension -, label) of the kernel a card below sm_80 may run:
    the Pascal op on sm_6x, the Turing op on sm_7x (OLD_CARD). With `device` (the engine's use; not the calibration's
    trial, which tries the shipped op) the Pascal op's twin the record chose for the fp32 P.V (`_pascal_twin`)."""
    if major == 6:
        op, twin = _pascal_twin(device, False) if device is not None else (_pascal_op(), "shipped")
        call = _pascal_call(op, False) if op is not None else None
        return "pascal", call, f"{_pascal_name(twin)} (int8 Q.K^T by dp4a, fp32 P.V)"
    op = _turing_op()
    return "turing", (_wheel_call(op) if op is not None else None), f"{TURING_OP} for sm_75 (per_warp, fp16+fp32)"


def _old_card_unchecked(device: torch.device, name: str) -> Optional[str]:
    """Why the kernel `name` of a card below sm_80 may not run in this process: None where the install's calibration
    ran it on this card at the flows' sizes and recorded so (the record's "candidate_ok"), or where the profile is
    forced (PIXAL3D_PRECISION: a developer's test of another card's path); else the reason, for the [SAGE] line."""
    stored = _stored(device)
    if stored is not None and stored.get("candidate") == name:
        if stored.get("candidate_ok"):
            return None
        return f"the install's calibration found it wanting on this card ({stored.get('why', 'no reason recorded')})"
    if os.environ.get("PIXAL3D_PRECISION", "").strip():
        return None
    return "the install's calibration has not run it on this card (python -m pixal3d.calibrate)"


def _old_card_trial(device: torch.device, progress: Optional[Callable[[dict], None]] = None) -> dict:
    """The kernel of a card below sm_80 tried on this card, as the calibration's child process runs it: the
    self-check's table (the 3 % bar of _check), then calls at RACE_TOKENS' sizes, timed (the flows' sizes: a kernel
    that faults does so here, not in the engine), and its int8 P.V twin against it on the table and in the race
    (_race_int8pv, CALIBRATION_RACE; sm_6x unless PIXAL3D_INT8PV is off, sm_7x always: `_turing_int8pv_trial`).
    `progress` is given the fields once the candidate's own trial has passed, before its twin runs. Returns the
    record's fields: "candidate", "candidate_ok", "why", "timings", "summary", and where the twin was tried "int8pv",
    "int8pv_timings", "int8pv_summary", "int8pv_why"."""
    major, _minor = precision.profile().capability or torch.cuda.get_device_capability(device)
    name, call, label = _old_card_candidate(major)
    found: dict = {"candidate": name, "candidate_ok": False}
    if call is None:
        found["why"] = f"the installed wheel has no {name} kernel"
        return found
    told = (lambda part: progress({**found, **part})) if progress is not None else None
    try:
        return {**found, **_old_card_trial_run(device, major, call, label, told)}
    except Exception as exc:        # noqa: BLE001 - whatever the wheel or the card raises: recorded, SDPA serves
        found["why"] = f"{type(exc).__name__}: {(str(exc).strip().splitlines() or [''])[0][:200]}"
        return found


def _old_card_trial_run(device: torch.device, major: int, call: Callable, label: str,
                        progress: Optional[Callable[[dict], None]] = None) -> dict:
    found: dict = {}
    q, k, vs, want_out = _check_table(device)

    def off(got: torch.Tensor) -> Optional[str]:
        rel = float((got.float() * 2.0 ** V_SHIFT - want_out).norm() / want_out.norm())
        if got.shape != want_out.shape or not bool(torch.isfinite(got).all()) or not rel <= 0.03:
            return f"its answer differs from fp32 attention (relative L2 {rel:.3g})"
        return None

    got = call(q, k, vs)
    torch.cuda.synchronize(device)
    why = off(got)
    if why is not None:
        found["why"] = why
        return found
    del got
    heads, dt, rows = 12, precision.torso(), []
    for n, calls in RACE_TOKENS:
        g = torch.Generator(device=device).manual_seed(n)
        tables = [torch.randn((1, n, heads, HEAD_DIM), dtype=dt, device=device, generator=g) for _ in range(3)]
        events = []
        for _ in range(calls):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            call(*tables)
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(device)
        rows.append((n, statistics.median(a.elapsed_time(b) for a, b in events)))
        del tables, events
    found.update(candidate_ok=True, why=f"{label} ran at the flows' sizes",
                 timings=[{"tokens": n, "ms": round(ms, 3)} for n, ms in rows],
                 summary=", ".join(f"{n:,} tokens {ms:.2f} ms" for n, ms in rows))
    if progress is not None:
        progress(dict(found))
    if major == 7:
        found.update(_turing_int8pv_trial(device, call, q, k, vs, off))
    if major == 6 and int8pv_switch() != "off":
        int8 = _pascal_call(_pascal_op(), True)
        got = int8(q, k, vs)
        torch.cuda.synchronize(device)
        why = off(got)
        del got
        if why is not None:
            found.update(int8pv=False, int8pv_summary="not timed",
                         int8pv_why=f"the int8 P.V kernel is left out: {why}")
        else:
            wins, line, rows8 = _race_int8pv(device, call, int8, CALIBRATION_RACE)
            found.update(int8pv=wins, int8pv_why=line,
                         int8pv_timings=[{"tokens": n, "fp32_ms": round(a, 3), "int8_ms": round(b, 3)}
                                         for n, a, b in rows8],
                         int8pv_summary=", ".join(f"{n:,} tokens {a:.2f} -> {b:.2f} ms ({100 * (b / a - 1):+.1f} %)"
                                                  for n, a, b in rows8) or "not timed")
    if major == 6:
        if progress is not None:        # a twin that takes the process down leaves the races above standing
            progress(dict(found))
        found["pascal_twins"] = _pascal_twins_trial(device, q, k, vs)
    return found


def _turing_int8pv_trial(device: torch.device, fp16: Callable, q: torch.Tensor, k: torch.Tensor, vs: torch.Tensor,
                         off: Callable[[torch.Tensor], Optional[str]]) -> dict:
    """The Turing int8 P.V twin in the calibration's trial, after the fp16 op passed: the self-check's table (`off`,
    the 3 % bar), then the race against the fp16 call (_race_int8pv, CALIBRATION_RACE). Timed whatever
    PIXAL3D_INT8PV says: the record holds how fast the twin is on this card, the switch decides in the engine
    whether it serves. Always returns "int8pv" (False where the wheel lacks the twin, it answers wrongly or raises),
    so that the race counts as made (int8pv_race_lacking); a failure here leaves the fp16 op's trial standing."""
    def left_out(why: str) -> dict:
        return {"int8pv": False, "int8pv_summary": "not timed", "int8pv_why": f"the int8 P.V kernel is left out: {why}"}

    try:
        op = _turing_int8pv_op()
        if op is None:
            return left_out("the installed wheel has no int8 P.V kernel for sm_75")
        int8 = _turing_int8pv_call(op)
        got = int8(q, k, vs)
        torch.cuda.synchronize(device)
        why = off(got)
        del got
        if why is not None:
            return left_out(why)
        wins, line, rows8 = _race_int8pv(device, fp16, int8, CALIBRATION_RACE)
    except Exception as exc:        # noqa: BLE001 - whatever the wheel or the card raises: the fp16 op serves
        return left_out(f"{type(exc).__name__}: {(str(exc).strip().splitlines() or [''])[0][:200]}")
    return {"int8pv": wins, "int8pv_why": line,
            "int8pv_timings": [{"tokens": n, "fp16_ms": round(a, 3), "int8_ms": round(b, 3)} for n, a, b in rows8],
            "int8pv_summary": ", ".join(f"{n:,} tokens {a:.2f} -> {b:.2f} ms ({100 * (b / a - 1):+.1f} %)"
                                        for n, a, b in rows8) or "not timed"}


def int8pv_race_lacking(section: object) -> bool:
    """Whether a calibration record's "sage" section lacks the race of the int8 P.V twin that `_old_card_trial` now
    makes on sm_7x: a Turing section whose fp16 op ran at the flows' sizes, without "int8pv". The installer's due
    rule (code/spellcast/install/components.py in Spellcast3D) is the same."""
    return (isinstance(section, dict) and section.get("candidate") == "turing" and bool(section.get("candidate_ok"))
            and "int8pv" not in section)


def _old_card_trial_in_child(major: int, minor: int) -> dict:
    """`_old_card_trial` in a child process with this process's environment and the fork on its path: a kernel
    that faults takes its process's CUDA context with it, so the parent (the calibration) stays up and records the
    failure, and the engine never tries that kernel on this card. The child's last line of JSON is the result; it
    also prints the candidate's fields before its int8 P.V twin runs, so a twin that takes the process down leaves
    the candidate's own trial standing (the twin recorded as left out). A process that printed its whole answer and
    then ended with an error (or ran out of time after it: its output is salvaged the same way) keeps that answer,
    with "exit_code" and "exit_note", the Pascal op's twins left out."""
    import json
    import subprocess
    import sys

    name = OLD_CARD.get(major, f"sm_{major}{minor}")
    code = ("import json, torch\nfrom pixal3d.modules import sage\n"
            "say = lambda found: print(json.dumps(found), flush=True)\n"
            "say(sage._old_card_trial(torch.device('cuda'), progress=say))")
    def text(out: object) -> str:
        return out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out or "")

    try:
        run = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[2]),
                             env=os.environ.copy(), capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=TRIAL_SECONDS)
        stdout, stderr, returncode = run.stdout or "", run.stderr or "", run.returncode
        ended = f"the trial process ended with code {returncode}"
    except subprocess.TimeoutExpired as exc:   # what it printed before its time ran out is salvaged like a crash's
        stdout, stderr, returncode = text(exc.stdout), text(exc.stderr), None
        ended = f"the trial did not finish in {TRIAL_SECONDS} s"
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    found = None
    if lines:
        try:
            found = json.loads(lines[-1])
        except ValueError:
            found = None
    if found is not None and returncode == 0:
        found.setdefault("summary", found.get("why", ""))
        return found
    tail = " | ".join((stderr or stdout).strip().splitlines()[-3:])
    if found is not None and found.get("candidate_ok") and ("pascal_twins" in found if major == 6
                                                            else "int8pv" in found if major == 7 else True):
        # the trial finished (its last line is the whole answer) and the process went wrong after it - in its
        # teardown, say, with a twin's sticky CUDA error: the candidate's and the int8 P.V race's results stand,
        # the Pascal op's twins are left out (they are only speed; a process that ended so promotes none)
        found.setdefault("summary", found.get("why", ""))
        found["exit_code"] = returncode
        found["exit_note"] = f"{ended} after the trial's last line: {tail[:300]}"
        twins = found.get("pascal_twins")
        if isinstance(twins, dict):
            chose = {mode: twins.get(mode) for mode in ("fp32", "int8") if twins.get(mode) not in (None, "shipped")}
            twins.update(fp32="shipped", int8="shipped")
            twins["error"] = "; ".join(filter(None, (twins.get("error"), f"{ended} after the twins ran"
                                                     + (f" (the race chose {chose}: left out)" if chose else "")
                                                     + f": {tail[:300]}")))
        return found
    if found is not None and found.get("candidate_ok") and "int8pv" not in found:
        found.setdefault("summary", found.get("why", ""))
        found.update(int8pv=False, int8pv_summary="not timed",
                     int8pv_why=f"the int8 P.V kernel is left out: {ended} while it ran: {tail[:300]}")
        return found
    if found is not None and found.get("candidate_ok") and major == 6 and "pascal_twins" not in found:
        found.setdefault("summary", found.get("why", ""))     # the Pascal op's twins took the trial down
        found["pascal_twins"] = {"present": [], "fp32": "shipped", "int8": "shipped",
                                 "error": f"{ended} while the twins ran: {tail[:300]}"}
        return found
    if returncode is None:
        return {"candidate": name, "candidate_ok": False, "summary": "not run", "why": ended}
    return {"candidate": name, "candidate_ok": False, "summary": "the trial failed", "why": f"{ended}: {tail[:300]}"}


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


def _turing_int8pv_op() -> Optional[Callable]:
    """The Turing extension's int8 P.V op (INT8PV_OP in _qattn_sm75, beside its v quantiser), or None for a wheel
    without it."""
    if _turing_op() is None:        # (imports the extension)
        return None
    try:
        ops = torch.ops.sageattention_qattn_sm75
        getattr(ops, "quant_per_block_int8_v_cuda")
        return getattr(ops, INT8PV_OP)
    except (AttributeError, RuntimeError):
        return None


def _turing_int8pv_call(op: Callable) -> Callable:
    """The int8 P.V attention on sm_7x ([B, N, H, 128] NHD fp16 -> the same): _int8pv_call's steps, with the v
    quantiser of the Turing extension (sageattention.quant.per_block_int8_v imports _qattn_sm80, which a build for
    sm_75 alone lacks; the extension compiles the same quantiser source)."""
    from sageattention.quant import per_warp_int8

    quantise_v = torch.ops.sageattention_qattn_sm75.quant_per_block_int8_v_cuda

    def run(q, k, v):
        torch.cuda.set_device(v.device)
        km = k.mean(dim=1, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=16, BLKK=64)
        b, n, h, d = v.shape
        padded = (n + 63) // 64 * 64
        v_int8 = torch.empty((b, h, d, padded), dtype=torch.int8, device=v.device)
        v_scale = torch.empty((b, h, padded // 64, d), dtype=torch.float32, device=v.device)
        v_mean = torch.empty((b, h, d), dtype=torch.float32, device=v.device)
        quantise_v(v, v_int8, v_scale, v_mean, 0, 1)
        o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
        op(q_int8, k_int8, v_int8, o, q_scale, k_scale, v_scale, v_mean, 0, 0, 2, q.size(-1) ** -0.5, 0)
        return o

    return run


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


def _pascal_name(twin: str) -> str:
    return PASCAL_OP if twin == "shipped" else f"{PASCAL_OP}_{twin} (the shipped kernel's twin {twin}, its bits)"


def _pascal_twin_op(twin: str) -> Optional[Callable]:
    """The wheel's twin `twin` of the Pascal op (sageattn_pascal_<twin>, PASCAL_TWINS), or None (an older wheel)."""
    if _pascal_op() is None:        # (imports the extension)
        return None
    try:
        return getattr(torch.ops.sageattention_qattn_sm61, f"{PASCAL_OP}_{twin}")
    except (AttributeError, RuntimeError):
        return None


_pascal_twins: dict = {}            # (device, int8pv) -> (op, twin name): this process's choice


def _pascal_twin_wanted(device: torch.device, mode: str) -> tuple[str, str]:
    """(the twin for the P.V mode "fp32" / "int8", why): PIXAL3D_PASCAL_TWIN (developers: a twin's name for both
    modes, or "fp32=<twin>,int8=<twin>", or shipped / off), else the calibration record's race ("pascal_twins", made
    on this card with the installed wheel: `_stored`), else "shipped"."""
    text = os.environ.get("PIXAL3D_PASCAL_TWIN", "").strip().lower()
    if text:
        named: dict = {}
        for part in (p.strip() for p in text.split(",") if p.strip()):
            if "=" in part:
                key, value = (s.strip() for s in part.split("=", 1))
                named[key] = value
            else:
                named.setdefault("fp32", part)
                named.setdefault("int8", part)
        name = named.get(mode, "shipped")
        return ("shipped" if name in ("off", "shipped", "") else name), f"PIXAL3D_PASCAL_TWIN={text}"
    stored = _stored(device)
    race = stored.get("pascal_twins") if stored is not None else None
    if isinstance(race, dict) and race.get(mode) in PASCAL_TWINS:
        return race[mode], (f"timed on this card when Spellcast3D was installed: "
                            f"{race.get(mode + '_summary', 'calibration record')}")
    return "shipped", "no race recorded"


def _pascal_twin(device: torch.device, int8pv: bool) -> tuple[Optional[Callable], str]:
    """The Pascal op for the P.V mode (`int8pv`) in this process: the twin `_pascal_twin_wanted` names, once it gave
    the shipped op's bits on the self-check's table (torch.equal; on a card or wheel where it does not, one line
    says so and the shipped op serves), else the shipped op. (op, the twin's name or "shipped"); cached."""
    key = (str(device), bool(int8pv))
    if key in _pascal_twins:
        return _pascal_twins[key]
    shipped = _pascal_op()
    mode = "int8" if int8pv else "fp32"
    name, why = _pascal_twin_wanted(device, mode)
    chosen = (shipped, "shipped")
    if name != "shipped" and shipped is not None:
        twin = _pascal_twin_op(name)
        if twin is None:
            print(f"[SAGE] the Pascal kernel's twin {name} is not in the installed wheel: the shipped kernel serves "
                  f"the {mode} P.V calls", flush=True)
        else:
            try:
                q, k, vs, _want = _check_table(device)
                same = torch.equal(_pascal_call(shipped, int8pv)(q, k, vs), _pascal_call(twin, int8pv)(q, k, vs))
                del q, k, vs, _want
            except Exception as exc:        # noqa: BLE001 - whatever the wheel or the card raises: the shipped op
                same, why = False, f"{type(exc).__name__}: {str(exc)[:200]}"
            if same:
                chosen = (twin, name)
                print(f"[SAGE] the Pascal kernel's twin {name} serves the {mode} P.V calls ({why})", flush=True)
            else:
                print(f"[SAGE] the Pascal kernel's twin {name} did not give the shipped kernel's bits on this card "
                      f"({why}): the shipped kernel serves the {mode} P.V calls", flush=True)
    _pascal_twins[key] = chosen
    return chosen


def _pascal_twins_trial(device: torch.device, q: torch.Tensor, k: torch.Tensor, vs: torch.Tensor) -> dict:
    """The calibration trial's race of the Pascal op's twins (PASCAL_TWINS in the wheel) against the shipped op, per
    P.V mode (fp32: PASCAL_TWINS_FP32; int8 unless PIXAL3D_INT8PV is off): each twin's answer on the self-check's
    table (q, k, vs) and on the race's tables must be the shipped op's bits (torch.equal), then whole calls at
    CALIBRATION_RACE's sizes, interleaved in a rotating order, CUDA events, medians (`_race_pascal_twins`); the twin
    with the best worst-size ratio serves where it is PASCAL_TWIN_MARGIN faster at every size. Returns the record's
    "pascal_twins": {"present", "fp32", "int8" (a twin or "shipped"), "<mode>_timings", "<mode>_summary",
    "<mode>_why"}. Never raises: a failure records the shipped op ("error")."""
    found: dict = {"present": [], "fp32": "shipped", "int8": "shipped"}
    try:
        shipped = _pascal_op()
        found["present"] = [t for t in PASCAL_TWINS if _pascal_twin_op(t) is not None]
        for mode, int8pv in (("fp32", False), ("int8", True)):
            names = [t for t in found["present"] if int8pv or t in PASCAL_TWINS_FP32]
            if not names:
                found[f"{mode}_why"] = "no twin in the installed wheel"
                continue
            if int8pv and int8pv_switch() == "off":
                found[f"{mode}_why"] = "PIXAL3D_INT8PV is off"
                continue
            calls = {"shipped": _pascal_call(shipped, int8pv)}
            want = calls["shipped"](q, k, vs)
            differ = []
            for t in names:
                call = _pascal_call(_pascal_twin_op(t), int8pv)
                if torch.equal(call(q, k, vs), want):
                    calls[t] = call
                else:
                    differ.append(t)
            del want
            winner, line, rows = _race_pascal_twins(device, calls, CALIBRATION_RACE)
            found[mode] = winner
            found[f"{mode}_why"] = line + (f"; left out (other bits on the self-check's table): "
                                           f"{', '.join(differ)}" if differ else "")
            found[f"{mode}_timings"] = rows
            found[f"{mode}_summary"] = "; ".join(
                f"{r['tokens']:,} tokens: shipped {r['ms']['shipped']:.2f} ms, " + ", ".join(
                    f"{t} {ms:.2f} ({100 * (ms / r['ms']['shipped'] - 1):+.1f} %)" for t, ms in r["ms"].items()
                    if t != "shipped") for r in rows) or "not timed"
    except Exception as exc:        # noqa: BLE001 - the twins never cost the shipped op its place
        found["error"] = f"{type(exc).__name__}: {(str(exc).strip().splitlines() or [''])[0][:200]}"
        found["fp32"] = found["int8"] = "shipped"
    return found


def _race_pascal_twins(device: torch.device, calls: dict, plan: tuple) -> tuple[str, str, list]:
    """The shipped call and its twins (`calls`: name -> call, "shipped" first) timed on this card at each of `plan`'s
    sizes on the same random q, k, v (12 heads x 128, the torso's dtype), the calls in a rotating order, CUDA events,
    the medians; a twin whose answer at a size differs from the shipped op's is left out. Returns (the winner or
    "shipped", the [SAGE] line, rows [{"tokens", "ms": {name: median ms}, "differ": [names]}])."""
    names = list(calls)
    heads, began = 12, time.perf_counter()
    rows, skipped, out = [], [], set()
    torch.cuda.synchronize(device)
    for n, reps in plan:
        need = n * heads * HEAD_DIM * 4 * 7                # q, k, v, two answers, the quantised copies
        free = torch.cuda.mem_get_info(device)[0]
        spare = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if need * 1.25 > free + spare:
            skipped.append(n)
            continue
        g = torch.Generator(device=device).manual_seed(n)
        dt = precision.torso()
        q, k, v = (torch.randn((1, n, heads, HEAD_DIM), dtype=dt, device=device, generator=g) for _ in range(3))
        want = calls["shipped"](q, k, v)
        differ = [t for t in names[1:] if t not in out and not torch.equal(calls[t](q, k, v), want)]
        out.update(differ)
        del want
        live = [t for t in names if t not in out]
        events: dict = {t: [] for t in live}
        for i in range(reps):
            for t in live[i % len(live):] + live[:i % len(live)]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                calls[t](q, k, v)
                end.record()
                events[t].append((start, end))
        torch.cuda.synchronize(device)
        rows.append({"tokens": n, "ms": {t: round(statistics.median(a.elapsed_time(b) for a, b in pairs), 3)
                                         for t, pairs in events.items()}, "differ": differ})
        del q, k, v, events
    seconds = time.perf_counter() - began
    if not rows:
        return "shipped", (f"[SAGE] the shipped Pascal kernel serves: its twins were not timed (no room for the "
                           f"tables of {', '.join(f'{n:,}' for n in skipped)} tokens)"), rows
    ratio = {t: max(r["ms"][t] / r["ms"]["shipped"] for r in rows) for t in names[1:]
             if t not in out and all(t in r["ms"] for r in rows)}
    best = min(ratio, key=ratio.get) if ratio else None
    text = ", ".join(f"{t} {100 * (x - 1):+.1f} %" for t, x in sorted(ratio.items(), key=lambda kv: kv[1]))
    if skipped:
        text += f"; {', '.join(f'{n:,}' for n in skipped)} tokens not timed (no room)"
    if best is not None and ratio[best] <= 1 - PASCAL_TWIN_MARGIN:
        return best, (f"[SAGE] the Pascal kernel's twin {best} serves, timed against the shipped one (the worst "
                      f"size: {text}; {seconds:.2f} s)"), rows
    return "shipped", (f"[SAGE] the shipped Pascal kernel serves: no twin is {100 * PASCAL_TWIN_MARGIN:.0f} % faster "
                       f"at every size timed ({text or 'none timed'}; {seconds:.2f} s)"), rows


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
    "capability", "tuned": bool, "summary", "timings": [{"tokens", "shipped_ms", "tuned_ms"}], "why"}. A card below
    sm_80 records its one candidate's trial instead (`_old_card_trial`, in a child process: "candidate",
    "candidate_ok", and on sm_6x the int8 P.V race)."""
    major, minor = precision.profile().capability or torch.cuda.get_device_capability(device)
    found = {"wheel": _wheel_version(), "capability": [major, minor], "tuned": False, "timings": []}
    if major in OLD_CARD:
        found.update(_old_card_trial_in_child(major, minor))
        return found
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


def int8pv_switch(major: Optional[int] = None) -> str:
    """PIXAL3D_INT8PV: "on" / unset (the int8 P.V kernel where the wheel has it, after its self-check and the
    race), "off" (never), "force" (developers: without the race). Unset on a card whose capability's major is
    `major` 7 (Turing): INT8PV_TURING_DEFAULT decides ("off" while it is False)."""
    text = os.environ.get("PIXAL3D_INT8PV", "").strip().lower()
    if text in _OFF:
        return "off"
    if text == "force":
        return "force"
    if text == "" and major == 7 and not INT8PV_TURING_DEFAULT:
        return "off"
    return "on"


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


_twin_device: Optional[torch.device] = None     # the card of this process's self-check (`_setup_int8pv` sets it):
#                                                 the one whose record names the Pascal op's twin for the int8 P.V


def _int8pv_candidate(major: int) -> tuple[Optional[Callable], str]:
    """The int8 P.V call of a card whose capability's major is `major`, and its label: on sm_8x INT8PV_OP of
    _qattn_sm80, on sm_7x its twin in _qattn_sm75, on sm_6x the Pascal op's int8 P.V (once `_setup_int8pv` named the
    card: the op's twin its record chose for the int8 P.V, `_pascal_twin`); (None, "") where the wheel has none."""
    if major == 8:
        op = _int8pv_op()
        return ((_int8pv_call(op) if op is not None else None),
                f"{INT8PV_OP} (int8 P.V: u8 P per 64-key block, v per block and channel)")
    if major == 7:
        op = _turing_int8pv_op()
        return ((_turing_int8pv_call(op) if op is not None else None),
                f"{INT8PV_OP} for sm_75 (int8 P.V: u8 P per 64-key block, v per block and channel)")
    if major == 6:
        op, twin = _pascal_twin(_twin_device, True) if _twin_device is not None else (_pascal_op(), "shipped")
        return ((_pascal_call(op, True) if op is not None else None),
                f"{_pascal_name(twin)} (int8 P.V by dp4a: u8 P per 32-key block, v per block and channel)")
    return None, ""


def _setup_int8pv(device: torch.device, q: torch.Tensor, k: torch.Tensor, vs: torch.Tensor,
                  want_out: torch.Tensor) -> None:
    """The int8 P.V kernel beside the chosen fp16 one (`_kernel_int8pv`): where the wheel has the op for the card
    (sm_8x, sm_7x, sm_6x: `_int8pv_candidate`) and the switch allows it (int8pv_switch: on sm_7x only when asked
    for, INT8PV_TURING_DEFAULT), it runs once on the self-check's table against fp32 attention (the 3 % bar
    of _check; it measures ~1.5 % there, the fp16 kernel ~1.2 %), then serves where the calibration record says
    so, or where it wins the race against the fp16 call (PIXAL3D_INT8PV=force: without the race). Off, or any
    failure, keeps the fp16 kernel for every block (the numbers as before)."""
    global _kernel_int8pv, KERNEL_INT8PV, _twin_device
    _kernel_int8pv, KERNEL_INT8PV = None, ""
    if _kernel is None:
        return
    try:
        major, _minor = precision.profile().capability or torch.cuda.get_device_capability(device)
        switch = int8pv_switch(major)
        if switch == "off":
            return
        _twin_device = device
        call, what = _int8pv_candidate(major)
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
