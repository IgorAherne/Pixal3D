"""The int8 (W8A8) kernels of the flows' linear layers (modules/int8.py chooses where they run): Triton GEMMs

    y = (x_int8 @ w_int8^T) [int32 accumulate] * row_scale[M] * col_scale[N] + bias   [-> GELU(tanh)]  -> bf16

Symmetric quantisation, no zero points: weights per output channel (one-off at load), activations per row (token)
at run time, or per row and per 128 input channels ("group" scales, a finer grain for the quality study); with
group scales on the activations the weights may have them too (one per output channel and 128 input channels:
the group GEMM multiplies each 128-K slab's int32 partial product by the row's and the channel's scale of that
slab), and SmoothQuant's per-input-channel vector s goes into the quantisers (x * (1 / s) as x is quantised, W * s
before W is), or into fc1's epilogue for fc2's input.

    quantize_weight(w, smooth, group)  bf16 [N, K] -> (int8 [N, K] row-major, fp32 scales [N] or [N, K / group])
    quantize_rows(x, group=0)     bf16 [..., K] -> (int8 [M, K], fp32 scales [M] or [M, K / group])
    norm_quant(x, eps, ...)       fused.norm_modulate's math (LayerNorm32 [affine] [* (1 + scale) + shift], the
                                  default "rounded" mode) emitting int8 rows + scales instead of bf16
    int8_linear(x, w_q, w_s, bias, *, gelu=False, path=None, out_int8=False)
                                  x: bf16 [..., K] or a pre-quantised (x_q, x_s) pair -> bf16 [..., N]
                                  (out_int8: the result requantised to int8 + scales, the next layer's input)
    int8_mlp(x, w1, w2)           fc1 + GELU -> int8 (group scales) -> fc2: the [M, 8192] hidden layer never in bf16

Paths (`path=`; None = the table PATHS, chosen from the round-I measurements on the RTX 3060):
    "triton"   quantize_rows + one Triton GEMM with the scaling / bias / GELU epilogue fused (per-row scales).
               THE FASTEST at every measured shape: 2.5-3.2x the bf16 F.linear with the quantiser in the call,
               2.9-3.6x when the input arrives as int8 (norm_quant, fc1's epilogue); 84-88 TOPS.
    "intmm"    quantize_rows + torch._int_mm (cuBLASLt IMMA, int32 [M, N] out, rows in chunks of CHUNK_ROWS)
               + one Triton epilogue kernel: 1.6-2.7x (the int32 round trip); ties "triton" only on fc2 at small M.
    "group"    quantize_rows(group=128) + the Triton GEMM with the accumulator rescaled every 128-K block:
               1.2-1.3x slower than "triton" (66-68 TOPS), for the quality study's finer scales; weight scales
               [N, K / 128] (quantize_weight(group=128)) add one multiply per element and slab (round J). With
               per-channel weight scales a layer form may run pixal3d_kernels' sm_80+ twin of that kernel instead
               (the same bits, where the install's race found it faster: group_cuda_tile; RTX 3060: to_qkv -5 %,
               fc1 -7 %, fc2 stays Triton's).
The whole block (3 norms + 7 token-sized linears + GELU) runs 2.67-2.83x faster than bf16 (README.md).

Capability gate (`capability()`): the Triton GEMM needs Triton's int8 mma.sync (m16n8k32), which Triton emits for
sm_80+ (Ampere and up; only sm_86, an RTX 3060, was measured). Below sm_80 (Turing, Pascal; Spellcast3D round L) the
GEMMs are pixal3d_kernels' (`cuda_linear`: the fork's own W8A8 kernel with the scaling, bias and GELU in its
epilogue, per-row or group scales on either operand - Turing's int8 tensor cores at 48-57 TOPS on an RTX 2060 Super
against 24-26 TFLOPS of fp16, Pascal's dp4a; the same sums as `reference`), or, where that package is not installed
or fails its self-check, torch._int_mm's (cuBLASLt: exact, per-row scales only, so the layers the policy gives
group scales then keep the torso's dtype - modules/int8.py) with the epilogue in torch ops (`epilogue_torch`) or
their Triton twin where Triton runs on the card and gives the same bits at the self-check (`_triton_elementwise`;
Triton has no int8 MMA for sm_75 but its elementwise kernels run there). The quantisers likewise (`quantize_rows_torch`
or the Triton kernel). The outputs come in the torso's dtype (`out_dtype`: bf16, fp16 or fp32; modules/precision.py).
Below sm_61, or when the first-use self-check (`self_check`) fails, every call runs F.linear on the dequantised
weight (the fallback) and says so once.

The dp4a kernel's twins (Spellcast3D round O; pixal3d_kernels 0.2.1 on: DP4A_TWINS): on the cards whose GEMMs are
dp4a (below sm_75: the GTX 10xx, Volta) the package also holds the dp4a kernel with other code shapes and the same
bits (csrc/int8_gemm_dp4a_twins.cu: q4 a K stage as 4 trips of 4 k4 rows, f5 a group's first row restarting the
integer sums so that the conversion needs no reset - both bring the group kernels' stage loop under Pascal's ~8 KB
instruction cache). The install's race times them against the shipped kernel on the card itself (old_cards'
calibration step: `calibrate_dp4a_twins`, the record's old_cards.dp4a) per scale form (group / rows); a process takes
the recorded twin (`_dp4a_twin`) once it gave the shipped kernel's bits on a small table at first use, the shipped
kernel otherwise (no record, an older package, PIXAL3D_DP4A_TWIN=shipped). On the old cards' profiles an MLP whose
hidden layer would pass MLP_SLAB_MIB runs in row slabs (`cuda_mlp`, round O: the same bits, a bounded table).
"""

from __future__ import annotations

import math
import os
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from . import precision

GROUP = 128                 # activation group size of the "group" path (= that kernel's BLOCK_K)
CHUNK_ROWS = 8192           # "intmm": rows per torch._int_mm call (bounds the int32 temporary: 8192 x N x 4 B)
TINY = 1e-30                # absmax floor: an all-zero row quantises to zeros with a tiny scale, not to NaN
NORM_WARPS = 4              # norm_quant's num_warps (C = 1536): 4 warps 0.461 ms, 8 warps 0.559 ms at M = 21504;
#                             the flows' blocks pass norm_modulate's 8, the warps that give its bits exactly

# Triton GEMM configurations: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_stages, num_warps), from the sweep of
# 14 configs x 6 shapes x M in {2024, 4096, 8192, 21504, 30000} on the RTX 3060 (bench_int8.py --parts triton):
CFG_BIG = (128, 128, 64, 8, 4, 4)       # the fastest for every K <= 2048 shape at every M (84-88 TOPS)
CFG_LONGK = (128, 256, 64, 8, 3, 8)     # K >= 4096 (fc2): CFG_BIG falls to 50 TOPS there (A re-read past the L2)
CFG_GROUP = (128, 128, 128, 8, 3, 8)    # group scales: BLOCK_K = 128 = the group; 4-warp / 256-wide tiles spill
CFG_FC1_INT8 = (128, 128, 64, 8, 4, 4)  # fc1 + GELU -> int8 tiles (BLOCK_N = GROUP): spills 144 B and is still
#                                         the faster (MLP 15.7 ms vs 18.1 ms with the 8-warp tile at M = 21504)

# per (N, K): the path; measured fastest everywhere: the Triton GEMM with the fused epilogue (README.md)
PATHS: dict = {}
DEFAULT_PATH = "triton"

_kernels = None
# gemm: "triton" (the fused GEMM, sm_80+) or "intmm" (torch._int_mm); elementwise: the quantiser and epilogue,
# "triton" or "torch" (decided by ready(), before any self-check: quantize_rows / epilogue dispatch on it)
_state = {"ok": None, "why": "", "warned": False, "gemm": "", "elementwise": ""}


# =====================================================================================================================
# Capability gate
# =====================================================================================================================

def capability(device: Optional[torch.device] = None) -> Tuple[bool, bool, str]:
    """(torch._int_mm usable [int8 GEMMs at all], the Triton int8 GEMM usable, the reason when not)."""
    if not torch.cuda.is_available():
        return False, False, "no CUDA device"
    major, minor = precision.profile().capability or torch.cuda.get_device_capability(device)
    if (major, minor) < (6, 1):
        return False, False, f"sm_{major}{minor}: no int8 GEMM below sm_61 (dp4a)"
    if major < 8:
        return True, False, f"sm_{major}{minor}: Triton's int8 mma.sync path needs sm_80+; torch._int_mm serves"
    try:
        import triton  # noqa: F401
    except Exception as exc:        # noqa: BLE001
        return True, False, f"Triton not importable ({type(exc).__name__})"
    return True, True, ""


def gemm_path() -> str:
    """"triton" (sm_80+), "cuda" (pixal3d_kernels' GEMM, Turing and Pascal) or "intmm" (torch._int_mm, per-row
    scales only) once ready() has decided ("" before)."""
    return str(_state["gemm"])


def has_group_gemm() -> bool:
    """Whether group scales on the activations can be applied (the Triton GEMM and pixal3d_kernels' rescale every
    128-K slab; torch._int_mm multiplies over the whole K and takes per-row scales only)."""
    return _state["gemm"] in ("triton", "cuda")


_cuda_op = None
DP4A_TWINS = ("q4", "f5")   # torch.ops.pixal3d_kernels.int8_gemm_group_<twin> (pixal3d_kernels 0.2.1 on): the dp4a
#                             kernel's twins, int8_gemm_group's arguments but `kernel`, its bits (the docstring)
DP4A_TWIN_MARGIN = 0.05     # a twin serves only where it is this much faster than the shipped kernel at every size
#                             (the same bits: the margin only guards against the timing's noise; 5 %, as the
#                             project's other races)
DP4A_RACE = {"group": (("qkv", 1536, 4608, False), ("fc1", 1536, 8192, True), ("fc2", 8192, 1536, False)),
             "rows": (("out", 1536, 1536, False),)}     # the hr / tex flows' layers by scale form: (name, K, N, GELU)
DP4A_RACE_ROWS = ((8192, 4), (21504, 4))    # (M, rounds): a 1024 prop's pass and a 1536 one's; 5 calls a round
_dp4a: dict = {}            # (device, grouped) -> the twin's op or None (the shipped kernel): this process's choice


def cuda_gemm_op() -> Optional[Callable]:
    """pixal3d_kernels' int8_gemm_group (Spellcast3D round L: W8A8 with group scales on Turing's tensor cores and
    Pascal's dp4a; its README holds the contract), or None where the package is not installed."""
    global _cuda_op
    if _cuda_op is None:
        try:
            import pixal3d_kernels  # noqa: F401  (registers torch.ops.pixal3d_kernels)

            _cuda_op = torch.ops.pixal3d_kernels.int8_gemm_group
        except Exception:       # noqa: BLE001 - no package, or one that does not load: torch._int_mm serves
            _cuda_op = False
    return _cuda_op or None


def cuda_linear(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                bias: Optional[torch.Tensor], *, gelu: bool = False, out_dtype: torch.dtype = torch.float16):
    """pixal3d_kernels' GEMM: int8 [M, K] with per-row or group scales x int8 [N, K] with per-channel or
    per-channel-and-group scales -> out_dtype [M, N], the bias and GELU in its epilogue (int8mm.reference's sums).
    On a dp4a card, the twin of the dp4a kernel the install's race chose for the scale form (`_dp4a_twin`)."""
    twin = _dp4a_twin(xq.device, xs.dim() == 2)
    if twin is not None:
        return twin(xq, xs, w_q, w_s, bias, gelu, out_dtype, 0)
    return cuda_gemm_op()(xq, xs, w_q, w_s, bias, gelu, out_dtype, 0, 0)


def _dp4a_card(device: torch.device) -> bool:
    """Whether the shipped op runs its dp4a kernel on this card: below sm_75 (the profile's capability where one is
    forced: PIXAL3D_CARD_SM)."""
    major, minor = precision.profile().capability or torch.cuda.get_device_capability(device)
    return (major, minor) < (7, 5)


def _dp4a_twin_op(name: str) -> Optional[Callable]:
    """pixal3d_kernels' twin `name` of the dp4a kernel (int8_gemm_group_<name>), or None (an older package)."""
    if cuda_gemm_op() is None:
        return None
    try:
        return getattr(torch.ops.pixal3d_kernels, f"int8_gemm_group_{name}")
    except (AttributeError, RuntimeError):
        return None


def _dp4a_wanted(device: torch.device, form: str) -> Tuple[str, str]:
    """(the twin for the scale form "group" / "rows", why): PIXAL3D_DP4A_TWIN (developers: a twin's name for both
    forms, "group=<twin>,rows=<twin>", or shipped / off), else the install's race (the calibration record's
    old_cards.dp4a, made on this card for this profile with the installed pixal3d_kernels build), else "shipped"."""
    text = os.environ.get("PIXAL3D_DP4A_TWIN", "").strip().lower()
    if text:
        named: dict = {}
        for part in (p.strip() for p in text.split(",") if p.strip()):
            if "=" in part:
                key, value = (s.strip() for s in part.split("=", 1))
                named[key] = value
            else:
                named.setdefault("group", part)
                named.setdefault("rows", part)
        name = named.get(form, "shipped")
        return ("shipped" if name in ("off", "shipped", "") else name), f"PIXAL3D_DP4A_TWIN={text}"
    try:
        from . import old_cards

        race = old_cards.dp4a_race(device)
    except Exception:       # noqa: BLE001 - no record: the shipped kernel
        race = None
    if isinstance(race, dict) and race.get(form) in DP4A_TWINS:
        return race[form], f"timed on this card when Spellcast3D was installed: {race.get('summary', 'record')}"
    return "shipped", "no race recorded"


def _twin_table(device: torch.device, grouped: bool, M: int, K: int, N: int, gelu: bool, seed: int):
    """Random int8 operands of the flows' kind (activations per row or per row and 128-K group, weights per channel,
    a bias), from pixal3d_kernels' own quantiser (the torch quantiser's bits)."""
    import pixal3d_kernels as pk

    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, K), device=device, generator=g)
    x[:, torch.randint(0, K, (8,), device=device, generator=g)] *= 20.0
    w = torch.randn((N, K), device=device, generator=g) * 0.05
    b = torch.randn((N,), device=device, generator=g)
    xq, xs = pk.quantize_rows(x, GROUP if grouped else 0)
    wq, ws = pk.quantize_rows(w, 0)
    return xq, xs, wq, ws, b, gelu


def _dp4a_twin(device: torch.device, grouped: bool) -> Optional[Callable]:
    """The twin of the dp4a kernel for this scale form in this process (`_dp4a_wanted`), once it gave the shipped
    op's bits on a small table here (torch.equal; else one line and the shipped kernel), or None: the shipped
    kernel (a card on the tensor cores, no twin wanted, an older package). Cached."""
    key = (str(device), bool(grouped))
    if key in _dp4a:
        return _dp4a[key]
    _dp4a[key] = None
    form = "group" if grouped else "rows"
    if not _dp4a_card(device):
        return None
    name, why = _dp4a_wanted(device, form)
    if name == "shipped":
        return None
    op = _dp4a_twin_op(name)
    if op is None:
        print(f"[INT8] the dp4a GEMM's twin {name} is not in the installed pixal3d_kernels: the shipped kernel serves "
              f"the {form} layers", flush=True)
        return None
    try:
        xq, xs, wq, ws, b, gelu = _twin_table(device, grouped, 300, 1536, 4608, grouped, 3)
        same = torch.equal(op(xq, xs, wq, ws, b, gelu, precision.torso(), 0),
                           cuda_gemm_op()(xq, xs, wq, ws, b, gelu, precision.torso(), 0, 0))
        del xq, xs, wq, ws, b
    except Exception as exc:        # noqa: BLE001 - whatever the package or the card raises: the shipped kernel
        same, why = False, f"{type(exc).__name__}: {str(exc)[:200]}"
    if not same:
        print(f"[INT8] the dp4a GEMM's twin {name} did not give the shipped kernel's bits on this card ({why}): the "
              f"shipped kernel serves the {form} layers", flush=True)
        return None
    print(f"[INT8] the dp4a GEMM's twin {name} serves the {form} layers ({why})", flush=True)
    _dp4a[key] = op
    return op


DP4A_RACE_SECONDS = 600     # the child process of the dp4a twins' race (calibrate_dp4a_twins_in_child): its limit


def calibrate_dp4a_twins_in_child(device: torch.device) -> dict:
    """calibrate_dp4a_twins in a child process with this process's environment and the fork on its path, as
    sage._old_card_trial_in_child runs the attention's trial (old_cards' calibration step calls this): the twins'
    sm_61 code has run on no GTX 10xx yet, and a kernel that faults leaves its process's CUDA context broken, which
    must cost only this race - never the element-wise race after it, nor the calibration's later steps. The
    child's last line of JSON is the result; a child that ends with an error, prints no result or runs out of time
    gives {"ok": False, "group": "shipped", "rows": "shipped", "error"}: the shipped kernel serves. Where nothing is
    to be raced (the tensor cores, no package, no twins) the answer comes without a child."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    if not _dp4a_card(device):
        return {"ok": True, "skipped": "the card runs the tensor-core kernel"}
    if cuda_gemm_op() is None:
        return {"ok": True, "skipped": "pixal3d_kernels is not installed"}
    if not any(_dp4a_twin_op(t) is not None for t in DP4A_TWINS):
        return {"ok": True, "skipped": "the installed pixal3d_kernels has no twins", "group": "shipped",
                "rows": "shipped"}
    code = ("import json, torch\nfrom pixal3d.modules import int8mm\n"
            "print(json.dumps(int8mm.calibrate_dp4a_twins(torch.device('cuda'))), flush=True)")

    def text(out: object) -> str:
        return out.decode("utf-8", "replace") if isinstance(out, bytes) else str(out or "")

    try:
        run = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[2]),
                             env=os.environ.copy(), capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=DP4A_RACE_SECONDS)
        stdout, stderr, returncode = run.stdout or "", run.stderr or "", run.returncode
        ended = f"the race's process ended with code {returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, returncode = text(exc.stdout), text(exc.stderr), None
        ended = f"the race did not finish in {DP4A_RACE_SECONDS} s"
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    found = None
    if lines:
        try:
            found = json.loads(lines[-1])
        except ValueError:
            found = None
    if isinstance(found, dict) and returncode == 0:
        return found
    tail = " | ".join((stderr or stdout).strip().splitlines()[-3:])
    failed = {"ok": False, "group": "shipped", "rows": "shipped", "error": f"{ended}: {tail[:300]}",
              "summary": "the shipped kernel serves (the twins' race did not finish)"}
    if isinstance(found, dict) and found.get("timings"):
        failed["timings"] = found["timings"]            # what it measured, for the record: no twin serves on it
    return failed


def calibrate_dp4a_twins(device: torch.device) -> dict:
    """The install's race of the dp4a kernel's twins (old_cards' calibration step calls it; the record's
    old_cards.dp4a): on a dp4a card, per scale form (DP4A_RACE: the hr / tex flows' group layers qkv, fc1 + GELU,
    fc2; the per-row out), the shipped op and each twin at DP4A_RACE_ROWS' sizes on the same random operands - each
    twin's answer must be the shipped op's bits (torch.equal) - timed in a rotating order (CUDA events, medians); the
    form's time is the sum of its layers'. The twin with the best worst-size ratio serves where it is
    DP4A_TWIN_MARGIN faster at every size. Returns {"ok", "group", "rows" (a twin or "shipped"), "timings",
    "summary", "why"}, or {"ok": True, "skipped": why}. Never raises."""
    import statistics

    if not _dp4a_card(device):
        return {"ok": True, "skipped": "the card runs the tensor-core kernel"}
    if cuda_gemm_op() is None:
        return {"ok": True, "skipped": "pixal3d_kernels is not installed"}
    present = [t for t in DP4A_TWINS if _dp4a_twin_op(t) is not None]
    if not present:
        return {"ok": True, "skipped": "the installed pixal3d_kernels has no twins", "group": "shipped",
                "rows": "shipped"}
    found: dict = {"ok": True, "present": present, "group": "shipped", "rows": "shipped", "timings": []}
    try:
        shipped = cuda_gemm_op()
        out_dtype = precision.torso()
        lines = []
        for form, layers in DP4A_RACE.items():
            grouped = form == "group"
            names = ["shipped", *present]
            per_size: dict = {}
            out: set = set()
            for M, rounds in DP4A_RACE_ROWS:
                totals = dict.fromkeys(names, 0.0)
                for layer, K, N, gelu in layers:
                    xq, xs, wq, ws, b, gelu = _twin_table(device, grouped, M, K, N, gelu, M + K + N)
                    calls = {"shipped": (lambda: shipped(xq, xs, wq, ws, b, gelu, out_dtype, 0, 0))}
                    for twin in present:
                        calls[twin] = (lambda op=_dp4a_twin_op(twin): op(xq, xs, wq, ws, b, gelu, out_dtype, 0))
                    want = calls["shipped"]()
                    for twin in present:
                        if twin not in out and not torch.equal(calls[twin](), want):
                            out.add(twin)
                    del want
                    live = [n for n in names if n not in out]
                    ms: dict = {n: [] for n in live}
                    for i in range(rounds):
                        for n in live[i % len(live):] + live[:i % len(live)]:
                            calls[n]()
                            events = []
                            for _ in range(5):
                                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                start.record()
                                calls[n]()
                                end.record()
                                events.append((start, end))
                            torch.cuda.synchronize(device)
                            ms[n] += [a.elapsed_time(e) for a, e in events]
                    med = {n: statistics.median(v) for n, v in ms.items()}
                    for n in live:
                        totals[n] += med[n]
                    found["timings"].append({"form": form, "layer": layer, "M": M, "K": K, "N": N,
                                             "ms": {n: round(v, 4) for n, v in med.items()}})
                    del xq, xs, wq, ws, b, calls, ms
                    torch.cuda.empty_cache()
                per_size[M] = totals
            ratio = {t: max(per_size[M][t] / per_size[M]["shipped"] for M in per_size) for t in present if t not in out}
            best = min(ratio, key=ratio.get) if ratio else None
            text = ", ".join(f"{t} {100 * (x - 1):+.1f} %" for t, x in sorted(ratio.items(), key=lambda kv: kv[1]))
            if out:
                text += f"; other bits: {', '.join(sorted(out))} (left out)"
            if best is not None and ratio[best] <= 1 - DP4A_TWIN_MARGIN:
                found[form] = best
            lines.append(f"{form} layers: {found[form]} (the worst size: {text or 'no twin timed'})")
        found["summary"] = "; ".join(lines)
        found["why"] = (f"a twin serves where it is {100 * DP4A_TWIN_MARGIN:.0f} % faster than the shipped kernel at "
                        f"every size (M {', '.join(str(m) for m, _ in DP4A_RACE_ROWS)})")
    except Exception as exc:        # noqa: BLE001 - the twins never cost the shipped kernel its place
        found.update(ok=False, group="shipped", rows="shipped",
                     error=f"{type(exc).__name__}: {(str(exc).strip().splitlines() or [''])[0][:200]}")
    return found


def kernel_name() -> str:
    gemm_how = {"triton": "Triton int8 GEMMs, the epilogue fused",
                "cuda": "pixal3d_kernels' int8 GEMM, the epilogue fused",
                "intmm": "torch._int_mm (cuBLASLt)"}.get(str(_state["gemm"]), "int8 GEMMs")
    if _state["gemm"] in ("intmm", "cuda"):
        what = "quantiser" if _state["gemm"] == "cuda" else "quantiser and epilogue"
        gemm_how += (f", the {what} in "
                     f"{ {'triton': 'Triton', 'cuda': 'pixal3d_kernels'}.get(_state['elementwise'], 'torch')}")
    return f"int8mm: {gemm_how}"


# =====================================================================================================================
# Kernels
# =====================================================================================================================

def _build():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def round_bf16(x):
        # as modules/fused.py: x rounded to the nearest bf16 (ties to even) by integer arithmetic on the bits
        bits = x.to(tl.uint32, bitcast=True)
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
        return tl.where(x != x, x, bits.to(tl.float32, bitcast=True))

    @triton.jit
    def quant_row(y, Q, S, row, cols, mask, stride_q):
        # one fp32 row (masked lanes 0) -> int8 + its scale; torch.round(y * (127 / amax)) bit for bit
        amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
        inv = tl.div_rn(127.0, amax)
        q = tl.minimum(tl.maximum(libdevice.rint(y * inv), -127.0), 127.0)
        tl.store(Q + row * stride_q + cols, q.to(tl.int8), mask=mask)
        tl.store(S + row, tl.div_rn(amax, 127.0))

    @triton.jit(do_not_specialize=["M", "stride_s"])
    def quant_rows_kernel(X, Q, S, SM, M, K, stride_x, stride_q, stride_s,
                          GROUP: tl.constexpr, NG: tl.constexpr, BLOCK_K: tl.constexpr, HAS_SMOOTH: tl.constexpr):
        # HAS_SMOOTH: x * SM[K], SM = 1 / s (SmoothQuant's per-input-channel divisor s; the weight holds W * s)
        row = tl.program_id(0).to(tl.int64)
        if GROUP == 0:
            cols = tl.arange(0, BLOCK_K)
            mask = cols < K
            x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_SMOOTH:
                x = x * tl.load(SM + cols, mask=mask, other=1.0).to(tl.float32)
            quant_row(x, Q, S, row, cols, mask, stride_q)
        else:
            g = tl.arange(0, NG)
            c = tl.arange(0, GROUP)
            cols = g[:, None] * GROUP + c[None, :]
            gmask = g < K // GROUP
            x = tl.load(X + row * stride_x + cols, mask=gmask[:, None], other=0.0).to(tl.float32)
            if HAS_SMOOTH:
                x = x * tl.load(SM + cols, mask=gmask[:, None], other=1.0).to(tl.float32)
            amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-30)
            inv = tl.div_rn(tl.full([NG], 127.0, tl.float32), amax)
            q = tl.minimum(tl.maximum(libdevice.rint(x * inv[:, None]), -127.0), 127.0)
            tl.store(Q + row * stride_q + cols, q.to(tl.int8), mask=gmask[:, None])
            tl.store(S + row * stride_s + g, tl.div_rn(amax, tl.full([NG], 127.0, tl.float32)), mask=gmask)

    @triton.jit(do_not_specialize=["n_rows", "rows_per_batch", "stride_s"])
    def norm_quant_kernel(X, Q, S, W, B, SCALE, SHIFT, BIDX, SM, n_rows, rows_per_batch, C, stride_x, stride_q,
                          stride_s, stride_mod, eps, AFFINE: tl.constexpr, MODULATE: tl.constexpr,
                          HAS_BIDX: tl.constexpr, ROUNDED: tl.constexpr, GROUP: tl.constexpr, NG: tl.constexpr,
                          BLOCK_C: tl.constexpr, HAS_SMOOTH: tl.constexpr):
        # modules/fused.py norm_modulate_kernel's arithmetic, the result rounded to bf16 (the value the bf16 path
        # hands to the linear), then quantised: equal to quantize_rows(norm_modulate(x)) bit for bit
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / C
        rstd = tl.math.rsqrt(var + eps)
        y = xc * rstd
        if AFFINE:
            w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
            y = w * y + b
        if MODULATE:
            if HAS_BIDX:
                bi = tl.load(BIDX + row).to(tl.int64)
            else:
                bi = row // rows_per_batch
            scale = tl.load(SCALE + bi * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(SHIFT + bi * stride_mod + cols, mask=mask, other=0.0).to(tl.float32)
            if ROUNDED:
                y = round_bf16(y)
                s1 = round_bf16(scale + 1.0)
                y = round_bf16(y * s1)
                y = y + shift
            else:
                y = y * (scale + 1.0) + shift
        y = tl.where(mask, round_bf16(y), 0.0)          # the bf16 value the eager / fused path would store
        if HAS_SMOOTH:
            y = y * tl.load(SM + cols, mask=mask, other=1.0).to(tl.float32)
        if GROUP == 0:
            quant_row(y, Q, S, row, cols, mask, stride_q)
        else:
            # [BLOCK_C] -> [NG, GROUP]: one scale per GROUP channels
            y2 = tl.reshape(y, (NG, GROUP))
            g = tl.arange(0, NG)
            c2 = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
            gmask = g < C // GROUP
            amax = tl.maximum(tl.max(tl.abs(y2), axis=1), 1e-30)
            inv = tl.div_rn(tl.full([NG], 127.0, tl.float32), amax)
            q = tl.minimum(tl.maximum(libdevice.rint(y2 * inv[:, None]), -127.0), 127.0)
            tl.store(Q + row * stride_q + c2, q.to(tl.int8), mask=gmask[:, None])
            tl.store(S + row * stride_s + g, tl.div_rn(amax, tl.full([NG], 127.0, tl.float32)), mask=gmask)

    @triton.jit(do_not_specialize=["M", "stride_asm", "stride_qsm"])
    def gemm_kernel(A, B, C, AS, BS, BIAS, QS, SMO, RX, RG, R2, M, N, K, stride_am, stride_bn, stride_cm, stride_asm,
                    stride_qsm, stride_bsn, HAS_BIAS: tl.constexpr, GELU: tl.constexpr, OUT: tl.constexpr,
                    A_GROUP: tl.constexpr, B_GROUP: tl.constexpr, OUT_SMOOTH: tl.constexpr, MAGIC: tl.constexpr,
                    RESID: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                    GROUP_M: tl.constexpr):
        # RESID (OUT 0): the block's residual kernel (fused.residual, rounded mode) in the epilogue - the result h,
        # rounded to bf16 as it would be stored, goes into x + h (1: x + round(h * RG[col]), the gate one row; 2: x +
        # round(R2 + h); 3: x + h), written to C instead of h; RX and R2 [M, N] with C's row stride
        # A: int8 [M, K] row-major; B: int8 [N, K] row-major (the weight as stored: both operands K-contiguous,
        # the "TN" layout int8 mma wants). OUT: 0 = bf16, 1 = the raw int32 accumulator (checks), 2 = int8 with one
        # fp32 scale per row and per BLOCK_N columns (QS [M, N / BLOCK_N]: the next layer's group-scaled input),
        # 3 = fp16 (Hi3D's transformer keeps its activations in fp16).
        # A_GROUP: 0 = AS holds one scale per row; else AS [M, K / BLOCK_K] and the int32 partial product of each
        # BLOCK_K slab is scaled into an fp32 accumulator (MAGIC: int32 -> fp32 by the 1.5 * 2^23 trick, exact for
        # |partial| < 2^22, i.e. BLOCK_K <= 256). B_GROUP (needs A_GROUP): BS [N, K / BLOCK_K] column-major (a
        # slab's BLOCK_N scales contiguous: stride 1 along N, N along K) holds one weight scale per output channel
        # and BLOCK_K slab, multiplied into the slab's partial product beside the row's scale (otherwise BS [N]
        # scales the columns in the epilogue). OUT_SMOOTH (OUT 2): the result times SMO [N] before it is quantised.
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        a_ptrs = A + ram[:, None].to(tl.int64) * stride_am + rk[None, :]
        b_ptrs = B + rbn[None, :].to(tl.int64) * stride_bn + rk[:, None]
        if A_GROUP:
            accf = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            s_ptrs = AS + ram.to(tl.int64) * stride_asm
            if B_GROUP:
                w_ptrs = BS + rbn.to(tl.int64) * stride_bsn
            for kb in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                part = tl.dot(a, b, out_dtype=tl.int32)
                s = tl.load(s_ptrs + kb)
                if MAGIC:
                    pf = (part + 0x4B400000).to(tl.float32, bitcast=True) - 12582912.0
                else:
                    pf = part.to(tl.float32)
                if B_GROUP:
                    ws = tl.load(w_ptrs + kb * N)
                    accf += pf * (s[:, None] * ws[None, :])
                else:
                    accf += pf * s[:, None]
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K
            y = accf
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kb in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K
            rs = tl.load(AS + ram)
            y = acc.to(tl.float32) * rs[:, None]
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        c_ptrs = C + rm[:, None].to(tl.int64) * stride_cm + rn[None, :]
        if OUT == 1:
            tl.store(c_ptrs, acc, mask=mask)
        else:
            if not B_GROUP:
                cs = tl.load(BS + rbn)
                y = y * cs[None, :]
            if HAS_BIAS:
                y += tl.load(BIAS + rbn).to(tl.float32)[None, :]
            if GELU:
                inner = 0.7978845608028654 * (y + 0.044715 * (y * y * y))
                y = 0.5 * y * (1.0 + libdevice.tanh(inner))
            if OUT_SMOOTH:
                y = y * tl.load(SMO + rbn)[None, :]
            if OUT == 2:
                amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-30)
                inv = tl.div_rn(tl.full([BLOCK_M], 127.0, tl.float32), amax)
                q = tl.minimum(tl.maximum(libdevice.rint(y * inv[:, None]), -127.0), 127.0)
                tl.store(c_ptrs, q.to(tl.int8), mask=mask)
                tl.store(QS + rm.to(tl.int64) * stride_qsm + pid_n,
                         tl.div_rn(amax, tl.full([BLOCK_M], 127.0, tl.float32)), mask=rm < M)
            elif OUT == 3:
                tl.store(c_ptrs, y.to(tl.float16), mask=mask)
            elif RESID:
                h = round_bf16(y)
                r_off = rm[:, None].to(tl.int64) * stride_cm + rn[None, :]
                if RESID == 1:
                    h = round_bf16(h * tl.load(RG + rbn).to(tl.float32)[None, :])
                elif RESID == 2:
                    h = round_bf16(tl.load(R2 + r_off, mask=mask, other=0.0).to(tl.float32) + h)
                xr = tl.load(RX + r_off, mask=mask, other=0.0).to(tl.float32)
                tl.store(c_ptrs, (xr + h).to(tl.bfloat16), mask=mask)
            else:
                tl.store(c_ptrs, y.to(tl.bfloat16), mask=mask)

    @triton.jit(do_not_specialize=["M"])
    def epilogue_kernel(C32, Y, AS, BS, BIAS, QS, M, N, stride_c, stride_y, HAS_BIAS: tl.constexpr,
                        GELU: tl.constexpr, OUT: tl.constexpr, BLOCK_N: tl.constexpr):
        # torch._int_mm's int32 [M, N] -> * row_scale * col_scale + bias [-> GELU] -> bf16 (OUT 0) or int8 with one
        # scale per row (OUT 2: the input of a following torch._int_mm). One row per program.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        acc = tl.load(C32 + row * stride_c + cols, mask=mask, other=0)
        rs = tl.load(AS + row)
        cs = tl.load(BS + cols, mask=mask, other=0.0)
        y = (acc.to(tl.float32) * rs) * cs
        if HAS_BIAS:
            y += tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
        if GELU:
            inner = 0.7978845608028654 * (y + 0.044715 * (y * y * y))
            y = 0.5 * y * (1.0 + libdevice.tanh(inner))
        if OUT == 2:
            quant_row(y, Y, QS, row, cols, mask, stride_y)
        else:
            tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)

    _kernels = {"quant": quant_rows_kernel, "norm_quant": norm_quant_kernel, "gemm": gemm_kernel,
                "epilogue": epilogue_kernel, "next_pow2": triton.next_power_of_2, "cdiv": triton.cdiv}
    return _kernels


def _p2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def _warps(width: int) -> int:
    return 4 if width <= 2048 else 8


# =====================================================================================================================
# Quantisers
# =====================================================================================================================

def quantize_rows(x: torch.Tensor, group: int = 0,
                  inv_smooth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 (or fp16 / fp32) [..., K] -> (int8 [M, K], fp32 scales [M] or [M, K // group]); symmetric,
    q = round_half_even(x * (127 / absmax)), scale = absmax / 127 (absmax floored at TINY). inv_smooth: 1 / s
    (fp32 [K], computed once at load) for SmoothQuant's per-input-channel vector s, applied as x * (1 / s) in the
    same pass (the weight was quantised as W * s: quantize_weight(w, smooth=s)). The Triton kernel where ready()
    chose it, else the torch ops (the same bits)."""
    if _state["elementwise"] == "cuda":           # below sm_80: pixal3d_kernels' quantiser (old_cards.py)
        from . import old_cards

        return old_cards.quantize_rows(x, group, inv_smooth)
    if _state["elementwise"] != "triton":
        return quantize_rows_torch(x, group, inv_smooth)
    return quantize_rows_triton(x, group, inv_smooth)


def quantize_rows_triton(x: torch.Tensor, group: int = 0,
                         inv_smooth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    k = _build()
    K = x.shape[-1]
    flat = x.reshape(-1, K)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    M = flat.shape[0]
    q = torch.empty((M, K), dtype=torch.int8, device=x.device)
    if group:
        if K % group or group & (group - 1):
            raise ValueError(f"group {group} must be a power of two dividing K = {K}")
        s = torch.empty((M, K // group), dtype=torch.float32, device=x.device)
        ng, block, stride_s = _p2(K // group), 1, s.stride(0)
    else:
        s = torch.empty((M,), dtype=torch.float32, device=x.device)
        ng, block, stride_s = 1, _p2(K), 1
    if M:
        k["quant"][(M,)](flat, q, s, inv_smooth if inv_smooth is not None else flat, M, K, flat.stride(0),
                         q.stride(0), stride_s, GROUP=group, NG=ng, BLOCK_K=block, HAS_SMOOTH=inv_smooth is not None,
                         num_warps=_warps(K))
    return q, s


def quantize_weight(w: torch.Tensor, smooth: Optional[torch.Tensor] = None,
                    group: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """nn.Linear weight [N, K] -> (int8 [N, K] row-major, fp32 per-output-channel scales [N], or [N, K // group]
    with `group`: one per output channel and group of K, the "group" GEMM's weight scales, column-major). The same row
    quantiser: a row of the weight is one output channel. One-off, at load. smooth: s [K]; W * s is quantised
    (the activations then pass inv_smooth = 1 / s to quantize_rows / norm_quant / int8_linear)."""
    w = w.detach()
    if smooth is not None:
        w = w.float() * smooth.float()[None, :]
    q, s = quantize_rows(w, group)
    if group:
        s = s.t().contiguous().t()          # column-major: the GEMM reads a 128-K slab's scales contiguously
    return q, s


def quantize_rows_torch(x: torch.Tensor, group: int = 0,
                        inv_smooth: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The reference quantiser in plain torch ops (the Triton kernels must equal it bit for bit). Note: torch's
    `127.0 / t` is t.reciprocal() * 127 (two roundings); the kernels divide once (IEEE), as torch.div of two
    tensors does."""
    K = x.shape[-1]
    xf = x.reshape(-1, K).float()
    if inv_smooth is not None:
        xf = xf * inv_smooth.float()
    if group:
        xf = xf.view(xf.shape[0], K // group, group)
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(TINY)
    q = torch.round(xf * torch.div(torch.full_like(amax, 127.0), amax)).clamp_(-127, 127).to(torch.int8)
    s = torch.div(amax, torch.full_like(amax, 127.0)).squeeze(-1)   # IEEE (t / 127.0 is t * (1 / 127))
    return q.reshape(-1, K), s


def norm_quant(x: torch.Tensor, eps: float, *, weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None, scale: Optional[torch.Tensor] = None,
               shift: Optional[torch.Tensor] = None, batch: int = 1, rows_per_batch: Optional[int] = None,
               batch_index: Optional[torch.Tensor] = None, group: int = 0, rounded: bool = True,
               inv_smooth: Optional[torch.Tensor] = None, num_warps: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """fused.norm_modulate(x, ...) followed by quantize_rows, in one kernel: (int8 [M, C], scales)."""
    k = _build()
    C = x.shape[-1]
    flat = x.reshape(-1, C)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    n_rows = flat.shape[0]

    def mod_rows(mod):
        if mod is None:
            return None, 0
        if mod.dim() != 2:
            mod = mod.reshape(mod.shape[0], -1)
        return mod, (0 if mod.shape[0] == 1 else mod.stride(0))

    scale, stride_mod = mod_rows(scale)
    shift, _ = mod_rows(shift)
    q = torch.empty((n_rows, C), dtype=torch.int8, device=x.device)
    if group:
        if C % group:
            raise ValueError(f"group {group} must divide C = {C}")
        s = torch.empty((n_rows, C // group), dtype=torch.float32, device=x.device)
        block = max(_p2(C), group)
        ng, stride_s = block // group, s.stride(0)
    else:
        s = torch.empty((n_rows,), dtype=torch.float32, device=x.device)
        block, ng, stride_s = _p2(C), 1, 1
    if n_rows:
        k["norm_quant"][(n_rows,)](
            flat, q, s, weight if weight is not None else flat, bias if bias is not None else flat,
            scale if scale is not None else flat, shift if shift is not None else flat,
            batch_index if batch_index is not None else flat, inv_smooth if inv_smooth is not None else flat, n_rows,
            rows_per_batch or n_rows, C, flat.stride(0), q.stride(0), stride_s, stride_mod, float(eps),
            AFFINE=weight is not None, MODULATE=scale is not None, HAS_BIDX=batch_index is not None,
            ROUNDED=rounded, GROUP=group, NG=ng, BLOCK_C=block, HAS_SMOOTH=inv_smooth is not None,
            num_warps=num_warps or NORM_WARPS)
    return q, s


# =====================================================================================================================
# GEMMs
# =====================================================================================================================

# group scales on both operands (weight scales per output channel and group, round J's policy "cs"): one more
# multiply per element and 128-K slab; 64 x 128 tiles of 4 warps keep it 6-10 % over the group GEMM at K <= 2048
# (CFG_GROUP: 10-14 %); fc2 (K = 8192) is fastest at CFG_GROUP (round-j/int8cs gg_sweep.json; the same bits)
CFG_GROUP_W = (64, 128, 128, 8, 3, 4)


def pick_config(M: int, N: int, K: int, group: int = 0, w_group: bool = False) -> tuple:
    if group:
        return CFG_GROUP_W if w_group and K <= 2048 else CFG_GROUP
    return CFG_LONGK if K >= 4096 else CFG_BIG


def gemm(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor],
         *, gelu: bool = False, out: str = "bf16", group: int = 0, cfg: Optional[tuple] = None,
         magic: bool = True, out_smooth: Optional[torch.Tensor] = None, resid: Optional[dict] = None,
         kernel: Optional[str] = None):
    """The Triton int8 GEMM with the epilogue fused. out: "bf16" -> bf16 [M, N]; "fp16" -> fp16 [M, N]; "int32" ->
    the raw accumulator (per-row mode only); "int8" -> (int8 [M, N], fp32 scales [M, N // BLOCK_N]). w_s: fp32 [N]
    (one scale per output channel) or [N, K // group] (one per output channel and group of K: needs group scales on x too).
    out_smooth: fp32 [N] that multiplies the result before an "int8" output is quantised (the next layer's 1 / s).
    resid (bf16 out): {"x", "gate" or "second"} - the block's residual (fused.residual, rounded mode, a gate of one
    row) done in the epilogue: returns x + round(h * gate) / x + round(second + h) / x + h, h the bf16 result.
    kernel: None = the group layers on pixal3d_kernels' CUDA kernel where the install's race chose it for the layer
    (group_cuda_tile: the same bits; cfg is then the Triton kernel's, unused), "triton" = the Triton kernel always."""
    if group and kernel != "triton" and w_s.dim() == 1 and out_smooth is None:
        tile = group_cuda_tile(xq, w_q, gelu, out, resid)
        if tile is not None:
            return group_cuda(xq, xs, w_q, w_s, bias, gelu=gelu, out=out, resid=resid, tile=tile)
    k = _build()
    M, K = xq.shape
    N = w_q.shape[0]
    BM, BN, BK, GM, stages, warps = cfg or pick_config(M, N, K, group, w_s.dim() == 2)
    if K % BK:
        raise ValueError(f"K = {K} is not a multiple of BLOCK_K = {BK}")
    if group and group != BK:
        raise ValueError(f"the group path needs BLOCK_K == group ({BK} != {group})")
    b_group = w_s.dim() == 2
    if b_group and (not group or w_s.shape[1] * group != K):
        raise ValueError(f"weight scales {tuple(w_s.shape)} per group need group scales on x (group {group}, K {K})")
    if b_group and (w_s.stride(0) != 1 or w_s.stride(1) != N):
        w_s = w_s.t().contiguous().t()      # the kernel reads them column-major (quantize_weight gives them so)
    mode = {"bf16": 0, "int32": 1, "int8": 2, "fp16": 3}[out]
    if mode == 1 and b_group:
        raise ValueError("the raw int32 output is for per-row scales only")
    dtype = {0: torch.bfloat16, 1: torch.int32, 2: torch.int8, 3: torch.float16}[mode]
    y = torch.empty((M, N), dtype=dtype, device=xq.device)
    rx = rg = r2 = None
    how = 0
    if resid is not None:
        if mode != 0:
            raise ValueError("the residual epilogue writes bf16 only")
        rx, rg, r2 = resid["x"], resid.get("gate"), resid.get("second")
        how = 1 if rg is not None else 2 if r2 is not None else 3
        if rg is not None:
            rg = rg.reshape(-1)
        if (rx.shape != (M, N) or rx.stride() != y.stride() or (r2 is not None and r2.stride() != y.stride())
                or (rg is not None and (rg.numel() != N or rg.stride(0) != 1))):
            raise ValueError("the residual epilogue takes x (and second) of the output's shape and strides, a gate "
                             "of one row")
    qs = None
    if mode == 2:
        if N % BN:
            raise ValueError(f"int8 output needs N % BLOCK_N == 0 ({N} % {BN})")
        qs = torch.empty((M, N // BN), dtype=torch.float32, device=xq.device)
    if M:
        grid = (k["cdiv"](M, BM) * k["cdiv"](N, BN),)
        k["gemm"][grid](xq, w_q, y, xs, w_s, bias if bias is not None else w_s, qs if qs is not None else w_s,
                        out_smooth if out_smooth is not None else w_s, rx if rx is not None else w_s,
                        rg if rg is not None else w_s, r2 if r2 is not None else w_s,
                        M, N, K, xq.stride(0), w_q.stride(0), y.stride(0), xs.stride(0) if group else 1,
                        qs.stride(0) if qs is not None else 1, w_s.stride(0) if b_group else 1,
                        HAS_BIAS=bias is not None, GELU=gelu, OUT=mode, A_GROUP=group, B_GROUP=b_group,
                        OUT_SMOOTH=out_smooth is not None, MAGIC=magic, RESID=how, BLOCK_M=BM, BLOCK_N=BN,
                        BLOCK_K=BK, GROUP_M=GM, num_stages=stages, num_warps=warps)
    return (y, qs) if mode == 2 else y


def epilogue(c32: torch.Tensor, xs: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor], *,
             gelu: bool = False, out_int8: bool = False, y: Optional[torch.Tensor] = None,
             ys: Optional[torch.Tensor] = None, out_dtype: torch.dtype = torch.bfloat16):
    """torch._int_mm's int32 [M, N] -> out_dtype [M, N] (or int8 [M, N] + per-row scales [M]); y / ys: optional
    outputs to write. The Triton kernel where ready() chose it, else the torch ops."""
    if _state["elementwise"] != "triton":
        return epilogue_torch(c32, xs, w_s, bias, gelu=gelu, out_int8=out_int8, y=y, ys=ys, out_dtype=out_dtype)
    return epilogue_triton(c32, xs, w_s, bias, gelu=gelu, out_int8=out_int8, y=y, ys=ys, out_dtype=out_dtype)


def epilogue_torch(c32: torch.Tensor, xs: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor], *,
                   gelu: bool = False, out_int8: bool = False, y: Optional[torch.Tensor] = None,
                   ys: Optional[torch.Tensor] = None, out_dtype: torch.dtype = torch.bfloat16):
    """The epilogue in torch ops: the kernel's arithmetic in fp32 in the same order (the scale products, the bias,
    GELU(tanh)), then the rounding to out_dtype or the row quantiser."""
    yf = c32.float()
    yf.mul_(xs[:, None]).mul_(w_s[None, :])
    if bias is not None:
        yf.add_(bias[None, :])
    if gelu:
        yf = F.gelu(yf, approximate="tanh")
    if out_int8:
        q, s = quantize_rows_torch(yf)
        if y is not None:
            y.copy_(q)
            q = y
        if ys is not None:
            ys.copy_(s)
            s = ys
        return q, s
    if y is not None:
        y.copy_(yf)
        return y
    return yf.to(out_dtype)


def epilogue_triton(c32: torch.Tensor, xs: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor], *,
                    gelu: bool = False, out_int8: bool = False, y: Optional[torch.Tensor] = None,
                    ys: Optional[torch.Tensor] = None, out_dtype: torch.dtype = torch.bfloat16):
    k = _build()
    M, N = c32.shape
    if y is None:
        y = torch.empty((M, N), dtype=torch.int8 if out_int8 else out_dtype, device=c32.device)
    if out_int8 and ys is None:
        ys = torch.empty((M,), dtype=torch.float32, device=c32.device)
    if M:
        k["epilogue"][(M,)](c32, y, xs, w_s, bias if bias is not None else w_s, ys if out_int8 else w_s, M, N,
                            c32.stride(0), y.stride(0), HAS_BIAS=bias is not None, GELU=gelu,
                            OUT=2 if out_int8 else 0, BLOCK_N=_p2(N), num_warps=_warps(N))
    return (y, ys) if out_int8 else y


def intmm_linear(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                 bias: Optional[torch.Tensor], *, gelu: bool = False, out_int8: bool = False,
                 chunk: Optional[int] = CHUNK_ROWS, out_dtype: torch.dtype = torch.bfloat16):
    """torch._int_mm (rows in chunks: the int32 temporary is chunk x N x 4 B) + the epilogue per chunk."""
    M, K = xq.shape
    N = w_q.shape[0]
    if M <= 16:                                     # torch._int_mm wants more than 16 rows
        pad = torch.zeros((17, K), dtype=xq.dtype, device=xq.device)
        pad[:M] = xq
        ps = torch.zeros((17,), dtype=xs.dtype, device=xs.device)
        ps[:M] = xs
        r = intmm_linear(pad, ps, w_q, w_s, bias, gelu=gelu, out_int8=out_int8, chunk=None, out_dtype=out_dtype)
        return (r[0][:M], r[1][:M]) if out_int8 else r[:M]
    wt = w_q.t()                                    # [K, N] column-major: the layout cuBLASLt's IMMA kernels take
    y = torch.empty((M, N), dtype=torch.int8 if out_int8 else out_dtype, device=xq.device)
    ys = torch.empty((M,), dtype=torch.float32, device=xq.device) if out_int8 else None
    step = M if not chunk or chunk >= M else chunk
    buf = torch.empty((step, N), dtype=torch.int32, device=xq.device)
    for r0 in range(0, M, step):
        r1 = min(M, r0 + step)
        if r1 - r0 <= 16:                           # a last chunk of <= 16 rows: redo the 17 last rows instead
            r0 = r1 - 17
        c = buf[: r1 - r0]
        torch._int_mm(xq[r0:r1], wt, out=c)
        epilogue(c, xs[r0:r1], w_s, bias, gelu=gelu, out_int8=out_int8, y=y[r0:r1],
                 ys=ys[r0:r1] if ys is not None else None, out_dtype=out_dtype)
    return (y, ys) if out_int8 else y


def dequantize_rows(xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """(int8 [M, K], scales [M] or [M, K / group]) -> fp32 [M, K]."""
    M, K = xq.shape
    if xs.dim() == 2:
        return (xq.float().view(M, xs.shape[1], -1) * xs.unsqueeze(-1)).view(M, K)
    return xq.float() * xs[:, None]


def _fallback(x2: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor, bias: Optional[torch.Tensor],
              gelu: bool, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    w = dequantize_rows(w_q, w_s).to(x2.dtype if x2.dtype.is_floating_point else out_dtype)
    y = F.linear(x2.to(w.dtype), w, bias.to(w.dtype) if bias is not None else None)
    return F.gelu(y, approximate="tanh") if gelu else y


def int8_linear(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], w_q: torch.Tensor, w_s: torch.Tensor,
                bias: Optional[torch.Tensor] = None, *, gelu: bool = False, path: Optional[str] = None,
                out_int8: bool = False, inv_smooth: Optional[torch.Tensor] = None,
                out_dtype: torch.dtype = torch.bfloat16, resid: Optional[dict] = None):
    """y = x @ W^T + b [GELU tanh] with W = w_q * w_s[:, None]; x: [..., K] in the torso's dtype or a pre-quantised
    (x_q [M, K], x_s) pair (x_s [M]: per row; x_s [M, K / 128]: group scales, the "group" path). Returns out_dtype
    [..., N], or (int8 [M, N], scales) with out_int8 (path "triton"/"group": [M, N / 128] group scales; "intmm": [M]
    per row; "cuda": the input's grain). w_s [N, K / 128] (weight scales per output channel and group: quantize_weight(group=GROUP)) takes the
    "group" path. inv_smooth: 1 / s [K] applied to x as it is quantised (the weight holds W * s). Where the card has
    no group GEMM (has_group_gemm), group scales take the fallback. resid: the block's residual in the Triton GEMM's
    epilogue (gemm), bf16 out: the result is then the new stream."""
    pre = isinstance(x, tuple)
    K = w_q.shape[1]
    N = w_q.shape[0]
    lead = None if pre else x.shape[:-1]
    if w_s.dim() == 2:
        path = "group"
    on = enabled(x[0] if pre else x)
    if on and not has_group_gemm():
        if path == "group" or (pre and x[1].dim() == 2):
            on = False
        else:
            path = "intmm"
    if not on:
        if pre:
            x2 = dequantize_rows(*x).to(out_dtype)
        else:
            x2 = x.reshape(-1, K)
            if inv_smooth is not None:
                x2 = (x2.float() * inv_smooth.float()).to(out_dtype)
        y = _fallback(x2, w_q, w_s, bias, gelu, out_dtype)
        if out_int8:
            return quantize_rows_torch(y)
        return y if pre else y.view(*lead, N)
    path = path or PATHS.get((N, K), DEFAULT_PATH)
    if pre:
        xq, xs = x
        group = (K // xs.shape[1]) if xs.dim() == 2 else 0
        if group and path != "group":
            path = "group"
        if not group and path == "group":
            path = "triton"
    else:
        group = GROUP if path == "group" else 0
        xq, xs = quantize_rows(x, group, inv_smooth)
    if path == "triton" and out_dtype not in _OUT_NAME:       # the Triton GEMM writes bf16 / fp16 (or int8)
        path = "intmm"
    if _state["gemm"] == "cuda":                                 # pixal3d_kernels, per-row or group scales alike
        path = "cuda"
    if resid is not None and (path in ("intmm", "cuda") or out_int8):
        raise ValueError("the residual epilogue is the Triton GEMM's, bf16 out")
    if path == "intmm":
        y = intmm_linear(xq, xs, w_q, w_s, bias, gelu=gelu, out_int8=out_int8, out_dtype=out_dtype)
    elif path == "cuda":
        y = cuda_linear(xq, xs, w_q, w_s, bias, gelu=gelu, out_dtype=out_dtype)
        if out_int8:
            y = quantize_rows(y, group)
    else:
        y = gemm(xq, xs, w_q, w_s, bias, gelu=gelu, out="int8" if out_int8 else _OUT_NAME[out_dtype], group=group,
                 resid=resid)
    if out_int8 or pre:
        return y
    return y.view(*lead, N)


_OUT_NAME = {torch.bfloat16: "bf16", torch.float16: "fp16"}     # gemm()'s `out` for an output dtype

# The old cards' MLP in row slabs (Spellcast3D round O): pixal3d_kernels' fc1 writes the hidden layer [M, 8192] in
# the torso's dtype - fp32 on the pascal profile, 32 KB a token, plus its int8 re-quantisation, 8 KB more - which at
# a 1536 prop's 43.5k tokens is 1.7 GB of one MLP's temporaries. Where that table would pass MLP_SLAB_MIB, fc1 ->
# quantise -> fc2 run on row slabs of at most that much hidden layer, fc2's rows written into one output. The same
# bits by construction: the dp4a and tensor-core kernels have fixed tiles and no split-K (csrc/int8_gemm_dp4a.cu,
# int8_gemm_imma.cu: an output's integer sums and its fp32 group chain read its own row and channel only, rows past
# M are not stored), the quantisers are per row; proven with torch.equal on random and captured real operands under
# both profiles (code/logs/profiling/pixal3d-lowmem/round-o/reports/finish-up-REPORT.md). Only the old cards'
# profiles slab (the ampere path is Triton's, untouched); PIXAL3D_MLP_SLAB_MIB = n sets the budget (checks; 0: never).
MLP_SLAB_MIB = 256
MLP_SLAB_ALIGN = 256        # slab rows a multiple of this (every operand's slice stays 16-byte aligned)
MLP_SLAB_ENV = "PIXAL3D_MLP_SLAB_MIB"


def mlp_slab_rows(M: int, hidden: int, dtype: torch.dtype) -> int:
    """The rows of the old cards' MLP slabs for M tokens and a hidden layer of `hidden` channels in `dtype`: M (one
    call, as before) on the ampere profile, where the hidden layer fits MLP_SLAB_MIB, or with the budget 0; else the
    budget's rows, a multiple of MLP_SLAB_ALIGN."""
    if precision.profile().name == "ampere":
        return M
    budget = MLP_SLAB_MIB
    text = os.environ.get(MLP_SLAB_ENV, "").strip()
    if text:
        try:
            budget = float(text)
        except ValueError:
            pass
    if budget <= 0:
        return M
    per_row = hidden * torch.empty((), dtype=dtype).element_size()
    rows = max(MLP_SLAB_ALIGN, int(budget * (1 << 20)) // per_row // MLP_SLAB_ALIGN * MLP_SLAB_ALIGN)
    return M if rows >= M else rows


def cuda_mlp(xq: torch.Tensor, xs: torch.Tensor, w1: tuple, w2: tuple, group: int,
             inv_smooth2: Optional[torch.Tensor], out_dtype: torch.dtype, step: int) -> torch.Tensor:
    """pixal3d_kernels' MLP (int8_mlp's "cuda" path): fc1 + GELU -> quantise (per row, or per row and `group`) ->
    fc2, on row slabs of `step` rows (step >= M: one call each, as before), fc2's rows written into one output."""
    M = xq.shape[0]
    if step >= M:
        h = cuda_linear(xq, xs, *w1, gelu=True, out_dtype=out_dtype)
        return cuda_linear(*quantize_rows(h, group, inv_smooth2), *w2, out_dtype=out_dtype)
    y = torch.empty((M, w2[0].shape[0]), dtype=out_dtype, device=xq.device)
    for r0 in range(0, M, step):
        r1 = min(M, r0 + step)
        h = cuda_linear(xq[r0:r1], xs[r0:r1], *w1, gelu=True, out_dtype=out_dtype)
        hq, hs = quantize_rows(h, group, inv_smooth2)
        del h
        y[r0:r1] = cuda_linear(hq, hs, *w2, out_dtype=out_dtype)
        del hq, hs
    return y


def int8_mlp(x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], w1: tuple, w2: tuple, *,
             path: str = "rows", cfg1: Optional[tuple] = None,
             inv_smooth2: Optional[torch.Tensor] = None, out_dtype: torch.dtype = torch.bfloat16,
             resid: Optional[dict] = None) -> torch.Tensor:
    """fc2(GELU(fc1(x))): w1, w2 = (w_q, w_s, bias).
    "rows" (fastest measured: 15.5 ms at M = 21504 vs 41.4 bf16): fc1 Triton with GELU -> bf16, quantize_rows,
           fc2 Triton (per-row scales everywhere, like the other layers);
    "group": fc1's Triton epilogue quantises each 128-column tile per row (int8 + group scales, the [M, 8192]
           hidden layer never in bf16; fc2 runs the group GEMM): 15.7 ms, 237 MiB peak vs 672 MiB bf16;
           weight scales per group ([N, K / 128]) on either layer take the group-weight GEMM; inv_smooth2: fc2's
           SmoothQuant 1 / s [8192], applied in fc1's epilogue before the hidden layer is quantised;
    "intmm": torch._int_mm, fc1's epilogue quantises each full row for fc2's torch._int_mm: 16.6 ms."""
    if inv_smooth2 is not None and path != "group":
        raise ValueError("fc2's smoothing is applied in the group path's fc1 epilogue only")
    if resid is not None and (_state["gemm"] != "triton" or path not in ("rows", "group")
                              or out_dtype not in _OUT_NAME):
        raise ValueError("the residual epilogue is the Triton GEMM's")
    if _state["gemm"] == "cuda" and path in ("rows", "group"):
        # pixal3d_kernels: fc1 with its GELU in the epilogue, the hidden layer quantised again for fc2 with the
        # scales fc1's input had (per row, or per row and group); on the old cards' profiles in row slabs where the
        # hidden layer would be big (mlp_slab_rows: the same bits)
        group = GROUP if path == "group" else 0
        xq, xs = x if isinstance(x, tuple) else quantize_rows(x, group)
        return cuda_mlp(xq, xs, w1, w2, group, inv_smooth2, out_dtype, mlp_slab_rows(xq.shape[0], w1[0].shape[0],
                                                                                     out_dtype))
    if path == "rows" and out_dtype not in _OUT_NAME:
        path = "intmm"
    if path == "rows":
        xq, xs = x if isinstance(x, tuple) else quantize_rows(x)
        h = gemm(xq, xs, *w1, gelu=True, group=GROUP if xs.dim() == 2 else 0, out=_OUT_NAME[out_dtype])
        return gemm(*quantize_rows(h), *w2, out=_OUT_NAME[out_dtype], resid=resid)
    if path == "intmm":
        if isinstance(x, tuple):
            xq, xs = x
        else:
            xq, xs = quantize_rows(x)
        hq, hs = intmm_linear(xq, xs, *w1, gelu=True, out_int8=True)
        return intmm_linear(hq, hs, *w2, out_dtype=out_dtype)
    if out_dtype not in _OUT_NAME:
        raise ValueError(f"the group path's fc2 GEMM writes bf16 or fp16, not {out_dtype}")
    if isinstance(x, tuple):
        xq, xs = x
    else:
        xq, xs = quantize_rows(x)
    M = xq.shape[0]
    if cfg1 is None:
        if xs.dim() == 2 and w1[1].dim() != 2:
            cfg1 = fc1_group_cfg(xq.device)
        else:
            cfg1 = (pick_config(M, w1[0].shape[0], xq.shape[1], GROUP, w1[1].dim() == 2) if xs.dim() == 2
                    else CFG_FC1_INT8)
    if cfg1[1] != GROUP:                            # fc1's BLOCK_N is the group size of fc2's input
        cfg1 = (cfg1[0], GROUP) + tuple(cfg1[2:])
    hq, hs = gemm(xq, xs, *w1, gelu=True, out="int8", group=GROUP if xs.dim() == 2 else 0, cfg=cfg1,
                  out_smooth=inv_smooth2)
    return gemm(hq, hs, *w2, group=GROUP, cfg=CFG_GROUP, out=_OUT_NAME[out_dtype], resid=resid)


# =====================================================================================================================
# References, the first-use self-check, the switch
# =====================================================================================================================

def reference_int(xq: torch.Tensor, w_q: torch.Tensor) -> torch.Tensor:
    """The exact integer product (fp64 holds every int8 x int8 sum below 2^53), as int64."""
    return (xq.double() @ w_q.double().t()).to(torch.int64)


def reference(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
              bias: Optional[torch.Tensor], gelu: bool = False) -> torch.Tensor:
    """fp64 y from the same quantised operands (per-row or group scales; weight scales per channel or per group),
    unrounded."""
    K = xq.shape[1]
    if w_s.dim() == 2:
        xd = dequantize_rows(xq.double(), xs.double()) if xs.dim() == 2 else xq.double() * xs.double()[:, None]
        y = xd @ dequantize_rows(w_q.double(), w_s.double()).t()
    elif xs.dim() == 2:
        g = K // xs.shape[1]
        xd = xq.double().view(xq.shape[0], -1, g) * xs.double().view(xq.shape[0], -1, 1)
        y = xd.view(xq.shape) @ w_q.double().t()
        y = y * w_s.double()[None, :]
    else:
        y = (xq.double() @ w_q.double().t()) * xs.double()[:, None] * w_s.double()[None, :]
    if bias is not None:
        y = y + bias.double()[None, :]
    if gelu:
        y = 0.5 * y * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (y + 0.044715 * y ** 3)))
    return y


def reference_mag(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
                  bias: Optional[torch.Tensor], rows: int = 128) -> torch.Tensor:
    """The magnitude the fp32 epilogue rounds against (fp64): per-row scales |acc * rs * cs| + |bias|; group scales
    sum over the K groups of |partial_g * s_g| * cs + |bias| (the fp32 accumulator adds those terms)."""
    M, K = xq.shape
    out = []
    for r0 in range(0, M, rows):
        q = xq[r0:r0 + rows].double()
        if w_s.dim() == 2:
            G = w_s.shape[1]
            part = torch.einsum("mgk,ngk->mng", q.view(q.shape[0], G, -1), w_q.double().view(w_q.shape[0], G, -1))
            sx = xs[r0:r0 + rows].double()
            sx = sx[:, None, :] if sx.dim() == 2 else sx[:, None, None]
            m = (part.abs() * sx * w_s.double()[None, :, :]).sum(-1)
        elif xs.dim() == 2:
            G = xs.shape[1]
            part = torch.einsum("mgk,ngk->mng", q.view(q.shape[0], G, -1), w_q.double().view(w_q.shape[0], G, -1))
            m = (part.abs() * xs[r0:r0 + rows].double()[:, None, :]).sum(-1) * w_s.double()[None, :]
        else:
            m = (q @ w_q.double().t()).abs() * xs[r0:r0 + rows].double()[:, None] * w_s.double()[None, :]
        if bias is not None:
            m = m + bias.double().abs()[None, :]
        out.append(m)
    return torch.cat(out)


def within(got: torch.Tensor, want64: torch.Tensor, mag64: torch.Tensor, c: float = 4.0,
           gelu: bool = False) -> Tuple[float, float]:
    """(largest |got - want| / (one bf16 ulp of want + c * 2^-24 * mag), share of elements != bf16(want)): <= 1 means
    the result is the fp64 value of the same int8 operands up to the epilogue's fp32 rounding (c roundings of the
    magnitude it works on: the int32 -> fp32 conversion, the two scale products, the bias add; + one per group for
    the group path's fp32 accumulator) and the final bf16 rounding. Through GELU (slope <= 1.13) the bound x 1.2."""
    w = want64.float()
    ulp = torch.where(w == 0, torch.full_like(w, 2.0 ** -133),
                      torch.exp2(torch.floor(torch.log2(w.abs())) - 7)).double()
    tol = ulp + c * (1.2 if gelu else 1.0) * 2.0 ** -24 * mag64
    d = (got.double() - want64).abs() / tol
    neq = (got.float() != w.to(torch.bfloat16).float()).double().mean()
    return float(d.max()), float(neq)


def bf16_ulps(got: torch.Tensor, want64: torch.Tensor) -> Tuple[float, float]:
    """(largest |got - want| in bf16 ulps of want, share of elements != bf16(want)). Misleading where the epilogue's
    terms cancel (a tiny result): use within()."""
    w = want64.float()
    ulp = torch.where(w == 0, torch.full_like(w, 2.0 ** -133),
                      torch.exp2(torch.floor(torch.log2(w.abs())) - 7))
    d = (got.double() - want64).abs() / ulp.double()
    neq = (got.float() != w.to(torch.bfloat16).float()).double().mean()
    return float(d.max()), float(neq)


# fc1 with GELU and group-scaled int8 output (int8_mlp's "group" path) on group-scaled input: its tile is chosen
# once per card at install (calibrate_fc1_tiles, the calibration record's "triton" section, "int8_tiles"). Every
# candidate gives the same bits: BLOCK_K is the group (each 128-K slab's int32 sum exact, the slabs summed in order
# per element), BLOCK_N fc2's input group (the output's scales per 128 columns). On an RTX 3060 (64, 128, 128, 8,
# 3, 4) was 13 % faster than CFG_GROUP at 19,928 rows.
FC1_GROUP_CANDIDATES = (CFG_GROUP, (64, 128, 128, 8, 3, 4))
FC1_GROUP_CFG: Optional[tuple] = None   # forced (developers); None: the calibration record's choice, else CFG_GROUP
FC1_TILE_MARGIN = 0.03                  # a candidate serves only where it is this much faster at every size timed
FC1_TILE_ROWS = (21504, 8192)           # the sizes timed at install
_fc1_cfg: dict = {}


def fc1_group_cfg(device: torch.device) -> tuple:
    """fc1's tile for this card (FC1_GROUP_CFG, else the install's record of calibrate_fc1_tiles where it was made
    on this card with this Triton, else CFG_GROUP)."""
    if FC1_GROUP_CFG is not None:
        return tuple(FC1_GROUP_CFG)
    key = str(device)
    if key not in _fc1_cfg:
        _fc1_cfg[key] = CFG_GROUP
        try:
            from .. import calibration
            import triton

            entry = ((calibration.section("triton", device) or {}).get("int8_tiles") or {})
            chosen = tuple(entry.get("fc1_group") or ())
            if entry.get("triton") == triton.__version__ and chosen in FC1_GROUP_CANDIDATES:
                _fc1_cfg[key] = chosen
        except Exception:       # noqa: BLE001 - no record, no Triton: the shipped tile
            pass
    return _fc1_cfg[key]


def warm_handoffs(device: torch.device, rows: tuple = (4096, 4099)) -> int:
    """The flows' blocks' hand-off kernels compiled at the flows' widths (the install's calibration calls this, so
    that the first prop finds them in Triton's cache): norm_quant for the modulated norms (group scales) and the
    affine one (per row) at norm_modulate's warps, the per-row GEMM with the residual in its epilogue (a gate, and a
    second term), and fused.residual_row. Returns the calls made (0 where the Triton GEMMs do not serve)."""
    from . import fused

    if not ready(device)[0] or gemm_path() != "triton":
        return 0
    g = torch.Generator(device=device).manual_seed(0)
    C = 1536
    w_q, w_s = quantize_weight(torch.randn((C, C), device=device, dtype=torch.bfloat16, generator=g) * 0.03)
    bias = torch.zeros(C, device=device)
    w32 = torch.ones(C, device=device, dtype=torch.float32)  # norm2's weight/bias: LayerNorm32 stays fp32
    calls = 0
    with torch.no_grad():
        for m in rows:
            x = torch.randn((m, C), device=device, dtype=torch.bfloat16, generator=g)
            mod = torch.randn((1, 2 * C), device=device, dtype=torch.bfloat16, generator=g)
            norm_quant(x, 1e-6, scale=mod[:, :C], shift=mod[:, C:], group=GROUP, num_warps=8)
            norm_quant(x, 1e-6, weight=w32, bias=w32, group=0, num_warps=8)
            int8_linear(x, w_q, w_s, bias, path="triton", resid={"x": x, "gate": mod[:, :C]})
            int8_linear(x, w_q, w_s, bias, path="triton", resid={"x": x, "second": x})
            if fused.active(x):
                saved = fused.ROW_RESIDUAL
                fused.ROW_RESIDUAL = True
                try:
                    fused.residual_row(x, x[:1])
                finally:
                    fused.ROW_RESIDUAL = saved
            calls += 5
    torch.cuda.synchronize(device)
    return calls


def calibrate_fc1_tiles(device: torch.device, reps: int = 6, iters: int = 5) -> dict:
    """The install's choice of fc1's tile (FC1_GROUP_CANDIDATES): each candidate on the same random tables at
    FC1_TILE_ROWS rows (1536 -> 8192, GELU, int8 out, as int8_mlp runs it), its output torch.equal to CFG_GROUP's,
    timed interleaved (CUDA events, medians); the fastest serves where it is FC1_TILE_MARGIN faster than CFG_GROUP at
    every size. Returns the record's entry: {"fc1_group", "triton", "timings", "summary", "ok"}."""
    import statistics

    import triton

    g = torch.Generator(device=device).manual_seed(0)
    w = torch.randn((8192, 1536), device=device, dtype=torch.bfloat16, generator=g) * 0.03
    wq, ws = quantize_weight(w)
    bias = torch.randn(8192, device=device, generator=g) * 0.1
    del w
    found: dict = {"fc1_group": list(CFG_GROUP), "triton": triton.__version__, "timings": [], "ok": True}
    wins = {c: True for c in FC1_GROUP_CANDIDATES[1:]}
    for m in FC1_TILE_ROWS:
        x = torch.randn((m, 1536), device=device, dtype=torch.bfloat16, generator=g)
        xq, xs = quantize_rows(x, GROUP)
        del x
        run = {c: (lambda c=c: gemm(xq, xs, wq, ws, bias, gelu=True, out="int8", group=GROUP, cfg=c,
                                    kernel="triton"))
               for c in FC1_GROUP_CANDIDATES}
        want = run[CFG_GROUP]()
        for c in FC1_GROUP_CANDIDATES[1:]:
            got = run[c]()
            if not (torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])):
                wins[c] = False
        del want, got
        ms: dict = {c: [] for c in FC1_GROUP_CANDIDATES}
        for rep in range(reps):
            for c in (FC1_GROUP_CANDIDATES if rep % 2 == 0 else FC1_GROUP_CANDIDATES[::-1]):
                run[c]()
                events = []
                for _ in range(iters):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    run[c]()
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize(device)
                ms[c] += [a.elapsed_time(b) for a, b in events]
        med = {c: statistics.median(v) for c, v in ms.items()}
        found["timings"].append({"rows": m, **{".".join(map(str, c)): round(t, 3) for c, t in med.items()}})
        for c in wins:
            wins[c] = wins[c] and med[c] <= med[CFG_GROUP] * (1 - FC1_TILE_MARGIN)
        del xq, xs, run, ms
    best = [c for c in FC1_GROUP_CANDIDATES[1:] if wins[c]]
    if best:
        def total(c):
            return sum(t[".".join(map(str, c))] for t in found["timings"])
        found["fc1_group"] = list(min(best, key=total))
    found["summary"] = "fc1 tile " + ".".join(map(str, found["fc1_group"])) + ": " + "; ".join(
        f"{t['rows']:,} rows " + ", ".join(f"{k} {v:.2f} ms" for k, v in t.items() if k != "rows")
        for t in found["timings"])
    del wq, ws, bias
    torch.cuda.empty_cache()
    _fc1_cfg.clear()
    try:
        found["group_cuda"] = calibrate_group_cuda(device, tuple(found["fc1_group"]), reps=reps, iters=iters)
    except Exception as exc:        # noqa: BLE001 - the Triton kernel serves every group layer
        found["group_cuda"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return found


# =====================================================================================================================
# The group GEMM on pixal3d_kernels' CUDA kernel (sm_80+)
# =====================================================================================================================

# The group GEMM's layers (activations scaled per row and 128-K slab, weights per output channel: the "detail"
# policy's to_qkv, fc1 and fc2) can run pixal3d_kernels' sm_80+ kernel (int8_gemm_group80) instead of the Triton one.
# Its arithmetic is the Triton kernel's instruction for instruction (each slab's exact int32 sum, the fp32
# accumulation in slab order, the same epilogue down to libdevice's tanhf), so the numbers are the same bit for bit
# (checked once a process on small tables: group_cuda_check, and on the flows' shapes when the install races the two
# kernels: calibrate_group_cuda). Which layer forms take which of its tiles is the race's (the record's "triton" ->
# "int8_tiles" -> "group_cuda"); a form it did not choose, or a record of another card, Triton or kernel build, keeps
# the Triton kernel. PIXAL3D_INT8_GEMM = triton | cuda forces one for every group layer (developers; "cuda" takes the
# record's tile, else GROUP_CUDA_DEFAULT).
GROUP_CUDA_ENV = "PIXAL3D_INT8_GEMM"
GROUP_CUDA_FORMS = {"qkv": (4608, 1536, False, "bf16"), "fc1": (8192, 1536, True, "int8"),
                    "fc2": (1536, 8192, False, "bf16")}     # the forms raced at install: N, K, GELU, out
GROUP_CUDA_CANDIDATES = {"bf16": (0, 1, 2, 3, 4), "int8": (2, 4, 6)}   # the kernel's tiles raced per output kind
GROUP_CUDA_DEFAULT = {"bf16": 2, "int8": 4, "fp16": 2}                # a forced run's tiles where the record has none
GROUP_CUDA_GROUP_M = 8                  # the tile order (row tiles per group: Triton's GROUP_M)
GROUP_CUDA_MARGIN = 0.03                # a tile serves only where it is this much faster than Triton at every size
GROUP_CUDA_ROWS = FC1_TILE_ROWS
_group_cuda: dict = {"op": None, "checked": {}, "choice": {}, "said": False}


def group_cuda_op() -> Optional[Callable]:
    """pixal3d_kernels' int8_gemm_group80 where the installed build has it and runs on this card, else None."""
    if _group_cuda["op"] is None:
        _group_cuda["op"] = False
        try:
            import pixal3d_kernels  # noqa: F401

            ops = torch.ops.pixal3d_kernels
            if bool(ops.int8_gemm_group80_loaded()):
                _group_cuda["op"] = ops.int8_gemm_group80
        except Exception:       # noqa: BLE001 - no package, an older build, no sm_80+ code for the card
            pass
    return _group_cuda["op"] or None


def _kernels_build() -> str:
    try:
        from importlib import metadata

        return metadata.version("pixal3d_kernels")
    except Exception:       # noqa: BLE001
        return ""


def group_cuda_stands(int8_tiles: Optional[dict]) -> bool:
    """Whether an install record's int8_tiles entry holds the race this build would make (calibrate.py's due rule):
    a "group_cuda" entry made with the installed pixal3d_kernels build, or one that could not race (no kernel then,
    or a failure: not tried again by itself), or no kernel to race now."""
    entry = (int8_tiles or {}).get("group_cuda")
    if not isinstance(entry, dict):
        return group_cuda_op() is None
    return not entry.get("ok") or entry.get("kernels") == _kernels_build() or group_cuda_op() is None


def _tile_info(tile: int) -> list:
    return [int(v) for v in torch.ops.pixal3d_kernels.int8_gemm_group80_info(int(tile))]


def _form_key(N: int, K: int, gelu: bool, out: str) -> str:
    return f"{N}x{K}{'.gelu' if gelu else ''}.{out}"


def _group_cuda_choice(device: torch.device) -> dict:
    """{form key: tile} for this card (PIXAL3D_INT8_GEMM, else the install's race), or {"*": None} for none."""
    key = str(device)
    if key not in _group_cuda["choice"]:
        forced = os.environ.get(GROUP_CUDA_ENV, "").strip().lower()
        choice: dict = {}
        if forced not in ("", "triton", "cuda"):
            print(f"[INT8] {GROUP_CUDA_ENV}={forced!r} is neither triton nor cuda: the install's choice serves",
                  flush=True)
            forced = ""
        if forced != "triton" and group_cuda_op() is not None:
            try:
                from .. import calibration
                import triton

                entry = (((calibration.section("triton", device) or {}).get("int8_tiles") or {}).get("group_cuda")
                         or {})
                tiles = entry.get("tiles") or {}
                if (entry.get("ok") and entry.get("triton") == triton.__version__
                        and entry.get("kernels") == _kernels_build()
                        and all(_tile_info(int(t)) == list(info) for t, info in tiles.items())):
                    choice = {k: int(t) for k, t in (entry.get("choice") or {}).items() if t is not None}
            except Exception:       # noqa: BLE001 - no record, no Triton: the Triton kernel serves
                choice = {}
            if forced == "cuda":
                choice["*"] = True
        _group_cuda["choice"][key] = choice
    return _group_cuda["choice"][key]


def group_cuda_tile(xq: torch.Tensor, w_q: torch.Tensor, gelu: bool, out: str,
                    resid: Optional[dict] = None) -> Optional[int]:
    """The CUDA kernel's tile for this group GEMM call, or None where the Triton kernel serves (no choice for its
    form, a form the kernel lacks: fp16 unless forced, a residual not in bf16, int8 out on N not in groups of 128;
    or the first-use check failed)."""
    if not xq.is_cuda or out not in ("bf16", "int8", "fp16"):
        return None
    choice = _group_cuda_choice(xq.device)
    if not choice:
        return None
    N, K = w_q.shape
    tile = choice.get(_form_key(N, K, gelu, out))
    if tile is None and choice.get("*"):
        tile = GROUP_CUDA_DEFAULT[out]
    if tile is None or (out == "fp16" and not choice.get("*")) or (out == "int8" and N % GROUP):
        return None
    if resid is not None and any(t is not None and t.dtype != torch.bfloat16
                                 for t in (resid.get("x"), resid.get("gate"), resid.get("second"))):
        return None
    if not group_cuda_check(xq.device, tile):
        return None
    if not _group_cuda["said"]:
        _group_cuda["said"] = True
        print(f"[INT8] the group-scaled layers run pixal3d_kernels' sm_80 GEMM where the install chose it "
              f"({', '.join(f'{k} tile {t}' for k, t in sorted(choice.items()) if k != '*') or 'forced'}; "
              f"{GROUP_CUDA_ENV})", flush=True)
    return tile


def group_cuda(xq: torch.Tensor, xs: torch.Tensor, w_q: torch.Tensor, w_s: torch.Tensor,
               bias: Optional[torch.Tensor], *, gelu: bool = False, out: str = "bf16", resid: Optional[dict] = None,
               tile: int = 0):
    """gemm()'s group path on pixal3d_kernels' kernel: the same arguments and results (bf16 / fp16 [M, N], or int8
    [M, N] + scales [M, N / 128]), the residual modes included."""
    rx = rg = r2 = None
    if resid is not None:
        rx, rg, r2 = resid["x"], resid.get("gate"), resid.get("second")
        if rg is not None:
            rg = rg.reshape(-1)
    y, qs = group_cuda_op()(xq, xs, w_q, w_s, bias, gelu, {"bf16": 0, "int8": 2, "fp16": 3}[out], rx, rg, r2,
                            int(tile), GROUP_CUDA_GROUP_M)
    return (y, qs) if out == "int8" else y


def _torch_resid(h: torch.Tensor, x: torch.Tensor, gate: Optional[torch.Tensor] = None,
                 second: Optional[torch.Tensor] = None) -> torch.Tensor:
    """fused.residual's rounded mode in torch ops on the bf16 result h (the Triton epilogue's RESID arithmetic)."""
    hf = h.float()
    if gate is not None:
        hf = (hf * gate.reshape(1, -1).float()).to(torch.bfloat16).float()
    elif second is not None:
        hf = (second.float() + hf).to(torch.bfloat16).float()
    return (x.float() + hf).to(torch.bfloat16)


def group_cuda_check(device: torch.device, tile: int) -> bool:
    """Once a process per tile: the CUDA kernel's outputs torch.equal to the Triton kernel's on small tables (300
    rows: masked tails; an outlier group) in the flows' forms - bf16 with a bias, int8 + scales with GELU (int8
    tiles), fp16 where forced: Triton kernels the install's warm-up compiled, none compiled here - and its residual
    modes equal to the residual's arithmetic on its own bf16 result. A mismatch turns the CUDA kernel off for the
    process (once said)."""
    key = (str(device), int(tile))
    if key in _group_cuda["checked"]:
        return _group_cuda["checked"][key]
    ok, why = True, ""
    try:
        info = _tile_info(tile)
        g = torch.Generator(device=device).manual_seed(3)
        M, K, N = 300, 256, 384
        x = torch.randn((M, K), device=device, dtype=torch.bfloat16, generator=g)
        x[7, :9] = 50.0
        w = torch.randn((N, K), device=device, dtype=torch.bfloat16, generator=g) * 0.05
        b = torch.randn((N,), device=device, generator=g)
        xq, xs = quantize_rows(x, GROUP)
        wq, ws = quantize_weight(w)
        forms = [("bf16", False, b)]
        if info[1] % GROUP == 0 and GROUP % info[3] == 0:
            forms.append(("int8", True, b))
        if _group_cuda_choice(device).get("*"):
            forms.append(("fp16", False, b))
        for out, gelu, bias in forms:
            cfg = fc1_group_cfg(device) if out == "int8" else CFG_GROUP
            want = gemm(xq, xs, wq, ws, bias, gelu=gelu, out=out, group=GROUP, cfg=cfg, kernel="triton")
            got = group_cuda(xq, xs, wq, ws, bias, gelu=gelu, out=out, tile=tile)
            same = (torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])) if out == "int8" else \
                torch.equal(got, want)
            if not same:
                ok, why = False, f"{out}{' + GELU' if gelu else ''} differs from the Triton kernel's"
                break
        if ok:
            h = group_cuda(xq, xs, wq, ws, b, tile=tile)
            rx = torch.randn((M, N), device=device, dtype=torch.bfloat16, generator=g)
            gate = torch.randn((1, N), device=device, dtype=torch.bfloat16, generator=g)
            for resid in ({"x": rx, "gate": gate}, {"x": rx, "second": h.flip(0).contiguous()}, {"x": rx}):
                got = group_cuda(xq, xs, wq, ws, b, resid=resid, tile=tile)
                if not torch.equal(got, _torch_resid(h, **resid)):
                    ok, why = False, f"the residual ({', '.join(sorted(resid))}) differs from its arithmetic"
                    break
        torch.cuda.synchronize(device)
    except Exception as exc:        # noqa: BLE001
        ok, why = False, f"{type(exc).__name__}: {str(exc)[:200]}"
    if not ok:
        print(f"[INT8] pixal3d_kernels' sm_80 group GEMM (tile {tile}) is left out: {why}; the Triton kernel serves",
              flush=True)
    _group_cuda["checked"][key] = ok
    return ok


def calibrate_group_cuda(device: torch.device, fc1_cfg: tuple, reps: int = 6, iters: int = 5) -> dict:
    """The install's race of the group layers (GROUP_CUDA_FORMS, as the flows run them: bias, fc1's GELU and int8
    output, fc2 on fc1's output) between the Triton kernel (fc1 at fc1_cfg) and the CUDA kernel's tiles
    (GROUP_CUDA_CANDIDATES) at GROUP_CUDA_ROWS rows: every tile's outputs torch.equal to Triton's, timed interleaved
    (CUDA events, medians); per form the fastest equal tile serves where it is GROUP_CUDA_MARGIN faster than Triton
    at every size. Returns the record's entry {"ok", "triton", "kernels", "tiles", "choice", "timings", "summary"}."""
    import statistics

    import triton

    if group_cuda_op() is None:
        return {"ok": False, "why": "pixal3d_kernels has no sm_80 group GEMM for this card"}
    g = torch.Generator(device=device).manual_seed(0)
    cands = sorted({t for ts in GROUP_CUDA_CANDIDATES.values() for t in ts})
    found: dict = {"ok": True, "triton": triton.__version__, "kernels": _kernels_build(),
                   "tiles": {str(t): _tile_info(t) for t in cands}, "choice": {}, "timings": []}
    weights = {name: quantize_weight(torch.randn((n, k), device=device, dtype=torch.bfloat16, generator=g) * 0.03)
               for name, (n, k, _, _) in GROUP_CUDA_FORMS.items()}
    biases = {name: torch.randn(n, device=device, generator=g) * 0.1 for name, (n, _, _, _) in GROUP_CUDA_FORMS.items()}
    wins = {name: {t: True for t in GROUP_CUDA_CANDIDATES[out]} for name, (_, _, _, out) in GROUP_CUDA_FORMS.items()}
    totals = {name: {t: 0.0 for t in GROUP_CUDA_CANDIDATES[out]} for name, (_, _, _, out) in
              GROUP_CUDA_FORMS.items()}
    for m in GROUP_CUDA_ROWS:
        x = torch.randn((m, 1536), device=device, dtype=torch.bfloat16, generator=g)
        xq, xs = quantize_rows(x, GROUP)
        del x
        hq, hs = gemm(xq, xs, *weights["fc1"], biases["fc1"], gelu=True, out="int8", group=GROUP, cfg=fc1_cfg,
                      kernel="triton")
        for name, (n, k, gelu, out) in GROUP_CUDA_FORMS.items():
            ax, axs = (hq, hs) if k == 8192 else (xq, xs)
            cfg = fc1_cfg if out == "int8" else CFG_GROUP
            run = {"triton": (lambda: gemm(ax, axs, *weights[name], biases[name], gelu=gelu, out=out, group=GROUP,
                                           cfg=cfg, kernel="triton"))}
            for t in GROUP_CUDA_CANDIDATES[out]:
                run[t] = (lambda t=t: group_cuda(ax, axs, *weights[name], biases[name], gelu=gelu, out=out, tile=t))
            want = run["triton"]()
            for t in GROUP_CUDA_CANDIDATES[out]:
                got = run[t]()
                same = (torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])) if out == "int8" else \
                    torch.equal(got, want)
                wins[name][t] = wins[name][t] and same
                del got
            del want
            ms: dict = {c: [] for c in run}
            order = list(run)
            for rep in range(reps):
                for c in (order if rep % 2 == 0 else order[::-1]):
                    run[c]()
                    events = []
                    for _ in range(iters):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        run[c]()
                        end.record()
                        events.append((start, end))
                    torch.cuda.synchronize(device)
                    ms[c] += [a.elapsed_time(b) for a, b in events]
            med = {c: statistics.median(v) for c, v in ms.items()}
            found["timings"].append({"rows": m, "form": name, **{str(c): round(v, 3) for c, v in med.items()}})
            for t in GROUP_CUDA_CANDIDATES[out]:
                wins[name][t] = wins[name][t] and med[t] <= med["triton"] * (1 - GROUP_CUDA_MARGIN)
                totals[name][t] += med[t]
            del run, ms
        del xq, xs, hq, hs
        torch.cuda.empty_cache()
    for name, (n, k, gelu, out) in GROUP_CUDA_FORMS.items():
        best = [t for t, w in wins[name].items() if w]
        found["choice"][_form_key(n, k, gelu, out)] = min(best, key=lambda t: totals[name][t]) if best else None
    found["summary"] = "group layers: " + ", ".join(
        f"{name} {'tile ' + str(found['choice'][_form_key(n, k, gelu, out)]) if found['choice'][_form_key(n, k, gelu, out)] is not None else 'Triton'}"
        for name, (n, k, gelu, out) in GROUP_CUDA_FORMS.items()) + "; " + "; ".join(
        f"{t['form']} {t['rows']:,} rows triton {t['triton']:.2f} ms, best tile "
        f"{min((c for c in t if c not in ('rows', 'form', 'triton')), key=lambda c: t[c])} "
        f"{min(v for c, v in t.items() if c not in ('rows', 'form', 'triton')):.2f} ms" for t in found["timings"])
    del weights, biases
    torch.cuda.empty_cache()
    _group_cuda["choice"].clear()
    return found


def self_check(device: torch.device) -> Tuple[bool, str]:
    """Every path the card uses once on small tables (M = 300: masked tails), against fp64 from the same int8
    operands: the integer part exactly, the outputs (in the torso's dtype) within the epilogue's fp32 rounding
    (within() <= 1). Done once per process before the first int8 call."""
    try:
        g = torch.Generator(device=device).manual_seed(0)
        dt = precision.torso()
        M, K, N = 300, 256, 384
        x = torch.randn((M, K), device=device, dtype=dt, generator=g)
        w = torch.randn((N, K), device=device, dtype=dt, generator=g) * 0.05
        b = torch.randn((N,), device=device, dtype=dt, generator=g)
        xq, xs = quantize_rows(x)
        rq, rs = quantize_rows_torch(x)
        if not (torch.equal(xq, rq) and torch.equal(xs, rs)):
            return False, "quantize_rows differs from the torch reference"
        wq, ws = quantize_weight(w)
        exact = reference_int(xq, wq)
        # the Triton GEMM's checks only where it serves: below sm_80 Triton cannot compile an int8 product at all
        # (a real RTX 2060 Super: "PassManager::run failed"), which would leave pixal3d_kernels out for nothing
        triton_gemm = _state["gemm"] == "triton"
        if triton_gemm:
            tri = gemm(xq, xs, wq, ws, None, out="int32", cfg=(64, 64, 64, 8, 3, 4))
            if not torch.equal(tri.to(torch.int64), exact):
                return False, "the Triton int8 GEMM's integer product is wrong"
        if not torch.equal(torch._int_mm(xq, wq.t()).to(torch.int64), exact):
            return False, "torch._int_mm's integer product is wrong"
        mag = reference_mag(xq, xs, wq, ws, b)
        for gelu in (False, True):
            want = reference(xq, xs, wq, ws, b, gelu)
            got_by = {"intmm": intmm_linear(xq, xs, wq, ws, b, gelu=gelu, out_dtype=dt)}
            if triton_gemm:
                got_by["triton"] = gemm(xq, xs, wq, ws, b, gelu=gelu, cfg=(64, 64, 64, 8, 3, 4))
            if _state["gemm"] == "cuda":
                got_by["pixal3d_kernels' per-row GEMM"] = cuda_linear(xq, xs, wq, ws, b, gelu=gelu, out_dtype=dt)
            for name, got in got_by.items():
                worst, _ = within(got, want, mag, gelu=gelu)
                if worst > 1.0:
                    return False, f"{name} (gelu={gelu}) is {worst:.2f}x the fp32-rounding bound off"
        if _state["gemm"] == "cuda":
            gq, gs = quantize_rows(x, GROUP)
            for gelu in (False, True):
                got = cuda_linear(gq, gs, wq, ws, b, gelu=gelu, out_dtype=dt)
                worst, _ = within(got, reference(gq, gs, wq, ws, b, gelu), reference_mag(gq, gs, wq, ws, b),
                                  c=4 + K // GROUP, gelu=gelu)
                if worst > 1.0:
                    return False, f"pixal3d_kernels' group GEMM (gelu={gelu}) {worst:.2f}x the fp32-rounding bound off"
            wg, wgs = quantize_weight(w, group=GROUP)
            got = cuda_linear(gq, gs, wg, wgs, b, out_dtype=dt)
            worst, _ = within(got, reference(gq, gs, wg, wgs, b), reference_mag(gq, gs, wg, wgs, b),
                              c=4 + 2 * K // GROUP)
            if worst > 1.0:
                return False, f"pixal3d_kernels' group-weight GEMM {worst:.2f}x the fp32-rounding bound off"
            if not torch.equal(cuda_linear(xq, xs, wq, ws, None, out_dtype=torch.int32).to(torch.int64), exact):
                return False, "pixal3d_kernels' integer product is wrong"
        if not triton_gemm:
            torch.cuda.synchronize(device)
            return True, ""
        gq, gs = quantize_rows(x, GROUP)
        rq, rs = quantize_rows_torch(x, GROUP)
        if not (torch.equal(gq, rq) and torch.equal(gs, rs)):
            return False, "group quantize_rows differs from the torch reference"
        got = gemm(gq, gs, wq, ws, b, group=GROUP, cfg=(64, 64, 128, 8, 3, 4), kernel="triton")
        worst, _ = within(got, reference(gq, gs, wq, ws, b), reference_mag(gq, gs, wq, ws, b), c=4 + K // GROUP)
        if worst > 1.0:
            return False, f"group GEMM {worst:.2f}x the fp32-rounding bound off"
        ok, why = _check_group_weights(device, g, x, w, b)
        if not ok:
            return False, why
        torch.cuda.synchronize(device)
    except Exception as exc:        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
    return True, ""


def _check_group_weights(device: torch.device, g: torch.Generator, x: torch.Tensor, w: torch.Tensor,
                         b: torch.Tensor) -> Tuple[bool, str]:
    """The "cs" pieces: SmoothQuant 1 / s in the quantiser (bit-equal to the torch reference), weights quantised per
    output channel and group (W * s), the GEMM with group scales on both operands (bf16 out within the fp32-rounding
    bound of the fp64 value), and fc1's int8 output with the next layer's 1 / s applied before the quantisation
    (each value within half a step of its group's scale of the fp64 value)."""
    M, K = x.shape
    N = w.shape[0]
    s = torch.rand((K,), device=device, generator=g) * 4 + 0.25
    s[3] = 20.0                                         # an outlier channel's divisor
    inv = 1.0 / s
    gq, gs = quantize_rows(x, GROUP, inv)
    rq, rs = quantize_rows_torch(x, GROUP, inv)
    if not (torch.equal(gq, rq) and torch.equal(gs, rs)):
        return False, "smoothed group quantize_rows differs from the torch reference"
    wq, ws = quantize_weight(w, smooth=s, group=GROUP)
    rq, rs = quantize_rows_torch(w.float() * s[None, :], GROUP)
    if not (torch.equal(wq, rq) and torch.equal(ws, rs)):
        return False, "the group weight quantiser differs from the torch reference"
    for gelu in (False, True):
        got = gemm(gq, gs, wq, ws, b, gelu=gelu, group=GROUP, cfg=(64, 64, 128, 8, 3, 4))
        worst, _ = within(got, reference(gq, gs, wq, ws, b, gelu), reference_mag(gq, gs, wq, ws, b),
                          c=4 + 2 * K // GROUP, gelu=gelu)
        if worst > 1.0:
            return False, f"group-weight GEMM (gelu={gelu}) {worst:.2f}x the fp32-rounding bound off"
    inv2 = 1.0 / (torch.rand((N,), device=device, generator=g) * 4 + 0.25)
    hq, hs = gemm(gq, gs, wq, ws, b, gelu=True, out="int8", group=GROUP, cfg=(64, GROUP, 128, 8, 3, 4),
                  out_smooth=inv2)
    want = reference(gq, gs, wq, ws, b, True) * inv2.double()[None, :]
    mag = reference_mag(gq, gs, wq, ws, b) * 1.2 * inv2.double()[None, :]
    step = hs.double().repeat_interleave(GROUP, dim=1)
    err = (dequantize_rows(hq.double(), hs.double()) - want).abs()
    worst = float((err / (0.5 * (1 + 1e-4) * step + (4 + 2 * K // GROUP) * 2.0 ** -24 * mag + 1e-30)).max())
    if worst > 1.0:
        return False, f"fc1's smoothed int8 output {worst:.2f}x half a step off"
    return True, ""


def _cuda_elementwise(device: torch.device) -> bool:
    """Below sm_80, where the GEMM is pixal3d_kernels': whether that package's quantiser serves (the torch ops'
    bits, checked once: modules/old_cards.py), where the Triton quantiser does not run (Pascal, a Turing card whose
    Triton fails its check): 10-16 times the torch ops' speed. Triton's runs 2-30 % faster where it runs (an RTX 2070's
    passes)."""
    if _state["gemm"] != "cuda":
        return False
    from . import old_cards

    return old_cards.quantiser_ok(device)


def _triton_elementwise(device: torch.device) -> bool:
    """Whether the Triton quantiser and epilogue run on this card and give the torch ops' bits (a card below sm_80:
    Triton may run there, or not - Windows builds for Turing are untested - so nothing depends on it)."""
    if not precision.profile().triton:
        return False
    try:
        g = torch.Generator(device=device).manual_seed(1)
        x = torch.randn((300, 256), device=device, dtype=precision.torso(), generator=g)
        for group in (0, GROUP):
            tq, ts = quantize_rows_triton(x, group)
            rq, rs = quantize_rows_torch(x, group)
            if not (torch.equal(tq, rq) and torch.equal(ts, rs)):
                return False
        c32 = torch.randint(-2 ** 20, 2 ** 20, (300, 384), device=device, dtype=torch.int32, generator=g)
        xs = torch.rand((300,), device=device, generator=g) + 0.5
        ws = torch.rand((384,), device=device, generator=g) + 0.5
        b = torch.randn((384,), device=device, generator=g)
        for gelu in (False, True):
            got = epilogue_triton(c32, xs, ws, b, gelu=gelu, out_dtype=torch.float32)
            want = epilogue_torch(c32, xs, ws, b, gelu=gelu, out_dtype=torch.float32)
            if not bool(((got - want).abs() <= 2.0 ** -20 * want.abs() + 2.0 ** -20).all()):
                return False
        torch.cuda.synchronize(device)
    except Exception:       # noqa: BLE001 - whatever Triton or the card raises: the torch ops serve
        return False
    return True


def ready(device: torch.device) -> Tuple[bool, str]:
    """Whether the kernels run on this card (the capability gate, then the self-check once per process): (yes,
    the reason when not). Cached. Decides the GEMM path (the Triton GEMM on sm_80+, torch._int_mm below) and the
    elementwise kernels (Triton where it gives the torch ops' bits, else the torch ops) before the self-check."""
    if _state["ok"] is None:
        ok_mm, ok_triton, why = capability(device)
        if not ok_mm:
            _state["ok"], _state["why"] = False, why
        else:
            if ok_triton:
                _state["gemm"], _state["elementwise"] = "triton", "triton"
            else:
                _state["gemm"] = "cuda" if cuda_gemm_op() is not None else "intmm"
                _state["elementwise"] = ("triton" if _triton_elementwise(device) else
                                         "cuda" if _cuda_elementwise(device) else "torch")
            _state["ok"], _state["why"] = self_check(device)
            if not _state["ok"] and _state["gemm"] == "cuda":       # the package's kernel failed: per-row layers only
                print(f"[INT8] pixal3d_kernels' GEMM is left out ({_state['why']}); torch._int_mm serves the per-row "
                      f"layers", flush=True)
                _state["gemm"] = "intmm"
                _state["ok"], _state["why"] = self_check(device)
    return bool(_state["ok"]), str(_state["why"])


def enabled(x: torch.Tensor) -> bool:
    """Whether the int8 path serves x (CUDA, no autograd, capable card, self-check passed); else the fallback.
    (The switch PIXAL3D_INT8 is read by modules/int8.py, which converts layers only when it says so; a converted
    layer always runs the kernels where they can run.)"""
    if not x.is_cuda or torch.is_grad_enabled():
        return False
    ok, why = ready(x.device)
    if not ok and not _state["warned"]:
        _state["warned"] = True
        print(f"[INT8] the int8 linear path is off ({why}); F.linear serves", flush=True)
    return ok
