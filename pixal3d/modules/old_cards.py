"""The flows' element-wise chains on the cards below sm_80, each in one of pixal3d_kernels' CUDA kernels with the
bits of the eager torch code it replaces.

On Turing and Pascal the blocks run their eager code (modules/fused.py's Triton kernels are bf16-only), where every
norm, modulation, residual, q / k RMS norm and rotary embedding is a chain of 3 to 10 torch kernels in fp32 (and fp16
casts on Turing): about a third of an hr / tex pass on an RTX 2070. Here each chain is one kernel that makes the same
roundings in the same order (pixal3d_kernels' README, "The element-wise chains"; the q / k norm sums its squares in
torch's own reduction order), so the flows' numbers do not change:

    modulate(h, scale, shift, dtype)   _inner(h * (1 + scale) + shift, dtype)   (after a LayerNorm32)
    gated(x, h, gate)                  x + _stream(h, x) * gate                 (a gated residual)
    qk(attn, q, k, rope)               q_rms_norm(q), k_rms_norm(k) [then the rotary embedding] of a sparse
                                       attention; qk_dense(attn, q, k, phases) for the dense one
    quantize_rows(x, group, inv)       int8mm.quantize_rows_torch (int8mm's dispatch below sm_80)

Each returns None where it does not apply (the ampere profile, no kernels, a batch of several items, a layout the
kernel does not take, a kind the install's race left out, PIXAL3D_OLD_CARD_KERNELS=off): the caller runs its eager
code then. Before the first use in a process every kernel runs once on small tables against the eager chain
(`_check`): a kernel that gives other bits (another torch version that reduces in another order, say) turns them all
off, with one line printed.

Which kinds serve is the install's (calibrate.py's step "old_cards", `calibrate` below): each kernel timed against
its eager chain on this card at two of the flows' sizes, on only where it is RACE_MARGIN faster at both (on an RTX
2070 they took a fifth off a pass; on Pascal, which no test card had, an fp32 chain may be as fast as one kernel). No
record (or one of another card, profile or kernels build): every kind serves, as before the race. Either way the
numbers are the same; only the time differs. PIXAL3D_OLD_CARD_KERNELS = off: none; force: every kind, whatever the
race chose (the first-use check still decides).

The same step races the dp4a GEMM's twins on the cards that run it (Spellcast3D round O: int8mm.calibrate_dp4a_twins,
the section's "dp4a"; int8mm reads it through `dp4a_race`): the same bits, other code shapes, the faster one per
scale form recorded - in a child process (calibrate_dp4a_twins_in_child), so that a twin that faults costs only
its own race.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

from . import precision

_state: dict = {"ok": None, "why": "", "on": {}}
_ON_DTYPES = (torch.float16, torch.float32)
KINDS = ("quantize_rows", "modulate", "residual_gated", "qk_norm_rope")     # pixal3d_kernels' operators
RACE_SHAPES = ((4096, 1024), (20000, 1536))     # rows x channels: a 64^3 flow's stream, a detail flow's at 1536
RACE_MARGIN = 0.05                              # a kind serves only where its kernel is this much faster at each size
SECTION = "old_cards"                           # the calibration record's section


def _kernels():
    import pixal3d_kernels

    return pixal3d_kernels


def _mode() -> str:
    """PIXAL3D_OLD_CARD_KERNELS: "off", "force", or "" (the install's race decides; on, 1 and the like too)."""
    text = os.environ.get("PIXAL3D_OLD_CARD_KERNELS", "").strip().lower()
    if text in ("off", "0", "false", "no"):
        return "off"
    return "force" if text == "force" else ""


def _switch() -> bool:
    return _mode() != "off"


def ready(device: torch.device) -> bool:
    """Whether the kernels serve in this process (the profile, the package, the switch, then `_check` once)."""
    if _state["ok"] is None:
        _state["ok"] = False
        if precision.profile().name == "ampere" or not _switch():
            _state["why"] = "not used on this card" if _switch() else "PIXAL3D_OLD_CARD_KERNELS is off"
            return False
        try:
            pk = _kernels()
            missing = [name for name in KINDS if not pk.has(name)]
        except Exception as exc:        # noqa: BLE001 - no package, or one that does not load: the eager code
            _state["why"] = f"pixal3d_kernels does not load ({type(exc).__name__})"
            return False
        if missing:
            _state["why"] = f"this pixal3d_kernels has no {', '.join(missing)}"
            return False
        _state["ok"], _state["why"] = _check(device)
        if not _state["ok"]:
            print(f"[OLD-CARDS] the element-wise kernels are left out ({_state['why']}); the eager code serves",
                  flush=True)
        else:
            _state["on"] = chosen(device)
            left = [kind for kind in KINDS if not _state["on"][kind]]
            if left:
                print(f"[OLD-CARDS] the install's race keeps the eager code for {', '.join(left)} (its kernel was "
                      f"not {RACE_MARGIN:.0%} faster on this card)", flush=True)
    return bool(_state["ok"])


def chosen(device: Any = None) -> dict[str, bool]:
    """{kind: whether its kernel serves}: the install's race (the record's section, when it was made on this card,
    for this profile, with the installed kernels build), every kind where there is none or PIXAL3D_OLD_CARD_KERNELS
    is force."""
    every = dict.fromkeys(KINDS, True)
    if _mode() == "force":
        return every
    try:
        from .. import calibration

        found = calibration.section(SECTION, device)
    except Exception:       # noqa: BLE001 - no record: as before the race
        found = None
    if not race_holds(found):
        return every
    return {kind: bool(found["on"].get(kind, True)) for kind in KINDS}


def race_holds(section: Optional[dict]) -> bool:
    """Whether a record's section is a race this process may follow: one that ran (ok, with its choices), on this
    profile, with the installed pixal3d_kernels build (another build's timings say nothing of this one's)."""
    return (isinstance(section, dict) and section.get("ok") is True and isinstance(section.get("on"), dict)
            and section.get("profile") == precision.profile().name and section.get("kernels") == _kernels_build())


def dp4a_race(device: Any = None) -> Optional[dict]:
    """The install's race of the dp4a GEMM's twins (the record's old_cards.dp4a, int8mm.calibrate_dp4a_twins) where
    it ran on this card, for this profile, with the installed pixal3d_kernels build; None otherwise (int8mm then runs
    the shipped kernel)."""
    try:
        from .. import calibration

        found = calibration.section(SECTION, device)
    except Exception:       # noqa: BLE001 - no record
        return None
    if (not isinstance(found, dict) or found.get("profile") != precision.profile().name
            or found.get("kernels") != _kernels_build()):
        return None
    race = found.get("dp4a")
    return race if isinstance(race, dict) and race.get("ok") is True and not race.get("skipped") else None


def _kernels_build() -> str:
    try:
        from importlib import metadata

        return metadata.version("pixal3d_kernels")
    except Exception:       # noqa: BLE001
        return ""


def _usable(t: torch.Tensor, kind: str) -> bool:
    return t.is_cuda and not torch.is_grad_enabled() and ready(t.device) and _state["on"].get(kind, True)


def _feats(x: Any) -> torch.Tensor:
    return getattr(x, "feats", x)


def _one_item(x: Any) -> bool:
    """One batch item (the modulation is one row): a SparseTensor of one item, or a dense [1, ...] tensor."""
    return x.shape[0] == 1


# ---- the chains -------------------------------------------------------------------------------------------------------

def modulate(h: Any, scale: torch.Tensor, shift: torch.Tensor, dtype: Optional[torch.dtype]) -> Any:
    """_inner(h * (1 + scale) + shift, dtype) for h fp32 (a SparseTensor's features [T, C], or a dense [1, N, C]) and
    one row of scale / shift, or None."""
    f = _feats(h)
    if (f.dtype != torch.float32 or not _usable(f, "modulate") or not _one_item(h) or scale.numel() != f.shape[-1]
            or shift.numel() != f.shape[-1] or scale.dtype != shift.dtype or scale.dtype not in _ON_DTYPES
            or not f.is_contiguous()):
        return None
    out_dtype = dtype if dtype is not None else f.dtype
    if out_dtype not in _ON_DTYPES:
        return None
    y = _kernels().modulate(f, 1 + scale, shift, out_dtype)
    return h.replace(y) if hasattr(h, "replace") and hasattr(h, "feats") else y


def gated(x: Any, h: Any, gate: torch.Tensor) -> Any:
    """x + _stream(h, x) * gate for an fp32 stream x, h of x's shape (fp16 or fp32) and one row of gate, or None."""
    fx, fh = _feats(x), _feats(h)
    if (fx.dtype != torch.float32 or not _usable(fx, "residual_gated") or not _one_item(x) or fh.shape != fx.shape
            or fh.dtype not in _ON_DTYPES or gate.numel() != fx.shape[-1] or gate.dtype not in _ON_DTYPES
            or not fx.is_contiguous() or not fh.is_contiguous()):
        return None
    y = _kernels().residual_gated(fx, fh, gate)
    return x.replace(y) if hasattr(x, "replace") and hasattr(x, "feats") else y


def _norm_one(t: torch.Tensor, norm: Any, phases: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """norm(t) [then the rotary embedding] for t [T, H, 128] (the last dimension contiguous) in t's dtype."""
    gamma = norm.gamma
    if (t.dim() != 3 or t.shape[-1] != 128 or t.stride(-1) != 1 or t.dtype not in _ON_DTYPES
            or gamma.dtype not in _ON_DTYPES or tuple(gamma.shape) != (t.shape[1], 128)
            or (t.dtype == torch.float32 and gamma.dtype != torch.float32)):
        return None
    if phases is not None and (phases.dtype != torch.complex64 or tuple(phases.shape) != (t.shape[0], 64)):
        return None
    return _kernels().qk_norm_rope(t, gamma, norm.scale, 1e-12, phases, t.dtype)


def qk(attn: Any, q: Any, k: Any, rope: bool) -> Optional[tuple]:
    """A sparse attention's q_rms_norm(q), k_rms_norm(k) and, with `rope` where the attention has it, its rotary
    embedding of both (SparseTensors of [T, H, 128] features), or None. Without the rotary embedding (the cross
    attention, whose k comes from the context: a VarLenTensor's [N, H, 128] or a dense [1, L, H, 128]) a k the kernel
    does not take goes through its eager module."""
    fq = _feats(q)
    if not _usable(fq, "qk_norm_rope"):
        return None
    phases = attn.rope.phases(q) if rope and getattr(attn, "use_rope", False) else None
    nq = _norm_one(fq, attn.q_rms_norm, phases)
    if nq is None:
        return None
    wrap = lambda t, f: t.replace(f) if hasattr(t, "replace") and hasattr(t, "feats") else f
    fk = _feats(k)
    if phases is not None:
        nk = _norm_one(fk, attn.k_rms_norm, phases) if fk.dim() == 3 and fk.shape[0] == fq.shape[0] else None
        return None if nk is None else (wrap(q, nq), wrap(k, nk))
    if fk.dim() == 4 and fk.shape[0] == 1:
        nk = _norm_one(fk[0], attn.k_rms_norm, None)
        nk = nk.unsqueeze(0) if nk is not None else None
    else:
        nk = _norm_one(fk, attn.k_rms_norm, None)
    return wrap(q, nq), (wrap(k, nk) if nk is not None else attn.k_rms_norm(k))


def qk_dense(attn: Any, q: torch.Tensor, k: torch.Tensor, phases: Optional[torch.Tensor]) -> Optional[tuple]:
    """A dense attention's q_rms_norm(q), k_rms_norm(k) [then RotaryPositionEmbedder.apply_rotary_embedding with
    phases [L, 64]] for q, k [1, L, H, 128], or None."""
    if not _usable(q, "qk_norm_rope") or q.dim() != 4 or k.dim() != 4 or q.shape[0] != 1 or k.shape[0] != 1:
        return None
    nq = _norm_one(q[0], attn.q_rms_norm, phases)
    if nq is None:
        return None
    if phases is not None:
        nk = _norm_one(k[0], attn.k_rms_norm, phases) if k.shape[1] == q.shape[1] else None
        return None if nk is None else (nq.unsqueeze(0), nk.unsqueeze(0))
    nk = _norm_one(k[0], attn.k_rms_norm, None)
    return nq.unsqueeze(0), (nk.unsqueeze(0) if nk is not None else attn.k_rms_norm(k))


def quantize_rows(x: torch.Tensor, group: int = 0,
                  inv_smooth: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """int8mm.quantize_rows_torch's bits (int8mm's dispatch calls this below sm_80 once `quantiser_ok` said so)."""
    return _kernels().quantize_rows(x, group, inv_smooth.float() if inv_smooth is not None else None)


def quantiser_ok(device: torch.device) -> bool:
    """Whether int8mm may take this module's quantiser (`ready`, and the install's race kept it)."""
    return ready(device) and _state["on"].get("quantize_rows", True)


# ---- the first-use check ----------------------------------------------------------------------------------------------

def _check(device: torch.device) -> tuple[bool, str]:
    """Every kernel once on small tables of the flows' widths against the eager chain it replaces: (all equal bit
    for bit, the first that is not)."""
    from .int8mm import quantize_rows_torch
    from .sparse.attention.modules import SparseMultiHeadRMSNorm
    from .sparse.attention.rope import SparseRotaryPositionEmbedder

    try:
        pk = _kernels()
        g = torch.Generator(device=device).manual_seed(0)
        torso = precision.torso()
        sub = torso if torso in _ON_DTYPES else torch.float32
        T, C, H = 333, 1536, 12
        with torch.no_grad():
            x = torch.randn((T, C), device=device, generator=g) * 3
            x[::50] *= 40
            for group in (0, 128):
                for dt in (sub, torch.float32):
                    if any(not torch.equal(a, b) for a, b in zip(pk.quantize_rows(x.to(dt), group),
                                                                  quantize_rows_torch(x.to(dt), group))):
                        return False, f"the quantiser (group {group}, {dt})"
            h = torch.randn((T, C), device=device, generator=g)
            s = (torch.randn((1, C), device=device, generator=g) * 0.3).to(sub)
            sh = (torch.randn((1, C), device=device, generator=g) * 0.3).to(sub)
            want = h * (1 + s) + sh
            if not torch.equal(pk.modulate(h, 1 + s, sh, sub), want.to(sub) if sub != torch.float32 else want):
                return False, "the modulation"
            hs = torch.randn((T, C), device=device, generator=g).to(sub)
            if not torch.equal(pk.residual_gated(x, hs, s), x + hs.float() * s):
                return False, "the gated residual"
            norm = SparseMultiHeadRMSNorm(128, H).to(device)
            norm.gamma.data = torch.rand((H, 128), device=device, generator=g) + 0.5
            rope = SparseRotaryPositionEmbedder(128)
            qkv = (torch.randn((T, 3, H, 128), device=device, generator=g) * 4).to(sub)
            q = qkv[:, 0]
            coords = torch.randint(0, 64, (T * 3,), device=device, generator=g)
            ph = rope._get_phases(coords).reshape(T, -1)
            ph = torch.cat([ph, torch.ones((T, 64 - ph.shape[1]), dtype=ph.dtype, device=device)], 1)
            want = norm(q)
            if not torch.equal(pk.qk_norm_rope(q, norm.gamma, norm.scale, 1e-12, None, sub), want):
                return False, "the q / k RMS norm"
            if not torch.equal(pk.qk_norm_rope(q, norm.gamma, norm.scale, 1e-12, ph, sub),
                               rope._rotary_embedding(want, ph)):
                return False, "the rotary embedding"
        torch.cuda.synchronize(device)
    except Exception as exc:        # noqa: BLE001 - a kernel that cannot run here: the eager code
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
    return True, ""


# ---- the install's race -----------------------------------------------------------------------------------------------

def _forms(device: torch.device, rows: int, channels: int, g: torch.Generator) -> dict:
    """{kind: [(form, kernel call, eager call)]} on random tables of one size, in the profile's dtypes, as the blocks
    call them (the stream in fp32, the sub-layers in the torso's dtype; the quantiser at both scale layouts, the q / k
    norm with the rotary embedding of a self-attention)."""
    from .int8mm import GROUP, quantize_rows_torch
    from .sparse.attention.modules import SparseMultiHeadRMSNorm
    from .sparse.attention.rope import SparseRotaryPositionEmbedder

    pk = _kernels()
    torso = precision.torso()
    sub = torso if torso in _ON_DTYPES else torch.float32
    heads = channels // 128
    x = torch.randn((rows, channels), device=device, generator=g)
    xs = x.to(sub)
    h = torch.randn((rows, channels), device=device, generator=g).to(sub)
    s = (torch.randn((1, channels), device=device, generator=g) * 0.3).to(sub)
    sh = (torch.randn((1, channels), device=device, generator=g) * 0.3).to(sub)
    cast = (lambda t: t.to(sub)) if sub != torch.float32 else (lambda t: t)
    stream = (lambda t: t.float()) if sub != torch.float32 else (lambda t: t)
    norm = SparseMultiHeadRMSNorm(128, heads).to(device)
    norm.gamma.data = torch.rand((heads, 128), device=device, generator=g) + 0.5
    rope = SparseRotaryPositionEmbedder(128)
    q = (torch.randn((rows, 3, heads, 128), device=device, generator=g) * 4).to(sub)[:, 0]
    coords = torch.randint(0, 64, (rows * 3,), device=device, generator=g)
    ph = rope._get_phases(coords).reshape(rows, -1)
    ph = torch.cat([ph, torch.ones((rows, 64 - ph.shape[1]), dtype=ph.dtype, device=device)], 1)
    return {
        "quantize_rows": [(f"group {group}", lambda group=group: pk.quantize_rows(xs, group),
                           lambda group=group: quantize_rows_torch(xs, group)) for group in (0, GROUP)],
        "modulate": [("", lambda: pk.modulate(x, 1 + s, sh, sub), lambda: cast(x * (1 + s) + sh))],
        "residual_gated": [("", lambda: pk.residual_gated(x, h, s), lambda: x + stream(h) * s)],
        "qk_norm_rope": [("rope", lambda: pk.qk_norm_rope(q, norm.gamma, norm.scale, 1e-12, ph, sub),
                          lambda: rope._rotary_embedding(norm(q), ph))],
    }


def _median_ms(calls: dict, device: torch.device, reps: int, iters: int) -> dict:
    """{name: median ms of one call}: the calls timed in turn, the order reversed every other round (CUDA events)."""
    import statistics

    names = list(calls)
    ms: dict = {name: [] for name in names}
    for rep in range(reps):
        for name in (names if rep % 2 == 0 else names[::-1]):
            calls[name]()
            events = []
            for _ in range(iters):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                calls[name]()
                end.record()
                events.append((start, end))
            torch.cuda.synchronize(device)
            ms[name] += [a.elapsed_time(b) for a, b in events]
    return {name: statistics.median(values) for name, values in ms.items()}


def calibrate(device: torch.device, reps: int = 6, iters: int = 5) -> dict:
    """The install's race (calibrate.py's step "old_cards"): every kind's kernel against its eager chain at
    RACE_SHAPES, after the first-use check; a kind serves where its kernel is RACE_MARGIN faster in every form at
    every size. Before it, on a dp4a card, the dp4a GEMM's twins against its shipped kernel in a child process
    (int8mm.calibrate_dp4a_twins_in_child: "dp4a"; a twin that faults costs only its own race, never this process's
    CUDA context). Returns the record's section {"ok", "profile", "kernels", "on", "timings", "summary", "dp4a"}."""
    name = precision.profile().name
    if name == "ampere":
        return {"ok": True, "skipped": "the flows' element-wise chains run Triton's fused kernels on this card "
                                       "(ampere profile)"}
    from . import int8mm

    dp4a = int8mm.calibrate_dp4a_twins_in_child(device)
    torch.cuda.empty_cache()
    found: dict = {"ok": True, "profile": name, "kernels": _kernels_build(), "torch": torch.__version__,
                   "on": dict.fromkeys(KINDS, True), "timings": [], "dp4a": dp4a}
    ok, why = _check(device)
    if not ok:
        return {"ok": False, "profile": name, "kernels": found["kernels"], "dp4a": dp4a,
                "error": f"the kernels do not give the eager chains' bits here ({why})"}
    g = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for rows, channels in RACE_SHAPES:
            forms = _forms(device, rows, channels, g)
            for kind in KINDS:
                for form, kernel, eager in forms[kind]:
                    med = _median_ms({"kernel": kernel, "eager": eager}, device, reps, iters)
                    faster = med["kernel"] <= med["eager"] * (1 - RACE_MARGIN)
                    found["on"][kind] = found["on"][kind] and faster
                    found["timings"].append({"kind": kind, "form": form, "rows": rows, "channels": channels,
                                             "kernel_ms": round(med["kernel"], 4), "eager_ms": round(med["eager"], 4)})
            del forms
            torch.cuda.empty_cache()
    found["summary"] = "; ".join(
        f"{kind} {'on' if found['on'][kind] else 'off'} (" + ", ".join(
            f"{t['rows']:,}x{t['channels']}{' ' + t['form'] if t['form'] else ''} {t['kernel_ms']:.3f} vs "
            f"{t['eager_ms']:.3f} ms" for t in found["timings"] if t["kind"] == kind) + ")" for kind in KINDS)
    if dp4a.get("summary"):
        found["summary"] += f"; the dp4a GEMM: {dp4a['summary']}"
    return found
