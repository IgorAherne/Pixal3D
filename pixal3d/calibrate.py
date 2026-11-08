"""Calibrate Pixal3D for this PC's graphics card, once, at install: Spellcast3D's installer runs

    python -m pixal3d.calibrate --version <N> [--quiet-wait 900] [--record <file>]

as the last step of Pixal3D's install (and again only when the card changes or the installer asks for a new
version of it). What it measures is kept in the install's calibration record (pixal3d/calibration.py), so that no
process times or tunes anything while the artist works:

  1. The card is let go quiet first: another program using it (a render in Blender, a game, a video) would make
     every timing below wrong. NVML's utilisation is read every second (nvidia-smi); the steps start after five
     quiet seconds in a row (QUIET_UTIL %), and the card is looked at again before every convolution key: a run
     that finds it busy waits again, within one budget of --quiet-wait seconds for the whole run. A run that had to
     go on with a busy card is written "complete": false; it stands all the same (the installer never measures again
     by itself: Spellcast3D's Settings > Debugging has "Measure the graphics card again" for that).
  2. The Triton kernels the engine compiles at first use are compiled now, through their own self-checks (the
     fused elementwise kernels, the int8 GEMMs, the convolutions' row codes): the first 3D job of the install no
     longer waits for them (the compiled kernels stay in Triton's cache). On sm_80+ the int8 GEMMs' fc1 tile is chosen
     here too ("int8_tiles": candidates that give the same bits, timed). A record made before that choice existed
     runs this step again, alone where the installer asks for it (--only triton), its other sections standing.
  3. Which of the SageAttention wheel's two CUDA kernels is faster on this card (modules/sage.py `calibrate`; they
     give the same bits).
  4. The decoders' sparse convolutions' tile configurations for every size a prop can have
     (modules/sparse/conv/tuning.py `calibrate`, written into flex_gemm's autotune_cache.json): afterwards a size
     that was not calibrated takes the configuration of the nearest one that was, instead of being tuned while the
     artist waits (every configuration gives the same values; only the time differs).
  5. Where Hi3D's code is on this PC (code/hi3d beside the fork, or SPELLCAST_HI3D_CODE): spconv's kernel choice
     for every sparse convolution Hi3D runs (sparc3d.ops.spconv_tuning), which spconv otherwise times again in
     every process on the first prop's own input, and which could differ between sessions. Spellcast3D's installer
     leaves this step out (--skip spconv) while Hi3D's AI models are not on the PC, and their download runs it alone
     afterwards (--only spconv: the other sections of the record stand).
  6. Below sm_80, where the decoders' convolutions run spconv's own kernels (RTX 20xx): spconv's kernel choice for
     each of the decoders' convolutions (modules/sparse/conv/spconv_choices.py, the step "spconv_decoders"), which
     spconv otherwise times again at the first prop of every process, differently from one process to the next.
  7. Below sm_80 (RTX 20xx, GTX 10xx): pixal3d_kernels' element-wise kernels (modules/old_cards.py `calibrate`, the
     step "old_cards") each timed against the eager chain it stands for at two of the flows' sizes; a kind serves
     only where its kernel is 5 % faster at both (an fp32 chain on Pascal may be as fast as one kernel). They give
     the same bits either way. A record without the section runs this step alone (--only old_cards).

Nothing here changes a result: every choice is between kernels that give the same numbers (the convolutions' split-K
choice aside, which flex_gemm has always timed per install). A step that fails is recorded and the others go on; the
record is "complete": true only when every step passed on a quiet card, and the exit code is then 0 (1 when a step
failed or the card was busy: the installer says so and tries again at its next start; 2 when the card cannot be used
at all). The record is written once, at the end - an interrupted run leaves the previous record (or none). The card
is named by its UUID (torch's; the installer compares it with nvidia-smi's list, whatever the device order).
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

FORK = Path(__file__).resolve().parents[1]
# flex_gemm reads its cache path when it is imported (app.py sets the same before its imports).
os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(FORK / "autotune_cache.json"))
# The Triton kernels compiled here must land where the launcher's processes look for them: Spellcast3D's launcher
# (pixal3d_server.configure_env) forces the venv's folder, not %USERPROFILE%\.triton (Triton's default) nor a global
# TRITON_CACHE_DIR; forced here too, so the calibration's kernels are the ones the artist's first prop finds.
os.environ["TRITON_CACHE_DIR"] = str(Path(sys.prefix) / "triton-cache")

QUIET_UTIL = 15          # % of the card's time busy (NVML) that still counts as quiet
QUIET_SECONDS = 5        # quiet samples in a row before the steps start
# A quick look between keys first lets the run's own kernels drop out of NVML's sample window: utilization.gpu is
# the share of the LAST sample period (up to a second) that had a kernel running, so a sample taken right after a
# benchmark reads 100 % of the run's own work (the first install blamed itself and waited after every key). The
# card is looked at again at most every CHECK_EVERY seconds of work (a look costs SETTLE_SECONDS), and two busy
# samples in a row are needed before a quick look counts the card as busy.
SETTLE_SECONDS = 1.0
CHECK_EVERY = 15.0
SAY_EVERY = 30.0         # while waiting: one line this often
MAX_ATTEMPTS = 3         # (kept for the record's "attempts" count; the installer no longer retries by itself)
# Where the run's bar stands when each step begins (the installer shows it on its page: components._calibrate reads
# the "NN%" of a progress line).
SHARE = {"start": 0.0, "triton": 0.04, "sage": 0.14, "old_cards": 0.16, "convolutions": 0.18, "spconv_decoders": 0.18,
         "spconv": 0.95, "done": 1.0}       # (convolutions: flex_gemm's cards; old_cards, spconv_decoders: the others)


def say(text: str) -> None:
    print(f"[CALIBRATE] {text}", flush=True)


def progress(fraction: float, text: str) -> None:
    """A line the installer reads for its page: how far the run is (0..1) and what it does now."""
    print(f"[CALIBRATE] {int(round(max(0.0, min(1.0, fraction)) * 100)):3d}% {text}", flush=True)


def _nvidia_smi() -> Optional[str]:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (Path(r"C:\Windows\System32\nvidia-smi.exe"),
                      Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")):
        if candidate.exists():
            return str(candidate)
    return None


def _query(fields: str) -> Optional[list[str]]:
    """One nvidia-smi query of the first card (`fields` comma-separated), or None when it cannot be asked."""
    exe = _nvidia_smi()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    return [part.strip() for part in lines[0].split(",")] if out.returncode == 0 and lines else None


class Quiet:
    """The card watched through nvidia-smi: `wait()` before the steps and again before every convolution key, within
    one budget of `limit_seconds` for the whole run; a wait that runs out of budget lets the run go on and marks it
    (`busy_during`: the record is then not complete)."""

    def __init__(self, limit_seconds: float):
        self.limit = limit_seconds
        self.spent = 0.0
        self.checks = 0
        self.waits = 0              # times the card was found busy
        self.busy_during = False    # a wait ran out of budget: the steps went on beside another program
        self.unknown = False        # nvidia-smi could not be asked
        self.told = -SAY_EVERY
        self.looked: Optional[float] = None     # time.monotonic() of the last look (a quick one is skipped for CHECK_EVERY)

    def wait(self, quick: bool = False) -> bool:
        """Until the card has been at most QUIET_UTIL % busy for QUIET_SECONDS samples in a row. `quick` (between
        keys: the card is the run's own unless another program started): nothing within CHECK_EVERY seconds of the
        last look; else SETTLE_SECONDS for the run's own kernels to leave NVML's window, then one sample - quiet,
        done; busy twice in a row, a proper quiet spell is waited for. True when it was quiet, False when the budget
        ran out."""
        self.checks += 1
        if self.busy_during:
            return False                # the budget is spent: said once, the run goes on without more waiting
        if quick and self.looked is not None and time.monotonic() - self.looked < CHECK_EVERY:
            return True
        if quick:
            time.sleep(SETTLE_SECONDS)
        began = time.monotonic()
        calm, need, found_busy, strikes = 0, (1 if quick else QUIET_SECONDS), False, 0
        try:
            while True:
                got = _query("utilization.gpu")
                if got is None:
                    self.unknown = True
                    return True
                try:
                    util = int(float(got[0]))
                except ValueError:
                    util = 100
                if util <= QUIET_UTIL:
                    calm += 1
                    strikes = 0
                else:
                    calm = 0
                    strikes += 1
                    if quick and not found_busy and strikes < 2:
                        time.sleep(1.0)     # one busy sample right after the run's own work proves nothing
                        continue
                    need = QUIET_SECONDS
                    if not found_busy:
                        found_busy = True
                        self.waits += 1
                if calm >= need:
                    self.spent += time.monotonic() - began
                    return True
                waited = self.spent + time.monotonic() - began
                if waited >= self.limit:
                    self.spent = waited
                    self.busy_during = True
                    say(f"the graphics card is still busy after {waited / 60:.0f} minutes of waiting (last {util} %): "
                        f"going on anyway - this run is not kept as final, the installer tries again at its next start")
                    return False
                if util > QUIET_UTIL and waited - self.told >= SAY_EVERY:
                    self.told = waited
                    say(f"the graphics card is busy ({util} % in use): waiting for it to be free before measuring. "
                        f"Please close heavy programs (Blender, games, videos, Photoshop, a browser with many tabs).")
                time.sleep(1.0)
        finally:
            self.looked = time.monotonic()

    def record(self) -> dict:
        return {"quiet": not self.busy_during and not self.unknown, "waited_s": round(self.spent, 1),
                "checks": self.checks, "found_busy": self.waits, "busy_during": self.busy_during,
                "how": ("nvidia-smi could not be asked: not checked" if self.unknown else
                        "busy: gave up waiting" if self.busy_during else "quiet")}


def wait_until_quiet(limit_seconds: float) -> dict:
    """One wait (the old entry point): see Quiet."""
    quiet = Quiet(limit_seconds)
    quiet.wait()
    return quiet.record()


def _timed(name: str, step: Callable[[], Any], record: dict) -> Any:
    began = time.perf_counter()
    try:
        found = step()
    except Exception as exc:        # noqa: BLE001 - one step's failure is recorded; the others go on
        record[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "seconds": round(time.perf_counter() - began, 2)}
        say(f"{name}: FAILED ({type(exc).__name__}: {str(exc)[:200]}) - Pixal3D still works; that part is then "
            f"done at its first use")
        return None
    if isinstance(found, dict):
        found.setdefault("ok", True)
        found["seconds"] = round(time.perf_counter() - began, 2)
        record[name] = found
    return found


def _triton(device: Any) -> dict:
    """Compile the engine's Triton kernels: each family's self-check (which says whether it can run here), then the
    kernels as the flows and decoders call them - the int8 layers of the shipped policy at the flows' widths, the
    fused elementwise kernels, the row codes, NAF's RoPE, and Hi3D's (its int8 layers, fused chains and int8
    attention) where its code is on this PC - at a row count divisible by 16 and one that is not (Triton compiles
    a kernel again for each), so that the first 3D job finds them in the cache."""
    import torch

    done: dict[str, Any] = {}
    from pixal3d.modules import fused, int8mm, precision

    # the fused kernels serve bf16 torsos only (modules/fused.py active): nothing to compile for the others
    done["fused"] = bool(fused._check(device)) if precision.torso() == torch.bfloat16 else "not used: the torso is "         + precision.profile().torso_name
    ok, why = int8mm.ready(device)
    done["int8"] = bool(ok) if ok else f"off: {why}"
    if ok and int8mm.gemm_path() == "triton":        # fc1's tile for this card (the same bits whichever serves)
        try:
            done["int8_tiles"] = int8mm.calibrate_fc1_tiles(device)
        except Exception as exc:        # noqa: BLE001 - the shipped tile serves
            done["int8_tiles"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    from pixal3d.modules.sparse.conv import codes

    done["codes"] = bool(codes._usable(torch.zeros(1, device=device), 27))     # its self-check, on its own table
    torch.cuda.synchronize(device)
    warmed: dict[str, Any] = {}
    for name, step in (("int8_layers", _warm_int8), ("fused", _warm_fused), ("codes", _warm_codes),
                       ("rope", _warm_rope), ("hi3d", _warm_hi3d)):
        began = time.perf_counter()
        try:
            count = step(device)
            warmed[name] = {"calls": count, "seconds": round(time.perf_counter() - began, 2)}
        except Exception as exc:        # noqa: BLE001 - a kernel that cannot be warmed compiles at first use
            warmed[name] = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    done["warmed"] = warmed
    return done


WARM_ROWS = (4096, 4099)        # a row count divisible by 16 and one that is not: both of Triton's specialisations


def _warm_int8(device: Any) -> int:
    """The int8 layers of the shipped policy at the flows' widths (1536 channels, 12 x 128 heads, 8192 hidden,
    2048 of conditioning): Int8Linear / Int8MLP built from random nn.Linear layers as int8.convert builds them."""
    import torch
    from torch import nn

    from pixal3d.modules import int8, precision

    if int8.policy() is None or not int8.available(device):
        return 0
    dt = precision.torso()
    rules = int8.POLICIES[int8.policy()].get("hr") or {}
    shapes = {"qkv": (1536, 4608), "out": (1536, 1536), "xq": (1536, 1536), "xout": (1536, 1536),
              "proj": (2048, 1536)}
    calls = 0
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(0)
        for name, (k, n) in shapes.items():
            how = rules.get(name)
            if how is None or how in ("cs",):           # a calibration vector it has not got: the engine's case
                continue
            layer = nn.Linear(k, n).to(device=device, dtype=dt)
            q = int8.Int8Linear(layer, how)
            for rows in WARM_ROWS:
                q(torch.randn(rows, k, device=device, dtype=dt, generator=g))
                calls += 1
        how = rules.get("fc1")
        if how is not None and how != "cs":
            fc1 = nn.Linear(1536, 8192).to(device=device, dtype=dt)
            fc2 = nn.Linear(8192, 1536).to(device=device, dtype=dt)
            mlp = int8.Int8MLP(fc1, fc2, how, rules.get("fc2") if rules.get("fc2") != "cs" else how)
            for rows in WARM_ROWS:
                mlp(torch.randn(rows, 1536, device=device, dtype=dt, generator=g))
                calls += 1
    return calls


def _warm_hi3d(device: Any) -> int:
    """Hi3D's Triton kernels at its transformer's widths (2112 channels, 22 x 96 heads, 8448 hidden): the int8
    layers of the shipped policy on Pixal3D's int8mm, sparc3d's fused norm / rope chains and its int8 attention,
    each through its own self-check first; nothing where Hi3D's code is not on this PC."""
    code = _hi3d_code()
    if code is None:
        return 0
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import torch
    from torch import nn

    from pixal3d.modules import int8, int8mm
    from sparc3d.nn import fused as sparc_fused
    from sparc3d.nn import int8_linear
    from sparc3d.nn.sparse.attention import int8_attention

    calls = 0
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(0)
        if int8.policy() is not None and int8.available(device):
            int8_linear.install(int8mm)
            rules = int8_linear.POLICIES[int8_linear.DEFAULT_POLICY]
            layers = {"qkv": (2112, 6336), "out": (2112, 2112)}
            for name, (k, n) in layers.items():
                q = int8_linear.Int8Linear(nn.Linear(k, n).to(device=device, dtype=torch.float16), rules[name])
                for rows in WARM_ROWS:
                    q(torch.randn(rows, k, device=device, dtype=torch.float16, generator=g))
                    calls += 1
            mlp = int8_linear.Int8FeedForward(nn.Linear(2112, 8448).to(device=device, dtype=torch.float16),
                                              nn.Linear(8448, 2112).to(device=device, dtype=torch.float16),
                                              rules["mlp"])
            for rows in WARM_ROWS:
                mlp(torch.randn(rows, 2112, device=device, dtype=torch.float16, generator=g))
                calls += 1
        if sparc_fused.check(device)[0]:
            weight = torch.ones(2112, device=device, dtype=torch.float32)
            for rows in WARM_ROWS:
                x = torch.randn(rows, 2112, device=device, dtype=torch.float16, generator=g)
                sparc_fused.norm_modulate(x, 1e-6, scale1=x[0], shift=x[1])
                sparc_fused.norm_modulate(x, 1e-6, weight=weight, bias=weight)
                qkv = torch.randn(rows, 3, 22, 96, device=device, dtype=torch.float16, generator=g)
                angles = torch.rand(rows, 48, device=device, generator=g) * 6.28
                gamma = torch.ones(22, 96, device=device, dtype=torch.float32)
                sparc_fused.rope_rms_(qkv, torch.polar(torch.ones_like(angles), angles), gamma, gamma, 96 ** 0.5)
                calls += 3
        if torch.cuda.get_device_capability(device) >= (8, 0) and int8_attention.self_check(device)[0]:
            for rows in WARM_ROWS:
                qkv = torch.randn(rows, 3, 22, 96, device=device, dtype=torch.float16, generator=g)
                q, k, v = qkv.unbind(1)
                int8_attention.attention(q, k, v)
                calls += 1
    torch.cuda.synchronize(device)
    return calls


def _warm_fused(device: Any) -> int:
    """The fused elementwise kernels at the flows' widths, with and without the modulation and the gate."""
    import torch

    from pixal3d.modules import fused

    calls = 0
    with torch.no_grad():       # the fused path serves inference only: active() says no with autograd on
        if not fused.active(torch.zeros(1, 1536, device=device, dtype=torch.bfloat16)):
            return 0
        g = torch.Generator(device=device).manual_seed(0)
        for rows in WARM_ROWS:
            x = torch.randn(rows, 1536, device=device, dtype=torch.bfloat16, generator=g)
            h = torch.randn(rows, 1536, device=device, dtype=torch.bfloat16, generator=g)
            scale = torch.randn(1, 1536, device=device, dtype=torch.bfloat16, generator=g)
            shift = torch.randn(1, 1536, device=device, dtype=torch.bfloat16, generator=g)
            weight = torch.ones(1536, device=device, dtype=torch.float32)
            fused.norm_modulate(x, 1e-6, scale=scale, shift=shift)
            fused.norm_modulate(x, 1e-6, weight=weight, bias=weight)
            fused.residual(x, h, gate=scale)
            fused.residual(x, h, second=h)
            fused.residual(x, h)
            qk = torch.randn(rows, 12, 128, device=device, dtype=torch.bfloat16, generator=g)
            gamma = torch.ones(128, device=device, dtype=torch.bfloat16)
            fused.qk_norm_rope(qk, gamma, 128 ** -0.5)
            calls += 6
    from pixal3d.modules import int8mm

    return calls + int8mm.warm_handoffs(device, WARM_ROWS)      # the blocks' norm -> int8 / residual hand-offs


def _warm_codes(device: Any) -> int:
    import torch

    from pixal3d.modules.sparse.conv import codes

    calls = 0
    g = torch.Generator(device=device).manual_seed(0)
    for rows in WARM_ROWS:
        nmap = torch.randint(-1, 1 << 20, (rows, 27), dtype=torch.int32, device=device, generator=g)
        codes.codes(nmap.view(torch.uint32), 27)
        calls += 1
    return calls


def _warm_rope(device: Any) -> int:
    """NAF's RoPE kernel on a guide map of the texture conditioning's size (1024 x 1024 at 1536; 512 x 512 too),
    through the module's own entry (the eager rope_apply serves if the kernel cannot run)."""
    import torch

    from pixal3d.modules import naf_sliced

    if not naf_sliced.FUSED_ROPE:
        return 0
    warm = getattr(naf_sliced, "warm_rope", None)
    if warm is None:
        return 0
    calls = 0
    with torch.no_grad():
        for side in (512, 1024):
            calls += int(bool(warm(device, side)))
    return calls


def _sage(device: Any) -> dict:
    from pixal3d.modules import sage

    if not sage.from_env():
        return {"tuned": False, "why": "PIXAL3D_SAGE is off", "summary": "off"}
    return sage.calibrate(device)


def _convolutions(device: Any, quiet: Quiet, grid: Optional[dict] = None) -> dict:
    """The convolution step: every shape of calibrate_conv.GRID (this card's earlier choices dropped), or `grid`
    alone with the earlier choices kept."""
    from pixal3d.modules.sparse.conv import calibrate_conv, tuning

    calibrate = getattr(tuning, "calibrate", None)
    if calibrate is None:
        return {"ok": False, "why": "this build has no convolution calibration"}
    shapes = max(1, len(grid if grid is not None else getattr(calibrate_conv, "GRID", {})) or 1)
    done = [0]

    def tell(text: str) -> None:
        """The step's own lines; one per finished shape ("convolutions 1024x1024: 9 sizes ...") moves the bar."""
        if text.startswith("convolutions ") and " sizes (" in text:
            done[0] += 1
            progress(SHARE["convolutions"] + (SHARE["spconv"] - SHARE["convolutions"]) * min(done[0], shapes) / shapes,
                     f"{text} - shape {done[0]} of {shapes}")
        else:
            say(text)
    if grid is None:
        return calibrate(device, say=tell, pause=lambda: quiet.wait(quick=True))
    return calibrate(device, say=tell, pause=lambda: quiet.wait(quick=True), grid=grid, drop=False)


def _conv_backend(device: Any) -> dict:
    """Below sm_80: whether flex_gemm's Triton convolutions give spconv's answers on this card (the record's
    "conv_backend" section: {"backend", "rel", "<backend>_ms", "why"}; modules/sparse/conv/backend_check.py)."""
    from pixal3d.modules.sparse.conv import backend_check

    return backend_check.compare(device)


def _hi3d_code() -> Optional[Path]:
    """Hi3D's checkout (sparc3d/ inside), as Spellcast3D's launcher finds it, or None."""
    code = Path(os.environ.get("SPELLCAST_HI3D_CODE") or FORK.parent / "hi3d")
    return code if (code / "sparc3d" / "__init__.py").is_file() else None


def _spconv(device: Any, quiet: Quiet, decoders: Optional[dict] = None) -> dict:
    """The spconv step: spconv's kernel choice for every sparse convolution Hi3D runs, measured on random points
    (sparc3d.ops.spconv_tuning.calibrate); skipped where Hi3D's code is not on this PC."""
    code = _hi3d_code()
    if code is None:
        return {"ok": True, "skipped": "Hi3D's code is not on this PC", "entries": []}
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    from sparc3d.ops import spconv_tuning

    from pixal3d.modules.sparse.conv import spconv_choices

    # a key the decoders use too keeps the decoders' choice (both share spconv's cache in one process)
    spconv_choices.restore_section(decoders)
    return spconv_tuning.calibrate(device, say=say, pause=lambda: quiet.wait(quick=True))


def _shapes_lacking(earlier: dict, record: dict) -> Optional[dict]:
    """The shapes of calibrate_conv.GRID a complete earlier record of this card does not hold, {} when it holds
    them all, None when there is no such record (a first calibration, another card, a failed convolution step).
    A new version of the calibration that adds shapes measures those alone: the choices Pixal3D's own decoders
    run with stand, and so do their bits."""
    from pixal3d.modules.sparse.conv import calibrate_conv

    if not earlier.get("complete") or not (earlier.get("flex_gemm") or {}).get("ok"):
        return None
    if (earlier.get("card_uuid") or earlier.get("card")) != (record.get("card_uuid") or record.get("card")):
        return None
    held = set(earlier["flex_gemm"].get("grid") or {})
    return {shape: span for shape, span in calibrate_conv.GRID.items() if calibrate_conv.family(*shape) not in held}


def _triton_stands(earlier: dict, device: Any) -> bool:
    """Whether the earlier record's Triton section holds what this build's Triton step records: where the card runs
    the Triton int8 GEMMs (sm_80+) and they passed their self-check then, fc1's tile choice (int8_tiles) made with
    this Triton (int8mm.fc1_group_cfg reads a choice of another Triton as none: two tiles proven to give the same
    bits under one Triton are not proven so under another). A section without it is made again, alone
    (Spellcast3D's installer runs --only triton for it): the step's kernels compile into Triton's cache again (those
    there are found), the tiles are timed once. A section whose int8 GEMMs did not run, or whose tile timing failed,
    stands."""
    section = earlier.get("triton") or {}
    if section.get("int8") is not True:
        return True
    tiles = section.get("int8_tiles")
    if tiles is not None:
        if not (isinstance(tiles, dict) and tiles.get("ok") is True):
            return True             # a timing that failed is not tried again by itself
        try:
            import triton

            if tiles.get("triton") == triton.__version__:
                from pixal3d.modules import int8mm

                return int8mm.group_cuda_stands(tiles)      # and the group GEMMs' race, made with this kernels build
        except Exception:           # noqa: BLE001 - no Triton to time with: the section stands
            return True
    from pixal3d.modules import int8mm

    return not int8mm.capability(device)[1]


def _sage_stands(earlier: dict) -> bool:
    """Whether the earlier record's attention section was made with the installed SageAttention wheel (the fork
    reads the section only then: modules/sage.py `_stored`); a section of another wheel is measured again."""
    from pixal3d.modules import sage

    section = earlier.get("sage") or {}
    return bool(section) and section.get("wheel") == sage._wheel_version()


def _merged(earlier: dict, added: dict) -> dict:
    """The earlier convolution section with the added shapes' keys and grid."""
    old = earlier["flex_gemm"]
    return {**old, **added, "keys": old.get("keys", []) + added.get("keys", []),
            "grid": {**(old.get("grid") or {}), **(added.get("grid") or {})},
            "skipped": old.get("skipped", []) + added.get("skipped", []),
            "dropped": old.get("dropped", {}), "seconds": old.get("seconds", 0) + added.get("seconds", 0),
            "summary": f"{old.get('summary', '')}; added {added.get('summary', '')}"}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate Pixal3D for this PC's graphics card (once, at install).")
    ap.add_argument("--version", type=int, default=1, help="the calibration's version (the installer's)")
    ap.add_argument("--record", default="", help="the record file (default: pixal3d/calibration.py's)")
    ap.add_argument("--quiet-wait", type=float, default=900.0, help="seconds to wait for a quiet card at most")
    ap.add_argument("--skip", default="",
                    help="comma-separated steps to leave out: triton, sage, old_cards, convolutions, spconv_decoders, "
                         "spconv")
    ap.add_argument("--only", default="",
                    help="comma-separated steps to run alone, the earlier record's other sections standing (spconv "
                         "once Hi3D's models arrived, say)")
    args = ap.parse_args(argv)
    if args.record:
        os.environ["PIXAL3D_CALIBRATION_FILE"] = args.record
    steps = {"triton": "triton", "sage": "sage", "old_cards": "old_cards", "convolutions": "flex_gemm",
             "spconv_decoders": "spconv_decoders", "spconv": "spconv"}  # step -> section
    skip = {part.strip() for part in args.skip.split(",") if part.strip()}
    only = {part.strip() for part in args.only.split(",") if part.strip()}
    if only:
        skip |= set(steps) - only

    progress(SHARE["start"], "checking that the graphics card is free (close heavy programs: Blender, games, videos, "
                              "Photoshop)")
    quiet = Quiet(args.quiet_wait)
    quiet.wait()

    import torch

    from pixal3d import calibration

    if not torch.cuda.is_available():
        say("PyTorch cannot use the graphics card: nothing calibrated")
        return 2
    device = torch.device("cuda", torch.cuda.current_device())
    driver = _query("driver_version")
    earlier = calibration.load() or {}
    attempts = int(earlier.get("attempts") or 0) + 1 if not earlier.get("complete") else 1
    record: dict[str, Any] = {
        "version": args.version, "complete": False, "attempts": attempts,
        "card": torch.cuda.get_device_name(device),
        "card_uuid": _card_uuid(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "driver": driver[0] if driver else None,
        "made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "torch": torch.__version__, "python": sys.version.split()[0],
    }
    say(f"calibrating for {record['card']} (sm_{record['capability'][0]}{record['capability'][1]})"
        + (f", attempt {attempts}" if attempts > 1 else ""))
    lacking = _shapes_lacking(earlier, record)
    if lacking is not None:         # the record stands; the shapes it lacks are added, the other steps kept
        if _triton_stands(earlier, device) and "triton" not in only:
            skip |= {"triton"}
            record["triton"] = earlier.get("triton")
        elif "triton" not in skip:
            say("the earlier calibration has no tile choice for the int8 GEMMs with this Triton: the Triton step "
                "runs again")
        if _sage_stands(earlier):   # the attention timings belong to the wheel they were made with
            skip |= {"sage"}
            record["sage"] = earlier["sage"]
        say(f"the earlier calibration stands: {len(lacking)} new convolution shapes are measured"
            if lacking else "the earlier calibration stands: no new convolution shapes to measure")
        if "sage" not in skip:
            say("the SageAttention wheel changed since: its kernels are timed again")
    began = time.monotonic()
    from pixal3d.modules import precision

    card = precision.profile()
    record["precision"] = card.name
    if "triton" not in skip and not card.triton:
        record["triton"] = {"ok": True, "skipped": f"no Triton on this card ({card.name} profile)"}
        skip |= {"triton"}
    if "convolutions" not in skip and card.conv != "flex_gemm":
        # Below sm_80 the decoders run on spconv. flex_gemm's Triton kernels give spconv's answers on an RTX 2060
        # Super, but its tile tuning (the step below) took 25-95 minutes a shape there (13 shapes: hours of an
        # install; Spellcast3D round L), so a card tries them only on request (PIXAL3D_TURING_FLEX_GEMM=on,
        # developers; conv/backend_check.py, the record's "conv_backend", which the precision profile reads)
        chosen = None
        if card.triton and os.environ.get("PIXAL3D_TURING_FLEX_GEMM", "").strip().lower() in ("1", "on", "true", "yes"):
            progress(SHARE["triton"], "trying flex_gemm's convolution kernels on this card")
            chosen = _timed("conv_backend", lambda: _conv_backend(device), record)
            if chosen:
                say(f"convolutions: {chosen.get('backend')} ({chosen.get('why', '')})")
        if chosen and chosen.get("backend") == "flex_gemm":
            from pixal3d.modules.sparse import config as sparse_config

            sparse_config.set_conv_backend("flex_gemm")        # the tuning step below works on flex_gemm's kernels
        else:
            record["flex_gemm"] = {"ok": True, "skipped": f"the sparse convolutions run on {card.conv} on this card"}
            skip |= {"convolutions"}
    if "triton" not in skip:
        progress(SHARE["triton"], "compiling Pixal3D's GPU kernels (step 1 of 4)")
        _timed("triton", lambda: _triton(device), record)
    if "sage" not in skip:
        progress(SHARE["sage"], "timing the attention kernels (step 2 of 4)")
        quiet.wait(quick=True)
        found = _timed("sage", lambda: _sage(device), record)
        if found and found.get("candidate"):       # a card below sm_80: its one kernel, tried in a child process
            say(f"attention: the {found['candidate']} kernel {'serves' if found.get('candidate_ok') else 'is not used'} "
                f"({found.get('why', '')})")
        elif found:
            say(f"attention: the {'tuned' if found.get('tuned') else 'shipped'} kernel ({found.get('summary', '')})")
    if "old_cards" not in skip:
        from pixal3d.modules import old_cards

        held = earlier.get("old_cards") or {}
        if lacking is not None and old_cards.race_holds(held):
            record["old_cards"] = held                  # the earlier record's race stands
        else:                                           # (on the ampere profile: "skipped", nothing timed)
            if card.name != "ampere":
                progress(SHARE["old_cards"], "timing the element-wise kernels for this card")
                quiet.wait(quick=True)
            found = _timed("old_cards", lambda: old_cards.calibrate(device), record)
            if found and not found.get("skipped"):
                say(f"element-wise kernels: {found.get('summary', '')}")
    if "convolutions" not in skip:
        progress(SHARE["convolutions"], "tuning the 3D decoders' convolutions for this card (step 3 of 4, the longest: "
                                        "several minutes)")
        if lacking is not None and not lacking:
            record["flex_gemm"] = earlier["flex_gemm"]
        else:
            found = _timed("flex_gemm", lambda: _convolutions(device, quiet, grid=lacking), record)
            if found and lacking:
                record["flex_gemm"] = _merged(earlier, record["flex_gemm"])
            if found:
                say(f"convolutions: {record['flex_gemm'].get('summary', found)}")
    if "spconv_decoders" not in skip:
        from pixal3d.modules.sparse.conv import spconv_choices

        held = earlier.get("spconv_decoders") or {}
        if lacking is not None and held.get("ok") and (held.get("entries") or held.get("skipped")):
            record["spconv_decoders"] = held            # the earlier record's choices stand
        elif not spconv_choices.applies():
            record["spconv_decoders"] = {"ok": True, "skipped": f"the decoders do not run spconv's own kernels on "
                                                                f"this card ({card.name} profile)", "entries": []}
        else:
            progress(SHARE["spconv_decoders"], "choosing spconv's kernels for the 3D decoders (step 3 of 4)")
            quiet.wait(quick=True)
            found = _timed("spconv_decoders", lambda: spconv_choices.calibrate(
                device, say=say, pause=lambda: quiet.wait(quick=True),
                others=calibration.section("spconv", device)), record)
            if found:
                say(f"spconv (decoders): {found.get('summary') or found.get('skipped')}")
    if "spconv" not in skip:
        if lacking is not None and (earlier.get("spconv") or {}).get("ok") and earlier["spconv"].get("entries"):
            record["spconv"] = earlier["spconv"]        # the earlier record's choices stand
        else:
            progress(SHARE["spconv"], "choosing spconv's kernels for Hi3D's convolutions (step 4 of 4)")
            quiet.wait(quick=True)
            decoders = record.get("spconv_decoders") or calibration.section("spconv_decoders", device)
            found = _timed("spconv", lambda: _spconv(device, quiet, decoders), record)
            if found:
                say(f"spconv: {found.get('summary') or found.get('skipped')}")
    for step, key in steps.items():     # a step left out keeps the earlier record's section (--only: all the others)
        if step in skip and key not in record and key in earlier:
            record[key] = earlier[key]
    record["seconds"] = round(time.monotonic() - began, 1)
    record["quiet"] = quiet.record()
    failed = [key for step, key in steps.items() if step not in skip and not (record.get(key) or {}).get("ok", False)]
    record["failed"] = failed
    record["complete"] = not failed and not quiet.busy_during and (not only or bool(earlier.get("complete", True)))
    where = calibration.save(record)
    if record["complete"]:
        progress(SHARE["done"], f"done in {record['seconds'] / 60:.1f} minutes; kept in {where}")
        return 0
    why = (f"the card was busy with another program during the run" if quiet.busy_during else "") \
        + (f"{'; ' if quiet.busy_during and failed else ''}{', '.join(failed)} failed" if failed else "")
    say(f"done in {record['seconds'] / 60:.1f} minutes, with a reservation ({why}); the record stands as it is (kept in "
        f"{where}). To measure again with the card free, use Settings > Debugging > Measure the graphics card again")
    return 1


def _card_uuid(device: Any) -> Optional[str]:
    """The card's UUID as nvidia-smi spells it (GPU-xxxxxxxx-...), from torch; None when torch has none."""
    try:
        import torch

        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        return f"GPU-{uuid}" if uuid is not None else None
    except Exception:       # noqa: BLE001
        return None


if __name__ == "__main__":
    sys.exit(main())
