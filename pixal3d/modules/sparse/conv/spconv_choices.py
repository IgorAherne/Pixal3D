"""spconv's kernel choices for the decoders' convolutions, measured once at install and put back in every process.

Below sm_80 the decoders' sparse convolutions run on spconv (modules/precision.py). spconv picks the implicit-GEMM
kernel of a convolution by timing every candidate on the first call of each key (the input, weight and output
dtypes, the output and input channels, the card): on an RTX 2070 that cost 25 s of the first 1024 prop of every
process (the shape decoder's first upsample 10.8 s, its first decode 14.1 s; 54 s for a batch of large props), was
timed on whatever the first prop brought (the upsample's tables, a quarter of the decode's), and chose differently
from one process to the next: another kernel sums in another order, and the same prop gave other meshes (other
voxel counts too) in another process. Here the choices are timed once at install on a quiet card, at the sizes of
a decode at 1024 (`calibrate`, the calibration's "spconv_decoders" section), and put back into spconv's cache
before the first convolution of a process (`restore`, called by conv_spconv at its first call): no convolution
is timed while the artist works, and every process runs the same kernels.

    record = calibrate(device)      # {"ok", "arch", "entries": [{"key", "algo", "splitk"}], "shapes", ...}
    restored = restore(device)      # the number of choices put back from this install's record (0: none)

A choice names one of spconv's prebuilt kernels by its descriptor's string; spconv 2.3's C++ tuner keeps its
choices where Python cannot read or set them, so both functions switch the convolutions to spconv's Python path
(ops.SPCONV_CPP_GEMM = False: the same kernels, the choices in spconv.algo.CONV's cache) - the mechanism of Hi3D's
sparc3d/ops/spconv_tuning.py, whose record (the "spconv" section) shares that cache. The one key the two share
gets one choice: whichever step calibrates second puts the other's choices back first (`restore_section`).
Cards whose spconv has no prebuilt kernels (sm_61: the products run on torch.mm, conv_spconv.py) and
cards on flex_gemm need nothing here.

The candidates are those of the C++ tuner, which the decoders ran with before: fp16 and fp32 accumulation alike
(the C++ tuner chose fp16-accumulating kernels for every one of the decoders' convolutions on an RTX 2070), so the
fastest wins as it did. spconv's Python path would leave out the fp16-accumulating ones for an input of more than
128 channels (`fp32_accum` None): that is a precision change of the 256- to 1024-channel convolutions, measured
(FP32_ACCUM below) but not taken.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

import torch

SECTION = "spconv_decoders"
HI3D_SECTION = "spconv"
# The shape and texture decoders (shape_dec_next_dc_f16c32_fp16 / tex_dec_next_dc_f16c32_fp16): channels per level,
# ConvNeXt blocks per level (one C -> C convolution each), and between levels a channel-to-space block whose first
# convolution makes C_next * 8 channels on the coarser level and whose second runs C_next -> C_next on the finer one.
CHANNELS = (1024, 512, 256, 128, 64)
BLOCKS = (4, 16, 8, 4, 0)
# Voxels per level the calibration times on: a prop of about 2.5 M fine voxels at 1024 (the boy single-view 1.7 M,
# multi-view props 1.2-5.1 M on an RTX 2070); the upsample runs the same keys on tables a quarter of the size.
ROWS = (8_000, 35_000, 150_000, 620_000, 2_500_000)
# The candidates' accumulation: False = fp16 or fp32, whichever is faster (the C++ tuner's set, the decoders' numbers
# as before); None = spconv's Python rule (fp32 only above 128 input channels: another precision, not taken)
FP32_ACCUM: Optional[bool] = False


def shapes(channels=CHANNELS, blocks=BLOCKS, rows=ROWS) -> list[tuple[int, int, int]]:
    """(input channels, output channels, voxels) of every distinct convolution of the decoders, each at the largest
    level it runs on."""
    found: dict[tuple[int, int], int] = {}

    def add(ci: int, co: int, n: int) -> None:
        found[(ci, co)] = max(found.get((ci, co), 0), n)

    for level, c in enumerate(channels):
        if blocks[level]:
            add(c, c, rows[level])
        if level + 1 < len(channels):
            nxt = channels[level + 1]
            add(c, nxt * 8, rows[level])
            add(nxt, nxt, rows[level + 1])
    return [(ci, co, n) for (ci, co), n in found.items()]


def applies() -> bool:
    """Whether this process's decoders run spconv's own GEMM kernels (whose choice is timed): the spconv backend on
    a card spconv ships kernels for."""
    from ... import precision

    profile = precision.profile()
    if profile.conv != "spconv" or profile.capability is None:
        return False
    try:
        from spconv.core_cc.cumm.common import CompileInfo

        return bool(CompileInfo.arch_is_compiled_gemm(tuple(profile.capability)))
    except Exception:       # noqa: BLE001 - no spconv: nothing to choose
        return False


def _tuner():
    from spconv.algo import CONV

    return CONV


def use_python_tuner() -> None:
    """spconv's Python convolution path from now on (its implicit_gemm reads the flag at every call)."""
    from spconv.pytorch import ops

    ops.SPCONV_CPP_GEMM = False


def _arch() -> tuple[int, int]:
    from spconv.pytorch.cppcore import get_arch

    return tuple(int(v) for v in get_arch())


def _entries(cache: dict, keys: Optional[set] = None) -> list[dict]:
    return [{"key": list(key), "algo": str(res.algo_desp), "splitk": int(res.splitk)} for key, res in cache.items()
            if keys is None or key in keys]


def _shell(rows: int, channels: int, dtype: torch.dtype, device: torch.device, seed: int = 0):
    """About `rows` voxels of a spherical shell two voxels thick (a decoder's tables are surfaces: 9-13 neighbours
    a voxel, rows in scan order) with random features, as a spconv SparseConvTensor."""
    import spconv.pytorch as spconv

    radius = max(4.0, 1.03 * math.sqrt(rows / (8.0 * math.pi)))
    side = int(2 * radius + 6)
    axis = torch.arange(side, device=device, dtype=torch.float32) - side / 2
    out = []
    for x in range(side):            # a slab at a time: the full grid of the largest shell is 360^3 floats
        yy, zz = torch.meshgrid(axis, axis, indexing="ij")
        r = torch.sqrt(axis[x] ** 2 + yy ** 2 + zz ** 2)
        hit = ((r >= radius - 1.0) & (r < radius + 1.0)).nonzero().to(torch.int32)
        if hit.shape[0]:
            out.append(torch.cat([torch.zeros_like(hit[:, :1]), torch.full_like(hit[:, :1], x), hit], 1))
    coords = torch.cat(out)[:rows].contiguous()
    g = torch.Generator(device="cpu").manual_seed(seed)
    feats = torch.randn((coords.shape[0], channels), generator=g).to(device=device, dtype=dtype)
    return spconv.SparseConvTensor(feats, coords, [side, side, side], 1)


def restore_section(record: Optional[dict]) -> int:
    """Put a record section's choices into spconv's cache (those not there yet): the number put back. Nothing for
    no section, one made on another card architecture, or no spconv."""
    if not record or not record.get("ok") or not record.get("entries"):
        return 0
    try:
        from spconv.algo import BestConvAlgoByProfile

        tuner = _tuner()
        arch = _arch()
    except Exception:       # noqa: BLE001 - no spconv, no card
        return 0
    if list(record.get("arch") or []) != list(arch):
        return 0
    use_python_tuner()
    by_name = {str(desp): desp for desp in tuner.all_desps}
    restored = 0
    with tuner.lock:
        for entry in record["entries"]:
            desp = by_name.get(entry.get("algo"))
            key = entry.get("key")
            if desp is None or not isinstance(key, list) or len(key) != 9:
                continue
            key = tuple(bool(v) if i == 8 else int(v) for i, v in enumerate(key))
            if tuple(key[5:7]) != tuple(arch) or key in tuner.kc_forward_cache:
                continue
            tuner.kc_forward_cache[key] = BestConvAlgoByProfile(desp, arch, splitk=int(entry.get("splitk", 1)))
            restored += 1
    return restored


_restored: Optional[int] = None


def restore(device: Any = None) -> int:
    """Once a process: this install's choices for the decoders (and Hi3D's, the same cache) put back into spconv's
    cache; one line says how many. 0 where nothing applies or no record holds them (spconv then times its kernels
    at first use, as it always did)."""
    global _restored
    if _restored is not None:
        return _restored
    _restored = 0
    if not applies():
        return 0
    try:
        from .... import calibration

        mine = calibration.section(SECTION, device)
        _restored = restore_section(mine)
        if mine and mine.get("ok"):
            _restored += restore_section(calibration.section(HI3D_SECTION, device))
    except Exception as exc:        # noqa: BLE001 - a record that cannot be read: spconv times its kernels
        print(f"[SPCONV] the decoders' kernel choices were not restored ({type(exc).__name__}: {str(exc)[:200]})",
              flush=True)
        return _restored
    if _restored:
        print(f"[SPCONV] {_restored} spconv kernel choices restored from the calibration record", flush=True)
    return _restored


@torch.no_grad()
def calibrate(device: Any, say: Callable[[str], None] = print, pause: Optional[Callable[[], Any]] = None,
              others: Optional[dict] = None) -> dict:
    """Time spconv's kernels for every convolution of the decoders (`shapes`) on a shell of their level's size and
    return the choices. `others`: another section of this card's record whose choices are put back first (Hi3D's:
    a key both use keeps one choice). `pause` is called before each shape (a check that the card is still quiet)."""
    import spconv.pytorch as spconv

    device = torch.device(device)
    if not applies():
        return {"ok": True, "skipped": "the decoders' convolutions do not time spconv's kernels on this card",
                "entries": []}
    tuner = _tuner()
    use_python_tuner()
    kept = restore_section(others)
    began = time.monotonic()
    timings, keys = [], set()
    dtype = torch.float16
    for ci, co, rows in shapes():
        if pause is not None:
            pause()
        before = set(tuner.kc_forward_cache)
        conv = spconv.SubMConv3d(ci, co, 3, bias=True, algo=spconv.ConvAlgo.MaskImplicitGemm,
                                 fp32_accum=FP32_ACCUM).to(device=device, dtype=dtype)
        x = _shell(rows, ci, dtype, device)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        conv(x)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - t0
        new = set(tuner.kc_forward_cache) - before
        keys |= new
        keys |= {k for k in tuner.kc_forward_cache if k[3] == co and k[4] == ci}
        timings.append({"in": ci, "out": co, "rows": int(x.features.shape[0]), "seconds": round(seconds, 2),
                        "new": len(new)})
        say(f"spconv {ci} -> {co} at {x.features.shape[0]:,} voxels: {seconds:.1f} s"
            + (" (timed)" if new else " (kept)"))
        del conv, x
        torch.cuda.empty_cache()
    entries = _entries(tuner.kc_forward_cache, keys)
    return {"ok": True, "arch": list(_arch()), "entries": entries, "shapes": timings, "kept_from_hi3d": kept,
            "fp32_accum": FP32_ACCUM,
            "seconds": round(time.monotonic() - began, 1),
            "summary": f"{len(entries)} kernel choices for {len(timings)} convolution shapes"}


def current() -> dict:
    """What spconv's cache holds now, in the record's form (a process's choices, for checks)."""
    return {"ok": True, "arch": list(_arch()), "entries": _entries(_tuner().kc_forward_cache)}
