"""The smallest GEMM call from which cuBLAS gives a row of the old cards' decoders the same bits whatever the call's
row count: measured once at install (`measure`, the calibration's "decode_products" section), read by the decode in
boxes on spconv (`floor`, `exact`, `verified`, `boundary_band`).

The decoders' products on a card without spconv's GEMM kernels (conv_spconv.py `products`: one torch.mm per kernel
offset) and their Linear layers (models/sc_vaes/chunked.py: the ConvNeXt MLPs, the subdivision heads, the output
layer) are cuBLAS calls, and cuBLAS chooses its kernel by the call's size: some kernels round differently (sliced-K,
split-K, s1688 against s16816). A row computed in a box's call is the whole table's row only when both calls take
kernels that agree. The decode in boxes therefore pads every product call that stands in for a whole-table call - a
box's, a slab's, a row subset's - to at least `floor` rows (zero rows, only the real ones kept), takes the products
of the offsets a whole level pairs fewer rows for from the whole level's own call (conv_spconv.whole_offset_products),
makes its box Linear calls exactly BOX_ROWS rows, and makes the whole-table decode's own calls where a level is
decoded whole (chunked.whole_level_calls). The shapes that stand in ("stand-in": the boxes' and the slabs'
convolutions, the box level's Linear layers and the output layers) decide; the others ("whole": the coarse levels'
ConvNeXt convolutions and MLPs, whose calls are the reference's own on spconv) are measured for the record only.

`measure`, for every shape (the products: the 9 (Ci, Co) pairs of the two decoders, with the KRSC weight's strided
offset slice conv_spconv multiplies by):
- sweeps the row count M on a table of at most 1.9 M rows / 768 MiB (MMAX_BYTES; the caching allocator's idle
  blocks are kept under as much again, `_trim_cache`): a call's rows - the first M rows, M rows from the middle, M
  gathered rows, M rows padded to the candidate floor - against one call on the whole table; the smallest M from
  which every tested M agrees is the shape's floor;
- for a stand-in shape, checks whole calls of the sizes of the level it serves in a 1536 decode (LEVEL_LADDER: level
  2 0.6-1.25 M rows, level 3 1.2-4 M, level 4 3-17 M), every row against the same rows in BOX_ROWS-row calls,
  compared window by window (the input and the whole call's output on the card, nothing else: at most BIG_BYTES
  (5.5 GiB) and BIG_FREE (70 %) of the free memory when the step starts - where a ladder size does not fit, the
  largest size that fits is checked instead); the largest whole call that agreed (or the swept table's own, where
  larger) is the shape's `verified` size, and a level's is the smallest of its stand-in shapes': a decode whose
  level is larger is outside what this card was measured at (chunked says so; `covers_voxels` lets the planner keep
  such props whole or at 1024);
- at cuBLAS's launch boundaries (BOUNDARY_SIZES: 2^20, 2^21, 2^22 rows) records which rows of a whole call differ
  from BOX_ROWS-row calls ("boundary": on an RTX 3060 under the pascal profile the fp32 output layer's last 64 / 128
  rows): a decode whose fine level lands in such a band (`boundary_band`) makes the output layer's whole call
  itself (chunked), and a box shape's band is either taken from the whole level's own calls (the up-block's conv1
  from the last level decoded whole: conv_spconv.whole_offset_products; the level-4 form's level-3 MLP:
  chunked._level_windows) or keeps the boxes from starting at that level (`plan_of`).

On an RTX 3060 (round O) the turing profile's (cuBLASLt) Linear shapes agreed from 6 Ki rows; the pascal profile's
non-Lt calls did not for the coarse levels' K = 1024 / 2048 shapes up to 131 Ki rows (hence the whole-level calls).

Where the boxes start (Spellcast3D round P, `plan`): each stand-in shape is either box-safe (`box_safe`: a product
with a floor of at most MAX_PRODUCTS_FLOOR rows, a Linear layer whose BOX_ROWS-row calls agreed with the whole
table's, and every whole call of its level's ladder equal to BOX_ROWS-row calls) or WHOLE-ONLY: its whole-level calls
are made as the whole-table decode makes them, never stood in for. The boxes start at the coarsest level whose box
shapes are all box-safe (`box_shapes`): level 3 (round O's form: the up-block from level 2 on in boxes) or level 4
(level 3 decoded whole, its 128 -> 128 convolutions and subdivision head the whole level's own calls, its MLP in
BOX_ROWS-row calls where those are box-safe; only the up-block from level 3 and the output layer in boxes); none where
neither is (`box_level` None: no decode in boxes, the whole-table decode alone). The output layer (fp32) never blocks
the boxes: up to `output_windows_to` rows (the largest whole call the sweep found equal to BOX_ROWS-row calls) it runs
in the boxes' windows, above it - and in a recorded launch-boundary band - the decode keeps the fine rows and makes
the whole decode's own call once (chunked, `output_whole`). On a real GTX 1080 Ti / 1070 Ti (sm_61, cuBLAS 12.6) the
128 -> 128 product's floor is 1,160,440 rows (its one disagreeing call above 6,144 rows is the one of exactly 2^20
rows: a launch boundary) and the output layer's whole calls of 12 M rows and more are another kernel's (6 M agree):
the boxes start at level 4 there, and the output layer makes its whole call above 6 M fine rows. Round O made such a
card decode everything in windows of 65,536 rows instead (twice the decode's time); that mode is now a forced check
only (PIXAL3D_DECODE_WINDOW).

Two more rules choose level 4 over level 3 (round P's review): a limit to the output layer's windows
(`output_windows_to` not None) - the level-4 form knows the fine level's size before its boxes, keeps level 3's table
in RAM beside the whole call's input and is what the planner's figures for that call were measured on (parts.py
SPCONV_L4_*), where round O's form would hold the fine rows and make the call uncounted -, and a launch-boundary band
of a box shape the form does not take from the whole level's own calls (`banded`: a recorded boundary with a first
differing row, or a sweep failure from BOX_ROWS rows on): round O's form stands in for level 3's MLP, subdivision head,
128 -> 128 and 128 -> 512 calls with box calls whatever the level's counts; both forms stand in for the finest level's
64 -> 64 products (a band there: no boxes).

    record = measure(device)    # {"ok", "card", "profile", "form", "products_floor", "linear_floor", "exact",
                                #  "box_level", "whole_only", "output_windows_to", "verified", "boundary", "shapes"...}
    floor("products")           # the class floor for the products (DEFAULT_FLOOR, or the record's where higher)
    floor_for(kind, ci, co)     # one shape's floor (DEFAULT_FLOOR, or its own where higher)
    box_level()                 # 3, 4 or None: the first level decoded in boxes (`plan`)
    box_safe(kind, ci, co)      # whether that shape's whole-level calls may be stood in for
    output_whole(rows)          # whether a fine level of `rows` rows makes the output layer's whole call
    in_band(kind, ci, co, rows) # whether a whole call of `rows` rows of the shape lands in a launch-boundary band
    band_from(kind, ci, co, rows)  # the first row of such a call that may be another kernel's (None: not in a band)
    exact()                     # boxes give the whole decode's values on this card (False without a record!)
    problem()                   # why there is no usable record ("" when there is one)
    window()                    # None, or the rows of every call (the forced windows-everywhere mode, below)
    verified(level)             # the largest whole call of that level the sweep checked
    beyond({level: rows})       # the levels larger than what was verified: [(level, rows, verified)]
    covers_voxels(voxels)       # every level of a prop of `voxels` foreseen fine voxels verified (LEVEL_SHARE)
    boundary_band(rows)         # whether a fine level of `rows` rows lands in a recorded boundary band

A record holds for the card and the profile it was measured under (the pascal profile turns cuBLASLt's fused
linear off: DISABLE_ADDMM_CUDA_LT, modules/precision.py). Without one for this card and profile - none, a failed
step, another card's, another profile's - `exact()` is False: spconv does not decode in boxes (chunked.backend), and
conv_spconv prints a warning naming the command that measures the card. DEFAULT_FLOOR (8,192, measured on sm_86) is
the floor a record whose card agreed from fewer rows gets.

A round O record (no "form") is read the same way: its shapes carry everything `plan` needs (floors, big checks, the
sweep's failing row counts), so a GTX 10xx install measured in round O decodes in the level-4 form without being
measured again; its "exact" / "window" flags are not read where it has its shapes. One without shapes (a hand-made
record) boxes from level 3 with its class floor where it says exact and names no window, else not at all.

The windows-everywhere mode (round O; now only where PIXAL3D_DECODE_WINDOW = n forces it, for measurements): every
GEMM call of the decoders - every product, every Linear layer, on whole levels and in boxes, and the whole-table
decode's too (it then goes through chunked.forward_chunked without boxes) - has exactly n rows (the last call of a
table padded), so a row meets the same kernel in every decode by construction, with other bits than the unwindowed
decode's where a whole call of another size took another kernel.

The record's section (written by calibrate.py; a floor is None where its class was not measured or has none):
    "decode_products": {"ok": true, "card": "<torch's device name>", "profile": "pascal", "capability": [6, 1],
                        "torch": "2.8.0+cu128", "cuda": "12.8", "form": 3, "products_floor": 8192,
                        "linear_floor": 4096, "exact": true, "window": null, "box_level": 4,
                        "whole_only": [{"kind", "ci", "co", "level", "why"}], "output_windows_to": 6000000,
                        "why_not_3": "...",
                        "verified": {"2": 1250000, "3": 4000000, "4": 17000000},
                        "boundary": [{"kind", "ci", "co", "M", "rows_differ", "first"}],
                        "peak_alloc_mib": 5200, "budget_mib": 5632,
                        "shapes": [{"kind", "role", "level", "ci", "co", "dtype", "mmax", "tested", "failed",
                                    "floor", "big": [{"M", "rows_differ", "first"}], "verified", "window_ok",
                                    "boundary"}, ...], "seconds": 41.2, "summary": "..."}
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

SECTION = "decode_products"
FORM = 3                        # the section's format: round P's "box_level", "whole_only", "output_windows_to" (2);
#                                 the launch-boundary check at 2^20 rows too and the plan's band and reach rules (3).
#                                 A section of an older form is read through `plan_of` (its shapes), and Spellcast3D's
#                                 installer measures it once more (DECODE_PRODUCTS_FORM there)
DEFAULT_FLOOR = 1 << 13         # measured on sm_86 (round O R2): every call from 8 Ki rows on agreed
DEFAULT_WINDOW = 1 << 16        # the rows of every call in the windows-everywhere mode
WINDOW_ENV = "PIXAL3D_DECODE_WINDOW"
MMAX_ROWS = 1_900_000           # the table a shape is swept on: a call of an ordinary size (not 2^21 rows, where
#                                 cuBLAS's grid limit changes how a call is cut: rows from the middle of one
#                                 2,097,152-row fp32 call differed from every smaller call's, round O)
# The whole calls each stand-in shape is checked at (rows), by the level it serves: the sizes of the levels of the
# test props at 1536 (level 0 12-50 k, level 1 52-200 k, level 2 0.2-0.75 M, level 3 0.87-3.1 M, level 4 2.4-16.5 M)
LEVEL_LADDER = {0: (60_000,), 1: (250_000,), 2: (600_000, 1_000_000, 1_250_000),
                3: (1_200_000, 2_500_000, 4_000_000), 4: (3_000_000, 6_000_000, 12_000_000, 17_000_000)}
BIG_BYTES = int(5.5 * (1 << 30))  # the most a whole-call check holds (its input and the whole call's output), and at
BIG_FREE = 0.7                  # most this share of the card's free memory (the calibration runs alone at install:
#                                 on an 8 GB card about 0.7 x 7.3 GiB = 5.1 GiB, the level-3 shapes to ~4.3 M rows)
MAX_PRODUCTS_FLOOR = 1 << 14    # a products floor above this: not exact (windows). It bounds the boxes' padded calls
#                                 and the whole level's own products of the offsets under the floor that the boxes
#                                 take theirs from (conv_spconv.whole_offset_products: under 26 x floor x Co x 2 bytes,
#                                 850 MiB for level 2's 1024 channels at 16 Ki rows)
BAND_ROWS = 1024                # the rows above a recorded boundary in which a fine level counts as in its band
# Each level's rows per fine voxel, the largest measured on the decoders at 1536 (round O; fine / level 3 / level 2):
# the boy 3,549,866 / 879,438 (0.2477) / 216,825 (0.0611), its turing twin 0.2494 / 0.0619, the boy's three views
# 0.2402 / 0.0569, the knight 0.2490 / 0.0609, the dragon 0.2486 / 0.0612, the ork 0.2392 / 0.0561, the tree
# 12,240,099 / 3,026,001 (0.2472) / 749,460 (0.0612); level 1 the boy's 51,883 (0.0146), level 0 the tokens (the boy
# 0.0034, the tree 43,529: 0.0036). They are applied to the planner's foreseen voxels, which carry its own margin
# (parts.VOXELS_PER_OCCUPIED, 1.15): the tree's foreseen 13.46 M voxels give 3.37 M level-3 rows (it has 3.03 M).
LEVEL_SHARE = {4: 1.0, 3: 0.25, 2: 0.062, 1: 0.0147, 0: 0.0036}
# Sizes at cuBLAS's launch boundaries (grid.y <= 65,535 tiles): on an RTX 3060 under the pascal profile the fp32
# output layer's whole call of 2^21 (2^22) rows gave its last 64 (128) rows another kernel's bits than any 64 Ki
# call (window-pascal.json, round O) - a band of a few hundred sizes a 2 M-row step. Recorded ("boundary"), not
# deciding `exact`: a prop whose fine level lands in such a band differs from its whole-table decode in those rows.
# 2^20 since round P's review: the real GTX 1080 Ti / 1070 Ti fail at exactly 2^20 rows in the products 128 -> 128
# and the subdivision heads, which their sweeps reach, while MMAX_BYTES keeps the 128 -> 512 sweeps under 630 k rows
# (2^20 x 640 x 2 bytes = 1.25 GiB: inside BIG_BYTES).
BOUNDARY_SIZES = (1 << 20, 1 << 21, 1 << 22)
BOX_ROWS = 1 << 16              # the boxes' Linear windows (chunked.BOX_WINDOW)
BOX_LEVELS = (3, 4)             # the levels the boxes may start at, tried in this order (`plan_of`)
MMAX_BYTES = 768 << 20          # and at most this much for its input and its output
SMALLEST = 64
RANDOM_POINTS = 48              # random row counts per shape beside the powers of two and their halves
# The decoders' channels per level and ConvNeXt blocks (spconv_choices.CHANNELS / BLOCKS), the output layers' widths
# (the shape decoder's 7 fields, the texture decoder's 6 attributes) and the latent's
CHANNELS = (1024, 512, 256, 128, 64)
OUTPUTS = (7, 6)
LATENT = 32

_found: Dict[str, Any] = {}


# ------------------------------------------------------------------------------------------------ reading the record

def forget() -> None:
    """Drop what was read (tests, or a process that wrote the record)."""
    _found.clear()


def _section(device: Any = None) -> Optional[dict]:
    """The record's section when it holds for this card and profile, else None (`problem` says why)."""
    if "section" not in _found:
        section, why = None, ""
        try:
            from ... import precision
            from .... import calibration

            record = calibration.load()
            got = calibration.section(SECTION, device)
            if not record:
                why = f"no calibration record at {calibration.path()}"
            elif not isinstance(record.get(SECTION), dict):
                why = "the calibration record has no decode_products section"
            elif got is None:
                why = f"its decode_products section was measured on another card ({record.get('card')})"
            elif not got.get("ok"):
                why = f"its decode_products step failed ({str(got.get('error') or 'no result')[:160]})"
            elif got.get("profile") != precision.profile().name:
                why = f"its decode_products section was measured under the {got.get('profile')} profile"
            else:
                section = got
        except Exception as exc:        # noqa: BLE001 - no record, no card
            why = f"the calibration record cannot be read ({type(exc).__name__})"
        _found["section"], _found["problem"] = section, why
    return _found["section"]


def problem(device: Any = None) -> str:
    """Why this card has no usable decode_products section ("" when it has one)."""
    _section(device)
    return _found.get("problem", "")


def floor(kind: str = "products", device: Any = None) -> int:
    """The class floor (rows) for `kind` ("products": the largest of the record's stand-in product floors):
    DEFAULT_FLOOR, or the install's measurement where it is higher (a lower one is not taken: every call from the
    measured floor on agreed, so DEFAULT_FLOOR rows do too). The decode pads a call to its own shape's floor
    (`floor_for`); this one stands in where the record has no shapes. "linear" reads the record's stand-in Linear
    floor, which is information only: the boxes' own Linear calls (BOX_ROWS rows) are judged by `box_safe`, and
    no caller sizes its calls by it (chunked.min_call_rows keeps MIN_CALL_ROWS: the encoders' windows never
    move)."""
    section = _section(device)
    value = section.get(f"{kind}_floor") if section else None
    return max(DEFAULT_FLOOR, int(value)) if isinstance(value, int) and value > 0 else DEFAULT_FLOOR


def _shape(kind: str, ci: int, co: int, device: Any = None) -> Optional[dict]:
    """The record's stand-in entry for the shape, None where it has none."""
    for entry in ((_section(device) or {}).get("shapes") or []):
        if isinstance(entry, dict) and entry.get("kind") == kind and entry.get("ci") == ci and entry.get("co") == co \
                and entry.get("role") == "stand-in":
            return entry
    return None


def floor_for(kind: str, ci: int, co: int, device: Any = None) -> int:
    """The smallest call (rows) a box, a slab or a row subset makes for the shape (kind, Ci, Co): DEFAULT_FLOOR, or
    the shape's own measured floor where higher; the class floor (`floor`) where the record has no such shape. A
    whole-only shape's (`box_safe` False) is never asked: its calls are the whole table's."""
    entry = _shape(kind, int(ci), int(co), device)
    if entry is None:
        return floor(kind, device)
    value = entry.get("floor")
    return max(DEFAULT_FLOOR, int(value)) if isinstance(value, int) and value > 0 else floor(kind, device)


def window(device: Any = None) -> Optional[int]:
    """The rows of every GEMM call of the decoders in the windows-everywhere mode, which PIXAL3D_DECODE_WINDOW = n
    forces (for measurements: round O's fallback, twice the decode's time on a GTX 10xx), else None. A record's
    "window" (round O's sweep wrote 65,536 where it found a shape without a floor) is not read: such a card decodes
    in the level-4 form (`plan`)."""
    text = os.environ.get(WINDOW_ENV, "").strip()
    if text:
        try:
            value = int(float(text))
            return value if value > 0 else None
        except ValueError:
            pass
    return None


def exact(device: Any = None) -> bool:
    """Whether this card decodes in boxes with the whole-table decode's values: `box_level` is not None (the
    install's sweep found every shape of the level the boxes start at box-safe), or the windows-everywhere mode is
    forced. False without this card's and profile's section (fail safe: the 1024 rule of small cards applies, see
    `problem`)."""
    return box_level(device) is not None


# ------------------------------------------------------------------------------------------------ where the boxes start

def _sweep_window_ok(entry: dict) -> bool:
    """Whether the sweep's own BOX_ROWS-row call agreed with the whole table's (round O's records overwrite
    "window_ok" with the big checks' verdict: this is the sweep's alone)."""
    return BOX_ROWS < int(entry.get("mmax") or 0) and BOX_ROWS not in (entry.get("failed") or [])


def shape_safe(entry: dict) -> Tuple[bool, str]:
    """Whether a recorded stand-in shape's whole-level calls may be stood in for by box, slab and subset calls, and
    why not: a product whose floor is at most MAX_PRODUCTS_FLOOR rows (its padded calls and the whole level's
    under-floor offsets stay bounded); a Linear layer (not the output layer) whose BOX_ROWS-row calls agreed with the
    whole table's; for both, every whole call of the level's ladder equal to BOX_ROWS-row calls. The output layer is
    always safe: above `output_windows_to` it makes the whole call itself."""
    kind = entry.get("kind")
    differ = [c.get("M") for c in entry.get("big") or [] if c.get("rows_differ")]
    if kind == "output":
        return True, ""
    if kind == "products":
        value = entry.get("floor")
        if not isinstance(value, int):
            return False, "no row count from which every call keeps a row's bits"
        if value > MAX_PRODUCTS_FLOOR:
            return False, f"its floor {value:,} rows is above {MAX_PRODUCTS_FLOOR:,}"
    elif not _sweep_window_ok(entry):
        return False, f"its {BOX_ROWS:,}-row calls differ from the whole table's"
    if differ:
        return False, "whole calls of " + ", ".join(f"{m:,}" for m in differ) + f" rows differ from {BOX_ROWS:,}-row calls"
    return True, ""


def box_shapes(level: int, channels: Iterable[int] = CHANNELS) -> List[Tuple[str, int, int]]:
    """The stand-in shapes a decode whose boxes start at `level` calls in boxes (chunked.forward_chunked): the
    up-blocks' conv1 from the level before it on (a box's parents, a row subset), the convolutions of the box levels
    and of the finer level each up-block makes, the box levels' MLP and subdivision head; the output layer apart
    (always safe, `shape_safe`)."""
    channels = list(channels)
    found: List[Tuple[str, int, int]] = []
    for lv, c in enumerate(channels[:-1]):
        nxt = channels[lv + 1]
        if lv >= level - 1:
            found.append(("products", c, 8 * nxt))
        if lv >= level:
            found.append(("products", c, c))
            found += [("mlp1", c, 4 * c), ("mlp2", 4 * c, c), ("to_subdiv", c, 8)]
        if lv + 1 >= level:
            found.append(("products", nxt, nxt))
    return list(dict.fromkeys(found))


def output_windows_to(shapes: Iterable[dict]) -> Optional[int]:
    """The largest fine level whose output layer's whole call the sweep found equal to BOX_ROWS-row calls, where
    it found a larger one that is not, for the output shapes together: the largest ladder size below the first big
    check that differed (the sweep's table where none agreed); 0 where the sweep's own BOX_ROWS-row call did not
    agree (the whole call always); None where every check agreed, or without an output shape (no limit: the windows,
    round O's rule - a prop above the ladder is not covered anyway, `covers_voxels`)."""
    found: Optional[int] = None
    for entry in shapes:
        if not isinstance(entry, dict) or entry.get("kind") != "output":
            continue
        if not _sweep_window_ok(entry):
            reach: Optional[int] = 0
        else:
            reach, last = None, int(entry.get("mmax") or 0)
            for check in sorted(entry.get("big") or [], key=lambda c: c.get("M") or 0):
                if check.get("rows_differ"):
                    reach = last
                    break
                last = max(last, int(check.get("M") or 0))
        if reach is not None:
            found = reach if found is None else min(found, reach)
    return found


def output_may_be_whole(device: Any = None) -> bool:
    """Whether some fine level makes the output layer's whole call on this card (`output_whole`): a limit to the
    windows' reach, or a launch-boundary band of the output layer (recorded, or a disagreeing sweep row count). The
    shape decoder, which learns its fine level's size at the end of round O's boxes, then keeps its fine rows."""
    if window(device):
        return False
    section = _section(device) or {}
    points = any(int(m) >= BOX_ROWS for s in section.get("shapes") or [] if isinstance(s, dict)
                 and s.get("kind") == "output" and s.get("role") == "stand-in" for m in s.get("failed") or [])
    return plan(device)["output_windows_to"] is not None or has_boundary(device) or points


def banded(entry: dict, boundary: Iterable[dict] = ()) -> List[int]:
    """The launch-boundary points of a recorded stand-in shape: the first differing rows of its boundary checks (its
    own "boundary" list, or the section's `boundary` entries of its kind, Ci and Co) and the row counts from BOX_ROWS
    on at which its sweep disagreed ([] where it has none: its whole calls of every size keep the BOX_ROWS calls'
    bits as far as the record knows)."""
    key = (entry.get("kind"), entry.get("ci"), entry.get("co"))
    found = {int(c["first"]) for c in entry.get("boundary") or [] if isinstance(c, dict) and c.get("rows_differ")
             and c.get("first")}
    found |= {int(b["first"]) for b in boundary if isinstance(b, dict) and b.get("first")
              and (b.get("kind"), b.get("ci"), b.get("co")) == key}
    found |= {int(m) for m in entry.get("failed") or [] if int(m) >= BOX_ROWS}
    return sorted(found)


def band_handled(level: int, channels: Iterable[int] = CHANNELS) -> Tuple[str, int, int]:
    """The box shape whose launch-boundary bands a decode whose boxes start at `level` takes from the whole level's own
    calls (conv_spconv.whole_offset_products: the offsets, and the centre, of the up-block's conv1 from the last
    level decoded whole, whose count lands in a band)."""
    channels = list(channels)
    return ("products", channels[level - 1], 8 * channels[level])


def plan_of(section: Optional[dict], products: bool) -> dict:
    """Where the boxes start for a record section (`box_level`: 3, 4, or None), its whole-only stand-in shapes and
    the output layer's reach (`output_windows_to`), from its shapes (round O's records and this round's alike; the
    section's own fields are what measure() wrote of the same rule). `products`: this card's convolutions are
    torch products (their shapes must be in the record); else the products' shapes are not looked at (spconv's
    implicit GEMM gives a row the same bits whatever the call). A section without shapes boxes from level 3 where it
    says exact and names no window (round O's reading), else not at all.

    A level is where the boxes may start when each of its box shapes (`box_shapes`) is box-safe (`shape_safe`) and has
    no launch-boundary band (`banded`), but for the one whose bands that form takes from the whole level's own calls
    (`band_handled`). Where the output layer's windows have a reach (output_windows_to not None) the boxes start at
    level 4 or nowhere: the level-4 form's whole output call is the one the planner counts. "why_not_3" says why the
    boxes do not start at level 3 ("" where they do)."""
    if not section:
        return {"box_level": None, "whole_only": [], "output_windows_to": None, "why": "no record", "why_not_3": ""}
    shapes = [s for s in section.get("shapes") or [] if isinstance(s, dict) and s.get("role") == "stand-in"]
    if not shapes:
        ok = section.get("exact", True) is not False and not section.get("window")
        return {"box_level": 3 if ok else None, "whole_only": [], "output_windows_to": None,
                "why": "" if ok else "the record has no shapes and is not exact", "why_not_3": ""}
    whole_only: List[dict] = []
    safe: Dict[Tuple[str, int, int], bool] = {}
    bands: Dict[Tuple[str, int, int], List[int]] = {}
    boundary = [b for b in section.get("boundary") or [] if isinstance(b, dict)]
    for entry in shapes:
        key = (entry.get("kind"), entry.get("ci"), entry.get("co"))
        if key[0] == "products" and not products:
            continue
        ok, why = shape_safe(entry)
        safe[key] = ok
        if not ok:
            whole_only.append({"kind": key[0], "ci": key[1], "co": key[2], "level": entry.get("level"), "why": why})
        points = banded(entry, boundary)
        if points:
            bands[key] = points
    reach = output_windows_to(shapes)
    if products and not any(k[0] == "products" for k in safe):
        return {"box_level": None, "whole_only": whole_only, "output_windows_to": reach,
                "why": "the record has no products' shapes for a card whose convolutions are products",
                "why_not_3": "no products' shapes"}

    def blocked(lv: int) -> List[str]:
        found = []
        for k in box_shapes(lv):
            if k[0] == "products" and not products:
                continue
            if not safe.get(k, True):
                found.append(f"{k[0]} {k[1]} -> {k[2]} is whole-only")
            elif k in bands and k != band_handled(lv):
                found.append(f"{k[0]} {k[1]} -> {k[2]} has a launch-boundary band (at "
                             + ", ".join(f"{m:,}" for m in bands[k][:3]) + " rows) its boxes would not keep")
        return found

    levels = [lv for lv in BOX_LEVELS if reach is None or lv == BOX_LEVELS[-1]]
    level = next((lv for lv in levels if not blocked(lv)), None)
    not_3 = [] if level == BOX_LEVELS[0] else blocked(BOX_LEVELS[0])
    if reach is not None and level != BOX_LEVELS[0]:
        not_3.append(f"the output layer's windows reach {reach:,} fine rows (its whole call above them: the "
                     f"level-4 form's)")
    why = "" if level is not None else "no level's box shapes are all box-safe: " + "; ".join(
        [f"{w['kind']} {w['ci']} -> {w['co']}: {w['why']}" for w in whole_only]
        + [f"level {lv}: " + ", ".join(b for b in blocked(lv) if "band" in b) for lv in levels
           if any("band" in b for b in blocked(lv))])
    return {"box_level": level, "whole_only": whole_only, "output_windows_to": reach, "why": why,
            "why_not_3": "; ".join(not_3)}


def _products_here() -> bool:
    try:
        from . import conv_spconv

        return conv_spconv.path() == "products"
    except Exception:       # noqa: BLE001 - spconv without its core: as the sweep would have measured it
        return True


def plan(device: Any = None) -> dict:
    """This card's plan (read once a process, with the section): the section's own fields where it is of this
    round's form (measure wrote them by `plan_of`), else `plan_of` its shapes (a round O record)."""
    if "plan" not in _found:
        section = _section(device)
        if not section:
            found = {"box_level": None, "whole_only": [], "output_windows_to": None, "why": problem(device),
                     "why_not_3": ""}
        elif int(section.get("form") or 0) >= FORM and all(k in section for k in
                                                            ("box_level", "whole_only", "output_windows_to")):
            level = section.get("box_level")
            found = {"box_level": level if level in BOX_LEVELS else None,
                     "whole_only": [w for w in section.get("whole_only") or [] if isinstance(w, dict)],
                     "output_windows_to": section.get("output_windows_to"),
                     "why": "" if level in BOX_LEVELS else "the record names no level to start the boxes at",
                     "why_not_3": str(section.get("why_not_3") or "")}
        else:
            found = plan_of(section, _products_here())
        _found["plan"] = found
    return _found["plan"]


def box_level(device: Any = None) -> Optional[int]:
    """The first decoder level decoded in boxes (chunked.forward_chunked): 3 (round O's form; also in the forced
    windows-everywhere mode), 4 (level 3 decoded whole), or None (no decode in boxes on this card)."""
    if window(device):
        return BOX_LEVELS[0]
    return plan(device)["box_level"]


def box_safe(kind: str, ci: int, co: int, device: Any = None) -> bool:
    """Whether the shape's whole-level calls may be stood in for (slabs, boxes, row subsets): not one of this card's
    whole-only shapes (`plan`; a shape the record does not name is, as before round P)."""
    if window(device):
        return True
    return not any(w["kind"] == kind and w["ci"] == ci and w["co"] == co for w in plan(device)["whole_only"])


def output_whole(rows: Optional[int], device: Any = None) -> bool:
    """Whether a fine level of `rows` rows makes the output layer's whole call (the whole decode's own) instead of
    the boxes' BOX_ROWS-row calls: above `output_windows_to`, or in a launch-boundary band of the output layer."""
    if rows is None or window(device):
        return False
    reach = plan(device)["output_windows_to"]
    return (reach is not None and rows > reach) or boundary_band(rows, device)


def in_band(kind: str, ci: int, co: int, rows: int, device: Any = None) -> bool:
    """Whether a whole call of `rows` rows of the shape lands in a launch-boundary band (boundary_band with the
    shape's own entries)."""
    return boundary_band(rows, device, (kind, int(ci), int(co)))


def band_from(kind: str, ci: int, co: int, rows: int, device: Any = None) -> Optional[int]:
    """The first row of a whole call of `rows` rows of the shape that may be another kernel's (None where the call is
    in no band, `in_band`): BAND_ROWS before the band's start where the band is a recorded boundary's (the rows that
    differ there are the call's last ones: 28-128 rows from the recorded first differing row on, real GTX 1080 Ti),
    0 where it is a row count the sweep found disagreeing (which rows differ there was not recorded)."""
    return _band(rows, device, (kind, int(ci), int(co)))


def verified(level: int, device: Any = None) -> Optional[int]:
    """The largest whole call the record checked for every stand-in shape of `level` (None: no limit known - an
    older record without the figure, or no shape of that level stands in)."""
    section = _section(device)
    value = ((section or {}).get("verified") or {}).get(str(level))
    return int(value) if isinstance(value, int) and value > 0 else None


def beyond(levels: Dict[int, int], device: Any = None) -> List[Tuple[int, int, int]]:
    """The levels whose rows exceed what the record verified: [(level, rows, verified)]. None in the windows-everywhere
    mode (`window`): there no call of the decoders has a level's rows, whole decodes included."""
    if window(device):
        return []
    out = []
    for level, rows in sorted(levels.items()):
        limit = verified(level, device)
        if limit is not None and rows > limit:
            out.append((level, int(rows), limit))
    return out


def covers_voxels(voxels: int, device: Any = None) -> bool:
    """Whether a prop of `voxels` fine voxels has every level within what the record verified (the level sizes
    estimated by LEVEL_SHARE): for the planner, which keeps a prop it does not cover whole, or at 1024."""
    return not beyond({level: int(share * voxels) for level, share in LEVEL_SHARE.items()}, device)


def _matches(entry: dict, shape: Tuple) -> bool:
    """Whether a record entry is of the shape (kind, ci) or (kind, ci, co)."""
    return entry.get("kind") == shape[0] and entry.get("ci") == shape[1] and (len(shape) < 3 or entry.get("co") == shape[2])


def has_boundary(device: Any = None, shape: Tuple = ("output", 64)) -> bool:
    """Whether the record found a launch-boundary band for the shape (kind, ci) or (kind, ci, co)."""
    section = _section(device)
    return any(_matches(b, shape) and b.get("first") for b in (section or {}).get("boundary") or [])


def boundary_band(rows: Optional[int], device: Any = None, shape: Tuple = ("output", 64)) -> bool:
    """Whether a whole call of `rows` rows of the shape (kind, ci) or (kind, ci, co) lands in a band the record
    found at cuBLAS's launch boundaries: within BAND_ROWS above one of the recorded first differing rows, or above a
    power-of-two multiple of the smallest of them (round P's review: not every multiple - a 2^20 boundary would
    otherwise make a band near 3 x 2^20 that no check found); or within BAND_ROWS of a row count from BOX_ROWS on at
    which the shape's own sweep disagreed (round P: on sm_61 the one such count is 2^20, on an RTX 3060 under the
    pascal profile the output layer's too - the launch boundary below the recorded ones)."""
    return _band(rows, device, shape) is not None


def _band(rows: Optional[int], device: Any, shape: Tuple) -> Optional[int]:
    """boundary_band's test, with the first row of the call that may differ where it holds (band_from)."""
    section = _section(device)
    if rows is None or not section:
        return None
    firsts = sorted({int(b["first"]) for b in section.get("boundary") or [] if _matches(b, shape) and b.get("first")})
    points = sorted({int(m) for s in section.get("shapes") or [] if isinstance(s, dict) and _matches(s, shape)
                     and s.get("role") == "stand-in" for m in s.get("failed") or [] if int(m) >= BOX_ROWS})
    if any(abs(rows - m) <= BAND_ROWS for m in points):
        return 0
    if not firsts or rows <= firsts[0]:
        return None
    starts = set(firsts)
    multiple = firsts[0]
    while multiple < rows:
        starts.add(multiple)
        multiple *= 2
    hits = [start for start in starts if 0 < rows - start <= BAND_ROWS]
    return max(0, min(hits) - BAND_ROWS) if hits else None


# ------------------------------------------------------------------------------------------------ the measurement

BOX_FROM = 2                     # the level the boxes start from (chunked.forward_chunked: the third-last)


def product_shapes(channels: Iterable[int] = CHANNELS) -> List[Tuple[int, int]]:
    """(Ci, Co) of every distinct convolution of the decoders (spconv_choices.shapes' pairs)."""
    return [(ci, co) for ci, co, _ in product_roles(channels)]


def product_roles(channels: Iterable[int] = CHANNELS) -> List[Tuple[int, int, str]]:
    """(Ci, Co, role) of every distinct convolution: "stand-in" where a box, a slab or a row subset computes it in
    place of the whole table (every up-block's conv1, the box levels' convolutions), else "whole"."""
    channels = list(channels)
    found: dict = {}
    for level, c in enumerate(channels[:-1]):
        nxt = channels[level + 1]
        found.setdefault((c, c), "stand-in" if level > BOX_FROM else "whole")
        found[(c, 8 * nxt)] = "stand-in"                    # conv1: slabs on the whole levels, boxes after
        if level >= BOX_FROM:
            found[(nxt, nxt)] = "stand-in"                  # conv2 on a box's finer level
        else:
            found.setdefault((nxt, nxt), "whole")
    return [(ci, co, role) for (ci, co), role in found.items()]


def product_level(ci: int, co: int, channels: Iterable[int] = CHANNELS) -> int:
    """The level whose whole calls a convolution of (Ci, Co) makes: its input's level (an up-block's conv1 runs on
    the coarser level, its conv2 on the finer one, whose channels are the next level's)."""
    channels = list(channels)
    return channels.index(ci)


def linear_level(name: str, ci: int, co: int, channels: Iterable[int] = CHANNELS) -> int:
    """The level a Linear layer runs on (the MLP's second layer by its output; the output layers on the finest)."""
    channels = list(channels)
    if name == "output":
        return len(channels) - 1
    if name == "from_latent":
        return 0
    return channels.index(co if name == "mlp2" else ci)


def linear_shapes(channels: Iterable[int] = CHANNELS) -> List[Tuple[str, int, int, torch.dtype]]:
    """(name, in, out, dtype) of every Linear layer of the decoders: per level with ConvNeXt blocks its MLP's two
    layers and the subdivision head (fp16, the torso), the output layers on the finest level and the latent's input
    layer (fp32: outside the torso)."""
    return [(name, ci, co, dtype) for name, ci, co, dtype, _ in linear_roles(channels)]


def linear_roles(channels: Iterable[int] = CHANNELS) -> List[Tuple[str, int, int, torch.dtype, str]]:
    """linear_shapes with their role: "stand-in" for the box level's layers and the output layers (the boxes' windows
    stand in for the whole level's calls), "whole" for the levels decoded whole (their calls are the reference's)."""
    channels = list(channels)
    out: List[Tuple[str, int, int, torch.dtype, str]] = []
    for level, c in enumerate(channels[:-1]):
        role = "stand-in" if level > BOX_FROM else "whole"
        out += [("mlp1", c, 4 * c, torch.float16, role), ("mlp2", 4 * c, c, torch.float16, role),
                ("to_subdiv", c, 8, torch.float16, role)]
    out += [("output", channels[-1], width, torch.float32, "stand-in") for width in OUTPUTS]
    out.append(("from_latent", LATENT, channels[0], torch.float32, "whole"))
    return out


def row_counts(mmax: int, points: int = RANDOM_POINTS, seed: int = 0) -> List[int]:
    """The M a shape is tested at: powers of two from SMALLEST and the counts halfway between them, below mmax, and
    `points` log-uniform random counts (fixed by `seed`)."""
    found = set()
    m = SMALLEST
    while m < mmax:
        found.add(m)
        if m + m // 2 < mmax:
            found.add(m + m // 2)
        m *= 2
    g = torch.Generator().manual_seed(seed)
    span = math.log(max(mmax - 1, SMALLEST + 1)) - math.log(SMALLEST)
    for value in torch.rand(points, generator=g).tolist():
        found.add(int(round(math.exp(math.log(SMALLEST) + value * span))))
    return sorted(m for m in found if SMALLEST <= m < mmax)


def floor_of(results: List[Tuple[int, bool]]) -> Optional[int]:
    """The smallest tested M from which every tested M agreed; None when the largest tested M did not (no floor
    below the whole table)."""
    ordered = sorted(results)
    if not ordered or not ordered[-1][1]:
        return None
    best = ordered[-1][0]
    for m, ok in reversed(ordered):
        if not ok:
            break
        best = m
    return best


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    bits = torch.int16 if a.element_size() == 2 else torch.int32
    return a.shape == b.shape and torch.equal(a.contiguous().view(bits), b.contiguous().view(bits))


def _sweep(fn: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, counts: List[int],
           seed: int) -> Tuple[List[Tuple[int, bool]], List[int]]:
    """fn's rows on M-row calls against one call on the whole of X: for every M the first M rows, M rows from
    the middle and M gathered rows. -> [(M, all three agreed)], the failing M."""
    full = fn(X)
    n = X.shape[0]
    g = torch.Generator(device=X.device).manual_seed(seed)
    results, failed = [], []
    for m in counts:
        off = (n - m) // 2
        pick = torch.sort(torch.randperm(n, device=X.device, generator=g)[:m]).values
        ok = (_same(fn(X[:m]), full[:m]) and _same(fn(X[off:off + m]), full[off:off + m])
              and _same(fn(X.index_select(0, pick)), full.index_select(0, pick)))
        results.append((m, ok))
        if not ok:
            failed.append(m)
        _trim_cache(X.device)
    return results, failed


def _trim_cache(device: torch.device) -> None:
    """Hand the caching allocator's idle blocks back to the card where they exceed MMAX_BYTES. The sweep's calls grow
    with the row count, each larger than every block the cache holds, so without this the cache kept one block per
    row count until the shape's end: on an RTX 3060 under the pascal profile 14.9 GB reserved for the 512 -> 2048
    products, past the card's 12 GB into the driver's system-memory fallback, while 1.6 GB were allocated (round O).
    Where a block lies does not change which kernel cuBLAS takes for a call's sizes: the bits stay."""
    if device.type == "cuda" and torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) > MMAX_BYTES:
        torch.cuda.empty_cache()


def _windowed(fn: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, window: int) -> torch.Tensor:
    """fn's rows of X in calls of exactly `window` rows, the last overlapping the one before (chunked.rowwise)."""
    n = X.shape[0]
    out = None
    for a in range(0, n, window):
        b = min(n, a + window)
        first = max(0, b - window)
        part = fn(X[first:b])[a - first:]
        if out is None:
            out = part.new_empty((n, part.shape[1]))
        out[a:b] = part
    return out


def _big_budget(device: Any) -> int:
    """The bytes a whole-call check may hold: BIG_BYTES, or BIG_FREE of the card's free memory where less."""
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:       # noqa: BLE001 - no card to ask (the CPU)
        return BIG_BYTES
    return int(min(BIG_BYTES, BIG_FREE * free))


def big_sizes(ladder: Iterable[int], ci: int, co: int, itemsize: int, budget: int) -> List[int]:
    """The ladder's sizes that fit `budget` (an input and the whole call's output), and the largest that fits
    where a ladder size does not (rounded down to 1,000 rows)."""
    per_row = itemsize * (ci + co)
    fit = budget // per_row
    sizes = [m for m in ladder if m <= fit]
    if any(m > fit for m in ladder) and fit >= BOX_ROWS and (not sizes or fit > sizes[-1] + 1000):
        sizes.append(int(fit // 1000 * 1000))
    return sizes


def _big_check(fn: Callable[[torch.Tensor], torch.Tensor], ci: int, co: int, dtype: torch.dtype, device: Any,
               seed: int, sizes: Iterable[int]) -> List[dict]:
    """A whole call of each of `sizes` rows against the same rows computed in calls of BOX_ROWS rows (the last
    overlapping the one before, as chunked.rowwise cuts), compared window by window in place: the input and the
    whole call's output are all it holds. -> [{"M", "rows_differ", "first"}]."""
    itemsize = torch.tensor([], dtype=dtype).element_size()
    bits = torch.int16 if itemsize == 2 else torch.int32
    found = []
    for m in sizes:
        g = torch.Generator(device=device).manual_seed(seed + m)
        X = torch.randn((m, ci), generator=g, device=device, dtype=dtype)
        whole = fn(X)
        differ, first = 0, None
        for a in range(0, m, BOX_ROWS):
            b = min(m, a + BOX_ROWS)
            lo = max(0, b - BOX_ROWS)
            part = fn(X[lo:b])[a - lo:]
            bad = (whole[a:b].view(bits) != part.view(bits)).any(dim=1).nonzero().squeeze(1)
            if bad.numel():
                differ += int(bad.numel())
                first = a + int(bad[0]) if first is None else first
            del part, bad
        found.append({"M": m, "rows_differ": differ, "first": first})
        del X, whole
        torch.cuda.empty_cache()
    return found


def _padded_ok(fn: Callable[[torch.Tensor], torch.Tensor], X: torch.Tensor, at: int) -> bool:
    """Calls of fewer than `at` rows padded with zero rows to `at` give the whole call's rows."""
    full = fn(X)
    for m in sorted({SMALLEST, 1000, at // 2, at - 1}):
        if not 0 < m < at:
            continue
        padded = X.new_zeros((at, X.shape[1]))
        padded[:m] = X[:m]
        if not _same(fn(padded)[:m], full[:m]):
            return False
    return True


def _table(rows: int, width: int, out: int, dtype: torch.dtype, device: Any, seed: int) -> torch.Tensor:
    itemsize = torch.tensor([], dtype=dtype).element_size()
    rows = min(rows, MMAX_ROWS, MMAX_BYTES // (itemsize * (width + out)))
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((rows, width), generator=g).to(device=device, dtype=dtype)


@torch.no_grad()
def measure(device: Any, say: Callable[[str], None] = print, pause: Optional[Callable[[], Any]] = None,
            points: int = RANDOM_POINTS, products: Optional[bool] = None) -> dict:
    """Sweep every product shape (where this card's convolutions run on torch.mm, or `products` True) and every
    Linear shape of the decoders on the card and return the record section (the module's docstring). `pause` is
    called before each shape (a check that the card is still quiet)."""
    from ... import precision
    from .. import config as sparse_config

    began = time.monotonic()
    profile = precision.profile()           # first: the pascal profile turns cuBLASLt's fused linear off
    device = torch.device(device)
    base = {"profile": profile.name, "capability": list(profile.capability) if profile.capability else None,
            "torch": torch.__version__, "cuda": torch.version.cuda}
    if profile.conv != "spconv" or sparse_config.CONV != "spconv":
        return {"ok": True, **base, "skipped": "the decoders run on flex_gemm here: no floor is needed",
                "products_floor": None, "linear_floor": None, "exact": True, "shapes": []}
    if products is None:
        from . import conv_spconv

        products = conv_spconv.path() == "products"
    base["card"] = torch.cuda.get_device_name(device)
    shapes: List[dict] = []
    budget = _big_budget(device)
    torch.cuda.reset_peak_memory_stats(device)

    def one(kind: str, ci: int, co: int, dtype: torch.dtype, fn_of: Callable[[torch.Tensor], Callable], seed: int,
            role: str, level: int):
        if pause is not None:
            pause()
        X = _table(MMAX_ROWS, ci, co, dtype, device, seed)
        fn = fn_of(X)
        counts = row_counts(X.shape[0], points, seed)
        results, failed = _sweep(fn, X, counts, seed)
        found = floor_of(results)
        window_ok = dict(results).get(BOX_ROWS)
        if found is not None and not _padded_ok(fn, X, found):
            found = None
        mmax = int(X.shape[0])
        del X
        torch.cuda.empty_cache()
        itemsize = torch.tensor([], dtype=dtype).element_size()
        big = (_big_check(fn, ci, co, dtype, device, seed,
                          big_sizes(LEVEL_LADDER.get(level, ()), ci, co, itemsize, budget))
               if role == "stand-in" else [])
        if any(check["rows_differ"] for check in big):
            found = None                    # a whole level's call would give rows no box call gives
            window_ok = False
        checked = [mmax] + [c["M"] for c in big if not c["rows_differ"]]
        boundary = (_big_check(fn, ci, co, dtype, device, seed,
                               [m for m in BOUNDARY_SIZES if m * itemsize * (ci + co) <= budget])
                    if role == "stand-in" else [])
        shapes.append({"kind": kind, "role": role, "level": level, "ci": ci, "co": co,
                       "dtype": str(dtype).replace("torch.", ""), "mmax": mmax, "tested": len(counts),
                       "failed": failed, "floor": found, "big": big,
                       "verified": max(checked) if found is not None or window_ok else None,
                       "window_ok": bool(window_ok), "boundary": boundary})
        sizes = ", ".join(format(check["M"], ",") for check in big)
        same = "yes" if not any(check["rows_differ"] for check in big) else "NO"
        say(f"cuBLAS {kind} {ci} -> {co} ({str(dtype).replace('torch.', '')}, {role}): {len(counts)} row counts up to "
            f"{mmax:,}, " + (f"the same rows from {found:,} rows a call" if found is not None else "no floor")
            + (f"; differ at {failed[:6]}" if failed else "")
            + (f"; whole calls of {sizes} rows = {BOX_ROWS:,}-row calls: {same}" if big else "")
            + "".join(f"; at {c['M']:,} rows {c['rows_differ']} rows differ" for c in boundary if c["rows_differ"]))
        del fn
        torch.cuda.empty_cache()

    def product_fn(co: int, ci: int, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        filters = (torch.randn((co, 27, ci), generator=g) / math.sqrt(27 * ci)).to(device, torch.float16)
        wt = filters[:, 0].t()              # [Ci, Co] with the leading dimension 27 * Ci, as conv_spconv's views
        return lambda X: (lambda rows: torch.mm(rows, wt))

    def linear_fn(ci: int, co: int, dtype: torch.dtype, seed: int):
        layer = torch.nn.Linear(ci, co).to(device=device, dtype=dtype)
        g = torch.Generator(device="cpu").manual_seed(seed + 1)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=g) / math.sqrt(ci))
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=g) * 0.1)
        return lambda X: (lambda rows: torch.nn.functional.linear(rows, layer.weight, layer.bias))

    seed = 0
    if products:
        for ci, co, role in product_roles():
            seed += 1
            one("products", ci, co, torch.float16, product_fn(co, ci, seed), seed, role, product_level(ci, co))
    for name, ci, co, dtype, role in linear_roles():
        seed += 1
        one(name, ci, co, dtype, linear_fn(ci, co, dtype, seed), seed, role, linear_level(name, ci, co))

    def class_floor(kinds: Callable[[str], bool]) -> Tuple[Optional[int], bool]:
        mine = [s for s in shapes if kinds(s["kind"]) and s["role"] == "stand-in"]
        if not mine:
            return None, True
        if any(s["floor"] is None for s in mine):
            return None, False
        return max(s["floor"] for s in mine), True

    # The products a box or a slab makes have any row count from the floor on: every one of them must agree. The
    # Linear calls a box makes have exactly BOX_ROWS rows (chunked.BOX_WINDOW): those, at the table's head, middle
    # and gathered rows, and whole calls of its level (LEVEL_LADDER) must agree; the smallest row count from which every
    # tested call agrees is recorded too ("linear_floor": on sm_86's non-Lt fp32 SGEMM a call of exactly 2^20 rows
    # gave its last rows another kernel's bits, which no box call meets).
    products_floor, products_ok = class_floor(lambda kind: kind == "products")
    if products_floor is not None and products_floor > MAX_PRODUCTS_FLOOR:
        products_ok = False                 # some product shape's padded calls would not be bounded: whole-only
    linear_floor, _ = class_floor(lambda kind: kind != "products")
    linear_ok = all(s["window_ok"] for s in shapes if s["kind"] not in ("products", "output") and s["role"] == "stand-in")
    found_plan = plan_of({"shapes": shapes}, products)
    level = found_plan["box_level"]
    verified_: Dict[str, int] = {}
    for s_ in shapes:
        if s_["role"] == "stand-in" and s_["verified"]:
            key = str(s_["level"])
            verified_[key] = min(verified_.get(key, s_["verified"]), s_["verified"])
    def said(value: Optional[int], ok: bool) -> str:
        return f"from {value:,} rows" if value else ("none" if not ok else "not measured")

    reach = found_plan["output_windows_to"]
    summary = (f"the shapes the boxes stand in for keep their rows: products {said(products_floor, products_ok)}, "
               f"Linear in {BOX_ROWS:,}-row calls {'yes' if linear_ok else 'NO'}"
               + (f" (every call {said(linear_floor, True)})" if linear_floor else "")
               + "".join(f"; whole-only {w['kind']} {w['ci']} -> {w['co']} ({w['why']})"
                         for w in found_plan["whole_only"])
               + ("; NO decode in boxes: " + found_plan["why"] if level is None else
                  "; boxes from level 3" if level == 3 else
                  f"; boxes from level 4 (level 3 decoded whole: {found_plan['why_not_3']})")
               + ("" if reach is None else f"; the output layer's whole call above {reach:,} fine rows"))
    boundary = [{"kind": s_["kind"], "ci": s_["ci"], "co": s_["co"], **c} for s_ in shapes
                for c in s_.get("boundary", []) if c["rows_differ"]]
    if verified_:
        summary += "; whole calls verified up to " + ", ".join(f"level {k} {v:,}" for k, v in sorted(verified_.items()))
    peak = round(torch.cuda.max_memory_allocated(device) / 2 ** 20)
    summary += f"; at most {peak:,} MiB on the card (budget {budget / 2 ** 20:,.0f} MiB)"
    return {"ok": True, **base, "form": FORM, "products_floor": products_floor, "linear_floor": linear_floor,
            "exact": level is not None, "window": None, "box_level": level,
            "whole_only": found_plan["whole_only"], "output_windows_to": reach,
            "why_not_3": found_plan["why_not_3"], "verified": verified_,
            "boundary": boundary, "peak_alloc_mib": peak, "budget_mib": round(budget / 2 ** 20),
            "shapes": shapes, "seconds": round(time.monotonic() - began, 1), "summary": summary}
