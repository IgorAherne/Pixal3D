"""A decode whose peak on the card stops growing with the prop: the last levels in spatial boxes.

With `decoder.chunk_voxels = n` (0, the default: off) a SparseUnetVaeDecoder - without gradients, outside training,
one item - decodes through `forward_chunked` instead of its loop:

- Levels 0-2 (R/16, R/8, R/4) on whole tables, as before but leaner: a ConvNeXt block normalises and runs its MLP on
  a slab of rows at a time, in place in its convolution's output (no [n, 4C] hidden table, no second [n, C] one);
  an up-block's conv1 runs on a slab of coarse rows at a time, gathered straight into the finer level (no [n, 8C']
  table of the whole level).
- The rest - the up-block from level 2 to 3, the ConvNeXt blocks of level 3, the up-block from 3 to 4 and the tail -
  on boxes of level-2 voxels, `n` fine voxels a box at most (about). A box is decoded together with every level-2
  voxel within `halo()` voxels of it (5 here): all its own fine voxels depend on through the 3x3x3 convolutions after
  level 2; deeper down only what is still needed is computed (the up-blocks' conv1 on the parents within reach).
  Only the box's own rows are kept.
- The kept rows go to the reference's row order (the whole-table decode's) when the boxes are done. The shape
  decoder's mesh is then extracted a slab of voxels at a time (fdg_vae.py, `dual_grid_faces`).

The values are those of the whole-table decode, bit for bit, because:
- LayerNorm, SiLU, the skip connections and the subdivision are computed row by row: a slab gives the same rows.
- A Linear layer gives a row the same values whatever other rows share its call, from 8 Ki rows a call on (cuBLAS;
  measured on every Linear shape of the two decoders at the golden props' table sizes, against windows of
  8 Ki-256 Ki rows and gathered subsets). Calls of a few thousand rows or fewer may take another kernel and round
  differently, so no call here is smaller than a window (`rowwise`), unless it is the reference's own call.
- flex_gemm's masked implicit GEMM computes an output row from its own neighbour list alone: the kernel offsets of
  a block of rows are visited in ascending order and an offset the row lacks adds exact zeros, so neither the rows
  it is grouped with, nor the tile configuration, nor the row count change it - with SPLITK 1, where the sum is
  not split. The boxes' convolutions run that kernel directly, always: the whole-table decode takes SPLITK 1 for
  every large level on the card this was measured on (`reference_splitk`), and on a card whose autotuner would
  split a level's sum the boxes give the SPLITK-1 values - those of the golden runs - in the bounded memory,
  rather than fall back to a whole-table decode that may not fit. The coarse levels, decoded whole, take the
  whole-table decode's own calls where that would split (`sliced`).
- A box's rows within the halo have all their neighbours, with the same values as in the whole table.

On spconv (the cards below sm_80, Spellcast3D round O) the convolutions run on a pair table built once per level and
kept in its spatial cache (modules/sparse/conv/conv_spconv.py `subm_rows`; whole levels and boxes alike):
- spconv's implicit GEMM (sm_70+) computes an output row from its own neighbours alone, as flex_gemm's masked
  kernel does: a block of rows visits the union of its rows' offsets in ascending order, a missing offset adds
  exact zeros, the forward split-K is always 1 and the kernel is chosen without the row count (the calibration
  record's choice, spconv_choices: the box decode has the whole-table decode's bits under the same record; with no
  record spconv times its kernels at the first call of each shape in the process, and a decode then has that
  process's bits, the boxes included). Every level may be sliced (`slices`).
- the products on torch.mm (sm_61) are spconv's Native sums - centre, offsets 0..26, bias, one fp16 rounding an
  offset. The whole levels' convolutions are the whole-table decode's own calls (unpadded: today's calls on any
  card); every call that stands in for one - a box's, a slab's, a row subset's - has at least
  products_floor.floor("products") rows a product (padded with zero rows): from there cuBLAS gives a row the
  same bits whatever the call's size, for the shapes the boxes and slabs run (measured at install, the record's
  "decode_products"). An offset the whole level pairs fewer rows for (0 < n < the floor) is the whole level's own
  call once (conv_spconv.whole_offset_products), and every slab or box adds its rows of it at that offset's turn
  (up_block's `fixed`): the boxes never fall back to a whole table the memory model did not plan.
- cuBLAS's fp16 GEMMs without cuBLASLt (the pascal profile) give the coarse levels' Linear shapes (K = 1024 / 2048)
  other bits at 8-131 Ki rows a call than at the level's size, so on spconv every Linear layer of a level decoded
  whole makes the reference's own call, one per layer (`whole_level_calls`: the ConvNeXt MLPs and the tail); the
  boxes' windows (BOX_WINDOW, padded) stand in for the box level's and the output layers' whole calls, which the
  install's sweep checks at the whole-level sizes of each level (products_floor.verified: a decode whose level is
  larger is outside what was measured - the planner's boxes_supported(voxels) keeps such props whole or at 1024, and
  a decode that gets here all the same says so). Where the record found a launch-boundary band for the output layer
  (products_floor.boundary_band) the fine rows before it are kept and, for a fine level in the band, the output
  layer makes the whole decode's own call. Without this card's ok decode_products section, or where it found a
  shape the boxes call with no such floor (products_floor.exact() False), spconv does not decode in boxes
  (`backend`).

PIXAL3D_CHUNK_VOXELS = n (> 0) decodes every decode that can be decoded in boxes in boxes of about n fine voxels,
whatever `decoder.chunk_voxels` says (0: none in boxes) - for the identity checks of a prop the memory model would
decode whole; unset, the decoder's own setting decides.
"""

from __future__ import annotations

import importlib
import math
import os
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ...modules import sparse as sp

SLAB_BYTES = 64 << 20       # the widest table of a slab: the MLP's hidden rows, conv1's [rows, 8C'] output
COARSE_SLAB_BYTES = 256 << 20   # the same on the levels decoded on whole tables (0-2), far below the boxes' peak:
#                             larger slabs there make fewer, larger convolution and Linear calls (the same rows)
EVEN_WINDOWS = True         # rowwise on a whole table: near-equal windows of at least MIN_CALL_ROWS rows instead of
#                             `window`-row windows the last of which overlaps the one before (whose rows it computes
#                             again); a ConvNeXt block's second MLP layer then writes straight into the table
MIN_CALL_ROWS = 1 << 13     # the smallest Linear call that gives a row the same values as any larger one (above)
GATHER_INTO = False         # an up-block's conv1 rows gathered straight into the finer level's table (index_select
#                             with out=: the same copy without the intermediate table and its second copy)
NORM_SLAB_ROWS = 1 << 16    # rows normalised at a time (LayerNorm32 makes two float32 copies of them)
BOX_WINDOW = 1 << 16        # rows of every Linear call inside a box (a box's tables are padded to it)
KERNEL_VOLUME = 27
MIN_BOX = 8                 # level-2 voxels: a box is not cut below this side
FINE_PER_LEVEL3 = 4.6       # fine voxels per level-3 voxel the shape decoder's boxes are cut for (4.12-4.47 on the
#                             golden props; the texture decoder knows its own)
LAST: Dict[str, Any] = {}   # what the last chunked decode did (boxes, rows, seconds): for measuring
BOXES = True                # False: no boxes, every level on whole tables with the slabbed layers only (for
#                             measuring what the slabs alone give)
CHUNK_ENV = "PIXAL3D_CHUNK_VOXELS"      # the override of decoder.chunk_voxels (the module's docstring)


# ------------------------------------------------------------------------------------------------ the convolutions

def _module(name: str) -> Any:
    # the package exports functions under its modules' names: take the module itself
    return importlib.import_module(f"flex_gemm.kernels.triton.spconv.{name}")


def backend() -> str:
    """The convolutions a decode in boxes runs on in this process: "flex_gemm" (its masked split-K algorithm, the one
    whose rows this module reproduces), "spconv" (conv_spconv's subm_rows), or "" (no decode in boxes)."""
    from ...modules.sparse import config as sparse_config
    from ...modules.sparse.conv import config as conv_config
    if sparse_config.CONV == "flex_gemm":
        return "flex_gemm" if conv_config.FLEX_GEMM_ALGO == "masked_implicit_gemm_splitk" else ""
    if sparse_config.CONV == "spconv":
        from ...modules.sparse.conv import conv_spconv, products_floor
        return "spconv" if conv_spconv.boxes_supported() and products_floor.exact() else ""
    return ""


def boxes_supported(voxels: Optional[int] = None) -> bool:
    """Whether this process's backend can decode in boxes (the whole-table decode's values in a bounded memory):
    flex_gemm's masked split-K, or spconv where this card's decode_products measurement says so
    (products_floor.exact: never without one). `voxels`: also whether a prop of that many fine voxels has every
    level within the whole calls the measurement verified (products_floor.covers_voxels; spconv only) - for the
    planner, which then keeps a prop it does not cover whole, or at 1024."""
    found = backend()
    if found != "spconv" or voxels is None:
        return bool(found)
    from ...modules.sparse.conv import products_floor
    return products_floor.covers_voxels(int(voxels))


def chunk_voxels(decoder: Any) -> int:
    """The box size (fine voxels) `decoder` decodes in: PIXAL3D_CHUNK_VOXELS when set, else decoder.chunk_voxels."""
    text = os.environ.get(CHUNK_ENV, "").strip()
    if text:
        try:
            return max(0, int(float(text)))
        except ValueError:
            pass
    return int(getattr(decoder, "chunk_voxels", 0) or 0)


def applies(decoder: Any, x: sp.SparseTensor) -> bool:
    """Whether `decoder` decodes x through forward_chunked: a box size set (chunk_voxels) - or the windows-everywhere
    mode, in which every decode does (`windows`) - no gradients, not training, one item, a backend whose rows this
    module reproduces (`backend`)."""
    if not (chunk_voxels(decoder) or windows()) or torch.is_grad_enabled() or decoder.training:
        return False
    if not backend():
        return False
    return x.shape[0] == 1


def _spconv() -> bool:
    return backend() == "spconv"


def slices(up: Any, x: sp.SparseTensor) -> bool:
    """Whether a whole level x's up-block conv1 may run a slab of parents at a time with the whole call's values: on
    flex_gemm where its split-K wrapper would not split the reference's call (reference_splitk); on spconv always
    (the implicit GEMM's rows do not depend on the call; the products' offsets that pair fewer rows than the floor
    are taken from the level's own calls: up_block, conv_spconv.whole_offset_products)."""
    if _spconv():
        return True
    return reference_splitk(x.feats.shape[0], up.channels, 8 * up.out_channels) == 1


def min_call_rows() -> int:
    """The smallest Linear call of rowwise's even windows: MIN_CALL_ROWS, whatever the backend and the record (the
    decoders' whole levels on spconv make whole-level calls instead, `whole_level_calls`; the other callers of
    rowwise - lean_encode's encoder, Hi3D's shape encoder - keep their own windows: a record's floor never moves
    them)."""
    return MIN_CALL_ROWS


def box_window() -> int:
    """The rows of every Linear call inside a box: BOX_WINDOW (on spconv the install's sweep checked exactly these
    calls against whole-level ones: products_floor.exact)."""
    return BOX_WINDOW


def windows() -> Optional[int]:
    """On spconv where the install found a box shape with no floor: the rows of every GEMM call of the decoders
    (products_floor.window: the windows-everywhere mode, whole decodes included), else None."""
    if _spconv():
        from ...modules.sparse.conv import products_floor
        return products_floor.window()
    return None


def _linear(layer: Any, feats: torch.Tensor) -> torch.Tensor:
    """layer's rows of feats (a Linear layer, a SparseLinear's module) in the windows-everywhere mode's calls."""
    out = torch.empty((feats.shape[0], layer.out_features), dtype=feats.dtype, device=feats.device)
    rowwise(lambda rows: torch.nn.Linear.forward(layer, rows), feats, out, windows(), pad=True)
    return out


def whole_level_calls() -> bool:
    """On spconv the Linear layers of a level decoded whole make the reference's own calls (one call per layer on the
    whole level, as SparseUnetVaeDecoder.forward's loop makes them): cuBLAS's fp16 GEMMs without cuBLASLt (the pascal
    profile) give a row other bits at 8-131 Ki rows a call than at the level's size for the coarse levels' shapes
    (K = 1024 / 2048; measured on sm_86, round O), so no window of them can stand in for the whole call. Not in the
    windows-everywhere mode, whose windows are the calls of every decode."""
    return _spconv() and not windows()


def reference_splitk(rows: int, ci: int, co: int) -> Optional[int]:
    """The SPLITK flex_gemm's split-K wrapper takes for a convolution of a whole level of `rows` rows (the
    reference's call): its cached choice for that key, or the one configuration its config function offers; None
    when it would have to measure (such a level is not split here)."""
    if rows < 1:
        return None
    module = _module("sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk")
    tuner = module.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk
    key = f"(2^{int(math.log2(rows))}, {ci}, {co}, {KERNEL_VOLUME})"
    if key in tuner.cache:
        return int(tuner.cache[key].get("SPLITK", 1))
    configs = module.sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk_configs(
        None, SimpleNamespace(shape=(co, KERNEL_VOLUME, ci)), None, SimpleNamespace(shape=(rows, KERNEL_VOLUME)),
        None, None, None)
    return int(configs[0]["SPLITK"]) if len(configs) == 1 else None


def neighbors(conv: Any, x: sp.SparseTensor) -> Any:
    """The neighbour cache of x's level for `conv`: the one conv_flex_gemm keeps in x's spatial cache, built and
    registered there the way it builds it when the level has none yet."""
    from flex_gemm.ops import spconv
    from flex_gemm.ops.spconv import SubMConv3dFunction
    from ...modules.sparse.conv import config, lean_cache
    from ...modules.sparse.conv.conv_flex_gemm import BACKWARD_ONLY

    Co, Kd, Kh, Kw, Ci = conv.weight.shape
    key = f"SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{conv.dilation}_forward"
    cache = x.get_spatial_cache(key)
    if cache is None:
        spconv.set_algorithm(config.FLEX_GEMM_ALGO)
        spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)
        shape = torch.Size([*x.shape, *x.spatial_shape])
        kernel = tuple(conv.weight.shape[1:4])
        cache = lean_cache.build(x.coords, shape, kernel, conv.dilation) if config.LEAN_NEIGHBOR_CACHE else None
        if cache is None:
            cache = SubMConv3dFunction._compute_neighbor_cache(x.coords, shape, kernel, conv.dilation)
            for name in BACKWARD_ONLY:
                if hasattr(cache, name):
                    delattr(cache, name)
        x.register_spatial_cache(key, cache)
    return cache


def subset(cache: Any, rows: Any) -> Any:
    """A level's neighbour cache restricted to `rows` (a slice, or ascending indices): their neighbour lists
    (indices into the whole level), Gray codes, and the order flex_gemm groups them in (lean_cache.build's)."""
    from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dNeighborCache

    if isinstance(rows, slice):
        nmap, gray = cache.neighbor_map[rows], cache.gray_code[rows]
    else:       # uint32 has no gather of its own: the same bits through an int32 view
        nmap, gray = cache.neighbor_map.view(torch.int32)[rows].view(torch.uint32), cache.gray_code[rows]
    from ...modules.sparse.conv import codes

    gray = gray.contiguous()
    binary = codes.binary_of(gray, KERNEL_VOLUME)       # lean_cache.build's binary code of these rows' Gray codes
    return SubMConv3dNeighborCache(neighbor_map=nmap.contiguous(), gray_code=gray,
                                   sorted_idx=torch.argsort(binary))


def conv_rows(conv: Any, x: sp.SparseTensor, rows: Any = None, reference: bool = False,
              fixed: Any = None) -> torch.Tensor:
    """conv(x).feats - of the rows `rows` only when given - by flex_gemm's masked implicit GEMM with SPLITK 1, or on
    spconv by conv_spconv.subm_rows (the level's cached pairs; `reference`: x is a whole level and the call covers
    it all: the whole-table decode's own call, not padded; `fixed`: the products' offsets taken from the whole
    level's own calls, with the whole level's row of each of x's rows)."""
    if _spconv():
        from ...modules.sparse.conv import conv_spconv
        return conv_spconv.subm_rows(conv, x, rows, reference=reference, fixed=fixed)
    cache = neighbors(conv, x)
    if rows is not None:
        cache = subset(cache, rows)
    Co, Kw, Kh, Kd, Ci = conv.weight.shape
    return _module("sparse_submanifold_conv_fwd_masked_implicit_gemm").sparse_submanifold_conv_fwd_masked_implicit_gemm(
        x.feats, conv.weight.reshape(Co, Kd * Kh * Kw, Ci), conv.bias, cache.neighbor_map, cache.sorted_idx,
        cache.valid_kernel_callback, cache.valid_kernel_seg_callback)


def whole_conv(conv: Any, x: sp.SparseTensor) -> torch.Tensor:
    """conv(x).feats the reference's way (conv_flex_gemm: the split-K wrapper and its autotuned SPLITK; on spconv
    subm_rows of every row, which is what the module's forward runs on the cached pairs)."""
    if _spconv():
        from ...modules.sparse.conv import conv_spconv
        return conv_spconv.subm_rows(conv, x, reference=True)
    return conv(x).feats


# ------------------------------------------------------------------------------------------------ the blocks

def _slab_rows(width: int, itemsize: int = 2, budget: Optional[int] = None) -> int:
    return max(1024, (SLAB_BYTES if budget is None else budget) // (width * itemsize))


def even_windows(n: int, window: int) -> Optional[List[Tuple[int, int]]]:
    """[a, b) row ranges covering n rows in calls of at most `window` rows, as equal as they can be, none smaller
    than MIN_CALL_ROWS; None when the table cannot be cut so (then rowwise's overlapping windows are used)."""
    k = -(-n // window)
    size = -(-n // k)
    last = n - (k - 1) * size
    if min(size, last) < min_call_rows():
        return None
    return [(a, min(n, a + size)) for a in range(0, n, size)]


def rowwise(fn: Callable[[torch.Tensor], torch.Tensor], src: torch.Tensor, out: torch.Tensor, window: int,
            pad: bool, into: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None) -> None:
    """out[i] = fn(src)[i] for every row i of a row-wise `fn` (Linear layers and what lies between them), in calls on
    exactly `window` rows: the last call overlaps the one before and only its new rows are kept (`out` may be `src`:
    the overlapped rows' second results are thrown away). A table of at most `window` rows is one call: on the table
    itself (`pad` False: on a whole level, the reference's own call), or on the table padded with zero rows to
    `window` rows (`pad` True: a box, so that no call is smaller than the window).

    With EVEN_WINDOWS a whole level (`pad` False) is cut into near-equal windows instead (even_windows), and
    `into(rows, out_rows)`, when given, computes fn(rows) straight into out_rows (rows and out_rows may be the same
    memory: `into` must have read its rows before it writes).

    In the windows-everywhere mode (`windows`) every call has exactly that many rows, padded."""
    n = src.shape[0]
    fixed = windows()
    if fixed:
        window, pad = fixed, True
    if EVEN_WINDOWS and not pad:
        ranges = [(0, n)] if n <= window else even_windows(n, window)
        if ranges is not None:
            for a, b in ranges:
                if into is not None:
                    into(src[a:b], out[a:b])
                else:
                    out[a:b] = fn(src[a:b])
            return
    if n <= window:
        if pad and n < window:
            buffer = src.new_zeros((window, *src.shape[1:]))
            buffer[:n] = src
            out[:n] = fn(buffer)[:n]
        else:
            out[:n] = fn(src)
        return
    for a in range(0, n, window):
        b = min(n, a + window)
        first = b - window
        out[a:b] = fn(src[first:b])[a - first:]


def norm_silu(norm: Any, feats: torch.Tensor) -> torch.Tensor:
    """F.silu(norm(feats)) (LayerNorm32, then SiLU) into a new table, a slab of rows at a time."""
    out = torch.empty_like(feats)
    for a in range(0, feats.shape[0], NORM_SLAB_ROWS):
        out[a:a + NORM_SLAB_ROWS] = F.silu(norm(feats[a:a + NORM_SLAB_ROWS]), inplace=True)
    return out


def convnext(block: Any, x: sp.SparseTensor, conv: Callable, boxed: bool, window: Optional[int] = None) -> sp.SparseTensor:
    """SparseConvNeXtBlock3d: conv, then LayerNorm and the MLP on windows of rows, in place in the conv's output
    (`rowwise`; a box's windows are BOX_WINDOW rows, a whole level's COARSE_SLAB_BYTES of hidden rows unless
    `window` says how many; on spconv a whole level's are the level itself: the reference's own calls, see
    `whole_level_calls`), then the input added in place."""
    h = conv(block.conv, x)
    if window is None:
        if boxed:
            window = box_window()
        elif whole_level_calls():
            window = max(1, h.shape[0])
        else:
            window = _slab_rows(block.mlp[0].out_features, budget=COARSE_SLAB_BYTES)
    rowwise(lambda rows: block.mlp(block.norm(rows)), h, h, window, pad=boxed,
            into=lambda rows, out: mlp_into(block, rows, out))
    h.add_(x.feats)
    return x.replace(h)


def mlp_into(block: Any, rows: torch.Tensor, out: torch.Tensor) -> None:
    """block.mlp(block.norm(rows)) written into `out` (which may be `rows` itself): the same calls as the
    Sequential's - Linear, SiLU over its input, Linear (addmm, as F.linear calls it) - the last one with out=."""
    first, act, last = block.mlp
    if whole_level_calls():         # a whole level at once: LayerNorm32's float32 copies a slab of rows at a time
        from .sparse_unet_vae import norm_rows
        hidden = act(first(norm_rows(block.norm, rows)))
    else:
        hidden = act(first(block.norm(rows)))
    if last.bias is not None:
        torch.addmm(last.bias, hidden, last.weight.t(), out=out)
    else:
        torch.mm(hidden, last.weight.t(), out=out)


def children(coords: torch.Tensor, sub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SparseChannel2Spatial.fine_level for parents at `coords` [p, 4] with subdivision `sub` [p, 8] bool: the
    children's coords, and for each child its parent (0..p-1) and its sub-cell, in the reference's order."""
    leaves = sub.sum(dim=-1)
    subidx = sub.nonzero()[:, -1]
    new = coords.clone()
    new[:, 1:] *= 2
    new = torch.repeat_interleave(new, leaves, dim=0, output_size=subidx.shape[0])
    for i in range(3):
        new[:, i + 1] += subidx // 2 ** i % 2
    idx = torch.repeat_interleave(torch.arange(coords.shape[0], device=coords.device), leaves, dim=0,
                                  output_size=subidx.shape[0])
    return new, idx, subidx


def up_block(up: Any, x: sp.SparseTensor, sub: torch.Tensor, parents: Optional[torch.Tensor], conv2: Callable,
             sliced: bool, forget: Callable, fixed: Any = None) -> Tuple[sp.SparseTensor, torch.Tensor]:
    """SparseResBlockC2S3d on x for the parent rows `parents` (ascending local rows; None: all), x's subdivision
    `sub` [n, 8] bool: the finer level's rows - the children of those parents in the reference's order - and for
    each child its parent's position in `parents`. conv1 runs a slab of parents at a time (`sliced`; else whole,
    the reference's call) and is gathered straight into the finer level; conv2 by `conv2`. On spconv's products
    the offsets the whole level pairs fewer rows than the floor for come from the level's own calls: `fixed` (a box:
    computed on the whole level beforehand, with the level's row of each of x's rows), or made here for a whole
    level's slabs."""
    n, C = x.feats.shape
    C2 = up.out_channels
    hx = x.replace(norm_silu(up.norm1, x.feats))
    psub = sub if parents is None else sub[parents]
    pcoords = x.coords if parents is None else x.coords[parents]
    coords, pidx, subidx = children(pcoords, psub)
    p, nf = psub.shape[0], coords.shape[0]
    if nf == 0:                     # no parent is subdivided: an empty finer level
        forget(x)
        empty = torch.empty((0, C2), dtype=x.feats.dtype, device=x.device)
        if _spconv():               # spconv's tensor takes the spatial shape from its coords: none for no rows
            return SimpleNamespace(feats=empty, coords=coords), pidx
        return sp.SparseTensor(empty, coords), pidx
    ends = torch.cumsum(psub.sum(dim=-1), 0)
    hf = torch.empty((nf, C2), dtype=x.feats.dtype, device=x.device)
    step = _slab_rows(8 * C2, budget=COARSE_SLAB_BYTES if parents is None else None) if sliced else max(p, 1)
    bounds = list(range(0, p, step)) + [p]
    fine_bounds = [0] + ends[torch.tensor(bounds[1:], device=ends.device) - 1].tolist() if p else [0]
    if sliced and parents is None and len(bounds) > 2 and fixed is None and _spconv():
        from ...modules.sparse.conv import conv_spconv
        whole = conv_spconv.whole_offset_products(up.conv1, hx)
        if whole:
            fixed = (whole, torch.arange(n, device=x.device))
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if sliced:
            out = conv_rows(up.conv1, hx, slice(a, b) if parents is None else parents[a:b],
                            reference=parents is None and a == 0 and b == p, fixed=fixed)
        else:
            out = up.conv1(hx).feats if parents is None else up.conv1(hx).feats[parents]
        fa, fb = fine_bounds[k], fine_bounds[k + 1]
        if GATHER_INTO:
            torch.index_select(out.view(-1, C2), 0, (pidx[fa:fb] - a) * 8 + subidx[fa:fb], out=hf[fa:fb])
        else:
            hf[fa:fb] = out.view(-1, C2)[(pidx[fa:fb] - a) * 8 + subidx[fa:fb]]
        del out
        for c in range(fa, fb, NORM_SLAB_ROWS):
            d = min(fb, c + NORM_SLAB_ROWS)
            hf[c:d] = F.silu(up.norm2(hf[c:d]), inplace=True)
    forget(x)                       # conv1 was the coarse level's last convolution
    del hx
    fine = sp.SparseTensor(hf, coords)
    h = conv2(up.conv2, fine)
    xs = (x.feats if parents is None else x.feats[parents]).reshape(p * 8, C // 8)
    repeat = C2 // (C // 8)
    for a in range(0, nf, NORM_SLAB_ROWS):
        b = min(nf, a + NORM_SLAB_ROWS)
        h[a:b].add_(xs[pidx[a:b] * 8 + subidx[a:b]].repeat_interleave(repeat, dim=1))
    return fine.replace(h), pidx


def tail(decoder: Any, feats: torch.Tensor, dtype: torch.dtype, boxed: bool) -> torch.Tensor:
    """The decoder's tail: the float32 layer norm of every row and the output layer, on windows of rows (a whole
    level on spconv: the reference's own call, `whole_level_calls`)."""
    from .sparse_unet_vae import cast_norm_rows

    layer = decoder.output_layer
    out = torch.empty((feats.shape[0], layer.out_features), dtype=dtype, device=feats.device)
    if boxed:
        window = box_window()
    else:
        window = max(1, feats.shape[0]) if whole_level_calls() else NORM_SLAB_ROWS * 4
    rowwise(lambda rows: torch.nn.Linear.forward(layer, cast_norm_rows(rows, dtype)), feats, out, window, pad=boxed)
    return out


def to_subdiv(up: Any, y: sp.SparseTensor) -> torch.Tensor:
    """up.to_subdiv(y).feats, a box's table: box_window() rows a call."""
    out = torch.empty((y.feats.shape[0], up.to_subdiv.out_features), dtype=y.feats.dtype, device=y.device)
    rowwise(lambda rows: torch.nn.Linear.forward(up.to_subdiv, rows), y.feats, out, box_window(), pad=True)
    return out


# ------------------------------------------------------------------------------------------------ the boxes

def halo(decoder: Any, start: int) -> int:
    """Level-`start` voxels around a box that its own fine voxels depend on: back from the finest level's last
    convolution, through every 3x3x3 convolution and every subdivision after level `start`'s ConvNeXt blocks."""
    h = 1                                           # the last up-block's conv2, in voxels of the finest level
    for level in range(len(decoder.blocks) - 2, start - 1, -1):
        h = -(-h // 2) + 1                          # their parents, and the up-block's conv1 on this level
        if level > start:
            h += decoder.num_blocks[level] + 1      # this level's ConvNeXt blocks and the conv2 that made it
    return h


def distance(c: torch.Tensor, lo: List[int], hi: List[int]) -> torch.Tensor:
    """Chebyshev distance of the voxels `c` [n, 3] to the box [lo, hi) (0 inside), int32 [n]."""
    low = torch.tensor(lo, dtype=c.dtype, device=c.device)
    high = torch.tensor(hi, dtype=c.dtype, device=c.device) - 1
    return torch.maximum(low - c, c - high).clamp_(min=0).amax(dim=1)


def inside(c: torch.Tensor, lo: List[int], hi: List[int]) -> torch.Tensor:
    low = torch.tensor(lo, dtype=c.dtype, device=c.device)
    high = torch.tensor(hi, dtype=c.dtype, device=c.device)
    return ((c >= low) & (c < high)).all(dim=1)


def cut_boxes(c: torch.Tensor, margin: int, budget: int) -> List[Tuple[List[int], List[int]]]:
    """Boxes [lo, hi) covering the voxels `c` [n, 3] (level 2), none empty, each with at most `budget` voxels
    within `margin` of it where a box of side MIN_BOX allows: halved along their longest side at the median."""
    todo = [(c.min(0).values.tolist(), (c.max(0).values + 1).tolist())]
    boxes = []
    while todo:
        lo, hi = todo.pop()
        mask = inside(c, lo, hi)
        if not bool(mask.any()):
            continue
        sides = [h - l for l, h in zip(lo, hi)]
        axis = max(range(3), key=lambda a: sides[a])
        within = int(inside(c, [v - margin for v in lo], [v + margin for v in hi]).sum())
        if within <= budget or sides[axis] <= MIN_BOX:
            boxes.append((lo, hi))
            continue
        cut = int(c[mask, axis].median()) + 1
        cut = min(max(cut, lo[axis] + 1), hi[axis] - 1)
        first_hi, second_lo = list(hi), list(lo)
        first_hi[axis] = second_lo[axis] = cut
        todo.append((second_lo, hi))
        todo.append((lo, first_hi))
    return boxes


def _guide(h: sp.SparseTensor, guide: Any) -> sp.SparseTensor:
    return h.replace(guide) if isinstance(guide, torch.Tensor) else guide


def _rank(counts: torch.Tensor) -> torch.Tensor:
    """0, 1, ..., counts[i] - 1 for every i, one after the other."""
    starts = torch.cumsum(counts, 0) - counts
    total = int(counts.sum())
    return torch.arange(total, device=counts.device) - torch.repeat_interleave(starts, counts, output_size=total)


@torch.no_grad()
def forward_chunked(decoder: Any, x: sp.SparseTensor, guide_subs: Optional[List[Any]],
                    fields: Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]
                    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[sp.SparseTensor]]:
    """The decode of x as SparseUnetVaeDecoder.forward computes it, in boxes after level 2 (see the module's
    docstring). `fields(rows, coords)` turns tail rows (float32 [n, out_channels]) of fine voxels at `coords`
    [n, 4] into the tables the caller wants per voxel. Returns those tables and the coords, all in the reference's
    row order, and the subdivisions (pred_subdiv: SparseTensors of logits, one per up-block)."""
    from .sparse_unet_vae import forget_neighbors

    began = time.perf_counter()
    blocks, nb = decoder.blocks, decoder.num_blocks
    last = len(blocks) - 1                  # the finest level: no blocks of its own
    start = last - 2                        # from its up-block on, in boxes
    pred = decoder.pred_subdiv
    subs: List[sp.SparseTensor] = []
    stats: Dict[str, Any] = {"boxes": 0}

    def subdivision(i: int, h: sp.SparseTensor) -> sp.SparseTensor:
        if pred and windows():
            sub = h.replace(_linear(blocks[i][-1].to_subdiv, h.feats))
        else:
            sub = blocks[i][-1].to_subdiv(h) if pred else _guide(h, guide_subs[i])
        if pred:
            subs.append(sub)
        return sub

    if windows():
        h = x.replace(_linear(decoder.from_latent, x.feats)).type(decoder.dtype)
    else:
        h = decoder.from_latent(x).type(decoder.dtype)
    level_rows: Dict[int, int] = {}
    for i in range(start + 1):
        level_rows[i] = int(h.feats.shape[0])
        for j in range(nb[i]):
            h = convnext(blocks[i][j], h, whole_conv, False)
        sub = subdivision(i, h)
        if i == start:
            break
        up = blocks[i][-1]
        sliced = slices(up, h)
        h, _ = up_block(up, h, sub.feats > 0, None, whole_conv, sliced, forget_neighbors)
        decoder.trim()
    stats["whole_seconds"] = time.perf_counter() - began

    x2, sub2 = h, sub.feats > 0
    up2, up3 = blocks[start][-1], blocks[start + 1][-1]
    cnt2 = sub2.sum(dim=-1)
    offs3 = torch.cumsum(cnt2, 0) - cnt2
    rows2, rows3 = x2.feats.shape[0], int(cnt2.sum())
    if pred:
        sub3_all, rows4 = None, None
    else:
        guide3 = guide_subs[start + 1]
        guide3 = guide3.feats if isinstance(guide3, sp.SparseTensor) else guide3
        sub3_all = guide3 if guide3.dtype == torch.bool else guide3 > 0
        cnt3 = sub3_all.sum(dim=-1)
        offs4 = torch.cumsum(cnt3, 0) - cnt3
        rows4 = int(cnt3.sum())
    fine_rows = rows4 if rows4 is not None else int(FINE_PER_LEVEL3 * rows3)
    boxed = BOXES and chunk_voxels(decoder) > 0 and min(rows3, fine_rows) >= box_window()
    if chunk_voxels(decoder) > 0 and not boxed:
        stats["not_boxed"] = (f"a small shape: {rows3:,} level-3 rows, about {fine_rows:,} fine ones (boxes from "
                              f"{box_window():,})" if BOXES else "BOXES off")
    if _spconv():
        from ...modules.sparse.conv import products_floor
        over = products_floor.beyond({**level_rows, start + 1: rows3, start + 2: fine_rows})
        if over:
            # the planner's covers_voxels keeps such props off the boxes; a decode that gets here all the same
            # goes on (a whole decode would not fit where boxes were planned) and says so
            stats["unverified"] = [list(o) for o in over]
            print("[CHUNKED] WARNING: this decode's levels are larger than the whole calls this card's "
                  "decode_products measurement verified (" + ", ".join(f"level {lv}: {r:,} rows, verified {v:,}"
                                                                        for lv, r, v in over)
                  + "): its boxes may differ from the whole-table decode in those levels' rows", flush=True)

    if not boxed:
        # A small shape (or BOXES off): the rest whole, the reference's calls.
        sliced = slices(up2, x2)
        h, _ = up_block(up2, x2, sub2, None, whole_conv, sliced, forget_neighbors)
        del x2
        decoder.trim()
        for j in range(nb[start + 1]):
            h = convnext(blocks[start + 1][j], h, whole_conv, False)
        sub3 = subdivision(start + 1, h)
        sliced = slices(up3, h)
        h, _ = up_block(up3, h, sub3.feats > 0, None, whole_conv, sliced, forget_neighbors)
        decoder.trim()
        forget_neighbors(h)
        out = fields(tail(decoder, h.feats, x.dtype, False), h.coords)
        LAST.clear()
        LAST.update(stats, boxed=False, seconds=time.perf_counter() - began)
        return out, h.coords, subs

    fixed2 = None
    hold = False
    if _spconv():
        from ...modules.sparse.conv import conv_spconv, products_floor
        # level 2's offsets that pair fewer rows than the floor: their products from the level's own calls, which
        # every box then takes its rows of (the same bits, never a fall back to the whole table)
        whole2 = conv_spconv.whole_offset_products(up2.conv1, x2, prep=lambda rows: norm_silu(up2.norm1, rows))
        if whole2:
            fixed2 = whole2
            stats["level2_whole_offsets"] = sorted(whole2)
            stats["level2_whole_offsets_mib"] = round(sum(p.numel() * p.element_size() for _, p in whole2.values())
                                                      / 2 ** 20, 1)
        # the output layer at a recorded launch boundary: the fine rows before it are kept and it makes the whole
        # decode's own call once the fine level's size is known (the texture decoder knows it now)
        hold = not windows() and products_floor.has_boundary() and (pred or products_floor.boundary_band(rows4))
    forget_neighbors(x2)                    # the boxes build their own
    decoder.trim()
    margin = halo(decoder, start)
    budget = max(1, int(chunk_voxels(decoder) * rows2 / max(fine_rows, 1)))
    boxes = cut_boxes(x2.coords[:, 1:], margin, budget)
    stats.update(boxes=len(boxes), halo=margin, budget_level2=budget, rows=[rows2, rows3, rows4],
                 chunk_voxels=chunk_voxels(decoder), backend=backend())
    if pred:
        sub3_logits = torch.empty((rows3, 8), dtype=decoder.dtype, device=x2.device)
    pieces: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]] = []
    out: Optional[Dict[str, torch.Tensor]] = None
    decoded = 0

    def place(gpar: torch.Tensor, counts: torch.Tensor, piece: Dict[str, torch.Tensor]) -> None:
        nonlocal out
        dest = torch.repeat_interleave(offs4[gpar], counts) + _rank(counts)
        if out is None:
            out = {k: torch.empty((rows4, *v.shape[1:]), dtype=v.dtype, device=v.device) for k, v in piece.items()}
        for k, v in piece.items():
            out[k][dest] = v

    for lo, hi in boxes:
        d2 = distance(x2.coords[:, 1:], lo, hi)
        g2 = (d2 <= margin).nonzero().squeeze(1)
        dc = d2[g2]
        del d2
        xc = sp.SparseTensor(x2.feats[g2], x2.coords[g2])
        par2 = (dc <= margin - 1).nonzero().squeeze(1)
        y, pidx2 = up_block(up2, xc, sub2[g2], par2, conv_rows, True, forget_neighbors,
                            fixed=(fixed2, g2) if fixed2 else None)
        del xc
        gp2 = g2[par2]
        cnt = cnt2[gp2]
        g3 = torch.repeat_interleave(offs3[gp2], cnt) + _rank(cnt)
        del g2, dc, par2, gp2, cnt, pidx2
        if y.feats.shape[0] == 0:           # the box's level-2 voxels have no children
            decoder.trim()
            continue
        for j in range(nb[start + 1]):
            y = convnext(blocks[start + 1][j], y, conv_rows, True)
        d3 = distance(y.coords[:, 1:], [2 * v for v in lo], [2 * v for v in hi])
        if pred:
            logits = to_subdiv(up3, y)
            own = d3 == 0
            sub3_logits[g3[own]] = logits[own]
            s3 = logits > 0
            del logits, own
        else:
            s3 = sub3_all[g3]
        par3 = (d3 <= 1).nonzero().squeeze(1)
        fine, pidx3 = up_block(up3, y, s3, par3, conv_rows, True, forget_neighbors)
        del y
        own_par = d3[par3] == 0
        if fine.feats.shape[0] == 0:        # nor the level-3 ones within reach
            del fine
            decoder.trim()
            continue
        keep = own_par[pidx3].nonzero().squeeze(1)
        if hold:
            piece = {"pre": fine.feats[keep]}
        else:
            piece = fields(tail(decoder, fine.feats[keep], x.dtype, True), fine.coords[keep])
        piece["coords"] = fine.coords[keep]
        decoded += int(fine.feats.shape[0])
        gpar = g3[par3[own_par]]
        counts = s3[par3[own_par]].sum(dim=-1)
        del fine, pidx3, keep, own_par, d3, s3, par3, g3
        if pred:
            pieces.append((gpar, counts, piece))
        else:
            place(gpar, counts, piece)
        del piece, gpar, counts
        decoder.trim()

    coords2 = x2.coords                     # level 2's features are no longer needed: not beside the assembly
    del x2, h
    if pred:
        cnt3 = (sub3_logits > 0).sum(dim=-1)
        offs4 = torch.cumsum(cnt3, 0) - cnt3
        rows4 = int(cnt3.sum())
        while pieces:
            place(*pieces.pop(0))
        coords3, _, _ = children(coords2, sub2)
        subs.append(sp.SparseTensor(sub3_logits, coords3))
    coords = out.pop("coords")
    if hold:
        from ...modules.sparse.conv import products_floor
        pre = out.pop("pre")
        band = products_floor.boundary_band(rows4)
        # in the band: the output layer's one call on the whole fine level, the whole decode's own; outside it the
        # boxes' 64 Ki-row calls on the assembled rows (the same calls a box makes on its own rows)
        rows_out = tail(decoder, pre, x.dtype, boxed=not band)
        del pre
        out = fields(rows_out, coords)
        del rows_out
        stats["tail"] = "the whole call (a launch-boundary band)" if band else "64 Ki calls on the assembled rows"
    stats.update(rows=[rows2, rows3, rows4], decoded_fine_rows=decoded, overhead=decoded / max(rows4, 1),
                 seconds=time.perf_counter() - began)
    LAST.clear()
    LAST.update(stats, boxed=True)
    return out, coords, subs


def raw_fields(rows: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {"feats": rows}
