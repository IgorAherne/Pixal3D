"""A SparseUnetVaeEncoder forward whose peak on the card stops growing with the prop's fine voxels: the fine level in
spatial boxes, the coarser levels on whole tables with slabbed norms and MLPs. The same latent, bit for bit.

TRELLIS.2's shape encoder reads a mesh's dual grid at the texture resolution (a knight at 1536: 6.9 M fine voxels)
and downsamples it 16x to the texture latent. The reference forward (SparseUnetVaeEncoder.forward) takes about 960
bytes of the card per fine voxel: the input layer's float32 [n, 64] table, LayerNorm32's two float32 copies of the
whole fine table, the fine level's neighbour tables kept to the end, the skip connection's re-laid [8 n_c, 64]
table, and the level-1 MLP's [n_1, 512] hidden table. This forward takes about 220:

- Level 0 (the fine voxels: no blocks of their own, one down-block) in spatial boxes of about `encoder.box_voxels`
  voxels, each with a halo of one voxel: a box's rows go through the input layer, the norm, SiLU and the down-block's
  conv1 (flex_gemm's masked implicit GEMM with SPLITK 1 on the box's own neighbour tables), and only the box's own
  rows are kept, straight into [n, C/8] tables of the whole level; the skip connection's means (4 channels of a
  voxel) are written the same way. The coarser level's layout (the cells and sub-cells SparseSpatial2Channel finds)
  is made first, from the coords alone, and the two tables are scattered into it after the boxes - a whole-level
  [n, 64] table or neighbour map never exists.
- Every down-block's second norm runs in place on slabs, and its conv2 is accumulated into the skip table a window
  of rows at a time (the level's neighbour tables built once, SPLITK 1; the whole table when the reference's own
  call would split the sum). Levels 1-4: a ConvNeXt block's norm and MLP on windows of rows, in place in its
  convolution's output (chunked.convnext); a down-block's skip connection from the per-voxel means, its first norm
  in place once the means are taken, and a level's neighbour tables dropped after its last convolution.

The values are the reference's for the reasons chunked.py gives: rows are independent through the Linear layers
(from 8 Ki rows a call; the input layer's float32 6 -> 64 measured on windows, gathered rows and 3,000 rows), the
norms and SiLU; a box's rows within the halo have all their neighbours; the masked implicit GEMM gives a row the
same sum whatever rows it is grouped with, with SPLITK 1 - the reference's own choice for the fine level's shape
(64 -> 16 channels at 2^20-2^22 rows in the calibration record); the per-voxel means are the same 4-element
reductions; the residual's add is the same elementwise sum. A single box (a small prop) takes the reference's own
calls. Measured bit-identical on the knight's and the boy's 1536 grids.
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch
import torch.nn.functional as F

from ...modules import sparse as sp
from . import chunked

BOX_VOXELS = 1 << 20        # fine voxels a box is cut for (its rows within a halo of one voxel): the default
#                             `encoder.box_voxels`; 0 turns this forward off
LINEAR_WINDOW = 1 << 18     # rows of an input-layer call inside a box (its float32 [rows, C] result precedes the
#                             fp16 copy)
CONV_WINDOW = 1 << 18       # rows of a down-block's conv2 call (its output, added into the skip table)
TRIM = True                 # PyTorch's unused blocks given back after the boxes and between the levels (no value
#                             depends on it: a level's tables would otherwise be carved out of the blocks the
#                             boxes left behind, which under a memory cap fails with memory to spare)
LAST: List[Tuple[str, int, int]] = []   # what the last forward did, phase by phase: (phase, allocated bytes at its
#                                         end, the peak during it), for measuring; filled while MEASURE is set
MEASURE = False


def _mark(phase: str) -> None:
    if MEASURE:
        LAST.append((phase, torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()))
        torch.cuda.reset_peak_memory_stats()


def applies(encoder: Any, x: sp.SparseTensor) -> bool:
    """Whether `encoder` runs x this way: box_voxels set, no gradients, not training, flex_gemm's masked split-K
    algorithm (the one whose rows this module reproduces), the fine level without blocks of its own, one item."""
    if not getattr(encoder, "box_voxels", 0) or torch.is_grad_enabled() or encoder.training or not x.coords.is_cuda:
        return False
    if encoder.num_blocks[0] != 0:
        return False
    from ...modules.sparse import config as sparse_config
    from ...modules.sparse.conv import config as conv_config

    if sparse_config.CONV != "flex_gemm" or conv_config.FLEX_GEMM_ALGO != "masked_implicit_gemm_splitk":
        return False
    return bool((x.coords[:, 0] == 0).all())


def trim() -> None:
    if TRIM:
        torch.cuda.empty_cache()


def linear_rows(layer: torch.nn.Linear, rows: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """layer(rows).type(dtype), the Linear on windows of LINEAR_WINDOW rows (no call below MIN_CALL_ROWS rows: a
    table smaller than that is padded with zero rows)."""
    n = rows.shape[0]
    out = torch.empty((n, layer.out_features), dtype=dtype, device=rows.device)
    if n < chunked.MIN_CALL_ROWS:
        padded = rows.new_zeros((chunked.MIN_CALL_ROWS, rows.shape[1]))
        padded[:n] = rows
        out[:] = torch.nn.Linear.forward(layer, padded)[:n].type(dtype)
        return out
    chunked.rowwise(lambda r: torch.nn.Linear.forward(layer, r).type(dtype), rows, out, LINEAR_WINDOW, pad=False)
    return out


def norm_silu_inplace(norm: Any, feats: torch.Tensor) -> None:
    """feats <- F.silu(norm(feats)) (LayerNorm32, then SiLU), a slab of rows at a time, in the table itself."""
    for a in range(0, feats.shape[0], chunked.NORM_SLAB_ROWS):
        b = min(feats.shape[0], a + chunked.NORM_SLAB_ROWS)
        feats[a:b] = F.silu(norm(feats[a:b]), inplace=True)


def box_conv(conv: Any, feats: torch.Tensor, coords: torch.Tensor, spatial: torch.Size) -> torch.Tensor:
    """conv over the voxels `coords` [m, 4] int32 with features `feats` [m, C] alone (a box with its halo), by
    flex_gemm's masked implicit GEMM with SPLITK 1 on neighbour tables built for these rows."""
    from flex_gemm.ops import spconv
    from flex_gemm.ops.spconv import SubMConv3dFunction

    from ...modules.sparse.conv import config, lean_cache
    from ...modules.sparse.conv.conv_flex_gemm import BACKWARD_ONLY

    spconv.set_algorithm(config.FLEX_GEMM_ALGO)
    spconv.set_hashmap_ratio(config.FLEX_GEMM_HASHMAP_RATIO)
    Co, Kw, Kh, Kd, Ci = conv.weight.shape
    shape = torch.Size([1, Ci, *spatial])
    kernel = tuple(conv.weight.shape[1:4])
    coords = coords.contiguous()
    cache = lean_cache.build(coords, shape, kernel, conv.dilation) if config.LEAN_NEIGHBOR_CACHE else None
    if cache is None:
        cache = SubMConv3dFunction._compute_neighbor_cache(coords, shape, kernel, conv.dilation)
        for name in BACKWARD_ONLY:
            if hasattr(cache, name):
                delattr(cache, name)
    module = chunked._module("sparse_submanifold_conv_fwd_masked_implicit_gemm")
    return module.sparse_submanifold_conv_fwd_masked_implicit_gemm(
        feats.contiguous(), conv.weight.reshape(Co, Kd * Kh * Kw, Ci), conv.bias, cache.neighbor_map,
        cache.sorted_idx, cache.valid_kernel_callback, cache.valid_kernel_seg_callback)


def conv_into(conv: Any, x: sp.SparseTensor, acc: torch.Tensor) -> None:
    """acc += conv(x).feats, a window of CONV_WINDOW rows at a time on the level's neighbour tables (built here, as
    the reference's call would build and keep them) with SPLITK 1; the whole table at once where the reference's
    own call splits the sum (chunked.reference_splitk), so that the rows keep its bits."""
    n = x.feats.shape[0]
    Co, Ci = conv.weight.shape[0], conv.weight.shape[-1]
    if chunked.reference_splitk(n, Ci, Co) != 1:
        acc.add_(conv(x).feats)
        return
    chunked.neighbors(conv, x)
    for a in range(0, n, CONV_WINDOW):
        b = min(n, a + CONV_WINDOW)
        acc[a:b].add_(chunked.conv_rows(conv, x, slice(a, b)))


def skip_means(feats: torch.Tensor, groups: int) -> torch.Tensor:
    """The down-block's skip connection per voxel: the mean of every `groups` consecutive channels - what
    SparseResBlockS2C3d's reshape(n_c, C_out, groups).mean(-1) of the re-laid table gives a cell's sub-cell, taken
    before the re-layout (a missing sub-cell's rows are zeros there, as they are here once scattered)."""
    n, C = feats.shape
    return feats.view(n, C // groups, groups).mean(dim=-1)


def layout(down: Any, fine: sp.SparseTensor) -> Tuple[sp.SparseTensor, torch.Tensor]:
    """The coarser level's layout from `fine`'s coords alone, the way down.updown (SparseSpatial2Channel) lays a
    table out: its SparseTensor (coords, scale, spatial cache; no features yet) and for every fine voxel its row in
    the wide [8 n_c, C] table (its cell's 8 sub-cells lie one after the other). The fine scale's cache entries and
    the coarse scale's way back (the fine coords, never used by an encoder) are dropped."""
    factor = down.updown.factor
    probe = fine.replace(torch.zeros((fine.coords.shape[0], 1), dtype=fine.dtype, device=fine.device))
    coarse = down.updown(probe)
    new_coords, idx, subidx = probe.get_spatial_cache(f"spatial2channel_{factor}")
    dest = idx * factor ** 3 + subidx
    del probe, idx, subidx
    coarse._spatial_cache.pop(str(fine._scale), None)
    coarse.get_spatial_cache().pop(f"channel2spatial_{factor}", None)
    return coarse.replace(torch.empty((new_coords.shape[0], 0), dtype=fine.dtype, device=fine.device)), dest


def relaid(table: torch.Tensor, dest: torch.Tensor, cells: int, factor: int = 2) -> torch.Tensor:
    """`table` (rows of the fine voxels) in the coarser level's layout: [cells, 8 * channels], zeros where a
    sub-cell has no voxel."""
    wide = torch.zeros((cells * factor ** 3, table.shape[1]), dtype=table.dtype, device=table.device)
    wide[dest] = table
    return wide.view(cells, -1)


def down_tail(down: Any, coarse: sp.SparseTensor, hc: torch.Tensor, skip: torch.Tensor) -> sp.SparseTensor:
    """The down-block from its re-laid tables on: norm2 and SiLU in place in `hc`, conv2 accumulated into `skip`
    (the block's residual), which becomes the coarser level's features."""
    norm_silu_inplace(down.norm2, hc)
    conv_into(down.conv2, coarse.replace(hc), skip)
    return coarse.replace(skip)


@torch.no_grad()
def forward_lean(encoder: Any, coords: torch.Tensor, rows: Callable[[Any], torch.Tensor], in_dtype: torch.dtype,
                 sample_posterior: bool = False, return_raw: bool = False) -> Any:
    """SparseUnetVaeEncoder.forward on the voxels `coords` [n, 4] int32 (one item) whose input rows `rows(g)` gives
    (float32 [len(g), in_channels] of the rows `g`, ascending indices or a slice): see the module's docstring.
    `in_dtype` is the input's dtype (the latent's)."""
    from .sparse_unet_vae import forget_neighbors

    blocks, dtype, device = encoder.blocks, encoder.dtype, coords.device
    last = len(blocks) - 1
    down = blocks[0][-1]
    n, C0, C1 = coords.shape[0], encoder.model_channels[0], encoder.model_channels[1]
    groups = C0 * 8 // C1
    xyz = coords[:, 1:]
    spatial = torch.Size((xyz.max(0)[0] + 1).tolist())
    fine = sp.SparseTensor(feats=torch.empty((n, 0), dtype=dtype, device=device), coords=coords)
    if MEASURE:
        LAST.clear()
        _mark("start")

    # --- level 0 in boxes: conv1's rows and the skip's means into tables of the whole level
    coarse, dest = layout(down, fine)
    cells = coarse.coords.shape[0]
    _mark(f"level 0 layout ({cells:,} cells)")
    boxes = chunked.cut_boxes(xyz, 1, encoder.box_voxels) if n > encoder.box_voxels else [None]
    q = torch.empty((n, down.conv1.out_channels), dtype=dtype, device=device)
    means = torch.empty((n, C0 // groups), dtype=dtype, device=device)
    _mark(f"boxes cut ({len(boxes)})")
    for box in boxes:
        if box is None:
            g, own = slice(None), slice(None)
        else:
            d = chunked.distance(xyz, *box)
            g = (d <= 1).nonzero().squeeze(1)
            own = d[g] == 0
            del d
        h = linear_rows(encoder.input_layer, rows(g), dtype)
        means[g if box is None else g[own]] = skip_means(h, groups)[own]
        a = chunked.norm_silu(down.norm1, h)
        del h
        if box is None:                                 # one box: the reference's own convolution call
            out = down.conv1(fine.replace(a)).feats
            forget_neighbors(fine)
        else:
            out = box_conv(down.conv1, a, coords[g], spatial)
        del a
        q[g if box is None else g[own]] = out[own]
        del out, g, own
        _mark("box" if box is not None else "one box")
    del fine, xyz
    trim()
    hc = relaid(q, dest, cells)
    del q
    skip = relaid(means, dest, cells)
    del means, dest
    _mark("level 0 re-laid")
    h = down_tail(down, coarse, hc, skip)
    del hc, skip, coarse
    trim()
    _mark("level 0 norm2, conv2")

    # --- levels 1 .. last on whole tables
    window = chunked._slab_rows(blocks[1][0].mlp[0].out_features) if encoder.num_blocks[1] else None
    for i in range(1, last + 1):
        level = blocks[i]
        for block in (level if i == last else level[:-1]):
            h = chunked.convnext(block, h, chunked.whole_conv, False, window)
        _mark(f"level {i} blocks ({h.feats.shape[0]:,} rows)")
        if i == last:
            break
        down = level[-1]
        groups = h.feats.shape[1] * 8 // down.out_channels
        means = skip_means(h.feats, groups)
        norm_silu_inplace(down.norm1, h.feats)          # the means are taken: nobody else reads these rows
        q = down.conv1(h).feats
        forget_neighbors(h)                             # conv1 was the level's last convolution
        coarse, dest = layout(down, h)
        cells = coarse.coords.shape[0]
        del h
        hc = relaid(q, dest, cells)
        del q
        skip = relaid(means, dest, cells)
        del means, dest
        trim()
        h = down_tail(down, coarse, hc, skip)
        del hc, skip, coarse
        _mark(f"level {i} down-block")
    forget_neighbors(h)
    z = encoder.latent(h, in_dtype, sample_posterior, return_raw)
    _mark("latent")
    return z
