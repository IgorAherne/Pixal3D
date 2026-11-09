"""The sparse convolutions on spconv (CUDA kernels compiled for sm_61 and up): the backend of the cards below sm_80,
where flex_gemm's Triton kernels are not used (modules/precision.py).

The weight and the bias are this module's own parameters, with the names and the layout flex_gemm's backend gives
them (`weight` [Co, Kd, Kh, Kw, Ci] - spconv's KRSC layout is the same - and `bias` [Co]), so that the checkpoints
load unchanged whatever the backend. The spconv module is kept aside, unregistered (its parameters would otherwise
appear under `conv.` in the state dict), and lent the parameters at every call: the loader replaces the Parameter
objects when it assigns the weights (load_state_dict(assign=True)), so a one-off alias would go stale. Being
unregistered it never sees the network's eval(): it runs spconv's training-mode forward - the bias added to the
GEMM's fp16 output afterwards, not in its epilogue - and that is the arithmetic every path here keeps.

spconv's GEMM kernels are prebuilt for sm_60, 70, 75, 80, 86, 89 and 90 only; on another card (a GTX 10xx is sm_61)
spconv compiles them at run time through NVRTC, which cannot work from the wheel: no headers are shipped and no
include path is given ("could not open source file tensorview/core/all.h", a GTX 1080). Such a card runs spconv's
Native algorithm with the products on torch.mm, one GEMM per kernel offset (`indice_conv_torch`, put in place of
spconv's ops.indice_conv, which its module reaches for subm, strided and inverse convolutions alike).

A submanifold convolution without gradients (every convolution of the decoders: 3x3x3, stride 1) runs on a pair
table built once per level (`level_pairs`, Spellcast3D round O): spconv's implicit-GEMM builder, kept in the level's
spatial cache like flex_gemm's neighbour cache (forget_neighbors drops both), instead of a table built and thrown
away by each of the level's convolutions (40 builds a decode, 5-6.5 % of it, become 5). `subm_rows(conv, x, rows)`
is the convolution of the rows `rows` of x only (models/sc_vaes/chunked.py, the decode in boxes):

    implicit GEMM (sm_70+: spconv's prebuilt kernels)   ops.implicit_gemm on the cached pairs, of all rows or of a
        subset (their pair columns, their masks sorted again, stably). A row's sum is a fixed function of its own
        neighbours, the weights and the kernel the calibration record names (spconv_choices): a CTA ORs the masks
        of its rows and visits the set offsets in ascending order, a row lacking one adds exact zeros, split-K is 1
        in the forward pass and spconv's kernel key holds no row count - so neither the rows a row is grouped with
        nor their number changes its bits. The whole call is the module's own call with the pairs taken from the
        cache (the same builder, which is deterministic: no atomic slot allocation, a stable sort).
    products (sm_61, `products`): spconv's Native order of the sums with each offset's GEMM on torch.mm - the
        centre offset's product first, then offsets 0..26 ascending, each added into the output rows it pairs
        (one fp16 rounding per offset), then the bias - with two changes that keep every value: (i) a call that
        stands in for the whole table's (a box's, a slab's, a subset's: `reference` False) has no product call of
        fewer than `products_floor.floor_for(shape)` rows (M0, 8,192 unless the install measured the shape higher): a
        smaller call is padded with zero rows and only its real rows are used, because cuBLAS chooses its kernel
        by the row count and some kernels round differently (sliced-K, split-K: measured on sm_86, round O), while
        from M0 rows on every call of the shapes the boxes run gives a row the same bits; the whole table's own
        calls are not padded: they are today's calls on any card; an offset the whole level pairs fewer rows for
        than M0 is that level's own call once (`whole_offset_products`), whose rows each box or slab adds in at
        that offset's turn (`products`' `fixed`); (ii) the accumulation without atomics: the
        rows an offset pairs are distinct (a submanifold row has one neighbour per offset), so out[r] + p is
        gathered, added and written back - the one IEEE fp16 addition index_add_'s atomicAdd makes (torch's
        Pascal atomicAdd is a CAS loop around the same float sum rounded to fp16: fp32 holds the sum of two fp16
        numbers exactly enough, 24 >= 2 * 11 + 2 bits, so the rounding is the correct one either way) - on a
        card below sm_70; from sm_70 on index_add_'s native fp16 atomics stay (faster there). The pairs
        an offset makes are taken in ascending order of the output row instead of the Native builder's (whose
        order its atomics set): a row's bits do not depend on its place in a cuBLAS call.

PIXAL3D_SPCONV_PAIRS=off sends every convolution of a whole-table decode through spconv's module as before (its pairs
built per call, the products with index_add_): the way back, and the "before" of the measurements;
PIXAL3D_SPCONV_BOXES=off keeps spconv decodes out of boxes (chunked.backend: the 1024 rule of small cards comes back);
PIXAL3D_SPCONV_ACCUMULATE=atomic keeps index_add_ in the products (a measurement).
"""
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from .. import SparseTensor
from ... import precision
from . import config, products_floor, spconv_choices
import spconv.pytorch as spconv
from spconv.pytorch import ops as spconv_ops


def _switch(name: str, off: Tuple[str, ...] = ("off", "0", "false", "no")) -> bool:
    return os.environ.get(name, "").strip().lower() not in off


PAIRS = _switch("PIXAL3D_SPCONV_PAIRS")                                  # the per-level pair table (above)
BOXES = _switch("PIXAL3D_SPCONV_BOXES")                                  # the decode in boxes on spconv
# The products' accumulation: None = by the card (the atomic-free sum below sm_70, where torch's fp16 atomicAdd is a
# CAS loop; index_add_ from sm_70 on, whose native fp16 atomics were faster on an RTX 3060: the ork's decode 14.0 s
# against 16.1 s); True / False = PIXAL3D_SPCONV_ACCUMULATE=free / atomic. Both give the same bits.
_ACCUMULATE = os.environ.get("PIXAL3D_SPCONV_ACCUMULATE", "").strip().lower()
ATOMIC_FREE: Optional[bool] = True if _ACCUMULATE == "free" else False if _ACCUMULATE == "atomic" else None
_cas_atomics: Optional[bool] = None
CACHE_KEY = "SubMConv3d_neighbor_cache_spconv"      # the prefix forget_neighbors (sparse_unet_vae.py) drops
FULL_MASK = 0xffffffff                              # MaskImplicitGemm's one split: every offset
ACCUMULATE_BYTES = 32 << 20                         # the atomic-free accumulation's gathered rows at a time
ACCUMULATE_MIN_ROWS = 4096


def _own(self, conv) -> None:
    """Take `conv`'s weight and bias as this module's parameters and keep `conv` aside."""
    want = (conv.out_channels, *conv.kernel_size, conv.in_channels)
    if tuple(conv.weight.shape) != tuple(want):
        raise RuntimeError(f"spconv's weight layout {tuple(conv.weight.shape)} is not KRSC {tuple(want)}")
    self.weight = nn.Parameter(conv.weight.detach())
    if conv.bias is not None:
        self.bias = nn.Parameter(conv.bias.detach())
    else:
        self.register_parameter("bias", None)
    conv._parameters.pop("weight", None)
    conv._parameters.pop("bias", None)
    object.__setattr__(self, "_conv", conv)
    self.in_channels, self.out_channels = conv.in_channels, conv.out_channels


def _lent(self):
    """The spconv module with this module's parameters in place."""
    conv = self._conv
    conv._parameters["weight"] = self.weight
    conv._parameters["bias"] = self.bias
    return conv


def indice_conv_torch(features, filters, indice_pairs, indice_pair_num, num_activate_out, inverse=False, subm=False,
                      algo=None, timer=None, bias=None, act_alpha=0.0, act_beta=0.0, act_type=None):
    """spconv's gather - product - scatter convolution (ops.indice_conv, its Native algorithm) with the products on
    torch.mm: for every kernel offset, the rows that offset pairs times the offset's [Ci, Co] slice of the KRSC
    weight, summed into the output rows (fp32 rounding order apart, spconv's kernels' sums)."""
    if act_type is not None and act_type != spconv_ops.tv.gemm.Activation.None_:
        raise NotImplementedError("a fused activation")
    co, ci = filters.shape[0], filters.shape[-1]
    w = filters.reshape(co, -1, ci)                         # [Co, kv, Ci]
    kv = w.shape[1]
    centre = kv // 2
    counts = indice_pair_num.tolist()                       # the host sync spconv's own path has
    pair_in, pair_out = indice_pairs[int(inverse)], indice_pairs[int(not inverse)]
    out = torch.mm(features, w[:, centre].t()) if subm else features.new_zeros((num_activate_out, co))
    for i, n in enumerate(counts):
        if subm and i == centre:
            continue
        if subm and i > centre:                             # a subm's far half mirrors the near half's pairs
            n = counts[kv - i - 1]
        if n <= 0:
            continue
        rows = features.index_select(0, pair_in[i, :n].long())
        out.index_add_(0, pair_out[i, :n].long(), torch.mm(rows, w[:, i].t()))
    if bias is not None:
        out += bias
    return out


_torch_products: Optional[bool] = None


def _products_in_torch() -> bool:
    """True on a card whose spconv GEMM kernels are not prebuilt (the module's docstring): the first call puts
    indice_conv_torch in place of spconv's ops.indice_conv, which spconv's functional module looks up at each call."""
    global _torch_products
    if _torch_products is None:
        from spconv.core_cc.cumm.common import CompileInfo
        cc = precision.profile().capability
        _torch_products = cc is not None and not CompileInfo.arch_is_compiled_gemm(tuple(cc))
        if _torch_products:
            spconv_ops.indice_conv = indice_conv_torch
    return _torch_products


def _algo():
    if config.SPCONV_ALGO == 'native' or _products_in_torch():
        return spconv.ConvAlgo.Native
    if config.SPCONV_ALGO == 'implicit_gemm':
        return spconv.ConvAlgo.MaskImplicitGemm
    return None


def path() -> str:
    """How this process's submanifold convolutions run on the cached pairs: "products" (torch.mm, the cards without
    spconv's GEMM kernels), "igemm" (spconv's implicit GEMM), or "" (spconv's own Native kernels, asked for by
    SPCONV_ALGO=native: not on the cached pairs)."""
    if _products_in_torch():
        return "products"
    return "" if config.SPCONV_ALGO == 'native' else "igemm"


def boxes_supported() -> bool:
    """Whether a decode on this backend can run in boxes (models/sc_vaes/chunked.py): its convolutions have a
    `subm_rows` that gives a row the whole table's bits."""
    if not BOXES:
        return False
    try:
        return bool(path())
    except Exception:       # noqa: BLE001 - spconv without its compiled core: no boxes
        return False


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    algo = _algo()
    if stride == 1 and (padding is None):
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo)
    else:
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias, indice_key=indice_key, algo=algo)
    _own(self, conv)
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
    self.padding = padding


# ------------------------------------------------------------------------------------------------ the pair table

class LevelPairs:
    """One level's submanifold pair table (`level_pairs`). For the implicit GEMM: `pair_fwd` [kv, n] int32 (the
    input row of output row r's neighbour at offset k, -1 where it has none), `mask_sorted` [n, 1] int32 (each
    row's offsets as bits, in ascending order of that number: the builder's stable sort) and `argsort` [n] int32
    (the rows in that order). For the products: `offsets[k]` = (output rows ascending, their neighbours' input rows),
    both int32, for every offset but the centre (None there), and no pair_fwd."""

    def __init__(self, n: int, kv: int, pair_fwd: Optional[torch.Tensor] = None,
                 mask_sorted: Optional[torch.Tensor] = None, argsort: Optional[torch.Tensor] = None,
                 offsets: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None) -> None:
        self.n, self.kv = n, kv
        self.pair_fwd, self.mask_sorted, self.argsort, self.offsets = pair_fwd, mask_sorted, argsort, offsets
        self.masks = [np.array([FULL_MASK], dtype=np.uint32)]

    def nbytes(self) -> int:
        tensors = [self.pair_fwd, self.mask_sorted, self.argsort]
        for pair in self.offsets or []:
            if pair is not None:
                tensors.extend(pair)
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def offset_lists(pair_fwd: torch.Tensor) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """The products' per-offset pairs of an implicit-GEMM pair table [kv, n]: for each offset but the centre, the
    output rows that have a neighbour there (ascending) and those neighbours' input rows, int32."""
    kv = pair_fwd.shape[0]
    out: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
    for k in range(kv):
        if k == kv // 2:
            out.append(None)
            continue
        dest = (pair_fwd[k] >= 0).nonzero().squeeze(1)
        out.append((dest.to(torch.int32), pair_fwd[k].index_select(0, dest).contiguous()))
    return out


def _pairs_key(conv) -> str:
    sc = conv._conv
    return f"{CACHE_KEY}_{'x'.join(map(str, sc.kernel_size))}_dilation{tuple(sc.dilation)}_{path()}"


def build_pairs(conv, x: SparseTensor) -> LevelPairs:
    """x's level's pair table for `conv`'s kernel: spconv's implicit-GEMM pair builder for an inference subm
    (ops.get_indice_pairs_implicit_gemm, as spconv's module calls it with is_train False; its kernels are compiled
    for sm_50-90, sm_61 included); for the products, the per-offset lists made of it."""
    sc = conv._conv
    indices = x.coords if x.coords.is_contiguous() else x.coords.contiguous()
    res = spconv_ops.get_indice_pairs_implicit_gemm(
        indices, x.data.batch_size, x.data.spatial_shape, spconv.ConvAlgo.MaskImplicitGemm, sc.kernel_size,
        sc.stride, sc.padding, sc.dilation, sc.output_padding, subm=True, transpose=False, is_train=False)
    pair_fwd = res[2]
    n, kv = int(indices.shape[0]), int(pair_fwd.shape[0])
    if path() == "products":
        offsets = offset_lists(pair_fwd)
        del res, pair_fwd
        return LevelPairs(n, kv, offsets=offsets)
    return LevelPairs(n, kv, pair_fwd=pair_fwd.contiguous(), mask_sorted=res[4][0], argsort=res[6][0])


def level_pairs(conv, x: SparseTensor) -> LevelPairs:
    """The pair table of x's level for `conv`'s kernel: the one kept in x's spatial cache (shared by every tensor of
    the level), built and registered there when the level has none yet."""
    key = _pairs_key(conv)
    found = x.get_spatial_cache(key)
    if found is None:
        found = build_pairs(conv, x)
        x.register_spatial_cache(key, found)
    return found


# ------------------------------------------------------------------------------------------------ the convolution

def _rows_index(rows: Any, n: int, device: Any) -> Optional[torch.Tensor]:
    """`rows` (None, a slice or ascending indices) as int64 indices, None for all n rows in order."""
    if rows is None:
        return None
    if isinstance(rows, slice):
        a, b, step = rows.indices(n)
        if a == 0 and b == n and step == 1:
            return None
        return torch.arange(a, b, step, device=device)
    rows = rows.to(device=device, dtype=torch.long)
    if rows.shape[0] == n and n and bool((rows == torch.arange(n, device=device)).all()):
        return None
    return rows


def _call(rows: torch.Tensor, wt: torch.Tensor, floor: int, window: Optional[int] = None) -> torch.Tensor:
    """rows @ wt, on no fewer than `floor` rows a call (zero rows below them, only the real rows kept), or in calls
    of exactly `window` rows when one is given (the last one padded)."""
    n = rows.shape[0]
    if window:
        out = rows.new_empty((n, wt.shape[1]))
        for a in range(0, n, window):
            b = min(n, a + window)
            out[a:b] = _call(rows[a:b], wt, window)
        return out
    if n >= floor:
        return torch.mm(rows, wt)
    padded = rows.new_zeros((floor, rows.shape[1]))
    padded[:n] = rows
    return torch.mm(padded, wt)[:n]


def _product(features: torch.Tensor, index: Optional[torch.Tensor], wt: torch.Tensor, floor: int,
             window: Optional[int] = None) -> torch.Tensor:
    """features[index] @ wt (features itself for index None) on a call of at least `floor` rows: the rows gathered
    straight into a zero-padded block of `floor` rows when they are fewer (or in `window`-row calls)."""
    if index is None:
        return _call(features, wt, floor, window)
    if window:
        return _call(features.index_select(0, index), wt, floor, window)
    n = index.shape[0]
    if n >= floor:
        return torch.mm(features.index_select(0, index), wt)
    padded = features.new_zeros((floor, features.shape[1]))
    torch.index_select(features, 0, index, out=padded[:n])
    return torch.mm(padded, wt)[:n]


def _accumulate(out: torch.Tensor, dest: torch.Tensor, product: torch.Tensor) -> None:
    """out[dest] += product for distinct rows `dest`: without atomics (gather, add, write back: the same fp16
    addition), or index_add_ as before (ATOMIC_FREE: by the card when None, `_atomic_free`)."""
    dest = dest.long()
    if _atomic_free(out.device):
        step = max(ACCUMULATE_MIN_ROWS, ACCUMULATE_BYTES // (product.shape[1] * product.element_size()))
        for a in range(0, dest.shape[0], step):         # the gathered rows a slab at a time: no table of the size
            rows, part = dest[a:a + step], product[a:a + step]      # of the product beside it
            part.add_(out.index_select(0, rows))
            out.index_copy_(0, rows, part)
    else:
        out.index_add_(0, dest, product)


def _atomic_free(device: Any) -> bool:
    """Whether the products add without atomics: ATOMIC_FREE when set, else on a card below sm_70 (the real card's
    capability, not the profile's: a forced pascal profile on an Ampere card keeps its native fp16 atomics)."""
    global _cas_atomics
    if ATOMIC_FREE is not None:
        return ATOMIC_FREE
    if _cas_atomics is None:
        try:
            _cas_atomics = tuple(torch.cuda.get_device_capability(device)) < (7, 0)
        except Exception:       # noqa: BLE001 - no card to ask (the CPU): the atomic-free sum
            _cas_atomics = True
    return _cas_atomics


def _window() -> Optional[int]:
    """The windows-everywhere mode's call size, only where the decode in boxes is on (PIXAL3D_SPCONV_BOXES=off: the
    unwindowed whole-table decode, today's)."""
    return products_floor.window() if BOXES else None


Fixed = Tuple[dict, torch.Tensor]     # ({offset: (the whole level's output rows ascending, their product)}, the
#                                       global row of each local output row). A product may lie in RAM (a band's whole
#                                       call, whole_offset_products): its rows are copied to the card per call
BAND_ROOM_MARGIN = 256 << 20           # free under the cap beside a band's whole call (whole_offset_products)
BAND_NOTES: List[str] = []             # what the last whole_offset_products calls left to the boxes' own calls, for
#                                        chunked's stats (cleared by the caller)


def card_room(device: Any) -> Optional[int]:
    """Bytes PyTorch may still allocate on `device`: the cap the staged loader set (set_per_process_memory_fraction)
    less what is allocated, and at most what the card has free plus PyTorch's idle reserve; None off the card."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    room = free + torch.cuda.memory_reserved(device) - allocated
    try:
        fraction = torch.cuda.memory.get_per_process_memory_fraction(device)
        room = min(room, int(fraction * total) - allocated)
    except Exception:       # noqa: BLE001 - a torch without the question: the card's own figure
        pass
    return int(room)


def fits(need: int, device: Any) -> bool:
    """Whether `need` bytes fit the card's room now with BAND_ROOM_MARGIN beside them (always off the card)."""
    room = card_room(device)
    return room is None or need + BAND_ROOM_MARGIN <= room


def _rows_of(product: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """product[index] on index's device: a product kept in RAM is gathered there and copied (the same values)."""
    if product.device == index.device:
        return product.index_select(0, index)
    return product.index_select(0, index.to(product.device)).to(index.device)


def _hits(fixed: Fixed, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(this call's output rows that offset k's whole-level entry covers, their rows in the entry)."""
    whole_dest = fixed[0][k][0]
    at = torch.searchsorted(whole_dest, fixed[1]).clamp_(max=max(whole_dest.shape[0] - 1, 0))
    hit = (whole_dest.index_select(0, at) == fixed[1]).nonzero().squeeze(1) if whole_dest.shape[0] else at[:0]
    return hit, at.index_select(0, hit)


def products(features: torch.Tensor, filters: torch.Tensor, pairs: LevelPairs, rows: Optional[torch.Tensor] = None,
             bias: Optional[torch.Tensor] = None, floor: Optional[int] = None,
             window: Optional[int] = None, fixed: Optional[Fixed] = None) -> torch.Tensor:
    """The submanifold convolution of the output rows `rows` (int64, ascending; None: all) in spconv's Native order
    with the products on torch.mm (the module's docstring): the centre offset's product, then offsets 0..kv-1
    ascending added in, then the bias. `floor`: the smallest product call (1: none; None: the install's for this
    shape, products_floor.floor_for); `window`: every product call of exactly that many rows (0: none; None: the install's
    windows-everywhere mode, `_window`). `fixed`: offsets whose product comes from the whole level's own call
    (whole_offset_products: the offsets that pair fewer rows than the floor there, or a count in a launch-boundary
    band), with the global row of each of this call's output rows - added in at their offset's turn, in place of this
    table's own product; the centre's entry (a level whose own count is in a band) replaces the rows it covers of this
    call's centre product before any offset is added."""
    co, ci = filters.shape[0], filters.shape[-1]
    floor = products_floor.floor_for("products", ci, co) if floor is None else max(1, floor)
    window = _window() if window is None else (window or None)
    w = filters.reshape(co, -1, ci)                         # [Co, kv, Ci], the views indice_conv_torch multiplies by
    kv = w.shape[1]
    centre = kv // 2
    out = _product(features, rows, w[:, centre].t(), floor, window)
    if fixed is not None and centre in fixed[0]:
        hit, at = _hits(fixed, centre)
        if hit.shape[0]:
            out.index_copy_(0, hit, _rows_of(fixed[0][centre][1], at))
    if rows is not None:
        where = torch.full((pairs.n,), -1, dtype=torch.int32, device=features.device)
        where[rows] = torch.arange(rows.shape[0], dtype=torch.int32, device=features.device)
    for k in range(kv):
        if k == centre:
            continue
        if fixed is not None and k in fixed[0]:
            hit, at = _hits(fixed, k)
            if hit.shape[0]:
                _accumulate(out, hit, _rows_of(fixed[0][k][1], at))
            continue
        dest, src = pairs.offsets[k]
        if rows is not None:
            local = where.index_select(0, dest)
            keep = (local >= 0).nonzero().squeeze(1)
            dest, src = local.index_select(0, keep), src.index_select(0, keep)
            del local, keep
        if dest.shape[0] == 0:
            continue
        _accumulate(out, dest, _product(features, src, w[:, k].t(), floor, window))
    if bias is not None:
        out += bias
    return out


def _sorted_masks(pairs: LevelPairs, rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The masks of the rows `rows` sorted again (stably, as the builder sorts them): [m, 1] and their order [m]."""
    masks = torch.empty((pairs.n,), dtype=torch.int32, device=rows.device)
    masks[pairs.argsort.long()] = pairs.mask_sorted[:, 0]           # each row's own mask back in row order
    values, order = torch.sort(masks.index_select(0, rows), stable=True)
    return values.reshape(-1, 1).contiguous(), order.to(torch.int32)


def implicit_gemm(conv, features: torch.Tensor, pairs: LevelPairs, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The submanifold convolution of the output rows `rows` (None: all) by spconv's implicit GEMM on the cached
    pairs, with the bias added afterwards as spconv's training-mode forward adds it (the module's docstring)."""
    sc = conv._conv
    if rows is None:
        pair_fwd, mask, order, n = pairs.pair_fwd, pairs.mask_sorted, pairs.argsort, pairs.n
    else:
        pair_fwd = pairs.pair_fwd.index_select(1, rows).contiguous()
        mask, order = _sorted_masks(pairs, rows)
        n = int(rows.shape[0])
    if n == 0:
        return features.new_empty((0, conv.out_channels))
    out, _, _ = spconv_ops.implicit_gemm(features, conv.weight, pair_fwd, [mask], [order], n, pairs.masks, False, True,
                                         fp32_accum=sc.fp32_accum, bias=None, act_alpha=sc.act_alpha,
                                         act_beta=sc.act_beta, act_type=sc.act_type)
    if conv.bias is not None:
        out += conv.bias
    return out


def subm_rows(conv, x: SparseTensor, rows: Any = None, reference: bool = False,
              fixed: Optional[Fixed] = None) -> torch.Tensor:
    """conv(x).feats - of the rows `rows` only (a slice, or ascending indices into x's rows) when given - on x's
    level's cached pairs: the implicit GEMM or the products (path()), every row with the whole table's bits.
    `reference`: x is a whole level and this is the whole-table decode's own call - its products are not padded
    (the floor is for the calls that stand in for it: boxes, slabs, subsets; the windows-everywhere mode cuts every
    call alike). `fixed`: products' offsets taken from the whole level's own calls (`products`)."""
    spconv_choices.restore(x.device)
    _note_choices()
    features = x.feats if x.feats.is_contiguous() else x.feats.contiguous()
    index = _rows_index(rows, features.shape[0], features.device)
    pairs = level_pairs(conv, x)
    if path() == "products":
        if index is not None and index.shape[0] == 0:
            return features.new_empty((0, conv.out_channels))
        if fixed is not None and index is not None:
            fixed = (fixed[0], fixed[1].index_select(0, index))
        return products(features, conv.weight, pairs, index, conv.bias, floor=1 if reference else None,
                        fixed=fixed)
    return implicit_gemm(conv, features, pairs, index)


def whole_offset_products(conv, x: SparseTensor, prep: Any = None, floor: Optional[int] = None) -> dict:
    """The offsets of x's whole level that pair fewer rows than the shape's floor (0 < n < floor), or a count of rows
    in one of the shape's launch-boundary bands (products_floor.in_band: the whole call's last rows are another
    kernel's there), each with the whole-table decode's own product of it: {offset: (output rows ascending, int64;
    prep(rows) @ W_offset, the unpadded call)}. A box or a slab of that level then takes those rows from here
    (`products`' `fixed`) instead of a call of its own, whose rows could be another kernel's: the same bits. Under
    the floor each entry stays on the card (under floor x Co rows). A band's entry - its whole product, n x Co (2 GiB
    for 2^21 rows of 128 -> 512), a count in about one in a thousand of a million rows - is copied to RAM once made
    (pageable: the boxes gather their rows there and copy them up, the same values), so the card holds it only while
    the call is made; and that call is made only where the card has room for its input and output beside what it
    holds (card_room): else the offset is left to the boxes' own calls, whose rows in the band's tail may then differ,
    and BAND_NOTES and a warning say so. The same for the level's own centre call where the level's row count is in a
    band (its rows from products_floor.band_from on). `prep`: what the convolution's input is made of the level's rows
    (a row-wise function: the up-block's norm and SiLU). Empty on the implicit GEMM and in the windows-everywhere
    mode."""
    if path() != "products" or _window():
        return {}
    filters = conv.weight
    co, ci = filters.shape[0], filters.shape[-1]
    floor = products_floor.floor_for("products", ci, co) if floor is None else floor
    pairs = level_pairs(conv, x)
    w = filters.reshape(co, -1, ci)
    itemsize = x.feats.element_size()
    found = {}

    def banded(n: int, what: str) -> bool:
        """Whether the band's whole call of n rows fits the card now (else noted, and left to the boxes)."""
        need = n * (max(2 * ci, ci + co) if prep is not None else ci + co) * itemsize
        if fits(need, x.feats.device):
            return True
        note = (f"{what} of {n:,} rows ({ci} -> {co}) is in a launch-boundary band, but its whole call "
                f"({need / 2 ** 20:,.0f} MiB) does not fit the card now: the boxes' own calls instead (its last rows "
                f"may differ from the whole-table decode)")
        BAND_NOTES.append(note)
        print(f"[SPCONV] WARNING: {note}", flush=True)
        return False

    for k, pair in enumerate(pairs.offsets):
        if pair is None or not pair[0].shape[0]:
            continue
        n = int(pair[0].shape[0])
        band = n >= floor
        if band and not products_floor.in_band("products", ci, co, n):
            continue
        if band and not banded(n, f"offset {k}"):
            continue
        dest, src = pair
        rows = x.feats.index_select(0, src)
        if prep is not None:
            rows = prep(rows)
        product = torch.mm(rows.contiguous(), w[:, k].t())
        del rows
        found[k] = (dest.long(), product.cpu() if band else product)
        del product
    total, centre = int(x.feats.shape[0]), w.shape[1] // 2
    lo = products_floor.band_from("products", ci, co, total) if total else None
    if lo is not None and banded(total, "the centre"):
        rows = x.feats if prep is None else prep(x.feats)
        product = torch.mm(rows.contiguous(), w[:, centre].t())
        del rows
        found[centre] = (torch.arange(lo, total, device=x.feats.device), product[lo:].cpu())
        del product
    return found


_noted = False


def _note_choices() -> None:
    """Once a process, loudly: spconv's implicit GEMM without the decoders' recorded kernel choices (no record, one
    from before round N, or another card's) - spconv then times every candidate kernel at the first call of each
    shape in the process (232-433 s of the first 1536 decode on an RTX 3060, round O R1), and a decode has that
    process's values, the boxes included. And the windows-everywhere mode, where the record names it."""
    global _noted
    if _noted:
        return
    _noted = True
    if path() == "igemm" and spconv_choices.applies() and not spconv_choices.restore():
        print("[SPCONV] WARNING: no recorded spconv kernel choices for the decoders (the calibration's "
              "spconv_decoders section): spconv times its kernels at the first call of each shape now - minutes at "
              "1536 - and this process's meshes may differ from another's; run the calibration again", flush=True)
    if path() and BOXES:
        problem = products_floor.problem()
        if problem:
            print(f"[SPCONV] WARNING: this card has no usable decode_products measurement ({problem}): no decode "
                  f"in boxes on it, so a 1536 prop that does not fit whole is made at 1024; measure it with "
                  f"`python -m pixal3d.calibrate --version <N> --only decode_products` (the calibration's "
                  f"\"Measure the graphics card again\")", flush=True)
        elif products_floor.box_level() is None:
            print(f"[SPCONV] WARNING: no decode in boxes on this card ({products_floor.plan()['why']}): a 1536 prop "
                  f"that does not fit whole is made at 1024", flush=True)
        elif products_floor.box_level() == 4 and not _window():
            print("[SPCONV] the decode in boxes starts at level 4 on this card (level 3 decoded whole: "
                  + (products_floor.plan().get("why_not_3") or "; ".join(
                      f"{w['kind']} {w['ci']} -> {w['co']} {w['why']}" for w in products_floor.plan()["whole_only"]))
                  + ")", flush=True)
    window = _window()
    if window:
        print(f"[SPCONV] every GEMM of the decoders in windows of {window:,} rows (the windows-everywhere mode, "
              f"forced by PIXAL3D_DECODE_WINDOW)", flush=True)


def counts_at_least(conv, x: SparseTensor, floor: Optional[int] = None) -> bool:
    """Whether every offset of x's level that pairs rows at all pairs at least `floor` rows (products_floor's by
    default): the whole table's own product calls are then all in the range where a padded box or slab call gives
    a row their bits (an offset without pairs makes no call on either side). True on the implicit GEMM (no row count
    matters there) and in the windows-everywhere mode."""
    if path() != "products" or _window():
        return True
    if floor is None:
        floor = products_floor.floor_for("products", conv.weight.shape[-1], conv.weight.shape[0])
    pairs = level_pairs(conv, x)
    return all(pair[0].shape[0] == 0 or pair[0].shape[0] >= floor for pair in pairs.offsets if pair is not None)


def _on_pairs(self, x: SparseTensor) -> bool:
    """Whether this convolution of x runs on the level's cached pairs: a submanifold one (stride 1, no padding)
    without gradients, on the card, on a path that has them."""
    return (PAIRS and self._conv.subm and not torch.is_grad_enabled() and x.feats.is_cuda and bool(path())
            and self._conv.algo in (spconv.ConvAlgo.MaskImplicitGemm, spconv.ConvAlgo.Native))


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    spconv_choices.restore(x.device)
    if _on_pairs(self, x):
        return x.replace(subm_rows(self, x, reference=True))
    spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
    new_data = _lent(self)(x.data)
    new_shape = [x.shape[0], self.out_channels]
    new_layout = None if spatial_changed else x.layout

    if spatial_changed and (x.shape[0] != 1):
        # spconv was non-1 stride will break the contiguous of the output tensor, sort by the coords
        fwd = new_data.indices[:, 0].argsort()
        bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
        sorted_feats = new_data.features[fwd]
        sorted_coords = new_data.indices[fwd]
        unsorted_data = new_data
        new_data = spconv.SparseConvTensor(sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size)  # type: ignore

    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )

    if spatial_changed and (x.shape[0] != 1):
        out.register_spatial_cache(f'conv_{self.stride}_unsorted_data', unsorted_data)
        out.register_spatial_cache(f'conv_{self.stride}_sort_bwd', bwd)

    return out


def sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
    _own(self, spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key))
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    spconv_choices.restore(x.device)
    spatial_changed = any(s != 1 for s in self.stride)
    if spatial_changed:
        # recover the original spconv order
        data = x.get_spatial_cache(f'conv_{self.stride}_unsorted_data')
        bwd = x.get_spatial_cache(f'conv_{self.stride}_sort_bwd')
        data = data.replace_feature(x.feats[bwd])
    else:
        data = x.data

    new_data = _lent(self)(data)
    new_shape = [x.shape[0], self.out_channels]
    new_layout = None if spatial_changed else x.layout
    out = SparseTensor(
        new_data, shape=torch.Size(new_shape), layout=new_layout,
        scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
        spatial_cache=x._spatial_cache,
    )
    return out
