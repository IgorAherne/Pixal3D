"""NAF's neighborhood attention in plain torch, for the cards natten's kernels do not serve.

NAF's guided upsampler (its src/layers/attentions.py) lets every pixel of the target map attend to a K x K window
of keys, dilated by the ratio between the target and the low-resolution features (T = 1024 against 64: 16). It
calls natten's `na2d`, and the natten wheel Spellcast3D installs carries kernels for compute capability 8.0 and
newer only. `na2d` here gives natten's answer with torch ops, on any CUDA card and on the CPU.

natten's windows are neighbourhoods, not clamped boxes: along a dimension of length L with kernel K, half-width
n = K // 2 and dilation D, query i sees the keys start, start + D, ..., start + (K - 1) * D, where `window_start`
is natten's get_window_start. A window never leaves its query's dilation group (the positions with the same
i mod D), and near a border it shifts inward and keeps all of its K keys.

How it is computed. The queries of one dilation group are cut into blocks of BLOCK consecutive members per
dimension; the windows of a block's queries all lie within BLOCK + K - 1 consecutive members of the group, so a
block's scores are one small matrix product (BLOCK^2 queries by (BLOCK + K - 1)^2 keys, 64 x 256 for K = 9), in
which every key outside a query's own window is masked to -inf before the softmax; then one more product with
the block's values. That is about 3 times the arithmetic of the 81 dot products a query needs, done as batched
GEMMs, instead of 81 gathers of the whole map. A group shorter than BLOCK + K - 1 takes smaller blocks; the last
block of a group starts early so that it stays inside the group, and of two blocks that hold the same query only
the first writes it. Every index (the blocks' queries and keys, the masks, which block writes each position) is
made once per (length, kernel, dilation) and kept on the device; the blocks run a tile of block rows at a time,
so that a tile's temporaries stay within TILE_BYTES whatever the map's size. float32 throughout, as natten's
float32 kernel that NAF runs; no host synchronisation inside the loop.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.machinery
import sys
import types
from typing import Iterator, NamedTuple, Optional, Sequence, Tuple, Union

import torch

# Measured at T = 1024 (Spellcast3D code/devtools/pixal3d-lowmem/round-l/naf): blocks of 8 are the fastest on an
# RTX 3060 and an RTX 2060 Super (4: 1.6-2.2 times slower, 16: 1.3-1.6); tiles of 256 MiB are within 2 % of tiles
# of 512 or 1024 MiB in time and keep a call's peak 140 / 420 MiB lower.
BLOCK = 8                   # queries of a block along each dimension (fewer where a dilation group is shorter)
TILE_BYTES = 256 << 20      # a tile's temporaries: gathered queries, keys and values, scores, weights, output

Pair = Union[int, Sequence[int]]


def window_start(i: int, length: int, kernel: int, dilation: int) -> int:
    """natten's get_window_start: the first key of query i's window along a dimension of `length`."""
    n = kernel // 2
    if dilation <= 1:
        return max(i - n, 0) + (i + n >= length) * (length - i - n - 1)
    if i - n * dilation < 0:
        return i % dilation
    if i + n * dilation >= length:
        imodd = i % dilation
        a = (length // dilation) * dilation
        b = length - a
        return length - b + imodd - 2 * n * dilation if imodd < b else a + imodd - kernel * dilation
    return i - n * dilation


class _Axis(NamedTuple):
    """The blocks along one dimension."""
    block: int                  # queries of a block
    keys: int                   # keys of a block: block + kernel - 1
    count: int                  # blocks
    queries: torch.Tensor       # [count, block] positions of each block's queries
    key_pos: torch.Tensor       # [count, keys] positions of each block's keys
    bias: torch.Tensor          # [count, block, keys] 0 for the keys of the query's window, -inf for the others
    writer: torch.Tensor        # [length] block * self.block + slot of the query that writes each position
    order: torch.Tensor         # [length] the positions in the order of the blocks that write them
    starts: Tuple[int, ...]     # order[starts[b]:starts[b + 1]]: the positions block b writes


def _pair(value: Pair, name: str) -> Tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(x) for x in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must be an int or a pair, not {value!r}")
    return pair


@functools.lru_cache(maxsize=64)
def _axis(length: int, kernel: int, dilation: int, device: torch.device) -> _Axis:
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError(f"the kernel size must be odd, not {kernel}")
    if dilation < 1 or kernel * dilation > length:
        raise ValueError(f"a kernel of {kernel} dilated by {dilation} does not fit in {length} positions")
    block = max(1, min(BLOCK, length // dilation - kernel + 1))     # length // dilation: the shortest group
    keys = block + kernel - 1
    queries, key_pos, rel = [], [], []
    writer = [0] * length
    order, starts = [], [0]
    for group in range(dilation):
        members = list(range(group, length, dilation))
        size = len(members)
        # each member's window as an index into the group (natten's windows never leave the group)
        first_key = []
        for t, position in enumerate(members):
            start = window_start(position, length, kernel, dilation)
            if (start - group) % dilation or not 0 <= start <= position <= start + (kernel - 1) * dilation < length:
                raise AssertionError(f"window of {position} (length {length}, kernel {kernel}, dilation {dilation})")
            first_key.append((start - group) // dilation)
        for first in range(0, size, block):
            t0 = min(first, size - block)
            k0 = min(first_key[t0], size - keys)
            b = len(queries)
            queries.append([members[t0 + j] for j in range(block)])
            key_pos.append([members[k0 + j] for j in range(keys)])
            rel.append([first_key[t0 + j] - k0 for j in range(block)])
            for t in range(first, min(first + block, size)):
                writer[members[t]] = b * block + t - t0
                order.append(members[t])
            starts.append(len(order))
    rel_t = torch.tensor(rel, dtype=torch.int64)
    span = torch.arange(keys, dtype=torch.int64)
    inside = (span >= rel_t[..., None]) & (span < rel_t[..., None] + kernel)
    if int(rel_t.min()) < 0 or int(rel_t.max()) > keys - kernel:
        raise AssertionError(f"a window leaves its block's keys (length {length}, kernel {kernel}, dilation {dilation})")
    bias = torch.zeros(inside.shape, dtype=torch.float32).masked_fill_(~inside, float("-inf"))

    def put(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.int64).to(device)

    return _Axis(block, keys, len(queries), put(queries), put(key_pos), bias.to(device), put(writer), put(order),
                 tuple(starts))


def _flat(x: torch.Tensor) -> torch.Tensor:
    """[B, H, W, heads, c] -> [B * heads, H * W, c] float32 (a view where the layout allows it)."""
    batch, height, width, heads, channels = x.shape
    x = x.float()
    if heads == 1:
        return x.reshape(batch, height * width, channels)
    return x.permute(0, 3, 1, 2, 4).reshape(batch * heads, height * width, channels)


@torch.no_grad()
def na2d(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kernel_size: Pair, dilation: Pair,
         scale: Optional[float] = None) -> torch.Tensor:
    """natten's na2d(q, k, v, kernel_size, dilation=dilation, stride=1): q, k [B, H, W, heads, d], v [B, H, W,
    heads, c] -> [B, H, W, heads, c] in q's dtype. `scale` multiplies the scores, d ** -0.5 by default."""
    if q.dim() != 5 or k.shape != q.shape or v.dim() != 5 or v.shape[:4] != q.shape[:4]:
        raise ValueError(f"expected q, k [B, H, W, heads, d] and v [B, H, W, heads, c], got {list(q.shape)}, "
                         f"{list(k.shape)}, {list(v.shape)}")
    batch, height, width, heads, dim = q.shape
    channels = v.shape[-1]
    kernel = _pair(kernel_size, "kernel_size")
    step = _pair(dilation, "dilation")
    rows = _axis(height, kernel[0], step[0], q.device)
    cols = _axis(width, kernel[1], step[1], q.device)
    qf, kf, vf = _flat(q), _flat(k), _flat(v)
    n = qf.shape[0]
    out = qf.new_empty(n, height, width, channels)
    by, bx = rows.block, cols.block
    ky, kx = rows.keys, cols.keys
    per_block = by * bx * (dim + 2 * channels) + ky * kx * (dim + channels) + 2 * by * bx * ky * kx
    per_row = 4 * n * cols.count * per_block + 8 * cols.count * (by * bx + ky * kx)
    tile = max(1, min(rows.count, TILE_BYTES // per_row))
    scale = dim ** -0.5 if scale is None else float(scale)
    # where each position of a row lands in a tile's output [tile rows, cols.count, by, bx]
    col_slot = (cols.writer // bx) * (by * bx) + cols.writer % bx
    col_bias = cols.bias.view(1, cols.count, 1, bx, 1, kx)
    for r0 in range(0, rows.count, tile):
        r1 = min(r0 + tile, rows.count)
        blocks = (r1 - r0) * cols.count
        at = (rows.queries[r0:r1, None, :, None] * width + cols.queries[None, :, None, :]).reshape(-1)
        qb = qf.index_select(1, at).view(n, blocks, by * bx, dim).mul_(scale)
        at = (rows.key_pos[r0:r1, None, :, None] * width + cols.key_pos[None, :, None, :]).reshape(-1)
        kb = kf.index_select(1, at).view(n, blocks, ky * kx, dim)
        scores = torch.matmul(qb, kb.transpose(-1, -2))
        del qb, kb
        view = scores.view(n, r1 - r0, cols.count, by, bx, ky, kx)
        view += rows.bias[r0:r1].view(1, r1 - r0, 1, by, 1, ky, 1)
        view += col_bias
        weights = torch.softmax(scores, dim=-1)
        del scores, view
        vb = vf.index_select(1, at).view(n, blocks, ky * kx, channels)
        del at
        o = torch.matmul(weights, vb).view(n, -1, channels)
        del weights, vb
        written = rows.order[rows.starts[r0]:rows.starts[r1]]
        slot = rows.writer.index_select(0, written) - r0 * by
        row_slot = (slot // by) * (cols.count * by * bx) + (slot % by) * bx
        at = (row_slot[:, None] + col_slot[None, :]).reshape(-1)
        out.index_copy_(1, written, o.index_select(1, at).view(n, -1, width, channels))
        del o, at
    if heads == 1:
        out = out.view(batch, height, width, 1, channels)
    else:
        out = out.view(batch, heads, height, width, channels).permute(0, 2, 3, 1, 4).contiguous()
    return out.to(q.dtype)


def natten_na2d(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, kernel_size: Pair, stride: Pair = 1,
                dilation: Pair = 1, is_causal=False, scale: Optional[float] = None, **options) -> torch.Tensor:
    """`na2d` called the way natten's is (NAF's CrossAttention, naf_sliced): stride 1 and no causal mask only;
    natten's backend and tiling options mean nothing here and are ignored."""
    if _pair(stride, "stride") != (1, 1) or any(_pair(is_causal, "is_causal")):
        raise ValueError("the torch neighborhood attention has stride 1 and no causal mask only")
    for name in ("additional_keys", "additional_values"):
        if options.get(name) is not None:
            raise ValueError(f"the torch neighborhood attention takes no {name}")
    if options.get("return_lse"):
        raise ValueError("the torch neighborhood attention does not return the log-sum-exp")
    return na2d(query, key, value, kernel_size, dilation, scale)


def natten_missing() -> Optional[str]:
    """None when natten can be imported, else why not."""
    try:
        import natten  # noqa: F401
    except Exception as exc:        # noqa: BLE001 - a missing package and a library that does not load alike
        return f"{type(exc).__name__}: {str(exc)[:160]}"
    return None


@contextlib.contextmanager
def natten_stand_in() -> Iterator[Optional[str]]:
    """While NAF's code is imported: when natten itself cannot be imported, a module of that name whose `na2d` is
    natten_na2d, so that NAF's attention module (which takes natten's na2d at import, and torch.hub, which asks
    for the package) loads and holds the torch attention. Yields None when natten is there (nothing is changed),
    else the reason it is not. The stand-in is gone again afterwards: other code does not see a natten."""
    missing = natten_missing()
    if missing is None:
        yield None
        return
    stand_in = types.ModuleType("natten")
    stand_in.__spec__ = importlib.machinery.ModuleSpec("natten", None)
    stand_in.na2d = natten_na2d
    held = "natten" in sys.modules              # e.g. None there, which blocks the import
    before = sys.modules.get("natten")
    sys.modules["natten"] = stand_in
    try:
        yield missing
    finally:
        if held:
            sys.modules["natten"] = before
        else:
            sys.modules.pop("natten", None)
