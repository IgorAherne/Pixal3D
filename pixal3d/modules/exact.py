"""The conditioning and the coarse stages computed exactly: on the ampere profile, the original code's numbers bit
for bit.

The lr, hr and tex stages draw their starting noise from one generator, a row per occupied voxel in the voxels'
order, so a rounding change in the coarse stages that moves a few voxels of the sparse structure re-deals all later
noise: another model, not a nearby one. The coarse stages therefore run the original's arithmetic - the ss and lr
flows without SageAttention and the fused kernels - and so does the pictures' conditioning that every stage reads
(the cut-out and DINOv3's tokens in float32); the faster paths serve the detail flows (hr, tex) and the decoders,
whose rounding changes only their own output.

The switch PIXAL3D_COARSE_EXACT, read when asked:

    unset, "", "on", "1", "true", "yes"   on (the default)
    "off", "0", "false", "no"             off: the conditioning and the coarse stages take the faster paths as well
    anything else                         on, with a warning

The turing and pascal profiles have no original arithmetic to match (their torsos are fp16 and fp32): there the
switch changes nothing.

Whatever runs inside `scope()` on a thread is exact: SageAttention (sage.usable) and the fused kernels
(fused.active) stand aside, the matte and DINOv3 keep float32 (builders.matte_fp16, proj_cond.dino_fp16). The
engine marks the ss and lr flows (`mark`: their passes run in the scope, and they are not folded or converted);
the parts make the cut-out and DINOv3's tokens in it (`conditioning`).
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Any, ContextManager, Iterator

FLOWS = frozenset({"ss", "lr"})
_ON = {"", "1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}
_depth = threading.local()
_warned: set = set()


def on() -> bool:
    """Whether the coarse stages run exactly in this process: on the ampere profile, unless the switch is off."""
    from . import precision

    text = os.environ.get("PIXAL3D_COARSE_EXACT", "").strip().lower()
    if text not in _ON and text not in _OFF and text not in _warned:
        _warned.add(text)
        print(f"[EXACT] PIXAL3D_COARSE_EXACT={text!r} is neither on nor off: the coarse stages stay exact",
              flush=True)
    return text not in _OFF and precision.profile().name == "ampere"


def active() -> bool:
    """Whether this thread is inside `scope()`."""
    return getattr(_depth, "value", 0) > 0


@contextlib.contextmanager
def scope() -> Iterator[None]:
    _depth.value = getattr(_depth, "value", 0) + 1
    try:
        yield
    finally:
        _depth.value -= 1


def conditioning() -> ContextManager[None]:
    """`scope()` where the coarse stages run exactly, else nothing: around the pictures' conditioning."""
    return scope() if on() else contextlib.nullcontext()


def wrap(module: Any) -> None:
    """The module's forward run inside `scope()`; once per module."""
    if getattr(module, "exact_scope", False):
        return
    forward = module.forward

    def exact_forward(*args: Any, **kwargs: Any) -> Any:
        with scope():
            return forward(*args, **kwargs)

    module.forward = exact_forward
    module.exact_scope = True


def mark(model: Any, kind: str) -> bool:
    """`wrap` a flow network of `kind` (ss, lr, hr, tex) where it runs exactly; whether it did."""
    if kind not in FLOWS or not on():
        return False
    wrap(model)
    print(f"[EXACT] {kind}: the original's arithmetic (PIXAL3D_COARSE_EXACT)", flush=True)
    return True
