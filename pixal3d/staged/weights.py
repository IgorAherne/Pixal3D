"""Copies of checkpoints in the dtypes their models keep them in, written once beside the originals, so that a load
reads half as much.

Pixal3D's flow checkpoints are stored in float32 (5.4 to 5.6 GB each), while the models keep 480 of their 700
tensors, 98.7 % of the bytes, in bfloat16: a load reads four bytes per weight and casts them to two on the card.
The copy of a checkpoint holds every tensor the model takes from it, each already in the dtype the model keeps it
in (2.72 GB for the sparse-structure flow, 2.81 GB for each of the other three: 11.14 GB a set, read in place of
22.0 GB), and the network loader (loader.py) reads it in place of the original while it is valid. Spellcast3D's
installer makes the copies once the models are downloaded (install/components.py, and multiview.py after the
multi-view download):

    python -m pixal3d.staged.weights <folder of pipeline.json>   # every checkpoint of pipeline.json, and of
                                                                  # pipeline_mv.json when its models are there
    ok = make_copies(folder, say=print)                          # the same in-process: False when one failed
    outcome = make_copy(ckpt)                                    # one checkpoint (its path without extension)

The copy of <ckpt>.safetensors is <ckpt>.model-dtypes.safetensors, a safetensors file holding the tensors the model
takes, in the original's order (the loader puts them on the card in file order: the copy changes nothing of that),
cast on the CPU with `tensor.to(dtype)` - round to nearest even, the cast the loader makes on the card. Its
header's __metadata__ (`record`) says what it was made from: the original's name, size, modification time (ns) and
the SHA-256 of its header, the torch version that cast it, and the copy's format. A copy is read only while that
record matches the original as it is now and this torch, and while its tensors are exactly the ones the model
takes, in the model's dtypes and shapes (`usable_copy`); otherwise it is ignored, and the next `make_copy` makes
it again. A copy is written as <copy>.tmp, flushed to the disk and renamed: a copy that exists is whole.

A checkpoint gets no copy when its model keeps every tensor as the file has it (the sparse-structure decoder), or
when the copy would not be smaller (a model that widens fp16 tensors to fp32, as the two sparse decoders do). A
file whose tensors all take two bytes or fewer cannot shrink and is not even looked at with its model (building
the sparse decoders' classes takes a card).

The flows the int8 policy converts (modules/int8.py: hr and tex under the default "detail") also get a pre-quantised
copy, <ckpt>.int8.model-dtypes.safetensors (`make_int8_copy`): the model's state with its int8 layers in place -
each converted linear's int8 weight, fp32 scales and bias as int8.convert makes them (Int8Linear and Int8MLP built
on the CPU, by the torch reference quantiser, which the card's kernel equals bit for bit), every other tensor in the
torso's dtype. A load reads it in place of the model-dtypes copy and quantises nothing: 1.63 GB in place of 2.81 GB
on an RTX 30xx, and on a GTX 10xx, whose fp32 torso has no smaller copy, ~2 GB in place of the 5.55 GB original (and
the whole fp32 flow no longer lands on the card before its conversion). An fp16 torso (RTX 20xx) gets none: there
SageAttention's fold changes small weights before convert quantises them (below). Its record also names the flow's
kind, the layer rules it was made for (int8.copy_rules: what a card of the PC's precision profile runs, told without
the card), the torso's dtype, the int8 group and INT8_FORMAT; the loader reads it only where the process's own
rules, after the int8 self-check on the card (int8.live_rules), are the same, and otherwise reads the model-dtypes
copy (or the original) and converts as before. PIXAL3D_INT8=off never reads it. SageAttention's fold of v's scaling
is applied after the load, to the int8 layers' channel scales (powers of two: the same bits as folding the bf16 or
fp32 weights before quantising them, which have fp32's exponent range; in fp16 the fold's 2^-8 makes small weights
subnormal, which loses bits before convert quantises them, so an fp16 torso has no int8 copy).

Memory: the original is read tensor by tensor through one 64 MiB buffer, and each tensor (or piece of one) is cast
and written before the next is read. Nothing here uses the card (Spellcast3D runs the command with it hidden all
the same).

The exit code of the command: 0 when every checkpoint has a valid copy or needs none, 1 otherwise (each line says
why). Any failure only means that Pixal3D reads that checkpoint's original, as slowly as before.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch

from . import stfile

COPY_SUFFIX = ".model-dtypes.safetensors"   # <ckpt>.safetensors -> <ckpt>.model-dtypes.safetensors
FORMAT = "1"                                # bump when what a copy holds changes: every copy is then made again
INT8_SUFFIX = ".int8.model-dtypes.safetensors"  # <ckpt>.safetensors -> the flow's pre-quantised copy
INT8_FORMAT = "1"                           # bump when what an int8 copy holds changes (modules/int8.py's layers,
#                                             int8mm's weight quantiser): every int8 copy is then made again
# The pipeline files' "models" keys of the four flows and their kinds (engine.py FLOWS: <name>_flow -> <kind>)
FLOW_KINDS = {"sparse_structure_flow_model": "ss", "shape_slat_flow_model_512": "lr",
              "shape_slat_flow_model_1024": "hr", "tex_slat_flow_model_1024": "tex"}
BUFFER_BYTES = 64 << 20                     # the read buffer, as large as the loader's
DISK_SPARE_BYTES = 1 << 30                  # free space a copy must leave on its drive


class NoRoom(OSError):
    """The drive has too little free space for a copy."""


@dataclass
class Outcome:
    """What make_copy did with one checkpoint."""
    checkpoint: str         # the checkpoint's file name without extension
    state: str              # "written", "up to date" or "not needed"
    source_bytes: int       # the original's size
    copy_bytes: int         # the copy's size (0: there is none)
    seconds: float
    why: str = ""           # "not needed": the reason

    def line(self) -> str:
        if self.state == "not needed":
            return f"{self.checkpoint}: no copy needed, {self.why} ({_gb(self.source_bytes)}, {self.seconds:.1f} s)"
        return (f"{self.checkpoint}: copy {self.state}, {_gb(self.copy_bytes)} read in place of "
                f"{_gb(self.source_bytes)} ({self.seconds:.1f} s)")


# ----------------------------------------------------------------------------------- the loader's side: reading

def copy_of(file: Path) -> Path:
    """Where the copy of a checkpoint's .safetensors file lies: beside it."""
    file = Path(file)
    return file.with_name(file.name.removesuffix(".safetensors") + COPY_SUFFIX)


def record(file: Path) -> dict[str, str]:
    """The __metadata__ a copy of `file` made now, by this torch, carries."""
    file = Path(file)
    status = os.stat(file)          # through a HuggingFace snapshot's link: the blob's
    return {"copy_of": file.name, "format": FORMAT, "source_bytes": str(status.st_size),
            "source_mtime_ns": str(status.st_mtime_ns), "source_header_sha256": stfile.header_sha256(file),
            "torch": torch.__version__}


def current_copy(file: Path) -> Path | None:
    """The copy of `file` when it was made from `file` as it is now, by this torch, in this format; None when
    there is none, when it is stale or foreign, and when it cannot be read. What it holds is not looked at here
    (usable_copy does that): this is the prefetch's quick look."""
    copy = copy_of(file)
    try:
        if copy.is_file() and stfile.read_metadata(copy) == record(file):
            return copy
    except Exception:       # noqa: BLE001 - whatever is wrong with a copy only means that the original is read
        pass
    return None


def usable_copy(file: Path, keys: list[str], want: dict[str, tuple[torch.dtype, tuple[int, ...]]]
                ) -> tuple[Path, dict[str, stfile.TensorInfo]] | None:
    """The copy of `file` and its header when a load may read it in place of `file`: current (current_copy),
    and holding exactly `keys` - the tensors of `file` the model takes, in file order - each in the dtype and
    shape `want` gives. None otherwise; this never raises."""
    copy = current_copy(file)
    if copy is None:
        return None
    try:
        header = stfile.read_header(copy)
        if list(header) == list(keys) and all(header[key].torch_dtype == want[key][0]
                                              and header[key].shape == want[key][1] for key in keys):
            return copy, header
    except Exception:       # noqa: BLE001 - as in current_copy
        pass
    return None


# ------------------------------------------------------------------------------------------- making the copies

def make_copy(ckpt: Path, buffer: torch.Tensor | None = None) -> Outcome:
    """Make the copy of one checkpoint (`ckpt`: its path without extension, as the loader registers it) unless it
    is up to date or not needed; a copy that is no longer needed is removed. `buffer`: a 1-D uint8 CPU tensor to
    read through (64 MiB of its own when None). Raises when the checkpoint cannot be read or does not fit its
    model (OSError, KeyError, ValueError), when the drive lacks room for the copy (NoRoom), and when the copy
    cannot be written; an unfinished copy is not left behind."""
    began = time.perf_counter()
    ckpt = Path(ckpt)
    file = Path(f"{ckpt}.safetensors")
    copy = copy_of(file)
    source_bytes = os.stat(file).st_size

    def outcome(state: str, why: str = "") -> Outcome:
        copy_bytes = os.stat(copy).st_size if state != "not needed" else 0
        return Outcome(ckpt.name, state, source_bytes, copy_bytes, round(time.perf_counter() - began, 1), why)

    if all(info.itemsize <= 2 for info in stfile.read_header(file).values()):
        _remove(copy)
        return outcome("not needed", "every tensor takes two bytes or fewer")
    from . import loader            # here, not above: the loader imports this module

    plan = loader._plan(Path(f"{ckpt}.json"), file)
    dtypes = {key: plan.want[key][0] for key in plan.keys}
    reads = sum(plan.header[key].nbytes for key in plan.keys)
    holds = sum(plan.header[key].nbytes // plan.header[key].itemsize * dtypes[key].itemsize for key in plan.keys)
    if all(plan.header[key].torch_dtype == dtypes[key] for key in plan.keys):
        _remove(copy)
        return outcome("not needed", "its model keeps every tensor as the file has it")
    if holds >= reads:
        _remove(copy)
        return outcome("not needed", "in its model's dtypes it would not be smaller")
    if usable_copy(file, plan.keys, plan.want) is not None:
        return outcome("up to date")

    _remove(copy)                   # stale or foreign: it is never read again, and its room is needed
    free = shutil.disk_usage(copy.parent).free
    if free < holds + DISK_SPARE_BYTES:
        raise NoRoom(f"the drive has {_gb(free)} free; the copy takes {_gb(holds)}, and {_gb(DISK_SPARE_BYTES)} "
                     f"must stay free")
    stamp = record(file)
    if buffer is None:
        buffer = torch.empty(BUFFER_BYTES, dtype=torch.uint8)
    layout = [(key, dtypes[key], plan.want[key][1]) for key in plan.keys]
    stfile.write_streamed(copy, layout, _cast(file, buffer, plan.keys, dtypes), metadata=stamp, fsync=True)
    if usable_copy(file, plan.keys, plan.want) is None:
        _remove(copy)
        raise RuntimeError(f"{file.name} changed while its copy was made (the copy is not used); make it again")
    return outcome("written")


def pipeline_checkpoints(folder: Path) -> list[Path]:
    """The checkpoints of a pipeline folder (paths without extension), each once: every "models" entry of
    pipeline.json, and of pipeline_mv.json when that file and all of its checkpoints are there (the multi-view
    models are an optional download). Those entries are the networks the staged engine registers (engine.py
    FLOWS and DECODERS: each file's four flows, and the decoders both files share)."""
    folder = Path(folder)
    sets = [_models(folder / "pipeline.json")]
    multi = folder / "pipeline_mv.json"
    if multi.is_file():
        checkpoints = _models(multi)
        if all(Path(f"{ckpt}.json").is_file() and Path(f"{ckpt}.safetensors").is_file() for ckpt in checkpoints):
            sets.append(checkpoints)
    found: dict[str, Path] = {}
    for checkpoints in sets:
        for ckpt in checkpoints:
            found.setdefault(os.path.normcase(os.path.abspath(ckpt)), ckpt)
    return list(found.values())


def make_copies(folder: Path, say: Callable[[str], None] | None = None) -> bool:
    """make_copy for every checkpoint of a pipeline folder (pipeline_checkpoints), and make_int8_copy for each flow
    the int8 policy converts, one line each through `say` (printed when None). False when one of them failed: the
    others are still done. Raises when the pipeline files cannot be read."""
    say = say or _print
    buffer = torch.empty(BUFFER_BYTES, dtype=torch.uint8)
    kinds = flow_kinds(folder)
    ok = True
    for ckpt in pipeline_checkpoints(folder):
        try:
            say(make_copy(ckpt, buffer).line())
        except Exception as error:  # noqa: BLE001 - one checkpoint's failure is told, and the others go on
            ok = False
            say(f"{ckpt.name}: no copy made ({type(error).__name__}: {error}); its original is read, as before")
        kind = kinds.get(os.path.normcase(os.path.abspath(ckpt)))
        if kind is None:
            continue
        try:
            outcome = make_int8_copy(ckpt, kind, buffer)
            if outcome.state != "not needed":       # the flows the policy leaves alone: nothing to tell
                say(outcome.line())
        except Exception as error:  # noqa: BLE001 - as above
            ok = False
            say(f"{ckpt.name}: no int8 copy made ({type(error).__name__}: {error}); its layers are quantised as it "
                f"is loaded, as before")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pixal3d.staged.weights",
                                     description="Copies of a Pixal3D pipeline's checkpoints in the dtypes their "
                                                 "models keep them in, which the staged engine reads in place of "
                                                 "the originals (half as many bytes for the flow models).")
    parser.add_argument("folder", type=Path, help="the folder of pipeline.json (and pipeline_mv.json)")
    args = parser.parse_args(argv)
    try:
        ok = make_copies(args.folder)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _print(f"{args.folder}: its pipeline files cannot be read ({type(error).__name__}: {error})")
        return 1
    return 0 if ok else 1


def _models(file: Path) -> list[Path]:
    """The checkpoints a pipeline file names under "models"."""
    with open(file, "r", encoding="utf-8") as handle:
        models = json.load(handle)["args"]["models"]
    return [file.parent / relative for relative in models.values()]


def _cast(file: Path, buffer: torch.Tensor, keys: list[str], dtypes: dict[str, torch.dtype]
          ) -> Iterator[tuple[str, torch.Tensor]]:
    """The tensors of `file` named by `keys`, in file order, each cast to its dtype of `dtypes`: whole when it fits
    the buffer, else piece by piece (a cast goes element by element, so the pieces cast are the tensor cast).
    What is yielded is valid until the next item is asked for."""
    reader = stfile.iter_tensors(file, buffer, keys)
    try:
        for key, info, data in reader:
            if isinstance(data, torch.Tensor):
                yield key, data.to(dtypes[key])
                continue
            for piece in data:
                elements = info._replace(shape=(piece.numel() // info.itemsize,), start=0, end=piece.numel())
                yield key, stfile.as_tensor(piece, elements).to(dtypes[key])
    finally:
        reader.close()


# ------------------------------------------------------------------------------------- the int8 copies of the flows

def int8_copy_of(file: Path) -> Path:
    """Where the pre-quantised copy of a flow checkpoint's .safetensors file lies: beside it."""
    file = Path(file)
    return file.with_name(file.name.removesuffix(".safetensors") + INT8_SUFFIX)


def int8_record(file: Path, kind: str, rules: dict[str, str]) -> dict[str, str]:
    """The __metadata__ an int8 copy of `file` made now for a flow of `kind` with `rules`, by this torch for this
    card's precision profile, carries."""
    from ..modules import int8, precision

    source = record(file)
    return {"copy_of": source["copy_of"], "int8_format": INT8_FORMAT, "source_bytes": source["source_bytes"],
            "source_mtime_ns": source["source_mtime_ns"], "source_header_sha256": source["source_header_sha256"],
            "torch": source["torch"], "torso": precision.profile().torso_name, "kind": kind,
            "rules": json.dumps(rules, sort_keys=True), "group": str(int8.GROUP)}


def current_int8_copy(file: Path) -> tuple[Path, dict[str, str]] | None:
    """(the int8 copy of `file`, its __metadata__) when it was made from `file` as it is now, by this torch, in
    this format, for this card's precision profile; None otherwise and when it cannot be read. The flow kind and
    the rules it holds are not judged here (the loader holds them against the process's own), nor its tensors."""
    copy = int8_copy_of(file)
    try:
        if not copy.is_file():
            return None
        found = stfile.read_metadata(copy)
        want = int8_record(file, found.get("kind", ""), {})
        if all(found.get(key) == value for key, value in want.items() if key not in ("kind", "rules")):
            return copy, found
    except Exception:       # noqa: BLE001 - whatever is wrong with a copy only means that it is not read
        pass
    return None


def recorded_rules(metadata: dict[str, str]) -> dict[str, str] | None:
    """The rules an int8 copy's __metadata__ says it holds, or None when they cannot be read."""
    try:
        rules = json.loads(metadata.get("rules", ""))
    except ValueError:
        return None
    return rules if isinstance(rules, dict) else None


def usable_int8_copy(file: Path, kind: str, rules: dict[str, str],
                     want: dict[str, tuple[torch.dtype, tuple[int, ...]]]
                     ) -> tuple[Path, dict[str, stfile.TensorInfo]] | None:
    """The int8 copy of `file` and its header when a load may read it: current (current_int8_copy), made for a flow
    of `kind` with `rules`, and holding exactly the tensors of `want` - the state of the model with its int8 layers
    in place (int8.skeleton) - each in that dtype and shape. None otherwise; this never raises."""
    found = current_int8_copy(file)
    if found is None:
        return None
    copy, metadata = found
    try:
        if metadata.get("kind") != kind or recorded_rules(metadata) != rules:
            return None
        header = stfile.read_header(copy)
        if header.keys() == want.keys() and all(header[key].torch_dtype == want[key][0]
                                                and header[key].shape == want[key][1] for key in header):
            return copy, header
    except Exception:       # noqa: BLE001 - as in current_int8_copy
        pass
    return None


def flow_kinds(folder: Path) -> dict[str, str]:
    """The flow checkpoints of a pipeline folder (os.path.normcase of the absolute path without extension) and
    their kinds (ss, lr, hr, tex), by their "models" keys in pipeline.json and pipeline_mv.json."""
    found: dict[str, str] = {}
    for name in ("pipeline.json", "pipeline_mv.json"):
        path = Path(folder) / name
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            models = json.load(handle)["args"]["models"]
        for key, relative in models.items():
            if key in FLOW_KINDS:
                found[os.path.normcase(os.path.abspath(path.parent / relative))] = FLOW_KINDS[key]
    return found


def make_int8_copy(ckpt: Path, kind: str, buffer: torch.Tensor | None = None) -> Outcome:
    """Make the pre-quantised copy of one flow checkpoint (`ckpt`: its path without extension; `kind`: ss, lr, hr
    or tex) for the rules a process on this PC's card runs (int8.copy_rules), unless it is up to date or not needed
    (the policy converts nothing of that kind); a copy no longer needed is removed. Its converted layers hold what
    int8.convert makes of them (Int8Linear / Int8MLP built on the CPU from the weights in the torso's dtype, by the
    torch reference quantiser: the card's kernel's bits), every other tensor the torso's dtype, as the model-dtypes
    copy has it. Read from that copy when it is current (half the bytes), else from the original. Raises as
    make_copy does."""
    from ..modules import int8
    from . import loader            # here, not above: the loader imports this module

    began = time.perf_counter()
    ckpt = Path(ckpt)
    file = Path(f"{ckpt}.safetensors")
    copy = int8_copy_of(file)
    source_bytes = os.stat(file).st_size
    label = f"{ckpt.name} (int8, {kind})"

    def outcome(state: str, why: str = "") -> Outcome:
        copy_bytes = os.stat(copy).st_size if state != "not needed" else 0
        return Outcome(label, state, source_bytes, copy_bytes, round(time.perf_counter() - began, 1), why)

    rules = int8.copy_rules(kind)
    if rules is None:
        _remove(copy)
        if not int8._copy_exact():
            return outcome("not needed", "its fp16 torso is quantised after SageAttention's fold, at load")
        return outcome("not needed", f"the int8 policy converts no layer of the {kind} flow on this card")
    plan = loader._plan(Path(f"{ckpt}.json"), file)
    if plan.computed:
        _remove(copy)
        return outcome("not needed", "its model computes tensors no file carries")
    made = int8.skeleton(plan.model, rules)
    want = {key: (value.dtype, tuple(value.shape)) for key, value in plan.model.state_dict().items()}
    if loader._meta_outside(plan.model, want):
        _remove(copy)               # the loader could not fill it from the copy (loader._int8_plan)
        return outcome("not needed", "its model makes a tensor on the meta device that no file carries")
    if usable_int8_copy(file, kind, rules, want) is not None:
        return outcome("up to date")

    _remove(copy)
    layout, owner = _int8_layout(plan, made, want)
    holds = sum(_elements(shape) * dtype.itemsize for _, dtype, shape in layout)
    free = shutil.disk_usage(copy.parent).free
    if free < holds + DISK_SPARE_BYTES:
        raise NoRoom(f"the drive has {_gb(free)} free; the copy takes {_gb(holds)}, and {_gb(DISK_SPARE_BYTES)} "
                     f"must stay free")
    stamp = int8_record(file, kind, rules)
    if buffer is None:
        buffer = torch.empty(BUFFER_BYTES, dtype=torch.uint8)
    source = usable_copy(file, plan.keys, plan.want)
    source_file = source[0] if source is not None else file
    pieces = _int8_pieces(source_file, buffer, plan, made, owner)
    stfile.write_streamed(copy, layout, pieces, metadata=stamp, fsync=True)
    if usable_int8_copy(file, kind, rules, want) is None:
        _remove(copy)
        raise RuntimeError(f"{file.name} changed while its int8 copy was made (the copy is not used); make it again")
    return outcome("written")


def _int8_layout(plan, made: list[tuple[str, list[str]]], want: dict[str, tuple[torch.dtype, tuple[int, ...]]]
                 ) -> tuple[list[tuple[str, torch.dtype, tuple[int, ...]]], dict[str, int]]:
    """The int8 copy's tensors in the order they are written - the original's file order, an int8 layer's
    tensors where the last of its sources lies - and which int8 layer (its index in `made`) each source key
    feeds. Raises when they do not make up `want` exactly."""
    owner = {key: index for index, (_, sources) in enumerate(made) for key in sources}
    left = [len(sources) for _, sources in made]
    layout: list[tuple[str, torch.dtype, tuple[int, ...]]] = []
    for key in plan.keys:
        index = owner.get(key)
        if index is None:
            layout.append((key, *want[key]))
            continue
        left[index] -= 1
        if not left[index]:
            path = made[index][0]
            layout += [(f"{path}.{name}", *want[f"{path}.{name}"])
                       for name in plan.model.get_submodule(path).state_dict()]
    names = [name for name, _, _ in layout]
    if any(left) or len(names) != len(set(names)) or set(names) != want.keys():
        raise RuntimeError("the int8 layers' tensors do not make up the model's state")
    return layout, owner


def _int8_pieces(source: Path, buffer: torch.Tensor, plan, made: list[tuple[str, list[str]]], owner: dict[str, int]
                 ) -> Iterator[tuple[str, torch.Tensor]]:
    """The int8 copy's tensors in _int8_layout's order: every tensor of `source` (the model-dtypes copy, or the
    original) in the model's dtype, cast as the loader casts it, and each int8 layer once its sources are read,
    made on the CPU by the module convert puts there. What is yielded is valid until the next item is asked for."""
    from ..modules import int8

    pending: dict[int, dict[str, torch.Tensor]] = {}
    reader = stfile.iter_tensors(source, buffer, plan.keys)
    try:
        for key, info, data in reader:
            dtype = plan.want[key][0]
            index = owner.get(key)
            if index is None:
                if isinstance(data, torch.Tensor):
                    yield key, data.to(dtype)
                    continue
                for piece in data:
                    elements = info._replace(shape=(piece.numel() // info.itemsize,), start=0, end=piece.numel())
                    yield key, stfile.as_tensor(piece, elements).to(dtype)
                continue
            if isinstance(data, torch.Tensor):
                whole = data.to(dtype, copy=True)
            else:
                flat = torch.cat([piece.clone() for piece in data])
                whole = stfile.as_tensor(flat, info).to(dtype)
            pending.setdefault(index, {})[key] = whole
            path, sources = made[index]
            if len(pending[index]) < len(sources):
                continue
            tensors = pending.pop(index)
            shell = plan.model.get_submodule(path)
            with torch.no_grad():
                if isinstance(shell, int8.Int8MLP):
                    layer = int8.Int8MLP(_linear(tensors, f"{path}.0"), _linear(tensors, f"{path}.2"), shell.how,
                                         shell.how2)
                else:
                    layer = int8.Int8Linear(_linear(tensors, path), shell.how)
            del tensors
            for name, tensor in layer.state_dict().items():
                yield f"{path}.{name}", tensor
            del layer
    finally:
        reader.close()


def _linear(tensors: dict[str, torch.Tensor], path: str) -> torch.nn.Linear:
    """An nn.Linear holding the weight and bias of `path` from `tensors` (no memory of its own)."""
    weight = tensors[f"{path}.weight"]
    bias = tensors.get(f"{path}.bias")
    layer = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)
    return layer


def _elements(shape: tuple[int, ...]) -> int:
    count = 1
    for side in shape:
        count *= int(side)
    return count


def _remove(copy: Path) -> None:
    """A copy and an unfinished one, when they are there and can go. One that cannot (open in another program)
    stays: a stale copy is never read, and the copy of a checkpoint that needs none holds what its original does;
    writing a new copy over it fails with words of its own."""
    for path in (copy, Path(f"{copy}.tmp")):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def _gb(count: int) -> str:
    return f"{count / 1e9:.2f} GB"


def _print(line: str) -> None:
    print(line, flush=True)


if __name__ == "__main__":
    sys.exit(main())
