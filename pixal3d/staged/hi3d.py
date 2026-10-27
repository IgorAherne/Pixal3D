"""Hi3D v1.0 as the engine's second geometry: its refiner (the `sparc3d` package) makes the shape from the sparse
structure's coarse occupancy, and Pixal3D's texture flow paints it. A Hi3D prop runs these parts where a Pixal3D prop
runs cond_lr, lr, upsample and hr (parts.applies):

    ss           as for every prop; the occupancy at 64^3 is kept (Pixal3D's path pools it to 32^3)
    hi3d_cond    DINOv2-L and CLIP-L on the front picture cut out; the coarse mesh voxelized for the first pass
    hi3d_pK      pass K's latents: transformer_512 on the 64 grid, transformer_1024 on the 128 and 192 grids
    hi3d_dK      pass K decoded and its mesh extracted, then voxelized for the next pass or kept as the prop's mesh
                 (the last pass: reduced to Hi3D's face target, fitted into the cube, and its dual grid for the
                 texture, at the resolution hi3d_encode is foreseen to settle on, started on a worker thread, so
                 that the next prop's decode has the card meanwhile)
    hi3d_encode  the mesh into TRELLIS.2's shape latent (the encoder paired with the shape decoder Pixal3D ships,
                 downloaded with Hi3D): the texture flow's input and tokens, and the texture decoder's guide tables
                 from the mesh's own voxels
then cond_hr (the texture's conditioning alone), tex and finish as for a Pixal3D prop, the mesh being Hi3D's.

The sampler's kernels follow Pixal3D's switches (`accelerate`, `int8_convert`): PIXAL3D_FUSED the fused elementwise
chains, PIXAL3D_SAGE the int8 self-attention, PIXAL3D_INT8 the blocks' linears as int8 GEMMs (PIXAL3D_HI3D_INT8=off
turns Hi3D's alone off); spconv's kernel choices come from the install's calibration record
(`_restore_spconv_choices`); the blocks keep the picture's K/V through a pass where the cap has the room
(`_kv_cache_fits`; the same bits).

Frames: Pixal3D's voxels lie in [-0.5, 0.5]^3, Y up with the picture's front towards +Z (its GLB step's axis swap and
the launcher's GLB_ROTATION cancel out), which is the frame Hi3D was trained in. Every Hi3D pass fits its input mesh
into its own grid; the chain of those maps (`frames`) brings the last pass's mesh back into the coarse mesh's frame.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

STRUCTURES = (64, 128, 192)                 # each pass's structure grid; the decoder extracts at 8x
BUDGETS = (80_000, 90_000, 110_000)         # the voxels a pass takes at most (sparc3d's PASS_BUDGETS)
FACE_TARGETS = (1_000_000, 1_000_000, 2_000_000)    # the faces a pass's mesh is reduced to (sparc3d's TARGET_FACE_COUNTS)
CUBE_FIT = 0.499                            # the mesh's furthest vertex from the centre once fitted into the cube
PASSES = {1024: 2, 1536: 3}                 # the request's resolution -> Hi3D's passes
STEPS = 25                                  # scheduler updates a pass (the checkpoint's scheduler_config)
SAMPLE_PARTS = ("hi3d_p1", "hi3d_p2", "hi3d_p3")
DECODE_PARTS = ("hi3d_d1", "hi3d_d2", "hi3d_d3")
PARTS = ("hi3d_cond", *[part for pair in zip(SAMPLE_PARTS, DECODE_PARTS) for part in pair], "hi3d_encode")
TRANSFORMERS = ("hi3d_t512", "hi3d_t1024", "hi3d_t1024")
LATENT_LEVELS = 4                           # the shape latent's grid is the texture resolution / 16
DECODE_MARGIN_MIB = 300     # under the cap: the caching allocator reserves up to ~100 MiB above the decode's budget
TIGHT_SHARE = 0.8           # the budget on the ladder's tight rung (an attempt that ran out of memory)

ENTER = {"hi3d_cond": ("hi3d_dino", "hi3d_clip"), "hi3d_encode": ("shape_enc",),
         **{part: (net,) for part, net in zip(SAMPLE_PARTS, TRANSFORMERS)},
         **{part: ("hi3d_vae",) for part in DECODE_PARTS}}
WANTS = {part: ("hi3d_cond",) for part in SAMPLE_PARTS}
PRODUCES = {"hi3d_cond": ("hi3d_cond",)}


@dataclass(frozen=True)
class Hi3DPaths:
    code: Path              # the Hi3D checkout (sparc3d/ inside), put on sys.path
    weights: Path           # Hi3Dv1.0: transformer_512/, transformer_1024/, vae/, scheduler/
    dino: Path              # facebook/dinov2-large
    clip: Path              # openai/clip-vit-large-patch14
    encoder: Path           # TRELLIS.2's ckpts/shape_enc_next_dc_f16c32_fp16, without its extension


def passes(req: Any) -> int:
    return PASSES.get(int(req.resolution), 3)


def applies(part: str, req: Any) -> bool:
    """Whether a Hi3D part runs for this request (the parts after its last pass are skipped)."""
    if part in SAMPLE_PARTS:
        return SAMPLE_PARTS.index(part) < passes(req)
    if part in DECODE_PARTS:
        return DECODE_PARTS.index(part) < passes(req)
    return True


def spent(part: str, req: Any) -> tuple[str, ...]:
    """What nobody reads once `part` is over for this prop."""
    if part == "hi3d_cond":
        return ("hi3d_ss64", "cutout")
    if part in SAMPLE_PARTS and SAMPLE_PARTS.index(part) == passes(req) - 1:
        return ("hi3d_cond", "hi3d_coords")
    if part in DECODE_PARTS:
        return ("hi3d_lat",)
    if part == "hi3d_encode":
        return ("hi3d_frames", "hi3d_rng", "rng_ss", "coords_ss")
    return ()


def register(loader: Any, paths: Hi3DPaths) -> None:
    """Tell the loader where Hi3D's networks lie; nothing is read."""
    if str(paths.code) not in sys.path:
        sys.path.insert(0, str(paths.code))
    for name, folder in (("hi3d_t512", "transformer_512"), ("hi3d_t1024", "transformer_1024")):
        with open(paths.weights / folder / "config.json", "r", encoding="utf-8") as handle:
            args = json.load(handle)
        loader.register_checkpoint(name, paths.weights / folder / "model", int8_convert,
                                   config={"name": "sparc3d.models.transformers:Sparc3DTransformer", "args": args},
                                   file=paths.weights / folder / "model.safetensors")
    loader.register_builder("hi3d_vae", lambda device: _vae(paths, device), paths.weights / "vae" / "model.safetensors")
    # The picture's two encoders as the loader's own checkpoints (picture_encoder builds each from its config.json
    # alone): read ahead while the part before runs and straight onto the card, CLIP's vision tower only (1.1 of
    # its file's 1.6 GB) - transformers' from_pretrained read both files cold from the disk (19 s here) and built
    # them on the CPU first.
    for name, folder, kind in (("hi3d_dino", paths.dino, "dino"), ("hi3d_clip", paths.clip, "clip")):
        loader.register_checkpoint(name, folder / "model", file=folder / "model.safetensors",
                                   config={"name": "pixal3d.staged.hi3d:picture_encoder",
                                           "args": {"kind": kind, "folder": str(folder)}})
    loader.register_checkpoint("shape_enc", paths.encoder)


def picture_encoder(kind: str, folder: str) -> Any:
    """DINOv2-L ("dino") or CLIP-L's vision tower ("clip") built from the config.json in `folder`, no weights read:
    the loader's class for the two checkpoints (their state dicts are the files' tensors, CLIP's the vision_model
    ones), the weights its own read."""
    from transformers import CLIPVisionConfig, CLIPVisionModel, Dinov2Config, Dinov2Model

    if kind == "dino":
        return Dinov2Model(Dinov2Config.from_pretrained(folder))
    return CLIPVisionModel(CLIPVisionConfig.from_pretrained(folder))


def int8_convert(model: Any) -> Any:
    """The loader's convert hook for Hi3D's two transformers: the blocks' token-sized linears as int8 tensor-core
    GEMMs (sparc3d.nn.int8_linear on Pixal3D's own int8mm kernels) where PIXAL3D_INT8 is on and the card runs them,
    as the flows' linears are converted (modules/int8.py); the model unchanged otherwise, or where PIXAL3D_HI3D_INT8
    says off (Hi3D's own veto, the flows' int8 untouched). Half the weights' bytes on the card, a guided step at the
    finest pass about a third faster; judged end to end like the other kernels."""
    from ..modules import int8

    if int8.policy() is None or os.environ.get("PIXAL3D_HI3D_INT8", "").strip().lower() in int8._OFF:
        return model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return model
    if device.type != "cuda" or not int8.available(device):
        return model
    from sparc3d.nn import int8_linear

    from ..modules import int8mm

    int8_linear.install(int8mm)
    before = int8._bytes(model)
    counts = int8_linear.convert(model, int8_linear.DEFAULT_POLICY)
    if counts:
        print(f"[INT8] Hi3D's {type(model).__name__} (policy {int8_linear.DEFAULT_POLICY}): "
              + ", ".join(f"{kind} x{count}" for kind, count in counts.items())
              + f" as int8 GEMMs ({int8.KERNEL}; PIXAL3D_INT8), {before / 2**20:,.0f} -> "
              f"{int8._bytes(model) / 2**20:,.0f} MiB of weights", flush=True)
    return model


def _vae(paths: Hi3DPaths, device: str) -> Any:
    from sparc3d.models.autoencoders.config import load_sparconv_vae

    return load_sparconv_vae(paths.weights / "vae", device=device)


def scheduler(paths: Hi3DPaths) -> Any:
    from sparc3d.pipelines import loader

    return loader.load_scheduler(paths.weights)


# =====================================================================================================================
# Frames
# =====================================================================================================================

def coarse_mesh(coords64: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """The outer surface of the sparse structure's occupancy (its voxels, [N, 3] on the 64 grid, holes inside filled
    so that a shell gives one surface and not two), as voxel faces in Pixal3D's frame: (vertices [V, 3] float64,
    faces [F, 3] int64)."""
    from scipy import ndimage

    grid = np.zeros((66, 66, 66), dtype=bool)
    c = coords64.cpu().numpy().astype(np.int64) + 1
    grid[c[:, 0], c[:, 1], c[:, 2]] = True
    grid = ndimage.binary_fill_holes(grid)
    quads = []
    for axis in range(3):
        for step in (-1, 1):
            neighbour = np.roll(grid, -step, axis=axis)
            cells = np.argwhere(grid & ~neighbour)
            if len(cells) == 0:
                continue
            u, v = (axis + 1) % 3, (axis + 2) % 3          # u x v = +axis: the faces wound outwards
            base = cells.copy()
            base[:, axis] += 1 if step > 0 else 0
            corners = np.repeat(base[:, None, :], 4, axis=1)
            corners[:, 1, u] += 1
            corners[:, 2, u] += 1
            corners[:, 2, v] += 1
            corners[:, 3, v] += 1
            quads.append(corners if step > 0 else corners[:, ::-1])
    corners = np.concatenate(quads).reshape(-1, 3)
    unique, index = np.unique(corners, axis=0, return_inverse=True)
    index = index.reshape(-1, 4)
    faces = np.concatenate([index[:, [0, 1, 2]], index[:, [0, 2, 3]]])
    vertices = (unique - 1) / 64.0 - 0.5
    return vertices.astype(np.float64), faces.astype(np.int64)


def voxelize(vertices: np.ndarray, faces: np.ndarray, structure: int, budget: int,
             device: str) -> tuple[torch.Tensor, dict]:
    """sparc3d's adaptive_vox (a grid shrunk by 5 % at a time until at most `budget` voxels are occupied), also
    giving the fit that maps the grid back: {"center", "scale", "res"}."""
    from sparc3d.ops.voxelize import MeshVoxelizer

    v = torch.tensor(vertices, dtype=torch.float32, device=device)
    f = torch.tensor(faces, dtype=torch.int32, device=device)
    scale = 1.0
    while True:
        res = int(structure * scale)
        if res < 7:
            raise RuntimeError(f"the mesh does not fit {budget} voxels on any grid down to {res}")
        voxelizer = MeshVoxelizer(v, f, resolution=res)
        coords = voxelizer.voxelize()["voxel_coords"].to(dtype=torch.int32)
        if coords.shape[0] <= budget:
            break
        scale *= 0.95
    fit = {"center": voxelizer.center.double().cpu(), "scale": float(voxelizer.scale), "res": res}
    return coords, fit


def to_previous(vertices: np.ndarray, matrix: np.ndarray, fit: dict) -> np.ndarray:
    """A pass's extracted vertices (its decode's unit frame) -> the frame of the mesh that pass voxelized: through the
    decode's grid (8 fine cells to a structure cell) and the voxelizer's fit."""
    homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
    grid = (np.linalg.inv(matrix) @ homogeneous.T).T[:, :3] / 8.0
    return (grid - fit["res"] / 2.0) / fit["scale"] + np.asarray(fit["center"], dtype=np.float64)


def back_to_coarse(vertices: np.ndarray, frames: dict) -> np.ndarray:
    """The last pass's mesh -> the coarse mesh's frame, Pixal3D's. Row-major: the native code downstream (o_voxel,
    cumesh) reads the buffer as [V, 3] rows, and the matrix products leave a column-major array."""
    for k in reversed(range(len(frames["res"]))):
        fit = {"center": frames["center"][k].numpy(), "scale": float(frames["scale"][k]), "res": int(frames["res"][k])}
        vertices = to_previous(vertices, frames["matrix"][k].numpy(), fit)
    return np.ascontiguousarray(vertices)


def into_cube(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """The mesh scaled about the cube's centre into Pixal3D's frame [-0.5, 0.5]^3 when a pass carried it past a
    face (the refiner may push the surface a voxel past the coarse one: a crate's face by 3 %): one uniform scale,
    so that nothing downstream clamps it flat. (The vertices, the scale: 1.0 when none was needed.)"""
    span = float(np.abs(vertices).max()) if len(vertices) else 0.0
    if span <= CUBE_FIT:
        return vertices, 1.0
    scale = CUBE_FIT / span
    return vertices * scale, scale


def simplified(vertices: np.ndarray, faces: np.ndarray, target: int,
               device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The mesh reduced to about `target` faces when it has more (the extraction's 7-14 M faces at 1536 are far
    beyond what carries the detail, and the texture's dual grid, the mesh's RAM and the hub's own reduction all go
    by the count): cumesh's quadric simplification on the card (the hub's finish reduces every mesh with it; a
    7.7 M-face mesh in 1.2 s, within a tenth of a fine voxel of pyfqmr's result on average), or pyfqmr as Hi3D's
    own pipeline does (9 s on the CPU, which holds Python's lock meanwhile) where cumesh or a card is missing."""
    if faces.shape[0] <= target:
        return vertices, faces
    if device is not None and torch.device(device).type == "cuda":
        try:
            import cumesh
        except ImportError:
            cumesh = None
        if cumesh is not None:
            torch.cuda.empty_cache()        # cumesh allocates outside PyTorch: the decode's cached blocks go first
            mesh = cumesh.CuMesh()
            mesh.init(torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)).to(device),
                      torch.from_numpy(np.ascontiguousarray(faces, dtype=np.int32)).to(device))
            mesh.simplify(target, verbose=False)
            fewer_vertices, fewer_faces = mesh.read()
            del mesh
            return (np.ascontiguousarray(fewer_vertices.double().cpu().numpy()),
                    np.ascontiguousarray(fewer_faces.long().cpu().numpy()))
    import pyfqmr

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(np.ascontiguousarray(vertices, dtype=np.float64), np.ascontiguousarray(faces, dtype=np.int64))
    simplifier.simplify_mesh(target_count=target, aggressiveness=5, preserve_border=True, verbose=False)
    fewer_vertices, fewer_faces, _ = simplifier.getMesh()
    return (np.ascontiguousarray(fewer_vertices, dtype=np.float64),
            np.ascontiguousarray(fewer_faces, dtype=np.int64))


def frames_with(frames: dict | None, fit: dict) -> dict:
    """The chain with one more pass's fit (its decode's matrix not known yet: zeros)."""
    row = {"center": fit["center"][None], "scale": torch.tensor([fit["scale"]], dtype=torch.float64),
           "res": torch.tensor([fit["res"]], dtype=torch.int64), "matrix": torch.zeros(1, 4, 4, dtype=torch.float64)}
    if frames is None:
        return row
    return {key: torch.cat([frames[key], row[key]]) for key in row}


# =====================================================================================================================
# The texture's latent and guide tables from the mesh
# =====================================================================================================================

def _keys(coords: torch.Tensor, side: int) -> torch.Tensor:
    c = coords.long()
    return (c[:, 0] * side + c[:, 1]) * side + c[:, 2]


def guide_subs(latent_xyz: torch.Tensor, fine_xyz: torch.Tensor, resolution: int) -> list[torch.Tensor]:
    """The texture decoder's four subdivision tables [n_l, 8] (bool) for the voxels `fine_xyz` of the resolution
    grid: rows in the decoder's order at each level (the parent's row order, then the child index x + 2y + 4z, as
    SparseChannel2Spatial lays children out), starting from the latent's rows `latent_xyz` at resolution / 16."""
    offsets = torch.tensor([[k % 2, k // 2 % 2, k // 4 % 2] for k in range(8)], device=fine_xyz.device)
    coords = latent_xyz.long()
    subs = []
    for level in range(LATENT_LEVELS):
        shift = LATENT_LEVELS - 1 - level
        side = resolution >> shift
        present = torch.unique(_keys(fine_xyz.long() >> shift, side))
        children = (coords[:, None, :] * 2 + offsets[None]).reshape(-1, 3)
        found = torch.isin(_keys(children, side), present)
        subs.append(found.view(-1, 8))
        coords = children[found]
    if coords.shape[0] != fine_xyz.shape[0]:
        raise RuntimeError(f"the guide tables reach {coords.shape[0]} of the mesh's {fine_xyz.shape[0]} voxels")
    return subs


def dual_grid(vertices: torch.Tensor, faces: torch.Tensor, resolution: int) -> tuple[torch.Tensor, ...]:
    """The mesh (Pixal3D's frame) as TRELLIS.2's encoder reads it: o_voxel's flexible dual grid in [-0.5, 0.5]^3
    with the settings of Trellis2TexturingPipeline.encode_shape_slat. CPU work."""
    import o_voxel

    inside = vertices.float().clamp(-0.5 + 1e-6, 0.5 - 1e-6).contiguous()
    return o_voxel.convert.mesh_to_flexible_dual_grid(
        inside.cpu(), faces.int().contiguous().cpu(), grid_size=resolution, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)


# =====================================================================================================================
# The parts
# =====================================================================================================================

def _parts() -> Any:
    from . import parts

    return parts


def _pipeline(transformer: Any, steps: Any, device: str) -> Any:
    """sparc3d's pipeline around one transformer, for its sampling, decoding and extraction functions."""
    from sparc3d.pipelines.sparc3d_pipeline import Sparc3DPipeline

    transformer = transformer if transformer is not None else torch.nn.Identity()
    pipe = Sparc3DPipeline(transformer_512=transformer, vae=None, conditioner=None, scheduler=steps,
                           transformer_1024=transformer)
    pipe._device = torch.device(device)
    return pipe


_kernels_chosen = False


def _restore_spconv_choices(card: torch.device) -> None:
    """spconv's kernel choices from this install's calibration record (sparc3d.ops.spconv_tuning), so that no
    convolution of the first prop times its candidates and every session runs the same kernels."""
    try:
        from sparc3d.ops import spconv_tuning

        from .. import calibration

        restored = spconv_tuning.restore(calibration.section("spconv", card), card)
    except Exception as exc:        # noqa: BLE001 - a record that cannot be read: spconv tunes as it did
        print(f"[HI3D] spconv's kernel choices were not restored ({type(exc).__name__}: {str(exc)[:200]})", flush=True)
        return
    if restored:
        print(f"[HI3D] {restored} spconv kernel choices restored from the calibration record", flush=True)


def accelerate(device: str) -> None:
    """Hi3D's sampler kernels, chosen once a process at its first pass (and spconv's kernel choices restored from
    the calibration record), under Pixal3D's own switches as Pixal3D reads them: the fused elementwise chains
    (sparc3d's SPARC3D_FUSED kernels) where PIXAL3D_FUSED is its default "rounded" (modules.fused.MODE), the int8
    self-attention on sm_80+ cards where PIXAL3D_SAGE is on (modules.sage.from_env). Both check themselves on the
    card first and stand down if that fails. The sampler's bit-identical savings (SPARC3D_FAST) are sparc3d's own
    default; the int8 GEMMs are chosen at load (`int8_convert`)."""
    global _kernels_chosen
    if _kernels_chosen:
        return
    _kernels_chosen = True
    from sparc3d.nn import accel

    from ..modules import fused, sage

    accel.FUSED = fused.MODE == "rounded"
    card = torch.device(device)
    _restore_spconv_choices(card)
    if card.type != "cuda" or not sage.from_env() or torch.cuda.get_device_capability(card) < (8, 0):
        return
    from sparc3d.nn.sparse.attention import int8_attention

    if int8_attention.install(device=card):
        print("[HI3D] the transformer's self- and cross-attention run int8 (q, k) with fp16 P.V (PIXAL3D_SAGE)",
              flush=True)
    else:
        print(f"[HI3D] int8 self-attention is off on this card: {int8_attention.state().get('why')}", flush=True)


# The model calls of the parts, each one function (the engine's checks stand fakes in for them).

def condition(dino: Any, clip: Any, picture: Any) -> torch.Tensor:
    """The cut-out front picture -> Hi3D's conditioning [1, 2917, 2048] from the two encoders on the card, framed as
    sparc3d's prepare_image does."""
    from sparc3d.models.conditioners.image import ImageConditioner
    from sparc3d.utils.image import prepare_image

    return ImageConditioner.from_encoders(dino, clip)(prepare_image(picture))


KV_CACHE_MIB = 24 * 2917 * 2 * 2112 * 2 / 2**20    # the context's K/V of the 24 blocks in fp16: 564 MiB
SAMPLE_KIB_PER_VOXEL = 32       # a pass's activations at their peak, per voxel (17-29 KiB on the knight's passes)
KV_CACHE_SLACK_MIB = 1024       # what stays free under the cap beside the model, the activations and the K/V


def _kv_cache_fits(ctx: Any, voxels: int) -> bool:
    """Whether a pass's blocks may keep the context's K/V (accel.KV_CACHE: the same bits, one projection of the 2917
    context rows a block and pass instead of one a forward - 15 % of a first-pass step, 1-3 % of the finer passes'):
    the cap in force has the room for them beside what the card holds, the pass's activations and a slack for the
    allocator's fragmentation (no cap: the card is not the limit)."""
    device = torch.device(ctx.device)
    if device.type != "cuda":
        return False
    cap = ctx.loader.stats().get("cap_mib")
    if not cap:
        return True
    held = torch.cuda.memory_allocated(device) / 2**20
    return held + voxels * SAMPLE_KIB_PER_VOXEL / 1024 + KV_CACHE_MIB + KV_CACHE_SLACK_MIB <= cap


def sample(transformer: Any, steps: Any, cond_tokens: torch.Tensor, coords: torch.Tensor, index: int, total: int,
           generator: torch.Generator, device: str, tick: Any, kv_cache: bool = False
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass `index`'s latents on the voxels `coords` [N, 3]: (coords [N, 4], feats [N, 32] float32) on the CPU.
    `tick(step, steps)` after every scheduler update; `kv_cache`: the blocks keep the context's K/V through the pass
    (_kv_cache_fits)."""
    from sparc3d.nn import accel

    accelerate(device)
    pipe = _pipeline(transformer, steps, device)
    pipe.progress_callback = lambda event: tick(event.step, event.num_inference_steps)
    accel.KV_CACHE = bool(kv_cache)
    try:
        latents = pipe._sample_latents(transformer, cond_tokens, coords, STRUCTURES[index], STEPS, generator,
                                       pass_index=index, num_passes=total)
    finally:
        accel.KV_CACHE = False
    return latents.coords.cpu(), latents.feats.float().cpu()


def decode(vae: Any, coords: torch.Tensor, feats: torch.Tensor, index: int, device: str,
           budget_mib: float) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """Pass `index`'s latents decoded and the mesh extracted within `budget_mib` of the card (sparc3d's streamed
    decode: what does not fit goes through page-locked RAM): (vertices [V, 3] float64 in the decode's unit frame,
    faces [F, 3] int64, the decode's grid-to-unit matrix [4, 4] float64)."""
    from sparc3d.pipelines.streamed_decode import decode_and_extract

    mesh = decode_and_extract(vae, coords, feats, STRUCTURES[index], budget_mib=budget_mib, device=device)
    print(f"[HI3D] pass {index + 1}: {mesh.decoded_points:,} points decoded in {mesh.seconds['decode']:.1f} s, "
          f"{len(mesh.faces):,} faces in {mesh.seconds['extract']:.1f} s (card {mesh.peak_reserved_mib:,.0f} MiB of "
          f"{budget_mib:,.0f}, page-locked RAM {mesh.host_mib:,.0f} MiB)", flush=True)
    return mesh.vertices.double().numpy(), mesh.faces.numpy(), mesh.matrix


def encode_mesh(encoder: Any, fine: torch.Tensor, dual: torch.Tensor, intersected: torch.Tensor, resolution: int,
                device: str) -> Any:
    """The dual grid of the mesh at `resolution` -> TRELLIS.2's shape latent (a SparseTensor on the card)."""
    from ..modules.sparse import SparseTensor

    rows = torch.cat([torch.zeros_like(fine[:, :1]), fine], dim=1)
    vertices = SparseTensor(feats=(dual * resolution - fine).float(), coords=rows).to(device)
    return encoder(vertices, vertices.replace(intersected.to(device)))


def cond(ctx: Any, prop: Any, mode: str) -> None:
    """The picture's conditioning and the first pass's voxels from the coarse occupancy."""
    store, device, parts = ctx.store, ctx.device, _parts()
    tokens = condition(parts._network(ctx, "hi3d_dino"), parts._network(ctx, "hi3d_clip"),
                       store.get_small(prop.id, "cutout"))
    vertices, faces = coarse_mesh(store.get_small(prop.id, "hi3d_ss64")["coords"])
    coords, fit = voxelize(vertices, faces, STRUCTURES[0], BUDGETS[0], device)
    generator = torch.Generator("cpu").manual_seed(int(prop.req.seed))
    store.put_device(prop.id, "hi3d_cond", {"c": tokens.half()})
    store.put_small(prop.id, "hi3d_coords", {"coords": coords.cpu()})
    store.put_small(prop.id, "hi3d_frames", frames_with(None, fit))
    store.put_small(prop.id, "hi3d_rng", {"state": generator.get_state()})


def _sample(ctx: Any, prop: Any, mode: str, index: int) -> None:
    store = ctx.store
    transformer = _parts()._network(ctx, TRANSFORMERS[index])
    generator = torch.Generator("cpu")
    generator.set_state(store.get_small(prop.id, "hi3d_rng")["state"])
    coords = store.get_small(prop.id, "hi3d_coords")["coords"]
    coords, feats = sample(transformer, ctx.engine.hi3d_scheduler, ctx.take(prop, "hi3d_cond")["c"], coords, index,
                           passes(prop.req), generator, ctx.device, _parts()._steps(ctx, prop, SAMPLE_PARTS[index], mode),
                           kv_cache=_kv_cache_fits(ctx, int(coords.shape[0])))
    store.put_small(prop.id, "hi3d_lat", {"coords": coords, "feats": feats})
    store.put_small(prop.id, "hi3d_rng", {"state": generator.get_state()})


def decode_budget(ctx: Any, mode: str) -> float:
    """MiB the streamed decode may hold on the card: the cap in force (the VAE's weights inside it) less
    DECODE_MARGIN_MIB for the allocator's own reserve, TIGHT_SHARE of that on the ladder's tight rung; without a cap,
    what the card has free."""
    cap = ctx.loader.stats().get("cap_mib")
    device = torch.device(ctx.device)
    if not cap and device.type != "cuda":
        return float("inf")
    if not cap:
        free, _ = torch.cuda.mem_get_info(device)
        cap = (free >> 20) + torch.cuda.memory_reserved(device) / 2**20 - getattr(ctx.loader, "safety_mib", 300)
    budget = cap - DECODE_MARGIN_MIB
    return budget * TIGHT_SHARE if mode == "tight" else budget


def _decoded(ctx: Any, prop: Any, mode: str, index: int) -> None:
    store, device, part = ctx.store, ctx.device, DECODE_PARTS[index]
    stored = store.get_small(prop.id, "hi3d_lat")
    vertices, faces, matrix = decode(_parts()._network(ctx, "hi3d_vae"), stored["coords"], stored["feats"], index,
                                     device, decode_budget(ctx, mode))
    ctx.emit(_parts().Event("step", part, prop.id, 1, 2))
    ctx.check()
    frames = {key: value.clone() for key, value in store.get_small(prop.id, "hi3d_frames").items()}
    frames["matrix"][index] = matrix
    if index + 1 < passes(prop.req):
        coords, fit = voxelize(vertices, faces, STRUCTURES[index + 1], BUDGETS[index + 1], device)
        store.put_small(prop.id, "hi3d_coords", {"coords": coords.cpu()})
        store.put_small(prop.id, "hi3d_frames", frames_with(frames, fit))
    else:
        began = time.perf_counter()
        coarse, faces = simplified(back_to_coarse(vertices, frames), faces, FACE_TARGETS[index], device)
        coarse, scale = into_cube(coarse)
        print(f"[HI3D] the mesh: {len(faces):,} faces of {len(vertices):,} points"
              + (f", scaled by {scale:.3f} into the cube" if scale < 1.0 else "")
              + f" ({time.perf_counter() - began:.1f} s)", flush=True)
        # a large item: the mesh (tens of MB) may go to the work folder while the prop waits for its texture
        store.put_device(prop.id, "hi3d_mesh", {"vertices": torch.from_numpy(coarse.astype(np.float32)),
                                                "faces": torch.from_numpy(faces.astype(np.int32))})
        store.put_small(prop.id, "hi3d_frames", frames)
        _grid_ahead(store, prop, _resolution_ahead(ctx, prop, coarse, faces))
    ctx.emit(_parts().Event("step", part, prop.id, 2, 2))


ENCODER_MIB = 720       # the shape encoder's weights on the card (encode's room is measured beside them) and
DECODER_MIB = 380       # Hi3D's mesh decoder's (on the card while the last decode foresees encode's room)


def surface_count(vertices: np.ndarray, faces: np.ndarray, resolution: int, device: str) -> tuple[int, int]:
    """The cells of the `resolution` grid over Pixal3D's cube that the mesh's surface crosses, and the texture's
    tokens over them (the occupied cells of the resolution / 16 grid), by sparc3d's GPU voxelizer: within a few per
    cent of the dual grid's fine voxels and tokens (o_voxel, CPU seconds) in a fraction of a second. Two vertices at
    the cube's corners (no face) hold the voxelizer's frame to the cube."""
    from sparc3d.ops.voxelize import MeshVoxelizer

    corners = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32)
    v = torch.as_tensor(np.concatenate([np.asarray(vertices, dtype=np.float32), corners]), device=device)
    f = torch.as_tensor(np.asarray(faces), dtype=torch.int32, device=device)
    coords = MeshVoxelizer(v, f, resolution=resolution, pad=0).voxelize()["voxel_coords"].int()
    tokens = int(torch.unique(_keys(coords >> 4, resolution >> 4)).shape[0])
    return int(coords.shape[0]), tokens


def _resolution_ahead(ctx: Any, prop: Any, vertices: np.ndarray, faces: np.ndarray) -> int:
    """The texture resolution `encode` will most likely settle on, so that the dual grid made ahead (_grid_ahead)
    is the one it takes: the request's, lowered as encode lowers it (_refused: the token cap, the card's room for
    the encoder) from the surface's count at that resolution (surface_count; the lower resolutions' counts scaled as
    the resolution squared, as _foreseen scales them). encode decides for real, and makes another grid when this
    guess was wrong - the CPU seconds it spent anyway before. The request's own resolution when no count can be
    made."""
    request = prop.req
    resolution = int(request.resolution)
    try:
        voxels, tokens = surface_count(vertices, faces, resolution, ctx.device)
    except Exception:  # noqa: BLE001 - a guess only: the card's memory, the voxelizer's limits
        return resolution
    # encode's room (encoder_room_mib: the cap less what the card holds then) as it will be, the encoder's weights
    # standing where the mesh decoder's are now, the rest of the card's content the same
    room = encoder_room_mib(ctx)
    if room is not None:
        size = getattr(ctx.loader, "size_mib", None)
        decoder = (size("hi3d_vae") if size is not None else None) or DECODER_MIB
        room -= ENCODER_MIB - int(decoder)
    while resolution > 1024:
        refused = _refused(ctx, request, tokens, voxels, resolution, room)
        if refused is None:
            break
        lower = resolution - 128 if refused == "tokens" else _foreseen(ctx, request, tokens, voxels, resolution, room)
        scale = (lower / resolution) ** 2
        voxels, tokens, resolution = int(voxels * scale), int(tokens * scale), lower
    return resolution


_grids: concurrent.futures.ThreadPoolExecutor | None = None


def _grid_ahead(store: Any, prop: Any, resolution: int) -> None:
    """The texture's dual grid of the prop's mesh at the request's resolution, started on a worker thread (o_voxel's
    CPU work, 8-35 s at 1536, which lets go of Python's lock) so that the next prop's decode has the card meanwhile;
    `encode` takes it (`_grid`). Kept on the store, which lives as long as the batch."""
    global _grids
    pending = getattr(store, "hi3d_grids", None)
    if pending is None:
        try:
            store.hi3d_grids = pending = {}
        except AttributeError:          # a store that takes no attribute: encode makes the grid itself
            return
    if _grids is None:
        _grids = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="hi3d-grid")
    mesh = store.get_large(prop.id, "hi3d_mesh")
    pending[prop.id] = (resolution, _grids.submit(dual_grid, mesh["vertices"], mesh["faces"], resolution))


def _grid(store: Any, prop: Any, mesh: dict, resolution: int) -> tuple[torch.Tensor, ...]:
    """The mesh's dual grid at `resolution`: the one made ahead when it is that resolution's (an error it met is
    raised here, the prop's), else made now."""
    pending = getattr(store, "hi3d_grids", None) or {}
    ahead = pending.pop(prop.id, None)
    if ahead is not None and ahead[0] == resolution:
        return ahead[1].result()
    return dual_grid(mesh["vertices"], mesh["faces"], resolution)


ENCODE_BYTES_PER_VOXEL = 1024       # the shape encoder's peak on the card per fine voxel beside its weights (0.91
#                                     KB measured on the knight at 1536 and 1024: its float32 norms, neighbour tables)


def encode_mib(voxels: int) -> float:
    return voxels * ENCODE_BYTES_PER_VOXEL / 2**20


def encoder_room_mib(ctx: Any) -> float | None:
    """MiB the shape encoder may take beside what the card holds (its weights among it) under the cap in force;
    None without one (a CPU run, or the cap was lifted: the card and shared memory)."""
    cap = ctx.loader.stats().get("cap_mib")
    device = torch.device(ctx.device)
    if not cap or device.type != "cuda":
        return None
    return cap - torch.cuda.memory_allocated(device) / 2**20


def _refused(ctx: Any, request: Any, tokens: int, voxels: int, resolution: int, room: float | None) -> str | None:
    """Why a texture of `tokens` tokens over `voxels` voxels at `resolution` would not do: "tokens" (the request's
    cap), "card" or "ram" (parts.resolution_fits: the texture sampler and the finish as Pixal3D foresees them; the
    encoder's own peak against `room`), or None."""
    if tokens >= request.max_num_tokens:
        return "tokens"
    reason = _parts().resolution_fits(ctx, 0, tokens, resolution, voxels=voxels)
    if reason is None and room is not None and encode_mib(voxels) > room / 1.1:
        reason = "card"
    return reason


def _foreseen(ctx: Any, request: Any, tokens: int, voxels: int, resolution: int, room: float | None) -> int:
    """The highest resolution under `resolution` (128 at a time, 1024 at least) at which the mesh is foreseen to
    do: its voxels and tokens go about as the resolution squared. Spares a dual grid (CPU seconds) per step."""
    lower = resolution - 128
    while lower > 1024:
        scale = (lower / resolution) ** 2
        if _refused(ctx, request, int(tokens * scale), int(voxels * scale), lower, room) is None:
            break
        lower -= 128
    return lower


def _tokens(fine: torch.Tensor, resolution: int) -> int:
    """The texture's tokens over the fine voxels: the occupied cells of the resolution / 16 grid."""
    return int(torch.unique(_keys(fine >> 4, resolution >> 4)).shape[0])


def _counted(store: Any, prop: Any, mesh: dict, resolution: int, device: str) -> tuple[int, int]:
    """(fine voxels, tokens) of the mesh at `resolution` for encode's decision: the dual grid's own when the one
    made ahead is this resolution's, else the surface's count on the card (within 0.1 % of it, a second instead
    of the grid's 5-15 s on the CPU: the grid is made for the resolution encode settles on, not for every one it
    weighs); the dual grid itself where the card cannot count (a CPU run)."""
    pending = getattr(store, "hi3d_grids", None)
    ahead = (pending or {}).get(prop.id)
    if ahead is not None and ahead[0] == resolution:
        fine = ahead[1].result()[0].int()           # looked at, not taken: _grid takes it when encode settles here
        return fine.shape[0], _tokens(fine, resolution)
    try:
        return surface_count(mesh["vertices"].cpu().numpy(), mesh["faces"].cpu().numpy(), resolution, device)
    except Exception:  # noqa: BLE001 - no card, or the voxelizer's limits: the grid counts
        pass
    grid = dual_grid(mesh["vertices"], mesh["faces"], resolution)
    if pending is not None:                         # kept for _grid, should encode settle on this resolution
        done: concurrent.futures.Future = concurrent.futures.Future()
        done.set_result(grid)
        pending[prop.id] = (resolution, done)
    fine = grid[0].int()
    return fine.shape[0], _tokens(fine, resolution)


def encode(ctx: Any, prop: Any, mode: str) -> None:
    """The mesh into the texture's shape latent at the request's resolution, lowered by 128 at a time (down to
    1024) while its tokens reach the request's cap or the memory this PC leaves would not take it (_refused), as
    Pixal3D's upsample does, and noted as "memory_cap" the same way (the artist is told). An encoder that runs out
    of memory all the same lowers it once more."""
    parts, store, device, request = _parts(), ctx.store, ctx.device, prop.req
    encoder = parts._network(ctx, "shape_enc")
    mesh = store.get_large(prop.id, "hi3d_mesh")
    room = encoder_room_mib(ctx)
    resolution, unbounded, why = int(request.resolution), None, None
    while True:
        voxels, tokens = _counted(store, prop, mesh, resolution, device)
        ctx.check()
        refused = _refused(ctx, request, tokens, voxels, resolution, room)
        if refused == "tokens" and resolution > 1024:
            resolution -= 128
            continue
        unbounded = resolution if unbounded is None else unbounded      # the token cap alone gives this
        if refused and refused != "tokens" and resolution > 1024:
            why = why or refused
            resolution = _foreseen(ctx, request, tokens, voxels, resolution, room)
            continue
        fine, dual, intersected = _grid(store, prop, mesh, resolution)
        fine = fine.int()
        voxels, tokens = fine.shape[0], _tokens(fine, resolution)
        ctx.check()
        try:
            slat = encode_mesh(encoder, fine, dual, intersected, resolution, device)
            subs = guide_subs(slat.coords[:, 1:], fine.to(device), resolution)
            break
        except Exception as exc:
            from .runner import classify

            if resolution <= 1024 or classify(exc) != "oom":
                raise
            why = why or "card"
        ctx.loader.trim()               # the attempt's tensors went with its exception
        resolution -= 128
    store.put_small(prop.id, "coords_hr", {"coords": slat.coords.int().cpu(),
                                          "res": torch.tensor([resolution], dtype=torch.int32),
                                          "voxels": torch.tensor([voxels], dtype=torch.int64)})
    store.put_small(prop.id, "shape_slat", {"feats": slat.feats.float().cpu()})
    store.put_small(prop.id, "hi3d_subs", {f"s{level}": sub.cpu() for level, sub in enumerate(subs)})
    store.put_small(prop.id, "rng_hr", store.get_small(prop.id, "rng_ss"))
    if resolution != unbounded:
        notes = store.get_small(prop.id, "notes")
        store.put_small(prop.id, "notes", {**notes, "memory_cap": {
            "by": why, "resolution": resolution, "without": unbounded, "tokens": tokens}})


TABLE = {"hi3d_cond": cond, "hi3d_encode": encode,
         **{part: (lambda index: lambda ctx, prop, mode: _sample(ctx, prop, mode, index))(i)
            for i, part in enumerate(SAMPLE_PARTS)},
         **{part: (lambda index: lambda ctx, prop, mode: _decoded(ctx, prop, mode, index))(i)
            for i, part in enumerate(DECODE_PARTS)}}
