"""Hi3D v1.0 as the engine's second geometry: its refiner (the `sparc3d` package) makes the shape from the sparse
structure's coarse occupancy, and Pixal3D's texture flow paints it. A Hi3D prop runs these parts where a Pixal3D prop
runs cond_lr, lr, upsample and hr (parts.applies):

    ss           as for every prop; the occupancy at 64^3 is kept (Pixal3D's path pools it to 32^3)
    hi3d_cond    DINOv2-L and CLIP-L on the front picture cut out; the coarse mesh voxelized for the first pass
    hi3d_pK      pass K's latents: transformer_512 on the 64 grid, transformer_1024 on the 128 and 192 grids
    hi3d_dK      pass K decoded and its mesh extracted, then voxelized for the next pass or kept as the prop's mesh
    hi3d_encode  the mesh into TRELLIS.2's shape latent (the encoder of the shape decoder Pixal3D ships, the same
                 file): the texture flow's input and tokens, and the texture decoder's guide tables from the mesh's
                 own voxels
then cond_hr (the texture's conditioning alone), tex and finish as for a Pixal3D prop, the mesh being Hi3D's.

Frames: Pixal3D's voxels lie in [-0.5, 0.5]^3, Y up with the picture's front towards +Z (its GLB step's axis swap and
the launcher's GLB_ROTATION cancel out), which is the frame Hi3D was trained in. Every Hi3D pass fits its input mesh
into its own grid; the chain of those maps (`frames`) brings the last pass's mesh back into the coarse mesh's frame.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

STRUCTURES = (64, 128, 192)                 # each pass's structure grid; the decoder extracts at 8x
BUDGETS = (80_000, 90_000, 110_000)         # the voxels a pass takes at most (sparc3d's PASS_BUDGETS)
PASSES = {1024: 2, 1536: 3}                 # the request's resolution -> Hi3D's passes
STEPS = 25                                  # scheduler updates a pass (the checkpoint's scheduler_config)
SAMPLE_PARTS = ("hi3d_p1", "hi3d_p2", "hi3d_p3")
DECODE_PARTS = ("hi3d_d1", "hi3d_d2", "hi3d_d3")
PARTS = ("hi3d_cond", *[part for pair in zip(SAMPLE_PARTS, DECODE_PARTS) for part in pair], "hi3d_encode")
TRANSFORMERS = ("hi3d_t512", "hi3d_t1024", "hi3d_t1024")
LATENT_LEVELS = 4                           # the shape latent's grid is the texture resolution / 16
DECODE_MARGIN_MIB = 300     # under the cap: the caching allocator reserves up to ~100 MiB above the decode's budget
TIGHT_SHARE = 0.8           # the budget on the ladder's tight rung (an attempt that ran out of memory)

ENTER = {"hi3d_cond": ("hi3d_cond",), "hi3d_encode": ("shape_enc",),
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
        return ("hi3d_ss64", "cutout", "dino512")
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
        loader.register_checkpoint(name, paths.weights / folder / "model",
                                   config={"name": "sparc3d.models.transformers:Sparc3DTransformer", "args": args},
                                   file=paths.weights / folder / "model.safetensors")
    loader.register_builder("hi3d_vae", lambda device: _vae(paths, device), paths.weights / "vae" / "model.safetensors")
    loader.register_builder("hi3d_cond", lambda device: _conditioner(paths, device),
                            paths.dino / "model.safetensors")
    loader.register_checkpoint("shape_enc", paths.encoder)


def _vae(paths: Hi3DPaths, device: str) -> Any:
    from sparc3d.models.autoencoders.config import load_sparconv_vae

    return load_sparconv_vae(paths.weights / "vae", device=device)


def _conditioner(paths: Hi3DPaths, device: str) -> Any:
    from sparc3d.models.conditioners.image import ImageConditioner

    return ImageConditioner(local_files_only=True, dino_model_id=str(paths.dino),
                            clip_model_id=str(paths.clip)).to(device).eval()


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
            u, v = [a for a in range(3) if a != axis]
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


# The model calls of the parts, each one function (the engine's checks stand fakes in for them).

def condition(conditioner: Any, picture: Any) -> torch.Tensor:
    """The cut-out front picture -> Hi3D's conditioning [1, 2917, 2048], framed as sparc3d's prepare_image does."""
    from sparc3d.utils.image import prepare_image

    return conditioner(prepare_image(picture))


def sample(transformer: Any, steps: Any, cond_tokens: torch.Tensor, coords: torch.Tensor, index: int, total: int,
           generator: torch.Generator, device: str, tick: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass `index`'s latents on the voxels `coords` [N, 3]: (coords [N, 4], feats [N, 32] float32) on the CPU.
    `tick(step, steps)` after every scheduler update."""
    pipe = _pipeline(transformer, steps, device)
    pipe.progress_callback = lambda event: tick(event.step, event.num_inference_steps)
    latents = pipe._sample_latents(transformer, cond_tokens, coords, STRUCTURES[index], STEPS, generator,
                                   pass_index=index, num_passes=total)
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
    store, device = ctx.store, ctx.device
    tokens = condition(_parts()._network(ctx, "hi3d_cond"), store.get_small(prop.id, "cutout"))
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
    coords, feats = sample(transformer, ctx.engine.hi3d_scheduler, ctx.take(prop, "hi3d_cond")["c"],
                           store.get_small(prop.id, "hi3d_coords")["coords"], index, passes(prop.req), generator,
                           ctx.device, _parts()._steps(ctx, prop, SAMPLE_PARTS[index], mode))
    store.put_small(prop.id, "hi3d_lat", {"coords": coords, "feats": feats})
    store.put_small(prop.id, "hi3d_rng", {"state": generator.get_state()})


def decode_budget(ctx: Any, mode: str) -> float:
    """MiB the streamed decode may hold on the card: the cap in force (the VAE's weights inside it) less
    DECODE_MARGIN_MIB for the allocator's own reserve, TIGHT_SHARE of that on the ladder's tight rung; without a cap,
    what the card has free."""
    cap = ctx.loader.stats().get("cap_mib")
    if not cap and torch.device(ctx.device).type != "cuda":
        return float("inf")
    if not cap:
        free, _ = torch.cuda.mem_get_info()
        cap = (free >> 20) + torch.cuda.memory_reserved() / 2**20 - ctx.loader.safety_mib
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
        coarse = back_to_coarse(vertices, frames)
        store.put_small(prop.id, "hi3d_mesh", {"vertices": torch.from_numpy(coarse.astype(np.float32)),
                                               "faces": torch.from_numpy(faces.astype(np.int32))})
        store.put_small(prop.id, "hi3d_frames", frames)
    ctx.emit(_parts().Event("step", part, prop.id, 2, 2))


def encode(ctx: Any, prop: Any, mode: str) -> None:
    """The mesh into the texture's shape latent at the request's resolution (lowered by 128 at a time, down to
    1024, while its tokens reach the request's cap, as Pixal3D's upsample does)."""
    store, device, request = ctx.store, ctx.device, prop.req
    mesh = store.get_small(prop.id, "hi3d_mesh")
    resolution = int(request.resolution)
    while True:
        fine, dual, intersected = dual_grid(mesh["vertices"], mesh["faces"], resolution)
        tokens = torch.unique(fine // 16, dim=0).shape[0]
        if tokens < request.max_num_tokens or resolution <= 1024:
            break
        resolution -= 128
    ctx.check()
    fine = fine.int()
    slat = encode_mesh(_parts()._network(ctx, "shape_enc"), fine, dual, intersected, resolution, device)
    subs = guide_subs(slat.coords[:, 1:], fine.to(device), resolution)
    store.put_small(prop.id, "coords_hr", {"coords": slat.coords.int().cpu(),
                                          "res": torch.tensor([resolution], dtype=torch.int32),
                                          "voxels": torch.tensor([fine.shape[0]], dtype=torch.int64)})
    store.put_small(prop.id, "shape_slat", {"feats": slat.feats.float().cpu()})
    store.put_small(prop.id, "hi3d_subs", {f"s{level}": sub.cpu() for level, sub in enumerate(subs)})
    store.put_small(prop.id, "rng_hr", store.get_small(prop.id, "rng_ss"))


TABLE = {"hi3d_cond": cond, "hi3d_encode": encode,
         **{part: (lambda index: lambda ctx, prop, mode: _sample(ctx, prop, mode, index))(i)
            for i, part in enumerate(SAMPLE_PARTS)},
         **{part: (lambda index: lambda ctx, prop, mode: _decoded(ctx, prop, mode, index))(i)
            for i, part in enumerate(DECODE_PARTS)}}
