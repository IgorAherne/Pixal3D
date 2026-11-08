"""The eleven parts of a prop, and what each needs: this is the plan the runner follows (runner.StagedRunner).

A part function is `fn(ctx, prop, mode)`. It reads its inputs from the batch's store, computes with the networks
the loader holds for the part, and writes its results to the store as its LAST actions, after everything that can
raise: an attempt that failed has changed nothing, and the next attempt (mode "tight", then "uncapped": the
runner's ladder) starts from the same inputs. The parts and the lines of the reference pipeline each reproduces
(p2d = pipelines/pixal3d_image_to_3d.py, pmv = pipelines/pixal3d_mv_image_to_3d.py):

    cutout    hooks.prepare: the picture(s) opened, cut out and framed (background removal loaded when first asked for)
    camera    hooks.camera with MoGe: single-view props only
    features  DINOv3 once at 512 px and once at 1024 px; the dense conditioning of the first stage (p2d:193-227)
    ss        the sparse structure (p2d:301-348), through engine.sample_sparse_structure so that a caller's guard
              wrapped around that attribute applies
    cond_lr   conditioning of the 512 shape stage at the prop's tokens (p2d:229-295)
    lr        the 512 shape latent (p2d:350-388)
    upsample  the shape decoder's up-sampling and the token cap (p2d:702-725)
    cond_hr   conditionings of the 1024 shape stage and of the texture stage (p2d:730-736, 762-769)
    hr        the 1024 shape latent (p2d:737-757)
    tex       the texture latent (p2d:510-551); then the checkpoint a restarted batch resumes from
    finish    both decoders and the hole filling (p2d:574-606); the mesh to the CPU; hooks.finish writes the file

The random numbers are the reference's: drawn on the CPU with the reference's shapes, in the reference's order.
`ss` seeds the generator; `ss`, `lr` and `hr` store its state after their draws under their own key, and the next
drawing part starts from the stored state. A part never overwrites the state it started from, so a second attempt
draws the same noise, whatever other props did in between.

The conditioning and decoding functions (pipelines/proj_cond.py, pipelines/decode.py) are imported when a part
first needs them, not with this module: NAF's code asks for the graphics card when it is imported, and building an
engine must not.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import torch

from ..modules import precision
from ..modules.sparse import SparseTensor
from . import hi3d
from .runner import PartContext, PropState
from .types import PARTS, DecodedMesh, Event, FinishSteps, PropFailed

# MiB outside PyTorch's allocator while a part runs (cuBLAS / cuDNN handles, flash_attn, natten, flex_gemm work
# areas; for "finish" its decode phase, which adds cumesh's fill_holes). The loader's cap leaves this much free.
NATIVE_RESERVE = {**{part: 200 for part in PARTS if part != "finish"}, "finish": 300}   # starting figures
# Both phases of "finish" leave more of the card alone than the other parts do: the page's viewer may load a
# finished model while the next prop is decoded.
FINISH_SAFETY_MIB = 500

# Parts whose networks run cuDNN convolutions. cuDNN quietly takes another algorithm when the work area of its
# first choice cannot be allocated, and another algorithm rounds differently: under the cap the same seed could
# give another cut-out, camera or conditioning without any error. PyTorch counts the failed allocation, so a
# counter that moved during such a part is treated as an out-of-memory failure of that attempt. Nothing inside
# such a part may swallow an out-of-memory of its own (see `_steps`), or the count would mean two things.
CONV_PARTS = frozenset({"cutout", "camera", "features", "cond_lr", "cond_hr", "ss", "hi3d_cond"})
# Parts that draw from the CPU random generator; each starts from the state its predecessor stored.
DRAWING_PARTS = frozenset({"ss", "lr", "hr", "tex"})
# Parts after which PyTorch's cache is given back at once: the conditioning passes and the up-sampling decoder
# leave large freed blocks that the next prop's part does not reuse.
TRIMMED_PARTS = frozenset({"features", "cond_lr", "upsample", "cond_hr", "hi3d_cond", "hi3d_encode",
                           *hi3d.DECODE_PARTS})
# The parts of Pixal3D's own geometry; a Hi3D prop runs hi3d.PARTS in their place.
PIXAL3D_GEOMETRY = frozenset({"cond_lr", "lr", "upsample", "hr"})

# The networks a group's `enter` loads, in loading order ({kind} = "sv" | "mv"), and those a part loads itself when
# a prop first needs them.
ENTER = {
    "cutout": (), "camera": ("moge",), "features": ("dino",), "ss": ("ss_flow.{kind}", "ss_dec"),
    "cond_lr": ("naf",), "lr": ("lr_flow.{kind}",), "upsample": ("shape_dec",), "cond_hr": ("naf",),
    "hr": ("hr_flow.{kind}",), "tex": ("tex_flow.{kind}",), "finish": (), **hi3d.ENTER,
}
LAZY = {"cutout": ("rembg",), "finish": ("shape_dec", "tex_dec")}
LIGHT = frozenset({"naf"})              # megabytes: reading its file ahead gains nothing
FIRST_NETWORK = LAZY["cutout"][0]       # what a batch loads first: read ahead when the engine is made, and after a
#                                         batch's last network for the next batch (engine.py, runner._read_next_batch)

# Large store items: what a part takes (the runner asks the store for the next prop's ahead of time), what it puts,
# and what nobody reads once the part is done.
WANTS = {"ss": ("cond_ss",), "cond_lr": ("dino512",), "lr": ("cond_lr",), "cond_hr": ("dino1024",),
         "hr": ("cond_hr",), "tex": ("cond_tex",), **hi3d.WANTS}
PRODUCES = {"features": ("dino512", "dino1024", "cond_ss"), "cond_lr": ("cond_lr",),
            "cond_hr": ("cond_hr", "cond_tex"), **hi3d.PRODUCES}
SPENT = {"ss": ("cond_ss",), "cond_lr": ("dino512",), "lr": ("cond_lr", "rng_ss"),
         "upsample": ("coords_ss", "lr_slat"), "cond_hr": ("dino1024", "prep", "bundle"),
         "hr": ("cond_hr", "rng_lr"), "tex": ("cond_tex", "rng_hr")}

# What the end of a prop needs, as measured on an RTX 3060 over the captured reference cases and the bulky prop
# (code/logs/profiling/pixal3d-lowmem/staged in Spellcast3D). The decoders: the weights of one; what a decode
# allocates per thousand voxels of the result; and how many voxels a shape of M occupied voxels on the 512 grid
# decodes to at resolution R: about M * (R / 512)^2 (0.97-1.12 of it over the measured props). The flow samplers at
# the 1024 grid: what they allocate at 0 tokens and per thousand tokens, with a quarter on top for the allocator's
# fragmentation under a cap. The finish hook on the CPU: what the finish of a prop adds to the process's working
# set beside what it holds at `upsample` (the decoded mesh and its textures in RAM, the BVH's copy of its tree,
# the simplification, the maps) and per million voxels of the shape - measured on the five test props at 1536
# (8.8-12.3 M voxels, up to four in their finish at once): the launcher's private working set rose from 3.0 GB
# before the finish part to 6.7 GB, 3.7 GB for all of them (round-e/e1536-full). These figures decide, in `upsample`, the
# highest resolution a 1536 prop can be made at within the card and the RAM, and in `finish` whether both decoders
# may share the card and whether a decode is hopeless under the cap. Where the decoders decode in boxes
# (_decodes_in_boxes), a big shape is decoded in boxes (below; on spconv only one whose whole-table decode would not
# fit: SPCONV_BOXES_WHERE_NEEDED) and its decode no longer grows with the prop; what then
# bounds a 1536 prop on a small card is the closing of its holes, the samplers, and the RAM. Where they do not, the
# whole-table decode is what is counted. The decode's figures here are flex_gemm's; spconv's (the cards below sm_80)
# are SPCONV_* below, and decode_figures() gives the ones of the backend in use.
DECODER_MIB = 905
DECODE_MIB_PER_KVOXEL = 0.52
VOXELS_PER_OCCUPIED = 1.15          # decoded voxels per occupied 512-grid voxel scaled to the resolution, with margin
VOXELS_PER_TOKEN = 400              # voxels a token decodes to at most (396 measured on the bulky prop): used only
#                                     where the occupied voxels are not known (a prop resumed from a checkpoint
#                                     that does not carry them: one written before round O)
CHECKPOINT_VOXELS = "decode_voxels"   # the key of the foreseen voxels in a checkpoint's notes (tex, runner._restore)
# The decode in boxes (models/sc_vaes/chunked.py: the same values, bit for bit). A shape whose whole-table decode
# would take more than CHUNK_ABOVE_MIB is decoded a box of CHUNK_VOXELS fine voxels at a time, and its peak stops
# growing with the prop: DECODER_MIB, CHUNKED_MIB_PER_KVOXEL per thousand voxels of the prop (the coarse levels on
# whole tables, the assembly, the mesh extraction; the texture decode's mesh on the card) and CHUNK_MIB_PER_KVOXEL
# per thousand voxels of a box (measured: a 16.45 M-voxel shape 3,103-3,423 MiB with 1-2 M boxes, a 7.3 M one
# 1,946 shape / 2,725 texture with 2 M boxes, the texture decode holding the mesh; the formula gives 4,002-4,302 and
# 2,743). On flex_gemm every 1536 prop of the test set goes this way, and a 1024 prop stays whole (on spconv the
# threshold is the backend's own: SPCONV_CHUNK_ABOVE_MIB). Boxes cost 4-9 % of the decode's time, and the first
# chunked decode of an install autotunes flex_gemm's kernels for the new row counts (a few seconds, kept in its
# cache).
CHUNK_ABOVE_MIB = 3_000
CHUNK_VOXELS = 2_000_000
CHUNKED_MIB_PER_KVOXEL = 0.17
CHUNK_MIB_PER_KVOXEL = 0.30
# Closing the mesh's holes (cumesh: native memory, outside PyTorch's cap but on the card): about 67 bytes a face
# (get_edges' sort: two buffers of 3F keys beside cumesh's copy of the mesh; cumesh 0.0.1.post2, whose get_edges
# frees its sort buffers and whose tables are allocated at their exact size, and Mesh.fill_holes without its clones
# of the mesh - 160 bytes a face before, round G), 2.0-2.2 faces a voxel, beside the mesh itself; the texture tables
# are off the card by then. Measured 0.129 / 0.138 MiB per thousand voxels above the mesh (sv1536 / big1536), kept
# with the same margin as before. The decode in boxes is the finish part's widest moment again.
FILL_MIB_PER_KVOXEL = 0.15
# The flow samplers at the 1024 grid: the network's weights (the loader's figure once it is loaded, else what this
# card's torso and the int8 policy give: flow_weights_mib) and what a sampling allocates beyond them per thousand
# tokens (40-45 MiB measured on the five test props' hr and tex at 1536 with the int8 networks and the proj_linear
# answers counted out), with a quarter on top for the allocator's fragmentation under a cap. Until 2026-11-07 the
# base was the bf16 network's 2,700 MiB whatever the policy, so on a card whose cap is ~4.6 GB the dragon, knight
# and ork of the test set were made at 1152-1408 where 1536 fits.
FLOW_BF16_MIB = 2_680               # a flow network's weights in bf16 (the loader reports 2,678-2,679 MiB)
FLOW_INT8_CONVERTED_MIB = 1_121     # the token-sized linears as int8 (policy "detail"; 2,242 MiB in bf16)
FLOW_REST_MIB = 436                 # what stays in the torso's dtype beside them, in bf16
SAMPLER_MIB_PER_KTOKEN = 50        # bf16 and fp16 torsos: the turing profile's fp16 torso with its fp32 stream
#                                     allocates 57.4-58.2 MiB a thousand tokens beyond the weights and the tables
#                                     (hr: the RTX 2070's boy at 1024, the 3060 forced to turing at 1024 (n5) and
#                                     1536 (M1); R3 section 1), inside the 62.5 this gives with the slack, which adds
#                                     a quarter of the weights (~390 MiB) on top: 530-590 MiB of margin at 30-44k tokens
SAMPLER_SLACK = 1.25
# An fp32 torso (the pascal profile, GTX 10xx) allocates 87.0-87.2 MiB a thousand tokens beyond the weights and the
# tables in hr (the boy at 1024, 5.1k tokens, round N's forced-pascal pas-on; the boy at 1536, 12.1k tokens, M1's
# pas-a: 5,114 - 2,129 of tables - 1,929 of weights), 73-74 in tex: the same rate at both token counts, its activations
# in fp32 twice bf16's. With bf16's 50 sampler_mib said 62.5 and let a GTX 10xx's bulkiest props through at 1536 with
# ~0.6 GB too little (R3 section 2c: the tree's hr at 43.5k tokens 5,130 counted against ~5,714). 90 (3 % over the
# measured rate), with the same slack. Checked under the pessimistic 8 GB cap (round O, runs/tree-pas-hr: the tree,
# the 3060 forced to pascal, PyTorch's cap 6,061 MiB from hr on, no proj tables): the first hr step allocated 4,482
# MiB at ~30k tokens (the tree made at 1280: its whole-table decode at 1536 does not fit a 12 GB card on the products
# path), 84.5 a thousand tokens beyond the weights, conditioning included; this rule counts 5,809 there. At 43.5k
# tokens it counts 7,305, so a GTX 10xx with 6-7 GB for PyTorch makes such a prop at 1280-1408 (unmeasured at
# 43.5k under that cap: ~5.7-6.0 GB allocated foreseen, within 5 % of it).
#   Since round O's finish-up the old cards' int8 MLP runs in row slabs on pixal3d_kernels' GEMM path
# (modules/int8mm.py mlp_slab_rows: 256 MiB of hidden rows, 8,192 rows a slab in fp32, 16,384 in fp16; the same bits),
# so beyond the slab the MLP's hidden table no longer grows with the prop: the MLP alone took 547 / 900 / 1,378 /
# 1,967 MiB whole and 393 / 439 / 501 / 578 slabbed at 12,110 / 19,928 / 30,500 / 43,529 tokens (pascal, isolated:
# finish-up-REPORT section 4) - 45.2 a thousand tokens whole, ~5.9 a thousand beyond the slab. The boy at 1536
# (12.1k tokens) then allocated 76.5 a thousand in hr (63.2 in tex) where it took 86.6 (M1's pas-a; finish/runs/
# pas1536-slab), the rest of a block ~44 a thousand (926 - 393 MiB of the MLP over 12.11k). So the fp32 figure is
# piecewise: SAMPLER_FP32_MIB_PER_KTOKEN for the tokens of the first slab (or all of them where nothing is slabbed),
# SAMPLER_FP32_SLABBED_MIB_PER_KTOKEN beyond it - the rest of a block (44) and the slabbed MLP's growth (5.9), 4 %
# over - both with SAMPLER_SLACK. The boy: 941 MiB counted against 926 measured beyond the weights (3,588 with the
# slack against 2,863 allocated beyond the proj tables). The tree at 43.5k tokens: 2,575 counted against ~2,492
# foreseen (44 x 43.5 + the MLP's measured 578), 5,710 with the weights before their load (1,993) and the slack,
# under the pessimistic 6,061 MiB cap: a GTX 10xx makes it at 1536, foreseen at ~4.4 GB allocated (1,936 + 2,492),
# ~27 % under that cap - a foresight, to be checked by the tree's run under the forced pascal profile and the
# emulated cap. Where the MLP is not slabbed (no pixal3d_kernels GEMM, the int8 policy off, PIXAL3D_MLP_SLAB_MIB=0)
# the whole figure stands for every token (_mlp_slab_rows). The fp16 torso (turing) keeps bf16's figure: its slabs
# start at 16,384 tokens and only lower its need (the MLP at 43,529 tokens 1,159 -> 516 MiB), inside a rule that
# already counts 62.5 a thousand against the 57.4-58.2 measured whole.
SAMPLER_FP32_MIB_PER_KTOKEN = 90
SAMPLER_FP32_SLABBED_MIB_PER_KTOKEN = 52
FLOW_MLP_HIDDEN = 8_192             # the 1024-grid flows' MLP hidden channels (4 x 2,048): what a slab is cut by
PROJ_CACHE_SLACK = 1.1              # over the kept proj_linear answers' bytes (proj_cache_count)
PROJ_CACHE_MARGIN_MIB = 512         # free under the cap beside the sampler and the kept tables (proj_cache_count).
#                                     sampler_mib is itself above a bf16 sampling's peak: on a 7.2 GiB card's 6.2 GB cap
#                                     the ork's hr (30.5k tokens) allocated 2,737 MiB without tables (sampler_mib
#                                     3,853) and 5,451 with all 30, the allocator's cache filling the cap with one
#                                     retry and no failure; this margin keeps 18-19 of its blocks there (4.4 GB
#                                     allocated) and all of a 20k-token prop's
PROJ_CACHE_FP32_TOKENS = 2.0        # the per-token part of the sampler beside the tables, times this for an fp32
#                                     torso (proj_cache_sampler_mib): a sampling there allocates 86-89 MiB per
#                                     thousand tokens beyond the weights and the tables (the boy, the pascal profile
#                                     forced on an RTX 3060), twice bf16's 39-45; the fp16 torso with its fp32 stream
#                                     (turing) measured 52, within sampler_mib's 62.5
HOOK_RAM_BASE_MIB = 1_500
HOOK_RAM_MIB_PER_MVOXEL = 200
RAM_FLOOR_MIB = 1_500               # what must stay available to Windows and the hub beside the hook's own need
BUNDLE_RAM_MIB = RAM_FLOOR_MIB + HOOK_RAM_BASE_MIB     # RAM that must stay available beside a multi-view prop's
#                                                       view bundle for it to be kept from `features` to `cond_hr`
# `cond_hr` makes the shape and the texture conditionings together (proj_cond.cond_tokens_pair: NAF's image encoder
# once per view instead of twice, about 0.36 s a view), for every prop: a multi-view prop adds each view into the
# two float32 sums in place, so at its widest moment it holds the two sums and no copy of a view, and it passes
# under a lower cap than the two calls (round G: 3,776 against 4,448 MiB for 3 views at 31k tokens, 3,552 against
# 4,704 for the 4 views of mv1024, 3,360 against 3,392 for sv1536). The two calls remain the ladder's second rung.


# What a finish hook's GPU step may keep on the card between its steps (types.FinishSteps: state["room_mib"]): a
# cuBVH of the decoded mesh, built for the remesh and needed again for the bake (60-70 bytes a face, 0.6-0.9 GB for
# the meshes of a 1024 prop; cuBVH keeps a copy of its tree in RAM as well). The room is what the card has beside
# what the props still to decode need under the finish cap, or beside what another prop's hook steps need with
# nothing held (HOLD_BESIDE_MIB: the remesh, nvdiffrast, the texture volume and a BVH of their own), whichever is
# more, less a slack; and never more than the RAM allows twice over beside the floor.
HOLD_BESIDE_MIB = 3_000
HOLD_SLACK_MIB = 512
HOLD_RAM_FACTOR = 2

# The paired texture and finish (runner._pair) take the texture network off the card before each decode and read it
# again for the next prop, and load both decoders again for every prop (0.8-1.4 s of reading and trimming a prop at
# 1536). Where the card has room for it they stay instead (pair_keep): the texture network through a prop's finish,
# when another prop of the group follows, and both decoders from one prop's finish to the next, when there is room
# for that as well. Room is judged against the caps the finish will set, with every margin the finish takes itself:
# its decode with both decoders (decoders_fit's), the closing of the holes beside what is kept (finish_mib's), the
# finish hook's GPU steps beside it (their native reserve, glb_reserve, the largest of this prop's and of the props
# in their steps, and what they allocate through PyTorch, HOOK_TORCH_MIB, within what the hook's cap leaves the
# card), and the next prop's texture sampler beside the decoders (sampler_mib, under the cap in force).
FACES_PER_VOXEL = 2.2               # faces of a decoded mesh per voxel of its shape (2.0-2.2 measured), for the
#                                     hook's reserve of a mesh that is not decoded yet
HOOK_TORCH_MIB = 1_500              # what Spellcast3D's GLB GPU steps reserve through PyTorch at the default texture
#                                     size of 2048 (measured: geometry 1,034-1,444 MiB, bake 1,068-1,234 on the
#                                     golden mv1024 / sv1536 meshes; 2,454-2,498 for the bake at 4096): the figure
#                                     where the hook says nothing. A hook that knows its texture size says it
#                                     (Hooks.gpu_torch_mib(req), hook_torch_mib below), so a larger texture keeps
#                                     less rather than running out of memory beside what is kept (that would only
#                                     cost the step a second run with it gone, runner._after_failure)
HOOK_TORCH_REF_MFACES = 20          # the decoded faces (millions) that figure, and the hook's own, hold for: the
#                                     golden sv1536 mesh (14.9 M faces) was the largest it was measured on; round M
#                                     (2026-11-07) measured the geometry step's reserve on a card that keeps no tree
#                                     flat at 1,006-1,194 MiB from 14 to 18 M faces and past 1,089 at 21 M
HOOK_TORCH_MIB_PER_MFACE = 120      # what the GLB GPU steps reserve through PyTorch per million decoded faces
#                                     beyond that: geometry and bake upload the decoded mesh, the bake the texture
#                                     volume (36 B a voxel, about 2 faces a voxel) and grid_sample's tables. Round
#                                     I's 7.2 GiB run: the geometry and the bake of a 21.5 M-face prop and the bake
#                                     of an 18.3 M-face one ran out of memory beside the kept texture network with
#                                     1,593 / 1,593 / 1,785 MiB left for them (pair_keep had counted 1,500 for every
#                                     prop), so each step ran again with the network gone (re-read from the disk)
#                                     and the bake built its BVH anew: 1,500 + 87 a million faces beyond 15 M is the
#                                     least those three say, 120 gives the 18.3 M-face bake ~100 MiB of margin; the
#                                     full RTX 3060 still keeps both decoders beside the int8 texture network for
#                                     every prop of the test set (round J, overlap)

# The decoders give their cache back level by level (decoder.lean: an empty_cache between the levels, the
# results moved through RAM once to leave the card unfragmented) only where the decode needs it. Without it a
# decode reserves more than it allocates - measured with no cap (check_decode.py --no-trim, RTX 3060): on whole
# tables 2.07 / 2.04 times (sv1024: 6,302 / 6,460 MiB reserved for 3,038 / 3,171 allocated, shape / texture), in
# boxes of 2 M voxels 1.27 / 1.23 times (sv1536: 2,476 / 3,354 for 1,943 / 2,724) - and saves 0.2 s a decoder
# (sv1024 5.20 -> 4.81 s for both, sv1536's texture 5.14 -> 4.75 s), the same values both ways (12 of 12
# comparisons each). Where that reserve, with the margin decoders_fit takes, does not fit the room the cap leaves,
# the lean form is used (lean_needed), and always on the ladder's tight rung.
UNLEAN_WHOLE = 2.1
UNLEAN_BOXES = 1.3

# The figures above are flex_gemm's (the Ampere cards, and a Turing card whose calibration found flex_gemm right).
# The cards below sm_80 decode on spconv (modules/precision.py), whose convolutions allocate otherwise: spconv's
# implicit GEMM with its own kernels ("spconv": the RTX 20xx / GTX 16xx), or spconv's gather-scatter with the
# products on torch.mm ("spconv_torch": a card spconv has no prebuilt kernels for, the GTX 10xx; conv_spconv.py).
# decode_figures() gives the figures of the backend this process decodes on; each one says what it was measured on.
# SPCONV_* hold the spconv figures per backend; a backend that cannot decode in boxes (_decodes_in_boxes) never
# uses its box figures: chunk_for gives it whole tables, and resolution_fits keeps a card under
# WHOLE_LEVEL_1536_MIB at 1024 (below).
#
# Whole tables, what a decode allocates per thousand voxels above its decoders' weights (round O, the RTX 3060 under
# the forced profiles, the finish part replayed from captured checkpoints: code/devtools/pixal3d-lowmem/round-o/
# memory, records code/logs/profiling/pixal3d-lowmem/round-o/memory/runs):
#   spconv        0.606-0.618: the boy single view at 1536 (3.49 M voxels) 0.615 with both decoders, 0.612 with one,
#                 0.618 with both beside the texture network (boy-tur), the ork multi view (10.57 M) 0.607 / 0.608 with
#                 one / both (whole-tur), 0.609 with both (M1's tur-ork); the real RTX 2070 at 1024 0.610-0.613 (R3,
#                 2070-u6). 0.63 keeps ~2 % over the largest. With conv_spconv.py's per-level pair tables (the
#                 default since the boxes landed; PIXAL3D_SPCONV_PAIRS=off is the way back, these figures' "before")
#                 a whole table takes less: 0.506-0.523 (boxes-tur: the boy 0.523, the ork 0.506, the tree 0.514),
#                 which 0.63 covers (the products path then 0.683-0.689: boxes-pas-b).
#   spconv_torch  0.80-0.81: the boy at 1536 (3.55 M) 0.807 with both decoders, 0.809 with both beside the
#                 texture network, 0.805 with one beside it (boy-pas), 0.797 / 0.796 with both / one (boy-pas2:
#                 conv_spconv.py with round O's product windows), 0.80 (M1's pas-a); the ork (10.56 M) 0.784 with one
#                 (whole-pas); the boy at 1024 on the products path 0.81 (R3, fp-pas). 0.84 keeps ~4 % over the largest.
# Without decoder.lean a whole-table decode reserved, times what it allocated, with no cap: spconv 1.69 (both
# decoders) and 1.94 (one: its shape decode leaves the cache the texture decode cannot reuse), boy-tur;
# spconv_torch 1.80 (both) and 2.02 (one), boy-pas2 (1.80 in M1's pas-a); the ork's one-decoder decode without
# the trims ran out of a 12 GB card's 10,414 MiB cap at 8,085 allocated / 10,258 reserved, and lean it fitted at
# 9,190 / 10,398 (whole-pas). 2.0 and 2.1. On the products path even
# the lean form reserves 1.45-1.56 times what it allocates with no cap (boy-pas2: the per-offset gathers and
# products of a level stay cached until the level ends); under a cap the allocator gives them back: the boy's
# both-decoder decode allocated 4,642 MiB, reserved 5,760, under the pessimistic 8 GB decode cap of 5,761 and fitted
# (boy-pas2 cap6861-auto), so what decides a fit there is the allocation, which decoders_fit takes with its 10 %.
SPCONV_DECODE_MIB_PER_KVOXEL = {"spconv": 0.63, "spconv_torch": 0.84}
SPCONV_UNLEAN_WHOLE = {"spconv": 2.0, "spconv_torch": 2.1}
# (R1's prototype on the tree capture, the same 3060 and profiles, gives whole decodes of 0.577 / 0.754 MiB a
# thousand voxels above the weights without the shape decoder's mesh, which the texture decode holds beside it:
# 36 bytes a voxel, 0.034; with it 0.611 / 0.788, the figures above. Round O, r1-boxes-REPORT section 3.)
# In boxes (models/sc_vaes/chunked.py on spconv, round O: on by default, PIXAL3D_SPCONV_BOXES=off), measured
# with the replay (runs/boxes-tur, boxes-pas-a / -b; both decoders, lean, 2 M-voxel boxes), above the decoders' weights:
#   the boy 3.49 / 3.55 M voxels   1,353 MiB (turing), 1,568 (pascal)
#   the ork 10.57 M                2,535 (turing), 2,549 (pascal); one decoder 2,525 / 2,539, beside the texture
#                                  network too
#   the tree 12.24 M               3,152 (turing), 3,154 (pascal)
# The same with 1 M boxes as with 2 M, and for the big props the same on both paths: the peak is the texture decode's
# whole-table part (the coarse levels, the assembly of the kept rows, the shape decoder's mesh beside it), not a box's
# convolutions (R1's prototype on a 10.59 M crop: 1,908 / 1,980 MiB without the mesh, 2,361 with it;
# r1-boxes-REPORT section 3). 0.22 a thousand voxels of the prop with 0.33 a thousand voxels of a 2 M box (660) holds
# spconv's (1,427 / 2,984 / 3,353 counted: 5 %, 18 %, 6 % over); the products path's boy needs 0.42 (840: 1,621 /
# 3,164 / 3,533 counted, 3 %, 24 %, 12 % over). Without decoder.lean the boxes reserved 1.44 (the ork, both decoders)
# to 1.61 (the tree, one) times what they allocated on spconv, 1.56-1.70 on the products path, whose lean form itself
# reserves 1.23-1.61 times: 1.8.
# The boxes gave the whole tables' meshes bit for bit wherever both ran (the boy and the ork on both paths, the tree
# on spconv: the decoded meshes' SHA-256), and take +2-10 % of the decode's time on spconv (the boy 4.2 s against 3.8
# whole, the ork 11.6 against 11.4) and ~0-10 % on the products path (the ork 26.4 against 25.1-25.8 s, the boy 8.9
# against 8.1).
#
# WHERE spconv boxes (chunk_for): only where the whole-table decode does not fit. The boxes' bit-equality with the
# whole tables is proven on this PC's RTX 3060 under the forced profiles only; no real RTX 20xx / GTX 10xx has run
# them (R1 section 5). Before round O those cards decoded every prop on whole tables, so a prop whose whole-table
# decode fits the finish's cap stays on that path - every 1024 prop that fitted before, and the boy at 1536 on an
# 8 GB card - and only a decode that could not run whole goes into boxes (the ork and the tree at 1536 on 8 GB). The
# test is decode_mib with one decoder on whole tables against the finish decode's cap with decoders_fit's margin
# (limit / 1.1): the tightest form a whole-table decode takes. The cap is the finish's own (the decode), or as
# card_limits foresees it (upsample's ladder, pair_keep), or what the card has left less the finish's reserves
# (tail_room, the uncapped rung). With no cap to judge (a CPU run) a spconv decode is whole. Once a real card has
# passed the whole-against-boxed gate, SPCONV_BOXES_WHERE_NEEDED = False boxes every decode above
# SPCONV_CHUNK_ABOVE_MIB as on flex_gemm (with these whole figures from ~3.3 M voxels on spconv and ~2.5 M on the
# products path, against 4.0 M on flex_gemm: some 1024 props too - the same meshes where the boxes are bit-equal).
# flex_gemm (Ampere) is not affected: it boxes above CHUNK_ABOVE_MIB as it always did.
SPCONV_BOXES_WHERE_NEEDED = True
# The hold mode (chunked.py, round O's review): where this card's decode_products record lists an output-layer
# boundary band (products_floor.has_boundary: this PC's pascal record does, its turing one does not), the shape
# decoder keeps the boxes' rows before the output layer (fp16 [n, 64] and their coords) and runs that layer at the
# end, so its box peak grows with the prop again: above the decoders, the boy (3.55 M voxels) 809 -> 1,082 MiB at
# 2 M boxes, the ork (10.56 M) 2,131 -> 3,077 (boxes-REPORT section 11.5); without it the shape decode peaked at 807 /
# 2,128 / 2,672 for the boy / ork / tree (12.24 M; runs/boxes-pas-a/-b), the texture decode at 1,568 / 2,549 / 3,154
# (unchanged by the hold). So with the band the box decode counts the larger of the box figures above and
# SPCONV_HOLD_MIB_PER_KVOXEL a thousand voxels of the prop with SPCONV_HOLD_BASE_MIB: the shape decode's ~0.215 a
# thousand (its fit over the three props) and the held rows' ~0.09 (946 MiB over the ork's 10.56 M), 0.31 with 100.
# Measured with the boxes agent's pascal record (runs/hold-pas, 2 M boxes, lean): the ork's shape decode 3,102 MiB
# above both decoders (3,375 counted, 9 % over), the tree's 3,594 (3,894, 8 %), the boy's 1,082 (1,201); under the
# pessimistic 8 GB cap the ork then decodes one decoder at a time, 4,001 / 5,550 MiB allocated / reserved under 5,761,
# the same mesh. The band's own sizes (a fine level
# within 1,024 rows above a multiple of 2,097,088 rows: about one prop in 2,000) also make the output layer's whole
# call once, fp32 [n, 64] beside the held rows (the boy 1,082 -> ~1,660-1,760 MiB, forced): not counted here, as no
# plan can tell those sizes before the decode (the residual risk named in the memory report).
SPCONV_HOLD_MIB_PER_KVOXEL = 0.31
SPCONV_HOLD_BASE_MIB = 100
SPCONV_CHUNKED_MIB_PER_KVOXEL = {"spconv": 0.22, "spconv_torch": 0.22}
SPCONV_CHUNK_MIB_PER_KVOXEL = {"spconv": 0.33, "spconv_torch": 0.42}
SPCONV_UNLEAN_BOXES = {"spconv": 1.8, "spconv_torch": 1.8}
SPCONV_CHUNK_ABOVE_MIB = {"spconv": 3_000, "spconv_torch": 3_000}     # with SPCONV_BOXES_WHERE_NEEDED off only


def hook_torch_mib(hooks: Any, req: Any) -> int:
    """MiB the finish hook's GPU steps reserve through PyTorch for `req`: what the hook says
    (`gpu_torch_mib(req)`, when it has the method and answers a number), else HOOK_TORCH_MIB."""
    ask = getattr(hooks, "gpu_torch_mib", None)
    if ask is None:
        return HOOK_TORCH_MIB
    try:
        return max(int(ask(req)), 0)
    except Exception:
        return HOOK_TORCH_MIB


def steps_torch_mib(hooks: Any, req: Any, faces: int) -> int:
    """MiB the finish hook's GPU steps reserve through PyTorch for a decoded mesh of `faces` faces: the hook's
    figure (hook_torch_mib, which holds up to HOOK_TORCH_REF_MFACES million faces) and HOOK_TORCH_MIB_PER_MFACE for
    every million beyond."""
    beyond = max(0.0, faces / 1e6 - HOOK_TORCH_REF_MFACES)
    return hook_torch_mib(hooks, req) + int(HOOK_TORCH_MIB_PER_MFACE * beyond)


# MiB of the card native code takes in the finish hook's GPU steps beside PyTorch's reserve (cumesh's remesh and
# its tables, the tree uploaded for the remesh and the bake, nvdiffrast): measured on an RTX 3060 emulating a 6 GB
# card (round M, 2026-11-07: the card's use during the geometry steps less PyTorch's reserve) 1,650-2,150 MiB at
# 12-18 M faces, so GLB_RESERVE_BASE_MIB + 60 a million faces keeps 300 MiB over the widest. The 2,300 base used
# until then left the ork at 1536 (21 M faces) 1,089 MiB of PyTorch under a 5.2 GB cap while 4 GB of the card
# were free, and its geometry step ran out.
GLB_RESERVE_BASE_MIB = 1_600
GLB_RESERVE_MIB_PER_MFACE = 60


def reserve_faces(native_mib: int) -> int:
    """The decoded faces glb_reserve gives `native_mib` for (its inverse): what a prop in its steps decoded."""
    return max(0, (int(native_mib) - GLB_RESERVE_BASE_MIB) * 1_000_000 // GLB_RESERVE_MIB_PER_MFACE)


def next_sampler_mib(flow: Any, tokens: int, cap_mib: float, weights_mib: int | None = None) -> float:
    """MiB the next prop's texture sampling takes on the card beside what is kept for it: its sampler
    (proj_cache_sampler_mib, with the flow's `weights_mib` where known) and the proj_linear answers of the blocks
    that will keep theirs under the cap in force (proj_cache_count, the rule of proj_cache_blocks, so that the
    decoders the pair keeps never take that room: the sampling would run out of memory and run again with them
    gone). A flow without such answers, or one that cannot say, needs its sampler alone."""
    sampler = proj_cache_sampler_mib(tokens, weights_mib)
    try:
        tables = proj_cache_tables_mib(flow, tokens) if flow is not None else []
    except Exception:           # noqa: BLE001 - no proj cache to make room for
        tables = []
    kept = proj_cache_count(sampler, tables, cap_mib)
    return sampler + PROJ_CACHE_SLACK * sum(tables[:kept])


def pair_keep(ctx: PartContext, prop: PropState, following: PropState, natives: list) -> frozenset:
    """The networks that may stay on the card through `prop`'s finish in the paired texture and finish, so that
    `following` (the next prop of the group) does not load them again: its texture network, and both decoders after
    the finish when there is room for them as well; nothing where the card cannot be judged (a CPU run, no cap).
    `natives`: the native reserves of the props in their finish steps, whose GPU steps run beside what is kept.
    The texture network stays only when, with it on the card, the decode with both decoders fits under the cap the
    finish will set (with decoders_fit's margin), the closing of the holes beside it does, and so do the finish
    steps: the largest of this prop's and of each prop in its steps of their native reserve (glb_reserve of the
    faces the voxels foresee, or the prop's own) and what they reserve through PyTorch (steps_torch_mib of those
    faces) within what the hook's cap leaves the card (the finish cap and its native reserve) - the decode counted
    with the backend's figures and on whole tables where it has no boxes (decode_mib, chunk_for: on spconv's whole
    tables the RTX 2070's ork at 1024 ran out of memory beside the texture network that flex_gemm's box figures had
    kept, round N);
    the decoders stay only when, beside it, the holes and the steps fit as well and the next prop's texture sampling
    (next_sampler_mib: its sampler, and its proj_linear answers where it will keep them) fits beside them under the
    cap in force."""
    loader = ctx.loader
    size_of = getattr(loader, "size_mib", None)
    headroom = loader.headroom_mib() if hasattr(loader, "headroom_mib") else None
    limits = card_limits(ctx)
    flow = f"tex_flow.{prop.kind}"
    if size_of is None or headroom is None or limits is None or loader.get(flow) is None:
        return frozenset()
    flow_mib = size_of(flow)
    try:
        voxels = decode_voxels(ctx.store.get_small(prop.id, "coords_hr"))
        tokens = int(ctx.store.get_small(following.id, "coords_hr")["coords"].shape[0])
    except Exception:
        return frozenset()
    if not flow_mib:
        return frozenset()
    finish_limit, cap_now = limits
    budget = finish_limit / 1.1
    decoders_mib = sum(size_of(name) or DECODER_MIB for name in LAZY["finish"])
    # the hook's cap leaves FINISH_SAFETY_MIB of the card free: what the decode's cap does, less its native reserve
    hooks = getattr(ctx, "hooks", None)
    faces = int(FACES_PER_VOXEL * voxels)
    steps = max([glb_reserve(faces) + steps_torch_mib(hooks, prop.req, faces),
                 *[int(mib) + steps_torch_mib(hooks, prop.req, reserve_faces(int(mib))) for mib in natives]])
    steps_room = finish_limit + NATIVE_RESERVE["finish"]
    sampler = next_sampler_mib(loader.get(flow), tokens, cap_now, flow_mib)

    chunk = chunk_for(voxels, finish_limit)          # the decode's own decision, under the cap foreseen for it

    def fits(decoders: bool) -> bool:
        after = flow_mib + (decoders_mib if decoders else 0)
        return (flow_mib + decode_mib(voxels, 2, chunk) <= budget and after + fill_mib(voxels) <= budget
                and after + steps <= steps_room
                and (not decoders or sampler + decoders_mib <= cap_now))

    if fits(decoders=True):
        return frozenset({flow, *LAZY["finish"]})
    return frozenset({flow}) if fits(decoders=False) else frozenset()


def lean_needed(voxels: int, decoders: int, chunk: int, room_mib: float) -> bool:
    """Must a decode of `voxels` voxels with `decoders` decoders on the card give its cache back level by level
    (decoder.lean) to fit `room_mib` (the cap less what else is kept on the card)? What a decode without it reserves
    is the backend's unlean factor (decode_figures) times decode_mib. A CPU run (no room known) needs nothing."""
    figures = decode_figures()
    factor = figures["unlean_boxes"] if chunk else figures["unlean_whole"]
    return room_mib > 0 and factor * decode_mib(voxels, decoders, chunk) > room_mib / 1.1


def pair_order(ctx: PartContext, props: list) -> list:
    """The order the props of a group go through the paired texture and finish (runner._pair_order): the biggest
    shape first, by the voxels `upsample` foresaw (coords_hr["voxels"]; a resumed prop by its token count), so
    that the last prop's finish steps - the only ones nothing hides under - are the smallest prop's. A prop's
    numbers do not depend on its place."""
    def voxels(prop: PropState) -> int:
        try:
            return decode_voxels(ctx.store.get_small(prop.id, "coords_hr"))
        except Exception:
            return 0
    return sorted(props, key=voxels, reverse=True)       # stable: equal shapes keep the batch's order


def glb_reserve(faces: int) -> int:
    """MiB of native memory the caller's finish hook needs for a decoded mesh of `faces` faces (to_glb: the
    nvdiffrast context, one BVH at a time, cumesh): GLB_RESERVE_BASE_MIB and GLB_RESERVE_MIB_PER_MFACE. Too small
    and the hook spills into shared memory, too large and PyTorch is left too little for the steps themselves."""
    return GLB_RESERVE_BASE_MIB + GLB_RESERVE_MIB_PER_MFACE * faces // 1_000_000


def voxels_at(occupied: int, resolution: int) -> int:
    """The voxels a shape with `occupied` voxels on the 512 grid decodes to at `resolution`, with margin."""
    return int(VOXELS_PER_OCCUPIED * occupied * (resolution / 512) ** 2)


def decode_backend() -> str:
    """What this process's sparse decoders convolve on: "flex_gemm" (also any backend other than spconv: the
    figures that always stood for it), "spconv" (spconv's own GEMM kernels) or "spconv_torch" (spconv's
    gather-scatter with the products on torch.mm, conv_spconv._products_in_torch: a card spconv has no prebuilt
    kernels for). Asked at every call: the configuration may be changed by a test."""
    from ..modules.sparse import config as sparse_config

    if sparse_config.CONV != "spconv":
        return "flex_gemm"
    from ..modules.sparse.conv import conv_spconv

    return "spconv_torch" if conv_spconv._products_in_torch() else "spconv"


def decode_figures() -> dict:
    """The decode's figures for this process's backend (decode_backend): "whole" (MiB a thousand voxels on whole
    tables), "chunked" / "chunk" (in boxes: a thousand voxels of the prop / of a box), "unlean_whole" /
    "unlean_boxes" (what a decode without decoder.lean reserves, times what it allocates), "chunk_above" (the
    whole-table decode above which boxes are used). flex_gemm's are this module's constants, read at the call."""
    backend = decode_backend()
    if backend == "flex_gemm":
        return {"whole": DECODE_MIB_PER_KVOXEL, "chunked": CHUNKED_MIB_PER_KVOXEL, "chunk": CHUNK_MIB_PER_KVOXEL,
                "unlean_whole": UNLEAN_WHOLE, "unlean_boxes": UNLEAN_BOXES, "chunk_above": CHUNK_ABOVE_MIB}
    return {"whole": SPCONV_DECODE_MIB_PER_KVOXEL[backend], "chunked": SPCONV_CHUNKED_MIB_PER_KVOXEL[backend],
            "chunk": SPCONV_CHUNK_MIB_PER_KVOXEL[backend], "unlean_whole": SPCONV_UNLEAN_WHOLE[backend],
            "unlean_boxes": SPCONV_UNLEAN_BOXES[backend], "chunk_above": SPCONV_CHUNK_ABOVE_MIB[backend],
            "hold": SPCONV_HOLD_MIB_PER_KVOXEL if _holds_tail_rows() else 0.0}


def _holds_tail_rows() -> bool:
    """Whether this card's decode_products record makes the shape decoder hold its boxes' rows before the output
    layer (an output-layer boundary band: products_floor.has_boundary; SPCONV_HOLD_MIB_PER_KVOXEL)."""
    try:
        from ..modules.sparse.conv import products_floor

        return bool(products_floor.has_boundary())
    except Exception:       # noqa: BLE001 - no reader (an older checkout), no record: no hold
        return False


def decode_mib(voxels: int, decoders: int = 1, chunk: int = 0) -> float:
    """MiB PyTorch allocates to decode a shape of `voxels` voxels with `decoders` decoders on the card: on whole
    tables (chunk 0), or in boxes of `chunk` fine voxels - with the figures of the backend the decoders run on
    (decode_figures)."""
    figures = decode_figures()
    if chunk:
        boxes = figures["chunked"] * voxels / 1000 + figures["chunk"] * chunk / 1000
        if figures.get("hold"):         # spconv's hold mode: the shape decode's held rows (SPCONV_HOLD_MIB_PER_KVOXEL)
            boxes = max(boxes, figures["hold"] * voxels / 1000 + SPCONV_HOLD_BASE_MIB)
        return decoders * DECODER_MIB + boxes
    return decoders * DECODER_MIB + figures["whole"] * voxels / 1000


def chunk_for(voxels: int, limit_mib: float | None = None) -> int:
    """The box size a shape of `voxels` voxels is decoded in (decoder.chunk_voxels), 0 for a whole-table decode -
    the one decision the decode, upsample's ladder, pair_keep, decoders_fit, lean_needed and tail_room all count
    with. Only where the backend decodes in boxes at all (_decodes_in_boxes: elsewhere every level is decoded whole
    whatever chunk_voxels says). flex_gemm: boxes where the whole-table decode would take more than CHUNK_ABOVE_MIB
    (`limit_mib` is not looked at). spconv: boxes only where the whole-table decode with one decoder does not fit
    `limit_mib`, the finish decode's cap, with decoders_fit's margin; whole tables where it fits or where there is no
    cap to judge (SPCONV_BOXES_WHERE_NEEDED: why)."""
    if not _decodes_in_boxes(voxels):
        return 0
    figures = decode_figures()
    if decode_backend() == "flex_gemm" or not SPCONV_BOXES_WHERE_NEEDED:
        return CHUNK_VOXELS if decode_mib(voxels) > figures["chunk_above"] else 0
    if not limit_mib or limit_mib <= 0:
        return 0
    return CHUNK_VOXELS if decode_mib(voxels) > limit_mib / 1.1 else 0


def fill_mib(voxels: int) -> float:
    """MiB of the card the closing of the holes of a shape of `voxels` voxels takes (cumesh's native tables beside
    the mesh)."""
    return FILL_MIB_PER_KVOXEL * voxels / 1000


def finish_mib(voxels: int, decoders: int = 1, limit_mib: float | None = None, chunk: int | None = None) -> float:
    """MiB of the card the finish part's widest moment takes for a shape of `voxels` voxels: its decode (with
    `decoders` decoders on the card, in the boxes `chunk` gives - by default chunk_for's under the finish decode's cap
    `limit_mib` - with the backend's figures: decode_mib) or the closing of its holes, whichever is more."""
    chunk = chunk_for(voxels, limit_mib) if chunk is None else chunk
    return max(decode_mib(voxels, decoders, chunk), fill_mib(voxels))


def sampler_mib(tokens: int, weights_mib: int | None = None) -> float:
    """MiB a 1024-grid flow sampler needs on the card for a prop of `tokens` tokens, fragmentation included:
    the network's `weights_mib` (a bf16 network's when not known) and what the sampling allocates beyond them -
    SAMPLER_MIB_PER_KTOKEN a thousand tokens for a bf16 or fp16 torso; for an fp32 one (the pascal profile)
    SAMPLER_FP32_MIB_PER_KTOKEN up to the MLP's slab rows (_mlp_slab_rows) and SAMPLER_FP32_SLABBED_MIB_PER_KTOKEN
    beyond them."""
    weights = FLOW_BF16_MIB if weights_mib is None else weights_mib
    if precision.torso() == torch.float32:
        slab = min(tokens, _mlp_slab_rows(tokens))
        per_token = (SAMPLER_FP32_MIB_PER_KTOKEN * slab + SAMPLER_FP32_SLABBED_MIB_PER_KTOKEN * (tokens - slab)) / 1000
        return SAMPLER_SLACK * (weights + per_token)
    return SAMPLER_SLACK * (weights + SAMPLER_MIB_PER_KTOKEN * tokens / 1000)


def _mlp_slab_rows(tokens: int) -> int:
    """The rows a slab of the hr / tex flows' int8 MLP holds for `tokens` tokens on this card (modules/int8mm.py
    mlp_slab_rows), where it is slabbed: the int8 policy converts those flows and pixal3d_kernels' GEMM path serves
    them (taken, or foreseen before the first conversion: the package loads); `tokens` (one call) elsewhere, and
    where int8mm cannot say."""
    try:
        from ..modules import int8mm

        if not (_int8_converts("hr") and _int8_converts("tex")):
            return tokens
        path = int8mm.gemm_path()
        if path not in ("cuda", "") or (path == "" and int8mm.cuda_gemm_op() is None):
            return tokens
        slab = getattr(int8mm, "mlp_slab_rows", None)
        return tokens if slab is None else int(slab(int(tokens), FLOW_MLP_HIDDEN, precision.torso()))
    except Exception:       # noqa: BLE001 - an int8mm that cannot say: no slabs, the whole figure
        return tokens


def proj_cache_sampler_mib(tokens: int, weights_mib: int | None = None) -> float:
    """MiB a flow sampler needs on the card beside the proj_linear answers it keeps (the one figure of
    proj_cache_blocks and next_sampler_mib): sampler_mib's for a bf16 or fp16 torso; for an fp32 one bf16's
    per-token part PROJ_CACHE_FP32_TOKENS times as wide (100 a thousand tokens: a little more than sampler_mib's
    SAMPLER_FP32_MIB_PER_KTOKEN, kept as it was measured and tested for the tables). Which props a card can make
    (resolution_fits) stays with sampler_mib."""
    if precision.torso() != torch.float32:
        return sampler_mib(tokens, weights_mib)
    weights = FLOW_BF16_MIB if weights_mib is None else weights_mib
    return SAMPLER_SLACK * (weights + PROJ_CACHE_FP32_TOKENS * SAMPLER_MIB_PER_KTOKEN * tokens / 1000)


def flow_weights_mib(ctx: PartContext, kind: str) -> int:
    """MiB the flow network of `kind` (hr, tex, lr; ss) takes on the card: the loader's figure while it is loaded,
    else what this card's torso and the int8 policy give (a converted flow keeps its token-sized linears as int8
    whatever the torso)."""
    size_of = getattr(getattr(ctx, "loader", None), "size_mib", None)
    if size_of is not None:
        for name in (f"{kind}_flow.sv", f"{kind}_flow.mv"):
            try:
                mib = size_of(name)
            except Exception:       # noqa: BLE001 - a loader that cannot say: the figures below
                mib = None
            if mib:
                return int(mib)
    scale = torch.finfo(precision.torso()).bits / 16          # 1 for a bf16 torso, 2 for fp32
    if _int8_converts(kind):
        return int(FLOW_INT8_CONVERTED_MIB + FLOW_REST_MIB * scale)
    return int(FLOW_BF16_MIB * scale)


def _int8_converts(kind: str) -> bool:
    """Whether the int8 policy in force turns the flow of `kind` into int8 layers (modules/int8.py), as far as can
    be known before it is loaded: the policy names it and the kernels have not been found wanting on this card."""
    try:
        from ..modules import int8
    except Exception:           # noqa: BLE001 - no int8 module (fakes): bf16
        return False
    name = int8.policy()
    if not name or not int8.POLICIES.get(name, {}).get(kind):
        return False
    return getattr(int8, "_state", None) is not False


def proj_cache_tables_mib(flow: Any, tokens: int) -> list:
    """MiB of the proj_linear answer each block of the flow keeps for a prop of `tokens` tokens, block by block."""
    from ..modules.sparse.attention.proj_attention import proj_cache_tables

    return [size / 2**20 for size in proj_cache_tables(flow, tokens)]


def proj_cache_mib(flow: Any, tokens: int) -> float:
    """MiB the flow's blocks keep when all of them cache their proj_linear answers for a prop of `tokens` tokens."""
    return sum(proj_cache_tables_mib(flow, tokens))


def proj_cache_count(sampler: float, tables: list, cap_mib: float) -> int:
    """How many blocks of a flow, the first ones, may keep their proj_linear answers (`tables`: the MiB of each
    block's) beside a sampler of `sampler` MiB under `cap_mib`: each table with PROJ_CACHE_SLACK on top, and
    PROJ_CACHE_MARGIN_MIB to spare beside them all (the one rule of proj_cache_blocks and next_sampler_mib)."""
    room = cap_mib - sampler - PROJ_CACHE_MARGIN_MIB
    count = 0
    for mib in tables:
        room -= PROJ_CACHE_SLACK * mib
        if mib <= 0 or room < 0:
            break
        count += 1
    return count


def proj_cache_blocks(ctx: PartContext, flow: Any, tokens: int, kind: str, mode: str) -> int:
    """How many of the flow's blocks keep their proj_linear answer across a sampling's conditional steps for a
    prop of `tokens` tokens (`kind`: ss, lr, hr or tex): the same bits, one GEMM a block cheaper every conditional
    pass after the first (all 30 blocks: 157 ms of a 3.1 s detail pass at 21.5k tokens, for 1.9 GB on the card
    there). As many blocks as the cap in force has room for beside the sampler's own need (proj_cache_sampler_mib;
    proj_cache_count: every block on the 12 GB card at 1536, some on a 7-8 GB one); none on the ladder's tight rung
    (an attempt that ran out of memory runs again without them), on the CPU, or when the cap was lifted."""
    if mode != "normal":
        return 0
    limits = card_limits(ctx)
    if limits is None:
        return 0
    tables = proj_cache_tables_mib(flow, tokens)
    return proj_cache_count(proj_cache_sampler_mib(tokens, flow_weights_mib(ctx, kind)), tables, limits[1])


def _drop_proj_cache(ctx: PartContext, flow: Any, part: str, prop: PropState, blocks: int) -> None:
    from ..modules.sparse.attention.proj_attention import drop_proj_caches, proj_cache_tables

    freed = drop_proj_caches(flow)
    if freed:
        print(f"[PROJ-CACHE] {part} {prop.id}: proj_linear's answers kept across the conditional steps in "
              f"{blocks} of {len(proj_cache_tables(flow, 1))} blocks ({freed / 2**20:,.0f} MiB), dropped", flush=True)


def hook_ram_mib(voxels: int) -> float:
    """MiB of RAM the finish hook takes for a mesh of `voxels` voxels beside what the process already holds."""
    return HOOK_RAM_BASE_MIB + HOOK_RAM_MIB_PER_MVOXEL * voxels / 1_000_000


def card_limits(ctx: PartContext) -> tuple[int, int] | None:
    """(the cap PyTorch will get for the decode phase of `finish`, the cap in force now) in MiB. The first is
    foreseen from the second: the parts before `finish` leave the same networks' room, only the reserves differ.
    None when nothing is capped: the card is not the limit (a CPU run, or the cap was lifted)."""
    now = ctx.loader.stats().get("cap_mib")
    if not now:
        return None
    safety = getattr(ctx.loader, "safety_mib", 300)
    return int(now - ((NATIVE_RESERVE["finish"] + FINISH_SAFETY_MIB) - (NATIVE_RESERVE[ctx.part] + safety))), int(now)


def ram_available_mib() -> int | None:
    """RAM Windows can still hand out, or None where it cannot be asked."""
    try:
        import psutil

        return int(psutil.virtual_memory().available) >> 20
    except Exception:
        return None


# Without the decode in boxes (_decodes_in_boxes false: a backend chunked.py cannot cut), the decoders decode a level
# whole: the texture decoder of a 1536 prop then ran out of an 8 GB card's memory (an RTX 2060 Super on spconv, the
# knight; Spellcast3D round L), so such a card makes its props at 1024. A card with this much decodes 1536 whole.
# Since round O the finish test of resolution_fits counts such a decode on whole tables with the backend's own
# figures (decode_figures), which would let a boy-sized prop through at 1536 on 8 GB; this rule stays as the safety
# net for a backend without boxes, and does not apply where the decoders cut their levels into boxes.
WHOLE_LEVEL_1536_MIB = 10 * 1024


def _decodes_in_boxes(voxels: int | None = None) -> bool:
    """Whether the decoders of this process decode a level in boxes when told to (decoder.chunk_voxels) - for a
    shape of `voxels` fine voxels where that is given: what models/sc_vaes/chunked.py says (boxes_supported: flex_gemm's
    masked split-K; spconv where this card's decode_products measurement says its boxes are exact, and with
    `voxels` where every level of such a shape is within the whole calls it verified - a shape it does not cover is
    planned whole, or at 1024 below WHOLE_LEVEL_1536_MIB), or - a chunked.py without that question - flex_gemm's
    masked split-K alone, the one algorithm whose rows its boxes reproduced before."""
    try:
        from ..models.sc_vaes import chunked

        supported = getattr(chunked, "boxes_supported", None)
        if supported is not None:
            if voxels is not None:
                try:
                    return bool(supported(int(voxels)))
                except TypeError:       # a chunked.py whose question takes no shape
                    pass
            return bool(supported())
    except Exception:       # noqa: BLE001 - a chunked.py that cannot be asked: the test it always made
        pass
    from ..modules.sparse import config as sparse_config
    from ..modules.sparse.conv import config as conv_config

    return sparse_config.CONV == "flex_gemm" and conv_config.FLEX_GEMM_ALGO == "masked_implicit_gemm_splitk"


def _card_total_mib(ctx: PartContext) -> float:
    try:
        return torch.cuda.get_device_properties(ctx.device).total_memory / (1 << 20)
    except Exception:       # noqa: BLE001 - no card to ask (the fakes): no cap from here
        return float("inf")


def resolution_fits(ctx: PartContext, occupied: int, tokens: int, resolution: int,
                    voxels: int | None = None, req: Any = None) -> str | None:
    """Why a prop of `tokens` tokens at `resolution` would not fit the memory this PC leaves Pixal3D - "card" or
    "ram" - or None when it fits. The card: the widest moment of `finish` (finish_mib: the decode with the figures of
    the backend the decoders run on - in boxes for a big shape where that backend has boxes, on whole tables where it
    has none - or the closing of the holes) under the cap it will get, the 1024-grid samplers (sampler_mib, with the
    torso's own figure a thousand tokens) under the cap in force, and the finish hook's GPU steps - what they reserve
    through PyTorch (steps_torch_mib, for the faces the
    voxels foresee) under the cap they get beside their native reserve (glb_reserve: on a card that leaves Pixal3D
    5.2 GB the ork of the test set at 1536, 21 M faces, got 1,089 MiB there and its geometry step ran out at about
    1,100); the RAM: what the finish hook needs beside the process. The voxels are foreseen from the `occupied`
    cells of the 512 grid, or given (a Hi3D prop's mesh has them exactly); `req`: the prop's request, for the
    hook's own texture-size figure. Where the decoders have no boxes, a card under WHOLE_LEVEL_1536_MIB keeps its
    props at 1024 whatever the figures say (the safety net of round L)."""
    voxels = voxels_at(occupied, resolution) if voxels is None else voxels
    limits = card_limits(ctx)
    if limits is not None:
        finish_limit, cap_now = limits
        if finish_mib(voxels, limit_mib=finish_limit) > finish_limit / 1.1 or \
                sampler_mib(tokens, flow_weights_mib(ctx, "hr")) > cap_now:
            return "card"
        faces = int(FACES_PER_VOXEL * voxels)
        steps_cap = finish_limit + NATIVE_RESERVE["finish"] - glb_reserve(faces)
        if steps_torch_mib(getattr(ctx, "hooks", None), req, faces) > steps_cap:
            return "card"
        if resolution > 1024 and not _decodes_in_boxes(voxels) and _card_total_mib(ctx) < WHOLE_LEVEL_1536_MIB:
            return "card"
    available = ram_available_mib()
    if available is not None and hook_ram_mib(voxels) > available - RAM_FLOOR_MIB:
        return "ram"
    return None


def tail_room(ctx: PartContext, pending: list, held_mib: int) -> int:
    """MiB a finish hook's GPU step may keep on the card when it returns, for a later step of the same prop
    (runner: state["room_mib"]). `pending`: the props of the finish part still to be decoded; `held_mib`: what the
    other props in their steps hold already (the pending props' finish_mib with both decoders: the backend's
    figures). 0 when nothing may stay: a CPU run, a prop whose decode size is not known, or no room."""
    headroom = ctx.loader.headroom_mib() if hasattr(ctx.loader, "headroom_mib") else None
    if headroom is None:
        return 0
    need = 0.0
    limit = headroom - (NATIVE_RESERVE["finish"] + FINISH_SAFETY_MIB)     # the finish cap a next decode would get
    for prop in pending:
        try:
            stored = ctx.store.get_small(prop.id, "coords_hr")
        except Exception:
            return 0
        need = max(need, 1.1 * finish_mib(decode_voxels(stored), decoders=2, limit_mib=limit))
    room = (headroom - (NATIVE_RESERVE["finish"] + FINISH_SAFETY_MIB) - max(need, HOLD_BESIDE_MIB) - held_mib
            - HOLD_SLACK_MIB)
    available = ram_available_mib()
    if available is not None:
        room = min(room, (available - RAM_FLOOR_MIB - HOOK_RAM_BASE_MIB) // HOLD_RAM_FACTOR)
    return max(0, int(room))


def networks(part: str, kind: str) -> tuple[str, ...]:
    """Every network a group of `part` may have on the card."""
    return tuple(name.format(kind=kind) for name in ENTER[part] + LAZY.get(part, ()))


def geometry(prop: PropState) -> str:
    return getattr(prop.req, "geometry", "pixal3d")


def applies(part: str, prop: PropState) -> bool:
    if part in PIXAL3D_GEOMETRY:
        return geometry(prop) == "pixal3d"
    if part in hi3d.PARTS:
        return geometry(prop) == "hi3d" and hi3d.applies(part, prop.req)
    return part != "camera" or not prop.req.multi


def wants(part: str) -> tuple[str, ...]:
    return WANTS.get(part, ())


def produces(part: str) -> tuple[str, ...]:
    return PRODUCES.get(part, ())


def discards(part: str, prop: PropState) -> tuple[str, ...]:
    if part in hi3d.PARTS:
        return hi3d.spent(part, prop.req)
    keys = tuple(key for key in SPENT.get(part, ()) if key != "bundle" or prop.bundled)
    if part == "cond_hr":
        keys += tuple(f"image_{index}" for index in range(prop.views))
    if part == "finish" and geometry(prop) == "hi3d":
        keys += ("hi3d_mesh", "hi3d_subs")      # the hook holds the mesh it was handed
    return keys


def enter(ctx: PartContext, part: str, kind: str) -> None:
    """The group's networks onto the card (those the group before left there stay as they are)."""
    for name in ENTER[part]:
        ctx.entering = name.format(kind=kind)
        _network(ctx, ctx.entering)


def leave(ctx: PartContext, part: str, kind: str) -> None:
    """Drop the group's networks, except those the next group needs (ctx.keep)."""
    held = [name for name in networks(part, kind) if name not in ctx.keep and ctx.loader.get(name) is not None]
    if held:
        ctx.loader.drop(*held)


# =====================================================================================================================
# Shared pieces
# =====================================================================================================================

def _network(ctx: PartContext, name: str) -> Any:
    """The network on the card, loaded now if it is not. Never keep what this returns beyond the part function:
    the loader can only free a network nobody else refers to."""
    module = ctx.loader.get(name)
    return module if module is not None else ctx.loader.load(name, ctx.cancelled)


def _decoder(ctx: PartContext, name: str, chunk: int = 0, lean: bool = True) -> Any:
    """A sparse decoder, in its memory-saving form with `lean` (it gives each level's cache back as soon as the
    level is done: lean_needed says where a decode needs it; the values are the same either way), decoding in
    boxes of `chunk` fine voxels when `chunk` is given (chunk_for; models/sc_vaes/chunked.py)."""
    decoder = _network(ctx, name)
    decoder.lean = lean
    decoder.chunk_voxels = chunk
    return decoder


def _kept_beside(ctx: PartContext) -> int:
    """MiB of the networks on the card that the finish does not use (what the pair kept there for the next prop)."""
    size_of = getattr(ctx.loader, "size_mib", None)
    if size_of is None:
        return 0
    return sum(int(size_of(name) or 0) for name in ctx.loader.loaded() if name not in LAZY["finish"])


def _proj_cond() -> Any:
    from ..pipelines import proj_cond

    return proj_cond


def _decode() -> Any:
    from ..pipelines import decode

    return decode


def _source(ctx: PartContext, prop: PropState) -> tuple[Any, Any]:
    """The prop's picture(s) as the conditioning functions take them, and its camera. A multi-view prop's bundle
    (47 MiB of floats for three views, 63 for four) is built by the first part that needs it (`features`) and kept
    in the store as the small item "bundle" for `cond_lr` and `cond_hr` (spent after cond_hr), where the RAM has
    room for it (BUNDLE_RAM_MIB); where the store has none, the part builds it again: the hook's function of the
    same pictures and the same views, so the same tensors either way. The part keeps it as its last action
    (_keep_bundle)."""
    proj_cond = _proj_cond()
    prep = ctx.store.get_small(prop.id, "prep")
    if prop.req.multi:
        if prop.bundled:
            views = ctx.store.get_small(prop.id, "bundle")
        else:
            images = [ctx.store.get_small(prop.id, f"image_{index}") for index in range(prop.views)]
            views = ctx.hooks.bundle(prop.req, images, prep["meta"])
        return views, proj_cond.Camera.from_bundle(views, ctx.device)
    images = [ctx.store.get_small(prop.id, f"image_{index}") for index in range(prop.views)]
    return images, proj_cond.Camera.single(prep["camera"], ctx.device)


def _keep_bundle(ctx: PartContext, prop: PropState, views: Any) -> None:
    """Keep a multi-view prop's bundle for the later parts that read it, unless the store has it already or the RAM
    would keep less than BUNDLE_RAM_MIB beside it."""
    if not prop.req.multi or prop.bundled or not isinstance(views, dict):
        return
    size = _bundle_mib(views)
    available = ram_available_mib()
    if available is not None and available - size < BUNDLE_RAM_MIB:
        return
    ctx.store.put_small(prop.id, "bundle", views)
    prop.bundled = True


def _bundle_mib(value: Any) -> float:
    """MiB of the tensors in a bundle (its dicts walked through)."""
    if isinstance(value, dict):
        return sum(_bundle_mib(entry) for entry in value.values())
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() / (1 << 20)
    return 0.0


def _torso(cond: dict) -> dict:
    """A conditioning as it is stored: the cast the denoisers make of it in every pass, made once."""
    return {name: value.type(precision.torso()) for name, value in cond.items()}


def _as_reference_layout(cond: dict, multi: bool) -> dict:
    """The dense conditioning laid out in memory as the reference pipelines hand it to the sparse-structure flow.

    Its first linear layer runs one fused addmm on a contiguous [B, N, D] input, and a matmul plus a separate bias
    add on a permuted view; in bfloat16 the two round differently in every block, so the layout decides the sample.
    Single view reaches the flow as the permuted view ProjGrid.forward ends with, multi view as a contiguous
    tensor. Anything the store gives back is contiguous (it writes its files straight from its own memory), so the
    single-view layout is made again here. It is already right for a tensor that never left the card.
    """
    if multi:
        return cond
    proj = cond["proj"]
    return {**cond, "proj": proj.transpose(1, 2).contiguous().transpose(1, 2)}


def _token_cond(cond: dict, coords: torch.Tensor, negative: bool = True, proj_cache: int = 0) -> dict:
    """{"cond", "neg_cond"} of a sparse stage: the stored rows on the prop's coords, and zeros of the same form.
    `proj_cache`: how many of the flow's blocks, the first ones, keep their proj_linear answer across the
    conditional steps (proj_cache_blocks)."""
    pair = {"cond": {"global": cond["global"], "proj": SparseTensor(feats=cond["proj"], coords=coords)},
            "neg_cond": None}
    if proj_cache:
        pair["cond"]["proj_cache"] = int(proj_cache)
    if negative:
        zero = _proj_cond().negative(cond)
        pair["neg_cond"] = {"global": zero["global"], "proj": SparseTensor(feats=zero["proj"], coords=coords),
                            "zero": True}       # the flows' blocks compute the zero context's row once
    return pair


def _denormalised(slat: SparseTensor, normalization: dict) -> SparseTensor:
    std = torch.tensor(normalization["std"])[None].to(slat.device)
    mean = torch.tensor(normalization["mean"])[None].to(slat.device)
    return slat * std + mean


def _steps(ctx: PartContext, prop: PropState, part: str, mode: str) -> Callable[[int, int], None]:
    """A sampler's step callback: progress, the next prop's inputs on their way to the card (when memory is
    tight they wait for their prop's turn, and what was sent ahead comes back), and a cancel check point.

    Nothing is sent ahead from inside a part whose convolutions the runner watches: the store refuses an
    upload that does not fit by catching the out-of-memory itself, and the runner reads the same counter to
    decide whether cuDNN quietly changed its plan. Only `ss` is both, and it gives up the overlap.
    """
    upload = mode == "normal" and part not in CONV_PARTS

    def callback(step: int, total: int) -> None:
        ctx.emit(Event("step", part, prop.id, step, total))
        ctx.store.pump(upload=upload)
        ctx.check()

    return callback


# =====================================================================================================================
# The parts
# =====================================================================================================================

def cutout(ctx: PartContext, prop: PropState, mode: str) -> None:
    def matte(image: Any) -> Any:
        return _network(ctx, "rembg")(image)

    prepared = ctx.hooks.prepare(prop.req, matte)
    if not prepared.images:
        raise PropFailed("The picture could not be prepared.", "input")
    for index, image in enumerate(prepared.images):
        ctx.store.put_small(prop.id, f"image_{index}", image)
    if geometry(prop) == "hi3d":
        front = prepared.cutout if prepared.cutout is not None else prepared.images[0]
        if getattr(front, "mode", "") != "RGBA":
            raise PropFailed("The picture's cut-out has no transparency for Hi3D to frame it by.", "input")
        ctx.store.put_small(prop.id, "cutout", front)
    ctx.store.put_small(prop.id, "prep", {"meta": prepared.meta, "camera": None})
    prop.views = len(prepared.images)


def camera(ctx: PartContext, prop: PropState, mode: str) -> None:
    store = ctx.store
    found = ctx.hooks.camera(prop.req, store.get_small(prop.id, "image_0"), _network(ctx, "moge"))
    store.put_small(prop.id, "prep", {**store.get_small(prop.id, "prep"), "camera": dict(found)})


def features(ctx: PartContext, prop: PropState, mode: str) -> None:
    proj_cond = _proj_cond()
    dino = _network(ctx, "dino")
    source, view = _source(ctx, prop)
    z512 = proj_cond.encode(dino, proj_cond.image_tensor(source, 512, ctx.device))
    z1024 = proj_cond.encode(dino, proj_cond.image_tensor(source, 1024, ctx.device))
    cond = _torso(proj_cond.cond_dense(proj_cond.SPECS["ss"], z512, view))
    if geometry(prop) != "hi3d":            # cond_lr's: a Hi3D prop has no cond_lr
        ctx.store.put_device(prop.id, "dino512", {"z": z512})
    ctx.store.put_device(prop.id, "dino1024", {"z": z1024})
    ctx.store.put_device(prop.id, "cond_ss", cond)
    _keep_bundle(ctx, prop, source)


def draw_sparse_structure(ctx: PartContext, prop: PropState, mode: str, cond: dict, resolution: int,
                          num_samples: int = 1, sampler_params: dict | None = None,
                          draws: list | None = None) -> torch.Tensor:
    """One draw of the sparse structure: coords [N0,4] int32 on the card. What engine.sample_sparse_structure
    runs while part `ss` works on `prop`; a guard around that attribute may call it several times, and every
    call draws new noise from the CPU generator. `draws` collects (the coords returned, the occupancy at the
    decoder's own grid before the pooling) for a Hi3D prop."""
    args = ctx.args(prop)
    flow = _network(ctx, f"ss_flow.{prop.kind}")
    reso = flow.resolution
    noise = torch.randn(num_samples, flow.in_channels, reso, reso, reso).to(ctx.device)
    params = {**args.sampler_params["ss"], **(sampler_params or {})}
    z_s = args.samplers["ss"].sample(
        flow, noise, **cond, **params, verbose=False, keep_trajectory=False,
        step_callback=_steps(ctx, prop, "ss", mode),
    ).samples
    decoded = _network(ctx, "ss_dec")(z_s) > 0
    full = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int() if draws is not None else None
    if resolution != decoded.shape[2]:
        ratio = decoded.shape[2] // resolution
        decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
    coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
    if draws is not None:
        draws.append((coords, full))
    return coords


def ss(ctx: PartContext, prop: PropState, mode: str) -> None:
    store, request = ctx.store, prop.req
    cond = _as_reference_layout(ctx.take(prop, "cond_ss"), prop.req.multi)
    # The dense flow's blocks keep their proj_linear answers like the sparse flows' (one [4096, 1536] table a
    # block, 377 MB in bf16 for all 30): the guard's draws all take this conditioning, so they share them.
    flow = _network(ctx, f"ss_flow.{prop.kind}")
    cached = proj_cache_blocks(ctx, flow, int(getattr(flow, "resolution", 0)) ** 3, "ss", mode)
    pair = {"cond": {**cond, "proj_cache": cached} if cached else cond, "neg_cond": _proj_cond().negative(cond)}
    draws = [] if geometry(prop) == "hi3d" else None
    draw = functools.partial(draw_sparse_structure, ctx, prop, mode, draws=draws)
    torch.manual_seed(request.seed)
    try:
        with ctx.hooks.prop_scope(request, "ss") as found, ctx.engine.drawing(draw):
            coords = ctx.engine.sample_sparse_structure(pair, 32, 1, request.sampler_params.get("ss", {}))
    finally:
        _drop_proj_cache(ctx, flow, "ss", prop, cached)
    rng = torch.get_rng_state()
    if coords.shape[0] == 0:
        raise PropFailed("The first stage found no shape in the picture.")
    if draws is not None:       # the draw the guard returned (the object itself; a copy of it by its values)
        full = next((occupied for returned, occupied in draws if returned is coords),
                    next((occupied for returned, occupied in draws if torch.equal(returned, coords)), None))
        if full is None:
            raise RuntimeError("the sparse structure returned is none of the draws made")
        store.put_small(prop.id, "hi3d_ss64", {"coords": full[:, 1:].cpu()})
    store.put_small(prop.id, "coords_ss", {"coords": coords.cpu()})
    store.put_small(prop.id, "rng_ss", {"state": rng})
    store.put_small(prop.id, "notes", {**store.get_small(prop.id, "notes"), **(found or {})})


def cond_lr(ctx: PartContext, prop: PropState, mode: str) -> None:
    proj_cond = _proj_cond()
    naf = _network(ctx, "naf")
    source, view = _source(ctx, prop)
    coords = ctx.store.get_small(prop.id, "coords_ss")["coords"].to(ctx.device)
    z512 = ctx.take(prop, "dino512")["z"]
    image = proj_cond.image_tensor(source, 512, ctx.device)
    cond = _torso(proj_cond.cond_tokens(proj_cond.SPECS["shape_512"], z512, image, naf, view, coords, 32))
    ctx.store.put_device(prop.id, "cond_lr", cond)
    _keep_bundle(ctx, prop, source)


def _shape_latent(ctx: PartContext, prop: PropState, mode: str, part: str, coords_key: str, rng_key: str,
                  slat_key: str) -> None:
    """A shape stage (`lr` at the 512 grid, `hr` at the 1024 grid): noise at the prop's tokens from the stored
    generator state, the sampler, the latent denormalised."""
    store, device, args = ctx.store, ctx.device, ctx.args(prop)
    flow = _network(ctx, f"{part}_flow.{prop.kind}")
    coords = store.get_small(prop.id, coords_key)["coords"].to(device)
    cached = proj_cache_blocks(ctx, flow, coords.shape[0], part, mode)
    pair = _token_cond(ctx.take(prop, f"cond_{part}"), coords, proj_cache=cached)
    params = {**args.sampler_params["shape"], **prop.req.sampler_params.get("shape", {})}
    torch.set_rng_state(store.get_small(prop.id, rng_key)["state"])
    noise = SparseTensor(feats=torch.randn(coords.shape[0], flow.in_channels).to(device), coords=coords)
    try:
        slat = args.samplers["shape"].sample(
            flow, noise, **pair, **params, verbose=False, keep_trajectory=False,
            step_callback=_steps(ctx, prop, part, mode),
        ).samples
    finally:
        _drop_proj_cache(ctx, flow, part, prop, cached)
    slat = _denormalised(slat, args.shape_normalization)
    rng = torch.get_rng_state()
    store.put_small(prop.id, slat_key, {"feats": slat.feats.cpu()})
    store.put_small(prop.id, f"rng_{part}", {"state": rng})


def lr(ctx: PartContext, prop: PropState, mode: str) -> None:
    _shape_latent(ctx, prop, mode, "lr", "coords_ss", "rng_ss", "lr_slat")


def upsample(ctx: PartContext, prop: PropState, mode: str) -> None:
    store, device, request = ctx.store, ctx.device, prop.req
    feats = store.get_small(prop.id, "lr_slat")["feats"].to(device)
    coords = store.get_small(prop.id, "coords_ss")["coords"].to(device)
    hr_coords = _decode().upsample(_decoder(ctx, "shape_dec"), feats, coords)
    occupied = hr_coords.shape[0]

    def tokens_at(resolution: int) -> torch.Tensor:
        grid_res = resolution // 16
        quant_coords = torch.cat([
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / 512 * (grid_res - 1)).round().int(),
        ], dim=1)
        return quant_coords.unique(dim=0)

    # The token cap of p2d:711-725: a 1536 prop with too many tokens is made at a lower resolution. The memory
    # this PC leaves Pixal3D caps it as well (resolution_fits): a prop that would not fit the card or the RAM at
    # a resolution is made at the next lower one, down to 1024, instead of spilling or failing at the end. With
    # the decode in boxes (flex_gemm; spconv where its whole-table decode would not fit: chunk_for), what can still
    # lower a 1536 prop on an 8 GB card is a sampler at a bulky prop's token count (sampler_mib: an fp32 torso, a
    # GTX 10xx), the closing of the holes of a shape far bulkier than the test set's (fill_mib), the finish hook's
    # steps, or the RAM; without boxes, the whole-table decode with the backend's figures, and below
    # WHOLE_LEVEL_1536_MIB 1024.
    resolution, unique = request.resolution, None
    unbounded, why = None, None            # the resolution the token cap alone gives, and what memory said against it
    while True:
        unique = tokens_at(resolution)
        under_cap = unique.shape[0] < request.max_num_tokens
        if unbounded is None and (under_cap or resolution == 1024):
            unbounded = resolution
        reason = resolution_fits(ctx, occupied, unique.shape[0], resolution, req=request) if under_cap else None
        why = why or reason
        if (under_cap and reason is None) or resolution == 1024:
            break
        resolution -= 128
    bound_by = why if resolution != unbounded else None
    stored = {"coords": unique.cpu(), "res": torch.tensor([resolution], dtype=torch.int32),
              "voxels": torch.tensor([voxels_at(occupied, resolution)], dtype=torch.int64)}
    store.put_small(prop.id, "coords_hr", stored)
    if bound_by is not None:
        notes = store.get_small(prop.id, "notes")
        store.put_small(prop.id, "notes", {**notes, "memory_cap": {
            "by": bound_by, "resolution": resolution, "without": unbounded, "tokens": unique.shape[0]}})


def cond_hr(ctx: PartContext, prop: PropState, mode: str) -> None:
    proj_cond = _proj_cond()
    naf = _network(ctx, "naf")
    source, view = _source(ctx, prop)
    stored = ctx.store.get_small(prop.id, "coords_hr")
    coords = stored["coords"].to(ctx.device)
    grid = int(stored["res"][0]) // 16
    z1024 = ctx.take(prop, "dino1024")["z"]
    image = proj_cond.image_tensor(source, 1024, ctx.device)
    if geometry(prop) == "hi3d":
        tex = _torso(proj_cond.cond_tokens(proj_cond.SPECS["tex_1024"], z1024, image, naf, view, coords, grid))
        ctx.store.put_device(prop.id, "cond_tex", tex)
        return
    together = mode == "normal"
    pair_fn = getattr(proj_cond, "cond_tokens_pair", None)      # a conditioning module without it: the two calls
    pair = None
    if together and pair_fn is not None:
        try:
            pair = pair_fn(proj_cond.SPECS["shape_1024"], proj_cond.SPECS["tex_1024"], z1024, image, naf, view,
                           coords, grid, first_dtype=precision.torso())
        except ValueError:          # nothing to share for this NAF or picture: the two calls, as before
            pair = None
    if pair is not None:
        shape, tex = _torso(pair[0]), _torso(pair[1])
    else:
        shape = _torso(proj_cond.cond_tokens(proj_cond.SPECS["shape_1024"], z1024, image, naf, view, coords, grid))
        ctx.loader.trim()       # the texture conditioning is the widest moment of a prop: it starts from a clean cache
        tex = _torso(proj_cond.cond_tokens(proj_cond.SPECS["tex_1024"], z1024, image, naf, view, coords, grid))
    ctx.store.put_device(prop.id, "cond_hr", shape)
    ctx.store.put_device(prop.id, "cond_tex", tex)


def hr(ctx: PartContext, prop: PropState, mode: str) -> None:
    _shape_latent(ctx, prop, mode, "hr", "coords_hr", "rng_lr", "shape_slat")


def tex(ctx: PartContext, prop: PropState, mode: str) -> None:
    store, device, args, request = ctx.store, ctx.device, ctx.args(prop), prop.req
    flow = _network(ctx, f"tex_flow.{prop.kind}")
    stored = store.get_small(prop.id, "coords_hr")
    shape_feats = store.get_small(prop.id, "shape_slat")["feats"]
    coords = stored["coords"].to(device)
    params = {**args.sampler_params["tex"], **request.sampler_params.get("tex", {})}
    # A guidance strength of 1 runs only the positive pass: the zero conditioning would never be read.
    cached = proj_cache_blocks(ctx, flow, coords.shape[0], "tex", mode)
    pair = _token_cond(ctx.take(prop, "cond_tex"), coords, negative=params.get("guidance_strength") != 1,
                       proj_cache=cached)

    std = torch.tensor(args.shape_normalization["std"])[None].to(device)
    mean = torch.tensor(args.shape_normalization["mean"])[None].to(device)
    shape_slat = (SparseTensor(feats=shape_feats.to(device), coords=coords) - mean) / std
    torch.set_rng_state(store.get_small(prop.id, "rng_hr")["state"])
    noise = shape_slat.replace(
        feats=torch.randn(shape_slat.coords.shape[0], flow.in_channels - shape_slat.feats.shape[1]).to(device))
    try:
        slat = args.samplers["tex"].sample(
            flow, noise, concat_cond=shape_slat, **pair, **params, verbose=False, keep_trajectory=False,
            step_callback=_steps(ctx, prop, "tex", mode),
        ).samples
    finally:
        _drop_proj_cache(ctx, flow, "tex", prop, cached)
    tex_feats = _denormalised(slat, args.tex_normalization).feats.cpu()

    store.put_small(prop.id, "tex_slat", {"feats": tex_feats})
    if request.request_hash:
        tensors = {"coords": stored["coords"], "res": stored["res"], "shape_slat": shape_feats, "tex_slat": tex_feats}
        noted = prop.noted(store.get_small(prop.id, "notes"))
        if geometry(prop) == "hi3d":
            mesh = store.get_large(prop.id, "hi3d_mesh")
            tensors.update({"hi3d_vertices": mesh["vertices"], "hi3d_faces": mesh["faces"],
                            "voxels": stored["voxels"], **{f"hi3d_{key}": sub for key, sub in
                                                           store.get_small(prop.id, "hi3d_subs").items()}})
        elif stored.get("voxels") is not None:
            # the voxels upsample foresaw, with the checkpoint's notes (runner._restore puts them back into
            # coords_hr): without them a resumed prop's memory rules count VOXELS_PER_TOKEN a token (decode_voxels),
            # 18-38 % above the props measured, and on spconv's whole tables a resumed 1536 tree (43.5k tokens, 17.4 M
            # voxels counted for 12.2 M) was refused at once on a 12 GB card it fits (round O, runs/whole-tur)
            noted = {**noted, CHECKPOINT_VOXELS: int(stored["voxels"][0])}
        store.checkpoint(prop.id, request.request_hash, tensors, noted)


def decoders_fit(voxels: int, limit_mib: int, chunk: int | None = None) -> bool:
    """May both decoders stay on the card while a shape of `voxels` voxels is decoded under a cap of `limit_mib`
    (decode_mib with two decoders, in the decode's boxes `chunk` - by default chunk_for's under `limit_mib`: whole
    tables on a backend without boxes, and on spconv where they fit)."""
    chunk = chunk_for(voxels, limit_mib) if chunk is None else chunk
    return decode_mib(voxels, decoders=2, chunk=chunk) <= limit_mib / 1.1


def _finish_limit(ctx: PartContext) -> int | None:
    """The finish decode's cap as the card stands now (what set_cap would give it): the room the loader reports
    less the finish's reserves; None where the loader cannot say (a CPU run). For the box decision of an uncapped
    rung (chunk_for)."""
    headroom = ctx.loader.headroom_mib() if hasattr(ctx.loader, "headroom_mib") else None
    return None if headroom is None else int(headroom - (NATIVE_RESERVE["finish"] + FINISH_SAFETY_MIB))


def decode_voxels(stored: dict) -> int:
    """The voxels the stored coords_hr decode to: as `upsample` foresaw them (a resumed prop: as its checkpoint
    noted them), or from the token count for a prop resumed from a checkpoint without them."""
    voxels = stored.get("voxels")
    return int(voxels[0]) if voxels is not None else VOXELS_PER_TOKEN * int(stored["coords"].shape[0])


def finish(ctx: PartContext, prop: PropState, mode: str) -> "FinishSteps | None":
    if geometry(prop) == "hi3d":
        return _finish_hi3d(ctx, prop, mode)
    decode = _decode()
    store, loader, device = ctx.store, ctx.loader, ctx.device
    stored = store.get_small(prop.id, "coords_hr")
    resolution = int(stored["res"][0])
    notes = prop.noted(store.get_small(prop.id, "notes"))
    capped = mode != "uncapped"
    voxels = decode_voxels(stored)

    # Phase 1, the decoders. A decoder's weights go onto the card before the latents it decodes. What the pair
    # keeps beside them for the next prop (its texture network: runner._set_aside, pair_keep) is counted out of
    # the cap, which counts what is allocated as room; the decoders themselves are in decode_mib.
    limit = loader.set_cap(NATIVE_RESERVE["finish"], safety_mib=FINISH_SAFETY_MIB) if capped else 0
    room = limit - _kept_beside(ctx) if capped else 0
    # A big shape in boxes (the same mesh, a peak that does not grow with it), decided under the decode's own cap -
    # with the cap lifted, under what the card has left (chunk_for) - and counted so everywhere below. On flex_gemm
    # the hopeless check keeps the decision for the margin-free voxels it always made.
    chunk = chunk_for(voxels, limit if capped else _finish_limit(ctx))
    hopeless = finish_mib(voxels / VOXELS_PER_OCCUPIED, chunk=None if decode_backend() == "flex_gemm" else chunk)
    if capped and hopeless > room:
        # Hopeless under the cap, even with every margin taken off: an attempt would run out of memory after a
        # minute and leave nothing. Said at once; the ladder lifts the cap when the caller allows a spill.
        raise PropFailed(f"Finishing this shape ({voxels / 1e6:.1f} M voxels) needs about "
                         f"{hopeless:,.0f} MiB of video memory; the card has {room:,} MiB free for it.", "oom")
    together = mode == "normal" and decoders_fit(voxels, room, chunk)
    # The lean form (each level's cache given back at once) where the decode needs it, and on the tight rung
    lean = mode != "normal" or lean_needed(voxels, 2 if together else 1, chunk, room)
    ctx.check()
    if not together and loader.get("tex_dec") is not None:      # left by an attempt that failed with both loaded
        loader.drop("tex_dec")
    for name in LAZY["finish"] if together else LAZY["finish"][:1]:
        _decoder(ctx, name, chunk, lean)
    coords = stored["coords"].to(device)
    shape_feats = store.get_small(prop.id, "shape_slat")["feats"].to(device)
    mesh, subs = decode.decode_shape(loader.get("shape_dec"), shape_feats, coords, resolution)
    del shape_feats
    if together:
        loader.trim()
    else:
        loader.drop("shape_dec")
    ctx.emit(Event("step", "finish", prop.id, 1, 3))
    ctx.check()
    _decoder(ctx, "tex_dec", chunk, lean)
    tex_feats = store.get_small(prop.id, "tex_slat")["feats"].to(device)
    tex_coords, attrs = decode.decode_texture(loader.get("tex_dec"), tex_feats, coords, subs)
    del tex_feats, subs, coords
    tex_coords, attrs = tex_coords.cpu(), attrs.cpu()     # off the card before the holes are closed: cumesh's
    gone = [name for name in LAZY["finish"] if loader.get(name) is not None and name not in ctx.hold]
    if gone:                                            # tables need the room (the decoders stay only where the
        loader.drop(*gone)                              # pair found room for them beside the holes: ctx.hold)
    else:
        loader.trim()
    ctx.emit(Event("step", "finish", prop.id, 2, 3))
    decode.fill_holes(mesh)
    ctx.emit(Event("step", "finish", prop.id, 3, 3))

    # Nothing of the prop stays on the card: the hook's own work needs the room, and uploads what it needs.
    decoded = DecodedMesh(vertices=mesh.vertices.cpu(), faces=mesh.faces.cpu(), coords=tex_coords,
                          attrs=attrs, resolution=resolution, attr_layout=ctx.args(prop).pbr_attr_layout)
    del mesh, tex_coords, attrs
    loader.trim()
    return _hand_over(ctx, prop, decoded, notes, capped)


def _hand_over(ctx: PartContext, prop: PropState, decoded: "DecodedMesh", notes: dict,
               capped: bool) -> "FinishSteps | None":
    """Phase 2 of `finish`, the caller's hook. Once it has started the prop is finished and saved, cancel or not.
    The hook may hand the rest back as steps (types.FinishSteps): the runner then runs them around the next props'
    decodes, the CPU ones on its worker thread."""
    ctx.check()
    reserve = glb_reserve(decoded.faces.shape[0])
    if capped:
        ctx.loader.set_cap(reserve, safety_mib=FINISH_SAFETY_MIB)
    prop.resolution = decoded.resolution
    outcome = ctx.hooks.finish(prop.req, decoded, notes)
    if isinstance(outcome, FinishSteps):
        outcome.native_mib = outcome.native_mib or reserve
        outcome.safety_mib = FINISH_SAFETY_MIB if outcome.safety_mib is None else outcome.safety_mib
        return outcome
    prop.result = dict(outcome or {})
    return None


def _finish_hi3d(ctx: PartContext, prop: PropState, mode: str) -> "FinishSteps | None":
    """`finish` for a Hi3D prop: its mesh is Hi3D's, so only the texture is decoded, along the guide tables made
    from the mesh's own voxels (hi3d.guide_subs); the mesh's holes were closed when it was extracted."""
    decode = _decode()
    store, loader, device = ctx.store, ctx.loader, ctx.device
    stored = store.get_small(prop.id, "coords_hr")
    resolution = int(stored["res"][0])
    notes = prop.noted(store.get_small(prop.id, "notes"))
    capped = mode != "uncapped"
    voxels = decode_voxels(stored)
    limit = loader.set_cap(NATIVE_RESERVE["finish"], safety_mib=FINISH_SAFETY_MIB) if capped else 0
    room = limit - _kept_beside(ctx) if capped else 0
    chunk = chunk_for(voxels, limit if capped else _finish_limit(ctx))
    lean = mode != "normal" or (capped and lean_needed(voxels, 1, chunk, room))
    if loader.get("shape_dec") is not None:
        loader.drop("shape_dec")
    ctx.check()
    _decoder(ctx, "tex_dec", chunk, lean)
    coords = stored["coords"].to(device)
    subs = [sub.to(device) for _, sub in sorted(store.get_small(prop.id, "hi3d_subs").items())]
    tex_feats = store.get_small(prop.id, "tex_slat")["feats"].to(device)
    ctx.emit(Event("step", "finish", prop.id, 1, 3))
    tex_coords, attrs = decode.decode_texture(loader.get("tex_dec"), tex_feats, coords, subs)
    del tex_feats, subs, coords
    tex_coords, attrs = tex_coords.cpu(), attrs.cpu()
    gone = [name for name in LAZY["finish"] if loader.get(name) is not None and name not in ctx.hold]
    if gone:
        loader.drop(*gone)
    else:
        loader.trim()
    ctx.emit(Event("step", "finish", prop.id, 2, 3))
    mesh = store.get_large(prop.id, "hi3d_mesh")
    decoded = DecodedMesh(vertices=mesh["vertices"], faces=mesh["faces"], coords=tex_coords, attrs=attrs,
                          resolution=resolution, attr_layout=ctx.args(prop).pbr_attr_layout)
    del tex_coords, attrs
    ctx.emit(Event("step", "finish", prop.id, 3, 3))
    loader.trim()
    return _hand_over(ctx, prop, decoded, notes, capped)


TABLE = {"cutout": cutout, "camera": camera, "features": features, "ss": ss, "cond_lr": cond_lr, "lr": lr,
         "upsample": upsample, "cond_hr": cond_hr, "hr": hr, "tex": tex, "finish": finish, **hi3d.TABLE}
