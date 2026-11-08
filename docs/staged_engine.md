# The staged engine

`pixal3d/staged/` runs the Pixal3D cascade in little memory. The reference pipelines
(`Pixal3DImageTo3DPipeline.run`, `Pixal3DMVImageTo3DPipeline.run_mv`) keep every network in RAM and move one after
the other to the card for each prop. The staged engine makes a whole batch of props **part by part**: every prop
goes through one part before any prop starts the next, and a part's networks are read from disk straight onto the
card when the part starts and dropped when it ends. No network rests in RAM (what a prop carries from part to
part does, within a budget, and spills to the work folder beyond it), and PyTorch's reserve on the card is capped
so that it recycles its cache instead of spilling into shared memory.

The results are those of the reference pipelines for the same picture, seed and settings: same random numbers in
the same order, same arithmetic. `run()` and `run_mv()` stay in the fork as that reference.

## Using it

```python
from pathlib import Path
from pixal3d.staged import StagedEngine, EngineOptions, PropRequest
from pixal3d.staged.builders import default_builders, default_paths

options = EngineOptions(pipeline_dir=snapshot,                       # pipeline.json, pipeline_mv.json, ckpts/
                        builders=default_builders(default_paths(models_root)))
engine = StagedEngine(options, hooks)                                # reads two JSON files; loads nothing
                                                                     # (reads background removal's file ahead)

requests = [PropRequest(id="chair", seed=11, resolution=1024, multi=False,
                        sampler_params={"ss": {...}, "shape": {...}, "tex": {...}})]
results = engine.run(requests, Path(work_dir), on_event=print, cancelled=lambda: False)
# {"chair": {"state": "done", "result": {..., "slow": [], "seconds": 187.2, "resolution": 1024, "resumed": False}}}

engine.release()        # before anything else needs the card (the reference pipeline, another tool)
engine.close()          # at exit: also stops the helper threads
```

- `run` blocks and must be called on the one thread that does all CUDA work of the process. Of the failures a
  batch can meet it raises only `BatchBroken` (the CUDA context is unusable; restart the process): anything else
  is laid on the props that had not finished, so the ones that had keep their results. Being called wrongly still
  raises (`ValueError` for two props with one id, `RuntimeError` for a second batch at the same time). A prop that
  fails does not stop the others:
  its entry is `{"state": "failed", "error": {"kind", "detail"}}` with kind `input`, `mv_missing`, `oom`, `ram`,
  `disk`, `missing_file` or `error`.
- `cancelled()` is asked before every prop of every part, between the tensors of a network being loaded, at every
  sampler step and between the steps of the last part. Unfinished props come back as `{"state": "cancelled"}`. A
  prop whose file is being written is finished first.
- `work_dir` belongs to the batch. The engine empties it at the start, except for the checkpoints `resume` is
  about to read, and removes each prop's folder as that prop ends.
- `wave_size=K` makes K props at a time go through all parts (the first results appear sooner, a crash loses
  less). `None` / `0`: all props of the batch together.
- `resume=True` on a second call with the same `work_dir` (after the process died): props whose sampling was
  finished are only decoded and saved. A prop is recognised by its `id` and `request_hash`, which the caller makes
  from everything that defines the result. Props without a `request_hash` are never resumed.

### Hooks

The engine never touches a prop's files. The caller passes an object with five methods (`types.Hooks`), all called
on the GPU thread; each may raise `PropFailed(detail, kind)` to fail its prop:

| hook | when | gives |
|---|---|---|
| `prepare(req, matte)` | part `cutout` | `Prepared(images, meta)`: single view `[RGB picture]`, cut out and framed; multi-view the framed RGBA views, front first. `matte(rgb picture)` is background removal on the card; its network is loaded the first time it is called |
| `camera(req, image, moge)` | part `camera`, single-view props | `{"camera_angle_x", "distance", "mesh_scale"}` as Python floats |
| `bundle(req, images, meta)` | part `features` (again in `cond_lr` / `cond_hr` only where the RAM had no room to keep it), multi-view props | the views dict of `run_mv` (`images` at 512 and 1024, `camera_angle_x`, `camera_distance`, `transform_matrix`, `mesh_scale`) |
| `prop_scope(req, "ss")` | around the sparse-structure stage | a context manager yielding a dict the hook has filled when the block is left; it becomes part of the prop's notes |
| `finish(req, mesh, notes)` | part `finish` | writes the prop's file from the `DecodedMesh` (four CPU tensors) and returns what the result should carry - or a `FinishSteps` (below) |

`engine.sample_sparse_structure(cond, resolution, num_samples, sampler_params)` is the one attribute a caller may
wrap, exactly as the pipeline's method of that name: part `ss` calls it through the engine, positionally. A wrapper
that calls it again draws new noise from the prop's generator, and the state stored for the next stage is the one
after all draws.

#### Finishing in steps

Most of what a finish hook does after the decoders (the UV unwrap, the texture clean-up, writing the file) is CPU
work that leaves the card idle. Instead of a result, `finish` may return `types.FinishSteps(steps, state, result)`:
`steps` is a list of `("gpu" | "cpu", fn)`, each `fn(state)` reading and writing the dict `state`; `result(state)`
gives the prop's result once every step ran. The runner runs the `"gpu"` steps on the GPU thread, between the
other props' turns, under the memory cap (`native_mib`, by default the engine's own figure for a mesh of this
size, and `safety_mib`); the `"cpu"` steps run on `runner.WORKERS` (2) worker threads meanwhile, so the next prop is
decoded while this one is unwrapped, and one prop's export does not wait behind another's BVH build. Rules: a GPU step may run twice after an out-of-memory (it must compute from its inputs and
may overwrite its outputs) and leaves nothing on the card when it returns; a CPU step touches neither the card
nor the engine (no events, no store). An exception in a step fails the prop as the hook's own would; a cancel does
not stop a prop that has reached its steps. At most `runner.IN_FLIGHT` (3) props are in their steps at once, which
bounds the RAM their meshes and textures hold; a prop in its last step (the hub's export, which holds little beside
the file it writes) no longer counts. The hook's steps are reported as `step` events `4..3+n` of `3+n`;
`prop_done` comes when the last step is over, so a prop's turn on the card and its `prop_done` are no longer back
to back. Two more rules of the runner keep the RAM and the tail down (round G): a tail that has run no step yet and
begins with a CPU step is handed to the worker after the GPU steps of the other tails at that boundary, not before
them, while props are still to be decoded (what its first step makes - the hub's cuBVH tree, a gigabyte for a 1536
prop - is needed one texture later, and the GPU steps that run now let go of an older prop's: two trees in RAM at
that moment, not three; the last prop's tail is not deferred); and the props of a paired group go through the
texture and the finish biggest shape first (`parts.pair_order`, by the voxels `upsample` foresaw), so that the last
prop's steps, which nothing hides, are the smallest prop's. After the wave's last decode the tails' GPU steps run
one at a time, the tail with the most steps left first (`runner._run_longest_first`: a geometry, which hands the
worker an unwrap, a bake and an export after it, before a bake, which hands it only an export), the CPU steps that
ended collected and the next ones handed over between two of them; before that the tails keep their own order (a
bake lets go of its prop's RAM). A prop's numbers do not depend on its place or on when its steps run: every part
draws from the generator state its own prop stored.

One thing a step may leave on the card: what the engine has room for. Before every step the runner writes
`state["room_mib"]`, the MiB the step may keep for a later step of the same prop (`parts.tail_room`: what the card
has beside PyTorch's reserve, less the finish part's reserves, less what the props still to be decoded need under
the cap at most - both decoders with the margin of `decoders_fit` - or what another prop's hook steps need with
nothing held (`parts.HOLD_BESIDE_MIB`), whichever is more, less what the other props in their steps hold already
and a slack; and no more than the RAM allows twice over beside the floor, for things that keep a copy in RAM). A
step that keeps something writes the MiB it holds into `state["card_mib"]` - a CPU step as soon as it starts
taking it, and until a step of the prop has held something the runner counts `FinishSteps.holds_mib`, the most a
step takes, from the offer of a CPU step until that step is over, since the step may not have written the key yet;
once a step has held, the key alone counts (a later CPU step that says nothing holds nothing); the step that uses
the hold up removes the key. A CPU step may touch the card for such a hold and for nothing else. Every `step`
event says what room the step was offered and what the prop holds now (`detail` room_mib, card_mib). `FinishSteps.release(state)` lets go of the hold at once: the runner calls
it on every tail when a decode runs out of memory, before the retry. The runner hands every tail's CPU step to
the worker before it runs any GPU step, so the worker never waits behind another prop's GPU step, and up to
`runner.IN_FLIGHT` (3) props are in their steps at once. Spellcast3D's GLB steps hold the cuBVH of the decoded
mesh this way: its CPU step builds it on the worker thread (cuBVH builds its tree on the CPU for 5-7 s; the
Spellcast3D build of cumesh lets go of Python's lock meanwhile) while the next prop is decoded, the remesh and
the bake both use it (60-70 bytes a face, 0.6-0.9 GB on the card and in RAM for the meshes of a 1024 prop), and
where the card has no room the GPU thread builds it twice as before.

### Events

`on_event(Event)` is called on the GPU thread and must not block.

| kind | meaning |
|---|---|
| `part_start` / `part_end` | a part begins / is over for the wave; `detail["props"]` lists the props in run order, `detail["wave"]` / `["waves"]` the wave |
| `prop_start` | a prop goes onto the card for this part; `detail["position"]` of `detail["of"]` |
| `step` | `step` of `total`: a sampler step, or one of the three steps of `finish` (shape decoded, texture decoded, holes filled), then the hook's own steps when it finishes in steps |
| `prop_rest` | the prop is done with this part and waits for the next |
| `prop_done` | the prop's file is written; `detail["result"]`, `detail["seconds"]` (the GPU thread's seconds on the prop; the worker's are not counted) |
| `prop_failed` | `detail["kind"]`, `detail["detail"]` |
| `prop_spilled` | the memory cap was lifted for this prop in this part (see the ladder) |
| `prop_resumed` | the prop's latents came from a checkpoint |
| `net_load`, `net_ready`, `net_drop`, `store_warning` | the loader's and the store's own events, passed on with their fields in `detail` |

## The parts

| part | networks on the card | reads | writes |
|---|---|---|---|
| `cutout` | background removal, when first asked for | - | `image_<i>`, `prep` |
| `camera` (single view) | MoGe | `image_0` | `prep` (the camera) |
| `features` | DINOv3 | pictures, `prep` | `dino512`, `dino1024`, `cond_ss` |
| `ss` | sparse-structure flow + its decoder | `cond_ss` | `coords_ss`, `rng_ss`, `notes` |
| `cond_lr` | NAF | `dino512`, pictures, `prep`, `coords_ss` | `cond_lr` |
| `lr` | shape flow 512 | `cond_lr`, `coords_ss`, `rng_ss` | `lr_slat`, `rng_lr` |
| `upsample` | shape decoder | `lr_slat`, `coords_ss` | `coords_hr` (with the resolution used) |
| `cond_hr` | NAF | `dino1024`, pictures, `prep`, `coords_hr` | `cond_hr`, `cond_tex` (both at once, `proj_cond.cond_tokens_pair`: NAF's image encoder once per view, every view added into the two sums in place, a head's values through the attention 128 channels at a time at 1024 px; the two calls are the ladder's tight rung) |
| `hr` | shape flow 1024 | `cond_hr`, `coords_hr`, `rng_lr` | `shape_slat`, `rng_hr` |
| `tex` | texture flow | `cond_tex`, `coords_hr`, `shape_slat`, `rng_hr` | `tex_slat`, the checkpoint |
| `finish` | shape decoder, texture decoder | `coords_hr`, `shape_slat`, `tex_slat`, `notes` | the prop's file, through the hook |

The two last parts go prop by prop (`EngineOptions.paired_finish`, on by default): a prop is decoded and handed
to its finish as soon as its texture is sampled, instead of after every prop's texture. The CPU work of its finish
steps (the BVH, the UV unwrap, the file) then runs on the worker threads while the next prop's texture is sampled -
a minute and more at 1536 - rather than only while the next prop is decoded, and only the last prop's steps are left
with nothing to hide under. What the store sent ahead for the next prop's texture goes back to its slot for the
decode. Where the card has room (`parts.pair_keep`, below), the texture network stays on the card through a prop's
decode and its finish steps when another prop of the group follows, and both decoders stay from that prop's finish
to the next one's when there is room for them as well: neither is read again for the next prop. Elsewhere - a card
of 8 GB, the last prop of a group - the texture network leaves the card before the decode (the decode needs the
room, and the cap `finish` sets counts what is allocated as room) and is read again for the next prop, from the
file cache in most cases (its file is read ahead again meanwhile), and the decoders are loaded for every prop, as
before. Whatever is kept for the next prop leaves the card at any failure, before the retry: the ladder's next rung
finds the card as it would have without it (and the texture network's file is read ahead again, as when it leaves
before a decode: from the disk it takes 1.1-1.8 s, from the file cache 0.4). Props resumed from a checkpoint, which run `finish` alone, go
first in their kind. The events are those of the two parts with overlapping brackets: `tex` opens before its
first prop, `finish` before its first prop, `tex` closes after its last prop, `finish` after the last prop's steps
are over. A process that dies in a prop's finish leaves that prop's checkpoint alone: the props after it have not
sampled their texture yet. With `paired_finish=False` every prop's texture is sampled, then every prop is finished,
as the other parts go.

The flow networks exist once per kind (`<name>.sv` from `pipeline.json`, `<name>.mv` from `pipeline_mv.json`);
inside every part the multi-view props go first, then the networks are exchanged and the single-view props follow.
The decoders, DINOv3, NAF and background removal serve both kinds.

The flows' transformer blocks run their elementwise chains (LayerNorm32 and the adaLN modulation, the gated
residuals, the q / k RMS norm with the rotary embedding) as fused Triton kernels (`pixal3d/modules/fused.py`), on
by default: `PIXAL3D_FUSED=off` in the environment gives the eager PyTorch ops, bit for bit the numbers of before.
The kernels reproduce every bf16 rounding of the eager chain; only its reductions are summed in another order, and
the samplers amplify that - as they amplify any change of card or cuBLAS - into a slightly different sample of the
same quality. A detail-flow pass is 11-14 % faster, a sparse-structure pass 21 %, a batch 8-10 %; the memory peak
of a pass is unchanged. Before the first fused pass of a process the kernels are tried against the eager ops on
small tables; a card or a Triton that cannot run them turns the switch off with a `[FUSED]` warning and the flows
run eagerly.

Four more savings in the flows' blocks change no bit (each has a switch whose False runs the code before it, the same
numbers). A norm followed by an int8 layer of the Triton GEMMs writes that layer's int8 input itself, instead of a
bf16 table the layer reads back to quantise (`modules/sparse/transformer/modulated.py` `NORM_QUANT`,
`int8mm.norm_quant`: norm_modulate's arithmetic and quantize_rows' in one kernel, at norm_modulate's 8 warps - at
Triton's default 4 a few codes come out otherwise), at the three hand-offs of the sparse blocks' fused path (sm_80+):
0.06-0.65 ms a hand-off. The attentions' to_out writes the new residual stream from its epilogue (`RESID_EPILOGUE`,
`int8mm.gemm(..., resid=)`: the residual kernel's arithmetic with its gate, its second term or a plain add, without
the bf16 table of to_out's results): to_out 2.06 -> 1.62 ms at 21.5k tokens; fc2's epilogue gained nothing, so fc2
keeps the residual kernel. The guided pair's second pass starts its first block from the stream the first pass kept
after that block's self-attention half, which reads no context (`classifier_free_guidance_mixin.SHARE_FIRST_BLOCK`,
`_self_half_shared` in both block files, the block's `cfg_pair`): one [tokens, 1536] table held through the
conditional pass, 11 MiB more of hr's peak for the boy at 1536; it runs on the eager path below sm_80 as well, the
same bits by construction (not run on such a card). And a zero pass adds its kept row without expanding it into a
table first (`fused.ROW_RESIDUAL`, `fused.residual_row`): 0.27 / 0.42 / 0.77 ms a block at 8k / 21.5k / 41.6k tokens.
With the fc1 tile (below), on an RTX 3060 for the boy's three views at 1536 (19,928 tokens): an hr conditional pass
1,992 -> 1,920 ms (-3.6 %), a zero pass 1,874 -> 1,756 ms (-6.3 %), tex -3.6 %, the prop 100.2 -> 97.5 s; every flow
pass's output and the decoded meshes equal to the code before (76 of 76 passes, and 152 of 152 on two more props).
The new kernels compile at their first use (about 3 s once per Triton cache); the install's calibration compiles them
ahead (`int8mm.warm_handoffs`).

The full self-attention of one prop's tokens (the sparse flows: lr, hr, tex; the dense sparse-structure flow's 4096
tokens) runs SageAttention (`pixal3d/modules/sage.py`: int8 Q.K^T, fp16 P.V with fp32 accumulation, the wheel in
`whl/`) on sm_80+ cards, on by default after a first-use self-check: `PIXAL3D_SAGE=off` gives flash_attn, bit for
bit the numbers of before. flash_attn and cuBLAS already run at the card's bf16 tensor peak, so this cheaper number
format is what moves the samplers: an hr or tex pass at 1536 is 26 % faster, a batch about 20 %, the memory peak of
a pass unchanged. Its error at the attention output is about 4x the bf16 rounding of flash_attn's output, which the
samplers amplify into a different sample of the same quality (Chamfer distance 0.17-0.24 % of the bounding-box
diagonal on the five test props, the class of the fused kernels' own change). The cross-attention (5 context tokens)
and every other call keep their backend.

The detail flows' blocks (hr and tex) run that kernel's int8 P.V twin where the card is sm_8x and the wheel has it
(post6 on; round K): the same q / k quantisation, Q.K^T and online softmax, then P as 8-bit integers per row and
64-key block (relative to the block's own maximum) times v as int8 per 64-key block and channel (its mean per
channel subtracted and added back after the normalisation) on the int8 tensor cores, which halves the tensor-core
work of the P.V half: a call at 21.5k tokens 50.4 -> 40.4 ms on an RTX 3060 (-18 to -20 % at every length), an
hr or tex pass about 12 % faster, the five test props' batch 660 -> 597 s. It rounds differently from the fp16
kernel - judged first by an emulation on captured passes (1.02-1.42x the fp16 kernel's error against fp32
attention on every block but the hr flow's block 0, where the attention is nearly uniform and an integer code
with one step loses the small weights: 7-13x), then end to end: the five props' meshes 0.03-0.06 % of the
bounding-box diagonal from the fp16 kernel's (sampling floors 0.02-0.04 %; the int8 GEMMs' accepted change
0.03-0.08 %, SageAttention's own 0.17-0.24 %), the base colour at matched surface points 3-12 levels (the int8
GEMMs' 3-11, SageAttention's 13-31), renders alike. So the hr flow's block 0 keeps the fp16 kernel
(`sage.INT8PV_SKIP`), ss and lr keep it everywhere (`sage.INT8PV_FLOWS`; a small share of their passes, and the
samples that flip the most), and the engine marks each block's self-attention module once a flow is loaded
(`sage.mark_int8pv`, with the fold). The choice per card is a timing: the install's calibration (or, without a
record of this wheel, the first use) races the whole int8 call against the fp16 one and keeps it only where it is
5 % faster at every size (`sage._race_int8pv`, `INT8PV_MARGIN`; the record's "sage" section carries "int8pv"),
so no card gets slower; `PIXAL3D_INT8PV=off` keeps the fp16 kernel for every block, `force` skips the race.
The wheel's Turing extension (post7, `_qattn_sm75`) carries the same twin on Turing's mma shapes with its own v
quantiser, wired the same way (the self-check, the marked blocks, the race - the install's trial records it in the
"sage" section, `_turing_int8pv_trial`), but on sm_7x an unset switch keeps the fp16 kernel
(`sage.INT8PV_TURING_DEFAULT = False`): it serves only under `PIXAL3D_INT8PV=on` (where the race found it 5 %
faster) or `force`, until the user decides on its end-to-end judgement on an RTX 2070.
The wheel's Pascal extension (post8, `_qattn_sm61`) and pixal3d_kernels 0.2.1 carry twins of the Pascal attention
kernel (`sage.PASCAL_TWINS`) and of the dp4a GEMM (`int8mm.DP4A_TWINS`, raced in the calibration's `old_cards` step):
the shipped kernels' bits with other code shapes (main loops under Pascal's ~8 KB instruction cache), timed once at
install on the card itself and kept only where 5 % faster at every size, after a bit check at first use in each
process; without a record the shipped kernels serve (`PIXAL3D_PASCAL_TWIN`, `PIXAL3D_DP4A_TWIN` for developers). No
GTX 10xx has timed them yet.

Two savings around the attention change no bit. A guided sampling runs the flow once on the prop's conditioning
and once on an all-zero conditioning per step inside the guidance interval (hr and lr: 9 of 21 passes, ss: 10 of
22). On the zero one, every block's cross-attention branch (norm2, to_q, the q/k norms, the attention over the five
zero context tokens, to_out, proj_linear) answers the same row for every token, and that row depends on the weights
alone: to_kv of zeros is its bias, so the five keys and values are identical, the softmax is exactly uniform and
the attention returns the value row whatever the query; to_out of identical rows gives identical rows; proj_linear
of zeros is its bias. The engine marks the zero conditioning (`"zero": True` in neg_cond, `parts._token_cond` and
`proj_cond.negative`); the block runs the branch in full on the first zero pass of a token count and keeps its first
row (`SparseProjectAttention._zero_rows`, keyed by the token count, since cuBLAS may pick another kernel for
another count), and the next zero passes of that count add the kept row instead: a zero hr pass at 1536 4.72 ->
4.24 s, a whole hr sampling 85.5 -> 82.2 s, ss 11.0 -> 10.4, lr 5.4 -> 5.1 (about 21 s of a five-prop batch); the
golden engine check and the fused + SageAttention samplings are bit-equal with and without it
(`code/devtools/pixal3d-lowmem/round-i/zero/`). And SageAttention's scaling of v by 2^-8 into the kernel and of
the answer by 2^8 after it is folded into the weights when the switch is on (`sage.fold`, called by the engine's
`_prepare_flow` once a flow is loaded: to_qkv's v rows and bias times 2^-8, to_out's weight times 2^8; powers of
two scale every product and sum exactly, so flash_attn, SDPA and the int8 layers give the same bits too): the
kernel is called without its two elementwise passes (`sage.attention(..., scaled=True)`).

A third saving of the same kind (round J; per block, and in part ss too, since 2026-11-08): on the conditional
passes, proj_linear's input is the projected conditioning, the same tensor at every step of a sampling, so each
block's answer is the same table every pass (one GEMM of one shape on one input gives the same bits each time). The
first K blocks of a flow keep theirs, K as many as the cap in force has room for beside the sampler's own need
(`parts.proj_cache_blocks` / `proj_cache_count`: `proj_cache_sampler_mib(tokens)` + 1.1 x each block's table - one
[tokens, 1536] table a block, 58 MiB at 20k tokens in bf16 - with `PROJ_CACHE_MARGIN_MIB` 512 to spare).
`proj_cache_sampler_mib` is `sampler_mib`'s figure, with its per-token part twice as wide for an fp32 torso (the
pascal profile: its samplings allocate 86-89 MiB per thousand tokens beyond the weights and the tables, the boy
forced on an RTX 3060, against 39-45 in bf16; turing's fp16 torso with its fp32 stream measured 52, within the 62.5
sampler_mib charges). In bf16 sampler_mib is itself above a sampling's peak: on a 7.2 GiB card (cap 6.2-6.3 GB) hr
allocated 2,455 / 2,737 MiB without tables for the boy / ork (sampler_mib 3,192 / 3,853), and with all 30 of the
ork's tables 5,451 MiB, its reserve filling the cap with one allocator retry and no failure. So all 30 blocks on the
12 GB card at 1536 and for the smaller 1536 props on a 7-8 GB card, 18-19 of the ork's there (the earlier rule, all
30 or none with 1,536 MiB to spare, kept none for the boy or the ork there); by the figures 14 of the ork's fp32
tables on a GTX 1080 Ti (not run on such a card); none on the ladder's tight rung (an attempt that ran out of memory
runs again without them), on the CPU or when the cap was lifted. The engine marks the conditioning with the count
(`"proj_cache": K`, `parts._token_cond`; True would mean every block), the flows hand it to their first K blocks
(`proj_attention.proj_cache_limit`; none in a zero pass), and a block keeps its answer with its input's identity
(storage address, shape, version: `SparseProjectAttention.proj_feats`, `ProjectAttention.proj_feats`) and gives it
again while the input is that tensor; the engine drops the tables when the sampling attempt ends
(`parts._drop_proj_cache` -> `drop_proj_caches`, in a `finally`: after an out-of-memory, a cancel or a failure too).
Part `ss` does the same for the dense flow (30 tables of [4096, 1536]: 360 MiB in bf16, 720 MiB in fp32 on Pascal),
across the guard's draws, which all take the same conditioning: -2.4 % a step (0.82 -> 0.80 s), about 0.2 s a prop on
the RTX 3060. One GEMM per block cheaper on every conditional pass after the first (all 30 blocks: 157 ms of a 3.1 s
detail pass at 21.5k tokens; on the 7.2 GiB card the boy's hr -3.6 % and tex -6.2 % a step, the ork's with 18-19
blocks -2.0 % and -3.2 %); bit-identical (the sv1024 golden engine check with the tables kept; the boy's and the
ork's decoded meshes equal with 0, 18-19 and 30 blocks and to the full card's). The fakes check the count, its marks
on the conditioning, the drop after every attempt, the flows' first-K rule and the modules' keep and drop
(check_engine.py --fake, the proj tests).

The detail flows' token-sized linear layers run as int8 GEMMs (`pixal3d/modules/int8.py`, the kernels in
`modules/int8mm.py`: Triton GEMMs on the int8 tensor cores with the scaling, bias and GELU fused into the epilogue,
and the quantisers; switch `PIXAL3D_INT8` on/off/a policy name, on by default after a first-use self-check against
fp64, sm_80+). cuBLAS already runs the bf16 GEMMs at the card's tensor peak (26 TFLOPS on an RTX 3060: half of a
detail pass), and the same card multiplies int8 at 84-88 TOPS, so the weights are quantised once at load (one scale
per output channel) and the activations at every call (one scale per token, or per token and 128 input channels:
"group" scales); the loader's convert hook does it (`register_checkpoint(..., convert=)`, the engine's
`_prepare_flow`, after the Sage fold), or the load reads the weights quantised already, from the flow's int8 copy
("The int8 copies of the detail flows", below). Which flows and layers, and how, is a policy: the default "detail"
converts the hr and tex flows only (group scales on to_qkv and the MLP, which carry the error; per-row on the three
1536 -> 1536 layers; proj_linear in bf16, whose input has two channels ~25x the others; the ss and lr flows in bf16,
whose samples a small change flips the most). Measured: an hr pass at 1536 4.21 -> 3.08 s (-27 %), the five test
props' batch 954 -> 760 s (-20 %), a flow's weights 2.7 -> 1.6 GB (the hr/tex memory peaks 1.1-1.5 GB lower); the
final latent moves ~2x as far as the accepted changes (the fused kernels, SageAttention) but the meshes and textures
do not: Chamfer 0.03-0.08 % of the bounding-box diagonal against the bf16 run (the sampling floor; the accepted
changes 0.16-0.43 %), the base colour at matched surface points 3-11 levels apart (the Sage change 13-31), renders
alike (`code/logs/profiling/pixal3d-lowmem/round-i/int8-detail-full/compare`). Off = the bf16 layers, bit for bit the
numbers of before; "safe" (the three 1536 -> 1536 layers only, -5 % of a pass) and "every" / "all" (every flow) are
policies for experiments. The kernels compile on first use (~6 s with a cold Triton cache, 0.4 s warm; one
compilation serves every token count). fc1's tile (its GELU and group-scaled int8 output) is chosen once per card by
the install's calibration, among tiles that give the same bits - BLOCK_K is the group, BLOCK_N fc2's input group
(`int8mm.calibrate_fc1_tiles`: the bits checked, a candidate kept only where it is 3 % faster at both 21,504 and
8,192 rows; the record's `triton.int8_tiles`, which `int8mm.fc1_group_cfg` reads only for the card and the Triton it
was timed with, else the shipped tile; `FC1_GROUP_CFG` forces one): on an RTX 3060 (64, 128, 128, 8, 3, 4), fc1
13-20 % faster (10.46 -> 8.93 ms at 21.5k rows). A record without that choice on a card whose int8 GEMMs are Triton's
(sm_80+), or with a choice timed under another Triton, has the calibration run its Triton step again, alone
(`calibrate.py --only triton`, which Spellcast3D's installer asks for), the record's other sections standing.

On sm_80+ the group-scaled layers (to_qkv, fc1, fc2) can run pixal3d_kernels' CUDA twin of the Triton group GEMM
(`torch.ops.pixal3d_kernels.int8_gemm_group80`, `csrc/int8_gemm_sm80.cu`): the same arithmetic instruction for
instruction, read from Triton 3.4's own SASS - each 128-K slab's exact int32 sum, the fp32 accumulation in slab
order, the epilogue down to libdevice's tanhf in GELU - so the same bits (checked on 2,840 random tables in every
form, on every fp32 input of GELU, on captured real operands and on 7,536 calls inside real passes). The Triton step
of the install's calibration races its tiles against Triton per layer form at 8,192 and 21,504 rows
(`int8mm.calibrate_group_cuda`, called from `calibrate_fc1_tiles`: every tile's output torch.equal to Triton's, a
tile serving a form only where it is 3 % faster at both; the record's `triton.int8_tiles.group_cuda`, with the
kernels build it was made with); a first-use check per process (`group_cuda_check`) compares the two before it
serves, and a form the race did not choose, or a record of another card, Triton or kernels build, keeps Triton. On
an RTX 3060: to_qkv tile 2 (-3.4 to -8.2 %), fc1 tile 4 (-6.8 to -8.8 %), fc2 on Triton (no tile beat it); the boy's
three views at 1536 hr 39.45 -> 39.06 s, tex 23.28 -> 23.00 s, all 76 flow passes and the decoded meshes equal.
`PIXAL3D_INT8_GEMM = triton | cuda` forces one kernel for every group layer (a bench of Triton's must pass
`kernel="triton"` to `int8mm.gemm`, which otherwise takes the record's choice). A record whose tile choice lacks the
race (or holds one of another kernels build) runs the Triton step again, alone, like a missing tile.

DINOv3 (part `features`) runs under float16 autocast on cards from sm_75 up (`proj_cond.dino_fp16`, switch
`PIXAL3D_DINO_FP16` auto/on/off; a GTX 10xx keeps float32, where fp16 is slow): the features part 3.6x faster
(18.7 -> 9.1 s of a five-prop batch at 1536), the tokens within 7e-4 relative L2 of float32, a different sample of
the same quality (Chamfer 0.16-0.24 % of the bounding-box diagonal on the five test props, the class of the fused
kernels' and SageAttention's changes); off = the float32 tokens, bit for bit the numbers of before. The background
removal (part `cutout`, BiRefNet) runs under float16 autocast on the same cards (`builders.matte_fp16`, switch
`PIXAL3D_MATTE_FP16` auto/on/off): 0.58 -> 0.35 s a view, the 8-bit alpha one or two levels off on a quarter of the
pixels - a cut-out of the same quality, not the same bytes.

A part function is `fn(ctx, prop, mode)` in `parts.py`. It reads from the batch's store, computes, and writes its
results **as its last actions**, so an attempt that failed has changed nothing.

### What is kept between the parts

Everything a prop needs later lies in the batch's `PropStore`, under the prop's id:

| key | content |
|---|---|
| `request`, `notes` | the request's numbers; what the hooks noted (`ss_retries`, `ss_blocked`, ...) and what the engine noted: `memory_cap` when the memory, not the token cap, chose the resolution (`{"by": "card" \| "ram", "resolution", "without", "tokens"}`). The finish hook and the checkpoint get the notes with `slow`: the parts that ran with the cap lifted |
| `image_<i>`, `prep` | the prepared pictures (PIL); `{"meta": ..., "camera": {...}}` |
| `dino512`, `dino1024` | `z` [V,1029,1024] / [V,4101,1024] float32: DINOv3's tokens, computed once per input size |
| `cond_ss` | `global` [1,5,1024], `proj` [1,4096,1024] bfloat16 |
| `cond_lr`, `cond_hr`, `cond_tex` | `global` [1,5,1024], `proj` [tokens,2048] bfloat16 |
| `coords_ss`, `coords_hr` | `coords` [tokens,4] int32 (batch, x, y, z), never re-ordered; `coords_hr` also `res` int32 [1] (the resolution used) and `voxels` int64 [1] (what the decode is foreseen to produce; see the memory rules) |
| `lr_slat`, `shape_slat`, `tex_slat` | `feats` [tokens,32] float32, denormalised |
| `rng_ss`, `rng_lr`, `rng_hr` | `state`: `torch.get_rng_state()` after the part's draws |
| checkpoint | `coords`, `res`, `shape_slat`, `tex_slat` and the notes, on disk: what a restarted batch resumes from |

Conditionings are stored in bfloat16, cast on the card with `.type(torch.bfloat16)`: the cast the denoisers make
of them in every pass. Items are thrown away as soon as their last reader is done (`parts.SPENT`). A multi-view
prop's view bundle (`hooks.bundle`: 47 MiB for three views) is built by `features` and kept as the small item
`bundle` for `cond_lr` and `cond_hr` (spent after `cond_hr`), where the RAM keeps `parts.BUNDLE_RAM_MIB` (3,000)
available beside it; where it does not, each of the three parts builds it again, from the same pictures.

### Random numbers

The reference draws from PyTorch's global CPU generator: once per draw of the sparse structure, then once per
latent. In a batch the props' stages interleave, so the generator state travels with each prop: `ss` seeds it
(`torch.manual_seed(seed)`), and `ss`, `lr`, `hr` store its state after their draws under **their own** key; the
next drawing part restores the stored state first. A part never overwrites the state it started from, so a second
attempt of a part draws the same noise. Noise is always drawn on the CPU with the reference's shapes.

### Memory: the cap and the ladder

Before every prop of every part the runner asks the loader to cap PyTorch's reserve to what is free on the card
now, minus a reserve for native code (`runner.NATIVE_RESERVE`; `finish` sets its own cap twice, once for the
decoders and once for the hook, which needs most of the card for native code). Under the cap an allocation that
does not fit raises instead of spilling. The runner then goes down a ladder, for that prop in that part only:

1. `normal`;
2. `tight`: after the cache was emptied; samplers stop bringing the next prop's inputs to the card ahead of time,
   `finish` holds one decoder at a time;
3. only with `EngineOptions.spill=True`, `uncapped`: the cap is lifted, PyTorch may spill into shared memory as the
   reference does: slow, but done. The part is listed in the result's `slow` and a `prop_spilled` event is sent;
4. then the prop fails with kind `oom`.

By default (`spill=False`) nothing a batch does goes beyond the card's own memory: a prop that does not fit after
the tight attempt fails with `oom` instead of running slowly in shared memory. Only out-of-memory failures are
tried again (`runner.classify`). Parts that run cuDNN convolutions are also treated as failed when PyTorch counted
a failed allocation without raising: cuDNN would have taken another algorithm, which rounds differently.

Three rules keep that failure rare (`parts.resolution_fits`, `parts.finish_mib`, `parts.chunk_for`, with the
figures at the top of `parts.py`, measured on an RTX 3060):

- **A big shape is decoded in boxes.** A shape whose whole-table decode would take more than
  `parts.CHUNK_ABOVE_MIB` (3,000 MiB: every 1536 prop of the test set, no 1024 one) is decoded a box of
  `parts.CHUNK_VOXELS` (2 M) fine voxels at a time (`models/sc_vaes/chunked.py`, `decoder.chunk_voxels`): the
  same vertices and faces, bit for bit, in a peak that no longer grows with the prop - about 0.13 MiB per thousand
  voxels plus 600 MiB for the boxes, 3.4 GB where the whole-table decode of a 16.45 M-voxel shape took 9.2 GB -
  for 4-9 % of the decode's time. What then bounded a big prop on the card was the closing of its holes (cumesh's
  native tables: `parts.fill_mib`), the widest moment of `finish` (`parts.finish_mib`) - about 67 bytes a face
  since round G (cumesh 0.0.1.post2 frees its edge sort's buffers and sizes its tables exactly, `Mesh.fill_holes`
  takes no clones of the mesh; 160 before), so that the decode in boxes is the widest moment again.
  Below sm_80 the decoders run on spconv, and since round O they decode in boxes too (`chunked.backend()`
  "spconv"): spconv's implicit GEMM (Turing) or its products on torch.mm (Pascal) on per-level pair tables
  (`conv_spconv.py`), a box's, a slab's and a row subset's GEMM calls padded to at least the install's floor (the
  calibration's `decode_products`, `modules/sparse/conv/products_floor.py`; no boxes on spconv without a section that
  passed), a whole level's calls the whole-table decode's own - the same meshes bit for bit on the RTX 3060 under
  both forced profiles, unproven on real old cards. `parts.decode_figures()` gives each backend's measured figures.
  `PIXAL3D_SPCONV_BOXES=off` turns spconv's boxes off.
- **The resolution of a 1536 prop follows the memory as well as the token cap.** `upsample` walks the resolutions
  down as the reference does (1536, 1408, ..., 1024) and stops at the first whose token count is under the cap
  AND whose finish (`finish_mib`: the decode, in boxes for a big shape, or the closing of the holes) fits the cap
  `finish` will get, whose samplers fit the cap in force (`sampler_mib`: the flow network's weights - the loader's
  figure once it is loaded, else what the int8 policy and the torso give, `flow_weights_mib` - plus 50 MiB per
  thousand tokens, with a quarter on top; until 2026-11-07 the base was the bf16 network's 2,700 MiB whatever
  the policy, which on a card whose cap is ~4.6 GB made the dragon, knight and ork at 1152-1408 where 1536 fits
  with the int8 networks), whose finish hook's GPU steps fit the cap they get (`steps_torch_mib` of the faces the
  voxels foresee, beside `glb_reserve`: on a card that leaves Pixal3D 5.2 GB the ork at 1536, 21 M faces, was given
  1,089 MiB and its geometry step ran out at about 1,100 - the one rule that lowers a prop there, since 2026-11-07),
  and whose finish hook leaves the RAM a floor; at 1024 it
  stops whatever the figures say. When the memory, not the token cap, chose, the notes carry `memory_cap`. With
  the decode in boxes, every prop of the test set is made at 1536 on a card that leaves Pixal3D 7.2 GiB; what can
  still lower one there is the closing of the holes of a far bulkier shape, or the RAM.
- **A finish that cannot fit under the cap is not attempted.** `finish` fails at once with `oom` (an attempt would
  run out of memory after a minute and leave nothing); with `spill=True` the ladder then runs it with the cap
  lifted, as before. What the pair keeps on the card beside the decode (the texture network) is counted out of
  the cap first.

Two rules use the room a big card leaves, and change nothing where it is short:

- **The paired texture and finish keep their networks on the card where it has room** (`parts.pair_keep`, asked
  by `runner._set_aside` after a prop's texture, when another prop of the group follows). The texture network
  stays through the prop's finish when, with it on the card, the decode with both decoders fits the cap the finish
  will set (with `decoders_fit`'s margin), the closing of the holes fits beside it, and the finish steps' GPU steps
  do within what their cap leaves the card: their native reserve (`glb_reserve` of the faces the voxels foresee -
  `FACES_PER_VOXEL` 2.2 - or that of a prop in its steps: `GLB_RESERVE_BASE_MIB` 1,600 and 60 MiB a million
  faces since round M, when the card's use during the geometry steps less PyTorch's reserve measured 1,650-2,150
  MiB at 12-18 M faces; the 2,300 base before left a 21 M-face prop on a 6 GB card 1,089 MiB of PyTorch while 4 GB
  of the card were free, and its geometry step ran out) and what they reserve through PyTorch, for this prop
  and for every prop in its steps by its own faces (`steps_torch_mib`: the hook's figure, `HOOK_TORCH_MIB` 1,500
  for Spellcast3D's geometry and bake at the default 2048 texture, 1.0-1.4 GB measured on meshes of up to 15 M
  faces and 1,006-1,194 MiB reserved from 14 to 18 M on a card that keeps no tree, and
  `HOOK_TORCH_MIB_PER_MFACE` 120 a million faces beyond `HOOK_TORCH_REF_MFACES` 20: in round I's 7.2 GiB run, with one flat 1,500
  counted, the geometry and the bake of a 21.5 M-face prop and the bake of an 18.3 M-face one ran out of memory
  beside the kept texture network and ran again with it gone; measured in isolation on golden meshes at 2048
  (round J): geometry 1,002 MiB at 14.9 M faces and 2,126 at 35.9 M, the bake 964 and 1,744, so the rule's
  4,008 MiB at 35.9 M faces is conservative; a 4096 texture's bake takes 2.5 GB and may then run out of memory
  once, to run again with what is kept gone);
  both decoders stay as well when the holes and the steps fit beside them too and the next prop's texture
  sampling fits beside them under the cap in force: its sampler (`proj_cache_sampler_mib`) and, where it will keep
  them, its flow's proj_linear answers (`next_sampler_mib`: the tables of the blocks `proj_cache_count` gives the
  next prop under the cap in force, so the decoders never take the room of that cache - 3.5 GB with the 1.1 slack
  at 36 k tokens; on the full RTX 3060 a next prop above about 43 k tokens with the int8 texture network, 34 k with
  a bf16 one, keeps its cache and the decoders are loaded again for it). The sizes are the loaded networks' own
  (`loader.size_mib`: 2,708 MiB for a bf16 texture network, 1,557 for an int8 one, 914 for a decoder). What is kept
  is counted out of the room a finish step is offered (`tail_room`). On the full RTX 3060 every prop of the test
  set keeps both; under the caps of a card that leaves Pixal3D 7.2 GiB nothing is kept for the test set's 1536 props
  (by the figures), as before (with a bf16 network a 1536 prop's decode with both decoders beside it needs 7,181 MiB of a 5,749 budget;
  an int8 one fits beside the decode, but not beside the GLB steps of the bulky props in their steps).
  Measured on golden props at 1024 on the full card, the pair's loads went from 14 (8.9 s) to 5 (1.4 s) and its
  drops from 15 to 6; a prop of the same kind turns round about 1.1 s sooner and decodes 0.4-0.9 s sooner; under a
  7,168 MiB cap the loads and drops are those of before, and the peaks too (4,116 MiB reserved).
- **The decoders give their cache back level by level only where the decode needs it** (`parts.lean_needed`):
  `decoder.lean` (an `empty_cache` between the levels, the results moved through RAM once to leave the card
  unfragmented) costs about 0.2 s a decoder; without it a decode reserves up to 2.1 times what it allocates on
  whole tables and 1.3 times in boxes (`UNLEAN_WHOLE`, `UNLEAN_BOXES`, measured with no cap on sv1024 and sv1536),
  so the lean form is used where that would not fit the cap less what is kept beside it, with `decoders_fit`'s
  margin, and on the ladder's tight rung. The values are the same either way (`check_decode.py --no-trim`: 12 of 12
  comparisons on both cases). On the full RTX 3060 no prop of the test set needs it; on a card that leaves
  Pixal3D 7.2 GiB the bulkiest 1536 props (12.3 M voxels foreseen) and any shape on whole tables keep it, while the
  golden 1024 props and sv1536 (8.7 M), decoded in boxes, do without. The up-sampling keeps it.
- **The decoders' own kernels (round J, 2026-10-19; every value the same):** the masked sparse convolutions' row
  codes (the Gray code of the present neighbours and its binary number, which `sorted_idx` orders the rows by) are
  made by one Triton pass (`modules/sparse/conv/codes.py`, checked against the eager loops at first use, which serve
  if the kernel cannot run) instead of 27 + 26 integer passes over the neighbour map (0.05-3.1 ms against 1.5-47
  ms a level; the sv1536 finish decode -0.19 s); `chunked.rowwise` cuts a whole table into near-equal windows of at
  least `MIN_CALL_ROWS` (`EVEN_WINDOWS`) and the coarse levels' slab is 256 MiB (`COARSE_SLAB_BYTES`: -0.1 to -0.2 s
  a prop, the peak unchanged); NAF's RoPE turns every head of the guide map in place by one Triton kernel
  (`naf_sliced.FUSED_ROPE`, the eager `rope_apply` serving at any failure: cond_hr 4.38 -> 3.90 s on a three-view
  prop); and flex_gemm's masked implicit-GEMM kernel is tuned with a steadier benchmark plus two tiles its list
  lacks (`modules/sparse/conv/tuning.py`, installed by `conv_flex_gemm`): Triton's own benchmark times one run of a
  50-80 ms kernel under whatever load the card has, which on the developer's PC had left 60 of 74 keys with slow
  tiles (the 512x512 convolutions at 76.6 ms where 43.8 exists; every tile gives the same bits, SPLITK 1). The
  tuning is per card and per install (`autotune_cache.json`, keyed by the card's name); a tile a card cannot hold
  (shared memory) is skipped by Triton's autotuner. First-run tuning of the largest keys takes about twice as long
  as before, once; the decode of a 1536 prop is then 9.5 -> 8.1 s on the RTX 3060.

Before the decode in boxes, a bulky prop (41.6k tokens at 1408 by the token cap alone, 16.45 M voxels) was thus
made at 1024 on a card that leaves Pixal3D 7.2 GiB, inside it, in 299 s - instead of at 1408 in 801 s with 2.6 GB
of its decode spilled into shared memory. By the rules now its decode fits at 1408 and the closing of its holes
(5.9 GB there) did not: it was made at 1280 (foreseen by the figures, not measured); with the leaner hole filling
(2.5 GB there by the figures) its decode is the widest moment and it is made at 1408 - its token cap.

Nothing is retried, closed or freed while an exception is being handled: its traceback holds the failed attempt's
tensors. The runner reduces a failure to strings, leaves the `except` block and only then acts.

## Reading the networks: the copies in the models' dtypes

A part's network is read from disk onto the card when the part starts (`loader.py`): built without weights, its
file read in file order, each tensor uploaded and cast on the card to the dtype the model keeps it in. Onto the
card, `loader.READERS` (3) threads read the file at once, `CHUNK_BYTES` (16 MiB) at a time, into slots of
page-locked RAM - the 64 MiB read buffer and a pool of `POOL_BYTES` (64 MiB) beside it, made at the first load and
given back by `close()` - and the copies to the card are queued on the stream while the next chunks are read; a
slot is read into again once an event behind its copies has completed. Every tensor lands in a block of the same
size, taken in the same order, with the same bytes as the one-buffer way gives it (a tensor larger than the buffer
assembled in a block of its own, a smaller one that is cast passing through the one block taken for the casts), and
`cancelled()` is still asked once per tensor. The file cache gives this PC about 11 GB/s whatever the number of
readers; a flow's 2.8 GB load from the cache takes 0.40-0.47 s instead of 0.71-0.82 s, a third of it the model's
construction on the meta device. Onto the CPU (tests) the file goes through the one buffer, tensor by tensor. The
first network a batch loads (background removal) is read into the system's file cache by the read-ahead thread
when the engine is made, and again after a batch's last network has been loaded, for the next batch (that read is
left to go on when the batch ends). The four flow checkpoints are stored in float32 (5.4-5.6 GB each), but the
models keep 480 of
their 700 tensors, 98.7 % of the bytes, in bfloat16: half of what is read is cast away.

`weights.py` writes, once, a copy of such a checkpoint beside it, `<ckpt>.model-dtypes.safetensors`, with every
tensor the model takes from the file already in the model's dtype (2.72 GB for the sparse-structure flow, 2.81 GB
for each other flow: 11.14 GB a set of four instead of 22.0 GB):

```
python -m pixal3d.staged.weights <folder of pipeline.json>      # every checkpoint of pipeline.json, and of
                                                                 # pipeline_mv.json when its models are there
```

```python
from pixal3d.staged import weights
ok = weights.make_copies(snapshot, say=print)    # the same in-process; False when a checkpoint got no copy
weights.make_copy(ckpt)                          # one checkpoint, its path without extension
```

- **What a copy holds.** The tensors the loader takes from the original (the model's state, nothing else), in the
  original's order - the loader puts them on the card in file order, so where each block lands does not change -
  each in the model's dtype, cast on the CPU with `tensor.to(dtype)`: round to nearest even, the cast the loader
  makes on the card. The original is read through a 64 MiB buffer and each tensor (or piece of one) is cast and
  written before the next is read; the copy is written as `<copy>.tmp`, flushed to the disk and renamed. The card
  is not used (Spellcast3D runs the maker with it hidden).
- **When it is read.** The header's `__metadata__` records the original's name, size, modification time (ns) and
  the SHA-256 of its header, the torch version that cast it and the copy's format (`weights.FORMAT`). The loader
  reads a copy in place of its original only while that record matches the original as it is now and the running
  torch, and the copy holds exactly the tensors the model takes, in its dtypes and shapes
  (`weights.usable_copy`). A copy that is missing, stale, foreign or unreadable, or that fails while it is read,
  only means that the original is read (from its first tensor); no load fails because of a copy. A load from a
  copy casts nothing and takes no block for casts; `net_load` / `net_ready` `bytes` are what it read: the copy's
  tensors, or the original's. The read-ahead (`prefetch`) reads the copy when its record is current.
- **Which checkpoints get one.** Those where the model keeps some tensor in another dtype than the file and the
  copy is smaller: the flows. Not the decoders: the sparse-structure decoder's model keeps its file's dtypes, and
  the two others would widen fp16 tensors to fp32 (a file whose tensors all take two bytes or fewer is not even
  looked at with its model, whose class needs a card to be built). A copy that is no longer needed is removed.
- **The maker.** It skips copies that are up to date (it can run at every install), makes the missing and stale
  ones, refuses to leave less than 1 GB free on the drive, and prints one line per checkpoint with sizes and
  seconds. It exits 1 when a checkpoint got no copy, 0 otherwise. Spellcast3D runs it after its installer has
  downloaded the models and after the multi-view download (its `install/components.py` and `multiview.py`).

### The int8 copies of the detail flows

The flows the int8 policy converts (hr and tex under "detail": four checkpoints with the multi-view set) get a
second copy from the same maker, `<ckpt>.int8.model-dtypes.safetensors` (`weights.make_int8_copy`): the model's state
with its int8 layers in place - each converted linear's int8 weight, fp32 scales and fp32 bias exactly as
`int8.convert` makes them, every other tensor in the torso's dtype. A load reads it into a model built on the meta
device with those int8 layers already in place (`int8.skeleton`: the modules convert makes, by the same
constructors and the same walk over the model), and the convert hook then finds nothing left to quantise (one
`[INT8] ... read pre-quantised` line instead of the conversion's). What it saves:

| profile | read before | int8 copy | on the card before the conversion |
|---|---|---|---|
| ampere (bf16 torso) | 2.81 GB model-dtypes copy | 1.63 GB | 2.8 GB -> 1.6 GB |
| turing (fp16 torso) | 2.81 GB model-dtypes copy | none (below: Same bits) | - |
| pascal (fp32 torso) | 5.55 GB original (no smaller copy) | 2.01 GB | 5.5 GB -> 2.0 GB |

(The copies' sizes as the maker wrote them for each profile; the pascal row's card figures follow from them and
were not measured on that card.) On the RTX 3060 the first hr load of a process went 1.94 -> 0.54 s
(the int8 self-check 0.22-0.62 s now runs before the read; the conversion's 0.5 s of first-use quantising is gone)
and a tex load 0.45 -> 0.34 s; a later load from the file cache saves ~0.1 s of reading, a cold one the time of
1.2 GB at the drive's speed.

- **Same bits.** The weights are quantised on the CPU by `int8mm.quantize_rows_torch`, the reference the card's
  Triton quantiser equals bit for bit (`int8mm.self_check` checks it in every process; for every converted weight of
  the four checkpoints in bf16, and of the hr flow in fp16 and fp32, the CPU reference, the Triton kernel and the
  card's torch ops gave the same bytes on an RTX 3060). SageAttention's fold (v's 2^-8 and the answer's 2^8) is
  applied after the load to the int8 layers' channel scales and biases, as it is applied to the bf16 or fp32 weights
  before they are quantised at load: in those dtypes (fp32's exponent range) a power of two scales a row's absmax,
  its scale and its bias exactly and leaves its int8 codes as they are. Every tensor and module attribute of the four
  checkpoints loaded both ways is the same, byte for byte (bf16), and so are the decoded meshes of a single-view prop
  made with the copies and without. **Not in fp16**: there the fold's 2^-8 turns to_qkv's v weights and biases below
  2^-6 into subnormals, which lose bits before the quantiser sees them, while a copy is quantised from the unfolded
  weights - on the hr checkpoint hundreds to thousands of other int8 codes per block. So the turing profile (fp16
  torso) makes no int8 copy and reads none (`int8._copy_exact`, in `copy_rules` and `live_rules`): the install cannot
  know whether a run folds (the SageAttention switch is read at run time), and a copy is right only for one of the
  two. Winning the read back with the same bits would take a copy folded as the load folds it (v's rows scaled in
  fp16, their channel scales and bias times 2^8 again) and read only under the same switch.
- **When it is read.** Its record holds what the model-dtypes copy's does (the original's name, size, modification
  time, header SHA-256, the torch version) and the torso's dtype, the flow's kind, the layer rules it was made for,
  the int8 group and `weights.INT8_FORMAT`. The maker cannot run the card's self-check (it runs with the card
  hidden): it makes the copy for the rules a card of the PC's profile runs (`int8.copy_rules`: the policy's, the
  group-scaled layers left out where no group GEMM will be there - Triton below sm_80 cannot, pixal3d_kernels can).
  The loader reads it only where the process's own rules after its self-check on the card are the same
  (`int8.live_rules`), the convert hook names the same kind, and the copy holds exactly that model's state;
  anything else - a self-check that failed, PIXAL3D_INT8 off or another policy, a stale or broken copy - reads the
  model-dtypes copy (or the original) and converts as before. A "cs" policy (calibrated smoothing) has no copies.
  The read-ahead (`NetworkLoader._read_ahead_file`) picks the int8 copy by the same kind and rules - the process's
  after its self-check, `copy_rules`' prediction before it - and never runs the self-check itself.
- **Install.** `pixal3d_install_inputs` in Spellcast3D's installer carries `INT8_FORMAT`: an install made before the
  int8 copies existed runs its install once more and makes them (the model-dtypes copies are up to date and stay).
  4 x 1.63 GB more on the disk of an RTX 30xx PC (4 x 2.01 GB on a GTX 10xx, none on an RTX 20xx); about 4-14 s a
  copy on the Ryzen 9 (CPU, read from the model-dtypes copy where it is current). Spellcast3D's multi-view download
  runs the same maker with the card's `PIXAL3D_CARD_SM`, as the installer does: a maker in another profile would
  remove the copies of this one.

### Reading ahead

The loader's read-ahead thread reads the next networks' files into the system's file cache (`prefetch`; the runner
asks for the next group's while a group runs). It waits while RAM is short for a file (less than its size and
`PREFETCH_SPARE_BYTES` available): it looks again every `PREFETCH_RAM_POLL_S` and reads the file once RAM allows,
instead of giving up - in a cold 6-prop batch on an RTX 3060 the single-view texture flow's read ahead found
3,960 MiB available at the kind switch (4,344 needed) and was skipped, 5,750 MiB were free eight seconds later, and the
load read the 2.81 GB cold in 6.2 s with the card at 7 %. A load takes its own file out of the read-ahead (`forget`):
what was read stays in the cache, and the rest of the request goes on (unless a newer request came meanwhile: that
one wins, as it would have); and while any load runs the read-ahead waits between two of its reads (`hold`), so that
the read the card waits for has the drive to itself. Spellcast3D's launcher holds its own read-ahead of libraries and
first networks the same way, from the engine's `net_load` to its `net_ready` (pixal3d_server.py `ReadAheadGate`).

## Cards below sm_80: the precision profile

The engine was made for Ampere and newer (bf16 flows, Triton kernels, natten, flash_attn, SageAttention: all
sm_80+). `modules/precision.py` decides once per process what an older card runs (Spellcast3D round L, measured on
an RTX 2060 Super and a GTX 1080):

| profile | cards | flows' torso | attention | int8 GEMMs | decoders | NAF | Triton |
|---|---|---|---|---|---|---|---|
| ampere | sm_80+ | bf16 | flash_attn / SageAttention | Triton (group + per-row scales) | flex_gemm, fp16 | natten | yes |
| turing | sm_70 - sm_79 | fp16 | torch SDPA (the wheel's Turing kernel after the calibration's trial) | pixal3d_kernels (int8 tensor cores; group + per-row scales) | spconv, fp16 (flex_gemm after the calibration's check) | torch | elementwise only, after self-checks |
| pascal | below sm_70 | fp32 | torch SDPA, fp32 (the wheel's Pascal kernel after the calibration's trial) | pixal3d_kernels (dp4a; group + per-row scales) | spconv, fp16 (fp32 decoders ran the dragon at 1024 out of a GTX 1080's 8 GB in finish) | torch | no |

Turing has no bf16 arithmetic (torch emulates it at a seventh of the fp16 rate and no bf16 attention kernel
exists); consumer Pascal runs fp16 at a 64th of the fp32 rate (an fp16 attention call takes 14 s there against
0.7 s in fp32). An fp16 torso keeps the flows' residual stream in fp32 (`modules/utils.py` `stream_dtype`): the
stream, the running sum over the 30 blocks, reaches 1.1e5 at the late steps of the lr and hr flows, and a
sub-layer's output times its modulation gate passes 65,504 too, while the sub-layers' own outputs stay below
5,000 (round L's `fp16probe.py`); the blocks cast the normalised, modulated input to the torso's dtype for the
sub-layers (`inner_dtype`, set by `convert_to`) and apply the gate in the stream's dtype - no-ops on a bf16 torso.
The decoders' backend below sm_80 is spconv. Its GEMM kernels are prebuilt for sm_60, 70, 75, 80, 86, 89 and 90
only: on another card (a GTX 10xx is sm_61) spconv compiles them at run time through NVRTC, which cannot work from
the wheel (no headers shipped, no include path given: a GTX 1080 failed its first decoder convolution so), and such
a card runs spconv's Native algorithm with the products on torch.mm, one GEMM per kernel offset
(`conv_spconv.indice_conv_torch`: spconv's kernels' sums to 4e-7, 1.2-1.6x their time on the RTX 3060's check,
`round-l/spconv_products_check.py`; its index tables are spconv's own kernels still). On the pascal profile
`DISABLE_ADDMM_CUDA_LT=1` (set by the profile and by the hub for the card's processes): torch's fused linear-with-bias
goes through cuBLASLt, whose fp32 SGEMM reads out of bounds on a GTX 1080 (compute-sanitizer: `maxwell_sgemm_64x64_tn`,
cuBLAS 12.6 and 12.8 alike; an illegal memory access a few seconds into the ss flow, which the first whole prop on
that card needed off: 169 s for the boy at 1024). flex_gemm's Triton convolutions give
spconv's answers on an RTX 2060 Super and decode in boxes (`chunked.py`), but their tile tuning at the install's
calibration took 25-95 minutes a shape there (13 shapes), so they are tried only on request:
`PIXAL3D_TURING_FLEX_GEMM=on` at the calibration runs `modules/sparse/conv/backend_check.py` and records the
`conv_backend` section the profile reads (Triton's Windows builds are untested on Turing besides).
Where the decoders cannot decode in boxes (`chunked.boxes_supported()` false: spconv's boxes off, no
`decode_products` section that passed, or one that found them inexact) a card under 10 GiB makes a 1536 prop at 1024 (`parts.py`
`WHOLE_LEVEL_1536_MIB`: the knight's texture decode ran an RTX 2060 Super out of memory at 1536). The int8 GEMMs
are pixal3d_kernels' (`int8mm.cuda_linear`: the fork's W8A8 kernel with the scaling, bias and GELU in its epilogue,
per-row and group scales alike, so the whole int8 policy applies; self-checked at first use against the fp64
reference); without that package torch._int_mm serves the per-row layers and the group-scaled ones keep the
torso's dtype. An MLP whose hidden layer [tokens, 8192] would pass 256 MiB in the torso's dtype (fp32 on Pascal)
runs in row slabs there (`int8mm.cuda_mlp`, `MLP_SLAB_MIB`: the kernels have fixed tiles and no split-K, so the
same bits). The attention kernels of those cards (SageAttention's `_qattn_sm75` / `_qattn_sm61`, `sage.py`
OLD_CARD) serve only after the install's calibration ran them at the flows' sizes on the card in a child process
and recorded so (the record's `sage.candidate_ok`; a faulting kernel takes its process down, which must never be
the engine's); until then torch's SDPA serves. NAF's attention follows the profile too (`naf_sliced._kernel`).
Under SDPA the sparse self-attention takes q, k and v apart like flash_attn's (`sparse/attention/modules.py`:
the call with the three is the only one that offers one sequence's attention to SageAttention and it saves the
stack); a SparseTensor on spconv keeps its spatial shape as ints (`sparse/basic.py`: spconv compared a tensor
batch size at every `replace()`, a host sync 27,000 times in one lr sampling), and `engine._import_ahead` loads
spconv before the first flow (2-4 s that the lr flow's first step paid). On the RTX 3060 with the turing
profile forced those three took the boy's lr from 9.1 to 4.6 s.
Every network's constructor arguments pass through `precision.model_args` (the loader and the
copies): a flow's `dtype` becomes the torso's, a decoder's `use_fp16` is turned off on Pascal; the copies of the
flows (above) then hold fp16 on Turing and are not made on Pascal (fp32 is the file's own dtype). `PIXAL3D_CARD_SM`
tells a process that cannot see the card (the installer's copies) which card it is; `PIXAL3D_PRECISION` forces a
profile (a forced profile on another card runs that profile's code paths with the card's own kernels: the
developer's test of the Turing and Pascal paths on an RTX 3060, `round-l/smoke_precision.py` in Spellcast3D).

On both old profiles the flows' eager element-wise chains - about a third of an hr / tex pass on an RTX 2070 - run
as pixal3d_kernels' CUDA kernels with the chains' own bits (`modules/old_cards.py`; `pixal3d_kernels/csrc/
elementwise.cu`, README "The element-wise chains"): the int8 quantiser (`int8mm.quantize_rows_torch`'s bits;
int8mm takes it below sm_80 only where the Triton quantiser does not pass its check), the norm modulation, the
gated residual, and the q / k RMS norm with the rotary embedding (its sum of squares in torch's own reduction
order), every rounding through intrinsics that round once. Before the first use in a process each runs once on
small tables against its eager chain (`_check`): a kernel that gives other bits turns them all off, with an
`[OLD-CARDS]` line. Which kinds serve is the install's race (calibrate.py's step "old_cards", `old_cards.calibrate`:
each kernel against its chain at 4,096 x 1024 and 20,000 x 1536 rows in the profile's dtypes, interleaved medians,
on only where it is 5 % faster at both; the record's `old_cards` section keeps the timings and the choice, read only
when made on this card for this profile with the installed kernels build; without such a record every kind serves).
`PIXAL3D_OLD_CARD_KERNELS=off` gives the eager chains everywhere, `force` every kind whatever the race chose. On an
RTX 2070 an ss pass 601.6 -> 498.0 ms, lr 149.0 -> 118.4, hr 559.9 -> 430.6, tex 557.4 -> 431.1 (the boy at 1024;
the meshes the same); the RTX 3060 under the forced profiles had every kind 1.7-16x faster than its chain (pascal
1.7-8x in fp32, turing 2.7-16x); no real Pascal card was measured, which is why the race decides there.
Below sm_80 the decoders' spconv kernel choices are timed once at install as well (`modules/sparse/conv/
spconv_choices.py`, the record's `spconv_decoders`, put back before the first convolution of every process by
conv_spconv.py): spconv's own tuner otherwise times every candidate at the first prop of each process (25 s of the
boy's first prop on an RTX 2070, 75 s of the ork's) on whatever that prop brings, and chose differently from one
process to the next, which moved the decoded voxel count. The candidates are the C++ tuner's, fp16 and fp32
accumulation alike (`FP32_ACCUM = False`: the decoders' numbers of before); spconv's Python rule (fp32 accumulation
above 128 input channels, `FP32_ACCUM = None`) is steadier but a precision change, measured and not taken. Pascal
needs no record (its products run on torch.mm). A record without the section on Turing, or without the old cards'
race below sm_80, runs that step alone (`--only spconv_decoders`, `--only old_cards`; Spellcast3D's installer asks),
and so does one without the decode in boxes' floor below sm_80 (`--only decode_products`).

## Extending it

- **Another part.** Write `fn(ctx, prop, mode)` in `parts.py`, add it to `TABLE` and its key to `types.PARTS` at
  its place in the order; name its networks in `ENTER` (loaded when a group enters the part) or `LAZY` (loaded by
  the part itself), its large inputs in `WANTS`, its large outputs in `PRODUCES`, and what nobody reads after it in
  `SPENT`. Register a new network in `engine.py`.
- **Intermediate data from elsewhere.** A part is a function of the store. To start a prop from data made
  elsewhere (an edited sparse structure, a latent from another tool), put the items the next part reads into the
  store under the prop's id - the table above gives names, shapes and dtypes - and let the batch begin at that
  part; the checkpoint used by `resume` is the built-in case (`runner._restore`).
- **Reading intermediate data out.** Pass `store_factory=` to `StagedEngine` with a wrapper around `PropStore` that
  copies what the parts put (`code/spellcast/profiling/p3d_checks/check_engine.py` in Spellcast3D does this).

## How it is verified

- The logic of `engine.py`, `runner.py` and `parts.py` - order, waves, ladder, classifier, cancel, failure
  isolation, resume, the generator's hand-over, events - is tested on the CPU with everything below replaced by
  fakes (`check_engine.py --fake` in Spellcast3D's `code/spellcast/profiling/p3d_checks/`).
- The copies (`weights.py`) and the loader reading them or the original - what a copy holds and how it is
  rounded, stale, foreign and broken copies, a copy breaking half-way, the read-ahead, a pipeline folder - are
  tested on the CPU with small stand-in models (`code/tests/test_pixal3d_weights.py` in Spellcast3D).
- On the card, the same script runs the engine on golden cases: tensors captured from the untouched reference
  path stage by stage (conditionings, coords, generator states, latents, the decoded mesh). Integers must be equal;
  floats must be bit-equal wherever the reference reproduces itself bit for bit. The comparison is always against
  those captured tensors, never against a live run of the reference path: the fork's own convolutions and decoders
  changed under both, so the reference is no longer an independent control.
