# pixal3d_kernels

Spellcast3D's own CUDA kernels for the Pixal3D fork, as one small torch extension (`pixal3d_kernels._C`, operators
under `torch.ops.pixal3d_kernels`), for the cards the fork's Triton kernels do not serve (Turing sm_75, Pascal sm_61):
the flows' W8A8 linear with group scales (`int8_gemm_group`), and the flows' element-wise chains of the eager code
path (`quantize_rows`, `modulate`, `residual_gated`, `qk_norm_rope`: below, "The element-wise chains"). On sm_80 and
newer the fork keeps its Triton kernels (modules/int8mm.py, fused.py) - but for the group-scaled GEMM, whose
sm_80+ twin `int8_gemm_group80` the install races against Triton's (below) - and these operators run there too, untuned.

## Build

    TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX" python setup.py bdist_wheel # the product's wheel
    TORCH_CUDA_ARCH_LIST="7.5" python setup.py build_ext --inplace                          # one card (a pod)

Verified: `TORCH_CUDA_ARCH_LIST="6.1;7.5" python setup.py bdist_wheel` on Linux (CUDA 12.8, torch 2.8.0+cu128) gives
`pixal3d_kernels-0.1.0-cp39-abi3-linux_x86_64.whl` with sm_61 and sm_75 cubins (`-gencode arch=compute_61,code=sm_61`
and `compute_75,code=sm_75`), which installs and passes the checks; `build_ext` on Windows (MSVC 2022, CUDA 12.8,
the fork's venv torch 2.8.0+cu128, arch list "6.1;7.5;8.6") builds without a warning from these sources (torch's
cusparse headers warn under MSVC), imports, and resolves the op on meta tensors.

torch's `CUDAExtension` turns TORCH_CUDA_ARCH_LIST into the `-gencode` flags (without it: the building machine's
cards). Plain CUDA C++17, no `--use_fast_math`, no inline PTX that a listed architecture lacks (the tensor-core
kernel's `mma.sync.m8n8k16`, `ldmatrix` and `ld.global.nc.L2::128B` exist from sm_75; below it the kernel compiles to
an empty stub that the dispatcher never launches), no VLAs or GCC-only constructs: the same sources build with MSVC
(SageAttention's Windows defines are set by setup.py). `py_limited_api` (abi3): one wheel for Python 3.9+.
`PIXAL3D_KERNELS_PTXAS_V=1` prints ptxas's registers / spills per kernel and architecture.

## The operator

    torch.ops.pixal3d_kernels.int8_gemm_group(xq, xs, wq, ws, bias, gelu, out_dtype, raster=0, kernel=0) -> y
    pixal3d_kernels.int8_gemm_group(xq, xs, wq, ws, bias=None, gelu=False, out_dtype=torch.float16)

| argument | |
|---|---|
| xq | int8 [M, K], rows contiguous (another layout is copied) |
| xs | fp32 [M] (one scale per row) or [M, K // 128] (per row and 128-K group) |
| wq | int8 [N, K], rows contiguous |
| ws | fp32 [N] (per output channel) or [N, K // 128] (per channel and group; column-major, as `int8mm.quantize_weight(group=128)` lays it out - another layout is copied) |
| bias | fp32 [N] or None (another float dtype is converted) |
| gelu | GELU(tanh) after the bias |
| out_dtype | torch.float16 / float32 / bfloat16; torch.int32 = the raw integer sum (per-row / per-channel scales, no bias, no GELU: for checks) |
| raster, kernel | developer knobs: the tile order (0 = default) and the kernel family (0 = by the card, 1 = tensor cores, 2 = dp4a) |

Requirements: K % 128 == 0, N % 16 == 0, any M >= 0 (M = 0 gives an empty result; M < 16 works, unhurried), all
tensors on one CUDA device of compute capability 6.1 or newer.

The kernel family by the card: the tensor-core kernel on sm_75 and newer, dp4a below. Each kernel's body exists only in
code compiled for its architecture or newer; a build whose code for the card would be JIT-compiled from older PTX
(e.g. TORCH_CUDA_ARCH_LIST "6.1+PTX" on an sm_75 card) holds an empty stub there. The dispatcher asks the loaded
code's PTX version once per device (`cudaFuncGetAttributes`): such a card gets the dp4a kernel (whose compute_61 code
is real), a forced tensor-core call is refused, and a build without any code for the card raises a clear error -
never an uninitialised result (checked with "6.1+PTX" and "6.1" builds on an RTX 2060 Super).

Semantics: exactly `int8mm.reference()`:

    y[m, n] = sum_g P[m, n, g] * xs[m, g] * ws[n, g] + bias[n]      (then GELU(tanh) if asked)
    P[m, n, g] = sum over the 128 k of group g of xq[m, k] * wq[n, k]   (exact int32)

A per-row (per-channel) scale is constant over the groups and is applied once in the epilogue. With per-row scales on
both operands the integer sum runs over the whole K in int32 and is converted once: `((float) acc * xs[m]) * ws[n]`.
Otherwise each group's exact int32 sum is converted to fp32 and added, times its scales, into an fp32 accumulator
(`acc += P * xs` with per-channel weight scales, `acc += P * (xs * ws)` with group scales on both) - the order of
int8mm's Triton "group" kernel. The int32 -> fp32 conversion uses the 1.5 * 2^23 trick (the group's sum starts from the
bits of 12,582,912.0f; exact for |P| < 2^22, and a 128-value int8 group stays below 2,064,512), two full-rate
instructions instead of an I2F, which Turing and Pascal issue at a quarter of the fp32 rate. GELU is computed as
y * sigmoid(2u) with the hardware exp2 and reciprocal (~1e-6 relative to the exact value; 0 below y ~ -10, where the
exact value is under 1e-38). The two kernel families give the same bits (same arithmetic, same order: checked).

## The kernels

Ports of the W8A8 GEMMs of Spellcast3D's image engine (`temp_code/ggml-kquants`, `w8a8-turing.cuh` and
`w8a8-dp4a.cuh`, their tiling and instruction choices measured in `NOTES-w8a8.md` / `NOTES-w8a8-dp4a.md`), with the
activation scaled per 128-K group instead of the engine's 5-bit sub-scale per 32:

- `csrc/int8_gemm_imma.cu` (sm_75+): block 128 x 128, K stages of 64, 8 warps of 32 x 64, one block per SM; each
  m16n8k32 product as four `mma.sync.m8n8k16` chained through the C operand, so a group's (or the whole K's) integer
  sum accumulates in the MMA itself; a group's first MMA starts from the 1.5 * 2^23 constant, its last is followed by
  one FADD + one FFMA per output; two stages (one group) per loop iteration. No cp.async on sm_75: three 16 KB slots,
  global -> registers -> shared one stage ahead, one barrier per stage. Epilogue per 16-row fragment through shared
  memory, GELU / conversion / 16-byte stores in a loop that is not unrolled.
- `csrc/int8_gemm_dp4a.cu` (sm_61+): block 128 x 64, 128 threads, two blocks per SM, each thread 8 x 8 outputs, a
  stage's operands transposed into int32 words in shared memory (two 16-byte loads give a thread 8 rows' or 8
  channels' words of one k4 step, 64 dp4a); the dp4a chains run through a group (or the whole K); a stage is a loop
  over two 8-k4 halves (Pascal's ~8 KB instruction-cache ceiling on unrolled loops).

ptxas (CUDA 12.8 for sm_75, 12.6 for sm_61; no spills anywhere):

| kernel (xs, ws) | sm_75 tensor cores | sm_61 dp4a |
|---|---|---|
| per row, per channel | 196 registers, 48 KB dynamic shared, 1 block (8 warps) / SM | 153 registers, 24.0 KB, 2 blocks (8 warps) / SM |
| group, per channel | 222 | 237, 25.0 KB |
| group, group | 244 | 239, 25.5 KB |
| per row, group | 245 | 208, 24.5 KB |

## The dp4a kernel's twins (0.2.1, Spellcast3D round O)

    torch.ops.pixal3d_kernels.int8_gemm_group_q4(xq, xs, wq, ws, bias, gelu, out_dtype, raster=0) -> y
    torch.ops.pixal3d_kernels.int8_gemm_group_f5(xq, xs, wq, ws, bias, gelu, out_dtype, raster=0) -> y

The dp4a kernel (`csrc/int8_gemm_dp4a.cu`) with other code shapes and its bits (`csrc/int8_gemm_dp4a_twins.cu`):
int8_gemm_group's arguments but `kernel`, always the dp4a code (sm_61+; a build without code for the card refuses).
The group kernels' K-stage loop is 8,832 bytes of sm_61 SASS, above Pascal's ~8 KB instruction cache for a loop; the
twins bring it under: **q4** a stage as 4 trips of 4 k4 rows (5,920 B; ~2.6 % more issue slots a group: the shorter
trips' overhead), **f5** a group's first k4 row taking 1.5 * 2^23 as its dp4a addend so the conversion needs no
restart (64 moves less a group), then 3 trips of 5 rows (7,648 B; the shipped kernel's issue count). The per-row
kernel (6,592 B, nothing to restart) runs q4's shape in both. Every output's integer sum is the same exact chain and
its fp32 conversion the same `__fmaf_rn` in the same group order, the epilogue the shipped one: the same bits
(torch.equal at the flows' shapes, M 1-43,529, every scale layout and output dtype, the flows' real operands - on an
RTX 3060 through the compute_61 PTX's JIT; the executed FFMA / FADD / FMUL counts per stage of the sm_61 SASS equal
the shipped kernel's). Which one serves is the install's race on the card itself (the fork's modules/int8mm.py
`calibrate_dp4a_twins`, called by old_cards' calibration step; the record's old_cards.dp4a; the shipped kernel
without a record): no GTX 10xx could time them when they were written
(code/devtools/pixal3d-lowmem/round-o/pascal-kernels/: the checks, the SASS report, a pod's bench).

## Figures

Kernel-only medians (CUDA events, every method once a round, 12 rounds; random int8 operands; ms and TOPS = 2MNK / t).
"policy" = the "detail" policy's scales (group on the activations of qkv / fc1 / fc2, per-row on out; per-channel
weights; bias; GELU on fc1), the card's torso dtype out; "rows" = per-row scales on both; "cg" = group scales on both;
"intmm+torch" = torch._int_mm in 8,192-row chunks + int8mm.epilogue_torch (per-row scales only: what these cards would
otherwise run); "native" = F.linear in the torso dtype (+ GELU on fc1).

RTX 2060 Super (sm_75, 135 W power cap; torch 2.8.0+cu128; fp16 out; no other process on the card during the run;
the card's int8 MMA ceiling measured with `ubench_imma.cu`: 117 T/s; cuBLASLt's own int8 GEMM reaches 72-80 T/s on
square sizes):

| M | layer (K x N) | policy | rows | cg | intmm+torch | intmm alone | native fp16 | policy vs native |
|---|---|---|---|---|---|---|---|---|
| 4096 | qkv 1536 x 4608 | 1.144 ms 50.7 | 0.995 58.3 | 1.142 50.8 | 2.758 21.0 | 0.898 64.6 | 2.244 25.8 | 1.96x |
| 4096 | fc1 1536 x 8192 + GELU | 2.117 48.7 | 1.880 54.8 | 2.127 48.5 | 5.518 18.7 | 1.543 66.8 | 4.376 23.6 | 2.07x |
| 4096 | fc2 8192 x 1536 | 2.018 51.1 | 1.629 63.3 | 2.119 48.6 | 2.012 51.2 | 1.381 74.7 | 4.014 25.7 | 1.99x |
| 4096 | out 1536 x 1536 | 0.376 51.3 | 0.372 51.9 | 0.422 45.9 | 0.981 19.7 | 0.350 55.2 | 0.821 23.5 | 2.18x |
| 21504 | qkv | 6.019 50.6 | 5.283 57.6 | 6.092 50.0 | 14.400 21.1 | 4.656 65.4 | 12.104 25.1 | 2.01x |
| 21504 | fc1 + GELU | 11.215 48.3 | 9.969 54.3 | 11.365 47.6 | 28.942 18.7 | 7.950 68.1 | 22.867 23.7 | 2.04x |
| 21504 | fc2 | 10.527 51.4 | 8.264 65.5 | 10.682 50.7 | 10.465 51.7 | 7.272 74.4 | 21.626 25.0 | 2.05x |
| 21504 | out | 1.772 57.3 | 1.765 57.5 | 2.032 49.9 | 4.897 20.7 | 1.617 62.8 | 3.988 25.4 | 2.25x |

GTX 1080 (sm_61): not measured yet - the pod's card was held by another process for the whole session. The dp4a
kernel was compiled for sm_61 (CUDA 12.6 and 12.8: 153-239 registers, no spills), and its compute_61 PTX - the code
ptxas turns into that SASS - ran on the RTX 2060 Super (a "6.1+PTX" build, JIT-compiled there): every check passes and
its outputs are bit-identical to the tensor-core kernel's and to the sm_75 build of the dp4a kernel (72 configurations
hashed). The dp4a kernel on the RTX 2060 Super: 15.8-17.6 TOPS (~60 % of that card's dp4a rate). To do on a Pascal
card: `check.py` and `bench.py` (the speed, and GELU's hardware exp2 / reciprocal there).

## Checks and benchmarks

code/devtools/pixal3d-lowmem/round-l/int8-gemm/: `check.py` (exactness and the fp32-rounding bound against int8mm's
fp64 references on every scale layout, epilogue and output dtype, masked tails, the flows' shapes at M = 21,504),
`bench.py` (the table above), `ubench_imma.cu` (the card's m8n8k16 ceiling).

## The element-wise chains

    torch.ops.pixal3d_kernels.quantize_rows(x, group, inv_smooth) -> (q, s)
    torch.ops.pixal3d_kernels.modulate(h, scale1, shift, out_dtype) -> y
    torch.ops.pixal3d_kernels.residual_gated(x, h, gate) -> y
    torch.ops.pixal3d_kernels.qk_norm_rope(x, gamma, scale, eps, phases, out_dtype) -> y

On Turing and Pascal the flows' blocks run their eager torch code (the fused Triton kernels are bf16-only), where each
norm, modulation, residual, q / k RMS norm and rotary embedding is a chain of 3 to 10 kernels in fp32 (with fp16
casts on Turing). Each operator here is one such chain in one kernel, with the chain's bits: every rounding the chain
makes, in its order, with intrinsics that round once (`__fmul_rn`, `__fadd_rn`, `__fdiv_rn`, IEEE `sqrtf`; separate
torch kernels are never contracted into an FMA, and neither are these). modules/old_cards.py calls them from the
blocks and checks each against its chain once a process on the card (a torch whose kernels round otherwise turns them
off); modules/int8mm.py takes the quantiser below sm_80. Which of them serve on a card is the install's race
(old_cards.calibrate, the calibration record's "old_cards": each against its chain at two of the flows' sizes, on only
where it is 5 % faster at both).

| operator | the eager chain (one item; rows of C channels) |
|---|---|
| quantize_rows | int8mm.quantize_rows_torch: x.float() [* inv_smooth], amax of the row's (group's) \|x\| (exact), floored at 1e-30, q = rint(x * (127 / amax)) clamped to +-127, s = amax / 127; fp16 / fp32 / bf16 in, per row or per 128 (any group dividing K) |
| modulate | after LayerNorm32: `(h * (1 + scale) + shift)` then the cast to the sub-layers' dtype (h fp32; 1 + scale is the caller's own torch op) |
| residual_gated | `x + h.float() * gate` (x fp32, h fp16 or fp32, gate one row) |
| qk_norm_rope | `(F.normalize(x.float(), dim=-1) * gamma * scale).to(x.dtype)` (MultiHeadRMSNorm) and, with phases, `view_as_real(view_as_complex(y.float()) * phases).to(x.dtype)` (the rotary embedding): x [T, H, 128] with any token / head strides |

The q / k norm's sum of squares is torch's own reduction order for 128 values a row (Reduce.cuh with its config for
that shape: 32 lanes along the row, no vectorised input since 128 is not above 128, four accumulators a lane at
stride 32 each holding one square, combined in order, then shuffles down at offsets 1, 2, 4, 8, 16); the complex
product is c10::complex's own operator, as torch's mul kernel has it. Checked bit for bit (code/devtools/pixal3d-lowmem
/round-n/old-cards/check_kernels.py: random tables with outliers and zero rows, every dtype the profiles use, odd row
counts, and the tensors captured from the flows' real passes) on an RTX 2070 (Linux, torch 2.8.0+cu128: 109 checks)
and on an RTX 3060 against Windows' torch 2.8.0+cu128 (sm_86 code; the turing profile's fp16 chains, 85 checks, and
the pascal profile's fp32 ones, 70); and the boy at 1024 and the ork made through the whole pipeline with them on the
RTX 2070 give the decoded meshes of the eager code, byte for byte. Not run on a Pascal card: there the first-use check
(old_cards.py) decides.

Times on the RTX 2070 (CUDA events, medians; 5,084 tokens = an hr / tex pass of the boy at 1024, C = 1536, 12 heads):

| chain | eager | kernel |
|---|---|---|
| q / k norm + rope (one of q, k) | 1.30 ms | 0.16 ms |
| q norm (cross attention) | 0.87 ms | 0.10 ms |
| modulate | 0.48 ms | 0.17 ms |
| residual_gated | 0.57 ms | 0.21 ms |
| quantize_rows, K 1536, group 128 | 1.11 ms torch ops, 0.115 ms Triton | 0.09 ms |
| quantize_rows, K 1536, per row | 1.08 ms torch ops, 0.115 ms Triton | 0.08 ms |
| quantize_rows, K 8192 (fc1's hidden), group 128 | 5.57 ms torch ops, 0.37 ms Triton | 0.34 ms |

(Whole calls, the launch included. Inside the flows' passes, where the CPU runs ahead, the Triton quantiser's kernel
alone is the faster one on the RTX 2070 - 61 against 64 us per row and group, 319 against 326 us on fc1's hidden
layer - so int8mm keeps Triton's where it passes its check and takes this one where Triton does not run: Pascal,
and a Turing card whose Triton fails.) The boy at 1024 through the whole pipeline on the RTX 2070 with these
kernels: one conditional pass of ss 601.6 -> 498.0 ms, lr 149.0 -> 118.4, hr 559.9 -> 430.6, tex 557.4 -> 431.1
(kernels a pass 2,234 -> 1,093 in hr); its decoded mesh the same bytes.

An int8 output for fc1 (the hidden layer quantised in the GEMM's epilogue, bit-identical to the quantiser on its fp16
table) was measured 16-21 % slower than the GEMM plus this quantiser on the RTX 2070 (its epilogue's arithmetic is
exposed at one block an SM) and is not part of the package (round-n/old-cards/int8-out-epilogue/).

## sm_80+: the group GEMM in the Triton kernel's bits

    torch.ops.pixal3d_kernels.int8_gemm_group80(xq, xs, wq, ws, bias, gelu, out, rx, rg, r2, tile, group_m) -> (y, qs)
    torch.ops.pixal3d_kernels.int8_gemm_group80_info(tile) -> [BM, BN, WM, WN, BK, STAGES, blocks per SM, shared, threads]
    torch.ops.pixal3d_kernels.int8_gemm_group80_loaded() -> bool

`csrc/int8_gemm_sm80.cu` (+ `.h`, `ops_sm80.cpp`, a fragment of the package's op library): the hr / tex flows'
group-scaled layers (activations scaled per row and 128-K slab, weights per output channel) for sm_80 and newer, with
modules/int8mm.py's Triton "group" kernel's numbers bit for bit in every output form: bf16 / fp16, the block's
residual on the bf16 result (rx with rg = a gate row, or r2 = a second term, or neither), int8 + one fp32 scale per
row and 128 columns (out 2: fc1's output for fc2). The arithmetic is the Triton kernel's PTX instruction for
instruction (int8_gemm_sm80.h: per slab the exact int32 sum, `fma.rn` into the fp32 accumulator in slab order; the
epilogue's `fma.rn`, libdevice's tanhf as Triton links it - its ftz build -, `div.rn.ftz`, `cvt.rni`). Requirements:
K % 128 == 0, N % 16 == 0 (int8 out: N % 128 == 0), xs [M, K / 128] with unit stride along K, ws [N], bias fp32 [N] or
None, rx / r2 bf16 [M, N], rg bf16 [N]. Below sm_80 the kernel compiles to an empty stub that `..._loaded()` reports
and the op refuses.

The kernel: mma.sync.m16n8k32, ldmatrix.x4 from XOR-swizzled tiles, cp.async in 2-4 slots, four warps and 48 KB a
block, two blocks per SM. `tile` picks one of seven configurations (64 x 128 or 128 x 64 blocks, 64- or 128-K
stages); modules/int8mm.py races them against the Triton kernel at install (calibrate_group_cuda) and records the
fastest per layer form where it beats Triton by 3 %; `group_m` is the tile order (Triton's GROUP_M; int8mm passes 8).

RTX 3060 (interleaved CUDA-event medians, random tables, bits equal on every tile; ms, Triton -> the best tile):

| rows | to_qkv 1536 -> 4608 | fc1 1536 -> 8192, GELU, int8 out | fc2 8192 -> 1536 |
|---:|---|---|---|
| 8,192 | 1.808 -> 1.675 (-7.4 %) | 3.625 -> 3.294 (-9.1 %) | 2.986 -> 2.877 (-3.6 %) |
| 19,928 | 4.394 -> 4.178 (-4.9 %) | 8.735 -> 8.069 (-7.6 %) | 7.175 -> 7.319 (+2.0 %) |
| 21,504 | 4.713 -> 4.474 (-5.1 %) | 9.276 -> 8.612 (-7.2 %) | 7.710 -> 7.821 (+1.4 %) |
| 43,529 | 9.323 -> 9.125 (-2.1 %) | 18.913 -> 17.574 (-7.1 %) | 16.011 -> 16.325 (+2.0 %) |

Without their global copies the four-warp tiles reach 100 TOPS on fc2's shape (92 % of the card's int8 peak); with
them 66-69: the 64 x 128 blocks move half again the bytes per product of the per-row Triton kernel's 128 x 128 ones
(86 TOPS), which need 64 x 64 warp tiles - more registers than the slab sums and the fp32 accumulators leave. One
8-warp block per SM, two halves of a 64 x 64 warp tile, fragments double-buffered across stages and a later copy issue
were all slower (code/devtools/pixal3d-lowmem/round-n/group-gemm/README.md).
