# pixal3d_kernels

Spellcast3D's own CUDA kernels for the Pixal3D fork, as one small torch extension (`pixal3d_kernels._C`, operators
under `torch.ops.pixal3d_kernels`). Today it holds one operator: the flows' W8A8 linear with group scales for the cards
Triton's int8 GEMM does not serve (Turing sm_75, Pascal sm_61). On sm_80 and newer the fork keeps its Triton kernel
(modules/int8mm.py); this operator runs there too (the tensor-core kernel is plain sm_75 code), untuned.

## Build

    TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9;9.0;12.0+PTX" python setup.py bdist_wheel     # the product's wheel
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
