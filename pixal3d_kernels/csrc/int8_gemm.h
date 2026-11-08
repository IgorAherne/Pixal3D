// The W8A8 GEMM with group scales (torch.ops.pixal3d_kernels.int8_gemm_group): what the two kernels share.
//
//     y[m, n] = sum over the 128-K groups g of  P[m, n, g] * sa[m, g] * sb[n, g]   (+ bias[n], then GELU(tanh))
//     P[m, n, g] = the exact int32 dot product of xq[m, 128 g : 128 g + 128] and wq[n, 128 g : 128 g + 128]
//
// sa is the activation's scale (one per row, or one per row and group), sb the weight's (one per output channel, or
// one per channel and group). A per-row (per-channel) scale is constant over the groups and is applied once in the
// epilogue; when both are per-row / per-channel the integer sum runs over the whole K in int32 and is converted once
// (the round I "rows" layers), otherwise each group's int32 partial sum is converted to fp32 and added, times its
// scales, into an fp32 accumulator (int8mm.py's Triton "group" kernel, the same order of operations).
//
// The int32 -> fp32 conversion of a group's partial sum is the 1.5 * 2^23 trick: the partial sum starts from the bits
// of 12,582,912.0f (MAGIC) instead of 0, and its bits read as a float minus 12,582,912.0f are the exact sum, for any
// |sum| < 2^22 (a 128-value group's |sum| <= 128 * 127^2 = 2,064,512). Two full-rate instructions instead of an I2F,
// which Pascal and Turing issue at a quarter of the fp32 rate.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pixal3d_kernels {

constexpr int GROUP = 128;                  // input channels per group scale

enum OutType : int { OUT_F32 = 0, OUT_F16 = 1, OUT_BF16 = 2, OUT_I32 = 3 };

struct Int8GemmParams {
    const int8_t * xq;      // [M, K], rows ldx bytes apart (16-byte aligned)
    const float *  xs;      // [M] (per row) or [M, K / 128] (rows ldxs apart)
    const int8_t * wq;      // [N, K], rows ldw bytes apart (16-byte aligned)
    const float *  ws;      // [N] (per channel) or [K / 128][N] (column-major [N, K / 128]: a group's N contiguous)
    const float *  bias;    // [N] or null
    void *         y;       // [M, N] contiguous, OutType out
    int64_t ldx, ldw, ldxs;
    int     M, N, K;
    int     out;            // OutType; OUT_I32 = the raw integer sum (per-row / per-channel scales, no bias / GELU)
    int     gelu;
    int     raster;         // the tile order: > 0 groups of `raster` row tiles, < 0 of -raster column tiles, 0 auto
};

// The launchers (stream-ordered; a_group / b_group: xs / ws hold one scale per group). They return the launch's error.
cudaError_t int8_gemm_imma(const Int8GemmParams & p, bool a_group, bool b_group, cudaStream_t stream);   // sm_75+
cudaError_t int8_gemm_dp4a(const Int8GemmParams & p, bool a_group, bool b_group, cudaStream_t stream);   // sm_61+

// Whether the code the current device would run is the kernel itself: its body exists only in code compiled for its
// architecture or newer (sm_75 / sm_61); a build whose code for this card is JIT-compiled from older PTX (an arch list
// such as "6.1+PTX" on an sm_75 card) holds an empty stub there, which must never run.
bool int8_gemm_imma_loaded();
bool int8_gemm_dp4a_loaded();

// Spellcast3D round O: the dp4a kernel's twins (int8_gemm_dp4a_twins.cu: twin 1 = q4, 2 = f5), the shipped kernel's
// bits with another code shape, raced against it at install (pixal3d/modules/int8mm.py); sm_61+.
cudaError_t int8_gemm_dp4a_twin(const Int8GemmParams & p, bool a_group, bool b_group, int twin, cudaStream_t stream);
bool int8_gemm_dp4a_twins_loaded();

}  // namespace pixal3d_kernels
