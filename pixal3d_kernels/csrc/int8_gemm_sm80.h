// The flows' group-scaled W8A8 GEMM on sm_80 and newer (torch.ops.pixal3d_kernels.int8_gemm_group80): the same
// numbers as modules/int8mm.py's Triton "group" kernel, bit for bit, in its every output form.
//
//     acc[m, n] = 0;  for each 128-K slab g in order:  acc = fma(float(P[m, n, g]), xs[m, g], acc)
//     P[m, n, g] = the exact int32 dot product of xq[m, 128 g : 128 g + 128] and wq[n, 128 g : 128 g + 128]
//     y = bias ? fma(acc, ws[n], bias[n]) : acc * ws[n]       (fp32, one rounding each)
//     then GELU(tanh) as libdevice computes it when asked, and one of:
//       bf16 / fp16 (round to nearest even), the block's residual on the bf16 result (resid 1: x + bf16(h * gate),
//       2: x + bf16(second + h), 3: x + h), or int8 with one fp32 scale per row and 128 columns (fc1 -> fc2).
//
// The int32 slab sums are exact whatever the order of their MMAs; the fp32 steps run in the Triton kernel's order with
// the same instructions (int8_gemm_sm80.cu says which).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pixal3d_kernels {

struct GroupGemmParams {
    const int8_t * xq;      // [M, K], rows ldx bytes apart (16-byte aligned)
    const float *  xs;      // [M, K / 128], rows ldxs floats apart
    const int8_t * wq;      // [N, K], rows ldw bytes apart (16-byte aligned)
    const float *  ws;      // [N]
    const float *  bias;    // [N] or null
    void *         y;       // [M, N], rows ldy elements apart: bf16, fp16 or int8
    float *        qs;      // int8 out: [M, N / 128], rows ldqs floats apart
    const void *   rx;      // the residual's x (bf16, y's layout) or null
    const void *   rg;      // its gate (bf16 [N]) or null
    const void *   r2;      // its second term (bf16, y's layout) or null
    int64_t ldx, ldw, ldxs, ldy, ldqs;
    int     M, N, K;
    int     out;            // 0 bf16, 2 int8 + scales, 3 fp16
    int     gelu;
    int     resid;          // 0 none, 1 gate, 2 second, 3 plain
    int     group_m;        // the tile order: groups of group_m row tiles (Triton's GROUP_M)
};

// The tile configurations: their count, and per tile {BM, BN, WM, WN, BK, STAGES, blocks per SM, shared bytes,
// threads}; false for an index out of range.
int  int8_gemm_sm80_tiles();
bool int8_gemm_sm80_info(int tile, int (&info)[9]);

// Whether the code the current device runs is the kernel itself (built for sm_80 or newer), not the empty stub of an
// older architecture's code.
bool int8_gemm_sm80_loaded();

// Stream-ordered launch; the launch's error.
cudaError_t int8_gemm_sm80(const GroupGemmParams & p, int tile, cudaStream_t stream);

}  // namespace pixal3d_kernels
