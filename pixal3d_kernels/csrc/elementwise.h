// The element-wise kernels (elementwise.cu): torch.ops.pixal3d_kernels.quantize_rows / modulate / residual_gated /
// qk_norm_rope. Each gives the bits of the eager torch chain it stands for (README.md, "The element-wise chains").

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pixal3d_kernels {

enum DType : int { DT_F32 = 0, DT_F16 = 1, DT_BF16 = 2 };

struct QuantParams {
    const void *  x;            // [M, K], rows ldx elements apart, DType dtype
    const float * inv_smooth;   // [K] or null
    int8_t *      q;            // [M, K] contiguous
    float *       s;            // [M] or [M, K / group] contiguous
    int64_t M, ldx;
    int     K, group;           // group 0: one scale per row
    int     dtype;
    int     team;               // threads per segment (a row, or a group): segment / 8 / vpt
    int     vpt;                // chunks of 8 values a thread holds (1, 2 or 4)
};

struct QkParams {
    const void *   x;           // [T, H, 128], tokens ld_t and heads ld_h elements apart, the last dimension contiguous
    const void *   gamma;       // [H, 128] contiguous
    const float2 * phases;      // complex64 [T, 64], tokens ld_ph complex values apart, or null (no rotary embedding)
    void *         out;         // [T, H, 128] contiguous
    float  scale, eps;
    int64_t rows, ld_t, ld_h, ld_ph;    // rows = T * H
    int    H;
    int    x_dtype, g_dtype, out_dtype;
};

cudaError_t quantize_rows(const QuantParams & p, cudaStream_t stream);
cudaError_t modulate(const void * h, const void * s1, const void * sh, void * out, int64_t n, int C, int s_dtype,
                     int out_dtype, cudaStream_t stream);
cudaError_t residual_gated(const void * x, const void * h, const void * gate, void * out, int64_t n, int C,
                           int h_dtype, int g_dtype, cudaStream_t stream);
cudaError_t qk_norm_rope(const QkParams & p, cudaStream_t stream);

}  // namespace pixal3d_kernels
