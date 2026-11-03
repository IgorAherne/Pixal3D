// Device helpers of the two int8 GEMM kernels: the tile order, the group conversion, the epilogue's arithmetic.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "int8_gemm.h"

namespace pixal3d_kernels {

constexpr int   MAGIC   = 0x4B400000;       // the bits of 12,582,912.0f = 1.5 * 2^23
constexpr float MAGIC_F = 12582912.0f;

// grouped tile order (the ggml W8A8 kernels'): raster > 0: `raster` row tiles side by side, walking the column tiles;
// raster < 0: -raster column tiles side by side (their weight rows stay in L2), walking the row tiles
static __device__ __forceinline__ void tile_order(const int bid, const int ntm, const int ntn, const int raster,
                                                  int & tm, int & tn) {
    if (raster > 0) {
        const int per_grp = raster * ntn;
        const int first_m = (bid / per_grp) * raster;
        const int gsize   = min(raster, ntm - first_m);
        const int in_grp  = bid % per_grp;
        tm = first_m + in_grp % gsize;
        tn = in_grp / gsize;
    } else {
        const int per_grp = -raster * ntm;
        const int first_n = (bid / per_grp) * -raster;
        const int gsize   = min(-raster, ntn - first_n);
        const int in_grp  = bid % per_grp;
        tn = first_n + in_grp % gsize;
        tm = in_grp / gsize;
    }
}

// a group's partial sum (started from MAGIC) times its scale, added into the fp32 accumulator; the partial sum restarts
static __device__ __forceinline__ void add_group(float & acc, int & part, const float s) {
    acc  = __fmaf_rn(__int_as_float(part) - MAGIC_F, s, acc);
    part = MAGIC;
}

// GELU(tanh): 0.5 y (1 + tanh(u)) = y * sigmoid(2u), u = sqrt(2 / pi) (y + 0.044715 y^3). The sigmoid form has no
// cancellation where 1 + tanh(u) is small; the hardware exp2 and reciprocal (two MUFU instructions) keep it ~1e-6
// relative to the exact value, three orders below the int8 operands' own error (the accurate expf and IEEE division
// are ~25 instructions an output, on cards whose fp32 rate is a sixteenth of their int8 MMA rate).
// For y below about -10, 1 + exp(-2u) passes 2^126 and the result is 0 (the exact one is below 1e-38).
static __device__ __forceinline__ float gelu_tanh(const float y) {
    const float u = 0.7978845608028654f * (y + 0.044715f * (y * y * y));
    return __fdividef(y, 1.0f + __expf(-2.0f * u));
}

static __device__ __forceinline__ unsigned pack_half2(const float a, const float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}

static __device__ __forceinline__ unsigned pack_bf162(const float a, const float b) {
    // __floats2bfloat162_rn is a native instruction from sm_80; below, cuda_bf16.h emulates it (round to nearest even)
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}

// 8 consecutive outputs of one row (v: fp32 values, or the int32 sums' bits for OUT_I32): GELU when asked, the output
// dtype, 16-byte stores (y + off: 32-byte aligned for 4-byte outputs, 16 for 2-byte). Called from a loop that is not
// unrolled: one copy of this code per call site keeps the epilogue small (an epilogue of ~4,000 instructions, run once
// a tile, cost more than its arithmetic in instruction fetches).
static __device__ __forceinline__ void store8(void * y, const int64_t off, float (&v)[8], const int out,
                                              const bool gelu) {
    if (gelu) {
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = gelu_tanh(v[j]);
    }
    if (out == OUT_F32 || out == OUT_I32) {
        float4 * d = reinterpret_cast<float4 *>(static_cast<float *>(y) + off);
        __stcs(d + 0, make_float4(v[0], v[1], v[2], v[3]));
        __stcs(d + 1, make_float4(v[4], v[5], v[6], v[7]));
    } else if (out == OUT_F16) {
        __stcs(reinterpret_cast<int4 *>(static_cast<__half *>(y) + off),
               make_int4((int) pack_half2(v[0], v[1]), (int) pack_half2(v[2], v[3]), (int) pack_half2(v[4], v[5]),
                         (int) pack_half2(v[6], v[7])));
    } else {
        __stcs(reinterpret_cast<int4 *>(static_cast<__nv_bfloat16 *>(y) + off),
               make_int4((int) pack_bf162(v[0], v[1]), (int) pack_bf162(v[2], v[3]), (int) pack_bf162(v[4], v[5]),
                         (int) pack_bf162(v[6], v[7])));
    }
}

}  // namespace pixal3d_kernels
