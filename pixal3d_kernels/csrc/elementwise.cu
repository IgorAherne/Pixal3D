// The flows' element-wise chains for the cards without the fork's fused Triton kernels (Turing, Pascal), each the
// eager torch chain it replaces to the bit: every rounding the chain makes, in its order, with the intrinsics that
// round once (__fmul_rn, __fadd_rn, __fdiv_rn: never contracted into an FMA, as separate torch kernels never are).
//
//   quantize_rows    int8mm.quantize_rows_torch: amax of |x| (exact), q = rint(x * (127 / amax)) clamped to +-127,
//                    s = amax / 127 (both divisions IEEE), per row or per row and group
//   modulate         after a LayerNorm32: (h * (1 + scale)) + shift, cast to the sub-layers' dtype
//   residual_gated   x + float(h) * gate (the stream in fp32, h in the sub-layers' dtype)
//   qk_norm_rope     the q / k RMS norm (F.normalize(x.float(), dim=-1) * gamma * scale, cast) and, when given the
//                    phases, the rotary embedding (x.float() as complex pairs times the phase, cast): torch's norm
//                    reduction in torch's own order (Reduce.cuh for 128 values a row: 32 lanes, four accumulators a
//                    lane at stride 32, combined in order, then the warp's shuffles at offsets 1, 2, 4, 8, 16)

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <c10/util/complex.h>

#include "elementwise.h"

namespace pixal3d_kernels {
namespace {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// v rounded to T and back (the value a cast to T and a later .float() leave)
template <typename T> __device__ __forceinline__ float round_to(float v) { return to_f<T>(from_f<T>(v)); }

constexpr float TINY = 1e-30f;      // int8mm.TINY: the amax floor

// ---- quantize_rows --------------------------------------------------------------------------------------------------
// A team of P threads per segment of `seg` values (a row, or a row's group): P = seg / 8 / VPT, a power of two up
// to 32 (teams inside a warp) or a multiple of 32 up to 1024 (a block); a thread holds VPT chunks of 8 consecutive
// values, chunk c at (lane + c P) * 8.

template <typename T, int VPT>
__global__ void quant_kernel(const T * __restrict__ x, const float * __restrict__ inv_smooth, int8_t * __restrict__ q,
                             float * __restrict__ s, const int64_t M, const int K, const int64_t ldx, const int seg,
                             const int team) {
    extern __shared__ float red[];
    const int nseg_row = K / seg;
    const int64_t segs = M * nseg_row;
    const int teams_per_block = blockDim.x / team;
    const int t_in_block = threadIdx.x / team;
    const int lane_t = threadIdx.x % team;
    const int64_t sid = (int64_t) blockIdx.x * teams_per_block + t_in_block;
    const bool live = sid < segs;
    const int64_t row = live ? sid / nseg_row : 0;
    const int g = live ? (int) (sid % nseg_row) : 0;
    const T * src = x + row * ldx + (int64_t) g * seg;
    float v[VPT][8];
    float amax = 0.0f;
#pragma unroll
    for (int c = 0; c < VPT; ++c) {
        const int k0 = (lane_t + c * team) * 8;
        const bool in = live && k0 < seg;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float f = in ? to_f<T>(src[k0 + j]) : 0.0f;
            if (inv_smooth != nullptr && in) {
                f = __fmul_rn(f, inv_smooth[(int64_t) g * seg + k0 + j]);
            }
            v[c][j] = f;
            amax = fmaxf(amax, fabsf(f));
        }
    }
    // the team's amax (exact in any order)
    if (team <= 32) {
        for (int off = team / 2; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
    } else {
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
        const int warps = team / 32;
        if (threadIdx.x % 32 == 0) {
            red[threadIdx.x / 32] = amax;
        }
        __syncthreads();
        const int w0 = t_in_block * warps;
        amax = red[w0];
        for (int w = 1; w < warps; ++w) {
            amax = fmaxf(amax, red[w0 + w]);
        }
    }
    if (!live) {
        return;
    }
    amax = fmaxf(amax, TINY);
    const float inv = __fdiv_rn(127.0f, amax);
#pragma unroll
    for (int c = 0; c < VPT; ++c) {
        const int k0 = (lane_t + c * team) * 8;
        if (k0 >= seg) {
            continue;
        }
        char4 lo, hi;
        int r[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            r[j] = min(127, max(-127, __float2int_rn(__fmul_rn(v[c][j], inv))));
        }
        lo = make_char4((signed char) r[0], (signed char) r[1], (signed char) r[2], (signed char) r[3]);
        hi = make_char4((signed char) r[4], (signed char) r[5], (signed char) r[6], (signed char) r[7]);
        int8_t * dst = q + row * (int64_t) K + (int64_t) g * seg + k0;
        *reinterpret_cast<char4 *>(dst) = lo;
        *reinterpret_cast<char4 *>(dst + 4) = hi;
    }
    if (lane_t == 0) {
        s[row * nseg_row + g] = __fdiv_rn(amax, 127.0f);
    }
}

template <typename T, int VPT>
cudaError_t launch_quant(const QuantParams & p, cudaStream_t stream) {
    const int seg = p.group > 0 ? p.group : p.K;
    const int team = p.team;
    const int threads = team > 32 ? team : 256;         // a block per segment, or 256 / team segments a block
    const int teams_per_block = threads / team;
    const int64_t segs = p.M * (int64_t) (p.K / seg);
    const int64_t blocks = (segs + teams_per_block - 1) / teams_per_block;
    if (blocks == 0) {
        return cudaSuccess;
    }
    const size_t smem = team > 32 ? (size_t) (threads / 32) * sizeof(float) : 0;
    quant_kernel<T, VPT><<<(unsigned) blocks, threads, smem, stream>>>(
        static_cast<const T *>(p.x), p.inv_smooth, p.q, p.s, p.M, p.K, p.ldx, seg, team);
    return cudaGetLastError();
}

template <typename T>
cudaError_t quant_dispatch(const QuantParams & p, cudaStream_t stream) {
    switch (p.vpt) {
        case 1: return launch_quant<T, 1>(p, stream);
        case 2: return launch_quant<T, 2>(p, stream);
        case 3: return launch_quant<T, 3>(p, stream);
        case 4: return launch_quant<T, 4>(p, stream);
        case 5: return launch_quant<T, 5>(p, stream);
        case 6: return launch_quant<T, 6>(p, stream);
        case 7: return launch_quant<T, 7>(p, stream);
        case 8: return launch_quant<T, 8>(p, stream);
        default: return cudaErrorInvalidValue;
    }
}

// ---- modulate / residual --------------------------------------------------------------------------------------------

template <typename TS, typename TO>
__global__ void modulate_kernel(const float * __restrict__ h, const TS * __restrict__ s1, const TS * __restrict__ sh,
                                TO * __restrict__ out, const int64_t n, const int C) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int c = (int) (i % C);
        const float t = __fmul_rn(h[i], to_f<TS>(s1[c]));
        out[i] = from_f<TO>(__fadd_rn(t, to_f<TS>(sh[c])));
    }
}

template <typename TH, typename TG>
__global__ void residual_kernel(const float * __restrict__ x, const TH * __restrict__ h, const TG * __restrict__ gate,
                                float * __restrict__ out, const int64_t n, const int C) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int c = (int) (i % C);
        out[i] = __fadd_rn(x[i], __fmul_rn(to_f<TH>(h[i]), to_f<TG>(gate[c])));
    }
}

unsigned grid_for(const int64_t n, const int threads) {
    const int64_t want = (n + threads - 1) / threads;
    return (unsigned) (want < (int64_t) 65535 * 16 ? want : (int64_t) 65535 * 16);
}

// ---- qk norm + rope -------------------------------------------------------------------------------------------------
// One warp per (token, head) row of 128 values: lane l holds the values l, l + 32, l + 64, l + 96 - the four
// accumulators of torch's thread_reduce for that lane (vt0 = 4, step 32), whose squares (each the fma of one value
// with itself and 0: its square rounded once) are combined in order, then summed over the lanes by shuffles down at
// offsets 1, 2, 4, 8, 16 (block_x_reduce), the result in lane 0, its sqrt the norm.

template <typename TI, typename TG, typename TO, bool ROPE>
__global__ void qk_kernel(const TI * __restrict__ x, const TG * __restrict__ gamma, const float scale, const float eps,
                          const float2 * __restrict__ phases, TO * __restrict__ out, const int64_t rows, const int H,
                          const int64_t ld_t, const int64_t ld_h, const int64_t ld_ph) {
    const int lane = threadIdx.x % 32;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (r >= rows) {
        return;
    }
    const int64_t t = r / H;
    const int hd = (int) (r % H);
    const TI * src = x + t * ld_t + (int64_t) hd * ld_h;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = to_f<TI>(src[lane + 32 * j]);
    }
    float acc = __fmul_rn(v[0], v[0]);
    acc = __fadd_rn(acc, __fmul_rn(v[1], v[1]));
    acc = __fadd_rn(acc, __fmul_rn(v[2], v[2]));
    acc = __fadd_rn(acc, __fmul_rn(v[3], v[3]));
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const float other = __shfl_down_sync(0xffffffffu, acc, off);
        acc = __fadd_rn(acc, other);
    }
    float norm = __shfl_sync(0xffffffffu, sqrtf(acc), 0);     // sqrtf: IEEE (no fast math)
    norm = norm < eps ? eps : norm;                            // clamp_min(eps)
    TO * dst = out + r * 128;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = lane + 32 * j;
        float y = __fdiv_rn(v[j], norm);
        y = __fmul_rn(y, to_f<TG>(gamma[(int64_t) hd * 128 + e]));
        y = __fmul_rn(y, scale);
        y = round_to<TO>(y);                                   // .to(x_type)
        if (ROPE) {
            // view_as_complex: (value 2i, value 2i + 1) = (real, imaginary); lanes l and l ^ 1 hold a pair
            const float other = __shfl_xor_sync(0xffffffffu, y, 1);
            const float2 ph = phases[t * ld_ph + e / 2];
            const c10::complex<float> a = (e & 1) ? c10::complex<float>(other, y) : c10::complex<float>(y, other);
            const c10::complex<float> b(ph.x, ph.y);
            const c10::complex<float> z = a * b;                // torch's complex multiply, as its kernel has it
            y = (e & 1) ? z.imag() : z.real();
        }
        dst[e] = from_f<TO>(y);
    }
}

}  // namespace

cudaError_t quantize_rows(const QuantParams & p, cudaStream_t stream) {
    switch (p.dtype) {
        case DT_F32: return quant_dispatch<float>(p, stream);
        case DT_F16: return quant_dispatch<__half>(p, stream);
        case DT_BF16: return quant_dispatch<__nv_bfloat16>(p, stream);
        default: return cudaErrorInvalidValue;
    }
}

cudaError_t modulate(const void * h, const void * s1, const void * sh, void * out, const int64_t n, const int C,
                     const int s_dtype, const int out_dtype, cudaStream_t stream) {
    const int threads = 256;
    const unsigned grid = grid_for(n, threads);
    if (n == 0) return cudaSuccess;
#define P3K_MOD(TS, TO) modulate_kernel<TS, TO><<<grid, threads, 0, stream>>>( \
        static_cast<const float *>(h), static_cast<const TS *>(s1), static_cast<const TS *>(sh), static_cast<TO *>(out), n, C)
    if (s_dtype == DT_F16 && out_dtype == DT_F16) P3K_MOD(__half, __half);
    else if (s_dtype == DT_F32 && out_dtype == DT_F16) P3K_MOD(float, __half);
    else if (s_dtype == DT_F16 && out_dtype == DT_F32) P3K_MOD(__half, float);
    else if (s_dtype == DT_F32 && out_dtype == DT_F32) P3K_MOD(float, float);
    else if (s_dtype == DT_BF16 && out_dtype == DT_BF16) P3K_MOD(__nv_bfloat16, __nv_bfloat16);
    else if (s_dtype == DT_F32 && out_dtype == DT_BF16) P3K_MOD(float, __nv_bfloat16);
    else return cudaErrorInvalidValue;
#undef P3K_MOD
    return cudaGetLastError();
}

cudaError_t residual_gated(const void * x, const void * h, const void * gate, void * out, const int64_t n, const int C,
                           const int h_dtype, const int g_dtype, cudaStream_t stream) {
    const int threads = 256;
    const unsigned grid = grid_for(n, threads);
    if (n == 0) return cudaSuccess;
#define P3K_RES(TH, TG) residual_kernel<TH, TG><<<grid, threads, 0, stream>>>( \
        static_cast<const float *>(x), static_cast<const TH *>(h), static_cast<const TG *>(gate), static_cast<float *>(out), n, C)
    if (h_dtype == DT_F16 && g_dtype == DT_F16) P3K_RES(__half, __half);
    else if (h_dtype == DT_F16 && g_dtype == DT_F32) P3K_RES(__half, float);
    else if (h_dtype == DT_F32 && g_dtype == DT_F16) P3K_RES(float, __half);
    else if (h_dtype == DT_F32 && g_dtype == DT_F32) P3K_RES(float, float);
    else if (h_dtype == DT_BF16 && g_dtype == DT_BF16) P3K_RES(__nv_bfloat16, __nv_bfloat16);
    else return cudaErrorInvalidValue;
#undef P3K_RES
    return cudaGetLastError();
}

cudaError_t qk_norm_rope(const QkParams & p, cudaStream_t stream) {
    if (p.rows == 0) return cudaSuccess;
    const int threads = 256;
    const unsigned grid = (unsigned) ((p.rows + threads / 32 - 1) / (threads / 32));
    const bool rope = p.phases != nullptr;
#define P3K_QK(TI, TG, TO, R) qk_kernel<TI, TG, TO, R><<<grid, threads, 0, stream>>>( \
        static_cast<const TI *>(p.x), static_cast<const TG *>(p.gamma), p.scale, p.eps, p.phases, static_cast<TO *>(p.out), \
        p.rows, p.H, p.ld_t, p.ld_h, p.ld_ph)
#define P3K_QK_R(TI, TG, TO) if (rope) P3K_QK(TI, TG, TO, true); else P3K_QK(TI, TG, TO, false)
    if (p.x_dtype == DT_F16 && p.g_dtype == DT_F32 && p.out_dtype == DT_F16) { P3K_QK_R(__half, float, __half); }
    else if (p.x_dtype == DT_F16 && p.g_dtype == DT_F16 && p.out_dtype == DT_F16) { P3K_QK_R(__half, __half, __half); }
    else if (p.x_dtype == DT_F32 && p.g_dtype == DT_F32 && p.out_dtype == DT_F32) { P3K_QK_R(float, float, float); }
    else return cudaErrorInvalidValue;
#undef P3K_QK_R
#undef P3K_QK
    return cudaGetLastError();
}

}  // namespace pixal3d_kernels
