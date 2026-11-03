// The int8 GEMM with group scales as dp4a (4 int8 products a lane per instruction) for the cards without int8 tensor
// cores: Pascal sm_61 (GTX 10xx), also Volta (int8_gemm.h has the arithmetic).
//
// A port of the dp4a W8A8 GEMM of Spellcast3D's image engine (temp_code/ggml-kquants, w8a8-dp4a.cuh; its design was
// measured on a GTX 1070 / 1070 Ti: NOTES-w8a8-dp4a.md):
// - Block tile 128 rows x 64 channels, 128 threads, two blocks per SM; each thread 8 rows x 8 channels.
// - A K stage of 64 values sits in shared memory transposed into int32 words ([k4][row], [k4][channel]: 4 int8
//   values of K per word), so that a thread reads its 8 rows' and 8 channels' words of one k4 with two 16-byte loads
//   each and issues 64 dp4a with them. Columns XOR-swizzled (bit 2 by bit 5): the copies (a warp = 32 consecutive rows
//   of one 16-byte chunk) and the reads (a warp = 4 row groups x 8 channel groups) are free of bank conflicts.
// - The dp4a chains run through a whole group (or the whole K, per-row scales); a group's sums are scaled into the
//   fp32 accumulators once per 128 K (two stages): one FADD + one FFMA per output (int8_gemm.h), against the image
//   engine's I2F + FFMA per output and 32 K. The group's row / channel scales come with its second stage's copy.
// - Pascal's instruction cache (~8 KB for an unrolled loop, NOTES-w8a8-dp4a.md: a fully unrolled 17 KB stage ran at
//   13.6 T/s, the same work as a ~7 KB loop 18.9): a stage is a loop over two 8-k4 halves, each half unrolled.
// - Pipeline: no cp.async on Pascal; the next stage's global loads go to registers before the current stage's math
//   and to the other shared buffer after it, one barrier per stage.
// - Rows past M and channels past N read the last row / channel; their results are not stored.

#include "int8_gemm.h"
#include "int8_gemm_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
#define P3K_DP4A_DEVICE 1
#else
#define P3K_DP4A_DEVICE 0
#endif

namespace pixal3d_kernels {
namespace {

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int KR = BK / 4;                      // k4 rows of a stage
constexpr int NT = 2 * BN;                      // 128 threads
constexpr int CA = BM * BK / 16 / NT;           // 16-byte chunks per thread and stage: A (4: one row's 64 bytes)
constexpr int CB = BN * BK / 16 / NT;           // B (2)
static_assert(NT == BM, "a thread copies one row of A");

// swizzled column of word c in a [k4][128 or 64] shared tile
static __host__ __device__ __forceinline__ int swz(const int c) {
    return c ^ (((c >> 5) & 1) << 2);
}

// AG: xs holds one scale per row and group (else per row); BG: ws one per channel and group (else per channel)
template <bool AG, bool BG>
__global__ void __launch_bounds__(NT, 2) gemm_kernel(const Int8GemmParams p) {
#if P3K_DP4A_DEVICE
    constexpr bool GROUPED = AG || BG;
    __shared__ __align__(16) int   As[2][KR][BM];
    __shared__ __align__(16) int   Bs[2][KR][BN];
    __shared__ __align__(16) float Xs[2][BM];   // a group's row scales (AG), with its second stage
    __shared__ __align__(16) float Ws[2][BN];   // a group's channel scales (BG)
    const int M = p.M, N = p.N, K = p.K;

    int tm, tn;
    tile_order(blockIdx.x, (M + BM - 1) / BM, (N + BN - 1) / BN, p.raster, tm, tn);
    const int m0 = tm * BM;
    const int n0 = tn * BN;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int tg   = warp * 4 + lane / 8;       // row group: tile rows tg * 8 .. tg * 8 + 7
    const int cg   = lane % 8;                  // channel group: tile channels cg * 8 .. cg * 8 + 7
    const int pa   = swz(tg * 8);               // where the group's rows 0..3 are (4..7 at pa ^ 4)
    const int pb   = swz(cg * 8);
    const int nk   = K / BK;

    // copies: A chunk s of the thread = row tid, bytes 16 s .. 16 s + 15 of the stage (k4 rows 4 s .. 4 s + 3);
    // B chunk s = row tid % 64, chunk tid / 64 + 2 s. A warp = 32 consecutive rows of one 16-byte chunk.
    const int xr = min(m0 + tid, M - 1);
    const int8_t * gA = p.xq + (int64_t) xr * p.ldx;
    const int8_t * gB = p.wq + (int64_t) min(n0 + tid % BN, N - 1) * p.ldw + 16 * (tid / BN);
    const int ca = swz(tid);
    const int cb = swz(tid % BN);
    const int kb = 4 * (tid / BN);
    const float * gX = p.xs + (int64_t) xr * p.ldxs;            // AG: the row's group scales
    const float * gW = p.ws + min(n0 + tid % BN, N - 1);        // BG: the channel's, N apart

    int4 ra[CA], rb[CB];
    float rx = 0.0f, rw = 0.0f;
    auto gload = [&](const int kt) {
        const int64_t ko = (int64_t) kt * BK;
#pragma unroll
        for (int s = 0; s < CA; ++s) ra[s] = __ldg((const int4 *) (gA + ko + 16 * s));
#pragma unroll
        for (int s = 0; s < CB; ++s) rb[s] = __ldg((const int4 *) (gB + ko + 32 * s));
        if (GROUPED && (kt & 1)) {
            if (AG) rx = __ldg(gX + kt / 2);
            if (BG && tid < BN) rw = __ldg(gW + (int64_t) (kt / 2) * N);
        }
    };
    auto sstore = [&](const int buf, const int kt) {
#pragma unroll
        for (int s = 0; s < CA; ++s) {
            As[buf][4 * s + 0][ca] = ra[s].x; As[buf][4 * s + 1][ca] = ra[s].y;
            As[buf][4 * s + 2][ca] = ra[s].z; As[buf][4 * s + 3][ca] = ra[s].w;
        }
#pragma unroll
        for (int s = 0; s < CB; ++s) {
            const int k = kb + 8 * s;
            Bs[buf][k + 0][cb] = rb[s].x; Bs[buf][k + 1][cb] = rb[s].y;
            Bs[buf][k + 2][cb] = rb[s].z; Bs[buf][k + 3][cb] = rb[s].w;
        }
        if (GROUPED && (kt & 1)) {
            if (AG) Xs[buf][tid] = rx;
            if (BG && tid < BN) Ws[buf][tid] = rw;
        }
    };

    // the integer sums (from MAGIC per group when grouped, else from 0 over the whole K) and the fp32 accumulators
    int   pi[8][8];
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            pi[i][j]  = GROUPED ? MAGIC : 0;
            acc[i][j] = 0.0f;
        }

    gload(0);
    sstore(0, 0);
    __syncthreads();

    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) {
            gload(kt + 1);
        }
#pragma unroll 1
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int k4 = 0; k4 < 8; ++k4) {
                const int row = half * 8 + k4;
                const int4 a0 = *(const int4 *) &As[buf][row][pa];
                const int4 a1 = *(const int4 *) &As[buf][row][pa ^ 4];
                const int4 b0 = *(const int4 *) &Bs[buf][row][pb];
                const int4 b1 = *(const int4 *) &Bs[buf][row][pb ^ 4];
                const int fa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const int fb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) pi[i][j] = __dp4a(fa[i], fb[j], pi[i][j]);
            }
        }
        if (GROUPED && (kt & 1)) {
            float sx[8], sw[8];
            if constexpr (AG) {
                const float4 x0 = *(const float4 *) &Xs[buf][tg * 8];
                const float4 x1 = *(const float4 *) &Xs[buf][tg * 8 + 4];
                sx[0] = x0.x; sx[1] = x0.y; sx[2] = x0.z; sx[3] = x0.w;
                sx[4] = x1.x; sx[5] = x1.y; sx[6] = x1.z; sx[7] = x1.w;
            }
            if constexpr (BG) {
                const float4 w0 = *(const float4 *) &Ws[buf][cg * 8];
                const float4 w1 = *(const float4 *) &Ws[buf][cg * 8 + 4];
                sw[0] = w0.x; sw[1] = w0.y; sw[2] = w0.z; sw[3] = w0.w;
                sw[4] = w1.x; sw[5] = w1.y; sw[6] = w1.z; sw[7] = w1.w;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float s;
                    if constexpr (AG && BG) {
                        s = sx[i] * sw[j];
                    } else if constexpr (AG) {
                        s = sx[i];
                    } else {
                        s = sw[j];
                    }
                    add_group(acc[i][j], pi[i][j], s);
                }
        }
        if (kt + 1 < nk) {
            sstore(buf ^ 1, kt + 1);
        }
        __syncthreads();
    }

    // epilogue: row m0 + tg * 8 + i, channels n0 + cg * 8 + j (N % 16 == 0: a thread's 8 channels are all in or out)
    const int n = n0 + cg * 8;
    if (n >= N) {
        return;
    }
    float cs[8], cbias[8];
    {
        const float4 w0 = BG ? make_float4(1.f, 1.f, 1.f, 1.f) : __ldg((const float4 *) (p.ws + n));
        const float4 w1 = BG ? make_float4(1.f, 1.f, 1.f, 1.f) : __ldg((const float4 *) (p.ws + n + 4));
        const float4 b0 = p.bias ? __ldg((const float4 *) (p.bias + n)) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 b1 = p.bias ? __ldg((const float4 *) (p.bias + n + 4)) : make_float4(0.f, 0.f, 0.f, 0.f);
        cs[0] = w0.x; cs[1] = w0.y; cs[2] = w0.z; cs[3] = w0.w; cs[4] = w1.x; cs[5] = w1.y; cs[6] = w1.z; cs[7] = w1.w;
        cbias[0] = b0.x; cbias[1] = b0.y; cbias[2] = b0.z; cbias[3] = b0.w;
        cbias[4] = b1.x; cbias[5] = b1.y; cbias[6] = b1.z; cbias[7] = b1.w;
    }
    // the values of 4 of the thread's rows at a time go through its own slots of the A tile (free after the K loop's
    // last barrier; float4 slot (r * 2 + half) * NT + tid: conflict-free), then GELU / the output dtype / the stores
    // in a loop that is not unrolled (store8: one copy of that code per pass)
    float4 * stage = reinterpret_cast<float4 *>(&As[0][0][0]);
    const bool gelu = p.gelu != 0;
    const int out = p.out;

#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int i = pass * 4 + r;
            const float rs = AG ? 1.0f : __ldg(p.xs + min(m0 + tg * 8 + i, M - 1));
            float y[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (out == OUT_I32) {
                    y[j] = __int_as_float(pi[i][j]);
                } else if constexpr (AG && BG) {
                    y[j] = acc[i][j] + cbias[j];
                } else if constexpr (AG) {
                    y[j] = acc[i][j] * cs[j] + cbias[j];
                } else if constexpr (BG) {
                    y[j] = acc[i][j] * rs + cbias[j];
                } else {
                    y[j] = ((float) pi[i][j] * rs) * cs[j] + cbias[j];
                }
            }
            stage[(r * 2 + 0) * NT + tid] = make_float4(y[0], y[1], y[2], y[3]);
            stage[(r * 2 + 1) * NT + tid] = make_float4(y[4], y[5], y[6], y[7]);
        }
#pragma unroll 1
        for (int r = 0; r < 4; ++r) {
            const int t = m0 + tg * 8 + pass * 4 + r;
            if (t < M) {
                const float4 a = stage[(r * 2 + 0) * NT + tid];
                const float4 b = stage[(r * 2 + 1) * NT + tid];
                float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                store8(p.y, (int64_t) t * N + n, v, out, gelu);
            }
        }
    }
#else
    (void) p;
#endif
}

template <bool AG, bool BG>
cudaError_t launch(const Int8GemmParams & p, cudaStream_t stream) {
    const int ntm = (p.M + BM - 1) / BM;
    const int ntn = (p.N + BN - 1) / BN;
    gemm_kernel<AG, BG><<<dim3((unsigned) (ntm * ntn)), NT, 0, stream>>>(p);
    return cudaGetLastError();
}

}  // namespace

bool int8_gemm_dp4a_loaded() {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, gemm_kernel<false, false>) != cudaSuccess) {
        cudaGetLastError();     // no image for this card: not loaded (and no error left behind)
        return false;
    }
    return a.ptxVersion >= 61;
}

cudaError_t int8_gemm_dp4a(const Int8GemmParams & p, const bool a_group, const bool b_group, cudaStream_t stream) {
    if (p.K % (2 * BK) || (p.out == OUT_I32 && (a_group || b_group))) {
        return cudaErrorInvalidValue;
    }
    Int8GemmParams q = p;
    if (q.raster == 0) {
        q.raster = -4;      // the image engine's dp4a choice below N = 16,384
    }
    if (a_group) {
        return b_group ? launch<true, true>(q, stream) : launch<true, false>(q, stream);
    }
    return b_group ? launch<false, true>(q, stream) : launch<false, false>(q, stream);
}

}  // namespace pixal3d_kernels
