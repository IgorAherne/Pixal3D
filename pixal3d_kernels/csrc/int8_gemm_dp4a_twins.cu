// Spellcast3D round O: twins of the dp4a GEMM (int8_gemm_dp4a.cu) for Pascal (sm_61), with another code shape and
// the shipped kernel's bits. The shipped group-scaled kernels' K-stage loop is 8,832 bytes of SASS (gemm_kernel<true,
// false>, the hr / tex flows' qkv / fc1 / fc2 layers), above Pascal's ~8 KB instruction cache for a loop (the image
// engine's dp4a GEMM on a GTX 1070 Ti: a 17 KB loop ran at 43 % of the card's dp4a rate, a ~7 KB one at 63 %); its
// per-group conversion spends 64 moves restarting the integer sums at 1.5 * 2^23. The twins:
//
//   q4  a stage as 4 trips of 4 k4 rows (shipped: 2 trips of 8): the stage loop about half the size;
//   f5  a stage as one k4 row, then 3 trips of 5; the first k4 row of a group (the even stage) takes 1.5 * 2^23 as
//       its dp4a's addend instead of the sum (two copies of that one row, chosen by the stage's parity), so the
//       group's conversion needs no restart: 64 moves less a group, the stage loop below 8 KB.
// (sm_61 SASS, CUDA 12.8: the group kernel <true, false>'s stage loop 8,832 B shipped, 5,920 B q4, 7,648 B f5;
// issue slots a 128-K group ~2,581 shipped, ~2,647 q4 (the 4-row trips' overhead), ~2,585 f5. A restart with 5 trips
// of 3 rows was 6,192 B for q4's issue count: not kept.)
//
// Every output's integer sum is the same exact int32 chain (integer additions in any order give the same bits, and
// the chain is the shipped one: rows in order, from 1.5 * 2^23 for each group), converted and scaled into its fp32
// accumulator by the same instruction (add_group's __fmaf_rn of (sum - 1.5 * 2^23) and the group's scale) in the
// same group order; the epilogue is the shipped one. So y is the shipped kernel's bit for bit (checked: torch.equal
// on random tables and the flows' shapes: code/devtools/pixal3d-lowmem/round-o/pascal-kernels/check_gemm.py).
// Which serves on a card is the install's race (pixal3d/modules/int8mm.py: DP4A_TWINS); the shipped kernel serves
// without a record. The per-row kernel (gemm_kernel<false, false>, 6.6 KB, no group conversion) is twinned as q4 in
// both (no restart to fold).

#include "int8_gemm.h"
#include "int8_gemm_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
#define P3K_DP4A_TWIN_DEVICE 1
#else
#define P3K_DP4A_TWIN_DEVICE 0
#endif

namespace pixal3d_kernels {
namespace {

constexpr int BM = 128, BN = 64, BK = 64;
constexpr int KR = BK / 4;                      // k4 rows of a stage
constexpr int NT = 2 * BN;                      // 128 threads
constexpr int CA = BM * BK / 16 / NT;           // 16-byte chunks per thread and stage: A (4)
constexpr int CB = BN * BK / 16 / NT;           // B (2)
static_assert(NT == BM, "a thread copies one row of A");

static __host__ __device__ __forceinline__ int swz(const int c) {
    return c ^ (((c >> 5) & 1) << 2);
}

// TWIN: 1 = q4, 2 = f5 (the comment above)
template <bool AG, bool BG, int TWIN>
__global__ void __launch_bounds__(NT, 2) gemm_twin_kernel(const Int8GemmParams p) {
#if P3K_DP4A_TWIN_DEVICE
    constexpr bool GROUPED = AG || BG;
    constexpr bool FOLD = GROUPED && TWIN == 2;
    constexpr int TRIP = 5;                     // FOLD: rows a trip after the first (15 = 3 x 5)
    __shared__ __align__(16) int   As[2][KR][BM];
    __shared__ __align__(16) int   Bs[2][KR][BN];
    __shared__ __align__(16) float Xs[2][BM];
    __shared__ __align__(16) float Ws[2][BN];
    const int M = p.M, N = p.N, K = p.K;

    int tm, tn;
    tile_order(blockIdx.x, (M + BM - 1) / BM, (N + BN - 1) / BN, p.raster, tm, tn);
    const int m0 = tm * BM;
    const int n0 = tn * BN;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int tg   = warp * 4 + lane / 8;
    const int cg   = lane % 8;
    const int pa   = swz(tg * 8);
    const int pb   = swz(cg * 8);
    const int nk   = K / BK;

    const int xr = min(m0 + tid, M - 1);
    const int8_t * gA = p.xq + (int64_t) xr * p.ldx;
    const int8_t * gB = p.wq + (int64_t) min(n0 + tid % BN, N - 1) * p.ldw + 16 * (tid / BN);
    const int ca = swz(tid);
    const int cb = swz(tid % BN);
    const int kb = 4 * (tid / BN);
    const float * gX = p.xs + (int64_t) xr * p.ldxs;
    const float * gW = p.ws + min(n0 + tid % BN, N - 1);

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

    int   pi[8][8];
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            pi[i][j]  = GROUPED ? MAGIC : 0;
            acc[i][j] = 0.0f;
        }

    // one k4 row of the stage: 64 dp4a into the sums, or (fresh) from 1.5 * 2^23: the group's first row
    auto k4row = [&](const int buf, const int row, const bool fresh) {
        const int4 a0 = *(const int4 *) &As[buf][row][pa];
        const int4 a1 = *(const int4 *) &As[buf][row][pa ^ 4];
        const int4 b0 = *(const int4 *) &Bs[buf][row][pb];
        const int4 b1 = *(const int4 *) &Bs[buf][row][pb ^ 4];
        const int fa[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const int fb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 8; ++j) pi[i][j] = __dp4a(fa[i], fb[j], fresh ? MAGIC : pi[i][j]);
    };

    gload(0);
    sstore(0, 0);
    __syncthreads();

    for (int kt = 0; kt < nk; ++kt) {
        const int buf = kt & 1;
        if (kt + 1 < nk) {
            gload(kt + 1);
        }
        if constexpr (FOLD) {
            if (kt & 1) {
                k4row(buf, 0, false);
            } else {
                k4row(buf, 0, true);        // the group's first row: its sums restart here
            }
#pragma unroll 1
            for (int t = 0; t < 15 / TRIP; ++t) {
#pragma unroll
                for (int k4 = 0; k4 < TRIP; ++k4) k4row(buf, 1 + t * TRIP + k4, false);
            }
        } else {
#pragma unroll 1
            for (int t = 0; t < 4; ++t) {
#pragma unroll
                for (int k4 = 0; k4 < 4; ++k4) k4row(buf, t * 4 + k4, false);
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
                    if constexpr (FOLD) {
                        // add_group's arithmetic; the next group's first row restarts the sum (k4row fresh)
                        acc[i][j] = __fmaf_rn(__int_as_float(pi[i][j]) - MAGIC_F, s, acc[i][j]);
                    } else {
                        add_group(acc[i][j], pi[i][j], s);
                    }
                }
        }
        if (kt + 1 < nk) {
            sstore(buf ^ 1, kt + 1);
        }
        __syncthreads();
    }

    // the shipped epilogue (int8_gemm_dp4a.cu)
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

template <bool AG, bool BG, int TWIN>
cudaError_t launch_twin(const Int8GemmParams & p, cudaStream_t stream) {
    const int ntm = (p.M + BM - 1) / BM;
    const int ntn = (p.N + BN - 1) / BN;
    gemm_twin_kernel<AG, BG, TWIN><<<dim3((unsigned) (ntm * ntn)), NT, 0, stream>>>(p);
    return cudaGetLastError();
}

template <int TWIN>
cudaError_t dispatch_twin(const Int8GemmParams & q, const bool a_group, const bool b_group, cudaStream_t stream) {
    if (a_group) {
        return b_group ? launch_twin<true, true, TWIN>(q, stream) : launch_twin<true, false, TWIN>(q, stream);
    }
    // the per-row kernel has no restart to fold: q4's shape for both twins
    return b_group ? launch_twin<false, true, TWIN>(q, stream) : launch_twin<false, false, 1>(q, stream);
}

}  // namespace

bool int8_gemm_dp4a_twins_loaded() {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, gemm_twin_kernel<true, false, 1>) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return a.ptxVersion >= 61;
}

cudaError_t int8_gemm_dp4a_twin(const Int8GemmParams & p, const bool a_group, const bool b_group, const int twin,
                                cudaStream_t stream) {
    if (p.K % (2 * BK) || (p.out == OUT_I32 && (a_group || b_group)) || (twin != 1 && twin != 2)) {
        return cudaErrorInvalidValue;
    }
    Int8GemmParams q = p;
    if (q.raster == 0) {
        q.raster = -4;      // the shipped kernel's tile order (int8_gemm_dp4a)
    }
    return twin == 1 ? dispatch_twin<1>(q, a_group, b_group, stream) : dispatch_twin<2>(q, a_group, b_group, stream);
}

}  // namespace pixal3d_kernels
