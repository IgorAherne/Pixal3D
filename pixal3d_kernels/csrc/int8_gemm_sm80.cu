// The group-scaled W8A8 GEMM on sm_80+ (Ampere, Ada, Hopper, Blackwell): int8_gemm_sm80.h has the arithmetic.
//
// - mma.sync.m16n8k32 (s8), fragments by ldmatrix.x4 from XOR-swizzled shared tiles (conflict-free both ways),
//   cp.async (L2 128-byte prefetch) in a ring of STAGES slots, one barrier per K stage; 4 warps and 48 KB a block, two
//   blocks per SM (one block's barriers, conversions and epilogue overlap the other's products). On the RTX 3060 the
//   four-warp tiles reach 100 TOPS on fc2's shape without their global copies, 66-69 with them (the per-row Triton
//   kernel's 128 x 128 tiles move a third fewer bytes per product: 86); one block of 8 warps per SM, two 64 x 32 halves
//   of a 64 x 64 warp tile, and fragments double-buffered across stages were all slower (round N's study).
// - A 128-K slab's integer sum accumulates in the MMAs from the bits of 12,582,912.0f (the first MMA of a slab takes
//   them as its C operand), so the sum read as a float minus 12,582,912 is exact (|sum| <= 128 * 127^2 < 2^22): one
//   FADD, then one FFMA into the fp32 accumulator with the row's scale of that slab - the instructions Triton's kernel
//   runs (it folds its `part + 0x4B400000` into the MMA the same way). Each tile's conversion comes just before its
//   first product of the next slab.
// - The epilogue's fp32 arithmetic is Triton's PTX instruction for instruction: fma.rn (scale and bias), mul.rn /
//   add.rn where it has an unfused mul / add, libdevice's tanhf as Triton links it (its ftz forms: abs.ftz,
//   fma.rn.ftz, ex2.approx.ftz, rcp.approx.ftz; branch-free here, the same value from the same branch's instructions),
//   tl.div_rn as div.rn.ftz, rint as cvt.rni. Outputs leave through shared memory in 16-byte stores.
// - Rows past M and channels past N read the last row / channel; their results are not stored.

#include "int8_gemm_sm80.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define P3K_SM80_DEVICE 1
#else
#define P3K_SM80_DEVICE 0
#endif

namespace pixal3d_kernels {
namespace {

constexpr int G128 = 128;                   // input channels per activation scale (and output group of int8 out)
constexpr int MAGIC = 0x4B400000;           // the bits of 12,582,912.0f = 1.5 * 2^23
constexpr float MAGIC_F = 12582912.0f;

template <int BM_, int BN_, int WM_, int WN_, int BK_, int STAGES_>
struct Tile {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, BK = BK_, STAGES = STAGES_, MINB = 2;
    static constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN, NT = WARPS_M * WARPS_N * 32;
    static constexpr int MF = WM / 16, NF = WN / 8;
    static constexpr int A_BYTES = BM * BK, STAGE = (BM + BN) * BK, PIPE = STAGES * STAGE;
    static constexpr int LDS16 = BN + 8;                    // staged 2-byte outputs per row (16 bytes of padding)
    static constexpr int LDS8 = BN + 16;                    // staged int8 outputs per row
    static constexpr int EPI16 = BM * LDS16 * 2;
    static constexpr int EPI8 = BM * LDS8 + BM * WARPS_N * 4;
    static constexpr int PIPE16 = PIPE + 16;                // + -12,582,912.0f in shared memory
    static constexpr int SMEM = PIPE16 > EPI16 ? (PIPE16 > EPI8 ? PIPE16 : EPI8) : (EPI16 > EPI8 ? EPI16 : EPI8);
    static_assert(NT == 128, "4 warps a block, two blocks per SM");
    static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0, "warp tiles of m16 x n16 pieces");
    static_assert(BK == 64 || BK == 128, "a K stage is half a slab or a slab");
    static_assert((NT / (BK / 16)) % 8 == 0, "a thread's copy rows share their swizzle");
    static_assert((BM * BK / 16) % NT == 0 && (BN * BK / 16) % NT == 0, "copies spread evenly");
};

// the tile table (int8_gemm_sm80_info); the race at install picks one per layer form
using T0 = Tile<128, 64, 64, 32, 64, 4>;        // 48 KB
using T1 = Tile<64, 128, 64, 32, 64, 4>;        // a row of 4 warps across 128 channels
using T2 = Tile<64, 128, 32, 64, 64, 4>;        // 2 x 2 warps of 32 x 64
using T3 = Tile<64, 128, 64, 32, 64, 3>;        // T1 in three stages (36 KB)
using T4 = Tile<64, 128, 32, 64, 128, 2>;       // T2 in whole-slab stages, double-buffered
using T5 = Tile<128, 64, 64, 32, 128, 2>;       // T0 in whole-slab stages
using T6 = Tile<64, 128, 64, 32, 128, 2>;       // T1 in whole-slab stages
constexpr int NTILES = 7;

#if P3K_SM80_DEVICE
__device__ __forceinline__ unsigned smem_u32(const void * p) {
    return (unsigned) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(const unsigned s, const void * g) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" : : "r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" : : : "memory");
}
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
}
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], const unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}
// d = a x b + c, one m16n8k32 int8 product (Triton's instruction, satfinite included; a slab's sum never nears it)
__device__ __forceinline__ void mma16832(int (&d)[4], const unsigned (&a)[4], const unsigned b0, const unsigned b1,
                                         const int c0, const int c1, const int c2, const int c3) {
    asm("mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

// the .ftz instructions of libdevice's tanhf as Triton links it, and of its div_rn
__device__ __forceinline__ float fma_ftz(const float a, const float b, const float c) {
    float d;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}
__device__ __forceinline__ float abs_ftz(const float a) {
    float d;
    asm("abs.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float ex2_ftz(const float a) {
    float d;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float rcp_ftz(const float a) {
    float d;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}
__device__ __forceinline__ float div_ftz(const float a, const float b) {
    float d;
    asm("div.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

// libdevice's __nv_tanhf (Triton's link): |x| < 0.6 (or NaN): an odd polynomial; else 1 - 2 / (2^(2|x| log2 e) + 1)
// with the sign of x, 1 from |x| >= 9.0109; both sides computed, the branch's own value kept
__device__ __forceinline__ float tanh_tri(const float x) {
    const float ax = abs_ftz(x);
    const float x2 = __fmul_rn(x, x);
    float p = fma_ftz(__int_as_float(0x3C80F082), x2, __int_as_float(0xBD563CAE));
    p = fma_ftz(p, x2, __int_as_float(0x3E085941));
    p = fma_ftz(p, x2, __int_as_float(0xBEAAA9ED));
    p = fma_ftz(p, x2, 0.0f);
    const float small = fma_ftz(p, x, x);
    const float e = ex2_ftz(__fmul_rn(ax, __int_as_float(0x4038AA3B)));
    float s = fma_ftz(rcp_ftz(__fadd_rn(e, 1.0f)), -2.0f, 1.0f);
    s = ax >= __int_as_float(0x41102CB4) ? 1.0f : s;
    const float big = __int_as_float((__float_as_int(x) & int(0x80000000)) | __float_as_int(s));
    return !(ax >= __int_as_float(0x3F19999A)) ? small : big;
}

// int8mm's GELU: inner = 0.79788456 * (y + 0.044715 * y^3) (the add fused with the 0.044715 product, as LLVM
// contracts it), 0.5 * y * (1 + tanh(inner))
__device__ __forceinline__ float gelu_tri(const float y) {
    const float y3 = __fmul_rn(y, __fmul_rn(y, y));
    const float u = __fmul_rn(__fmaf_rn(y3, __int_as_float(0x3D372713), y), __int_as_float(0x3F4C422A));
    return __fmul_rn(__fmul_rn(y, 0.5f), __fadd_rn(tanh_tri(u), 1.0f));
}

__device__ __forceinline__ float bf16r(const float x) {            // x rounded to bf16 and back (nearest even)
    return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ unsigned pack_bf16(const float a, const float b) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}
__device__ __forceinline__ unsigned pack_f16(const float a, const float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned *>(&h);
}

// byte offset of 16-byte chunk c of row r in a [rows][BK] int8 tile, chunks XOR-swizzled by the row (period 8 rows)
template <int BK>
__device__ __forceinline__ int swz(const int r, const int c) {
    if constexpr (BK == 64) {
        return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
    } else {
        return r * 128 + ((c ^ (r & 7)) << 4);
    }
}
#endif

template <class T>
__global__ void __launch_bounds__(T::NT, T::MINB) group_gemm_kernel(const GroupGemmParams p) {
#if P3K_SM80_DEVICE
    constexpr int BM = T::BM, BN = T::BN, WM = T::WM, WN = T::WN, BK = T::BK, STAGES = T::STAGES, NT = T::NT;
    constexpr int MF = T::MF, NF = T::NF, KPS = BK / 32, SPS = G128 / BK;
    extern __shared__ __align__(128) uint8_t smem[];
    const int M = p.M, N = p.N, K = p.K;

    // Triton's tile order: groups of group_m row tiles, the column tiles walked within a group
    const int ntm = (M + BM - 1) / BM, ntn = (N + BN - 1) / BN;
    const int per_grp = p.group_m * ntn;
    const int first_m = (int) (blockIdx.x / per_grp) * p.group_m;
    const int gsize = min(ntm - first_m, p.group_m);
    const int in_grp = (int) (blockIdx.x % per_grp);
    const int m0 = (first_m + in_grp % gsize) * BM;
    const int n0 = (in_grp / gsize) * BN;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % T::WARPS_M, wn = warp / T::WARPS_M;
    const int gid = lane >> 2, tig = lane & 3;
    const unsigned sbase = smem_u32(smem);
    // the conversions' constant -12,582,912.0f, read from shared memory after the stage's barrier: they then cannot
    // be scheduled above that barrier, where they would wait for the slab's last products to drain
    float * negm_slot = reinterpret_cast<float *>(smem + T::PIPE);
    if (tid == 0) *negm_slot = -MAGIC_F;

    // copies: thread tid takes 16-byte chunk tid % CPR of rows tid / CPR + j * RSTEP (one swizzle for all its rows)
    constexpr int CPR = BK / 16, RSTEP = NT / CPR;
    constexpr int ACP = BM / RSTEP, BCP = BN / RSTEP;
    const int crow = tid / CPR, cchunk = tid % CPR;
    const unsigned cp_off = (unsigned) swz<BK>(crow, cchunk);
    const int8_t * ag[ACP];
    const int8_t * bg[BCP];
#pragma unroll
    for (int j = 0; j < ACP; ++j) ag[j] = p.xq + (int64_t) min(m0 + crow + j * RSTEP, M - 1) * p.ldx + cchunk * 16;
#pragma unroll
    for (int j = 0; j < BCP; ++j) bg[j] = p.wq + (int64_t) min(n0 + crow + j * RSTEP, N - 1) * p.ldw + cchunk * 16;
    auto load_stage = [&](const int slot, const int kt) {
        const unsigned s = sbase + slot * T::STAGE + cp_off;
        const int koff = kt * BK;
#pragma unroll
        for (int j = 0; j < ACP; ++j) cp_async16(s + j * RSTEP * BK, ag[j] + koff);
#pragma unroll
        for (int j = 0; j < BCP; ++j) cp_async16(s + T::A_BYTES + j * RSTEP * BK, bg[j] + koff);
    };

    // ldmatrix offsets per k32 step: A rows lane & 15, chunk 2 ks + lane / 16 (a0..a3 of the m16 fragment); B rows
    // (lane & 7) + 8 (lane / 16), chunk 2 ks + (lane / 8) % 2 (b0 / b1 of two n8 fragments); the m16 / n16 pieces are
    // whole multiples of 16 rows away (the same swizzle)
    unsigned a_off[KPS], b_off[KPS];
#pragma unroll
    for (int ks = 0; ks < KPS; ++ks) {
        a_off[ks] = (unsigned) swz<BK>(wm * WM + (lane & 15), 2 * ks + (lane >> 4));
        b_off[ks] = (unsigned) (T::A_BYTES + swz<BK>(wn * WN + (lane & 7) + ((lane >> 4) << 3),
                                                      2 * ks + ((lane >> 3) & 1)));
    }

    // the thread's rows (wm WM + mf 16 + gid [+ 8]) for the slab scales, clamped
    const float * sxp[MF][2];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int h = 0; h < 2; ++h)
            sxp[mf][h] = p.xs + (int64_t) min(m0 + wm * WM + mf * 16 + gid + 8 * h, M - 1) * p.ldxs;

    // the slab's integer sums (from MAGIC), the fp32 accumulators, the slab's row scales. Before the first slab the
    // sums read 0 and the scales are 0: its "conversion" adds fma(0, 0, 0) = +0 to +0, which changes nothing.
    int part[MF][NF][4];
    float acc[MF][NF][4];
    float sx[MF][2];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        sx[mf][0] = sx[mf][1] = 0.0f;
#pragma unroll
        for (int nf = 0; nf < NF; ++nf)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                part[mf][nf][i] = MAGIC;
                acc[mf][nf][i] = 0.0f;
            }
    }

    // STAGES - 1 stages in flight; per stage: wait, barrier, the next copy, its fragments and products. At a slab's
    // first k32 step each tile's previous-slab sums go into its accumulators just before its first product, and the
    // slab's row scales are read (for its own conversion, at the next slab)
    const int nk = K / BK, ng = K / G128;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nk) load_stage(s, s);
        cp_commit();
    }
    int kt = 0;
    for (int g = 0; g < ng; ++g) {
#pragma unroll
        for (int h = 0; h < SPS; ++h, ++kt) {
            cp_wait<STAGES - 2>();
            __syncthreads();
            {
                const int kn = kt + STAGES - 1;
                if (kn < nk) load_stage(kn % STAGES, kn);
                cp_commit();
            }
            const unsigned slot = sbase + (kt % STAGES) * T::STAGE;
#pragma unroll
            for (int ks = 0; ks < KPS; ++ks) {
                const bool first = h == 0 && ks == 0;
                unsigned fa[MF][4], fb[NF / 2][4];
#pragma unroll
                for (int mf = 0; mf < MF; ++mf) ldsm4(fa[mf], slot + a_off[ks] + mf * 16 * BK);
#pragma unroll
                for (int np = 0; np < NF / 2; ++np) ldsm4(fb[np], slot + b_off[ks] + np * 16 * BK);
                const float negm = first ? *static_cast<volatile float *>(negm_slot) : 0.0f;
#pragma unroll
                for (int mf = 0; mf < MF; ++mf)
#pragma unroll
                    for (int nf = 0; nf < NF; ++nf) {
                        const unsigned b0 = fb[nf / 2][(nf & 1) * 2], b1 = fb[nf / 2][(nf & 1) * 2 + 1];
                        int (&c)[4] = part[mf][nf];
                        if (first) {
#pragma unroll
                            for (int i = 0; i < 4; ++i)
                                acc[mf][nf][i] = __fmaf_rn(__fadd_rn(__int_as_float(c[i]), negm), sx[mf][i >> 1],
                                                           acc[mf][nf][i]);
                            mma16832(c, fa[mf], b0, b1, MAGIC, MAGIC, MAGIC, MAGIC);
                        } else {
                            mma16832(c, fa[mf], b0, b1, c[0], c[1], c[2], c[3]);
                        }
                    }
                if (first) {
#pragma unroll
                    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
                        for (int r = 0; r < 2; ++r) sx[mf][r] = __ldg(sxp[mf][r] + g);
                }
            }
        }
    }
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int nf = 0; nf < NF; ++nf)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                acc[mf][nf][i] = __fmaf_rn(__fadd_rn(__int_as_float(part[mf][nf][i]), -MAGIC_F), sx[mf][i >> 1],
                                           acc[mf][nf][i]);
    cp_wait<0>();
    __syncthreads();                        // the pipeline's shared memory is the epilogue's from here

    // y = acc * ws + bias (one fma), [GELU]
#pragma unroll
    for (int nf = 0; nf < NF; ++nf) {
        const int col = n0 + wn * WN + nf * 8 + 2 * tig;
        const int c = col < N ? col : 0;
        const float2 cs = *reinterpret_cast<const float2 *>(p.ws + c);
        float2 bb = make_float2(0.0f, 0.0f);
        if (p.bias) bb = *reinterpret_cast<const float2 *>(p.bias + c);
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float w = (i & 1) ? cs.y : cs.x;
                float y = p.bias ? __fmaf_rn(acc[mf][nf][i], w, (i & 1) ? bb.y : bb.x) : __fmul_rn(acc[mf][nf][i], w);
                if (p.gelu) y = gelu_tri(y);
                acc[mf][nf][i] = y;
            }
    }

    if (p.out == 2) {
        if constexpr (BN % G128 == 0 && G128 % WN == 0) {
        // int8 with one scale per row and 128 columns: the row's |y| maximum over its group (the quad's lanes, then
        // the group's warps through shared memory), floored at 1e-30; q = rint(y * (127 / amax)) clamped to +-127,
        // scale = amax / 127
        constexpr int WPG = G128 / WN;                              // warps across a group's 128 columns
        int8_t * st = reinterpret_cast<int8_t *>(smem);
        float * red = reinterpret_cast<float *>(smem + BM * T::LDS8);
        float amax[MF][2];
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float m = 0.0f;
#pragma unroll
                for (int nf = 0; nf < NF; ++nf) {
                    m = fmaxf(m, fabsf(acc[mf][nf][2 * h]));
                    m = fmaxf(m, fabsf(acc[mf][nf][2 * h + 1]));
                }
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                amax[mf][h] = m;
                if (WPG > 1 && tig == 0) red[(wm * WM + mf * 16 + gid + 8 * h) * T::WARPS_N + wn] = m;
            }
        if (WPG > 1) {
            __syncthreads();
            const int w0 = (wn / WPG) * WPG;
#pragma unroll
            for (int mf = 0; mf < MF; ++mf)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float * r = red + (wm * WM + mf * 16 + gid + 8 * h) * T::WARPS_N + w0;
                    float m = r[0];
#pragma unroll
                    for (int w = 1; w < WPG; ++w) m = fmaxf(m, r[w]);
                    amax[mf][h] = m;
                }
        }
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const float a = fmaxf(amax[mf][h], 1e-30f);
                const float inv = div_ftz(127.0f, a);
                const int row = wm * WM + mf * 16 + gid + 8 * h;
#pragma unroll
                for (int nf = 0; nf < NF; ++nf) {
                    const float q0 = fminf(fmaxf(rintf(__fmul_rn(acc[mf][nf][2 * h], inv)), -127.0f), 127.0f);
                    const float q1 = fminf(fmaxf(rintf(__fmul_rn(acc[mf][nf][2 * h + 1], inv)), -127.0f), 127.0f);
                    const unsigned short v = (unsigned short) (((unsigned) (int) q0 & 0xffu) |
                                                               (((unsigned) (int) q1 & 0xffu) << 8));
                    *reinterpret_cast<unsigned short *>(st + row * T::LDS8 + wn * WN + nf * 8 + 2 * tig) = v;
                }
                const int gm = m0 + row;
                if (tig == 0 && wn % WPG == 0 && gm < M) {
                    const int gcol = n0 + wn * WN;
                    if (gcol < N) p.qs[(int64_t) gm * p.ldqs + gcol / G128] = div_ftz(a, 127.0f);
                }
            }
        __syncthreads();
        constexpr int CH = BN / 16;
        int8_t * y = static_cast<int8_t *>(p.y);
        for (int c = tid; c < BM * CH; c += NT) {
            const int row = c / CH, cc = (c % CH) * 16;
            const int gm = m0 + row, gn = n0 + cc;
            if (gm < M && gn < N)
                *reinterpret_cast<uint4 *>(y + (int64_t) gm * p.ldy + gn) =
                    *reinterpret_cast<const uint4 *>(st + row * T::LDS8 + cc);
        }
        }
        return;                             // (tiles without whole 128-column groups are refused int8 out by the host)
    }

    // 2-byte outputs: staged, then 8 columns a thread in 16-byte stores (the residual applied there)
    unsigned short * st = reinterpret_cast<unsigned short *>(smem);
    const bool f16 = p.out == 3;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int nf = 0; nf < NF; ++nf)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = wm * WM + mf * 16 + gid + 8 * h, col = wn * WN + nf * 8 + 2 * tig;
                const float a = acc[mf][nf][2 * h], b = acc[mf][nf][2 * h + 1];
                *reinterpret_cast<unsigned *>(st + row * T::LDS16 + col) = f16 ? pack_f16(a, b) : pack_bf16(a, b);
            }
    __syncthreads();
    constexpr int CH = BN / 8;
    unsigned short * y = static_cast<unsigned short *>(p.y);
    for (int c = tid; c < BM * CH; c += NT) {
        const int row = c / CH, cc = (c % CH) * 8;
        const int gm = m0 + row, gn = n0 + cc;
        if (gm >= M || gn >= N) continue;
        uint4 v = *reinterpret_cast<const uint4 *>(st + row * T::LDS16 + cc);
        const int64_t off = (int64_t) gm * p.ldy + gn;
        if (p.resid) {
            // fused.residual's rounded mode on h, the bf16 result: x + bf16(h * gate) / x + bf16(second + h) / x + h
            const uint4 xv = *reinterpret_cast<const uint4 *>(static_cast<const unsigned short *>(p.rx) + off);
            uint4 tv = make_uint4(0, 0, 0, 0);
            if (p.resid == 1) tv = *reinterpret_cast<const uint4 *>(static_cast<const unsigned short *>(p.rg) + gn);
            if (p.resid == 2) tv = *reinterpret_cast<const uint4 *>(static_cast<const unsigned short *>(p.r2) + off);
            const unsigned * hv = &v.x;
            const unsigned * xw = &xv.x;
            const unsigned * tw = &tv.x;
            unsigned o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float hh[2], xx[2], tt[2];
                hh[0] = __uint_as_float(hv[j] << 16);
                hh[1] = __uint_as_float(hv[j] & 0xffff0000u);
                xx[0] = __uint_as_float(xw[j] << 16);
                xx[1] = __uint_as_float(xw[j] & 0xffff0000u);
                tt[0] = __uint_as_float(tw[j] << 16);
                tt[1] = __uint_as_float(tw[j] & 0xffff0000u);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    if (p.resid == 1) hh[e] = bf16r(__fmul_rn(hh[e], tt[e]));
                    if (p.resid == 2) hh[e] = bf16r(__fadd_rn(tt[e], hh[e]));
                }
                o[j] = pack_bf16(__fadd_rn(xx[0], hh[0]), __fadd_rn(xx[1], hh[1]));
            }
            v = make_uint4(o[0], o[1], o[2], o[3]);
        }
        *reinterpret_cast<uint4 *>(y + off) = v;
    }
#endif
}

template <class T>
cudaError_t launch(const GroupGemmParams & p, cudaStream_t stream) {
    const auto kernel = group_gemm_kernel<T>;
    static bool attr_set[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !attr_set[dev]) {
        const cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM);
        if (e != cudaSuccess) return e;
        attr_set[dev] = true;
    }
    const long long tiles = (long long) ((p.M + T::BM - 1) / T::BM) * ((p.N + T::BN - 1) / T::BN);
    if (tiles <= 0) return cudaSuccess;
    kernel<<<(unsigned) tiles, T::NT, T::SMEM, stream>>>(p);
    return cudaGetLastError();
}

template <class T>
void info_of(int (&info)[9]) {
    info[0] = T::BM; info[1] = T::BN; info[2] = T::WM; info[3] = T::WN; info[4] = T::BK; info[5] = T::STAGES;
    info[6] = T::MINB; info[7] = T::SMEM; info[8] = T::NT;
}

}  // namespace

int int8_gemm_sm80_tiles() {
    return NTILES;
}

bool int8_gemm_sm80_info(const int tile, int (&info)[9]) {
    switch (tile) {
        case 0: info_of<T0>(info); return true;
        case 1: info_of<T1>(info); return true;
        case 2: info_of<T2>(info); return true;
        case 3: info_of<T3>(info); return true;
        case 4: info_of<T4>(info); return true;
        case 5: info_of<T5>(info); return true;
        case 6: info_of<T6>(info); return true;
        default: return false;
    }
}

bool int8_gemm_sm80_loaded() {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, group_gemm_kernel<T0>) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return attr.binaryVersion >= 80 && attr.ptxVersion >= 80;
}

cudaError_t int8_gemm_sm80(const GroupGemmParams & p, const int tile, cudaStream_t stream) {
    switch (tile) {
        case 0: return launch<T0>(p, stream);
        case 1: return launch<T1>(p, stream);
        case 2: return launch<T2>(p, stream);
        case 3: return launch<T3>(p, stream);
        case 4: return launch<T4>(p, stream);
        case 5: return launch<T5>(p, stream);
        case 6: return launch<T6>(p, stream);
        default: return cudaErrorInvalidValue;
    }
}

}  // namespace pixal3d_kernels
