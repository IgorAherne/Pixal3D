// The int8 GEMM with group scales on the int8 tensor cores of sm_75 (Turing: RTX 20xx, GTX 16xx), also correct on
// every later card (int8_gemm.h has the arithmetic).
//
// A port of the Turing W8A8 GEMM of Spellcast3D's image engine (temp_code/ggml-kquants, w8a8-turing.cuh; its tiling,
// copy schedule, fragment schedule and epilogue were measured on Turing code: NOTES-w8a8.md, "Turing port"):
// - Block tile 128 x 128, K stages of 64, 8 warps of 32 rows x 64 channels (4 x 2), 256 threads, one block per SM.
// - MMA: sm_75 has no m16n8k32 int8 shape: each m16n8k32 product of the ldmatrix x4 fragments is four
//   mma.sync.m8n8k16.s32.s8.s8.s32 (rows gid / gid + 8, k 0-15 then 16-31), all chained through the C operand, so
//   the integer sum of a whole group (or of the whole K, per-row scales) accumulates in the MMA itself: no
//   instruction per product beside it. A group's sum is scaled into the fp32 accumulators once per 128 K (two stages):
//   one FADD + one FFMA per output (int8_gemm.h: the 1.5 * 2^23 conversion), against the image engine's one IMAD per
//   output and 32 K.
// - No cp.async on sm_75: a stage goes global -> registers (16-byte ld.global.nc, the L2::128B prefetch hint: the
//   row's next stage is an L2 hit) -> shared. Three slots of 16 KB (48 KB: the default limit, no opt-in): stage kt + 2
//   is loaded into registers during stage kt (in two pieces) and stored during stage kt + 1. This copy path is what
//   keeps the kernel at ~55 % of the card's int8 MMA rate (RTX 2060 Super: what-if builds without the global loads, or
//   without the shared stores, ran 40-60 % faster); measured without effect or worse: the A chunks two stages ahead (a
//   second register set), ld.global.nc / .cg / .ca without the hint, one or four pieces, no gap between store and load,
//   a 64 x 128 block tile of 4 warps with two blocks per SM (two slots: -17 % per-row, -10 % group scales).
// - Rows past M and channels past N read the last row / channel (no padding needed; their results are not stored).
// - Epilogue per 16-row fragment through the warp's shared-memory buffer, 16-byte streaming stores.

#include "int8_gemm.h"
#include "int8_gemm_common.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define P3K_IMMA_DEVICE 1
#else
#define P3K_IMMA_DEVICE 0
#endif

namespace pixal3d_kernels {
namespace {

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int WM = 32;                          // rows per warp tile
constexpr int MF = WM / 16;                     // m16 fragments per warp
constexpr int WARPS_M = BM / WM;
constexpr int NT = WARPS_M * 64;                // WARPS_M (rows) x 2 (channels) warps: 256 threads
constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK, STAGE = A_BYTES + B_BYTES;
constexpr int SLOTS = 3;
constexpr int SMEM = SLOTS * STAGE;             // 49,152 bytes
constexpr int CPO = BM * BK / 16 / NT;          // 16-byte chunks per thread and operand per stage (2)
constexpr int PIECES = 2;                       // a stage's chunks go in this many pieces (A, then B)
constexpr int CH = 2 * CPO / PIECES;            // chunks per piece
constexpr int LDG_GAP = 1;                      // pairs of n fragments between a piece's store and the next load into its registers
constexpr int EPI_BYTES = (NT / 32) * 16 * (64 + 8) * 4;     // the epilogue's per-warp staging (16 fp32 rows of 64 + 8)
static_assert(EPI_BYTES <= SMEM, "the epilogue reuses the pipeline's shared memory");

#if P3K_IMMA_DEVICE
static __device__ __forceinline__ int4 ldg16(const int8_t * p) {
    int4 v;
    asm volatile("ld.global.nc.L2::128B.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
static __device__ __forceinline__ void sts16(const unsigned addr, const int4 v) {
    asm volatile("st.shared.v4.s32 [%0], {%1, %2, %3, %4};" : : "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}
// byte offset of 16-byte chunk c (0..3) of row r in a [rows][64] int8 tile: chunks XOR-swizzled by bits 1-2 of the row,
// so that both the copies and ldmatrix are free of bank conflicts
static __device__ __forceinline__ int swz(const int r, const int c) {
    return r * 64 + ((c ^ ((r >> 1) & 3)) << 4);
}
static __device__ __forceinline__ void ldsm_x4(int (&r)[4], const unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// d = a x b + c for one 8 x 8 output block of 16 k values (a: row gid, k 4 tig .. +3; b: column gid, the same k;
// c / d: row gid, columns 2 tig, 2 tig + 1)
static __device__ __forceinline__ void mma8816(int & d0, int & d1, const int a, const int b, const int c0, const int c1) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};"
                 : "=r"(d0), "=r"(d1) : "r"(a), "r"(b), "r"(c0), "r"(c1));
}
// the m16n8k32 product of one A fragment set (a0 / a1: rows gid / gid + 8, k 0-15; a2 / a3: k 16-31) and one B pair,
// added to c (c[0..1]: row gid, c[2..3]: row gid + 8), or with `first` to `start` (a group's first product: the sum
// restarts from MAGIC without an instruction of its own). |sum| stays far below 2^31: satfinite never triggers.
static __device__ __forceinline__ void mma_k32(int (&c)[4], const int (&a)[4], const int b0, const int b1,
                                               const bool first, const int start) {
    mma8816(c[0], c[1], a[0], b0, first ? start : c[0], first ? start : c[1]);
    mma8816(c[2], c[3], a[1], b0, first ? start : c[2], first ? start : c[3]);
    mma8816(c[0], c[1], a[2], b1, c[0], c[1]);
    mma8816(c[2], c[3], a[3], b1, c[2], c[3]);
}
#endif

// AG: xs holds one scale per row and group (else per row); BG: ws one per channel and group (else per channel)
template <bool AG, bool BG>
__global__ void __launch_bounds__(NT, 1) gemm_kernel(const Int8GemmParams p) {
#if P3K_IMMA_DEVICE
    constexpr bool GROUPED = AG || BG;
    extern __shared__ __align__(128) uint8_t smem[];
    const int M = p.M, N = p.N, K = p.K;

    int tm, tn;
    tile_order(blockIdx.x, (M + BM - 1) / BM, (N + BN - 1) / BN, p.raster, tm, tn);
    const int m0 = tm * BM;
    const int n0 = tn * BN;

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int wm   = warp % WARPS_M;    // 32-row slice
    const int wn   = warp / WARPS_M;    // 64-channel slice
    const int gid  = lane / 4;
    const int tig  = lane % 4;
    const int nk   = K / BK;
    const unsigned sbase = (unsigned) __cvta_generic_to_shared(smem);

    // copies: chunk c (0 .. 2 CPO - 1) of a stage = A (c < CPO) or B, tile row (c % CPO) * 64 + tid / 4, 16-byte chunk
    // tid % 4 (rows r and r + 64 have the same swizzle); rows past M / N read the last one
    const int8_t * src[2 * CPO];
#pragma unroll
    for (int c = 0; c < CPO; ++c) {
        const int ra = min(m0 + c * (NT / 4) + tid / 4, M - 1);
        const int rb = min(n0 + c * (NT / 4) + tid / 4, N - 1);
        src[c]       = p.xq + (int64_t) ra * p.ldx + 16 * (tid % 4);
        src[CPO + c] = p.wq + (int64_t) rb * p.ldw + 16 * (tid % 4);
    }
    const unsigned cp_soff = (unsigned) swz(tid / 4, tid % 4);
    auto soff = [&](const int c) { return cp_soff + (c < CPO ? 0 : A_BYTES) + (c % CPO) * (NT / 4) * BK; };
    int4 stg[2 * CPO];              // three slots: a whole stage waits in the registers for a stage
    auto ldg_piece = [&](const int j) {
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            const int c = j * CH + i;
            stg[c] = ldg16(src[c]);
            src[c] += BK;
        }
    };
    auto sts_piece = [&](const unsigned slot_base, const int j) {
#pragma unroll
        for (int i = 0; i < CH; ++i) {
            sts16(slot_base + soff(j * CH + i), stg[j * CH + i]);
        }
    };

    // the integer sums (from MAGIC per group when grouped: each group's first MMA starts from `magic`; else from 0
    // over the whole K) and the fp32 accumulators
    const int magic = MAGIC;
    int   acc[MF][8][4];
    float accf[MF][8][4];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int nf = 0; nf < 8; ++nf)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[mf][nf][j]  = GROUPED ? MAGIC : 0;
                accf[mf][nf][j] = 0.0f;
            }

    // the thread's rows (wm * 32 + mf * 16 + h * 8 + gid) and channel pairs (wn * 64 + nf * 8 + 2 tig), clamped
    int xrow[MF][2];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int h = 0; h < 2; ++h) xrow[mf][h] = min(m0 + wm * WM + mf * 16 + h * 8 + gid, M - 1);
    const int ncol = n0 + wn * 64 + 2 * tig;    // + nf * 8
    // a group's scales: loaded at the group's first stage, used after its second
    float sx[MF][2], sw[8][2];
    auto load_scales = [&](const int g) {
        if constexpr (AG) {
#pragma unroll
            for (int mf = 0; mf < MF; ++mf)
#pragma unroll
                for (int h = 0; h < 2; ++h) sx[mf][h] = __ldg(p.xs + (int64_t) xrow[mf][h] * p.ldxs + g);
        }
        if constexpr (BG) {
#pragma unroll
            for (int nf = 0; nf < 8; ++nf) {
                const int n = ncol + nf * 8 < N ? ncol + nf * 8 : 0;
                const float2 v = __ldg((const float2 *) (p.ws + (int64_t) g * N + n));
                sw[nf][0] = v.x;
                sw[nf][1] = v.y;
            }
        }
    };
    auto scale_group = [&]() {
#pragma unroll
        for (int mf = 0; mf < MF; ++mf)
#pragma unroll
            for (int nf = 0; nf < 8; ++nf)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float s;
                    if constexpr (AG && BG) {
                        s = sx[mf][j / 2] * sw[nf][j % 2];
                    } else if constexpr (AG) {
                        s = sx[mf][j / 2];
                    } else {
                        s = sw[nf][j % 2];
                    }
                    accf[mf][nf][j] = __fmaf_rn(__int_as_float(acc[mf][nf][j]) - MAGIC_F, s, accf[mf][nf][j]);
                }
    };

    // per-lane ldmatrix offsets in a stage (rows r and r + 16 i have the same swizzle, so the m / n fragment is an
    // immediate offset of mf * 1024 / np * 1024); index = k step (32 values)
    unsigned a_off[2], b_off[2];
    {
        const int a_row = wm * WM + (lane % 16);
        const int a_x   = (lane / 16) ^ ((a_row >> 1) & 3);
        const int b_row = wn * 64 + (lane % 8) + (lane / 16) * 8;
        const int b_x   = ((lane / 8) % 2) ^ ((b_row >> 1) & 3);
        a_off[0] = a_row * 64 + (a_x << 4);
        a_off[1] = a_row * 64 + ((a_x ^ 2) << 4);
        b_off[0] = A_BYTES + b_row * 64 + (b_x << 4);
        b_off[1] = A_BYTES + b_row * 64 + ((b_x ^ 2) << 4);
    }
    auto load_a = [&](int (&f)[4], const unsigned st, const int ks, const int mf) {
        ldsm_x4(f, st + a_off[ks] + mf * 1024);
    };
    auto load_b = [&](int (&f)[4], const unsigned st, const int ks, const int np) {
        ldsm_x4(f, st + b_off[ks] + np * 1024);
    };

    int fa[MF][4] = {};
    int fb[2][4]  = {};

    // one k step of 32 values: fa / fb[0] hold its A fragments and B pair 0, its B pairs 1..3 are at st / ks, the next
    // k step at nst / nks (its fragments are loaded as this step's are used up); hook(np) runs before the loads and
    // MMAs of pair np. The m order is serpentine (the A fragment of the last MMA is reused).
    auto kstep = [&](const int par, const unsigned st, const int ks, const unsigned nst, const int nks, const bool first,
                     auto && hook) {
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            hook(np);
            if (np < 3) {
                load_b(fb[(np + 1) % 2], st, ks, np + 1);
            } else {
                load_b(fb[0], nst, nks, 0);
            }
            const int (&b)[4] = fb[np % 2];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
#pragma unroll
                for (int j = 0; j < MF; ++j) {
                    const int mf = (nn ^ par) ? MF - 1 - j : j;
                    mma_k32(acc[mf][2 * np + nn], fa[mf], b[2 * nn], b[2 * nn + 1], first, magic);
                    if (np == 3 && nn == 1) {
                        load_a(fa[mf], nst, nks, mf);   // its last use in this k step
                    }
                }
            }
        }
    };

    // prologue: stage 0 into slot 0, stage 1 into the registers, then the fragments of k step 0
    {
        int4 v[2 * CPO];
#pragma unroll
        for (int c = 0; c < 2 * CPO; ++c) {
            v[c] = ldg16(src[c]);
            src[c] += BK;
        }
#pragma unroll
        for (int c = 0; c < 2 * CPO; ++c) {
            sts16(sbase + soff(c), v[c]);
        }
    }
    if (nk > 1) {
#pragma unroll
        for (int j = 0; j < PIECES; ++j) {
            ldg_piece(j);
        }
    }
    __syncthreads();
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        load_a(fa[mf], sbase, 0, mf);
    }
    load_b(fb[0], sbase, 0, 0);

    // The copy schedule in pairs q = 4 ks + np = 0..7 of stage kt (LAG = 8 / PIECES): stage kt + 1 (in the registers
    // since stage kt - 1) is stored into its slot (last read in stage kt - 2) at q = j LAG, piece j of stage kt + 2 is
    // loaded LDG_GAP pairs later; the barrier at q = 7, before the loads of the next stage's first fragments.
    constexpr int LAG = 8 / PIECES;
    int slot = 0;
    auto stage = [&](const int kt, const bool first) {
        const bool more1 = kt + 1 < nk;
        const bool more2 = kt + 2 < nk;
        const int nslot = slot == SLOTS - 1 ? 0 : slot + 1;
        const unsigned st  = sbase + slot * STAGE;
        const unsigned nst = sbase + nslot * STAGE;
        auto copy_hook = [&](const int q) {
            if (q % LAG == 0 && more1) {
                sts_piece(nst, q / LAG);
            }
            if ((q + 8 - LDG_GAP) % LAG == 0 && more2) {
                ldg_piece(((q + 8 - LDG_GAP) % 8) / LAG);
            }
            if (q == 7) {
                __syncthreads();
            }
        };
        kstep(0, st, 0, st, 1, first, [&](const int np) { copy_hook(np); });
        kstep(1, st, 1, nst, 0, false, [&](const int np) { copy_hook(4 + np); });
        slot = nslot;
    };
    if constexpr (GROUPED) {
        // one group (two stages) per iteration: its scales load at its start, its sums are scaled at its end, in the
        // same block of code as the MMAs (the scheduler interleaves them)
        for (int kt = 0; kt < nk; kt += 2) {
            load_scales(kt / 2);
            stage(kt, true);
            stage(kt + 1, false);
            scale_group();
        }
    } else {
        for (int kt = 0; kt < nk; ++kt) {
            stage(kt, false);
        }
    }

    // epilogue: per m fragment, the warp's 16 x 64 values (fp32, or the int32 sums' bits) through its shared-memory
    // buffer, then 8 values a lane: GELU, the output dtype, 16-byte stores (store8, in a loop that is not unrolled)
    __syncthreads();    // the pipeline's slots are free
    constexpr int ELD = 64 + 8;                 // staged row stride in floats (+8: conflict-free fragment writes)
    float * ebuf = reinterpret_cast<float *>(smem) + warp * 16 * ELD;
    float cs[8][2], cb[8][2];
#pragma unroll
    for (int nf = 0; nf < 8; ++nf) {
        const int n = ncol + nf * 8 < N ? ncol + nf * 8 : 0;
        const float2 w = BG ? make_float2(1.0f, 1.0f) : __ldg((const float2 *) (p.ws + n));
        const float2 b = p.bias ? __ldg((const float2 *) (p.bias + n)) : make_float2(0.0f, 0.0f);
        cs[nf][0] = w.x;
        cs[nf][1] = w.y;
        cb[nf][0] = b.x;
        cb[nf][1] = b.y;
    }
    float rs[MF][2];
#pragma unroll
    for (int mf = 0; mf < MF; ++mf)
#pragma unroll
        for (int h = 0; h < 2; ++h) rs[mf][h] = AG ? 1.0f : __ldg(p.xs + xrow[mf][h]);
    const int  out  = p.out;
    const bool gelu = p.gelu != 0;
    const int  c8   = (lane % 8) * 8;           // the lane's 8 columns in the store phase
    const int  col  = n0 + wn * 64 + c8;
#pragma unroll
    for (int mf = 0; mf < MF; ++mf) {
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = h * 8 + gid;
#pragma unroll
            for (int nf = 0; nf < 8; ++nf) {
                float y[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int a = acc[mf][nf][2 * h + i];
                    if (out == OUT_I32) {
                        y[i] = __int_as_float(a);
                    } else if constexpr (AG && BG) {
                        y[i] = accf[mf][nf][2 * h + i] + cb[nf][i];
                    } else if constexpr (AG) {
                        y[i] = accf[mf][nf][2 * h + i] * cs[nf][i] + cb[nf][i];
                    } else if constexpr (BG) {
                        y[i] = accf[mf][nf][2 * h + i] * rs[mf][h] + cb[nf][i];
                    } else {
                        y[i] = ((float) a * rs[mf][h]) * cs[nf][i] + cb[nf][i];
                    }
                }
                *(float2 *) (ebuf + r * ELD + nf * 8 + 2 * tig) = make_float2(y[0], y[1]);
            }
        }
        __syncwarp();
#pragma unroll 1
        for (int i = 0; i < 4; ++i) {
            const int r = i * 4 + lane / 8;
            const int t = m0 + wm * WM + mf * 16 + r;
            const float4 v0 = *(const float4 *) (ebuf + r * ELD + c8);
            const float4 v1 = *(const float4 *) (ebuf + r * ELD + c8 + 4);
            float v[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
            if (t < M && col < N) {
                store8(p.y, (int64_t) t * N + col, v, out, gelu);
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
    gemm_kernel<AG, BG><<<dim3((unsigned) (ntm * ntn)), NT, SMEM, stream>>>(p);
    return cudaGetLastError();
}

}  // namespace

bool int8_gemm_imma_loaded() {
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, gemm_kernel<false, false>) != cudaSuccess) {
        cudaGetLastError();     // no image for this card: not loaded (and no error left behind)
        return false;
    }
    return a.ptxVersion >= 75;
}

cudaError_t int8_gemm_imma(const Int8GemmParams & p, const bool a_group, const bool b_group, cudaStream_t stream) {
    if (p.K % (2 * BK) || (p.out == OUT_I32 && (a_group || b_group))) {
        return cudaErrorInvalidValue;
    }
    Int8GemmParams q = p;
    if (q.raster == 0) {
        q.raster = -4;      // four column tiles side by side (the image engine's choice below N = 16,384)
    }
    if (a_group) {
        return b_group ? launch<true, true>(q, stream) : launch<true, false>(q, stream);
    }
    return b_group ? launch<false, true>(q, stream) : launch<false, false>(q, stream);
}

}  // namespace pixal3d_kernels
