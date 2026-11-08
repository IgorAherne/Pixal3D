// torch.ops.pixal3d_kernels: the operator schemas, the argument checks and the kernel choice per card.

#include <Python.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "elementwise.h"
#include "int8_gemm.h"

extern "C" {
// an empty module: importing pixal3d_kernels._C loads the library, whose static initialisers register the operators
PyObject * PyInit__C(void) {
    static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL};
    return PyModule_Create(&module_def);
}
}

namespace pixal3d_kernels {
namespace {

// per device: which kernel families this build really holds for it (int8_gemm.h), asked once (bit 0: the
// tensor-core kernel, bit 1: dp4a; 0 = not asked yet)
int loaded(const int device) {
    static int known[64] = {};
    if (device >= 0 && device < 64 && known[device]) {
        return known[device] & 3;
    }
    const int bits = 4 | (int8_gemm_imma_loaded() ? 1 : 0) | (int8_gemm_dp4a_loaded() ? 2 : 0);
    if (device >= 0 && device < 64) {
        known[device] = bits;
    }
    return bits & 3;
}

// kernel: 0 = by the card (sm_75 and up: the int8 tensor cores; sm_61 .. sm_72, or a build without the tensor-core
// kernel's code for the card: dp4a), 1 = tensor cores, 2 = dp4a
bool use_imma(const at::Device & device, const int64_t kernel) {
    const cudaDeviceProp * prop = at::cuda::getDeviceProperties(device.index());
    const int cc = prop->major * 10 + prop->minor;
    TORCH_CHECK(cc >= 61, "int8_gemm_group needs dp4a (sm_61 or newer), the card is sm_", cc);
    TORCH_CHECK(kernel >= 0 && kernel <= 2, "kernel must be 0 (auto), 1 (tensor cores) or 2 (dp4a)");
    const int have = loaded(device.index());
    const bool imma = cc >= 75 && (have & 1);
    if (kernel == 1) {
        TORCH_CHECK(imma, "the tensor-core kernel needs sm_75 or newer and this build's code for sm_", cc);
        return true;
    }
    if (kernel == 0 && imma) {
        return true;
    }
    TORCH_CHECK(have & 2, "this build of pixal3d_kernels has no int8_gemm_group code for sm_", cc,
                " (TORCH_CUDA_ARCH_LIST at build time)");
    return false;
}

// a 2-D int8 operand with 16-byte aligned, K-contiguous rows (else a contiguous copy)
at::Tensor rows16(const at::Tensor & t) {
    if (t.stride(1) == 1 && t.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0) {
        return t;
    }
    return t.contiguous();
}

// an fp32 vector read in 8- and 16-byte pieces
at::Tensor vec16(const at::Tensor & t) {
    at::Tensor v = t.to(at::kFloat).contiguous();
    if (reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 != 0) {
        v = v.clone();
    }
    return v;
}

}  // namespace

at::Tensor int8_gemm_group(const at::Tensor & xq_in, const at::Tensor & xs_in, const at::Tensor & wq_in,
                           const at::Tensor & ws_in, const c10::optional<at::Tensor> & bias_in, const bool gelu,
                           const at::ScalarType out_dtype, const int64_t raster, const int64_t kernel) {
    TORCH_CHECK(xq_in.is_cuda() && wq_in.is_cuda() && xs_in.is_cuda() && ws_in.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(xq_in.scalar_type() == at::kChar && wq_in.scalar_type() == at::kChar, "xq and wq must be int8");
    TORCH_CHECK(xs_in.scalar_type() == at::kFloat && ws_in.scalar_type() == at::kFloat, "xs and ws must be fp32");
    TORCH_CHECK(xq_in.dim() == 2 && wq_in.dim() == 2, "xq [M, K] and wq [N, K] expected");
    const int64_t M = xq_in.size(0), K = xq_in.size(1), N = wq_in.size(0);
    TORCH_CHECK(wq_in.size(1) == K, "xq [M, K] and wq [N, K] differ in K");
    TORCH_CHECK(N % 16 == 0 && N > 0, "N must be a positive multiple of 16, got ", N);
    TORCH_CHECK(K % GROUP == 0 && K > 0, "K must be a positive multiple of 128, got ", K);
    TORCH_CHECK(M < (int64_t(1) << 31) && N < (int64_t(1) << 31) && K < (int64_t(1) << 24), "sizes out of range");
    const bool a_group = xs_in.dim() == 2;
    const bool b_group = ws_in.dim() == 2;
    if (a_group) {
        TORCH_CHECK(xs_in.size(0) == M && xs_in.size(1) * GROUP == K, "xs must be [M] or [M, K / 128]");
    } else {
        TORCH_CHECK(xs_in.dim() == 1 && xs_in.size(0) == M, "xs must be [M] or [M, K / 128]");
    }
    if (b_group) {
        TORCH_CHECK(ws_in.size(0) == N && ws_in.size(1) * GROUP == K, "ws must be [N] or [N, K / 128]");
    } else {
        TORCH_CHECK(ws_in.dim() == 1 && ws_in.size(0) == N, "ws must be [N] or [N, K / 128]");
    }
    TORCH_CHECK(out_dtype == at::kFloat || out_dtype == at::kHalf || out_dtype == at::kBFloat16 ||
                    out_dtype == at::kInt, "out_dtype must be float32, float16, bfloat16 or int32");
    const bool raw = out_dtype == at::kInt;
    TORCH_CHECK(!raw || (!a_group && !b_group && !bias_in.has_value() && !gelu),
                "int32 out (the raw integer sum) takes per-row / per-channel scales, no bias, no GELU");
    const at::Device device = xq_in.device();
    TORCH_CHECK(wq_in.device() == device && xs_in.device() == device && ws_in.device() == device,
                "the tensors are on different devices");
    const c10::cuda::CUDAGuard guard(device);

    const at::Tensor xq = rows16(xq_in);
    const at::Tensor wq = rows16(wq_in);
    at::Tensor xs = xs_in;
    if (a_group ? xs.stride(1) != 1 : xs.stride(0) != 1) {
        xs = xs.contiguous();
    }
    at::Tensor ws = ws_in;
    if (b_group) {
        if (ws.stride(0) != 1 || ws.stride(1) != N || reinterpret_cast<uintptr_t>(ws.data_ptr()) % 16 != 0) {
            ws = ws.t().contiguous().t();       // column-major (a group's N scales contiguous), aligned
        }
    } else {
        ws = vec16(ws);
    }
    at::Tensor bias;
    if (bias_in.has_value()) {
        TORCH_CHECK(bias_in->is_cuda() && bias_in->device() == device && bias_in->dim() == 1 && bias_in->size(0) == N,
                    "bias must be [N] on the same device");
        bias = vec16(*bias_in);
    }
    at::Tensor y = at::empty({M, N}, xq.options().dtype(out_dtype));
    if (M == 0) {
        return y;
    }

    Int8GemmParams p;
    p.xq = xq.data_ptr<int8_t>();
    p.xs = xs.data_ptr<float>();
    p.wq = wq.data_ptr<int8_t>();
    p.ws = ws.data_ptr<float>();
    p.bias = bias.defined() ? bias.data_ptr<float>() : nullptr;
    p.y = y.data_ptr();
    p.ldx = xq.stride(0);
    p.ldw = wq.stride(0);
    p.ldxs = a_group ? xs.stride(0) : 1;
    p.M = (int) M;
    p.N = (int) N;
    p.K = (int) K;
    p.out = out_dtype == at::kFloat ? OUT_F32 : out_dtype == at::kHalf ? OUT_F16
                                              : out_dtype == at::kBFloat16 ? OUT_BF16 : OUT_I32;
    p.gelu = gelu ? 1 : 0;
    p.raster = (int) raster;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const cudaError_t err = use_imma(device, kernel) ? int8_gemm_imma(p, a_group, b_group, stream)
                                                     : int8_gemm_dp4a(p, a_group, b_group, stream);
    TORCH_CHECK(err == cudaSuccess, "int8_gemm_group: ", cudaGetErrorString(err));
    return y;
}

// ---- the element-wise chains ----------------------------------------------------------------------------------------

namespace {

int dt_of(const at::ScalarType t) {
    TORCH_CHECK(t == at::kFloat || t == at::kHalf || t == at::kBFloat16, "a float32, float16 or bfloat16 tensor expected");
    return t == at::kFloat ? DT_F32 : t == at::kHalf ? DT_F16 : DT_BF16;
}

// a [C] (or [1, C]) row operand, contiguous
at::Tensor row_of(const at::Tensor & t, const int64_t C) {
    TORCH_CHECK(t.numel() == C, "a row of ", C, " values expected (one item), got ", t.sizes());
    return t.reshape({C}).contiguous();
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> op_quantize_rows(const at::Tensor & x_in, const int64_t group,
                                                    const c10::optional<at::Tensor> & inv_smooth) {
    TORCH_CHECK(x_in.is_cuda() && x_in.dim() == 2, "x must be a 2-D CUDA tensor");
    const c10::cuda::CUDAGuard guard(x_in.device());
    const at::Tensor x = x_in.stride(1) == 1 ? x_in : x_in.contiguous();
    const int64_t M = x.size(0), K = x.size(1);
    const int64_t seg = group > 0 ? group : K;
    TORCH_CHECK(K > 0 && K % 8 == 0 && seg % 8 == 0 && K % seg == 0, "K % 8 == 0 and a group dividing K expected");
    // a warp's share of a segment (or less) in a part of one warp; up to 8 chunks of 8 a lane in one warp (a row
    // of 1536: 6 a lane, shuffles alone); longer segments a block, its warps' maxima through shared memory
    const int64_t chunks = seg / 8;
    int vpt = 1;
    if (chunks > 32 && chunks <= 256 && chunks % 32 == 0) {
        vpt = (int) (chunks / 32);
    } else {
        while (chunks / vpt > 1024 && vpt < 4) vpt *= 2;
    }
    const int64_t team = chunks / vpt;
    TORCH_CHECK(team * 8 * vpt == seg && team <= 1024 && (team > 32 ? team % 32 == 0 : (team & (team - 1)) == 0),
                "segment of ", seg, " values: no team shape for it");
    at::Tensor inv;
    if (inv_smooth.has_value()) {
        TORCH_CHECK(inv_smooth->numel() == K, "inv_smooth must hold K values");
        inv = inv_smooth->to(at::kFloat).contiguous();
    }
    at::Tensor q = at::empty({M, K}, x.options().dtype(at::kChar));
    at::Tensor s = group > 0 ? at::empty({M, K / group}, x.options().dtype(at::kFloat))
                             : at::empty({M}, x.options().dtype(at::kFloat));
    QuantParams p;
    p.x = x.data_ptr();
    p.inv_smooth = inv.defined() ? inv.data_ptr<float>() : nullptr;
    p.q = q.data_ptr<int8_t>();
    p.s = s.data_ptr<float>();
    p.M = M;
    p.ldx = x.stride(0);
    p.K = (int) K;
    p.group = (int) group;
    p.dtype = dt_of(x.scalar_type());
    p.team = (int) team;
    p.vpt = vpt;
    TORCH_CHECK(reinterpret_cast<uintptr_t>(p.x) % 16 == 0 || p.dtype == DT_F32, "x must be 16-byte aligned");
    const cudaError_t err = pixal3d_kernels::quantize_rows(p, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "quantize_rows: ", cudaGetErrorString(err));
    return {q, s};
}

at::Tensor op_modulate(const at::Tensor & h_in, const at::Tensor & scale1, const at::Tensor & shift,
                       const at::ScalarType out_dtype) {
    TORCH_CHECK(h_in.is_cuda() && h_in.scalar_type() == at::kFloat, "h must be a float32 CUDA tensor");
    TORCH_CHECK(scale1.scalar_type() == shift.scalar_type(), "scale and shift of one dtype expected");
    const c10::cuda::CUDAGuard guard(h_in.device());
    const at::Tensor h = h_in.contiguous();
    const int64_t C = h.size(-1);
    const at::Tensor s1 = row_of(scale1, C), sh = row_of(shift, C);
    at::Tensor out = at::empty(h.sizes(), h.options().dtype(out_dtype));
    const cudaError_t err = pixal3d_kernels::modulate(h.data_ptr(), s1.data_ptr(), sh.data_ptr(), out.data_ptr(),
                                                      h.numel(), (int) C, dt_of(s1.scalar_type()), dt_of(out_dtype),
                                                      at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "modulate: ", cudaGetErrorString(err));
    return out;
}

at::Tensor op_residual_gated(const at::Tensor & x_in, const at::Tensor & h_in, const at::Tensor & gate) {
    TORCH_CHECK(x_in.is_cuda() && x_in.scalar_type() == at::kFloat, "x must be a float32 CUDA tensor");
    TORCH_CHECK(h_in.sizes() == x_in.sizes(), "x and h of one shape expected");
    const c10::cuda::CUDAGuard guard(x_in.device());
    const at::Tensor x = x_in.contiguous(), h = h_in.contiguous();
    const int64_t C = x.size(-1);
    const at::Tensor g = row_of(gate, C);
    at::Tensor out = at::empty_like(x);
    const cudaError_t err = pixal3d_kernels::residual_gated(x.data_ptr(), h.data_ptr(), g.data_ptr(), out.data_ptr(),
                                                            x.numel(), (int) C, dt_of(h.scalar_type()),
                                                            dt_of(g.scalar_type()), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "residual_gated: ", cudaGetErrorString(err));
    return out;
}

at::Tensor op_qk_norm_rope(const at::Tensor & x, const at::Tensor & gamma_in, const double scale, const double eps,
                           const c10::optional<at::Tensor> & phases_in, const at::ScalarType out_dtype) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && x.size(2) == 128 && x.stride(2) == 1,
                "x must be a CUDA [T, H, 128] tensor with its last dimension contiguous");
    const c10::cuda::CUDAGuard guard(x.device());
    const int64_t T = x.size(0), H = x.size(1);
    TORCH_CHECK(gamma_in.numel() == H * 128, "gamma must be [H, 128]");
    const at::Tensor gamma = gamma_in.contiguous();
    at::Tensor ph;
    if (phases_in.has_value()) {
        ph = *phases_in;
        TORCH_CHECK(ph.scalar_type() == at::kComplexFloat && ph.dim() == 2 && ph.size(0) == T && ph.size(1) == 64,
                    "phases must be complex64 [T, 64]");
        if (ph.stride(1) != 1) ph = ph.contiguous();
    }
    at::Tensor out = at::empty({T, H, 128}, x.options().dtype(out_dtype));
    QkParams p;
    p.x = x.data_ptr();
    p.gamma = gamma.data_ptr();
    p.phases = ph.defined() ? reinterpret_cast<const float2 *>(ph.data_ptr()) : nullptr;
    p.out = out.data_ptr();
    p.scale = (float) scale;
    p.eps = (float) eps;
    p.rows = T * H;
    p.ld_t = x.stride(0);
    p.ld_h = x.stride(1);
    p.ld_ph = ph.defined() ? ph.stride(0) : 0;
    p.H = (int) H;
    p.x_dtype = dt_of(x.scalar_type());
    p.g_dtype = dt_of(gamma.scalar_type());
    p.out_dtype = dt_of(out_dtype);
    const cudaError_t err = pixal3d_kernels::qk_norm_rope(p, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "qk_norm_rope: ", cudaGetErrorString(err));
    return out;
}

}  // namespace pixal3d_kernels

TORCH_LIBRARY(pixal3d_kernels, m) {
    // raster / kernel: developer knobs (the tile order; the kernel family), 0 = the defaults
    m.def("int8_gemm_group(Tensor xq, Tensor xs, Tensor wq, Tensor ws, Tensor? bias, bool gelu, "
          "ScalarType out_dtype, int raster=0, int kernel=0) -> Tensor");
    // the flows' element-wise chains below sm_80 (elementwise.cu): the eager torch code's bits
    m.def("quantize_rows(Tensor x, int group, Tensor? inv_smooth) -> (Tensor, Tensor)");
    m.def("modulate(Tensor h, Tensor scale1, Tensor shift, ScalarType out_dtype) -> Tensor");
    m.def("residual_gated(Tensor x, Tensor h, Tensor gate) -> Tensor");
    m.def("qk_norm_rope(Tensor x, Tensor gamma, float scale, float eps, Tensor? phases, ScalarType out_dtype) -> Tensor");
}

TORCH_LIBRARY_IMPL(pixal3d_kernels, CUDA, m) {
    m.impl("int8_gemm_group", &pixal3d_kernels::int8_gemm_group);
    m.impl("quantize_rows", &pixal3d_kernels::op_quantize_rows);
    m.impl("modulate", &pixal3d_kernels::op_modulate);
    m.impl("residual_gated", &pixal3d_kernels::op_residual_gated);
    m.impl("qk_norm_rope", &pixal3d_kernels::op_qk_norm_rope);
}
