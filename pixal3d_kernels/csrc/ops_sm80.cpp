// torch.ops.pixal3d_kernels.int8_gemm_group80 (sm_80+): its schema, argument checks and launch, registered as a
// fragment of the package's library (ops.cpp defines the library; this file adds to it).
//
//     int8_gemm_group80(xq, xs, wq, ws, bias, gelu, out, rx, rg, r2, tile, group_m) -> (y, qs)
//     int8_gemm_group80_info(tile) -> [BM, BN, WM, WN, BK, STAGES, blocks per SM, shared bytes, threads] ([] past the
//                                     table), int8_gemm_group80_loaded() -> whether the card runs the kernel's code

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "int8_gemm_sm80.h"

#ifdef P3K_SM80_STANDALONE          // a developer's build of these files alone (its own library and module)
#include <Python.h>
#define P3K_SM80_LIB p3k_sm80dev
extern "C" {
PyObject * PyInit_p3k_sm80dev(void) {
    static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "p3k_sm80dev", NULL, -1, NULL};
    return PyModule_Create(&module_def);
}
}
#else
#define P3K_SM80_LIB pixal3d_kernels
#endif

namespace pixal3d_kernels {
namespace {

bool aligned(const at::Tensor & t, const int bytes) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
}

// a [N] operand (fp32 / bf16 as asked), contiguous and 16-byte aligned
at::Tensor vec_of(const at::Tensor & t, const int64_t n, const at::ScalarType dt, const char * what) {
    TORCH_CHECK(t.numel() == n, what, " must hold ", n, " values");
    at::Tensor v = t.reshape({n}).to(dt).contiguous();
    if (!aligned(v, 16)) v = v.clone();
    return v;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> op_int8_gemm_group80(
        const at::Tensor & xq_in, const at::Tensor & xs_in, const at::Tensor & wq_in, const at::Tensor & ws_in,
        const c10::optional<at::Tensor> & bias_in, const bool gelu, const int64_t out,
        const c10::optional<at::Tensor> & rx_in, const c10::optional<at::Tensor> & rg_in,
        const c10::optional<at::Tensor> & r2_in, const int64_t tile, const int64_t group_m) {
    TORCH_CHECK(xq_in.is_cuda() && wq_in.is_cuda() && xs_in.is_cuda() && ws_in.is_cuda(), "CUDA tensors expected");
    TORCH_CHECK(xq_in.scalar_type() == at::kChar && wq_in.scalar_type() == at::kChar, "xq and wq must be int8");
    TORCH_CHECK(xs_in.scalar_type() == at::kFloat && ws_in.scalar_type() == at::kFloat, "xs and ws must be fp32");
    TORCH_CHECK(xq_in.dim() == 2 && wq_in.dim() == 2 && xs_in.dim() == 2, "xq [M, K], xs [M, K / 128], wq [N, K]");
    const int64_t M = xq_in.size(0), K = xq_in.size(1), N = wq_in.size(0);
    TORCH_CHECK(wq_in.size(1) == K && K % 128 == 0 && K > 0 && K < (int64_t(1) << 24), "K a multiple of 128");
    TORCH_CHECK(N % 16 == 0 && N > 0 && N < (int64_t(1) << 31) && M < (int64_t(1) << 31), "N a multiple of 16");
    TORCH_CHECK(xs_in.size(0) == M && xs_in.size(1) * 128 == K, "xs must be [M, K / 128]");
    TORCH_CHECK(ws_in.dim() == 1 && ws_in.size(0) == N, "ws must be [N] (one scale per output channel)");
    TORCH_CHECK(out == 0 || out == 2 || out == 3, "out: 0 = bf16, 2 = int8 + scales per 128 columns, 3 = fp16");
    const at::Device device = xq_in.device();
    TORCH_CHECK(wq_in.device() == device && xs_in.device() == device && ws_in.device() == device,
                "the tensors are on different devices");
    int info[9];
    TORCH_CHECK(int8_gemm_sm80_info((int) tile, info), "no tile ", tile);
    TORCH_CHECK(out != 2 || (info[1] % 128 == 0 && 128 % info[3] == 0 && N % 128 == 0),
                "int8 out needs N and the tile's columns in whole groups of 128");
    TORCH_CHECK(group_m >= 1 && group_m <= 1024, "group_m in 1 .. 1024");
    const c10::cuda::CUDAGuard guard(device);
    const cudaDeviceProp * prop = at::cuda::getDeviceProperties(device.index());
    TORCH_CHECK(prop->major >= 8, "int8_gemm_group80 needs sm_80 or newer, the card is sm_", prop->major, prop->minor);
    TORCH_CHECK(int8_gemm_sm80_loaded(), "this build of pixal3d_kernels has no sm_80+ code for sm_", prop->major,
                prop->minor, " (TORCH_CUDA_ARCH_LIST at build time)");
    TORCH_CHECK(info[7] <= (int) prop->sharedMemPerBlockOptin, "tile ", tile, " needs ", info[7],
                " bytes of shared memory, the card gives ", prop->sharedMemPerBlockOptin);

    auto rows16 = [](const at::Tensor & t) {
        return (t.stride(1) == 1 && t.stride(0) % 16 == 0 && aligned(t, 16)) ? t : t.contiguous();
    };
    const at::Tensor xq = rows16(xq_in), wq = rows16(wq_in);
    const at::Tensor xs = xs_in.stride(1) == 1 ? xs_in : xs_in.contiguous();
    const at::Tensor ws = vec_of(ws_in, N, at::kFloat, "ws");
    at::Tensor bias;
    if (bias_in.has_value()) {
        TORCH_CHECK(bias_in->is_cuda() && bias_in->device() == device, "bias on the same device");
        bias = vec_of(*bias_in, N, at::kFloat, "bias");
    }
    const at::ScalarType ydt = out == 2 ? at::kChar : out == 3 ? at::kHalf : at::kBFloat16;
    at::Tensor y = at::empty({M, N}, xq.options().dtype(ydt));
    at::Tensor qs = out == 2 ? at::empty({M, N / 128}, xq.options().dtype(at::kFloat))
                             : at::empty({0}, xq.options().dtype(at::kFloat));
    int resid = 0;
    at::Tensor rx, rg, r2;
    if (rx_in.has_value()) {
        TORCH_CHECK(out == 0, "the residual epilogue writes bf16");
        TORCH_CHECK(!(rg_in.has_value() && r2_in.has_value()), "a gate or a second term, not both");
        auto like_y = [&](const at::Tensor & t, const char * what) {
            TORCH_CHECK(t.is_cuda() && t.device() == device && t.scalar_type() == at::kBFloat16 && t.dim() == 2 &&
                            t.size(0) == M && t.size(1) == N, what, " must be bf16 [M, N] on the same device");
            return t.is_contiguous() && aligned(t, 16) ? t : t.contiguous();
        };
        rx = like_y(*rx_in, "rx");
        if (rg_in.has_value()) {
            TORCH_CHECK(rg_in->is_cuda() && rg_in->device() == device && rg_in->scalar_type() == at::kBFloat16,
                        "rg must be bf16 on the same device");
            rg = vec_of(*rg_in, N, at::kBFloat16, "rg");
            resid = 1;
        } else if (r2_in.has_value()) {
            r2 = like_y(*r2_in, "r2");
            resid = 2;
        } else {
            resid = 3;
        }
    }
    if (M == 0) return {y, qs};

    GroupGemmParams p;
    p.xq = xq.data_ptr<int8_t>();
    p.xs = xs.data_ptr<float>();
    p.wq = wq.data_ptr<int8_t>();
    p.ws = ws.data_ptr<float>();
    p.bias = bias.defined() ? bias.data_ptr<float>() : nullptr;
    p.y = y.data_ptr();
    p.qs = out == 2 ? qs.data_ptr<float>() : nullptr;
    p.rx = rx.defined() ? rx.data_ptr() : nullptr;
    p.rg = rg.defined() ? rg.data_ptr() : nullptr;
    p.r2 = r2.defined() ? r2.data_ptr() : nullptr;
    p.ldx = xq.stride(0);
    p.ldw = wq.stride(0);
    p.ldxs = xs.stride(0);
    p.ldy = y.stride(0);
    p.ldqs = out == 2 ? qs.stride(0) : 0;
    p.M = (int) M;
    p.N = (int) N;
    p.K = (int) K;
    p.out = (int) out;
    p.gelu = gelu ? 1 : 0;
    p.resid = resid;
    p.group_m = (int) group_m;
    const cudaError_t err = int8_gemm_sm80(p, (int) tile, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(err == cudaSuccess, "int8_gemm_group80: ", cudaGetErrorString(err));
    return {y, qs};
}

std::vector<int64_t> op_int8_gemm_group80_info(const int64_t tile) {
    int info[9];
    if (!int8_gemm_sm80_info((int) tile, info)) return {};
    return std::vector<int64_t>(info, info + 9);
}

bool op_int8_gemm_group80_loaded() {
    return int8_gemm_sm80_loaded();
}

}  // namespace pixal3d_kernels

TORCH_LIBRARY_FRAGMENT(P3K_SM80_LIB, m) {
    // tile / group_m: the tile configuration (the install's race picks one per layer shape) and the tile order
    m.def("int8_gemm_group80(Tensor xq, Tensor xs, Tensor wq, Tensor ws, Tensor? bias, bool gelu, int out, "
          "Tensor? rx, Tensor? rg, Tensor? r2, int tile, int group_m) -> (Tensor, Tensor)");
    m.def("int8_gemm_group80_info(int tile) -> int[]", &pixal3d_kernels::op_int8_gemm_group80_info);
    m.def("int8_gemm_group80_loaded() -> bool", &pixal3d_kernels::op_int8_gemm_group80_loaded);
}

TORCH_LIBRARY_IMPL(P3K_SM80_LIB, CUDA, m) {
    m.impl("int8_gemm_group80", &pixal3d_kernels::op_int8_gemm_group80);
}
