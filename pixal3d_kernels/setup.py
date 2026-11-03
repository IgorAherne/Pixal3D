"""pixal3d_kernels: Spellcast3D's own CUDA kernels for the Pixal3D fork, one small torch extension (`pixal3d_kernels._C`).

Built for the architectures in TORCH_CUDA_ARCH_LIST (torch's own variable, e.g. "6.1;7.5;8.0;8.6;8.9;9.0;12.0+PTX"
for the product's wheel; without it, the cards of the building machine). Plain CUDA C++17, no --use_fast_math (the
GELU's expf must be the accurate one), MSVC- and gcc-clean. Environment:
    PIXAL3D_KERNELS_PTXAS_V=1   print ptxas's register / shared-memory / spill report per kernel and architecture
    NVCC_THREADS                nvcc's parallel -gencode compiles (default: the CPU count)
"""

import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if os.name == "nt":
    CXX_FLAGS = ["/O2", "/std:c++17", "/permissive-"]
else:
    CXX_FLAGS = ["-O3", "-std=c++17"]
CXX_FLAGS += ["-DPy_LIMITED_API=0x03090000"]

NVCC_FLAGS = ["-O3", "-std=c++17", f"--threads={os.getenv('NVCC_THREADS', os.cpu_count())}",
              # the kernels' template variants (scale layouts) leave some locals unused
              "-diag-suppress=177", "-diag-suppress=550"]
if os.name == "nt":
    NVCC_FLAGS += ["-D_WIN32=1", "-DUSE_CUDA=1"]      # https://github.com/pytorch/pytorch/issues/148317
if os.getenv("PIXAL3D_KERNELS_PTXAS_V", "") == "1":
    NVCC_FLAGS += ["-Xptxas=-v"]

setup(
    name="pixal3d_kernels",
    version="0.1.0" + os.environ.get("PIXAL3D_KERNELS_VERSION_SUFFIX", ""),
    description="Spellcast3D's CUDA kernels for Pixal3D below sm_80: a group-scaled W8A8 GEMM (Turing IMMA, Pascal dp4a)",
    packages=find_packages(),
    python_requires=">=3.9",
    ext_modules=[
        CUDAExtension(
            name="pixal3d_kernels._C",
            sources=["csrc/ops.cpp", "csrc/int8_gemm_imma.cu", "csrc/int8_gemm_dp4a.cu"],
            extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
