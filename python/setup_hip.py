"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Tuple

import pathlib
import os
import re
import itertools
import subprocess
import platform

import setuptools
import argparse
import torch
import torch.utils.cpp_extension as torch_cpp_ext
import generate_dispatch_inc

root = pathlib.Path(__name__).parent

enable_bf16 = 1
enable_fp8 = 0

if enable_bf16:
    torch_cpp_ext.COMMON_HIPCC_FLAGS.append("-DFLASHINFER_ENABLE_BF16")
if enable_fp8:
    torch_cpp_ext.COMMON_HIPCC_FLAGS.append("-DFLASHINFER_ENABLE_FP8")

torch_cpp_ext.COMMON_HIPCC_FLAGS.append("-DFLASHINFER_WITH_HIP")

def write_if_different(path: pathlib.Path, content: str) -> None:
    if path.exists():
        with open(path, "r") as f:
            if f.read() == content:
                return
    with open(path, "w") as f:
        f.write(content)

def get_instantiation_cu() -> None:
    prefix = "csrc/generated"
    (root / prefix).mkdir(parents=True, exist_ok=True)
    logits_hooks = os.environ.get("FLASHINFER_LOGITS_POST_HOOKS", "0,1").split(",")
    head_dims = os.environ.get("FLASHINFER_HEAD_DIMS", "64,128,256").split(",")
    pos_encoding_modes = os.environ.get("FLASHINFER_POS_ENCODING_MODES", "0,1,2").split(
        ","
    )
    allow_fp16_qk_reduction_options = os.environ.get(
        "FLASHINFER_ALLOW_FP16_QK_REDUCTION_OPTIONS", "0,1"
    ).split(",")
    mask_modes = os.environ.get("FLASHINFER_MASK_MODES", "0,1,2").split(",")
    # dispatch.inc
    path = root / prefix / "dispatch.inc"
    write_if_different(
        path,
        generate_dispatch_inc.get_dispatch_inc_str(
            argparse.Namespace(
                head_dims=map(int, head_dims),
                logits_post_hooks=map(int, logits_hooks),
                pos_encoding_modes=map(int, pos_encoding_modes),
                allow_fp16_qk_reductions=map(int, allow_fp16_qk_reduction_options),
                mask_modes=map(int, mask_modes),
            )
        ),
    )

def get_version():
    version = os.getenv("FLASHINFER_BUILD_VERSION")
    if version is None:
        with open(root / "version.txt") as f:
            version = f.read().strip()
    return version


def generate_build_meta() -> None:
    d = {}
    version = get_version()
    d["torch"] = torch.__dcu_version__
    d["python"] = platform.python_version()
    with open(root / "flashinfer/_build_meta.py", "w") as f:
        f.write(f"__version__ = {version!r}\n")
        f.write(f"build_meta = {d!r}")


def remove_unwanted_pytorch_hipcc_flags():
    REMOVE_HIPCC_FLAGS = [
        "-D__HIP_NO_HALF_OPERATORS__=1",
        "-D__HIP_NO_HALF_CONVERSIONS__=1",
    ]
    for flag in REMOVE_HIPCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_HIPCC_FLAGS.remove(flag)
        except ValueError:
            pass


class NinjaBuildExtension(torch_cpp_ext.BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            max_num_jobs_cores = max(1, os.cpu_count())
            os.environ["MAX_JOBS"] = str(max_num_jobs_cores)

        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    remove_unwanted_pytorch_hipcc_flags()
    generate_build_meta()
    get_instantiation_cu()
    include_dirs = [
        str(root.resolve() / "include"),
    ]
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-Wno-switch-bool",
            "-Wno-macro-redefined",
            "-Wno-deprecated-declarations",
        ],
        "nvcc": [
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-D__HIP_PLATFORM_HCC__=1",
            "--offload-arch=gfx928",
            "--gpu-max-threads-per-block=1024",
            "-Wno-macro-redefined",
            "-Wno-deprecated-declarations",
        ],
    }
    ext_modules = []
    ext_modules.append(
        torch_cpp_ext.CUDAExtension(
            name="flashinfer._kernels",
            sources=[
                "csrc/sampling.cu",
                "csrc/norm.cu",
                "csrc/activation.cu",
                "csrc/rope.cu",
                "csrc/quantization.cu",
                "csrc/flashinfer_ops.cu",
            ],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    )
    setuptools.setup(
        name="flashinfer",
        version=get_version(),
        packages=setuptools.find_packages(),
        author="FlashInfer team",
        license="Apache License 2.0",
        description="FlashInfer: Kernel Library for LLM Serving",
        url="https://github.com/flashinfer-ai/flashinfer",
        python_requires=">=3.8",
        ext_modules=ext_modules,
        cmdclass={"build_ext": NinjaBuildExtension},
    )
