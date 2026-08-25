"""setup_ext.py — build the freetoken_vulkan_ext C++ extension.

Usage (from python/ dir):
    python setup_ext.py build_ext --inplace

This produces `freetoken_vulkan_ext.<pyver>.pyd` next to this file,
which can be imported directly:

    import freetoken_vulkan_ext
    y = freetoken_vulkan_ext.rmsnorm(x, weight, eps=1e-6)

Requires: MSVC (VS Build Tools 2022), Vulkan SDK, PyTorch (has cpp_extension).
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
vulkan_sdk = os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")

setup(
    name="freetoken_vulkan_ext",
    ext_modules=[
        CppExtension(
            name="freetoken_vulkan_ext",
            sources=[str(HERE / "ext_module.cpp")],
            include_dirs=[
                os.path.join(vulkan_sdk, "Include"),
                str(REPO / "include"),
            ],
            library_dirs=[os.path.join(vulkan_sdk, "Lib")],
            libraries=["vulkan-1"],
            extra_compile_args={"cxx": ["/std:c++17", "/O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
