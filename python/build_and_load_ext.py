"""build_and_load_ext.py — JIT-build + import the Vulkan Torch extension.

Uses torch.utils.cpp_extension.load(), which handles the Windows/MSVC
quirks better than the setup.py path (auto-detects ninja from the venv,
threads through /std:c++17 correctly, honors extra_include_paths).

First call: compiles the .cpp into a cached .pyd under ~/.cache/torch_extensions/
Subsequent calls: reuses the cached .pyd (fast).
"""
from __future__ import annotations
import os
import pathlib
import sys
import time

# Add venv Scripts to PATH so torch can find ninja
_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")

# Vulkan env
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]

# Silence torch's compiler-version probe which spams warnings on Windows
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

# OpenBLAS threadpool fix
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])

print("Building freetoken_vulkan_ext (JIT compile)...")
t0 = time.time()
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(HERE / "ext_module.cpp")],
    extra_include_paths=[
        str(vulkan_sdk / "Include"),
        str(REPO / "include"),
    ],
    extra_ldflags=[
        f"/LIBPATH:{vulkan_sdk / 'Lib'}",
        "vulkan-1.lib",
    ],
    # torch 2.13 sets /std:c++20 for us — don't override
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=True,
)
print(f"Build complete in {time.time()-t0:.1f}s.")
print(f"Extension: {ext}")
print(f"Available ops: {[n for n in dir(ext) if not n.startswith('_')]}")

# ---- Sanity check: run RMSNorm through the extension ----
print("\n=== Sanity check ===")
torch.manual_seed(0)
x = torch.randn(4, 128)
w = torch.rand(128) + 0.5

y_ext = ext.rmsnorm(x, w, 1e-6)
y_ref = torch.nn.functional.rms_norm(x, (128,), w, eps=1e-6)
diff = (y_ext - y_ref).abs().max().item()
print(f"max |ext - torch.rms_norm| = {diff:.3e}")
if diff < 1e-5:
    print("PASS — extension gives correct result")
else:
    print("FAIL")
