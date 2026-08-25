"""bench_ext_vs_subprocess.py — measure how much faster the C++ extension is
than the subprocess wrapper. This is the whole reason we built the extension.
"""
from __future__ import annotations
import os, pathlib, sys, time

# Bootstrap env (same as build_and_load_ext.py)
_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import numpy as np

# ---- Import BOTH wrappers ----
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import freetoken_vulkan as fv_subprocess   # the disk-round-trip wrapper

from torch.utils.cpp_extension import load
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])

print("(re-loading C++ extension — should be fast now, cached build)...")
t0 = time.time()
fv_ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(HERE / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)
print(f"  ext load: {time.time()-t0:.2f}s")

# ---- Shape: something we'd use in a real transformer forward pass ----
torch.manual_seed(0)
x = torch.randn(16, 2048)      # 16 tokens, hidden 2048 (Qwen-ish size)
w = torch.rand(2048) + 0.5

# Correctness sanity
y_ext = fv_ext.rmsnorm(x, w, 1e-6)
y_sub = fv_subprocess.rmsnorm(x, w, 1e-6)
y_ref = torch.nn.functional.rms_norm(x, (2048,), w, eps=1e-6)
print(f"correctness — ext vs ref: {(y_ext - y_ref).abs().max().item():.2e}")
print(f"correctness — sub vs ref: {(y_sub - y_ref).abs().max().item():.2e}")

# ---- Benchmark ----
N_WARM = 3
N_TIME = 30

# subprocess wrapper: warmup + timed
for _ in range(N_WARM):
    fv_subprocess.rmsnorm(x, w, 1e-6)

t0 = time.time()
for _ in range(N_TIME):
    fv_subprocess.rmsnorm(x, w, 1e-6)
t_sub = (time.time() - t0) / N_TIME * 1000

# C++ extension: warmup + timed
for _ in range(N_WARM):
    fv_ext.rmsnorm(x, w, 1e-6)

t0 = time.time()
for _ in range(N_TIME):
    fv_ext.rmsnorm(x, w, 1e-6)
t_ext = (time.time() - t0) / N_TIME * 1000

# Also time pure PyTorch CPU for reference
for _ in range(N_WARM):
    torch.nn.functional.rms_norm(x, (2048,), w, eps=1e-6)
t0 = time.time()
for _ in range(N_TIME):
    torch.nn.functional.rms_norm(x, (2048,), w, eps=1e-6)
t_torch = (time.time() - t0) / N_TIME * 1000

print()
print(f"=== RMSNorm on x[16, 2048], {N_TIME}-iter mean ===")
print(f"  subprocess wrapper:  {t_sub:8.2f} ms/call")
print(f"  C++ extension:       {t_ext:8.2f} ms/call")
print(f"  PyTorch CPU ref:     {t_torch:8.2f} ms/call")
print()
print(f"  Extension speedup vs subprocess: {t_sub/t_ext:.1f}x")
print(f"  Extension vs PyTorch CPU:        {t_torch/t_ext:.2f}x  "
      "(negative = we're slower)")
