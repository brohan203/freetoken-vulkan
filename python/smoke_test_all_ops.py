"""smoke_test_all_ops.py — build extension and exercise every op.

Verifies correctness vs PyTorch for the whole op set now that we've added
matmul, softmax, flash_attention, swiglu, moe_router, moe_mlp on top of
the RMSNorm POC.
"""
from __future__ import annotations
import math
import os
import pathlib
import sys
import time

# Bootstrap env
_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])

print("Building freetoken_vulkan_ext (extended op set)...")
t0 = time.time()
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(HERE / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)
print(f"  build/load: {time.time()-t0:.1f}s")
print(f"  ops available: {[n for n in dir(ext) if not n.startswith('_')]}")

torch.manual_seed(42)

def check(name, gpu, ref, tol_abs=1e-4, tol_rel=1e-3):
    diff = (gpu - ref).abs()
    max_abs = diff.max().item()
    tag = "PASS" if torch.allclose(gpu, ref, rtol=tol_rel, atol=tol_abs) else "FAIL"
    print(f"  {name:22s}  max|D| = {max_abs:.3e}  {tag}")
    return tag == "PASS"

all_ok = True

# ---- rmsnorm ----
x = torch.randn(4, 128); w = torch.rand(128) + 0.5
gpu = ext.rmsnorm(x, w, 1e-6)
ref = F.rms_norm(x, (128,), w, eps=1e-6)
all_ok &= check("rmsnorm", gpu, ref)

# ---- matmul ----
A = torch.randn(64, 128); B = torch.randn(128, 64)
gpu = ext.matmul(A, B)
ref = A @ B
all_ok &= check("matmul", gpu, ref, tol_abs=1e-3)

# ---- softmax ----
x = torch.randn(8, 512)
gpu = ext.softmax(x)
ref = torch.softmax(x, dim=-1)
all_ok &= check("softmax", gpu, ref, tol_abs=1e-6)

# ---- flash_attention ----
S, D = 128, 64
Q = torch.randn(S, D) * 0.1
K = torch.randn(S, D) * 0.1
V = torch.randn(S, D) * 0.1
scale = 1.0 / math.sqrt(D)
gpu = ext.flash_attention(Q, K, V, scale)
scores = (Q @ K.T) * scale
weights = torch.softmax(scores, dim=-1)
ref = weights @ V
all_ok &= check("flash_attention", gpu, ref, tol_abs=1e-4)

# ---- swiglu ----
g = torch.randn(1024); u = torch.randn(1024)
gpu = ext.swiglu(g, u)
ref = F.silu(g) * u
all_ok &= check("swiglu", gpu, ref, tol_abs=1e-6)

# ---- moe_router ----
logits = torch.randn(16, 8) * 2.0
idx_gpu, w_gpu = ext.moe_router(logits, 2)
probs = torch.softmax(logits, dim=-1)
w_ref, idx_ref = torch.topk(probs, 2, dim=-1)
w_ref = w_ref / w_ref.sum(-1, keepdim=True)
idx_match = (idx_gpu == idx_ref).all().item()
w_diff = (w_gpu - w_ref).abs().max().item()
tag = "PASS" if idx_match and w_diff < 1e-5 else "FAIL"
print(f"  {'moe_router':22s}  idx_ok={idx_match}  w_diff={w_diff:.2e}  {tag}")
all_ok &= (tag == "PASS")

# ---- moe_mlp ----
T, D, Dff, E, K = 8, 64, 128, 4, 2
x = torch.randn(T, D) * 0.1
W_gate = torch.randn(E, Dff, D) * (1/math.sqrt(D))
W_up   = torch.randn(E, Dff, D) * (1/math.sqrt(D))
W_down = torch.randn(E, D, Dff) * (1/math.sqrt(Dff))
logits = torch.randn(T, E) * 2.0
probs = torch.softmax(logits, dim=-1)
w_ref, idx_ref = torch.topk(probs, K, dim=-1)
w_ref = w_ref / w_ref.sum(-1, keepdim=True)

gpu = ext.moe_mlp(x, idx_ref, w_ref, W_gate, W_up, W_down)

# Reference
y_ref = torch.zeros(T, D)
for t in range(T):
    for k in range(K):
        e = idx_ref[t, k].item()
        w = w_ref[t, k].item()
        g_ = W_gate[e] @ x[t]
        u_ = W_up[e] @ x[t]
        m_ = F.silu(g_) * u_
        o_ = W_down[e] @ m_
        y_ref[t] += w * o_
all_ok &= check("moe_mlp", gpu, y_ref, tol_abs=1e-4)

print()
print("ALL PASS" if all_ok else "SOME FAILED")
