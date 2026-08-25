"""test_moe_mlp_lg.py — verify Dff-blocked MoE MLP at large shapes.

Target: real Phi-3.5-MoE geometry (D=4096, Dff=6400, E=16, K=2), plus a
regression case at the small-shader boundary to make sure the Dff-blocked
variant gives the same answer as the direct variant at overlap.
"""
from __future__ import annotations
import math, os, pathlib, sys, time

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

print("Loading extension (with moe_mlp_lg)...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(HERE / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)


def moe_mlp_ref(x, indices, weights, W_gate, W_up, W_down):
    T, D = x.shape
    K = indices.shape[1]
    y = torch.zeros(T, D, dtype=torch.float32)
    for t in range(T):
        for k in range(K):
            e = indices[t, k].item()
            w = weights[t, k].item()
            g = W_gate[e] @ x[t]
            u = W_up[e]   @ x[t]
            m = F.silu(g) * u
            y[t] += w * (W_down[e] @ m)
    return y


# (T, D, Dff, E, K, label)
CASES = [
    (   4,  128,  256,  4, 2, "small (regression vs moe_mlp)"),
    (   2,  512, 1024,  4, 2, "medium — beyond small shader limits"),
    (   2, 1024, 2048,  4, 2, "wide"),
    (   1, 2048, 3072,  8, 2, "large single-token"),
    (   1, 4096, 6400, 16, 2, "PHI-3.5-MoE shape (real)"),
]


def run_case(T, D, Dff, E, K, label):
    torch.manual_seed(0xB0B + T * 131 + D * 17)
    x = torch.randn(T, D) * 0.1
    W_gate = torch.randn(E, Dff, D) * (1.0 / math.sqrt(D))
    W_up   = torch.randn(E, Dff, D) * (1.0 / math.sqrt(D))
    W_down = torch.randn(E, D, Dff) * (1.0 / math.sqrt(Dff))

    logits = torch.randn(T, E) * 2.0
    probs = torch.softmax(logits, dim=-1)
    weights, indices = torch.topk(probs, K, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)

    t_ref_start = time.time()
    y_ref = moe_mlp_ref(x, indices, weights, W_gate, W_up, W_down)
    t_ref = time.time() - t_ref_start

    t_gpu_start = time.time()
    y_gpu = ext.moe_mlp_lg(x, indices, weights, W_gate, W_up, W_down)
    t_gpu = time.time() - t_gpu_start

    max_abs = (y_gpu - y_ref).abs().max().item()
    ok = torch.allclose(y_gpu, y_ref, rtol=1e-3, atol=1e-3)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{label:40s}]  T={T:2d} D={D:5d} Dff={Dff:5d} E={E:2d} K={K}  "
          f"max|D|={max_abs:.2e}  {tag}  gpu={t_gpu*1000:.0f}ms  ref={t_ref*1000:.0f}ms")
    return ok


all_ok = True
for c in CASES:
    all_ok &= run_case(*c)
print("\nALL PASS" if all_ok else "\nSOME FAILED")
