"""demo_fast_transformer.py — full transformer blocks with the C++ extension.

Compares three code paths:
  1. Subprocess wrapper (freetoken_vulkan.py) — ~130 ms/kernel overhead
  2. C++ Torch extension (freetoken_vulkan_ext) — ~1 ms/kernel overhead
  3. Pure PyTorch CPU baseline

Runs the same two blocks used in demo_mini_transformer.py and
demo_moe_transformer.py, but via the extension. Same weights, same math.
"""
from __future__ import annotations
import math
import os
import pathlib
import sys
import time

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
PYDIR = REPO / "python"
vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])

sys.path.insert(0, str(PYDIR))
import freetoken_vulkan as fv_sub   # subprocess wrapper

# Load the C++ extension (cached build, fast)
print("Loading C++ extension...")
t0 = time.time()
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)
print(f"  loaded in {time.time()-t0:.1f}s")

# ============================================================
# Reference computations in pure PyTorch (baselines).
# ============================================================
def dense_torch(x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2, Wgate, Wup, Wdown):
    D = x_in.shape[-1]
    x1 = F.rms_norm(x_in, (D,), w_ln1, eps=1e-6)
    q, k, v = x1 @ Wq, x1 @ Wk, x1 @ Wv
    scale = 1.0 / math.sqrt(q.shape[-1])
    weights = torch.softmax((q @ k.T) * scale, dim=-1)
    a = weights @ v
    o = a @ Wo
    x = x_in + o
    x2 = F.rms_norm(x, (D,), w_ln2, eps=1e-6)
    m = F.silu(x2 @ Wgate) * (x2 @ Wup)
    y = m @ Wdown
    return x + y


def moe_torch(x_in, w_ln1, Wq, Wk, Wv, Wo,
              w_ln2, W_router, W_gate, W_up, W_down, K):
    D = x_in.shape[-1]
    x1 = F.rms_norm(x_in, (D,), w_ln1, eps=1e-6)
    q, k, v = x1 @ Wq, x1 @ Wk, x1 @ Wv
    scale = 1.0 / math.sqrt(q.shape[-1])
    weights = torch.softmax((q @ k.T) * scale, dim=-1)
    a = weights @ v
    o = a @ Wo
    x = x_in + o
    x2 = F.rms_norm(x, (D,), w_ln2, eps=1e-6)

    logits = x2 @ W_router
    probs = torch.softmax(logits, dim=-1)
    w_r, indices = torch.topk(probs, K, dim=-1)
    w_r = w_r / w_r.sum(-1, keepdim=True)

    T = x2.shape[0]
    m = torch.zeros_like(x2)
    for t in range(T):
        for kk in range(K):
            e = indices[t, kk].item()
            w = w_r[t, kk].item()
            g_ = W_gate[e] @ x2[t]
            u_ = W_up[e]   @ x2[t]
            mid = F.silu(g_) * u_
            m[t] += w * (W_down[e] @ mid)
    return x + m


# ============================================================
# Same block via the two Vulkan paths.
# ============================================================
def dense_via(fv_module, x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2, Wgate, Wup, Wdown):
    """Works with either the subprocess module or the C++ extension —
    both expose the same op names."""
    scale = 1.0 / math.sqrt(x_in.shape[-1])
    x1 = fv_module.rmsnorm(x_in, w_ln1)
    q = fv_module.matmul(x1, Wq)
    k = fv_module.matmul(x1, Wk)
    v = fv_module.matmul(x1, Wv)
    a = fv_module.flash_attention(q, k, v, scale) if fv_module is ext \
        else fv_module.flash_attention(q, k, v)
    o = fv_module.matmul(a, Wo)
    x = x_in + o
    x2 = fv_module.rmsnorm(x, w_ln2)
    g_ = fv_module.matmul(x2, Wgate)
    u_ = fv_module.matmul(x2, Wup)
    m  = fv_module.swiglu(g_, u_)
    y  = fv_module.matmul(m, Wdown)
    return x + y


def moe_via(fv_module, x_in, w_ln1, Wq, Wk, Wv, Wo,
            w_ln2, W_router, W_gate, W_up, W_down, K):
    scale = 1.0 / math.sqrt(x_in.shape[-1])
    x1 = fv_module.rmsnorm(x_in, w_ln1)
    q = fv_module.matmul(x1, Wq)
    k = fv_module.matmul(x1, Wk)
    v = fv_module.matmul(x1, Wv)
    a = fv_module.flash_attention(q, k, v, scale) if fv_module is ext \
        else fv_module.flash_attention(q, k, v)
    o = fv_module.matmul(a, Wo)
    x = x_in + o
    x2 = fv_module.rmsnorm(x, w_ln2)
    logits = fv_module.matmul(x2, W_router)
    if fv_module is ext:
        indices, weights = fv_module.moe_router(logits, K)
    else:
        indices, weights = fv_module.moe_router(logits, K)
    m = fv_module.moe_mlp(x2, indices, weights, W_gate, W_up, W_down)
    return x + m


def bench(name, fn, warmup=1, n=3):
    for _ in range(warmup): fn()
    t0 = time.time()
    for _ in range(n): fn()
    return (time.time() - t0) / n * 1000


def main():
    torch.manual_seed(0xF00DBABE)

    # Match the demo_moe_transformer shapes so this is comparable.
    S = 32
    D = 128
    Dff = 256
    E = 8
    K = 2
    print(f"Shapes: S={S}  D={D}  D_ff={Dff}  experts={E}  K={K}\n")

    x_in = torch.randn(S, D) * 0.1
    w_ln1 = torch.rand(D) * 0.5 + 0.75
    w_ln2 = torch.rand(D) * 0.5 + 0.75
    Wq = torch.randn(D, D) * (1/math.sqrt(D))
    Wk = torch.randn(D, D) * (1/math.sqrt(D))
    Wv = torch.randn(D, D) * (1/math.sqrt(D))
    Wo = torch.randn(D, D) * (1/math.sqrt(D))
    Wgate_dense = torch.randn(D, Dff) * (1/math.sqrt(D))
    Wup_dense   = torch.randn(D, Dff) * (1/math.sqrt(D))
    Wdown_dense = torch.randn(Dff, D) * (1/math.sqrt(Dff))
    W_router = torch.randn(D, E) * (1/math.sqrt(D))
    W_gate = torch.randn(E, Dff, D) * (1/math.sqrt(D))
    W_up   = torch.randn(E, Dff, D) * (1/math.sqrt(D))
    W_down = torch.randn(E, D, Dff) * (1/math.sqrt(Dff))

    # Correctness cross-check
    print("--- DENSE transformer block correctness ---")
    y_torch = dense_torch(x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2,
                          Wgate_dense, Wup_dense, Wdown_dense)
    y_ext = dense_via(ext, x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2,
                      Wgate_dense, Wup_dense, Wdown_dense)
    print(f"  ext vs torch: max|D| = {(y_ext - y_torch).abs().max():.3e}")

    print("\n--- MoE transformer block correctness ---")
    y_torch_moe = moe_torch(x_in, w_ln1, Wq, Wk, Wv, Wo,
                            w_ln2, W_router, W_gate, W_up, W_down, K)
    y_ext_moe = moe_via(ext, x_in, w_ln1, Wq, Wk, Wv, Wo,
                         w_ln2, W_router, W_gate, W_up, W_down, K)
    print(f"  ext vs torch: max|D| = {(y_ext_moe - y_torch_moe).abs().max():.3e}")

    # Timings
    print("\n--- DENSE transformer wall time ---")
    t_torch = bench("torch",
                    lambda: dense_torch(x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2,
                                        Wgate_dense, Wup_dense, Wdown_dense))
    t_ext   = bench("ext",
                    lambda: dense_via(ext, x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2,
                                       Wgate_dense, Wup_dense, Wdown_dense))
    print(f"  PyTorch CPU:         {t_torch:8.2f} ms")
    print(f"  Vulkan (extension):  {t_ext:8.2f} ms")

    # subprocess wrapper — SLOW. n=1 to keep this reasonable.
    t_sub = bench("sub",
                  lambda: dense_via(fv_sub, x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2,
                                     Wgate_dense, Wup_dense, Wdown_dense),
                  warmup=0, n=1)
    print(f"  Vulkan (subprocess): {t_sub:8.2f} ms  ({t_sub/t_ext:.0f}x slower than extension)")

    print("\n--- MoE transformer wall time ---")
    t_torch_moe = bench("torch_moe",
                        lambda: moe_torch(x_in, w_ln1, Wq, Wk, Wv, Wo,
                                          w_ln2, W_router, W_gate, W_up, W_down, K))
    t_ext_moe = bench("ext_moe",
                      lambda: moe_via(ext, x_in, w_ln1, Wq, Wk, Wv, Wo,
                                       w_ln2, W_router, W_gate, W_up, W_down, K))
    print(f"  PyTorch CPU:         {t_torch_moe:8.2f} ms")
    print(f"  Vulkan (extension):  {t_ext_moe:8.2f} ms")

    t_sub_moe = bench("sub_moe",
                      lambda: moe_via(fv_sub, x_in, w_ln1, Wq, Wk, Wv, Wo,
                                       w_ln2, W_router, W_gate, W_up, W_down, K),
                      warmup=0, n=1)
    print(f"  Vulkan (subprocess): {t_sub_moe:8.2f} ms  ({t_sub_moe/t_ext_moe:.0f}x slower than extension)")


if __name__ == "__main__":
    main()
