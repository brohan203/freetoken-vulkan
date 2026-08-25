"""test_flash_attention_mh.py — verify multi-head + causal attention.

Compares our shader against explicit softmax(Q@K^T/sqrt(d) [+ causal mask]) @ V
for various (B, H, S, D) shapes, with and without causal masking.
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

# Rebuild extension (adds flash_attention_mh_f32.comp)
print("Loading extension (with multi-head/causal support)...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(HERE / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

# Cases: (B, H, S, D, causal, label)
CASES = [
    (1, 1,   64,  64, False, "single head, no mask"),
    (1, 1,   64,  64, True,  "single head, CAUSAL"),
    (1, 8,  128, 128, False, "8 heads"),
    (1, 8,  128, 128, True,  "8 heads CAUSAL"),
    (2, 4,  256,  64, False, "batch=2, 4 heads"),
    (2, 4,  256,  64, True,  "batch=2 CAUSAL"),
    (1, 8, 1024, 128, True,  "S=1024 CAUSAL (LDS scale test)"),
]


def naive_mh_ref(Q, K, V, scale, causal):
    """[B, H, S, D] → [B, H, S, D] via explicit softmax."""
    # scores: [B, H, S, S]
    scores = torch.einsum('bhsd,bhtd->bhst', Q, K) * scale
    if causal:
        S = Q.size(-2)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum('bhst,bhtd->bhsd', weights, V)


def run_case(B, H, S, D, causal, label):
    torch.manual_seed(0xABBA + B * 131 + H * 17 + S)
    Q = torch.randn(B, H, S, D) * 0.1
    K = torch.randn(B, H, S, D) * 0.1
    V = torch.randn(B, H, S, D) * 0.1
    scale = 1.0 / math.sqrt(D)

    y_ref = naive_mh_ref(Q, K, V, scale, causal)
    y_gpu = ext.flash_attention_mh(Q, K, V, scale, causal)

    max_abs = (y_gpu - y_ref).abs().max().item()
    ok = torch.allclose(y_gpu, y_ref, rtol=1e-3, atol=1e-4)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{label:36s}]  B={B} H={H} S={S:4d} D={D:3d}  "
          f"max|D|={max_abs:.2e}  {tag}")
    return ok


all_ok = True
for c in CASES:
    all_ok &= run_case(*c)
print("\nALL PASS" if all_ok else "\nSOME FAILED")
