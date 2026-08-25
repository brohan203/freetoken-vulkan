"""test_rope_partial.py — verify partial RoPE vs transformers reference."""
from __future__ import annotations
import os, pathlib, sys, math, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])

ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)


def rotate_half(x):
    d = x.shape[-1]
    return torch.cat((-x[..., d//2:], x[..., :d//2]), dim=-1)


def rope_ref(x, cos, sin, rotary_dim):
    """Reference: apply rotate_half to first rotary_dim of head_dim,
    passthrough the rest. cos, sin: [S, rotary_dim]."""
    B, H, S, D = x.shape
    x_rot  = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    # Broadcast cos/sin from [S, rot] to [1, 1, S, rot].
    cos_b = cos[None, None, :, :]
    sin_b = sin[None, None, :, :]
    y_rot = x_rot * cos_b + rotate_half(x_rot) * sin_b
    return torch.cat([y_rot, x_pass], dim=-1)


# (B, H, S, D, rotary_dim, label)
CASES = [
    (1, 1,   64, 64, 16, "gpt-oss partial 25%"),
    (1, 8,  128, 64, 16, "H=8 partial"),
    (2, 4,  256, 64, 32, "batch=2 partial 50%"),
    (1, 16,  32, 128, 128, "full rotary (rotary_dim = D)"),
    (1, 8,  128, 64, 16, "H_q=8 real gpt-oss"),
]


all_ok = True
for B, H, S, D, rd, label in CASES:
    torch.manual_seed(B * H * S + rd)
    x = torch.randn(B, H, S, D)
    # Generate reasonable cos/sin (as if from RoPE base=10000 with linear positions).
    theta = 10000.0 ** (-torch.arange(0, rd, 2).float() / rd)  # [rd/2]
    pos = torch.arange(S).float()
    freqs = torch.outer(pos, theta)                            # [S, rd/2]
    freqs = torch.cat([freqs, freqs], dim=-1)                  # [S, rd]
    cos, sin = freqs.cos(), freqs.sin()

    y_ref = rope_ref(x, cos, sin, rd)
    y_gpu = ext.rope_partial(x, cos, sin, rd)
    diff = (y_gpu - y_ref).abs().max().item()
    ok = torch.allclose(y_gpu, y_ref, rtol=1e-5, atol=1e-6)
    tag = "PASS" if ok else "FAIL"
    all_ok &= ok
    print(f"  [{label:28s}]  B={B} H={H:2d} S={S:4d} D={D} rot={rd}  "
          f"max|D|={diff:.2e}  {tag}")

print("\nALL PASS" if all_ok else "\nSOME FAILED")
