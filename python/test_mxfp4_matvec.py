"""test_mxfp4_matvec.py — verify the Vulkan MXFP4 matvec kernel against
PyTorch reference on ACTUAL gpt-oss-20b expert weights.

Uses a subset of expert 0's gate_up_proj (well-defined shape [M, N_blocks, 16]
uint8 + scales [M, N_blocks] uint8) and a random activation vector. Computes:

    y_ref = dequant(W).mm(x)     (via mxfp4_ref.py + torch.matmul)
    y_gpu = ext.mxfp4_matvec(blocks, scales, x)
    diff  = |y_ref - y_gpu|

Expected error: dominated by FP32 accumulation reordering. Because we've
already verified our dequant is bit-exact with HF's reference (see e028),
the only source of difference here is the summation order of ~2880 FMAs —
which for FP32 is ~sqrt(K) * epsilon ~ 6e-6 relative.
"""
from __future__ import annotations
import os, pathlib, sys, time, json, math

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))
from gpt_oss.mxfp4_ref import mxfp4_dequant

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("Loading Vulkan extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

# ---- Load a real MXFP4 tensor from gpt-oss-20b ----
MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
with open(MODEL / "model.safetensors.index.json") as f:
    idx = json.load(f)["weight_map"]

def get(name):
    with safe_open(MODEL / idx[name], framework="pt") as f:
        return f.get_tensor(name)

blocks_all = get("model.layers.0.mlp.experts.gate_up_proj_blocks")  # [32, 5760, 90, 16] u8
scales_all = get("model.layers.0.mlp.experts.gate_up_proj_scales")  # [32, 5760, 90]     u8

# One expert
E, OUT, NB, PACK = blocks_all.shape
K = NB * 32
print(f"blocks: {list(blocks_all.shape)}  {blocks_all.dtype}")
print(f"scales: {list(scales_all.shape)}  {scales_all.dtype}")
print(f"Full weight: E={E}, out={OUT}, K={K}  (K = NB*32 = {NB}*32)")

# ---- Test cases: various M sizes ----
CASES = [
    (   1, "single row"),
    (   8, "small"),
    (  64, "1 workgroup"),
    ( 128, "2 workgroups"),
    ( 512, "medium"),
    (5760, "full row count"),
]

torch.manual_seed(0xC0DE)
x = torch.randn(K, dtype=torch.float32) * 0.1
print(f"\nActivation x: [{K}] fp32, mean={x.mean().item():.3f}, std={x.std().item():.3f}")

all_ok = True
for M, label in CASES:
    # Subset expert 0, first M rows
    blocks = blocks_all[0, :M].contiguous()        # [M, NB, 16]
    scales = scales_all[0, :M].contiguous()        # [M, NB]

    # Reference: dequantize then matmul on CPU
    W = mxfp4_dequant(blocks, scales)              # [M, K] fp32
    y_ref = W @ x                                   # [M]

    t0 = time.time()
    y_gpu = ext.mxfp4_matvec(blocks, scales, x)
    t_gpu = time.time() - t0

    abs_diff = (y_gpu - y_ref).abs()
    rel_diff = abs_diff / y_ref.abs().clamp_min(1e-6)
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    ok = torch.allclose(y_gpu, y_ref, rtol=5e-5, atol=1e-5)
    tag = "PASS" if ok else "FAIL"
    all_ok &= ok
    print(f"  [{label:16s}]  M={M:5d} K={K}  "
          f"max|D|={max_abs:.2e}  max|D/y|={max_rel:.2e}  {tag}  "
          f"gpu={t_gpu*1000:.1f}ms")

print("\nALL PASS" if all_ok else "\nSOME FAILED")
