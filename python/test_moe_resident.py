"""test_moe_resident.py — verify moe_mlp_gpt_oss_resident produces IDENTICAL
output to moe_mlp_gpt_oss (same math, same weights, just weights uploaded
once instead of per-call).
"""
from __future__ import annotations
import os, pathlib, sys, time, json

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from safetensors import safe_open

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("Loading extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
with open(MODEL / "model.safetensors.index.json") as f:
    idx = json.load(f)["weight_map"]

def load_tensor(name):
    with safe_open(MODEL / idx[name], framework="pt") as f:
        return f.get_tensor(name)

# Real layer 0 MoE weights.
L = 0
gu_blocks = load_tensor(f"model.layers.{L}.mlp.experts.gate_up_proj_blocks").contiguous()
gu_scales = load_tensor(f"model.layers.{L}.mlp.experts.gate_up_proj_scales").contiguous()
gu_bias   = load_tensor(f"model.layers.{L}.mlp.experts.gate_up_proj_bias").float().contiguous()
d_blocks  = load_tensor(f"model.layers.{L}.mlp.experts.down_proj_blocks").contiguous()
d_scales  = load_tensor(f"model.layers.{L}.mlp.experts.down_proj_scales").contiguous()
d_bias    = load_tensor(f"model.layers.{L}.mlp.experts.down_proj_bias").float().contiguous()

E   = gu_blocks.size(0)
Dff = gu_blocks.size(1) // 2
D   = 2880
print(f"E={E}  D={D}  Dff={Dff}")

# Fake input.
torch.manual_seed(0xA55E7)
T, K = 4, 4
x = torch.randn(T, D) * 0.05
logits = torch.randn(T, E)
top_vals, top_idx = torch.topk(logits, K, dim=-1)
routing = torch.softmax(top_vals, dim=-1)

# --- Transient (upload each call) ---
t0 = time.time()
y_transient = ext.moe_mlp_gpt_oss(
    x, top_idx, routing,
    gu_blocks, gu_scales, gu_bias,
    d_blocks,  d_scales,  d_bias,
)
t_trans = time.time() - t0
print(f"Transient call: {t_trans*1000:.1f} ms")

# --- Upload to resident VRAM ---
print("\nUploading MoE weights to resident VRAM...")
t0 = time.time()
h_gu_b = ext.upload_resident(gu_blocks)
h_gu_s = ext.upload_resident(gu_scales)
h_gu_bi = ext.upload_resident(gu_bias)
h_d_b = ext.upload_resident(d_blocks)
h_d_s = ext.upload_resident(d_scales)
h_d_bi = ext.upload_resident(d_bias)
upload_time = time.time() - t0
resident_bytes = ext.resident_bytes_total()
print(f"Upload: {upload_time:.2f}s   resident: {resident_bytes/1024**2:.1f} MB")

# --- Resident call ---
t0 = time.time()
y_resident = ext.moe_mlp_gpt_oss_resident(
    x, top_idx, routing,
    h_gu_b, h_gu_s, h_gu_bi,
    h_d_b,  h_d_s,  h_d_bi,
    E, D, Dff,
)
t_res = time.time() - t0
print(f"Resident call:  {t_res*1000:.1f} ms  ({t_trans/t_res:.2f}x speedup)")

# --- Time a few more calls to average out ---
for i in range(3):
    t0 = time.time()
    _ = ext.moe_mlp_gpt_oss(x, top_idx, routing,
        gu_blocks, gu_scales, gu_bias, d_blocks, d_scales, d_bias)
    dt = time.time() - t0
    t0 = time.time()
    _ = ext.moe_mlp_gpt_oss_resident(x, top_idx, routing,
        h_gu_b, h_gu_s, h_gu_bi, h_d_b, h_d_s, h_d_bi, E, D, Dff)
    dt_r = time.time() - t0
    print(f"  iter {i+1}:  transient={dt*1000:.1f}ms  resident={dt_r*1000:.1f}ms  "
          f"({dt/dt_r:.2f}x)")

# --- Correctness ---
diff = (y_resident - y_transient).abs().max().item()
print(f"\nmax|D| (resident vs transient): {diff:.3e}")
ok = torch.allclose(y_resident, y_transient, rtol=1e-5, atol=1e-5)
print(f"{'PASS' if ok else 'FAIL'}")

ext.free_resident(h_gu_b); ext.free_resident(h_gu_s); ext.free_resident(h_gu_bi)
ext.free_resident(h_d_b);  ext.free_resident(h_d_s);  ext.free_resident(h_d_bi)
print(f"After free: resident={ext.resident_bytes_total()/1024**2:.1f} MB")
