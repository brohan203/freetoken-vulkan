"""test_resident_moe.py — verify moe_mlp_gpt_oss_resident produces IDENTICAL
output to moe_mlp_gpt_oss on real gpt-oss-20b layer-0 weights, and measure
the speedup from skipping the ~424 MB per-call weight upload.
"""
from __future__ import annotations
import os, pathlib, sys, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

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
import json
with open(MODEL / "model.safetensors.index.json") as f:
    idx = json.load(f)["weight_map"]
def get(n):
    with safe_open(MODEL / idx[n], framework="pt") as f:
        return f.get_tensor(n)

# Load layer 0 MoE weights.
gu_blocks = get("model.layers.0.mlp.experts.gate_up_proj_blocks").contiguous()
gu_scales = get("model.layers.0.mlp.experts.gate_up_proj_scales").contiguous()
gu_bias   = get("model.layers.0.mlp.experts.gate_up_proj_bias").float().contiguous()
d_blocks  = get("model.layers.0.mlp.experts.down_proj_blocks").contiguous()
d_scales  = get("model.layers.0.mlp.experts.down_proj_scales").contiguous()
d_bias    = get("model.layers.0.mlp.experts.down_proj_bias").float().contiguous()

E, TWO_DFF = gu_blocks.shape[0], gu_blocks.shape[1]
Dff = TWO_DFF // 2
D = 2880
K = 4

# Random input, indices, weights.
torch.manual_seed(0xF00D)
T = 8
x = (torch.randn(T, D) * 0.05).float().contiguous()
indices = torch.randint(0, E, (T, K)).contiguous()
weights = torch.softmax(torch.randn(T, K), dim=-1).contiguous()

# ---- Pass 1: non-resident. Upload weights every call. ----
print("\n== Pass 1: non-resident (upload every call) ==")
for i in range(3):
    t0 = time.time()
    y_nonres = ext.moe_mlp_gpt_oss(
        x, indices, weights,
        gu_blocks, gu_scales, gu_bias,
        d_blocks,  d_scales,  d_bias)
    print(f"  call {i}: {(time.time()-t0)*1000:.1f} ms")

# ---- Pass 2: upload weights ONCE, then reuse. ----
print("\n== Pass 2: resident (upload once, reuse) ==")
t0 = time.time()
h_gub = ext.upload_resident(gu_blocks)
h_gus = ext.upload_resident(gu_scales)
h_gubi = ext.upload_resident(gu_bias)
h_db  = ext.upload_resident(d_blocks)
h_ds  = ext.upload_resident(d_scales)
h_dbi = ext.upload_resident(d_bias)
upload_time = time.time() - t0
resident_mb = ext.resident_bytes_total() / 1024**2
print(f"  upload of 6 weight tensors: {upload_time*1000:.1f} ms  "
      f"(resident total: {resident_mb:.1f} MB)")

for i in range(3):
    t0 = time.time()
    y_res = ext.moe_mlp_gpt_oss_resident(
        x, indices, weights,
        h_gub, h_gus, h_gubi,
        h_db,  h_ds,  h_dbi,
        E, D, Dff)
    print(f"  call {i}: {(time.time()-t0)*1000:.1f} ms")

# ---- Compare ----
diff = (y_res - y_nonres).abs().max().item()
print(f"\nmax|D| = {diff:.3e}")
print(f"{'PASS' if diff < 1e-4 else 'FAIL'}")

# Free
for h in [h_gub, h_gus, h_gubi, h_db, h_ds, h_dbi]:
    ext.free_resident(h)
print(f"After free: resident bytes = {ext.resident_bytes_total()}")
