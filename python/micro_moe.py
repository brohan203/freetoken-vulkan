"""micro_moe.py — micro-benchmark a single resident MoE MLP call to see where
the 35 ms per-call time actually goes."""
from __future__ import annotations
import os, pathlib, sys, time, json

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

E   = gu_blocks.size(0)
Dff = gu_blocks.size(1) // 2
D   = 2880
K = 4

# Upload weights to resident VRAM.
h_gub = ext.upload_resident(gu_blocks)
h_gus = ext.upload_resident(gu_scales)
h_gubi = ext.upload_resident(gu_bias)
h_db = ext.upload_resident(d_blocks)
h_ds = ext.upload_resident(d_scales)
h_dbi = ext.upload_resident(d_bias)

# --- Warmup ---
for T in [1, 4, 16, 64]:
    torch.manual_seed(42)
    x = torch.randn(T, D) * 0.05
    indices = torch.randint(0, E, (T, K))
    weights = torch.softmax(torch.randn(T, K), dim=-1)
    for _ in range(3):
        _ = ext.moe_mlp_gpt_oss_resident(x, indices, weights,
            h_gub, h_gus, h_gubi, h_db, h_ds, h_dbi, E, D, Dff)

# --- Measure ---
print("Single-call timings (Python transition per call):")
print("T\t| avg_ms/call\t| ms/token")
for T in [1, 2, 4, 8, 16, 32, 64, 128]:
    torch.manual_seed(42)
    x = torch.randn(T, D) * 0.05
    indices = torch.randint(0, E, (T, K))
    weights = torch.softmax(torch.randn(T, K), dim=-1)
    N = 20
    t0 = time.time()
    for _ in range(N):
        _ = ext.moe_mlp_gpt_oss_resident(x, indices, weights,
            h_gub, h_gus, h_gubi, h_db, h_ds, h_dbi, E, D, Dff)
    dt = (time.time() - t0) / N * 1000  # ms per call
    print(f"{T:3d}\t| {dt:10.2f}\t| {dt/T:8.2f}")

# --- Bench-mode: N kernel launches in ONE Python transition ---
# This isolates Python↔C++ overhead vs Vulkan submit/wait overhead.
print("\nN-batched-in-one-call:")
print("T=1, N=24 calls back-to-back inside one C++ transition")
torch.manual_seed(42)
x = torch.randn(1, D) * 0.05
indices = torch.randint(0, E, (1, K))
weights = torch.softmax(torch.randn(1, K), dim=-1)
# Warmup.
for _ in range(3):
    ext.moe_mlp_gpt_oss_resident_bench(x, indices, weights,
        h_gub, h_gus, h_gubi, h_db, h_ds, h_dbi, E, D, Dff, 24)
# Measure.
N_iter = 10
t0 = time.time()
for _ in range(N_iter):
    ext.moe_mlp_gpt_oss_resident_bench(x, indices, weights,
        h_gub, h_gus, h_gubi, h_db, h_ds, h_dbi, E, D, Dff, 24)
total_ms = (time.time() - t0) / N_iter * 1000
per_kernel_ms = total_ms / 24
print(f"  Total per Python call: {total_ms:.2f} ms")
print(f"  Per kernel launch:      {per_kernel_ms:.2f} ms")

for h in [h_gub, h_gus, h_gubi, h_db, h_ds, h_dbi]:
    ext.free_resident(h)