"""Validate 120b compact expert streaming against eager layer weights.

This test loads one real gpt-oss-120b layer twice:
  * eager: all 128 MXFP4 experts are present in CPU memory;
  * streaming: expert tensors remain file-backed and only the router-selected
    experts are materialized into a compact local table.

The complete layer output must match. Attention uses identical weights and
inputs; the only differing code path is expert selection/materialization.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = (
    os.path.join(os.environ["VULKAN_SDK"], "Bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from transformers import AutoConfig

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from gpt_oss.config import GptOssConfig
from gpt_oss.layer import gpt_oss_layer_forward
from gpt_oss.loader import ExpertStore, Safetensors, load_layer
from gpt_oss.rope import compute_cos_sin_for_positions

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-120b")

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[120b] Loading Vulkan extension...", flush=True)
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(HERE / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

cfg = GptOssConfig.from_json(MODEL_DIR / "config.json")
hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
sf = Safetensors(MODEL_DIR)

print("[120b] Loading eager layer 0 (all 128 experts)...", flush=True)
t0 = time.time()
eager = load_layer(sf, 0, load_experts=True)
print(f"[120b] Eager layer loaded in {time.time() - t0:.2f}s", flush=True)

print("[120b] Loading streaming layer 0 metadata...", flush=True)
streamed = load_layer(sf, 0, load_experts=False)
store = ExpertStore(sf)

# One token is the decode-critical path and selects at most four experts.
torch.manual_seed(120)
x = torch.randn(1, 1, cfg.hidden_size, dtype=torch.float32) * 0.01
positions = torch.tensor([0], dtype=torch.long)
cos, sin = compute_cos_sin_for_positions(hf_cfg, positions)

print("[120b] Running eager layer...", flush=True)
t0 = time.time()
y_eager = gpt_oss_layer_forward(
    ext, x, layer_idx=0, weights=eager, cfg=cfg, cos=cos, sin=sin
)
eager_s = time.time() - t0

print("[120b] Running compact streamed layer...", flush=True)
t0 = time.time()
y_stream = gpt_oss_layer_forward(
    ext, x, layer_idx=0, weights=streamed, cfg=cfg, cos=cos, sin=sin,
    expert_store=store,
)
stream_s = time.time() - t0

diff = (y_eager - y_stream).abs()
max_abs = diff.max().item()
mean_abs = diff.mean().item()
print(f"[120b] eager_s={eager_s:.3f} stream_s={stream_s:.3f}")
print(f"[120b] max_abs={max_abs:.9e} mean_abs={mean_abs:.9e}")
print(f"[120b] output_mean={y_stream.mean().item():.6f} output_std={y_stream.std().item():.6f}")
assert torch.equal(y_eager, y_stream), (
    f"streamed layer differs from eager layer: max_abs={max_abs}"
)
print("[120b] STREAMING_LAYER_BIT_EXACT")
