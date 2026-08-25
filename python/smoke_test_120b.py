"""First full 36-layer gpt-oss-120b inference with streamed experts."""
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
from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from gpt_oss import GptOssModel

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-120b")
PROMPT = "Hello"

sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[120b] Loading Vulkan extension...", flush=True)
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(HERE / "ext_module.cpp")],
    extra_include_paths=[str(sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

print("[120b] Loading 36 layers with file-backed experts...", flush=True)
t0 = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR, stream_experts=True)
load_s = time.time() - t0
print(f"[120b] Model metadata/non-expert weights loaded in {load_s:.2f}s", flush=True)
assert model.cfg.num_hidden_layers == 36
assert model.cfg.num_experts == 128
assert model.weights.expert_store is not None
assert all(layer.gate_up_blocks is None for layer in model.weights.layers)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
ids = tokenizer.encode(PROMPT, return_tensors="pt").long()
print(f"[120b] Prompt IDs: {ids.tolist()}", flush=True)

print("[120b] Running full 36-layer forward...", flush=True)
t0 = time.time()
with torch.no_grad():
    logits = model.forward(ids, only_last_logits=True)
forward_s = time.time() - t0
print(f"[120b] Forward completed in {forward_s:.2f}s", flush=True)
print(f"[120b] Logits shape: {list(logits.shape)}", flush=True)
print(f"[120b] Logits finite: {bool(torch.isfinite(logits).all())}", flush=True)

top_values, top_ids = torch.topk(logits[0, -1], 15)
print("[120b] Top-15 next-token predictions:")
for value, token_id in zip(top_values.tolist(), top_ids.tolist()):
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    print(f"  id={token_id:6d} logit={value:9.3f} text={text!r}")
print("[120b] FULL_FORWARD_OK")
