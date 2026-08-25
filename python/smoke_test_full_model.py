"""smoke_test_full_model.py — load ALL 24 layers, run ONE forward on a 4-token
prompt, print top-K predictions. If this works, the full model is wired
correctly and we can move on to KV cache + full generation.
"""
from __future__ import annotations
import os, pathlib, sys, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))
from gpt_oss import GptOssModel

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
PROMPT = "Hello"

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[smoke] Loading Vulkan extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

print(f"\n[smoke] Loading full model (24 layers)...")
t_load = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR)
print(f"[smoke] Model loaded in {time.time()-t_load:.1f}s")

def _sizeof(t):
    return 0 if t is None else t.numel() * t.element_size()

tot = _sizeof(model.weights.embed_tokens) + _sizeof(model.weights.lm_head)
tot += _sizeof(model.weights.final_norm)
for lw in model.weights.layers:
    for name in vars(lw):
        tot += _sizeof(getattr(lw, name))
print(f"[smoke] Total in-memory weight bytes: {tot / 1024**3:.2f} GB")

print(f"\n[smoke] Tokenizing prompt: {PROMPT!r}")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
input_ids = torch.tensor(tok.encode(PROMPT), dtype=torch.long).unsqueeze(0)
print(f"[smoke] Prompt tokens: {input_ids.tolist()}  shape={list(input_ids.shape)}")

print(f"\n[smoke] Running ONE forward pass through 24 layers + lm_head...")
print(f"        Expected: 30-90s depending on IO overhead per Vulkan call.")
t0 = time.time()
with torch.no_grad():
    logits = model.forward(input_ids)
elapsed = time.time() - t0
print(f"[smoke] Forward pass: {elapsed:.1f}s   logits shape: {list(logits.shape)}")

# Top-K predictions for the last position
print(f"\n[smoke] Top-15 next-token predictions:")
top_vals, top_ids = torch.topk(logits[0, -1], k=15)
for v, i in zip(top_vals.tolist(), top_ids.tolist()):
    piece = tok.decode([i], skip_special_tokens=False)
    print(f"  id={i:6d}  logit={v:8.3f}  {piece!r}")
print("\n[smoke] DONE")
