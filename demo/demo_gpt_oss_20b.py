"""demo_gpt_oss_20b.py — end-to-end demo.

Loads all 24 layers of gpt-oss-20b from safetensors, tokenizes a prompt
with the o200k_harmony tokenizer, and generates a few tokens greedy.

Expected memory: ~16 GB RAM (MoE experts stay MXFP4 packed; attention/
embed/lm_head/norms upgraded to FP32).

Expected time: model load ~30-60s, then per-token decode is SLOW (~30-60s
each) because the LM head is a 578M FMA CPU matmul and we recompute the
full sequence every step. This is a correctness demo, not a perf demo.

If the generated text is coherent English, the port is working end-to-end.
"""
from __future__ import annotations
import os, pathlib, sys, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# Give torch more CPU threads for the giant LM head matmul.
os.environ["MKL_NUM_THREADS"] = "12"

import torch
torch.set_num_threads(12)

from torch.utils.cpp_extension import load
from transformers import AutoTokenizer

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))

from gpt_oss import GptOssModel
from gpt_oss.generate import greedy_generate

MODEL_DIR = r"C:\Users\rohanborkar\Downloads\gpt-oss-20b"
PROMPT    = "The Vulkan API"
MAX_NEW   = 10

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[demo] Loading Vulkan extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

# Load all 24 layers of gpt-oss-20b.
model = GptOssModel.from_pretrained(ext, MODEL_DIR)   # all layers by default

print(f"[demo] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"\n[demo] Prompt: {PROMPT!r}")
input_ids = tok.encode(PROMPT, return_tensors="pt").long()
print(f"[demo] Prompt tokens: {input_ids.tolist()}  ({input_ids.shape[1]} tokens)")

print(f"\n[demo] Running full-sequence forward pass to warm up...")
t0 = time.time()
with torch.no_grad():
    logits = model.forward(input_ids)
print(f"[demo] First forward pass: {time.time()-t0:.1f}s  (logits shape: {list(logits.shape)})")

# Top-K next-token predictions for the last position — sanity check.
top_vals, top_ids = torch.topk(logits[0, -1], k=10)
print(f"\n[demo] Top-10 next-token predictions for prompt {PROMPT!r}:")
for v, i in zip(top_vals.tolist(), top_ids.tolist()):
    piece = tok.decode([i], skip_special_tokens=False)
    print(f"  {i:6d}  logit={v:8.3f}  {piece!r}")

print(f"\n[demo] Greedy generation ({MAX_NEW} new tokens):")
print(f"---")
text, new_ids = greedy_generate(model, tok, PROMPT,
                                 max_new_tokens=MAX_NEW, print_stream=True)
print(f"---")
print(f"[demo] new token IDs: {new_ids}")
