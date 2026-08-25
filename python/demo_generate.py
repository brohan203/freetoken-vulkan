"""demo_generate.py — the moment of truth: run gpt-oss-20b end-to-end
on a 6800 XT. This is the sanity-check demo.

Two-phase test:
    Phase A: single forward pass, show top-10 next-token predictions.
             If those are reasonable English continuations, the model is
             functionally correct.
    Phase B: greedy generate a few tokens to see actual output.

Runtime warning: no KV cache, no persistent VRAM weights, no batched
uploads. This is deliberately unoptimized correctness code. Expect 10-60s
per token depending on sequence length.
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
from gpt_oss.generate import top_k_predictions, greedy_generate

MODEL_DIR       = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
PROMPT          = "The capital of France is"
MAX_NEW_TOKENS  = 3

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[demo] Compiling Vulkan extension...")
t0 = time.time()
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)
print(f"[demo] Extension ready in {time.time()-t0:.1f}s")

print(f"\n[demo] Loading model (all 24 layers)...")
t_load = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR)
print(f"[demo] Model loaded in {time.time()-t_load:.1f}s")

# --- Rough weight-footprint estimate ---
def _sizeof(t):
    return 0 if t is None else t.numel() * t.element_size()
total = _sizeof(model.weights.embed_tokens) + _sizeof(model.weights.lm_head)
total += _sizeof(model.weights.final_norm)
for lw in model.weights.layers:
    for name in vars(lw):
        total += _sizeof(getattr(lw, name))
print(f"[demo] In-memory weight bytes: {total / 1024**3:.2f} GB")

print(f"\n[demo] Loading tokenizer (o200k_harmony)...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"[demo] Vocab size: {tok.vocab_size if hasattr(tok, 'vocab_size') else 'n/a'}")

# ============ PHASE A: single forward + top-10 ============
print(f"\n[demo] ---- PHASE A: single forward, top-10 predictions ----")
print(f"[demo] Prompt: {PROMPT!r}")
input_ids = tok.encode(PROMPT, return_tensors="pt").long()
print(f"[demo] Prompt token IDs: {input_ids[0].tolist()}")
print(f"[demo] Prompt decoded piece-by-piece:")
for tid in input_ids[0].tolist():
    print(f"  {tid:6d}  {tok.decode([tid], skip_special_tokens=False)!r}")

vals, ids = top_k_predictions(model, tok, PROMPT, k=10)

print(f"\n[demo] ---- PHASE B: greedy generate {MAX_NEW_TOKENS} tokens ----")
t_gen = time.time()
text, new_ids = greedy_generate(model, tok, PROMPT,
                                 max_new_tokens=MAX_NEW_TOKENS,
                                 print_stream=True)
gen_elapsed = time.time() - t_gen
print(f"\n[demo] Generation total: {gen_elapsed:.1f}s "
      f"({gen_elapsed/max(1,len(new_ids)):.1f}s/tok)")
print(f"[demo] New token IDs: {new_ids}")
print(f"[demo] Full text: {text!r}")
