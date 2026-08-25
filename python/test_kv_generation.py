"""test_kv_generation.py — verify KV cache produces identical tokens to
no-cache generation, and measure the speedup.

Correctness invariant: greedy_generate and greedy_generate_kv MUST produce
the same tokens for the same prompt (up to tiny FP32 accumulation drift
that can flip a token when top-2 logits are close, but for most cases
this should be identical).
"""
from __future__ import annotations
import os, pathlib, sys, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))

from gpt_oss import GptOssModel
from gpt_oss.generate import greedy_generate, greedy_generate_kv, top_k_predictions

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
PROMPT = "The capital of France is"
MAX_NEW = 3

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

print(f"\nLoading model (all 24 layers)...")
t0 = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR)
print(f"Model ready in {time.time()-t0:.1f}s")

print(f"\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

print("\n" + "="*60)
print("PASS 1: no-KV-cache greedy generation (slow path)")
print("="*60)
t_slow = time.time()
text_slow, ids_slow = greedy_generate(model, tok, PROMPT,
                                       max_new_tokens=MAX_NEW,
                                       print_stream=True)
slow_elapsed = time.time() - t_slow
print(f"slow: {slow_elapsed:.1f}s ({slow_elapsed/MAX_NEW:.1f}s/tok)")
print(f"slow tokens: {ids_slow}")

print("\n" + "="*60)
print("PASS 2: KV-cache greedy generation (fast path)")
print("="*60)
t_fast = time.time()
text_fast, ids_fast, stats = greedy_generate_kv(model, tok, PROMPT,
                                                  max_new_tokens=MAX_NEW,
                                                  max_seqlen=64,
                                                  print_stream=True)
fast_elapsed = time.time() - t_fast
print(f"fast: {fast_elapsed:.1f}s  (prefill {stats['prefill_time']:.1f}s + "
      f"{sum(stats['decode_times']):.1f}s decode)")
if stats["decode_times"]:
    avg_decode = sum(stats["decode_times"]) / len(stats["decode_times"])
    print(f"       avg decode step: {avg_decode:.2f}s")
print(f"fast tokens: {ids_fast}")

print("\n" + "="*60)
print("Correctness comparison")
print("="*60)
if ids_slow == ids_fast:
    print(f"IDENTICAL tokens: {ids_slow}")
else:
    print(f"DIFFER:")
    print(f"  slow: {ids_slow}")
    print(f"  fast: {ids_fast}")

if slow_elapsed > 0:
    speedup = slow_elapsed / max(fast_elapsed, 0.001)
    print(f"\nSpeedup: {speedup:.2f}x  ({slow_elapsed:.1f}s vs {fast_elapsed:.1f}s)")
