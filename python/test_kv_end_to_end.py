"""test_kv_end_to_end.py — verify KV-cache generation produces IDENTICAL
tokens to no-cache generation on real gpt-oss-20b weights, and measure
the speedup.

If tokens differ between the two paths, we have a bug in the KV plumbing
(most likely: RoPE positions, past_len wiring, or K/V write index).
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
from gpt_oss.generate import greedy_generate, greedy_generate_kv

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")
PROMPT = "The capital of France is"
MAX_NEW = 2

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[test] Loading extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

print(f"\n[test] Loading model (all 24 layers)...")
t0 = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR)
print(f"[test] Model loaded in {time.time()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# --- Path A: no KV cache (slow, O(N^2)) ---
print(f"\n[test] === Path A (no KV cache, recompute each step) ===")
t_a = time.time()
text_a, ids_a = greedy_generate(model, tok, PROMPT, max_new_tokens=MAX_NEW,
                                  print_stream=True)
elapsed_a = time.time() - t_a
print(f"[test] Path A total: {elapsed_a:.1f}s   tokens={ids_a}")

# --- Path B: WITH KV cache (fast, O(N)) ---
print(f"\n[test] === Path B (KV cache) ===")
t_b = time.time()
text_b, ids_b, stats_b = greedy_generate_kv(
    model, tok, PROMPT, max_new_tokens=MAX_NEW, print_stream=True)
elapsed_b = time.time() - t_b
print(f"[test] Path B total: {elapsed_b:.1f}s   tokens={ids_b}")
print(f"[test] Path B prefill: {stats_b['prefill_time']:.1f}s   "
      f"decode: {[f'{t:.1f}' for t in stats_b['decode_times']]}")

# --- Compare ---
print(f"\n[test] === Correctness check ===")
match = ids_a == ids_b
print(f"  no-cache tokens: {ids_a}")
print(f"  kv-cache tokens: {ids_b}")
print(f"  IDENTICAL: {match}")

if elapsed_a > 0 and elapsed_b > 0:
    speedup = elapsed_a / elapsed_b
    print(f"\n[test] === Perf ===")
    print(f"  no-cache: {elapsed_a:.1f}s ({elapsed_a/MAX_NEW:.2f}s/tok)")
    print(f"  kv-cache: {elapsed_b:.1f}s (prefill {stats_b['prefill_time']:.1f}s "
          f"+ decode ~{sum(stats_b['decode_times'])/max(1,len(stats_b['decode_times'])):.2f}s/tok)")
    print(f"  speedup:  {speedup:.2f}x")

print(f"\n[test] {'PASS' if match else 'FAIL'}")
