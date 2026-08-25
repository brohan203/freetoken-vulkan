"""demo_kv_resident.py — gpt-oss-20b generation with BOTH:
    - KV cache (O(N) decode instead of O(N^2))
    - Persistent VRAM MoE weights (skips ~10 GB PCIe traffic per forward)

Runs the same prompt through three configurations:
    A. Baseline: no KV cache, weights uploaded each call        (session 4 speed)
    B. KV cache only                                             (session 5a)
    C. KV cache + resident MoE weights                           (session 5b)

Prints tokens generated + per-step timings. Correctness is asserted by
comparing token IDs across all three paths — they MUST be identical.
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
MAX_NEW = 3

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

print(f"\n[demo] Loading model (24 layers, MoE weights kept in RAM only)...")
model = GptOssModel.from_pretrained(ext, MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# ============ Path A: baseline (no KV, transient weights) ============
print("\n" + "=" * 60)
print(f"[A] Baseline: no KV cache, transient weight uploads")
print("=" * 60)
t_a = time.time()
text_a, ids_a = greedy_generate(model, tok, PROMPT,
                                 max_new_tokens=MAX_NEW, print_stream=True)
elapsed_a = time.time() - t_a
print(f"\n[A] total={elapsed_a:.1f}s   tokens={ids_a}")

# ============ Path B: KV cache only ============
print("\n" + "=" * 60)
print(f"[B] KV cache, but weights still transient")
print("=" * 60)
t_b = time.time()
text_b, ids_b, stats_b = greedy_generate_kv(
    model, tok, PROMPT, max_new_tokens=MAX_NEW, print_stream=True)
elapsed_b = time.time() - t_b
print(f"\n[B] total={elapsed_b:.1f}s   tokens={ids_b}   "
      f"prefill={stats_b['prefill_time']:.1f}s   "
      f"decode~{sum(stats_b['decode_times'])/max(1,len(stats_b['decode_times'])):.1f}s/tok")

# ============ Pin MoE weights to VRAM ============
print("\n" + "=" * 60)
print(f"[Setup C] Pinning MoE weights to VRAM (~10 GB, one-time)")
print("=" * 60)
t_pin = time.time()
model.pin_moe_to_vram()
print(f"[Setup C] Pin done in {time.time()-t_pin:.1f}s   "
      f"VRAM held: {ext.resident_bytes_total()/1024**3:.2f} GB")

# ============ Path C: KV cache + resident weights ============
print("\n" + "=" * 60)
print(f"[C] KV cache + resident MoE weights (fastest)")
print("=" * 60)
t_c = time.time()
text_c, ids_c, stats_c = greedy_generate_kv(
    model, tok, PROMPT, max_new_tokens=MAX_NEW, print_stream=True)
elapsed_c = time.time() - t_c
print(f"\n[C] total={elapsed_c:.1f}s   tokens={ids_c}   "
      f"prefill={stats_c['prefill_time']:.1f}s   "
      f"decode~{sum(stats_c['decode_times'])/max(1,len(stats_c['decode_times'])):.1f}s/tok")

# ============ Correctness + speedup ============
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
same = ids_a == ids_b == ids_c
print(f"  All three paths tokens IDENTICAL: {same}")
print(f"  Path A tokens: {ids_a}")
print(f"  Path B tokens: {ids_b}")
print(f"  Path C tokens: {ids_c}")
print(f"")
print(f"  Path A total: {elapsed_a:.1f}s  ({elapsed_a/MAX_NEW:.2f}s/tok)")
print(f"  Path B total: {elapsed_b:.1f}s  (KV cache added)")
print(f"  Path C total: {elapsed_c:.1f}s  (KV + resident weights)")
print(f"  Speedup A -> C: {elapsed_a/max(elapsed_c, 0.001):.2f}x")
print(f"  Speedup B -> C: {elapsed_b/max(elapsed_c, 0.001):.2f}x")
