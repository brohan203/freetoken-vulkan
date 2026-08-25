"""demo_long_gen.py — long-form generation test with all optimizations
enabled (KV cache + resident MoE + resident LM head).

Runs several prompts through ~64-token greedy decode each. Purpose:
    (1) prove stability over long sequences (no OOM, no drift)
    (2) collect real sample output — what does gpt-oss-20b actually produce?
    (3) measure decode time as sequence length grows
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
from gpt_oss.generate import greedy_generate_kv

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In a shocking discovery, scientists have found that",
    "The three laws of robotics are:",
    "SELECT name, age FROM users WHERE",
]
MAX_NEW = 64
MAX_SEQLEN = 128

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[demo] Compiling Vulkan extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

print(f"\n[demo] Loading model...")
t0 = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR)
print(f"[demo] Model loaded in {time.time()-t0:.1f}s")

print(f"\n[demo] Pinning MoE + LM head to VRAM...")
t0 = time.time()
model.pin_moe_to_vram()
model.pin_lm_head_to_vram()
print(f"[demo] Pinned in {time.time()-t0:.1f}s   "
      f"VRAM: {ext.resident_bytes_total()/1024**3:.2f} GB")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# ============ Run each prompt ============
print(f"\n{'='*72}")
print(f"  Generating {MAX_NEW} tokens per prompt, {len(PROMPTS)} prompts")
print(f"{'='*72}")

results = []
for pi, prompt in enumerate(PROMPTS):
    print(f"\n--- Prompt {pi+1}/{len(PROMPTS)}: {prompt!r} ---")
    t0 = time.time()
    text, new_ids, stats = greedy_generate_kv(
        model, tok, prompt,
        max_new_tokens=MAX_NEW,
        max_seqlen=MAX_SEQLEN,
        print_stream=False,   # print full output at end instead
    )
    elapsed = time.time() - t0

    prefill = stats['prefill_time']
    decode_times = stats['decode_times']
    decode_avg = sum(decode_times) / max(1, len(decode_times))

    print(f"\n{text}")
    print(f"\n  [stats] prefill={prefill:.1f}s   "
          f"decode_avg={decode_avg:.2f}s/tok   "
          f"first={decode_times[0] if decode_times else 0:.2f}s   "
          f"last={decode_times[-1] if decode_times else 0:.2f}s   "
          f"total={elapsed:.1f}s")

    results.append({
        "prompt": prompt,
        "text": text,
        "n_prompt": stats['num_prompt_tokens'],
        "n_new": stats['num_new_tokens'],
        "prefill_s": prefill,
        "decode_avg_s": decode_avg,
        "decode_first_s": decode_times[0] if decode_times else 0,
        "decode_last_s": decode_times[-1] if decode_times else 0,
        "total_s": elapsed,
    })

# ============ Summary ============
print(f"\n{'='*72}")
print(f"  SUMMARY")
print(f"{'='*72}")
print(f"  {'Prompt':<48} | prefill | avg dec | total")
for r in results:
    tag = r['prompt'][:44] + ('...' if len(r['prompt']) > 44 else '')
    print(f"  {tag:<48} | {r['prefill_s']:6.1f}s | {r['decode_avg_s']:6.2f}s | {r['total_s']:5.1f}s")

# How does per-token time change as sequence grows?
if results and results[-1].get("decode_first_s"):
    r = results[-1]
    print(f"\n  Growth check (last prompt):")
    print(f"    First decode step: {r['decode_first_s']:.2f}s")
    print(f"    Last decode step:  {r['decode_last_s']:.2f}s")
    print(f"    Change:           {r['decode_last_s'] - r['decode_first_s']:+.2f}s "
          f"({(r['decode_last_s']/r['decode_first_s'] - 1)*100:+.1f}%)")
