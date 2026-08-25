"""bench_full_pipeline.py — end-to-end benchmark of gpt-oss-20b generation
on the 6800 XT with all optimizations enabled:

    (1) KV cache (avoids O(N^2) recompute per step)
    (2) Persistent VRAM MoE weights (avoids ~10 GB PCIe traffic per forward)

Also runs the unoptimized reference for comparison. Reports:
    - Model load time
    - Persistent-VRAM upload time
    - Prefill time
    - Per-decode-token time
    - Total generation time
    - Speedup vs unoptimized baseline
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
PROMPT = "The capital of France is"
MAX_NEW = 6

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("[bench] Loading extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

t0 = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR)
print(f"[bench] Model loaded: {time.time()-t0:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)


# ============ Run 1: KV cache only (no resident) ============
print("\n[bench] === Run 1: KV cache, NO resident MoE ===")
t_a = time.time()
text_a, ids_a, stats_a = greedy_generate_kv(model, tok, PROMPT,
                                              max_new_tokens=MAX_NEW,
                                              max_seqlen=64,
                                              print_stream=True)
elapsed_a = time.time() - t_a
print(f"[bench] Run 1 total: {elapsed_a:.1f}s  "
      f"(prefill {stats_a['prefill_time']:.1f}s + "
      f"decode {sum(stats_a['decode_times']):.1f}s)")
print(f"[bench] tokens: {ids_a}")


# ============ Run 2: KV cache + resident MoE ============
print("\n[bench] === Run 2: KV cache + RESIDENT MoE ===")
print(f"[bench] Pinning MoE weights to VRAM (one-time)...")
t_upload = time.time()
model.pin_moe_to_vram()
upload_time = time.time() - t_upload
print(f"[bench] Upload complete: {upload_time:.1f}s   VRAM used: "
      f"{ext.resident_bytes_total()/1024**3:.2f} GB")

t_b = time.time()
text_b, ids_b, stats_b = greedy_generate_kv(model, tok, PROMPT,
                                              max_new_tokens=MAX_NEW,
                                              max_seqlen=64,
                                              print_stream=True)
elapsed_b = time.time() - t_b
print(f"[bench] Run 2 total: {elapsed_b:.1f}s  "
      f"(prefill {stats_b['prefill_time']:.1f}s + "
      f"decode {sum(stats_b['decode_times']):.1f}s)")
print(f"[bench] tokens: {ids_b}")


# ============ Compare ============
print("\n[bench] === Correctness ===")
print(f"  Run 1 tokens: {ids_a}")
print(f"  Run 2 tokens: {ids_b}")
print(f"  IDENTICAL: {ids_a == ids_b}")

print(f"\n[bench] === Performance summary ===")
print(f"  Prefill:")
print(f"    KV only:            {stats_a['prefill_time']:.1f}s")
print(f"    KV + resident:      {stats_b['prefill_time']:.1f}s "
      f"(speedup {stats_a['prefill_time']/max(0.001,stats_b['prefill_time']):.2f}x)")
if stats_a['decode_times'] and stats_b['decode_times']:
    avg_a = sum(stats_a['decode_times']) / len(stats_a['decode_times'])
    avg_b = sum(stats_b['decode_times']) / len(stats_b['decode_times'])
    print(f"  Decode (avg per token):")
    print(f"    KV only:            {avg_a:.2f}s")
    print(f"    KV + resident:      {avg_b:.2f}s "
          f"(speedup {avg_a/max(0.001,avg_b):.2f}x)")
print(f"  Total (excl. one-time upload):")
print(f"    KV only:            {elapsed_a:.1f}s")
print(f"    KV + resident:      {elapsed_b:.1f}s "
      f"(speedup {elapsed_a/max(0.001,elapsed_b):.2f}x)")
