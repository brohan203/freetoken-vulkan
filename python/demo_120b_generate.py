"""Short gpt-oss-120b greedy generation with streamed experts and KV cache."""
from __future__ import annotations

import os
import pathlib
import sys
import time

os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ.get("PATH", "")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from gpt_oss import GptOssModel
from gpt_oss.generate import greedy_generate_kv

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-120b")
PROMPT = os.environ.get("FREETOKEN_PROMPT", "The capital of France is")
MAX_NEW = int(os.environ.get("FREETOKEN_MAX_NEW", "6"))
CACHE_SLOTS = int(os.environ.get("FREETOKEN_CACHE_SLOTS", "24"))
PIN_LM_HEAD = os.environ.get("FREETOKEN_PIN_LM_HEAD", "1") == "1"
ENABLE_GPU_CACHE = os.environ.get("FREETOKEN_GPU_CACHE", "1") == "1"
PREFILL_CHUNK_SIZE = int(os.environ.get("FREETOKEN_PREFILL_CHUNK", "0")) or None

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

print("[120b] Loading model with streamed experts...", flush=True)
t0 = time.time()
model = GptOssModel.from_pretrained(ext, MODEL_DIR, stream_experts=True)
print(f"[120b] Loaded in {time.time()-t0:.2f}s", flush=True)
if ENABLE_GPU_CACHE:
    model.enable_streamed_vram_cache(slots_per_layer=CACHE_SLOTS)
    print(f"[120b] Enabled {CACHE_SLOTS}-slot per-layer VRAM expert cache", flush=True)
else:
    print("[120b] GPU expert cache disabled", flush=True)
if PIN_LM_HEAD:
    model.pin_lm_head_to_vram()
    print(f"[120b] Resident VRAM bytes: {ext.resident_bytes_total()/1024**3:.3f} GiB", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print(f"[120b] Prompt: {PROMPT!r}")
t0 = time.time()
text, new_ids, stats = greedy_generate_kv(
    model, tokenizer, PROMPT,
    max_new_tokens=MAX_NEW,
    max_seqlen=128,
    print_stream=True,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
)
elapsed = time.time() - t0
print(f"[120b] Text: {text!r}")
print(f"[120b] Tokens: {new_ids}")
print(f"[120b] Prefill: {stats['prefill_time']:.3f}s")
print(f"[120b] Decode times: {[round(x, 3) for x in stats['decode_times']]}")
if stats['decode_times']:
    print(f"[120b] Decode average: {sum(stats['decode_times'])/len(stats['decode_times']):.3f}s/token")
store = model.weights.expert_store
print(f"[120b] Expert materialization: {store.materialize_seconds:.3f}s, "
      f"{store.materialized_bytes/1024**3:.3f} GiB")
# Consecutive-step expert reuse across the generated sequence.
reused = selected = 0
for history in store.selection_history.values():
    for previous, current in zip(history, history[1:]):
        reused += len(set(previous).intersection(current))
        selected += len(current)
print(f"[120b] Consecutive expert reuse: {reused}/{selected} "
      f"({100.0*reused/max(1, selected):.1f}%)")
print(f"[120b] CPU expert cache: hits={store.cache_hits} misses={store.cache_misses} "
      f"rate={100.0*store.cache_hits/max(1, store.cache_hits+store.cache_misses):.1f}%")
gpu_cache = model.streamed_resident
if gpu_cache is not None:
    print(f"[120b] GPU expert cache: hits={gpu_cache.hits} misses={gpu_cache.misses} "
          f"rate={100.0*gpu_cache.hits/max(1, gpu_cache.hits+gpu_cache.misses):.1f}% "
          f"uploads={gpu_cache.uploaded_bytes/1024**3:.3f} GiB "
          f"upload_s={gpu_cache.upload_seconds:.3f}")
print(f"[120b] Total: {elapsed:.3f}s")
print("[120b] GENERATION_OK")
