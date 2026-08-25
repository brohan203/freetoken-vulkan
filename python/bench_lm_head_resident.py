"""bench_lm_head_resident.py — measure impact of moving LM head to GPU.

Compares three configs:
    A. KV cache + resident MoE (baseline from previous session)
    B. KV cache + resident MoE + resident LM head
    (with only_last_logits enabled, which only matters when h_lm_head is set
     but works either way)
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
MAX_NEW = 12

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

model = GptOssModel.from_pretrained(ext, MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

# ============ Config A: MoE resident only ============
model.pin_moe_to_vram()
print(f"\n[bench] After pin_moe: resident = "
      f"{ext.resident_bytes_total()/1024**3:.2f} GB")

print("\n=== Config A: KV + resident MoE only ===")
t_a = time.time()
text_a, ids_a, stats_a = greedy_generate_kv(
    model, tok, PROMPT, max_new_tokens=MAX_NEW,
    max_seqlen=64, print_stream=True)
elapsed_a = time.time() - t_a
print(f"\ntotal={elapsed_a:.1f}s  tokens={ids_a}")

# ============ Config B: MoE + LM head resident ============
model.pin_lm_head_to_vram()
print(f"\n[bench] After pin_lm_head: resident = "
      f"{ext.resident_bytes_total()/1024**3:.2f} GB")

print("\n=== Config B: KV + resident MoE + resident LM head ===")
t_b = time.time()
text_b, ids_b, stats_b = greedy_generate_kv(
    model, tok, PROMPT, max_new_tokens=MAX_NEW,
    max_seqlen=64, print_stream=True)
elapsed_b = time.time() - t_b
print(f"\ntotal={elapsed_b:.1f}s  tokens={ids_b}")

# ============ Summary ============
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Config A (MoE resident):        {elapsed_a:.1f}s   tokens={ids_a}")
print(f"  Config B (MoE + LM head):       {elapsed_b:.1f}s   tokens={ids_b}")
print(f"  Identical: {ids_a == ids_b}")
if elapsed_b > 0:
    print(f"  Speedup A -> B: {elapsed_a/elapsed_b:.2f}x")
print()
print(f"  A prefill: {stats_a['prefill_time']:.1f}s  "
      f"decode avg: "
      f"{sum(stats_a['decode_times'])/max(1,len(stats_a['decode_times'])):.2f}s/tok")
print(f"  B prefill: {stats_b['prefill_time']:.1f}s  "
      f"decode avg: "
      f"{sum(stats_b['decode_times'])/max(1,len(stats_b['decode_times'])):.2f}s/tok")
