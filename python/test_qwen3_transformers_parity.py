"""Compare full Qwen3-4B logits with the canonical Transformers implementation."""
from __future__ import annotations

import gc
import os
import pathlib
import sys
import time

os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = (
    os.path.join(os.environ["VULKAN_SDK"], "Bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from qwen3 import Qwen3Model

MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\Qwen3-4B")
REPORT = HERE / "qwen3_transformers_parity.txt"


def main() -> None:
    torch.set_num_threads(12)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    input_ids = tokenizer.encode(
        "The capital of France is", return_tensors="pt"
    ).long()

    started = time.time()
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    with torch.no_grad():
        reference = reference_model(input_ids).logits[:, -1:, :].float()
    reference_seconds = time.time() - started
    del reference_model
    gc.collect()

    sdk = pathlib.Path(os.environ["VULKAN_SDK"])
    ext = load(
        name="freetoken_vulkan_ext",
        sources=[str(HERE / "ext_module.cpp")],
        extra_include_paths=[str(sdk / "Include"), str(REPO / "include")],
        extra_ldflags=[f"/LIBPATH:{sdk / 'Lib'}", "vulkan-1.lib"],
        extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
        verbose=False,
    )
    model = Qwen3Model.from_pretrained(ext, MODEL)
    started = time.time()
    actual = model.forward(input_ids, only_last_logits=True)
    actual_seconds = time.time() - started

    difference = (actual - reference).abs()
    ref_values, ref_indices = torch.topk(reference[0, -1], 10)
    act_values, act_indices = torch.topk(actual[0, -1], 10)
    ref_top = [int(index) for index in ref_indices]
    act_top = [int(index) for index in act_indices]
    lines = [
        f"reference_seconds={reference_seconds:.6f}",
        f"actual_seconds={actual_seconds:.6f}",
        f"max_difference={difference.max().item():.9g}",
        f"mean_difference={difference.mean().item():.9g}",
        f"reference_top_ids={ref_top!r}",
        f"actual_top_ids={act_top!r}",
        f"reference_top_tokens={[tokenizer.decode([i]) for i in ref_top]!r}",
        f"actual_top_tokens={[tokenizer.decode([i]) for i in act_top]!r}",
        f"top1_equal={ref_top[0] == act_top[0]}",
        f"top5_overlap={len(set(ref_top[:5]) & set(act_top[:5]))}",
        f"finite={torch.isfinite(actual).all().item()}",
    ]
    REPORT.write_text("\n".join(lines), encoding="ascii", errors="backslashreplace")
    assert torch.isfinite(actual).all()
    assert ref_top[0] == act_top[0]
    assert len(set(ref_top[:5]) & set(act_top[:5])) >= 4
    print("QWEN3_TRANSFORMERS_PARITY_OK")


if __name__ == "__main__":
    main()
