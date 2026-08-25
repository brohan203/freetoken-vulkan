"""Full Qwen3-4B next-token smoke with lazy per-layer loading."""
from __future__ import annotations

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

import psutil
import torch
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from qwen3 import Qwen3Model

MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\Qwen3-4B")
REPORT = HERE / "qwen3_full_model_report.txt"


def rss_gib() -> float:
    return psutil.Process().memory_info().rss / 1024**3


def main() -> None:
    torch.set_num_threads(12)
    sdk = pathlib.Path(os.environ["VULKAN_SDK"])
    ext = load(
        name="freetoken_vulkan_ext",
        sources=[str(HERE / "ext_module.cpp")],
        extra_include_paths=[str(sdk / "Include"), str(REPO / "include")],
        extra_ldflags=[f"/LIBPATH:{sdk / 'Lib'}", "vulkan-1.lib"],
        extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
        verbose=False,
    )
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = Qwen3Model.from_pretrained(ext, MODEL)
    load_seconds = time.time() - started
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").long()
    before = rss_gib()
    started = time.time()
    logits = model.forward(
        input_ids, only_last_logits=True, collect_layer_times=True
    )
    forward_seconds = time.time() - started
    after = rss_gib()
    values, indices = torch.topk(logits[0, -1], 10)
    top = [
        (int(index), float(value), tokenizer.decode([int(index)]))
        for value, index in zip(values, indices)
    ]
    lines = [
        f"prompt={prompt!r}",
        f"input_ids={input_ids[0].tolist()!r}",
        f"shape={tuple(logits.shape)!r}",
        f"finite={torch.isfinite(logits).all().item()}",
        f"load_seconds={load_seconds:.6f}",
        f"forward_seconds={forward_seconds:.6f}",
        f"rss_before_gib={before:.6f}",
        f"rss_after_gib={after:.6f}",
        f"layer_mean_seconds={sum(model.layer_times)/len(model.layer_times):.6f}",
        f"layer_max_seconds={max(model.layer_times):.6f}",
        f"top={top!r}",
    ]
    REPORT.write_text("\n".join(lines), encoding="ascii", errors="backslashreplace")
    assert logits.shape == (1, 1, model.config.vocab_size)
    assert torch.isfinite(logits).all()
    assert len(model.layer_times) == model.config.num_hidden_layers
    print("QWEN3_FULL_MODEL_OK")


if __name__ == "__main__":
    main()
