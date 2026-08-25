"""Generic long-lived gpt-oss-20b/120b resident prompt loop."""
from __future__ import annotations

import argparse
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
from transformers import AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from gpt_oss import GptOssConfig, GptOssModel, greedy_generate_resident
from gpt_oss.generate import greedy_generate_kv
from gpt_oss.resident_decode import ResidentDecodeWorkspace


def load_runtime(model_dir, threads, slots, legacy):
    torch.set_num_threads(threads)
    sdk = pathlib.Path(os.environ["VULKAN_SDK"])
    ext = load(
        name="freetoken_vulkan_ext",
        sources=[str(HERE / "ext_module.cpp")],
        extra_include_paths=[str(sdk / "Include"), str(REPO / "include")],
        extra_ldflags=[f"/LIBPATH:{sdk / 'Lib'}", "vulkan-1.lib"],
        extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
        verbose=False,
    )
    config = GptOssConfig.from_json(model_dir / "config.json")
    all_experts_resident = config.num_experts == 32
    model = GptOssModel.from_pretrained(
        ext, model_dir, stream_experts=not all_experts_resident
    )
    if all_experts_resident:
        model.pin_moe_to_vram()
    else:
        model.enable_streamed_vram_cache(slots or 18, "lfu")
    model.pin_lm_head_to_vram()
    if not legacy:
        model.pin_projections_to_vram(False)
    return model, AutoTokenizer.from_pretrained(model_dir)


def run_prompt(model, tokenizer, prompt, max_new, capacity, legacy, workspace):
    cache = model.streamed_resident
    hits_before = cache.hits if cache is not None else 0
    misses_before = cache.misses if cache is not None else 0
    started = time.time()
    generator = greedy_generate_kv if legacy else greedy_generate_resident
    kwargs = {} if legacy else {"workspace": workspace}
    text, _, stats = generator(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new,
        max_seqlen=capacity,
        print_stream=True,
        **kwargs,
    )
    elapsed = time.time() - started
    hits = cache.hits - hits_before if cache is not None else 0
    misses = cache.misses - misses_before if cache is not None else 0
    average = sum(stats["decode_times"]) / max(1, len(stats["decode_times"]))
    cache_text = (
        f"{100.0 * hits / max(1, hits + misses):.1f}%"
        if cache is not None
        else "all-resident"
    )
    print(
        f"[stats] total={elapsed:.2f}s prefill={stats['prefill_time']:.2f}s "
        f"decode={average:.3f}s/token cache={cache_text}"
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--legacy-decode", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_runtime(
        args.model_dir, args.threads, args.slots, args.legacy_decode
    )
    workspace = (
        None
        if args.legacy_decode
        else ResidentDecodeWorkspace(
            model.ext,
            model.cfg,
            model.cfg.num_hidden_layers,
            args.max_seq_len,
        )
    )
    try:
        if args.prompt is not None:
            run_prompt(
                model, tokenizer, args.prompt, args.max_new_tokens,
                args.max_seq_len, args.legacy_decode, workspace,
            )
            return
        mode = "legacy" if args.legacy_decode else "resident"
        print(f"gpt-oss ready ({mode} decode). Empty line exits.")
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            run_prompt(
                model, tokenizer, prompt, args.max_new_tokens,
                args.max_seq_len, args.legacy_decode, workspace,
            )
    finally:
        if workspace is not None:
            workspace.free()
        model.close()


if __name__ == "__main__":
    main()
