"""Long-lived gpt-oss-120b prompt loop with resident decode by default."""
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

from gpt_oss import GptOssModel, greedy_generate_resident
from gpt_oss.generate import greedy_generate_kv
from gpt_oss.resident_decode import ResidentDecodeWorkspace


def build_model(
    model_dir: pathlib.Path,
    slots: int,
    threads: int,
    resident_decode: bool,
):
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
    model = GptOssModel.from_pretrained(ext, model_dir, stream_experts=True)
    model.enable_streamed_vram_cache(slots, "lfu")
    model.pin_lm_head_to_vram()
    if resident_decode:
        model.pin_projections_to_vram(False)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def run_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new: int,
    max_seq: int,
    resident_decode: bool,
    workspace: ResidentDecodeWorkspace | None = None,
):
    hits_before = model.streamed_resident.hits
    misses_before = model.streamed_resident.misses
    started = time.perf_counter()
    generator = greedy_generate_resident if resident_decode else greedy_generate_kv
    text, _, stats = generator(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new,
        max_seqlen=max_seq,
        print_stream=True,
        **({"workspace": workspace} if resident_decode else {}),
    )
    elapsed = time.perf_counter() - started
    hits = model.streamed_resident.hits - hits_before
    misses = model.streamed_resident.misses - misses_before
    decode_times = stats["decode_times"]
    decode_average = sum(decode_times) / max(1, len(decode_times))
    print(
        f"[stats] total={elapsed:.2f}s "
        f"prefill={stats['prefill_time']:.2f}s "
        f"decode={decode_average:.3f}s/token "
        f"cache={100.0 * hits / max(1, hits + misses):.1f}%"
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            r"C:\Users\rohanborkar\Downloads\gpt-oss-120b"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--slots", type=int, default=18)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--legacy-decode",
        action="store_true",
        help="Use the CPU/tensor decode path instead of resident decode.",
    )
    args = parser.parse_args()
    resident_decode = not args.legacy_decode
    model, tokenizer = build_model(
        args.model_dir, args.slots, args.threads, resident_decode
    )
    workspace = (
        ResidentDecodeWorkspace(
            model.ext, model.cfg, model.cfg.num_hidden_layers,
            args.max_seq_len,
        )
        if resident_decode
        else None
    )

    if args.prompt is not None:
        try:
            run_prompt(
                model, tokenizer, args.prompt, args.max_new_tokens,
                args.max_seq_len, resident_decode, workspace,
            )
        finally:
            if workspace is not None:
                workspace.free()
        return

    mode = "resident" if resident_decode else "legacy"
    print(f"gpt-oss-120b ready ({mode} decode). Empty line exits.")
    mode = "resident" if resident_decode else "legacy"
    print(f"gpt-oss-120b ready ({mode} decode). Empty line exits.")
    try:
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            run_prompt(
                model, tokenizer, prompt, args.max_new_tokens,
                args.max_seq_len, resident_decode, workspace,
            )
    finally:
        if workspace is not None:
            workspace.free()
if __name__ == "__main__":
    main()
