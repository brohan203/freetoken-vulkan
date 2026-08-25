"""Long-lived fully resident Qwen3 prompt loop."""
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

from qwen3 import Qwen3Model, ResidentQwen3Workspace, greedy_generate_resident


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?")
    parser.add_argument(
        "--model-dir",
        type=pathlib.Path,
        default=pathlib.Path(r"C:\Users\rohanborkar\Downloads\Qwen3-4B"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    sdk = pathlib.Path(os.environ["VULKAN_SDK"])
    ext = load(
        name="freetoken_vulkan_ext",
        sources=[str(HERE / "ext_module.cpp")],
        extra_include_paths=[str(sdk / "Include"), str(REPO / "include")],
        extra_ldflags=[f"/LIBPATH:{sdk / 'Lib'}", "vulkan-1.lib"],
        extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
        verbose=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = Qwen3Model.from_pretrained(ext, args.model_dir)
    started = time.time()
    model.pin_to_vram(False)
    pin_seconds = time.time() - started
    workspace = ResidentQwen3Workspace(ext, model.config, args.max_seq_len)
    print(
        f"[ready] pin={pin_seconds:.2f}s "
        f"resident={ext.resident_bytes_total()/1024**3:.2f}GiB"
    )

    def run(prompt: str) -> str:
        started = time.time()
        text, _, stats = greedy_generate_resident(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.max_seq_len,
            True,
            workspace,
        )
        elapsed = time.time() - started
        decode = sum(stats["decode_times"]) / max(1, len(stats["decode_times"]))
        prompt_average = sum(stats["prompt_times"]) / max(
            1, len(stats["prompt_times"])
        )
        print(
            f"[stats] total={elapsed:.2f}s "
            f"prompt={prompt_average:.3f}s/token "
            f"decode={decode:.3f}s/token"
        )
        return text

    try:
        if args.prompt is not None:
            run(args.prompt)
            return
        print("Qwen3 ready. Empty line exits.")
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                break
            run(prompt)
    finally:
        workspace.free()
        model.close()


if __name__ == "__main__":
    main()
