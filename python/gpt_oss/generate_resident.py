"""KV prefill plus fully resident single-token gpt-oss decode."""
from __future__ import annotations

import time

import torch

from .resident_decode import ResidentDecodeWorkspace, resident_decode_model_step


@torch.no_grad()
def greedy_generate_resident(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 48,
    max_seqlen: int = 256,
    print_stream: bool = True,
):
    """Generate greedily with CPU prefill and GPU-resident decode."""
    if model.resident_projections is None or model.h_lm_head is None:
        raise RuntimeError(
            "pin projection weights and LM head before resident decode"
        )
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    input_ids = tokenizer.encode(prompt, return_tensors="pt").long()
    prompt_length = input_ids.shape[1]
    if prompt_length + max_new_tokens > max_seqlen:
        raise ValueError(
            f"prompt ({prompt_length}) + generation ({max_new_tokens}) "
            f"exceeds max_seqlen {max_seqlen}"
        )

    cache = model.make_kv_cache(max_seqlen)
    prefill_started = time.time()
    logits = model.forward(
        input_ids, past_kv=cache, past_len=0, only_last_logits=True
    )
    cache.advance(prompt_length)
    prefill_time = time.time() - prefill_started

    workspace = ResidentDecodeWorkspace(
        model.ext, model.cfg, model.cfg.num_hidden_layers, max_seqlen
    )
    workspace.load_kv_cache(cache)
    next_id = int(logits[0, -1].argmax())
    new_ids = [next_id]
    decode_times: list[float] = []

    if print_stream:
        print(prompt, end="", flush=True)
        print(f"[prefill {prefill_time:.1f}s]", end="", flush=True)
        print(
            tokenizer.decode([next_id], skip_special_tokens=False),
            end="",
            flush=True,
        )

    try:
        position = prompt_length
        for _ in range(max_new_tokens - 1):
            started = time.time()
            step_logits, _ = resident_decode_model_step(
                model, workspace, next_id, position
            )
            decode_times.append(time.time() - started)
            next_id = int(step_logits[0].argmax())
            new_ids.append(next_id)
            position += 1
            if print_stream:
                piece = tokenizer.decode([next_id], skip_special_tokens=False)
                print(piece, end=f"[{decode_times[-1]:.2f}s]", flush=True)
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None and next_id == eos:
                break
    finally:
        workspace.free()

    if print_stream:
        print()
    text = tokenizer.decode(
        input_ids[0].tolist() + new_ids, skip_special_tokens=False
    )
    stats = {
        "prefill_time": prefill_time,
        "decode_times": decode_times,
        "num_prompt_tokens": prompt_length,
        "num_new_tokens": len(new_ids),
    }
    return text, new_ids, stats
