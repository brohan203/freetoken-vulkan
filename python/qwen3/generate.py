"""KV-cached greedy generation for the lazy Qwen3 runtime."""
from __future__ import annotations

import time
import torch

from .model import Qwen3Model
from .resident import ResidentQwen3Workspace, resident_qwen3_model_step


@torch.no_grad()
def greedy_generate(
    model: Qwen3Model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    max_seqlen: int = 256,
    print_stream: bool = False,
):
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").long()
    prompt_length = input_ids.shape[1]
    if prompt_length + max_new_tokens > max_seqlen:
        raise ValueError("prompt plus generation exceeds max_seqlen")
    cache = model.make_kv_cache(max_seqlen=max_seqlen)
    started = time.time()
    logits = model.forward(
        input_ids, only_last_logits=True, past_kv=cache, past_len=0
    )
    cache.advance(prompt_length)
    prefill_seconds = time.time() - started
    next_id = int(logits[0, -1].argmax())
    generated = [next_id]
    decode_times: list[float] = []
    if print_stream:
        print(prompt, end="", flush=True)
        print(tokenizer.decode([next_id]), end="", flush=True)
    for _ in range(max_new_tokens - 1):
        started = time.time()
        token = torch.tensor([[next_id]], dtype=torch.long)
        logits = model.forward(
            token,
            only_last_logits=True,
            past_kv=cache,
            past_len=cache.cur_len,
        )
        cache.advance(1)
        decode_times.append(time.time() - started)
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        if print_stream:
            print(tokenizer.decode([next_id]), end="", flush=True)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
    if print_stream:
        print()
    all_ids = input_ids[0].tolist() + generated
    return tokenizer.decode(all_ids), generated, {
        "prefill_seconds": prefill_seconds,
        "decode_times": decode_times,
        "cache_bytes": cache.memory_bytes(),
    }


@torch.no_grad()
def greedy_generate_resident(
    model: Qwen3Model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    max_seqlen: int = 256,
    print_stream: bool = False,
    workspace: ResidentQwen3Workspace | None = None,
):
    if model.resident_weights is None:
        raise RuntimeError("call model.pin_to_vram() before resident generation")
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) + max_new_tokens > max_seqlen:
        raise ValueError("prompt plus generation exceeds max_seqlen")
    owns_workspace = workspace is None
    if workspace is None:
        workspace = ResidentQwen3Workspace(model.ext, model.config, max_seqlen)
    elif workspace.capacity != max_seqlen:
        raise ValueError("workspace capacity must match max_seqlen")
    prompt_times: list[float] = []
    logits = None
    for position, token_id in enumerate(input_ids):
        started = time.time()
        logits = resident_qwen3_model_step(model, workspace, token_id, position)
        prompt_times.append(time.time() - started)
    assert logits is not None
    next_id = int(logits[0].argmax())
    generated = [next_id]
    decode_times: list[float] = []
    if print_stream:
        print(prompt + tokenizer.decode([next_id]), end="", flush=True)
    position = len(input_ids)
    for _ in range(max_new_tokens - 1):
        started = time.time()
        logits = resident_qwen3_model_step(model, workspace, next_id, position)
        decode_times.append(time.time() - started)
        next_id = int(logits[0].argmax())
        generated.append(next_id)
        position += 1
        if print_stream:
            print(tokenizer.decode([next_id]), end="", flush=True)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break
    if print_stream:
        print()
    text = tokenizer.decode(input_ids + generated)
    if owns_workspace:
        workspace.free()
    return text, generated, {
        "prompt_times": prompt_times,
        "decode_times": decode_times,
    }
