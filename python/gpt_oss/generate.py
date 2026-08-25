"""generate.py — greedy / top-k text generation for gpt-oss.

Two decode strategies:

    greedy_generate:    O(N^2) — recomputes the full sequence each step.
                        Kept for correctness reference.
    greedy_generate_kv: O(N) — uses the KV cache, only processes the new
                        token each step. Should be dramatically faster
                        once past the prefill step.
"""
from __future__ import annotations
import time
from typing import Tuple

import torch


# ============================================================
# Slow reference: no KV cache. Recompute everything each step.
# ============================================================
@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 20,
                    print_stream: bool = True) -> Tuple[str, list[int]]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").long()
    if print_stream:
        print(prompt, end="", flush=True)
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        t0 = time.time()
        logits = model.forward(input_ids)
        next_id = int(logits[0, -1].argmax().item())
        elapsed = time.time() - t0
        new_ids.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
        if print_stream:
            piece = tokenizer.decode([next_id], skip_special_tokens=False)
            print(piece, end="", flush=True)
            print(f"[{elapsed:.1f}s]", end="", flush=True)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and next_id == eos:
            break
    if print_stream:
        print()
    text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
    return text, new_ids


# ============================================================
# Fast: KV-cache decode.
# ============================================================
@torch.no_grad()
def greedy_generate_kv(model, tokenizer, prompt: str,
                       max_new_tokens: int = 20,
                       max_seqlen: int | None = None,
                       print_stream: bool = True
                       ) -> Tuple[str, list[int], dict]:
    """Returns (full_text, new_token_ids, stats). stats includes:
        prefill_time, decode_times (list per token), total_time.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").long()
    B, S_prompt = input_ids.shape

    max_seqlen = max_seqlen or (S_prompt + max_new_tokens + 4)
    cache = model.make_kv_cache(max_seqlen=max_seqlen, batch=B)

    # ---- Prefill ----
    t_pre = time.time()
    logits = model.forward(input_ids, past_kv=cache, past_len=0,
                            only_last_logits=True)
    cache.advance(S_prompt)
    prefill_time = time.time() - t_pre

    next_id = int(logits[0, -1].argmax().item())
    if print_stream:
        print(prompt, end="", flush=True)
        print(f"[prefill {S_prompt}tok {prefill_time:.1f}s]", flush=True)

    new_ids: list[int] = [next_id]
    if print_stream:
        piece = tokenizer.decode([next_id], skip_special_tokens=False)
        print(piece, end="", flush=True)

    decode_times = []

    # ---- Decode: one token at a time ----
    for step in range(max_new_tokens - 1):
        t0 = time.time()
        cur_pos = cache.cur_len
        step_ids = torch.tensor([[next_id]], dtype=torch.long)
        logits = model.forward(step_ids, past_kv=cache, past_len=cur_pos,
                                only_last_logits=True)
        cache.advance(1)
        next_id = int(logits[0, -1].argmax().item())
        dt = time.time() - t0
        decode_times.append(dt)
        new_ids.append(next_id)
        if print_stream:
            piece = tokenizer.decode([next_id], skip_special_tokens=False)
            print(piece, end="", flush=True)
            print(f"[{dt:.1f}s]", end="", flush=True)

        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and next_id == eos:
            break

    if print_stream:
        print()
    full_ids = torch.cat([input_ids, torch.tensor([new_ids], dtype=torch.long)], dim=1)
    text = tokenizer.decode(full_ids[0].tolist(), skip_special_tokens=False)
    stats = {
        "prefill_time": prefill_time,
        "decode_times": decode_times,
        "num_prompt_tokens": S_prompt,
        "num_new_tokens": len(new_ids),
    }
    return text, new_ids, stats


@torch.no_grad()
def top_k_predictions(model, tokenizer, prompt: str, k: int = 10):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").long()
    t0 = time.time()
    logits = model.forward(input_ids)
    top_vals, top_ids = torch.topk(logits[0, -1], k=k)
    print(f"[top_k] forward: {time.time()-t0:.1f}s")
    print(f"[top_k] Top-{k} predictions for {prompt!r}:")
    for v, i in zip(top_vals.tolist(), top_ids.tolist()):
        piece = tokenizer.decode([i], skip_special_tokens=False)
        print(f"  {i:6d}  logit={v:9.3f}  {piece!r}")
    return top_vals.tolist(), top_ids.tolist()
