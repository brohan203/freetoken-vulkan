"""One dense Qwen3 transformer layer using reusable Vulkan primitives."""
from __future__ import annotations

import math
import torch

from dense_kv_cache import DenseKVCache
from model_contracts import DenseDecoderConfig, DenseDecoderLayerWeights


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _cpu_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return x * cos[None, None, :, :] + _rotate_half(x) * sin[None, None, :, :]


def _cpu_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    repeats = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    probabilities = torch.softmax(
        torch.matmul(query, key.transpose(-1, -2)) * scale,
        dim=-1,
    )
    return torch.matmul(probabilities, value)


def qwen3_layer_forward(
    ext,
    hidden_states: torch.Tensor,
    weights: DenseDecoderLayerWeights,
    config: DenseDecoderConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_idx: int = 0,
    past_kv: DenseKVCache | None = None,
    past_len: int = 0,
) -> torch.Tensor:
    """Run a Qwen3 prefill sequence or KV-cached decode step."""
    if hidden_states.dtype != torch.float32:
        raise TypeError("Qwen3 layer input must be FP32")
    if hidden_states.dim() != 3:
        raise ValueError("Qwen3 layer input must be [B,S,D]")
    batch, sequence, hidden = hidden_states.shape
    if hidden != config.hidden_size:
        raise ValueError("Qwen3 hidden size mismatch")

    normalized = ext.rmsnorm(
        hidden_states, weights.input_norm, config.rms_norm_eps
    )
    q = (normalized @ weights.q_weight.T).reshape(
        batch, sequence, config.num_attention_heads, config.head_dim
    )
    k = (normalized @ weights.k_weight.T).reshape(
        batch, sequence, config.num_key_value_heads, config.head_dim
    )
    v = (normalized @ weights.v_weight.T).reshape(
        batch, sequence, config.num_key_value_heads, config.head_dim
    )
    if weights.q_norm is None or weights.k_norm is None:
        raise ValueError("Qwen3 requires per-head Q/K RMSNorm weights")
    q = ext.rmsnorm(q, weights.q_norm, config.rms_norm_eps)
    k = ext.rmsnorm(k, weights.k_norm, config.rms_norm_eps)
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()

    single_decode = past_kv is not None and sequence == 1
    if single_decode:
        q = _cpu_rope(q, cos, sin)
        k = _cpu_rope(k, cos, sin)
    else:
        q = ext.rope_partial(q, cos, sin, config.head_dim)
        k = ext.rope_partial(k, cos, sin, config.head_dim)

    if past_kv is None:
        sinks = torch.zeros(config.num_attention_heads, dtype=torch.float32)
        attention = ext.flash_attention_gpt_oss(
            q,
            k,
            v,
            sinks,
            1.0 / math.sqrt(config.head_dim),
            0,
            False,
        )
    else:
        past_kv.append(layer_idx, k, v, past_len)
        key, value = past_kv.slice(layer_idx, past_len + sequence)
        if single_decode:
            attention = _cpu_gqa_attention(
                q, key, value, 1.0 / math.sqrt(config.head_dim)
            )
        else:
            sinks = torch.zeros(config.num_attention_heads, dtype=torch.float32)
            attention = ext.flash_attention_gpt_oss_kv(
                q,
                key,
                value,
                sinks,
                1.0 / math.sqrt(config.head_dim),
                past_len,
                0,
                False,
            )
    attention = attention.permute(0, 2, 1, 3).reshape(
        batch, sequence, config.query_size
    )
    hidden_states = hidden_states + attention @ weights.o_weight.T

    normalized = ext.rmsnorm(
        hidden_states, weights.post_attention_norm, config.rms_norm_eps
    )
    gate = normalized @ weights.gate_weight.T
    up = normalized @ weights.up_weight.T
    activated = ext.swiglu(gate, up)
    return hidden_states + activated @ weights.down_weight.T
