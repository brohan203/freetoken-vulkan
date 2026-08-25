"""One gpt-oss transformer layer with optional KV and expert caches."""
from __future__ import annotations

import math

import torch

from .config import GptOssConfig
from .kv_cache import KVCache
from .loader import ExpertStore, LayerWeights
from .resident import ResidentLayerHandles


def rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def gpt_oss_layer_forward(
    ext,
    x: torch.Tensor,
    layer_idx: int,
    weights: LayerWeights,
    cfg: GptOssConfig,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_kv: KVCache | None = None,
    past_len: int = 0,
    use_sinks: bool = True,
    resident_moe: ResidentLayerHandles | None = None,
    expert_store: ExpertStore | None = None,
    streamed_resident=None,
) -> torch.Tensor:
    batch, new_tokens, hidden = x.shape
    query_heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    residual = x
    normalized = rmsnorm(
        x, weights.input_layernorm_weight, cfg.rms_norm_eps
    )
    query = normalized @ weights.q_proj_weight.T + weights.q_proj_bias
    key = normalized @ weights.k_proj_weight.T + weights.k_proj_bias
    value = normalized @ weights.v_proj_weight.T + weights.v_proj_bias
    query = query.reshape(
        batch, new_tokens, query_heads, head_dim
    ).transpose(1, 2).contiguous()
    key = key.reshape(
        batch, new_tokens, kv_heads, head_dim
    ).transpose(1, 2).contiguous()
    value = value.reshape(
        batch, new_tokens, kv_heads, head_dim
    ).transpose(1, 2).contiguous()
    query = ext.rope_partial(query, cos, sin, head_dim)
    key = ext.rope_partial(key, cos, sin, head_dim)

    sliding_window = (
        cfg.sliding_window if cfg.layer_is_sliding(layer_idx) else 0
    )
    if past_kv is None:
        attention = ext.flash_attention_gpt_oss(
            query,
            key,
            value,
            weights.sinks,
            scale,
            sliding_window=sliding_window,
            use_sinks=use_sinks,
        )
    else:
        past_kv.append(layer_idx, key, value, positions_start=past_len)
        kv_length = past_len + new_tokens
        full_key, full_value = past_kv.slice(layer_idx, seq_end=kv_length)
        attention = ext.flash_attention_gpt_oss_kv(
            query,
            full_key,
            full_value,
            weights.sinks,
            scale,
            past_len=past_len,
            sliding_window=sliding_window,
            use_sinks=use_sinks,
        )
    attention = attention.transpose(1, 2).contiguous().reshape(
        batch, new_tokens, query_heads * head_dim
    )
    attention = attention @ weights.o_proj_weight.T + weights.o_proj_bias
    x = residual + attention

    residual = x
    normalized = rmsnorm(
        x, weights.post_attention_layernorm_weight, cfg.rms_norm_eps
    )
    router_logits = normalized @ weights.router_weight.T + weights.router_bias
    top_values, top_indices = torch.topk(
        router_logits, cfg.num_experts_per_tok, dim=-1
    )
    routing_weights = torch.softmax(top_values, dim=-1)
    tokens = batch * new_tokens
    flat_x = normalized.reshape(tokens, hidden).contiguous()
    flat_indices = top_indices.reshape(
        tokens, cfg.num_experts_per_tok
    ).contiguous()
    flat_weights = routing_weights.reshape(
        tokens, cfg.num_experts_per_tok
    ).contiguous()

    if expert_store is not None and streamed_resident is not None:
        mlp = streamed_resident.call(
            layer_idx,
            expert_store,
            flat_x,
            flat_indices,
            flat_weights,
        )
    elif expert_store is not None:
        compact, local_indices = expert_store.materialize_selected(
            layer_idx, flat_indices
        )
        mlp = ext.moe_mlp_gpt_oss(
            flat_x,
            local_indices,
            flat_weights,
            compact.gate_up_blocks,
            compact.gate_up_scales,
            compact.gate_up_bias,
            compact.down_blocks,
            compact.down_scales,
            compact.down_bias,
        )
    elif resident_moe is not None:
        mlp = resident_moe.call(
            ext, flat_x, flat_indices, flat_weights
        )
    else:
        if weights.gate_up_blocks is None:
            raise RuntimeError(
                "expert weights are not loaded and no ExpertStore was provided"
            )
        mlp = ext.moe_mlp_gpt_oss(
            flat_x,
            flat_indices,
            flat_weights,
            weights.gate_up_blocks,
            weights.gate_up_scales,
            weights.gate_up_bias,
            weights.down_blocks,
            weights.down_scales,
            weights.down_bias,
        )
    return residual + mlp.reshape(batch, new_tokens, hidden)
