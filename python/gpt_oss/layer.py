"""layer.py — one gpt-oss transformer layer forward pass, with optional
KV cache support.

Two modes:
    (a) No KV cache (past_kv=None): recompute everything. Used for prefill
        as a fallback, though prefill can also use the KV path.
    (b) KV cache (past_kv=KVCache): compute K, V for the query positions,
        append into the cache, then run attention against ALL cached K, V.

CPU-side pieces (still, until BF16 GEMM in shaders):
    - RMSNorm, Q/K/V/O projections, router, residual.
GPU-side pieces (our Vulkan shaders):
    - RoPE on Q, K
    - FlashAttention (KV-aware variant when past_kv is provided)
    - MoE MLP with MXFP4 experts
"""
from __future__ import annotations
import math
import torch

from .config import GptOssConfig
from .loader import LayerWeights
from .kv_cache import KVCache
from .resident import ResidentLayerHandles


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(var + eps)
    return x_norm * weight


def gpt_oss_layer_forward(
    ext,
    x: torch.Tensor,                # [B, S_new, D] input for the new positions
    layer_idx: int,
    weights: LayerWeights,
    cfg: GptOssConfig,
    cos: torch.Tensor,              # [S_new, head_dim] RoPE cos for NEW positions only
    sin: torch.Tensor,              # [S_new, head_dim]
    past_kv: KVCache | None = None,
    past_len: int = 0,              # absolute position of x[0]. Ignored if past_kv is None.
    use_sinks: bool = True,
    resident_moe: ResidentLayerHandles | None = None,
) -> torch.Tensor:
    B, S_new, D = x.shape
    H_q       = cfg.num_attention_heads
    H_kv      = cfg.num_key_value_heads
    head_dim  = cfg.head_dim
    scale     = 1.0 / math.sqrt(head_dim)

    # ---- Attention ----
    residual = x
    x_n = rmsnorm(x, weights.input_layernorm_weight, cfg.rms_norm_eps)

    q = x_n @ weights.q_proj_weight.T + weights.q_proj_bias
    k = x_n @ weights.k_proj_weight.T + weights.k_proj_bias
    v = x_n @ weights.v_proj_weight.T + weights.v_proj_bias

    q = q.reshape(B, S_new, H_q,  head_dim).transpose(1, 2).contiguous()
    k = k.reshape(B, S_new, H_kv, head_dim).transpose(1, 2).contiguous()
    v = v.reshape(B, S_new, H_kv, head_dim).transpose(1, 2).contiguous()

    # RoPE on new positions.
    q = ext.rope_partial(q, cos, sin, head_dim)
    k = ext.rope_partial(k, cos, sin, head_dim)

    sliding_window = cfg.sliding_window if cfg.layer_is_sliding(layer_idx) else 0

    if past_kv is None:
        # Backward-compat path: use the original shader (no cache).
        attn_out = ext.flash_attention_gpt_oss(
            q, k, v, weights.sinks, scale,
            sliding_window=sliding_window, use_sinks=use_sinks,
        )
    else:
        # Append to KV cache, then read the full slice up to past_len + S_new.
        past_kv.append(layer_idx, k, v, positions_start=past_len)
        S_kv = past_len + S_new
        K_full, V_full = past_kv.slice(layer_idx, seq_end=S_kv)
        attn_out = ext.flash_attention_gpt_oss_kv(
            q, K_full, V_full, weights.sinks, scale,
            past_len=past_len,
            sliding_window=sliding_window,
            use_sinks=use_sinks,
        )

    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, S_new, H_q * head_dim)
    attn_out = attn_out @ weights.o_proj_weight.T + weights.o_proj_bias
    x = residual + attn_out

    # ---- MoE MLP ----
    residual = x
    x_n = rmsnorm(x, weights.post_attention_layernorm_weight, cfg.rms_norm_eps)

    router_logits = x_n @ weights.router_weight.T + weights.router_bias
    top_vals, top_idx = torch.topk(router_logits, cfg.num_experts_per_tok, dim=-1)
    routing_weights = torch.softmax(top_vals, dim=-1)

    T = B * S_new
    x_flat  = x_n.reshape(T, D).contiguous()
    idx_flat = top_idx.reshape(T, cfg.num_experts_per_tok).contiguous()
    w_flat  = routing_weights.reshape(T, cfg.num_experts_per_tok).contiguous()

    if resident_moe is not None:
        # Fast path — MoE weights already in VRAM.
        mlp_out = resident_moe.call(ext, x_flat, idx_flat, w_flat)
    else:
        # Slow path — upload weights each call.
        mlp_out = ext.moe_mlp_gpt_oss(
            x_flat, idx_flat, w_flat,
            weights.gate_up_blocks, weights.gate_up_scales, weights.gate_up_bias,
            weights.down_blocks,    weights.down_scales,    weights.down_bias,
        )
    mlp_out = mlp_out.reshape(B, S_new, D)

    x = residual + mlp_out
    return x
