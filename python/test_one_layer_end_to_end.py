"""test_one_layer_end_to_end.py — the critical integration test.

Loads REAL gpt-oss-20b layer-0 weights, runs one full transformer layer
via our Vulkan kernels, and diffs against a pure-PyTorch reference that
matches transformers' modeling code exactly.

The reference:
    - rmsnorm on input
    - Q/K/V projections + bias, reshape to [B, H, S, head_dim]
    - RoPE via transformers' apply_rotary_pos_emb
    - Attention with GQA + causal + optional sliding window + sinks
      (implemented as scores | sinks → softmax → drop sink)
    - O projection + bias + residual
    - rmsnorm
    - Router → top-K + softmax
    - MoE MLP with MXFP4 experts (dequantized) + gpt-oss activation + biases
    - Residual

If our Vulkan layer matches to ~1e-3 absolute error, we know every kernel
in the layer is composed correctly.
"""
from __future__ import annotations
import os, pathlib, sys, math, time

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from transformers import AutoConfig
from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))

from gpt_oss import (
    GptOssConfig, load_model, compute_cos_sin_for_positions,
    gpt_oss_layer_forward, rmsnorm,
)
from gpt_oss.mxfp4_ref import mxfp4_dequant

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
print("Loading extension...")
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")


# ==================== Reference forward (pure PyTorch) ====================

def ref_attn_with_sinks(Q, K, V, sinks, scale, sliding_window=0, use_sinks=True):
    """Matches ext.flash_attention_gpt_oss + transformers.gpt_oss attention.
    Q [B, H_q, S, D],  K/V [B, H_kv, S, D],  sinks [H_q]. Returns [B, H_q, S, D]."""
    B, H_q, S, D = Q.shape
    _, H_kv, _, _ = K.shape
    n_rep = H_q // H_kv

    Kr = K.repeat_interleave(n_rep, dim=1)
    Vr = V.repeat_interleave(n_rep, dim=1)

    scores = torch.einsum('bhsd,bhtd->bhst', Q, Kr) * scale
    # Causal
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    # Sliding window (also masks kv_pos < q_idx - window + 1).
    if sliding_window > 0:
        for q_i in range(S):
            lo = max(0, q_i - sliding_window + 1)
            scores[..., q_i, :lo] = float('-inf')

    if use_sinks:
        sink_col = sinks.reshape(1, H_q, 1, 1).expand(B, H_q, S, 1)
        combined = torch.cat([scores, sink_col], dim=-1)
        probs = torch.softmax(combined, dim=-1)
        probs = probs[..., :-1]
    else:
        probs = torch.softmax(scores, dim=-1)

    out = torch.einsum('bhst,bhtd->bhsd', probs, Vr)
    return out


def ref_moe_mlp(x, indices, routing_weights, W_gu_deq, gu_bias, W_d_deq, d_bias,
                alpha=1.702, limit=7.0):
    """Matches ext.moe_mlp_gpt_oss + transformers.GptOssExperts._apply_gate."""
    T, D = x.shape
    K = indices.shape[1]
    y = torch.zeros(T, D, dtype=torch.float32)

    for t in range(T):
        for k in range(K):
            e = indices[t, k].item()
            w = routing_weights[t, k].item()

            gate_up = W_gu_deq[e] @ x[t] + gu_bias[e]   # [2*Dff]
            gate = gate_up[0::2]
            up   = gate_up[1::2]

            gate = torch.clamp(gate, max=limit)
            up   = torch.clamp(up,   min=-limit, max=limit)
            glu  = gate * torch.sigmoid(gate * alpha)
            hidden = (up + 1.0) * glu

            out = W_d_deq[e] @ hidden + d_bias[e]
            y[t] += w * out
    return y


def ref_layer_forward(x, layer_idx, weights, cfg, hf_cos, hf_sin,
                      W_gu_deq, W_d_deq, sliding_window):
    B, S, D = x.shape
    H_q = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    # Attention
    residual = x
    x_n = rmsnorm(x, weights.input_layernorm_weight, cfg.rms_norm_eps)
    q = x_n @ weights.q_proj_weight.T + weights.q_proj_bias
    k = x_n @ weights.k_proj_weight.T + weights.k_proj_bias
    v = x_n @ weights.v_proj_weight.T + weights.v_proj_bias
    q = q.reshape(B, S, H_q,  head_dim).transpose(1, 2).contiguous()
    k = k.reshape(B, S, H_kv, head_dim).transpose(1, 2).contiguous()
    v = v.reshape(B, S, H_kv, head_dim).transpose(1, 2).contiguous()

    # RoPE via transformers' apply_rotary_pos_emb — cos/sin are [B=1, S, D/2]
    q, k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)

    attn_out = ref_attn_with_sinks(q, k, v, weights.sinks, scale,
                                    sliding_window=sliding_window, use_sinks=True)
    attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, S, H_q * head_dim)
    attn_out = attn_out @ weights.o_proj_weight.T + weights.o_proj_bias
    x = residual + attn_out

    # MoE
    residual = x
    x_n = rmsnorm(x, weights.post_attention_layernorm_weight, cfg.rms_norm_eps)
    router_logits = x_n @ weights.router_weight.T + weights.router_bias
    top_vals, top_idx = torch.topk(router_logits, cfg.num_experts_per_tok, dim=-1)
    routing_weights = torch.softmax(top_vals, dim=-1)

    T = B * S
    x_flat  = x_n.reshape(T, D)
    idx_flat = top_idx.reshape(T, cfg.num_experts_per_tok)
    w_flat  = routing_weights.reshape(T, cfg.num_experts_per_tok)

    mlp_out = ref_moe_mlp(x_flat, idx_flat, w_flat,
                           W_gu_deq, weights.gate_up_bias,
                           W_d_deq,  weights.down_bias)
    mlp_out = mlp_out.reshape(B, S, D)

    x = residual + mlp_out
    return x


# ==================== Test ====================

def main():
    print("\n=== Loading config + layer 0 weights ===")
    t0 = time.time()
    model = load_model(MODEL_DIR, layers=[0])
    cfg = model.config
    weights = model.layers[0]
    print(f"  loaded in {time.time()-t0:.1f}s")
    print(f"  cfg: D={cfg.hidden_size}  H_q={cfg.num_attention_heads}  "
          f"H_kv={cfg.num_key_value_heads}  head_dim={cfg.head_dim}  "
          f"E={cfg.num_experts}  K={cfg.num_experts_per_tok}")
    print(f"  sinks: {list(weights.sinks.shape)} {weights.sinks.dtype}")
    print(f"  layer 0 is {'sliding' if cfg.layer_is_sliding(0) else 'full'} attention")

    print("\n=== Dequantizing MoE experts for reference (slow, one-time) ===")
    t0 = time.time()
    W_gu_deq = mxfp4_dequant(weights.gate_up_blocks, weights.gate_up_scales)
    W_d_deq  = mxfp4_dequant(weights.down_blocks,    weights.down_scales)
    print(f"  dequant in {time.time()-t0:.1f}s  "
          f"gate_up: {list(W_gu_deq.shape)}  down: {list(W_d_deq.shape)}")

    print("\n=== Preparing RoPE cos/sin ===")
    hf_cfg = AutoConfig.from_pretrained(MODEL_DIR)
    B, S = 1, 8
    positions = torch.arange(S)
    cos_shader, sin_shader = compute_cos_sin_for_positions(hf_cfg, positions)   # [S, head_dim]

    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding
    rot = GptOssRotaryEmbedding(hf_cfg)
    dummy = torch.zeros(B, S, cfg.hidden_size)
    hf_cos, hf_sin = rot(dummy, positions.unsqueeze(0))   # [1, S, head_dim/2]
    print(f"  hf cos: {list(hf_cos.shape)}  shader cos: {list(cos_shader.shape)}")

    print("\n=== Generating random input activation ===")
    torch.manual_seed(0xF00D)
    x = torch.randn(B, S, cfg.hidden_size) * 0.05
    print(f"  x: {list(x.shape)}  mean={x.mean():.4f}  std={x.std():.4f}")

    sliding_window = cfg.sliding_window if cfg.layer_is_sliding(0) else 0

    print("\n=== Running Vulkan layer forward ===")
    t0 = time.time()
    y_gpu = gpt_oss_layer_forward(
        ext, x, layer_idx=0, weights=weights, cfg=cfg,
        cos=cos_shader, sin=sin_shader, use_sinks=True,
    )
    print(f"  Vulkan layer: {time.time()-t0:.2f}s  out shape={list(y_gpu.shape)}")

    print("\n=== Running reference layer forward (pure PyTorch, slow) ===")
    t0 = time.time()
    y_ref = ref_layer_forward(x, 0, weights, cfg, hf_cos, hf_sin,
                               W_gu_deq, W_d_deq, sliding_window)
    print(f"  reference:    {time.time()-t0:.2f}s")

    # ==== Diff ====
    print("\n=== Comparison ===")
    diff = (y_gpu - y_ref).abs()
    max_abs = diff.max().item()
    rel = diff / y_ref.abs().clamp_min(1e-4)
    max_rel = rel.max().item()
    print(f"  y_ref: mean={y_ref.mean():.4f}  std={y_ref.std():.4f}  "
          f"range=[{y_ref.min():.3f}, {y_ref.max():.3f}]")
    print(f"  y_gpu: mean={y_gpu.mean():.4f}  std={y_gpu.std():.4f}  "
          f"range=[{y_gpu.min():.3f}, {y_gpu.max():.3f}]")
    print(f"  max|D|       = {max_abs:.3e}")
    print(f"  max|D/y|     = {max_rel:.3e}")
    ok = torch.allclose(y_gpu, y_ref, rtol=5e-3, atol=1e-3)
    print(f"\n  {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
