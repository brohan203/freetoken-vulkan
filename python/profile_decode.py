"""profile_decode.py — measure where time goes in a single decode step.

Layer-forward breakdown: RMSNorm, Q/K/V/O CPU matmul, RoPE, attention,
router, MoE MLP, residuals, LM head. Instrumented via time.time()
around each stage.
"""
from __future__ import annotations
import os, pathlib, sys, time, math

_VENV_SCRIPTS = pathlib.Path(sys.executable).parent
os.environ["PATH"] = str(_VENV_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = os.path.join(os.environ["VULKAN_SDK"], "Bin") + os.pathsep + os.environ["PATH"]
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer

HERE  = pathlib.Path(__file__).resolve().parent
REPO  = HERE.parent
PYDIR = REPO / "python"
sys.path.insert(0, str(PYDIR))
from gpt_oss import GptOssModel

MODEL_DIR = pathlib.Path(r"C:\Users\rohanborkar\Downloads\gpt-oss-20b")

vulkan_sdk = pathlib.Path(os.environ["VULKAN_SDK"])
ext = load(
    name="freetoken_vulkan_ext",
    sources=[str(PYDIR / "ext_module.cpp")],
    extra_include_paths=[str(vulkan_sdk / "Include"), str(REPO / "include")],
    extra_ldflags=[f"/LIBPATH:{vulkan_sdk / 'Lib'}", "vulkan-1.lib"],
    extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
    verbose=False,
)

model = GptOssModel.from_pretrained(ext, MODEL_DIR)
model.pin_moe_to_vram()

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
PROMPT = "The capital of France is"
input_ids = tok.encode(PROMPT, return_tensors="pt").long()
S_prompt = input_ids.shape[1]

# Prefill with cache.
cache = model.make_kv_cache(max_seqlen=32)
_ = model.forward(input_ids, past_kv=cache, past_len=0)
cache.advance(S_prompt)

# Now instrument ONE decode step end-to-end.
next_tok = 12650  # ' Paris' from earlier
step_ids = torch.tensor([[next_tok]], dtype=torch.long)

# Repeat the forward manually with timing hooks.
cfg = model.cfg
w = model.weights
D = cfg.hidden_size
H_q = cfg.num_attention_heads
H_kv = cfg.num_key_value_heads
head_dim = cfg.head_dim
scale = 1.0 / math.sqrt(head_dim)

from gpt_oss.layer import rmsnorm

# Warm-up.
for _ in range(2):
    _ = model.forward(step_ids, past_kv=cache, past_len=cache.cur_len)
    cache.cur_len -= 1  # rewind (dirty; only for timing)
# Actually let cur_len advance now
cache.cur_len -= 2  # reset from warmup

TOTAL_LAYERS = cfg.num_hidden_layers

# Now the instrumented step.
t_total = time.time()
positions = torch.arange(cache.cur_len, cache.cur_len + 1, dtype=torch.long)
cos, sin = model.compute_rope(positions)

x = w.embed_tokens[step_ids].float()

# Aggregate timings.
t_rmsnorm_1 = 0.0
t_qkv_proj = 0.0
t_rope = 0.0
t_attn = 0.0
t_o_proj = 0.0
t_res = 0.0
t_rmsnorm_2 = 0.0
t_router = 0.0
t_moe = 0.0

for i in range(TOTAL_LAYERS):
    lw = w.layers[i]
    resident = model.resident_moe.for_layer(i)
    B, S_new, _ = x.shape

    residual = x
    t0 = time.time(); x_n = rmsnorm(x, lw.input_layernorm_weight, cfg.rms_norm_eps); t_rmsnorm_1 += time.time()-t0

    t0 = time.time()
    q = x_n @ lw.q_proj_weight.T + lw.q_proj_bias
    k = x_n @ lw.k_proj_weight.T + lw.k_proj_bias
    v = x_n @ lw.v_proj_weight.T + lw.v_proj_bias
    q = q.reshape(B, S_new, H_q,  head_dim).transpose(1, 2).contiguous()
    k = k.reshape(B, S_new, H_kv, head_dim).transpose(1, 2).contiguous()
    v = v.reshape(B, S_new, H_kv, head_dim).transpose(1, 2).contiguous()
    t_qkv_proj += time.time() - t0

    t0 = time.time()
    q = ext.rope_partial(q, cos, sin, head_dim)
    k = ext.rope_partial(k, cos, sin, head_dim)
    t_rope += time.time() - t0

    sw = cfg.sliding_window if cfg.layer_is_sliding(i) else 0
    t0 = time.time()
    cache.append(i, k, v, positions_start=cache.cur_len)
    K_full, V_full = cache.slice(i, seq_end=cache.cur_len + 1)
    a = ext.flash_attention_gpt_oss_kv(q, K_full, V_full, lw.sinks, scale,
                                         past_len=cache.cur_len,
                                         sliding_window=sw, use_sinks=True)
    t_attn += time.time() - t0

    t0 = time.time()
    a = a.transpose(1, 2).contiguous().reshape(B, S_new, H_q * head_dim)
    a = a @ lw.o_proj_weight.T + lw.o_proj_bias
    x = residual + a
    t_o_proj += time.time() - t0

    residual = x
    t0 = time.time(); x_n = rmsnorm(x, lw.post_attention_layernorm_weight, cfg.rms_norm_eps); t_rmsnorm_2 += time.time()-t0

    t0 = time.time()
    router_logits = x_n @ lw.router_weight.T + lw.router_bias
    top_vals, top_idx = torch.topk(router_logits, cfg.num_experts_per_tok, dim=-1)
    routing_weights = torch.softmax(top_vals, dim=-1)
    t_router += time.time() - t0

    t0 = time.time()
    T = B * S_new
    x_flat  = x_n.reshape(T, D).contiguous()
    idx_flat = top_idx.reshape(T, cfg.num_experts_per_tok).contiguous()
    w_flat  = routing_weights.reshape(T, cfg.num_experts_per_tok).contiguous()
    E   = lw.gate_up_blocks.shape[0]
    Dff = lw.gate_up_blocks.shape[1] // 2
    mlp = ext.moe_mlp_gpt_oss_resident(
        x_flat, idx_flat, w_flat,
        resident.h_gu_blocks, resident.h_gu_scales, resident.h_gu_bias,
        resident.h_d_blocks,  resident.h_d_scales,  resident.h_d_bias,
        E, D, Dff,
    )
    mlp = mlp.reshape(B, S_new, D)
    x = residual + mlp
    t_moe += time.time() - t0

cache.advance(1)

# Final norm + LM head.
t0 = time.time()
x = rmsnorm(x, w.final_norm, cfg.rms_norm_eps)
logits = x @ w.lm_head.T
t_lm_head = time.time() - t0

t_step = time.time() - t_total

print(f"\n=== Decode step breakdown (single token, cache cur_len={cache.cur_len}) ===")
print(f"  Total step time:                {t_step*1000:8.1f} ms")
print(f"  Per-layer aggregate:")
print(f"    rmsnorm 1 (all layers):       {t_rmsnorm_1*1000:8.1f} ms")
print(f"    Q/K/V/O reshape (CPU):        {t_qkv_proj*1000:8.1f} ms")
print(f"    RoPE (Vulkan, 2x per layer):  {t_rope*1000:8.1f} ms")
print(f"    Attention (Vulkan):           {t_attn*1000:8.1f} ms")
print(f"    O proj + residual (CPU):      {t_o_proj*1000:8.1f} ms")
print(f"    rmsnorm 2 (all layers):       {t_rmsnorm_2*1000:8.1f} ms")
print(f"    Router (CPU):                 {t_router*1000:8.1f} ms")
print(f"    MoE MLP (Vulkan, resident):   {t_moe*1000:8.1f} ms")
print(f"  Final norm + LM head (CPU):     {t_lm_head*1000:8.1f} ms")
print(f"")
print(f"  Predicted next token: {int(logits[0, -1].argmax().item())} "
      f"({tok.decode([int(logits[0, -1].argmax().item())])!r})")
