"""demo_moe_transformer.py — full MoE transformer block on Vulkan.

Same as demo_mini_transformer, but the MLP is a real routed MoE (top-K
experts + weighted sum), not a single MLP. This chains:

    RMSNorm → 4 matmuls (Q,K,V, output proj) → FlashAttention → residual
    RMSNorm → MoE router → fused MoE MLP (with SwiGLU per expert) → residual

12 kernel dispatches, one of which is the real MoE block that mirrors the
Phi-3.5-MoE / Mixtral / Qwen3 architecture family.
"""
from __future__ import annotations
import math
import time

import torch
import torch.nn.functional as F

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "python"))
import freetoken_vulkan as fv


def moe_transformer_block_vulkan(
    x_in, w_ln1, Wq, Wk, Wv, Wo,
    w_ln2, W_router, W_gate, W_up, W_down, K,
):
    x1 = fv.rmsnorm(x_in, w_ln1)
    q  = fv.matmul(x1, Wq)
    k  = fv.matmul(x1, Wk)
    v  = fv.matmul(x1, Wv)
    a  = fv.flash_attention(q, k, v)
    o  = fv.matmul(a, Wo)
    x  = x_in + o
    x2 = fv.rmsnorm(x, w_ln2)
    logits = fv.matmul(x2, W_router)                # [T, E]
    indices, weights = fv.moe_router(logits, K)     # [T, K], [T, K]
    m  = fv.moe_mlp(x2, indices, weights, W_gate, W_up, W_down)
    return x + m


def moe_transformer_block_torch(
    x_in, w_ln1, Wq, Wk, Wv, Wo,
    w_ln2, W_router, W_gate, W_up, W_down, K,
):
    D = x_in.shape[-1]
    x1 = F.rms_norm(x_in, (D,), w_ln1, eps=1e-6)
    q, k, v = x1 @ Wq, x1 @ Wk, x1 @ Wv
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.T) * scale
    weights_attn = torch.softmax(scores, dim=-1)
    a = weights_attn @ v
    o = a @ Wo
    x = x_in + o
    x2 = F.rms_norm(x, (D,), w_ln2, eps=1e-6)

    # Router
    logits = x2 @ W_router
    probs = torch.softmax(logits, dim=-1)
    weights_r, indices = torch.topk(probs, K, dim=-1)
    weights_r = weights_r / weights_r.sum(dim=-1, keepdim=True)

    # MoE MLP
    T = x2.shape[0]
    m = torch.zeros_like(x2)
    for t in range(T):
        for kk in range(K):
            e = indices[t, kk].item()
            w = weights_r[t, kk].item()
            g = W_gate[e] @ x2[t]
            u = W_up[e]   @ x2[t]
            mid = F.silu(g) * u
            oo = W_down[e] @ mid
            m[t] += w * oo

    return x + m


def main() -> None:
    torch.manual_seed(0xF00DBABE)

    # Small enough shapes to fit our fused MoE MLP kernel's MAX_D=256, MAX_DFF=512
    # and small enough for the reference torch loop to complete quickly.
    S   = 32       # tokens per block
    D   = 128
    Dff = 256
    E   = 8
    K   = 2

    print(f"MoE transformer block:")
    print(f"  S={S}  D={D}  D_ff={Dff}  experts={E}  top-K={K}")
    print()

    x_in  = torch.randn(S, D) * 0.1
    w_ln1 = torch.rand(D) * 0.5 + 0.75
    w_ln2 = torch.rand(D) * 0.5 + 0.75
    Wq = torch.randn(D, D) * (1.0 / math.sqrt(D))
    Wk = torch.randn(D, D) * (1.0 / math.sqrt(D))
    Wv = torch.randn(D, D) * (1.0 / math.sqrt(D))
    Wo = torch.randn(D, D) * (1.0 / math.sqrt(D))
    W_router = torch.randn(D, E) * (1.0 / math.sqrt(D))
    W_gate = torch.randn(E, Dff, D) * (1.0 / math.sqrt(D))
    W_up   = torch.randn(E, Dff, D) * (1.0 / math.sqrt(D))
    W_down = torch.randn(E, D, Dff) * (1.0 / math.sqrt(Dff))

    print("running through Vulkan kernels...")
    t0 = time.time()
    out_vk = moe_transformer_block_vulkan(
        x_in, w_ln1, Wq, Wk, Wv, Wo,
        w_ln2, W_router, W_gate, W_up, W_down, K)
    t_vk = time.time() - t0
    print(f"  Vulkan wall time: {t_vk*1000:.1f} ms "
          "(dominated by subprocess IPC, not compute)")

    print("running through PyTorch reference...")
    t0 = time.time()
    out_ref = moe_transformer_block_torch(
        x_in, w_ln1, Wq, Wk, Wv, Wo,
        w_ln2, W_router, W_gate, W_up, W_down, K)
    t_ref = time.time() - t0
    print(f"  PyTorch wall time: {t_ref*1000:.2f} ms")

    diff = (out_vk - out_ref).abs()
    max_abs = diff.max().item()
    tol_abs = 5e-4
    ok = max_abs < tol_abs

    print()
    print(f"output shape: {tuple(out_vk.shape)}")
    print(f"max |Vulkan - Torch|   = {max_abs:.3e}    (tol {tol_abs:.1e})")
    print()
    if ok:
        print("PASS — full MoE transformer block ran on your 6800 XT")
        print("       and matched PyTorch within accumulated FP32 tolerance.")
    else:
        print("FAIL — investigate divergence.")


if __name__ == "__main__":
    main()
