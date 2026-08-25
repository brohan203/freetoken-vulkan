"""demo_mini_transformer.py — end-to-end mini transformer block on Vulkan.

Chains our kernels through a realistic-shape transformer decoder block:

    x_in  = [S, D]                       (batch=1 for simplicity)
    x1 = rmsnorm(x_in, w_ln1)            [S, D]
    q  = x1 @ Wq                         [S, D]     ← matmul
    k  = x1 @ Wk                         [S, D]     ← matmul
    v  = x1 @ Wv                         [S, D]     ← matmul
    a  = flash_attention(q, k, v)        [S, D]     ← flash_attention
    o  = a @ Wo                          [S, D]     ← matmul
    x  = x_in + o                        [S, D]     (residual)
    x2 = rmsnorm(x, w_ln2)               [S, D]
    g  = x2 @ Wgate                      [S, Dff]   ← matmul  (MoE MLP: single expert)
    u  = x2 @ Wup                        [S, Dff]   ← matmul
    m  = swiglu(g, u)                    [S, Dff]   ← swiglu
    y  = m @ Wdown                       [S, D]     ← matmul
    out = x + y                          [S, D]

We compare against the same computation in pure PyTorch, expecting agreement
to ~1e-4 (accumulated across many ops).

Note: this is a "single expert" MLP — MoE routing (`moe_router` + weighted
expert MLPs) is tested in tests/test_moe_router.py but not composed here
because a real MoE MLP needs grouped GEMM, which is on the TODO list.
"""
from __future__ import annotations
import math
import time

import torch

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "python"))
import freetoken_vulkan as fv


def transformer_block_vulkan(x_in, w_ln1, Wq, Wk, Wv, Wo,
                              w_ln2, Wgate, Wup, Wdown):
    """One transformer block through Vulkan kernels."""
    x1 = fv.rmsnorm(x_in, w_ln1)
    q  = fv.matmul(x1, Wq)
    k  = fv.matmul(x1, Wk)
    v  = fv.matmul(x1, Wv)
    a  = fv.flash_attention(q, k, v)
    o  = fv.matmul(a, Wo)
    x  = x_in + o
    x2 = fv.rmsnorm(x, w_ln2)
    g  = fv.matmul(x2, Wgate)
    u  = fv.matmul(x2, Wup)
    m  = fv.swiglu(g, u)
    y  = fv.matmul(m, Wdown)
    return x + y


def transformer_block_torch(x_in, w_ln1, Wq, Wk, Wv, Wo,
                             w_ln2, Wgate, Wup, Wdown):
    """Same block in pure PyTorch."""
    F = torch.nn.functional
    x1 = F.rms_norm(x_in, (x_in.shape[-1],), w_ln1, eps=1e-6)
    q, k, v = x1 @ Wq, x1 @ Wk, x1 @ Wv
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.T) * scale
    weights = torch.softmax(scores, dim=-1)
    a = weights @ v
    o = a @ Wo
    x = x_in + o
    x2 = F.rms_norm(x, (x.shape[-1],), w_ln2, eps=1e-6)
    g, u = x2 @ Wgate, x2 @ Wup
    m = F.silu(g) * u
    y = m @ Wdown
    return x + y


def main() -> None:
    torch.manual_seed(0xDEC0DE)

    # Modest shapes so subprocess wrapper doesn't take forever.
    # This is roughly a tiny MoE block: D=128 hidden, Dff=256 intermediate.
    S = 128
    D = 128
    Dff = 256

    print(f"Mini transformer block: S={S}  D={D}  D_ff={Dff}")
    print()

    x_in  = torch.randn(S, D) * 0.1
    w_ln1 = torch.rand(D) * 0.5 + 0.75
    Wq    = torch.randn(D, D) * (1.0 / math.sqrt(D))
    Wk    = torch.randn(D, D) * (1.0 / math.sqrt(D))
    Wv    = torch.randn(D, D) * (1.0 / math.sqrt(D))
    Wo    = torch.randn(D, D) * (1.0 / math.sqrt(D))
    w_ln2 = torch.rand(D) * 0.5 + 0.75
    Wgate = torch.randn(D, Dff) * (1.0 / math.sqrt(D))
    Wup   = torch.randn(D, Dff) * (1.0 / math.sqrt(D))
    Wdown = torch.randn(Dff, D) * (1.0 / math.sqrt(Dff))

    print("running through Vulkan kernels...")
    t0 = time.time()
    out_vk = transformer_block_vulkan(
        x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2, Wgate, Wup, Wdown)
    t_vk = time.time() - t0
    print(f"  Vulkan wall time: {t_vk*1000:.1f} ms "
          "(dominated by subprocess IPC, not compute)")

    print("running through PyTorch reference...")
    t0 = time.time()
    out_ref = transformer_block_torch(
        x_in, w_ln1, Wq, Wk, Wv, Wo, w_ln2, Wgate, Wup, Wdown)
    t_ref = time.time() - t0
    print(f"  PyTorch wall time: {t_ref*1000:.2f} ms")

    diff = (out_vk - out_ref).abs()
    max_abs = diff.max().item()
    rel = diff / out_ref.abs().clamp_min(1e-6)
    max_rel = rel.max().item()

    tol_abs = 5e-4   # accumulated across ~10 kernels of FP32 math
    tol_rel = 1e-2
    ok = max_abs < tol_abs

    print()
    print(f"output shape: {tuple(out_vk.shape)}")
    print(f"max |Vulkan - Torch|   = {max_abs:.3e}    (tol {tol_abs:.1e})")
    print(f"max relative diff       = {max_rel:.3e}    (tol {tol_rel:.1e})")
    print()
    if ok:
        print("PASS — the whole transformer block ran on your 6800 XT")
        print("       and matched PyTorch within accumulated FP32 tolerance.")
    else:
        print("FAIL — investigate divergence.")


if __name__ == "__main__":
    main()
