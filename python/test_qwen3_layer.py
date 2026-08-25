"""Compare one real Qwen3-4B layer against an independent PyTorch reference."""
from __future__ import annotations

import math
import os
import pathlib
import sys

os.environ.setdefault("VULKAN_SDK", r"C:\VulkanSDK\1.4.357.0")
os.environ["PATH"] = (
    os.path.join(os.environ["VULKAN_SDK"], "Bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from qwen3 import (
    ShardedSafetensors,
    compute_rope,
    load_qwen3_config,
    load_qwen3_layer,
    qwen3_layer_forward,
)

MODEL = pathlib.Path(r"C:\Users\rohanborkar\Downloads\Qwen3-4B")


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference_layer(x, weights, config, cos, sin):
    batch, sequence, _ = x.shape
    normalized = rmsnorm(x, weights.input_norm, config.rms_norm_eps)
    q = (normalized @ weights.q_weight.T).reshape(
        batch, sequence, config.num_attention_heads, config.head_dim
    )
    k = (normalized @ weights.k_weight.T).reshape(
        batch, sequence, config.num_key_value_heads, config.head_dim
    )
    v = (normalized @ weights.v_weight.T).reshape(
        batch, sequence, config.num_key_value_heads, config.head_dim
    )
    q = rmsnorm(q, weights.q_norm, config.rms_norm_eps)
    k = rmsnorm(k, weights.k_norm, config.rms_norm_eps)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    cos4 = cos[None, None, :, :]
    sin4 = sin[None, None, :, :]
    q = q * cos4 + rotate_half(q) * sin4
    k = k * cos4 + rotate_half(k) * sin4
    repeats = config.num_attention_heads // config.num_key_value_heads
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(config.head_dim)
    causal = torch.triu(
        torch.full((sequence, sequence), float("-inf")), diagonal=1
    )
    probabilities = torch.softmax(scores + causal, dim=-1)
    attention = probabilities @ v
    attention = attention.permute(0, 2, 1, 3).reshape(
        batch, sequence, config.query_size
    )
    residual = x + attention @ weights.o_weight.T
    normalized = rmsnorm(
        residual, weights.post_attention_norm, config.rms_norm_eps
    )
    gate = normalized @ weights.gate_weight.T
    up = normalized @ weights.up_weight.T
    return residual + (F.silu(gate) * up) @ weights.down_weight.T


def main() -> None:
    torch.set_num_threads(12)
    sdk = pathlib.Path(os.environ["VULKAN_SDK"])
    ext = load(
        name="freetoken_vulkan_ext",
        sources=[str(HERE / "ext_module.cpp")],
        extra_include_paths=[str(sdk / "Include"), str(REPO / "include")],
        extra_ldflags=[f"/LIBPATH:{sdk / 'Lib'}", "vulkan-1.lib"],
        extra_cflags=["/O2", "/D_CRT_SECURE_NO_WARNINGS"],
        verbose=False,
    )
    config = load_qwen3_config(MODEL)
    tensors = ShardedSafetensors(MODEL)
    weights = load_qwen3_layer(tensors, 0)
    torch.manual_seed(101)
    x = torch.randn(1, 3, config.hidden_size, dtype=torch.float32) * 0.02
    cos, sin = compute_rope(config, torch.arange(3))
    expected = reference_layer(x, weights, config, cos, sin)
    actual = qwen3_layer_forward(ext, x, weights, config, cos, sin)
    difference = (actual - expected).abs()
    print(
        "shape", tuple(actual.shape),
        "max", difference.max().item(),
        "mean", difference.mean().item(),
        "finite", torch.isfinite(actual).all().item(),
    )
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected, rtol=1e-4, atol=5e-4)
    print("QWEN3_LAYER_OK")


if __name__ == "__main__":
    main()
