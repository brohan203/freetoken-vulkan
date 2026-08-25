"""Standard full-dimension RoPE tables for Qwen3."""
from __future__ import annotations

import torch

from model_contracts import DenseDecoderConfig


def compute_rope(
    config: DenseDecoderConfig,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.to(torch.float32)
    dimensions = torch.arange(0, config.head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (
        config.rope_theta ** (dimensions / float(config.head_dim))
    )
    frequencies = torch.outer(positions, inv_freq)
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return torch.cos(embedding).contiguous(), torch.sin(embedding).contiguous()
