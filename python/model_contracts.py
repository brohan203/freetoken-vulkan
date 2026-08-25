"""Architecture-neutral contracts for dense decoder-only transformer models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DenseDecoderConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    tie_word_embeddings: bool

    @property
    def query_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def validate(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim <= 0 or self.head_dim > 128:
            raise ValueError("Vulkan attention requires 0 < head_dim <= 128")
        if self.head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        if self.hidden_act != "silu":
            raise ValueError("dense decoder currently requires SiLU/SwiGLU")


@dataclass
class DenseDecoderLayerWeights:
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    v_weight: torch.Tensor
    o_weight: torch.Tensor
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    q_norm: Optional[torch.Tensor] = None
    k_norm: Optional[torch.Tensor] = None
    q_bias: Optional[torch.Tensor] = None
    k_bias: Optional[torch.Tensor] = None
    v_bias: Optional[torch.Tensor] = None
    o_bias: Optional[torch.Tensor] = None
