"""Safetensors loader for dense Qwen3 checkpoints."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from safetensors import safe_open

from model_contracts import DenseDecoderConfig, DenseDecoderLayerWeights
from .config import load_qwen3_config


def _to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise TypeError(f"unexpected dense weight dtype {tensor.dtype}")
    return tensor.float().contiguous()


class ShardedSafetensors:
    def __init__(self, model_dir: str | pathlib.Path):
        self.model_dir = pathlib.Path(model_dir).resolve()
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: Dict[str, str] = index["weight_map"]
        self._handles: Dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._handles:
            self._handles[shard] = safe_open(
                self.model_dir / shard, framework="pt", device="cpu"
            )
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        return self._handle(shard).get_tensor(name)


@dataclass
class Qwen3ModelWeights:
    config: DenseDecoderConfig
    embed_tokens: torch.Tensor
    final_norm: torch.Tensor
    layers: List[DenseDecoderLayerWeights] = field(default_factory=list)

    @property
    def lm_head(self) -> torch.Tensor:
        return self.embed_tokens


def load_qwen3_layer(
    tensors: ShardedSafetensors, layer_idx: int
) -> DenseDecoderLayerWeights:
    prefix = f"model.layers.{layer_idx}"
    get = lambda suffix: _to_fp32(tensors.get(f"{prefix}.{suffix}"))
    return DenseDecoderLayerWeights(
        input_norm=get("input_layernorm.weight"),
        post_attention_norm=get("post_attention_layernorm.weight"),
        q_weight=get("self_attn.q_proj.weight"),
        k_weight=get("self_attn.k_proj.weight"),
        v_weight=get("self_attn.v_proj.weight"),
        o_weight=get("self_attn.o_proj.weight"),
        q_norm=get("self_attn.q_norm.weight"),
        k_norm=get("self_attn.k_norm.weight"),
        gate_weight=get("mlp.gate_proj.weight"),
        up_weight=get("mlp.up_proj.weight"),
        down_weight=get("mlp.down_proj.weight"),
    )


def load_qwen3_model(
    model_dir: str | pathlib.Path,
    layers: List[int] | None = None,
) -> Qwen3ModelWeights:
    model_dir = pathlib.Path(model_dir)
    config = load_qwen3_config(model_dir)
    tensors = ShardedSafetensors(model_dir)
    embed = _to_fp32(tensors.get("model.embed_tokens.weight"))
    final_norm = _to_fp32(tensors.get("model.norm.weight"))
    if layers is None:
        layers = list(range(config.num_hidden_layers))
    return Qwen3ModelWeights(
        config=config,
        embed_tokens=embed,
        final_norm=final_norm,
        layers=[load_qwen3_layer(tensors, index) for index in layers],
    )
