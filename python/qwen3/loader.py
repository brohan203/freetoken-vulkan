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


def _dequantize_awq4(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    k_size = qweight.shape[0]
    n_size = scales.shape[1]
    packed_weights = qweight.to(torch.int64)
    packed_zeros = qzeros.to(torch.int64)
    weights = torch.empty((k_size, n_size), dtype=torch.float32)
    zeros = torch.empty((k_size, n_size), dtype=torch.float32)
    for packed in range(qweight.shape[1]):
        for slot, logical in enumerate(order):
            column = packed * 8 + logical
            weights[:, column] = (
                (packed_weights[:, packed] >> (4 * slot)) & 15
            ).float()
            zero_column = (
                (packed_zeros[:, packed] >> (4 * slot)) & 15
            ).float()
            zeros[:, column] = zero_column.repeat_interleave(group_size)[:k_size]
    expanded_scales = scales.float().repeat_interleave(group_size, 0)[:k_size]
    return ((weights - zeros) * expanded_scales).T.contiguous()


def _dequantize_fp8(
    weight: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    if weight.dtype not in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
        raise TypeError(f"unexpected FP8 dtype {weight.dtype}")
    if weight.dim() != 2 or scales.dim() != 2:
        raise ValueError("FP8 weight and scale tensors must be matrices")
    block_rows = (weight.shape[0] + scales.shape[0] - 1) // scales.shape[0]
    block_cols = (weight.shape[1] + scales.shape[1] - 1) // scales.shape[1]
    expanded = scales.float().repeat_interleave(block_rows, 0)
    expanded = expanded.repeat_interleave(block_cols, 1)
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    return (weight.float() * expanded).contiguous()


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

    def contains(self, name: str) -> bool:
        return name in self.weight_map

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
    def get(suffix: str) -> torch.Tensor:
        return _to_fp32(tensors.get(f"{prefix}.{suffix}"))

    def get_matrix(suffix: str) -> torch.Tensor:
        name = f"{prefix}.{suffix}"
        if tensors.contains(name):
            tensor = tensors.get(name)
            if tensor.dtype in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
                scales = tensors.get(name + "_scale_inv")
                return _dequantize_fp8(tensor, scales)
            return _to_fp32(tensor)
        stem = name.removesuffix(".weight")
        return _dequantize_awq4(
            tensors.get(stem + ".qweight"),
            tensors.get(stem + ".qzeros"),
            tensors.get(stem + ".scales"),
        )

    return DenseDecoderLayerWeights(
        input_norm=get("input_layernorm.weight"),
        post_attention_norm=get("post_attention_layernorm.weight"),
        q_weight=get_matrix("self_attn.q_proj.weight"),
        k_weight=get_matrix("self_attn.k_proj.weight"),
        v_weight=get_matrix("self_attn.v_proj.weight"),
        o_weight=get_matrix("self_attn.o_proj.weight"),
        q_norm=get("self_attn.q_norm.weight"),
        k_norm=get("self_attn.k_norm.weight"),
        gate_weight=get_matrix("mlp.gate_proj.weight"),
        up_weight=get_matrix("mlp.up_proj.weight"),
        down_weight=get_matrix("mlp.down_proj.weight"),
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
