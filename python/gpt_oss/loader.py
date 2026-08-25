"""Safetensors loader for gpt-oss checkpoints.

Expert-loading modes:
* eager: load every MXFP4 expert tensor into CPU memory (gpt-oss-20b).
* streaming: materialize only router-selected experts (gpt-oss-120b).
"""
from __future__ import annotations

import json
import pathlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

from .config import GptOssConfig


@dataclass
class LayerWeights:
    input_layernorm_weight: torch.Tensor
    q_proj_weight: torch.Tensor
    q_proj_bias: torch.Tensor
    k_proj_weight: torch.Tensor
    k_proj_bias: torch.Tensor
    v_proj_weight: torch.Tensor
    v_proj_bias: torch.Tensor
    o_proj_weight: torch.Tensor
    o_proj_bias: torch.Tensor
    sinks: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor
    router_weight: torch.Tensor
    router_bias: torch.Tensor
    gate_up_blocks: Optional[torch.Tensor]
    gate_up_scales: Optional[torch.Tensor]
    gate_up_bias: Optional[torch.Tensor]
    down_blocks: Optional[torch.Tensor]
    down_scales: Optional[torch.Tensor]
    down_bias: Optional[torch.Tensor]


@dataclass
class ExpertTensors:
    """One or more experts laid out as a compact local expert table."""

    gate_up_blocks: torch.Tensor
    gate_up_scales: torch.Tensor
    gate_up_bias: torch.Tensor
    down_blocks: torch.Tensor
    down_scales: torch.Tensor
    down_bias: torch.Tensor

    @property
    def num_experts(self) -> int:
        return self.gate_up_blocks.shape[0]


@dataclass
class ModelWeights:
    config: GptOssConfig
    embed_tokens: torch.Tensor
    lm_head: torch.Tensor
    final_norm: torch.Tensor
    layers: List[LayerWeights] = field(default_factory=list)
    expert_store: Optional["ExpertStore"] = None


def _bf16_to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.bfloat16:
        return tensor.float().contiguous()
    if tensor.dtype == torch.float32:
        return tensor.contiguous()
    raise TypeError(f"unexpected dtype {tensor.dtype}")


class Safetensors:
    """Open a sharded safetensors checkpoint and fetch tensors by name."""

    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = pathlib.Path(model_dir).resolve()
        with open(self.model_dir / "model.safetensors.index.json") as stream:
            self.weight_map = json.load(stream)["weight_map"]
        self._handles: Dict[str, object] = {}

    def _handle(self, chunk_name: str):
        if chunk_name not in self._handles:
            self._handles[chunk_name] = safe_open(
                self.model_dir / chunk_name,
                framework="pt",
                device="cpu",
            )
        return self._handles[chunk_name]

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)


class ExpertStore:
    """File-backed MXFP4 experts with a bounded per-layer CPU LRU cache."""

    def __init__(self, safetensors: Safetensors, cache_size_per_layer: int = 16):
        self.sf = safetensors
        self.cache_size_per_layer = cache_size_per_layer
        self._cache: Dict[int, OrderedDict[int, ExpertTensors]] = {}
        self.selection_history: Dict[int, List[tuple[int, ...]]] = {}
        self.materialized_bytes = 0
        self.materialize_seconds = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _names(layer_idx: int) -> tuple[str, ...]:
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        return (
            f"{prefix}.gate_up_proj_blocks",
            f"{prefix}.gate_up_proj_scales",
            f"{prefix}.gate_up_proj_bias",
            f"{prefix}.down_proj_blocks",
            f"{prefix}.down_proj_scales",
            f"{prefix}.down_proj_bias",
        )

    def _selected(self, name: str, ids: list[int]) -> torch.Tensor:
        chunk = self.sf.weight_map[name]
        return self.sf._handle(chunk).get_slice(name)[ids].contiguous()

    def _load_many(
        self, layer_idx: int, expert_ids: list[int]
    ) -> Dict[int, ExpertTensors]:
        """Load all misses using six batched safetensors slices."""
        if not expert_ids:
            return {}
        gu_blocks, gu_scales, gu_bias, down_blocks, down_scales, down_bias = (
            self._selected(name, expert_ids) for name in self._names(layer_idx)
        )
        gu_bias = _bf16_to_fp32(gu_bias)
        down_bias = _bf16_to_fp32(down_bias)
        self.materialized_bytes += sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                gu_blocks,
                gu_scales,
                gu_bias,
                down_blocks,
                down_scales,
                down_bias,
            )
        )
        loaded: Dict[int, ExpertTensors] = {}
        for position, expert_id in enumerate(expert_ids):
            row = slice(position, position + 1)
            loaded[expert_id] = ExpertTensors(
                gate_up_blocks=gu_blocks[row],
                gate_up_scales=gu_scales[row],
                gate_up_bias=gu_bias[row],
                down_blocks=down_blocks[row],
                down_scales=down_scales[row],
                down_bias=down_bias[row],
            )
        return loaded

    def select(
        self, layer_idx: int, global_indices: torch.Tensor
    ) -> tuple[list[int], torch.Tensor, Dict[int, ExpertTensors]]:
        """Resolve selected IDs and return stable tensors for this invocation."""
        unique, inverse = torch.unique(
            global_indices.to(torch.int64),
            sorted=True,
            return_inverse=True,
        )
        ids = unique.tolist()
        self.selection_history.setdefault(layer_idx, []).append(tuple(ids))
        cache = self._cache.setdefault(layer_idx, OrderedDict())
        missing = [expert_id for expert_id in ids if expert_id not in cache]
        self.cache_hits += len(ids) - len(missing)
        self.cache_misses += len(missing)
        loaded = self._load_many(layer_idx, missing)

        # Preserve an invocation-local map before LRU eviction. The GPU cache
        # may need a newly loaded expert even if a small CPU cache evicts it.
        available = {
            expert_id: cache[expert_id]
            for expert_id in ids
            if expert_id in cache
        }
        available.update(loaded)

        if self.cache_size_per_layer > 0:
            for expert_id in missing:
                cache[expert_id] = loaded[expert_id]
            for expert_id in ids:
                if expert_id in cache:
                    cache.move_to_end(expert_id)
            while len(cache) > self.cache_size_per_layer:
                cache.popitem(last=False)

        local_indices = inverse.reshape_as(global_indices).contiguous()
        return ids, local_indices, available

    def load_ids(
        self, layer_idx: int, expert_ids: list[int]
    ) -> Dict[int, ExpertTensors]:
        """Materialize explicit GPU-cache misses directly from safetensors."""
        if not expert_ids:
            return {}
        t0 = time.perf_counter()
        loaded = self._load_many(layer_idx, expert_ids)
        self.materialize_seconds += time.perf_counter() - t0
        return loaded

    def record_selection(self, layer_idx: int, ids: list[int]) -> None:
        self.selection_history.setdefault(layer_idx, []).append(tuple(ids))

    def materialize_selected(
        self, layer_idx: int, global_indices: torch.Tensor
    ) -> tuple[ExpertTensors, torch.Tensor]:
        t0 = time.perf_counter()
        ids, local_indices, available = self.select(layer_idx, global_indices)
        experts = [available[expert_id] for expert_id in ids]
        compact = ExpertTensors(
            gate_up_blocks=torch.cat(
                [expert.gate_up_blocks for expert in experts], dim=0
            ),
            gate_up_scales=torch.cat(
                [expert.gate_up_scales for expert in experts], dim=0
            ),
            gate_up_bias=torch.cat(
                [expert.gate_up_bias for expert in experts], dim=0
            ),
            down_blocks=torch.cat(
                [expert.down_blocks for expert in experts], dim=0
            ),
            down_scales=torch.cat(
                [expert.down_scales for expert in experts], dim=0
            ),
            down_bias=torch.cat(
                [expert.down_bias for expert in experts], dim=0
            ),
        )
        self.materialize_seconds += time.perf_counter() - t0
        return compact, local_indices


def load_layer(
    safetensors: Safetensors,
    layer_idx: int,
    *,
    load_experts: bool = True,
) -> LayerWeights:
    """Load one transformer layer."""
    prefix = f"model.layers.{layer_idx}"
    if load_experts:
        gate_up_blocks = safetensors.get(
            f"{prefix}.mlp.experts.gate_up_proj_blocks"
        ).contiguous()
        gate_up_scales = safetensors.get(
            f"{prefix}.mlp.experts.gate_up_proj_scales"
        ).contiguous()
        gate_up_bias = _bf16_to_fp32(
            safetensors.get(f"{prefix}.mlp.experts.gate_up_proj_bias")
        )
        down_blocks = safetensors.get(
            f"{prefix}.mlp.experts.down_proj_blocks"
        ).contiguous()
        down_scales = safetensors.get(
            f"{prefix}.mlp.experts.down_proj_scales"
        ).contiguous()
        down_bias = _bf16_to_fp32(
            safetensors.get(f"{prefix}.mlp.experts.down_proj_bias")
        )
    else:
        gate_up_blocks = gate_up_scales = gate_up_bias = None
        down_blocks = down_scales = down_bias = None

    return LayerWeights(
        input_layernorm_weight=_bf16_to_fp32(
            safetensors.get(f"{prefix}.input_layernorm.weight")
        ),
        q_proj_weight=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.q_proj.weight")
        ),
        q_proj_bias=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.q_proj.bias")
        ),
        k_proj_weight=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.k_proj.weight")
        ),
        k_proj_bias=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.k_proj.bias")
        ),
        v_proj_weight=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.v_proj.weight")
        ),
        v_proj_bias=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.v_proj.bias")
        ),
        o_proj_weight=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.o_proj.weight")
        ),
        o_proj_bias=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.o_proj.bias")
        ),
        sinks=_bf16_to_fp32(
            safetensors.get(f"{prefix}.self_attn.sinks")
        ),
        post_attention_layernorm_weight=_bf16_to_fp32(
            safetensors.get(f"{prefix}.post_attention_layernorm.weight")
        ),
        router_weight=_bf16_to_fp32(
            safetensors.get(f"{prefix}.mlp.router.weight")
        ),
        router_bias=_bf16_to_fp32(
            safetensors.get(f"{prefix}.mlp.router.bias")
        ),
        gate_up_blocks=gate_up_blocks,
        gate_up_scales=gate_up_scales,
        gate_up_bias=gate_up_bias,
        down_blocks=down_blocks,
        down_scales=down_scales,
        down_bias=down_bias,
    )


def load_model(
    model_dir: str | pathlib.Path,
    layers: List[int] | None = None,
    *,
    stream_experts: bool = False,
    expert_cache_size: int = 16,
) -> ModelWeights:
    """Load config, shared weights, and selected transformer layers."""
    model_dir = pathlib.Path(model_dir)
    cfg = GptOssConfig.from_json(model_dir / "config.json")
    safetensors = Safetensors(model_dir)
    embed = _bf16_to_fp32(safetensors.get("model.embed_tokens.weight"))
    final_norm = _bf16_to_fp32(safetensors.get("model.norm.weight"))
    try:
        lm_head = _bf16_to_fp32(safetensors.get("lm_head.weight"))
    except KeyError:
        lm_head = embed
    if layers is None:
        layers = list(range(cfg.num_hidden_layers))
    model = ModelWeights(
        config=cfg,
        embed_tokens=embed,
        lm_head=lm_head,
        final_norm=final_norm,
        expert_store=(
            ExpertStore(safetensors, expert_cache_size)
            if stream_experts
            else None
        ),
    )
    for layer_idx in layers:
        model.layers.append(
            load_layer(
                safetensors,
                layer_idx,
                load_experts=not stream_experts,
            )
        )
    return model
