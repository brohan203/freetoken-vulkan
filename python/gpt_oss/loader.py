"""Loader — read gpt-oss-20b safetensors + build a Python-side weight dict.

Design: keep MXFP4 expert weights as-is (uint8 blocks + uint8 scales); dequant
BF16 attention/embed/norm/router weights to FP32 for our shaders that expect
FP32 activations and can't currently consume BF16 storage.

Memory footprint on load:
    MXFP4 experts (all layers)      ~10 GB      (kept as bytes)
    BF16→FP32 attention (24 layers) ~1 GB       (2× original)
    BF16→FP32 embed + lm_head       ~4.6 GB
    FP32 norms, router, sinks       small
    Total                           ~16 GB      of RAM

That's fine on 64 GB. Future optimization: keep BF16 as uint16 in shader
storage and dequant per-element (saves ~4 GB, adds shader complexity).
"""
from __future__ import annotations
import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from safetensors import safe_open

from .config import GptOssConfig


@dataclass
class LayerWeights:
    # Attention block
    input_layernorm_weight: torch.Tensor       # [D]
    q_proj_weight: torch.Tensor                # [H_q * head_dim, D]   fp32
    q_proj_bias:   torch.Tensor                # [H_q * head_dim]      fp32
    k_proj_weight: torch.Tensor                # [H_kv * head_dim, D]  fp32
    k_proj_bias:   torch.Tensor
    v_proj_weight: torch.Tensor
    v_proj_bias:   torch.Tensor
    o_proj_weight: torch.Tensor                # [D, H_q * head_dim]   fp32
    o_proj_bias:   torch.Tensor                # [D]                   fp32
    sinks:         torch.Tensor                # [H_q]                 fp32
    # MoE block
    post_attention_layernorm_weight: torch.Tensor
    router_weight: torch.Tensor                # [E, D]                fp32
    router_bias:   torch.Tensor                # [E]                   fp32
    # MoE experts (MXFP4)
    gate_up_blocks: torch.Tensor               # [E, 2*Dff, NB_D, 16]  uint8
    gate_up_scales: torch.Tensor               # [E, 2*Dff, NB_D]      uint8
    gate_up_bias:   torch.Tensor               # [E, 2*Dff]            fp32
    down_blocks:    torch.Tensor               # [E, D,     NB_Dff, 16] uint8
    down_scales:    torch.Tensor               # [E, D,     NB_Dff]     uint8
    down_bias:      torch.Tensor               # [E, D]                 fp32


@dataclass
class ModelWeights:
    config: GptOssConfig
    embed_tokens: torch.Tensor                 # [vocab_size, D]        fp32
    lm_head:      torch.Tensor                 # [vocab_size, D]        fp32
    final_norm:   torch.Tensor                 # [D]                    fp32
    layers: List[LayerWeights] = field(default_factory=list)


def _bf16_to_fp32(t: torch.Tensor) -> torch.Tensor:
    """Idempotent BF16→FP32 upgrade."""
    if t.dtype == torch.bfloat16:
        return t.float().contiguous()
    if t.dtype == torch.float32:
        return t.contiguous()
    raise TypeError(f"unexpected dtype {t.dtype}")


class Safetensors:
    """Small wrapper that opens the multi-file safetensors weight set once
    and lets us fetch tensors by name."""

    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = pathlib.Path(model_dir)
        with open(self.model_dir / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]
        self._handles: Dict[str, any] = {}

    def _handle(self, chunk_name: str):
        if chunk_name not in self._handles:
            self._handles[chunk_name] = safe_open(
                self.model_dir / chunk_name, framework="pt")
        return self._handles[chunk_name]

    def get(self, name: str) -> torch.Tensor:
        chunk = self.weight_map[name]
        return self._handle(chunk).get_tensor(name)


def load_layer(sf: Safetensors, layer_idx: int) -> LayerWeights:
    """Load and dequant one transformer layer's weights."""
    p = f"model.layers.{layer_idx}"

    return LayerWeights(
        input_layernorm_weight = _bf16_to_fp32(sf.get(f"{p}.input_layernorm.weight")),
        q_proj_weight = _bf16_to_fp32(sf.get(f"{p}.self_attn.q_proj.weight")),
        q_proj_bias   = _bf16_to_fp32(sf.get(f"{p}.self_attn.q_proj.bias")),
        k_proj_weight = _bf16_to_fp32(sf.get(f"{p}.self_attn.k_proj.weight")),
        k_proj_bias   = _bf16_to_fp32(sf.get(f"{p}.self_attn.k_proj.bias")),
        v_proj_weight = _bf16_to_fp32(sf.get(f"{p}.self_attn.v_proj.weight")),
        v_proj_bias   = _bf16_to_fp32(sf.get(f"{p}.self_attn.v_proj.bias")),
        o_proj_weight = _bf16_to_fp32(sf.get(f"{p}.self_attn.o_proj.weight")),
        o_proj_bias   = _bf16_to_fp32(sf.get(f"{p}.self_attn.o_proj.bias")),
        sinks         = _bf16_to_fp32(sf.get(f"{p}.self_attn.sinks")),
        post_attention_layernorm_weight = _bf16_to_fp32(sf.get(f"{p}.post_attention_layernorm.weight")),
        router_weight = _bf16_to_fp32(sf.get(f"{p}.mlp.router.weight")),
        router_bias   = _bf16_to_fp32(sf.get(f"{p}.mlp.router.bias")),
        # MoE MXFP4: keep as uint8
        gate_up_blocks = sf.get(f"{p}.mlp.experts.gate_up_proj_blocks").contiguous(),
        gate_up_scales = sf.get(f"{p}.mlp.experts.gate_up_proj_scales").contiguous(),
        gate_up_bias   = _bf16_to_fp32(sf.get(f"{p}.mlp.experts.gate_up_proj_bias")),
        down_blocks    = sf.get(f"{p}.mlp.experts.down_proj_blocks").contiguous(),
        down_scales    = sf.get(f"{p}.mlp.experts.down_proj_scales").contiguous(),
        down_bias      = _bf16_to_fp32(sf.get(f"{p}.mlp.experts.down_proj_bias")),
    )


def load_model(model_dir: str | pathlib.Path,
               layers: List[int] | None = None) -> ModelWeights:
    """Load config + selected layers (default: all).

    Passing `layers=[0]` skips the other 23 layers and lm_head for fast dev.
    """
    model_dir = pathlib.Path(model_dir)
    cfg = GptOssConfig.from_json(model_dir / "config.json")
    sf = Safetensors(model_dir)

    embed = _bf16_to_fp32(sf.get("model.embed_tokens.weight"))
    final_norm = _bf16_to_fp32(sf.get("model.norm.weight"))
    try:
        lm_head = _bf16_to_fp32(sf.get("lm_head.weight"))
    except KeyError:
        # Some models tie lm_head to embed_tokens.
        lm_head = embed

    if layers is None:
        layers = list(range(cfg.num_hidden_layers))

    m = ModelWeights(config=cfg, embed_tokens=embed, lm_head=lm_head,
                     final_norm=final_norm)
    for li in layers:
        m.layers.append(load_layer(sf, li))
    return m
