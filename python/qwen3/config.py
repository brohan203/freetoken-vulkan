"""Qwen3 dense-decoder configuration."""
from __future__ import annotations

import json
import pathlib

from model_contracts import DenseDecoderConfig


def load_qwen3_config(path: str | pathlib.Path) -> DenseDecoderConfig:
    source = pathlib.Path(path)
    if source.is_dir():
        source = source / "config.json"
    data = json.loads(source.read_text())
    if data.get("model_type") != "qwen3":
        raise ValueError(f"expected qwen3 model_type, got {data.get('model_type')!r}")
    config = DenseDecoderConfig(
        model_type="qwen3",
        hidden_size=int(data["hidden_size"]),
        intermediate_size=int(data["intermediate_size"]),
        num_hidden_layers=int(data["num_hidden_layers"]),
        num_attention_heads=int(data["num_attention_heads"]),
        num_key_value_heads=int(data["num_key_value_heads"]),
        head_dim=int(data["head_dim"]),
        vocab_size=int(data["vocab_size"]),
        rms_norm_eps=float(data.get("rms_norm_eps", 1e-6)),
        rope_theta=float(data.get("rope_theta", 1_000_000.0)),
        hidden_act=str(data.get("hidden_act", "silu")),
        tie_word_embeddings=bool(data.get("tie_word_embeddings", True)),
    )
    config.validate()
    if data.get("attention_bias", False):
        raise ValueError("Qwen3 attention bias is not supported by this loader")
    return config
