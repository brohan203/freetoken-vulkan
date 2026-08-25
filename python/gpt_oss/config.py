"""GptOssConfig - dataclass mirroring the fields we need from config.json."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import json
import pathlib


@dataclass
class GptOssConfig:
    # Architecture
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int      # H_q
    num_key_value_heads: int      # H_kv
    head_dim: int
    intermediate_size: int
    num_experts: int              # E
    num_experts_per_tok: int      # K
    experts_per_token: int = None # alias in some configs
    # Attention flavor per layer
    layer_types: List[str] = None
    sliding_window: int = 128
    # RoPE
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.25
    rope_scaling: dict = None
    max_position_embeddings: int = 131072
    original_max_position_embeddings: int = 4096
    # Activation
    hidden_act: str = "swiglu"    # gpt-oss's custom variant
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702
    # Norm
    rms_norm_eps: float = 1e-5
    # Vocab
    vocab_size: int = 201088
    # Misc
    tie_word_embeddings: bool = False

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "GptOssConfig":
        with open(path) as f:
            j = json.load(f)
        # Some fields have alternate names - normalize.
        num_experts = (j.get("num_experts") or j.get("num_local_experts")
                       or j.get("n_experts"))
        top_k = j.get("num_experts_per_tok") or j.get("experts_per_token") or j.get("top_k")
        return cls(
            hidden_size=j["hidden_size"],
            num_hidden_layers=j["num_hidden_layers"],
            num_attention_heads=j["num_attention_heads"],
            num_key_value_heads=j["num_key_value_heads"],
            head_dim=j.get("head_dim") or j["hidden_size"] // j["num_attention_heads"],
            intermediate_size=j["intermediate_size"],
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            layer_types=j.get("layer_types"),
            sliding_window=j.get("sliding_window", 128),
            rope_theta=j.get("rope_theta", 10000.0),
            partial_rotary_factor=j.get("partial_rotary_factor", 0.25),
            rope_scaling=j.get("rope_scaling"),
            max_position_embeddings=j.get("max_position_embeddings", 131072),
            original_max_position_embeddings=j.get("original_max_position_embeddings", 4096),
            hidden_act=j.get("hidden_act", "swiglu"),
            swiglu_limit=j.get("swiglu_limit", 7.0),
            swiglu_alpha=j.get("swiglu_alpha", 1.702),
            rms_norm_eps=j.get("rms_norm_eps", 1e-5),
            vocab_size=j["vocab_size"],
            tie_word_embeddings=j.get("tie_word_embeddings", False),
        )

    # --- Derived properties ---
    @property
    def rotary_dim(self) -> int:
        return int(self.partial_rotary_factor * self.head_dim)

    @property
    def h_q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def layer_is_sliding(self, layer_idx: int) -> bool:
        """True if this layer uses sliding window attention.
        For gpt-oss, this is one of {sliding_attention, full_attention} per
        layer, listed in the layer_types array in config.json."""
        if self.layer_types is None:
            return False
        return self.layer_types[layer_idx] == "sliding_attention"
