"""Lazy full-model runtime for dense Qwen3 checkpoints."""
from __future__ import annotations

import gc
import pathlib
import time

import torch

from dense_kv_cache import DenseKVCache
from .config import load_qwen3_config
from .layer import qwen3_layer_forward
from .loader import ShardedSafetensors, _to_fp32, load_qwen3_layer
from .rope import compute_rope


class Qwen3Model:
    """Run Qwen3 while materializing only one FP32 layer at a time."""

    def __init__(self, ext, model_dir: str | pathlib.Path):
        self.ext = ext
        self.model_dir = pathlib.Path(model_dir).resolve()
        self.config = load_qwen3_config(self.model_dir)
        self.tensors = ShardedSafetensors(self.model_dir)
        self.embed_tokens = self.tensors.get("model.embed_tokens.weight")
        self.final_norm = _to_fp32(self.tensors.get("model.norm.weight"))
        self.layer_times: list[float] = []

    @classmethod
    def from_pretrained(cls, ext, model_dir: str | pathlib.Path) -> "Qwen3Model":
        return cls(ext, model_dir)

    def make_kv_cache(
        self, max_seqlen: int = 1024, batch: int = 1
    ) -> DenseKVCache:
        return DenseKVCache(self.config, batch=batch, max_seqlen=max_seqlen)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        only_last_logits: bool = False,
        collect_layer_times: bool = False,
        past_kv: DenseKVCache | None = None,
        past_len: int = 0,
    ) -> torch.Tensor:
        if input_ids.dtype != torch.long or input_ids.dim() != 2:
            raise ValueError("input_ids must be int64 [B,S]")
        batch, sequence = input_ids.shape
        if batch != 1:
            raise ValueError("Qwen3 milestone currently supports batch=1")
        hidden = self.embed_tokens[input_ids].float().contiguous()
        positions = torch.arange(
            past_len, past_len + sequence, dtype=torch.long
        )
        cos, sin = compute_rope(self.config, positions)
        self.layer_times.clear()

        for layer_idx in range(self.config.num_hidden_layers):
            started = time.perf_counter()
            weights = load_qwen3_layer(self.tensors, layer_idx)
            hidden = qwen3_layer_forward(
                self.ext,
                hidden,
                weights,
                self.config,
                cos,
                sin,
                layer_idx=layer_idx,
                past_kv=past_kv,
                past_len=past_len,
            )
            if collect_layer_times:
                self.layer_times.append(time.perf_counter() - started)
            del weights
            if layer_idx % 6 == 5:
                gc.collect()

        hidden = self.ext.rmsnorm(
            hidden, self.final_norm, self.config.rms_norm_eps
        )
        if only_last_logits:
            hidden = hidden[:, -1:, :]
        lm_head = self.embed_tokens.float()
        return hidden @ lm_head.T
