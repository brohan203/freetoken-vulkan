"""Preallocated FP32 K/V cache for architecture-neutral dense decoders."""
from __future__ import annotations

import torch

from model_contracts import DenseDecoderConfig


class DenseKVCache:
    def __init__(
        self,
        config: DenseDecoderConfig,
        batch: int = 1,
        max_seqlen: int = 1024,
        dtype: torch.dtype = torch.float32,
    ):
        if batch != 1:
            raise ValueError("dense KV cache currently supports batch=1")
        self.config = config
        self.batch = batch
        self.max_seqlen = max_seqlen
        self.dtype = dtype
        self.cur_len = 0
        shape = (
            batch,
            config.num_key_value_heads,
            max_seqlen,
            config.head_dim,
        )
        self._k = [
            torch.empty(shape, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]
        self._v = [
            torch.empty(shape, dtype=dtype)
            for _ in range(config.num_hidden_layers)
        ]

    def append(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        start: int,
    ) -> None:
        count = key.shape[2]
        end = start + count
        if end > self.max_seqlen:
            raise ValueError(f"KV cache overflow: {end} > {self.max_seqlen}")
        self._k[layer_idx][:, :, start:end, :] = key
        self._v[layer_idx][:, :, start:end, :] = value

    def slice(
        self, layer_idx: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._k[layer_idx][:, :, :end, :].contiguous(),
            self._v[layer_idx][:, :, :end, :].contiguous(),
        )

    def advance(self, count: int) -> None:
        if self.cur_len + count > self.max_seqlen:
            raise ValueError("KV cache overflow")
        self.cur_len += count

    def reset(self) -> None:
        self.cur_len = 0

    def memory_bytes(self) -> int:
        return (
            2
            * len(self._k)
            * self._k[0].numel()
            * self._k[0].element_size()
        )
