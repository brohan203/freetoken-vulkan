"""kv_cache.py — pre-allocated K/V cache per transformer layer.

Layout per layer:
    K [B, H_kv, max_seqlen, head_dim] float32
    V [B, H_kv, max_seqlen, head_dim] float32

The cache tracks a single scalar `cur_len` — the number of positions
filled so far. During prefill this jumps by S_prompt; during decode it
increments by 1 per step. All layers stay in lockstep because gpt-oss is
strictly autoregressive.

Attention only reads positions [0, cur_len), so it's fine that the
underlying tensors are pre-allocated at max_seqlen — no writes there
means no correctness issue.

Memory footprint (fp32) for gpt-oss-20b:
    24 layers × 2 tensors × 8 KV heads × 64 head_dim × 4 B/elem
    × max_seqlen  =>  ~48 KB × max_seqlen  =>  6 MB @ max_seqlen=128,
    50 MB @ max_seqlen=1024. Cheap.
"""
from __future__ import annotations
import torch
from .config import GptOssConfig


class KVCache:
    def __init__(self, cfg: GptOssConfig, batch: int = 1,
                 max_seqlen: int = 1024, dtype=torch.float32):
        self.cfg = cfg
        self.batch = batch
        self.max_seqlen = max_seqlen
        self.dtype = dtype
        self.cur_len = 0
        shape = (batch, cfg.num_key_value_heads, max_seqlen, cfg.head_dim)
        self._k = [torch.zeros(shape, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self._v = [torch.zeros(shape, dtype=dtype) for _ in range(cfg.num_hidden_layers)]

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
               positions_start: int) -> None:
        n = k_new.size(2)
        end = positions_start + n
        assert end <= self.max_seqlen, \
            f"KV cache overflow: {end} > {self.max_seqlen}. Increase max_seqlen."
        self._k[layer_idx][:, :, positions_start:end, :] = k_new
        self._v[layer_idx][:, :, positions_start:end, :] = v_new

    def slice(self, layer_idx: int, seq_end: int) -> tuple[torch.Tensor, torch.Tensor]:
        K = self._k[layer_idx][:, :, :seq_end, :].contiguous()
        V = self._v[layer_idx][:, :, :seq_end, :].contiguous()
        return K, V

    def advance(self, n: int) -> None:
        """Bump cur_len after all layers have written n new positions."""
        assert self.cur_len + n <= self.max_seqlen, "cache overflow"
        self.cur_len += n

    def reset(self) -> None:
        self.cur_len = 0

    def memory_bytes(self) -> int:
        # 2 (K + V) tensors × num_layers × per-tensor bytes
        return 2 * len(self._k) * self._k[0].numel() * self._k[0].element_size()
