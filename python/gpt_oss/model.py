"""model.py — full gpt-oss forward pass with KV cache, resident MoE,
and resident LM head (all optional).
"""
from __future__ import annotations
import pathlib, time

import torch

from .config import GptOssConfig
from .loader import ModelWeights, load_model
from .layer import gpt_oss_layer_forward, rmsnorm
from .rope import compute_cos_sin_for_positions
from .kv_cache import KVCache
from .resident import ResidentMoEWeights


class GptOssModel:
    def __init__(self, ext, weights: ModelWeights):
        self.ext = ext
        self.weights = weights
        self.cfg: GptOssConfig = weights.config
        self._hf_cfg = None
        self._model_dir = None
        self.resident_moe: ResidentMoEWeights | None = None
        self.h_lm_head: int | None = None
        self._lm_head_shape: tuple | None = None

    @classmethod
    def from_pretrained(cls, ext, model_dir: str | pathlib.Path,
                        layers: list[int] | None = None) -> "GptOssModel":
        model_dir = pathlib.Path(model_dir)
        print(f"[GptOssModel] Loading from {model_dir}")
        t0 = time.time()
        w = load_model(model_dir, layers=layers)
        print(f"[GptOssModel] Loaded {len(w.layers)}/{w.config.num_hidden_layers} "
              f"layers in {time.time()-t0:.1f}s")
        m = cls(ext, w)
        m._model_dir = model_dir
        return m

    # ---------------- Resident VRAM setup ----------------
    def pin_moe_to_vram(self, verbose: bool = True) -> None:
        if self.resident_moe is None:
            self.resident_moe = ResidentMoEWeights(self.ext, self.weights, verbose=verbose)

    def pin_lm_head_to_vram(self) -> None:
        """Upload lm_head [vocab, D] to VRAM as an fp32 buffer (2.32 GB for
        gpt-oss-20b).  Moves the biggest CPU matmul off the CPU."""
        if self.h_lm_head is not None:
            return
        print(f"[GptOssModel] Pinning lm_head to VRAM "
              f"({self.weights.lm_head.numel() * 4 / 1024**3:.2f} GB)...")
        t0 = time.time()
        self.h_lm_head = self.ext.upload_resident(self.weights.lm_head.contiguous())
        self._lm_head_shape = tuple(self.weights.lm_head.shape)
        print(f"[GptOssModel] lm_head pinned in {time.time()-t0:.1f}s")

    def pin_all_to_vram(self) -> None:
        self.pin_moe_to_vram()
        self.pin_lm_head_to_vram()

    def _get_hf_cfg(self):
        if self._hf_cfg is None:
            from transformers import AutoConfig
            self._hf_cfg = AutoConfig.from_pretrained(self._model_dir)
        return self._hf_cfg

    def make_kv_cache(self, max_seqlen: int = 1024, batch: int = 1) -> KVCache:
        return KVCache(self.cfg, batch=batch, max_seqlen=max_seqlen)

    def compute_rope(self, positions: torch.Tensor):
        return compute_cos_sin_for_positions(self._get_hf_cfg(), positions)

    # ---------------- Forward ----------------
    def forward(self, input_ids: torch.Tensor,
                past_kv: KVCache | None = None,
                past_len: int = 0,
                only_last_logits: bool = False) -> torch.Tensor:
        """input_ids: [B, S_new] int64.
        Returns logits [B, S_new, vocab] or [B, 1, vocab] if only_last_logits.
        """
        B, S_new = input_ids.shape
        cfg = self.cfg
        w   = self.weights

        x = w.embed_tokens[input_ids].float()
        positions = torch.arange(past_len, past_len + S_new, dtype=torch.long)
        cos, sin = self.compute_rope(positions)

        for i, layer_weights in enumerate(w.layers):
            handles = self.resident_moe.for_layer(i) if self.resident_moe else None
            x = gpt_oss_layer_forward(
                self.ext, x, layer_idx=i, weights=layer_weights, cfg=cfg,
                cos=cos, sin=sin,
                past_kv=past_kv, past_len=past_len,
                use_sinks=True,
                resident_moe=handles,
            )

        x = rmsnorm(x, w.final_norm, cfg.rms_norm_eps)

        if only_last_logits:
            x = x[:, -1:, :].contiguous()   # [B, 1, D]
            S_out = 1
        else:
            S_out = S_new

        if self.h_lm_head is not None:
            V, D = self._lm_head_shape
            x_flat = x.reshape(B * S_out, D).contiguous()
            logits_flat = self.ext.linear_resident(
                x_flat, self.h_lm_head, 0, V, D, False)   # [B*S_out, V]
            return logits_flat.reshape(B, S_out, V)
        else:
            return x @ w.lm_head.T
