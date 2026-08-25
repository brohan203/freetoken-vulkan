"""Full gpt-oss forward pass with optional KV and resident weights."""
from __future__ import annotations

import pathlib
import time

import torch

from .config import GptOssConfig
from .kv_cache import KVCache
from .layer import gpt_oss_layer_forward, rmsnorm
from .loader import ModelWeights, load_model
from .resident import ResidentMoEWeights
from .rope import compute_cos_sin_for_positions


class GptOssModel:
    def __init__(self, ext, weights: ModelWeights):
        self.ext = ext
        self.weights = weights
        self.cfg: GptOssConfig = weights.config
        self._hf_cfg = None
        self._model_dir = None
        self.resident_moe: ResidentMoEWeights | None = None
        self.resident_projections = None
        self.h_final_norm: int | None = None
        self.streamed_resident = None
        self.h_lm_head: int | None = None
        self._lm_head_shape: tuple[int, int] | None = None
        self._lm_head_fp16 = False
        self._closed = False

    @classmethod
    def from_pretrained(
        cls,
        ext,
        model_dir: str | pathlib.Path,
        layers: list[int] | None = None,
        *,
        stream_experts: bool = False,
        expert_cache_size: int = 16,
    ) -> "GptOssModel":
        model_dir = pathlib.Path(model_dir)
        print(f"[GptOssModel] Loading from {model_dir}")
        t0 = time.time()
        weights = load_model(
            model_dir,
            layers=layers,
            stream_experts=stream_experts,
            expert_cache_size=expert_cache_size,
        )
        print(
            f"[GptOssModel] Loaded {len(weights.layers)}/"
            f"{weights.config.num_hidden_layers} layers in {time.time()-t0:.1f}s"
        )
        model = cls(ext, weights)
        model._model_dir = model_dir
        return model

    def pin_moe_to_vram(self, verbose: bool = True) -> None:
        self._require_open()
        if self.resident_moe is None:
            self.resident_moe = ResidentMoEWeights(
                self.ext, self.weights, verbose=verbose
            )

    def enable_streamed_vram_cache(
        self, slots_per_layer: int = 24, policy: str = "lfu"
    ) -> None:
        """Cache streamed experts in bounded per-layer VRAM slabs."""
        self._require_open()
        if self.weights.expert_store is None:
            raise RuntimeError("streamed VRAM cache requires stream_experts=True")
        if self.streamed_resident is None:
            from .streaming_resident import StreamedResidentMoECache
            self.streamed_resident = StreamedResidentMoECache(
                self.ext, len(self.weights.layers), slots_per_layer, policy
            )

    def pin_projections_to_vram(self, verbose: bool = True) -> None:
        self._require_open()
        if self.resident_projections is None:
            from .resident_projections import ResidentProjectionWeights
            self.resident_projections = ResidentProjectionWeights(
                self.ext, self.weights, verbose=verbose
            )
            self.h_final_norm = self.ext.upload_resident(
                self.weights.final_norm.contiguous()
            )

    def pin_lm_head_to_vram(self, fp16: bool = False) -> None:
        self._require_open()
        if self.h_lm_head is not None:
            return
        source = (
            self.weights.lm_head.half().contiguous()
            if fp16 else self.weights.lm_head.contiguous()
        )
        self._lm_head_fp16 = fp16
        print(
            f"[GptOssModel] Pinning lm_head to VRAM "
            f"({source.numel()*source.element_size()/1024**3:.2f} GB, "
            f"fp16={fp16})..."
        )
        t0 = time.time()
        self.h_lm_head = self.ext.upload_resident(source)
        self._lm_head_shape = tuple(self.weights.lm_head.shape)
        print(f"[GptOssModel] lm_head pinned in {time.time()-t0:.1f}s")

    def pin_all_to_vram(self) -> None:
        self.pin_moe_to_vram()
        self.pin_lm_head_to_vram()

    def close(self) -> None:
        """Release model-owned resident allocations exactly once."""
        if self._closed:
            return
        if self.resident_moe is not None:
            self.resident_moe.free()
            self.resident_moe = None
        if self.streamed_resident is not None:
            self.streamed_resident.free()
            self.streamed_resident = None
        if self.resident_projections is not None:
            self.resident_projections.free()
            self.resident_projections = None
        if self.h_final_norm is not None:
            self.ext.free_resident(self.h_final_norm)
            self.h_final_norm = None
        if self.h_lm_head is not None:
            self.ext.free_resident(self.h_lm_head)
            self.h_lm_head = None
        self._lm_head_shape = None
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("GptOssModel has been closed")

    def __enter__(self) -> "GptOssModel":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _get_hf_cfg(self):
        if self._hf_cfg is None:
            from transformers import AutoConfig
            self._hf_cfg = AutoConfig.from_pretrained(self._model_dir)
        return self._hf_cfg

    def make_kv_cache(self, max_seqlen: int = 1024, batch: int = 1) -> KVCache:
        self._require_open()
        return KVCache(self.cfg, batch=batch, max_seqlen=max_seqlen)

    def compute_rope(self, positions: torch.Tensor):
        self._require_open()
        return compute_cos_sin_for_positions(self._get_hf_cfg(), positions)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_kv: KVCache | None = None,
        past_len: int = 0,
        only_last_logits: bool = False,
    ) -> torch.Tensor:
        """Run new tokens through the model and return FP32 logits."""
        self._require_open()

        batch, new_tokens = input_ids.shape
        cfg = self.cfg
        weights = self.weights
        x = weights.embed_tokens[input_ids].float()
        positions = torch.arange(past_len, past_len + new_tokens, dtype=torch.long)
        cos, sin = self.compute_rope(positions)

        for layer_idx, layer_weights in enumerate(weights.layers):
            handles = (
                self.resident_moe.for_layer(layer_idx)
                if self.resident_moe is not None
                else None
            )
            x = gpt_oss_layer_forward(
                self.ext,
                x,
                layer_idx=layer_idx,
                weights=layer_weights,
                cfg=cfg,
                cos=cos,
                sin=sin,
                past_kv=past_kv,
                past_len=past_len,
                use_sinks=True,
                resident_moe=handles,
                expert_store=weights.expert_store,
                streamed_resident=self.streamed_resident,
            )

        x = rmsnorm(x, weights.final_norm, cfg.rms_norm_eps)
        if only_last_logits:
            x = x[:, -1:, :].contiguous()
            output_tokens = 1
        else:
            output_tokens = new_tokens

        if self.h_lm_head is not None and not self._lm_head_fp16:
            vocab, hidden = self._lm_head_shape
            x_flat = x.reshape(batch * output_tokens, hidden).contiguous()
            logits = self.ext.linear_resident(
                x_flat, self.h_lm_head, 0, vocab, hidden, False
            )
            return logits.reshape(batch, output_tokens, vocab)
        return x @ weights.lm_head.T
