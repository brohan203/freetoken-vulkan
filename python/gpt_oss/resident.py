"""resident.py — persistent VRAM manager for gpt-oss MoE expert weights.

Uploads all 6 MoE weight tensors (gate_up_blocks/scales/bias +
down_blocks/scales/bias) for every transformer layer to VRAM ONCE at
model-load time, then reuses them across every kernel call. Eliminates
~424 MB × 24 layers = ~10 GB of per-forward PCIe traffic.

Estimated VRAM footprint on gpt-oss-20b: ~404 MB × 24 = ~9.7 GB. Fits
comfortably in the 6800 XT's 16 GB budget alongside activations, KV
cache, and Vulkan's own scratch.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import time

import torch

from .loader import ModelWeights, LayerWeights


@dataclass
class ResidentLayerHandles:
    """Handles + shape metadata for one layer's VRAM-resident MoE weights."""
    h_gu_blocks: int
    h_gu_scales: int
    h_gu_bias:   int
    h_d_blocks:  int
    h_d_scales:  int
    h_d_bias:    int
    E: int
    D: int
    Dff: int

    def call(self, ext, x: torch.Tensor, indices: torch.Tensor,
             weights: torch.Tensor, two_stage: bool = False) -> torch.Tensor:
        op = (
            ext.moe_mlp_gpt_oss_twostage
            if two_stage
            else ext.moe_mlp_gpt_oss_resident
        )
        return op(
            x, indices, weights,
            self.h_gu_blocks, self.h_gu_scales, self.h_gu_bias,
            self.h_d_blocks,  self.h_d_scales,  self.h_d_bias,
            self.E, self.D, self.Dff,
        )


class ResidentMoEWeights:
    """Uploads MoE experts for every layer to VRAM once. Access per-layer
    handles via `resident.for_layer(layer_idx)`.
    """
    def __init__(self, ext, weights: ModelWeights, verbose: bool = True):
        self.ext = ext
        self.handles: List[ResidentLayerHandles] = []
        self.total_bytes = 0

        cfg = weights.config
        D = cfg.hidden_size

        t0 = time.time()
        for li, lw in enumerate(weights.layers):
            self.handles.append(self._upload_one(lw, D))
            if verbose and (li + 1) % 6 == 0:
                elapsed = time.time() - t0
                mb = ext.resident_bytes_total() / 1024**3
                print(f"  [{li+1:2d}/{len(weights.layers)}] layers uploaded  "
                      f"({elapsed:.1f}s, {mb:.2f} GB in VRAM)")
        self.total_bytes = ext.resident_bytes_total()
        if verbose:
            print(f"[Resident] Uploaded {len(self.handles)} layers, "
                  f"{self.total_bytes/1024**3:.2f} GB in VRAM in {time.time()-t0:.1f}s")

    def _upload_one(self, lw: LayerWeights, D: int) -> ResidentLayerHandles:
        E, two_Dff = lw.gate_up_blocks.shape[0], lw.gate_up_blocks.shape[1]
        Dff = two_Dff // 2
        return ResidentLayerHandles(
            h_gu_blocks = self.ext.upload_resident(lw.gate_up_blocks),
            h_gu_scales = self.ext.upload_resident(lw.gate_up_scales),
            h_gu_bias   = self.ext.upload_resident(lw.gate_up_bias),
            h_d_blocks  = self.ext.upload_resident(lw.down_blocks),
            h_d_scales  = self.ext.upload_resident(lw.down_scales),
            h_d_bias    = self.ext.upload_resident(lw.down_bias),
            E=E, D=D, Dff=Dff,
        )

    def for_layer(self, layer_idx: int) -> ResidentLayerHandles:
        return self.handles[layer_idx]

    def free(self):
        for h in self.handles:
            for hid in (h.h_gu_blocks, h.h_gu_scales, h.h_gu_bias,
                        h.h_d_blocks,  h.h_d_scales,  h.h_d_bias):
                self.ext.free_resident(hid)
        self.handles.clear()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass
