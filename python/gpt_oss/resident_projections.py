"""Resident FP32 projection and normalization weights for decode layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .loader import LayerWeights, ModelWeights


@dataclass
class ResidentProjectionLayer:
    input_norm: int
    q_weight: int
    q_bias: int
    k_weight: int
    k_bias: int
    v_weight: int
    v_bias: int
    o_weight: int
    o_bias: int
    post_norm: int
    router_weight: int
    router_bias: int
    sinks: int

    def handles(self) -> list[int]:
        return [
            self.input_norm,
            self.q_weight,
            self.q_bias,
            self.k_weight,
            self.k_bias,
            self.v_weight,
            self.v_bias,
            self.o_weight,
            self.o_bias,
            self.post_norm,
            self.router_weight,
            self.router_bias,
            self.sinks,
        ]


class ResidentProjectionWeights:
    def __init__(self, ext, weights: ModelWeights, verbose: bool = True):
        import time

        self.ext = ext
        self.layers: List[ResidentProjectionLayer] = []
        started = time.time()
        for index, layer in enumerate(weights.layers):
            self.layers.append(self._upload(layer))
            if verbose and (index + 1) % 6 == 0:
                gib = ext.resident_bytes_total() / 1024**3
                print(
                    f"  [projections {index+1}/{len(weights.layers)}] "
                    f"{gib:.2f} GiB total resident"
                )
        if verbose:
            print(
                f"[Resident projections] {len(self.layers)} layers in "
                f"{time.time()-started:.1f}s"
            )

    def _upload(self, layer: LayerWeights) -> ResidentProjectionLayer:
        upload = self.ext.upload_resident
        tensors = [
            layer.input_layernorm_weight,
            layer.q_proj_weight,
            layer.q_proj_bias,
            layer.k_proj_weight,
            layer.k_proj_bias,
            layer.v_proj_weight,
            layer.v_proj_bias,
            layer.o_proj_weight,
            layer.o_proj_bias,
            layer.post_attention_layernorm_weight,
            layer.router_weight,
            layer.router_bias,
            layer.sinks,
        ]
        return ResidentProjectionLayer(*(upload(tensor) for tensor in tensors))

    def for_layer(self, layer_idx: int) -> ResidentProjectionLayer:
        return self.layers[layer_idx]

    def free(self) -> None:
        if not self.layers:
            return
        for layer in self.layers:
            for handle in layer.handles():
                self.ext.free_resident(handle)
        self.layers.clear()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass
