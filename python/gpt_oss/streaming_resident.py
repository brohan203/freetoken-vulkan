"""Bounded per-layer VRAM cache for streamed gpt-oss experts."""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch

from .loader import ExpertStore, ExpertTensors
from .resident import ResidentLayerHandles


@dataclass
class _LayerSlab:
    handles: ResidentLayerHandles
    slots: OrderedDict[int, int]
    free_slots: List[int]
    row_bytes: tuple[int, int, int, int, int, int]
    frequencies: Counter[int]
    recency: dict[int, int]
    clock: int


class StreamedResidentMoECache:
    """Cache a fixed number of experts per layer in device-local VRAM."""

    def __init__(
        self,
        ext,
        num_layers: int,
        slots_per_layer: int = 24,
        policy: str = "lfu",
    ):
        if slots_per_layer < 4:
            raise ValueError("slots_per_layer must be at least top-K (4)")
        if policy not in {"lru", "lfu"}:
            raise ValueError("policy must be 'lru' or 'lfu'")
        self.ext = ext
        self.slots_per_layer = slots_per_layer
        self.policy = policy
        self.layers: List[Optional[_LayerSlab]] = [None] * num_layers
        self.hits = 0
        self.misses = 0
        self.upload_seconds = 0.0
        self.uploaded_bytes = 0

    @staticmethod
    def _tensors(expert: ExpertTensors) -> tuple[torch.Tensor, ...]:
        return (
            expert.gate_up_blocks,
            expert.gate_up_scales,
            expert.gate_up_bias,
            expert.down_blocks,
            expert.down_scales,
            expert.down_bias,
        )

    def _allocate(self, layer_idx: int, prototype: ExpertTensors) -> _LayerSlab:
        tensors = self._tensors(prototype)
        row_bytes = tuple(t.numel() * t.element_size() for t in tensors)
        handles = [
            self.ext.allocate_resident(size * self.slots_per_layer)
            for size in row_bytes
        ]
        slab = _LayerSlab(
            handles=ResidentLayerHandles(
                h_gu_blocks=handles[0],
                h_gu_scales=handles[1],
                h_gu_bias=handles[2],
                h_d_blocks=handles[3],
                h_d_scales=handles[4],
                h_d_bias=handles[5],
                E=self.slots_per_layer,
                D=prototype.down_blocks.shape[1],
                Dff=prototype.gate_up_blocks.shape[1] // 2,
            ),
            slots=OrderedDict(),
            free_slots=list(range(self.slots_per_layer)),
            row_bytes=row_bytes,
            frequencies=Counter(),
            recency={},
            clock=0,
        )
        self.layers[layer_idx] = slab
        return slab

    def _evict(self, slab: _LayerSlab, selected: set[int]) -> int:
        candidates = [key for key in slab.slots if key not in selected]
        if not candidates:
            raise RuntimeError("no cache victim available outside selected experts")
        if self.policy == "lfu":
            victim = min(
                candidates,
                key=lambda key: (slab.frequencies[key], slab.recency[key]),
            )
        else:
            victim = candidates[0]
        slot = slab.slots.pop(victim)
        slab.recency.pop(victim, None)
        return slot

    def call(
        self,
        layer_idx: int,
        store: ExpertStore,
        x: torch.Tensor,
        global_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        import time

        ids = torch.unique(global_indices.to(torch.int64), sorted=True).tolist()
        store.record_selection(layer_idx, ids)
        if len(ids) > self.slots_per_layer:
            compact, local_indices = store.materialize_selected(
                layer_idx, global_indices
            )
            return self.ext.moe_mlp_gpt_oss(
                x,
                local_indices,
                weights,
                compact.gate_up_blocks,
                compact.gate_up_scales,
                compact.gate_up_bias,
                compact.down_blocks,
                compact.down_scales,
                compact.down_bias,
            )

        slab = self.layers[layer_idx]
        resident_ids = set(slab.slots) if slab is not None else set()
        missing = [expert_id for expert_id in ids if expert_id not in resident_ids]
        self.hits += len(ids) - len(missing)
        self.misses += len(missing)
        loaded = store.mapped_experts(layer_idx, missing)
        if slab is None:
            if not missing:
                raise RuntimeError("cannot initialize VRAM slab without an expert")
            slab = self._allocate(layer_idx, loaded[missing[0]])

        for expert_id in ids:
            slab.clock += 1
            slab.frequencies[expert_id] += 1
            slab.recency[expert_id] = slab.clock

        selected = set(ids)
        upload_handles: list[int] = []
        upload_tensors: list[torch.Tensor] = []
        upload_offsets: list[int] = []
        for expert_id in ids:
            if expert_id in slab.slots:
                slab.slots.move_to_end(expert_id)
                continue
            slot = (
                slab.free_slots.pop(0)
                if slab.free_slots
                else self._evict(slab, selected)
            )
            slab.slots[expert_id] = slot
            expert = loaded[expert_id]
            for handle, tensor, row_bytes in zip(
                (
                    slab.handles.h_gu_blocks,
                    slab.handles.h_gu_scales,
                    slab.handles.h_gu_bias,
                    slab.handles.h_d_blocks,
                    slab.handles.h_d_scales,
                    slab.handles.h_d_bias,
                ),
                self._tensors(expert),
                slab.row_bytes,
            ):
                upload_handles.append(handle)
                upload_tensors.append(tensor)
                upload_offsets.append(slot * row_bytes)
                self.uploaded_bytes += tensor.numel() * tensor.element_size()

        if upload_handles:
            t0 = time.perf_counter()
            self.ext.upload_resident_batch(
                upload_handles, upload_tensors, upload_offsets
            )
            self.upload_seconds += time.perf_counter() - t0

        local_indices = global_indices.clone()
        for expert_id in ids:
            local_indices[global_indices == expert_id] = slab.slots[expert_id]
        return slab.handles.call(
            self.ext, x, local_indices.contiguous(), weights
        )

    def free(self) -> None:
        for slab in self.layers:
            if slab is None:
                continue
            for handle in (
                slab.handles.h_gu_blocks,
                slab.handles.h_gu_scales,
                slab.handles.h_gu_bias,
                slab.handles.h_d_blocks,
                slab.handles.h_d_scales,
                slab.handles.h_d_bias,
            ):
                self.ext.free_resident(handle)
        self.layers = [None] * len(self.layers)
