"""Bounded per-layer device-local cache for streamed MXFP4 experts."""
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
        row_bytes = tuple(tensor.numel() * tensor.element_size() for tensor in tensors)
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
        candidates = [expert for expert in slab.slots if expert not in selected]
        if not candidates:
            raise RuntimeError("no cache victim available outside selected experts")
        if self.policy == "lfu":
            victim = min(
                candidates,
                key=lambda expert: (
                    slab.frequencies[expert], slab.recency[expert]
                ),
            )
        else:
            victim = candidates[0]
        slot = slab.slots.pop(victim)
        slab.recency.pop(victim, None)
        return slot

    def _prepare(
        self,
        layer_idx: int,
        store: ExpertStore,
        global_indices: torch.Tensor,
    ) -> tuple[_LayerSlab, torch.Tensor]:
        import time

        ids = torch.unique(global_indices.to(torch.int64), sorted=True).tolist()
        store.record_selection(layer_idx, ids)
        if len(ids) > self.slots_per_layer:
            raise RuntimeError("resident expert path exceeds configured slots")

        slab = self.layers[layer_idx]
        resident_ids = set(slab.slots) if slab is not None else set()
        missing = [expert for expert in ids if expert not in resident_ids]
        self.hits += len(ids) - len(missing)
        self.misses += len(missing)
        loaded = store.mapped_experts(layer_idx, missing)

        if slab is None:
            if not missing:
                raise RuntimeError("cannot initialize slab without an expert")
            slab = self._allocate(layer_idx, loaded[missing[0]])

        for expert in ids:
            slab.clock += 1
            slab.frequencies[expert] += 1
            slab.recency[expert] = slab.clock

        selected = set(ids)
        upload_handles: list[int] = []
        upload_tensors: list[torch.Tensor] = []
        upload_offsets: list[int] = []
        for expert in ids:
            if expert in slab.slots:
                slab.slots.move_to_end(expert)
                continue
            slot = (
                slab.free_slots.pop(0)
                if slab.free_slots
                else self._evict(slab, selected)
            )
            slab.slots[expert] = slot
            tensors = loaded[expert]
            for handle, tensor, bytes_per_row in zip(
                (
                    slab.handles.h_gu_blocks,
                    slab.handles.h_gu_scales,
                    slab.handles.h_gu_bias,
                    slab.handles.h_d_blocks,
                    slab.handles.h_d_scales,
                    slab.handles.h_d_bias,
                ),
                self._tensors(tensors),
                slab.row_bytes,
            ):
                upload_handles.append(handle)
                upload_tensors.append(tensor)
                upload_offsets.append(slot * bytes_per_row)
                self.uploaded_bytes += tensor.numel() * tensor.element_size()

        if upload_handles:
            started = time.perf_counter()
            self.ext.upload_resident_batch(
                upload_handles, upload_tensors, upload_offsets
            )
            self.upload_seconds += time.perf_counter() - started

        local = global_indices.to(torch.int32).clone()
        for expert in ids:
            local[global_indices == expert] = slab.slots[expert]
        return slab, local.contiguous()

    def call(
        self,
        layer_idx: int,
        store: ExpertStore,
        x: torch.Tensor,
        global_indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if torch.unique(global_indices).numel() > self.slots_per_layer:
            compact, local = store.materialize_selected(layer_idx, global_indices)
            return self.ext.moe_mlp_gpt_oss(
                x,
                local,
                weights,
                compact.gate_up_blocks,
                compact.gate_up_scales,
                compact.gate_up_bias,
                compact.down_blocks,
                compact.down_scales,
                compact.down_bias,
            )
        slab, local = self._prepare(layer_idx, store, global_indices)
        return slab.handles.call(
            self.ext, x, local, weights, two_stage=True
        )

    def call_resident(
        self,
        layer_idx: int,
        store: ExpertStore,
        x_handle: int,
        global_indices: torch.Tensor,
        weights_handle: int,
        indices_handle: int,
        hidden_handle: int,
        output_handle: int,
        rows: int = 1,
    ) -> None:
        slab, local = self._prepare(layer_idx, store, global_indices)
        self.ext.update_resident(indices_handle, local, 0)
        self.ext.moe_mlp_gpt_oss_twostage_io(
            x_handle,
            indices_handle,
            weights_handle,
            hidden_handle,
            output_handle,
            slab.handles.h_gu_blocks,
            slab.handles.h_gu_scales,
            slab.handles.h_gu_bias,
            slab.handles.h_d_blocks,
            slab.handles.h_d_scales,
            slab.handles.h_d_bias,
            slab.handles.E,
            slab.handles.D,
            slab.handles.Dff,
            rows,
            4,
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
