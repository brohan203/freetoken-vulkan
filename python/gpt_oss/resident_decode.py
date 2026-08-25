"""Fully GPU-resident single-token decode for gpt-oss models."""
from __future__ import annotations

import math

import torch

from .resident_tensor import ResidentTensor
from .streaming_resident import StreamedResidentMoECache


class ResidentDecodeWorkspace:
    """Owned resident activations, control buffers, and per-layer KV slabs."""

    def __init__(self, ext, cfg, num_layers: int, capacity: int = 256):
        if capacity <= 0:
            raise ValueError("resident decode capacity must be positive")
        self.ext = ext
        self.cfg = cfg
        self.capacity = capacity
        self.owned: list[ResidentTensor] = []
        hidden = cfg.hidden_size
        query_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        top_k = cfg.num_experts_per_tok

        def empty(shape) -> ResidentTensor:
            tensor = ResidentTensor.empty(ext, shape)
            self.owned.append(tensor)
            return tensor

        # Ping-pong layer hidden state.
        self.hidden = [empty((1, hidden)), empty((1, hidden))]
        self.normalized = empty((1, hidden))
        self.q = empty((1, query_heads * head_dim))
        self.k = empty((1, kv_heads * head_dim))
        self.v = empty((1, kv_heads * head_dim))
        self.q_rotated = empty((1, query_heads, 1, head_dim))
        self.k_rotated = empty((1, kv_heads, 1, head_dim))
        self.attention = empty((1, query_heads * head_dim))
        self.projected = empty((1, hidden))
        self.residual = empty((1, hidden))
        self.post_norm = empty((1, hidden))
        self.router_logits = empty((1, cfg.num_experts))
        self.router_weights = empty((1, top_k))
        self.router_indices = ext.allocate_resident(top_k * 4)
        self.moe_hidden = empty((1, top_k, cfg.intermediate_size))
        self.moe_output = empty((1, hidden))
        self.cos = empty((1, head_dim))
        self.sin = empty((1, head_dim))
        self.final_normalized = empty((1, hidden))
        self.final_logits = empty((1, cfg.vocab_size))
        self.k_cache = [
            empty((1, kv_heads, capacity, head_dim)) for _ in range(num_layers)
        ]
        self.v_cache = [
            empty((1, kv_heads, capacity, head_dim)) for _ in range(num_layers)
        ]
        self._freed = False

    def upload_step(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> int:
        """Upload embedding plus RoPE tables and return the first ping-pong slot."""
        self._require_live()
        self.ext.upload_resident_batch(
            [self.hidden[0].handle, self.cos.handle, self.sin.handle],
            [
                hidden.reshape(1, -1).float().contiguous(),
                cos.float().contiguous(),
                sin.float().contiguous(),
            ],
            [0, 0, 0],
        )
        return 0

    def upload_input(self, hidden: torch.Tensor) -> int:
        """Test helper for updating only the input hidden state."""
        self._require_live()
        self.ext.update_resident(
            self.hidden[0].handle,
            hidden.reshape(1, -1).float().contiguous(),
            0,
        )
        return 0

    def upload_rope(self, cos: torch.Tensor, sin: torch.Tensor) -> None:
        """Test helper for updating only the RoPE tables."""
        self._require_live()
        self.ext.upload_resident_batch(
            [self.cos.handle, self.sin.handle],
            [cos.float().contiguous(), sin.float().contiguous()],
            [0, 0],
        )

    def load_kv_cache(self, cache) -> None:
        """Copy a CPU prefill KV cache into capacity-strided resident slabs."""
        self._require_live()
        if cache.max_seqlen != self.capacity:
            raise ValueError(
                f"CPU KV capacity {cache.max_seqlen} != resident capacity "
                f"{self.capacity}"
            )
        handles: list[int] = []
        tensors: list[torch.Tensor] = []
        offsets: list[int] = []
        for layer_idx in range(len(self.k_cache)):
            handles.extend(
                [self.k_cache[layer_idx].handle, self.v_cache[layer_idx].handle]
            )
            tensors.extend(
                [cache._k[layer_idx].contiguous(), cache._v[layer_idx].contiguous()]
            )
            offsets.extend([0, 0])
        self.ext.upload_resident_batch(handles, tensors, offsets)

    def free(self) -> None:
        if self._freed:
            return
        self.ext.free_resident(self.router_indices)
        for tensor in self.owned:
            tensor.free()
        self.owned.clear()
        self._freed = True

    def _require_live(self) -> None:
        if self._freed:
            raise RuntimeError("ResidentDecodeWorkspace has been freed")

    def __enter__(self) -> "ResidentDecodeWorkspace":
        self._require_live()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.free()

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass


@torch.no_grad()
def resident_decode_layer(
    ext,
    workspace: ResidentDecodeWorkspace,
    layer_idx: int,
    current_hidden: int,
    projection,
    expert_store,
    expert_cache: StreamedResidentMoECache,
    position: int,
    sliding_window: int,
) -> tuple[int, torch.Tensor]:
    """Run one single-token layer with only top-K IDs crossing to CPU."""
    cfg = workspace.cfg
    hidden = cfg.hidden_size
    query_heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    next_hidden = 1 - current_hidden
    x = workspace.hidden[current_hidden]
    output = workspace.hidden[next_hidden]

    ext.rmsnorm_qkv_resident(
        x.handle,
        projection.input_norm,
        workspace.normalized.handle,
        projection.q_weight,
        projection.q_bias,
        workspace.q.handle,
        projection.k_weight,
        projection.k_bias,
        workspace.k.handle,
        projection.v_weight,
        projection.v_bias,
        workspace.v.handle,
        1,
        hidden,
        query_heads * head_dim,
        kv_heads * head_dim,
        cfg.rms_norm_eps,
    )
    ext.rope_kv_attention_resident(
        workspace.q.handle,
        workspace.k.handle,
        workspace.v.handle,
        workspace.cos.handle,
        workspace.sin.handle,
        workspace.q_rotated.handle,
        workspace.k_rotated.handle,
        workspace.k_cache[layer_idx].handle,
        workspace.v_cache[layer_idx].handle,
        projection.sinks,
        workspace.attention.handle,
        1,
        query_heads,
        kv_heads,
        1,
        head_dim,
        position,
        workspace.capacity,
        sliding_window,
        True,
        1.0 / math.sqrt(head_dim),
    )
    ext.oproj_router_resident(
        workspace.attention.handle,
        projection.o_weight,
        projection.o_bias,
        workspace.projected.handle,
        x.handle,
        workspace.residual.handle,
        projection.post_norm,
        workspace.post_norm.handle,
        projection.router_weight,
        projection.router_bias,
        workspace.router_logits.handle,
        1,
        hidden,
        query_heads * head_dim,
        cfg.num_experts,
        cfg.rms_norm_eps,
    )
    ext.topk_resident(
        workspace.router_logits.handle,
        workspace.router_indices,
        workspace.router_weights.handle,
        1,
    )
    global_ids = ext.download_resident_i32(
        workspace.router_indices, [1, cfg.num_experts_per_tok]
    ).long()
    expert_cache.call_resident(
        layer_idx,
        expert_store,
        workspace.post_norm.handle,
        global_ids,
        workspace.router_weights.handle,
        workspace.router_indices,
        workspace.moe_hidden.handle,
        workspace.moe_output.handle,
        1,
    )
    ext.add_resident(
        workspace.residual.handle,
        workspace.moe_output.handle,
        output.handle,
        hidden,
    )
    return next_hidden, global_ids


@torch.no_grad()
def resident_decode_model_step(
    model, workspace: ResidentDecodeWorkspace, token_id: int, position: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run all layers and return CPU logits plus per-layer global expert IDs."""
    if position >= workspace.capacity:
        raise ValueError(
            f"position {position} exceeds resident capacity {workspace.capacity}"
        )
    cfg = model.cfg
    token = torch.tensor([[token_id]], dtype=torch.long)
    embedding = model.weights.embed_tokens[token]
    cos, sin = model.compute_rope(torch.tensor([position]))
    current = workspace.upload_step(embedding, cos, sin)
    all_ids: list[torch.Tensor] = []
    for layer_idx in range(cfg.num_hidden_layers):
        sliding_window = (
            cfg.sliding_window if cfg.layer_is_sliding(layer_idx) else 0
        )
        current, ids = resident_decode_layer(
            model.ext,
            workspace,
            layer_idx,
            current,
            model.resident_projections.for_layer(layer_idx),
            model.weights.expert_store,
            model.streamed_resident,
            position,
            sliding_window,
        )
        all_ids.append(ids)

    model.ext.rmsnorm_resident(
        workspace.hidden[current].handle,
        model.h_final_norm,
        workspace.final_normalized.handle,
        1,
        cfg.hidden_size,
        cfg.rms_norm_eps,
    )
    model.ext.linear_resident_io(
        workspace.final_normalized.handle,
        model.h_lm_head,
        0,
        workspace.final_logits.handle,
        1,
        cfg.vocab_size,
        cfg.hidden_size,
        False,
    )
    return workspace.final_logits.download(), all_ids
