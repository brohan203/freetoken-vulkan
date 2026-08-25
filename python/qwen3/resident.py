"""Resident native-BF16 weights and single-token Qwen3 layer execution."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from gpt_oss.resident_tensor import ResidentTensor
from model_contracts import DenseDecoderConfig, DenseDecoderLayerWeights


@dataclass
class ResidentQwen3LayerWeights:
    input_norm: int
    post_norm: int
    q_norm: int
    k_norm: int
    q_weight: int
    k_weight: int
    v_weight: int
    o_weight: int
    gate_weight: int
    up_weight: int
    down_weight: int
    weight_format: str = "bf16"
    q_scale: int = 0
    k_scale: int = 0
    v_scale: int = 0
    o_scale: int = 0
    gate_scale: int = 0
    up_scale: int = 0
    down_scale: int = 0

    def handles(self) -> list[int]:
        return [handle for handle in [
            self.input_norm, self.post_norm, self.q_norm, self.k_norm,
            self.q_weight, self.k_weight, self.v_weight, self.o_weight,
            self.gate_weight, self.up_weight, self.down_weight,
            self.q_scale, self.k_scale, self.v_scale, self.o_scale,
            self.gate_scale, self.up_scale, self.down_scale,
        ] if handle]


class ResidentQwen3Weights:
    def __init__(self, ext):
        self.ext = ext
        self.layers: list[ResidentQwen3LayerWeights] = []
        self.embed_tokens: int | None = None
        self.final_norm: int | None = None
        self.lm_head: int | None = None
        self.lm_head_scale: int | None = None
        self.lm_head_format = "bf16"

    def pin_global(
        self, embed_tokens: torch.Tensor, final_norm: torch.Tensor,
        lm_head: torch.Tensor | None = None,
        lm_head_scale: torch.Tensor | None = None,
    ) -> None:
        if self.embed_tokens is not None or self.final_norm is not None:
            raise RuntimeError("Qwen3 global weights already pinned")
        self.embed_tokens = self.ext.upload_resident(
            embed_tokens.bfloat16().contiguous()
        )
        self.final_norm = self.ext.upload_resident(
            final_norm.float().contiguous()
        )
        if lm_head is None:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = self.ext.upload_resident(lm_head.contiguous())
            if lm_head_scale is not None:
                self.lm_head_scale = self.ext.upload_resident(
                    lm_head_scale.float().contiguous()
                )
                self.lm_head_format = "fp8"

    def append(self, weights: DenseDecoderLayerWeights) -> ResidentQwen3LayerWeights:
        if weights.q_norm is None or weights.k_norm is None:
            raise ValueError("Qwen3 resident weights require Q/K norm vectors")
        upload = self.ext.upload_resident
        layer = ResidentQwen3LayerWeights(
            input_norm=upload(weights.input_norm.float().contiguous()),
            post_norm=upload(weights.post_attention_norm.float().contiguous()),
            q_norm=upload(weights.q_norm.float().contiguous()),
            k_norm=upload(weights.k_norm.float().contiguous()),
            q_weight=upload(weights.q_weight.bfloat16().contiguous()),
            k_weight=upload(weights.k_weight.bfloat16().contiguous()),
            v_weight=upload(weights.v_weight.bfloat16().contiguous()),
            o_weight=upload(weights.o_weight.bfloat16().contiguous()),
            gate_weight=upload(weights.gate_weight.bfloat16().contiguous()),
            up_weight=upload(weights.up_weight.bfloat16().contiguous()),
            down_weight=upload(weights.down_weight.bfloat16().contiguous()),
        )
        self.layers.append(layer)
        return layer

    def append_fp8(self, tensors, layer_idx: int) -> ResidentQwen3LayerWeights:
        prefix = f"model.layers.{layer_idx}"
        upload = self.ext.upload_resident

        def norm(suffix: str) -> int:
            return upload(tensors.get(f"{prefix}.{suffix}").float().contiguous())

        def matrix(suffix: str) -> tuple[int, int]:
            name = f"{prefix}.{suffix}"
            return (
                upload(tensors.get(name).contiguous()),
                upload(tensors.get(name + "_scale_inv").float().contiguous()),
            )

        q, qs = matrix("self_attn.q_proj.weight")
        k, ks = matrix("self_attn.k_proj.weight")
        v, vs = matrix("self_attn.v_proj.weight")
        o, os = matrix("self_attn.o_proj.weight")
        gate, gates = matrix("mlp.gate_proj.weight")
        up, ups = matrix("mlp.up_proj.weight")
        down, downs = matrix("mlp.down_proj.weight")
        layer = ResidentQwen3LayerWeights(
            input_norm=norm("input_layernorm.weight"),
            post_norm=norm("post_attention_layernorm.weight"),
            q_norm=norm("self_attn.q_norm.weight"),
            k_norm=norm("self_attn.k_norm.weight"),
            q_weight=q, k_weight=k, v_weight=v, o_weight=o,
            gate_weight=gate, up_weight=up, down_weight=down,
            weight_format="fp8",
            q_scale=qs, k_scale=ks, v_scale=vs, o_scale=os,
            gate_scale=gates, up_scale=ups, down_scale=downs,
        )
        self.layers.append(layer)
        return layer

    def free(self) -> None:
        for layer in self.layers:
            for handle in layer.handles():
                self.ext.free_resident(handle)
        self.layers.clear()
        embed_handle = self.embed_tokens
        if self.lm_head is not None and self.lm_head != embed_handle:
            self.ext.free_resident(self.lm_head)
        self.lm_head = None
        if embed_handle is not None:
            self.ext.free_resident(embed_handle)
            self.embed_tokens = None
        if self.final_norm is not None:
            self.ext.free_resident(self.final_norm)
            self.final_norm = None
        if self.lm_head_scale is not None:
            self.ext.free_resident(self.lm_head_scale)
            self.lm_head_scale = None


class ResidentQwen3Workspace:
    def __init__(self, ext, config: DenseDecoderConfig, capacity: int = 256):
        self.ext = ext
        self.config = config
        self.capacity = capacity
        self.owned: list[ResidentTensor] = []

        def empty(shape):
            tensor = ResidentTensor.empty(ext, shape)
            self.owned.append(tensor)
            return tensor

        d = config.hidden_size
        q = config.query_size
        kv = config.kv_size
        ff = config.intermediate_size
        self.hidden = [empty((1, d)), empty((1, d))]
        self.normalized = empty((1, d))
        self.q = empty((1, q))
        self.k = empty((1, kv))
        self.v = empty((1, kv))
        self.q_rotated = empty((1, config.num_attention_heads, 1, config.head_dim))
        self.k_rotated = empty((1, config.num_key_value_heads, 1, config.head_dim))
        self.attention = empty((1, q))
        self.projected = empty((1, d))
        self.residual = empty((1, d))
        self.post_normalized = empty((1, d))
        self.gate = empty((1, ff))
        self.up = empty((1, ff))
        self.activated = empty((1, ff))
        self.mlp_output = empty((1, d))
        self.final_normalized = empty((1, d))
        self.final_logits = empty((1, config.vocab_size))
        self.cos = empty((1, config.head_dim))
        self.sin = empty((1, config.head_dim))
        self.sinks = empty((config.num_attention_heads,))
        self.ext.update_resident(
            self.sinks.handle,
            torch.zeros(config.num_attention_heads, dtype=torch.float32),
            0,
        )
        self.k_cache = [
            empty((1, config.num_key_value_heads, capacity, config.head_dim))
            for _ in range(config.num_hidden_layers)
        ]
        self.v_cache = [
            empty((1, config.num_key_value_heads, capacity, config.head_dim))
            for _ in range(config.num_hidden_layers)
        ]
        self._freed = False

    def upload_input(self, hidden, cos, sin) -> int:
        self.ext.upload_resident_batch(
            [self.hidden[0].handle, self.cos.handle, self.sin.handle],
            [hidden.reshape(1, -1).float().contiguous(), cos.float().contiguous(), sin.float().contiguous()],
            [0, 0, 0],
        )
        return 0

    def free(self) -> None:
        if self._freed:
            return
        for tensor in self.owned:
            tensor.free()
        self.owned.clear()
        self._freed = True


def resident_qwen3_layer(
    ext,
    workspace: ResidentQwen3Workspace,
    weights: ResidentQwen3LayerWeights,
    layer_idx: int,
    input_slot: int,
    position: int,
) -> int:
    cfg = workspace.config
    output_slot = 1 - input_slot
    if weights.weight_format == "fp8":
        x = workspace.hidden[input_slot]
        out = workspace.hidden[output_slot]
        d = cfg.hidden_size
        hd = cfg.head_dim
        hq = cfg.num_attention_heads
        hkv = cfg.num_key_value_heads

        def linear(source, weight, scale, target, output_size, input_size):
            ext.linear_fp8e4m3_resident_io(
                source.handle, weight, scale, target.handle,
                1, output_size, input_size, (input_size + 127) // 128,
            )

        ext.rmsnorm_resident(x.handle, weights.input_norm, workspace.normalized.handle, 1, d, cfg.rms_norm_eps)
        linear(workspace.normalized, weights.q_weight, weights.q_scale, workspace.q, cfg.query_size, d)
        linear(workspace.normalized, weights.k_weight, weights.k_scale, workspace.k, cfg.kv_size, d)
        linear(workspace.normalized, weights.v_weight, weights.v_scale, workspace.v, cfg.kv_size, d)
        ext.rmsnorm_resident(workspace.q.handle, weights.q_norm, workspace.q.handle, hq, hd, cfg.rms_norm_eps)
        ext.rmsnorm_resident(workspace.k.handle, weights.k_norm, workspace.k.handle, hkv, hd, cfg.rms_norm_eps)
        ext.rope_kv_attention_resident(
            workspace.q.handle, workspace.k.handle, workspace.v.handle,
            workspace.cos.handle, workspace.sin.handle,
            workspace.q_rotated.handle, workspace.k_rotated.handle,
            workspace.k_cache[layer_idx].handle, workspace.v_cache[layer_idx].handle,
            workspace.sinks.handle, workspace.attention.handle,
            1, hq, hkv, 1, hd, position, workspace.capacity, 0, False,
            1.0 / (hd ** 0.5),
        )
        linear(workspace.attention, weights.o_weight, weights.o_scale, workspace.projected, d, cfg.query_size)
        ext.add_resident(workspace.projected.handle, x.handle, workspace.residual.handle, d)
        ext.rmsnorm_resident(workspace.residual.handle, weights.post_norm, workspace.post_normalized.handle, 1, d, cfg.rms_norm_eps)
        linear(workspace.post_normalized, weights.gate_weight, weights.gate_scale, workspace.gate, cfg.intermediate_size, d)
        linear(workspace.post_normalized, weights.up_weight, weights.up_scale, workspace.up, cfg.intermediate_size, d)
        ext.swiglu_resident(workspace.gate.handle, workspace.up.handle, workspace.activated.handle, cfg.intermediate_size)
        linear(workspace.activated, weights.down_weight, weights.down_scale, workspace.mlp_output, d, cfg.intermediate_size)
        ext.add_resident(workspace.residual.handle, workspace.mlp_output.handle, out.handle, d)
        return output_slot

    ext.qwen3_decode_layer_resident(
        workspace.hidden[input_slot].handle,
        weights.input_norm,
        workspace.normalized.handle,
        weights.q_weight,
        workspace.q.handle,
        weights.k_weight,
        workspace.k.handle,
        weights.v_weight,
        workspace.v.handle,
        weights.q_norm,
        weights.k_norm,
        workspace.cos.handle,
        workspace.sin.handle,
        workspace.q_rotated.handle,
        workspace.k_rotated.handle,
        workspace.k_cache[layer_idx].handle,
        workspace.v_cache[layer_idx].handle,
        workspace.sinks.handle,
        workspace.attention.handle,
        weights.o_weight,
        workspace.projected.handle,
        workspace.residual.handle,
        weights.post_norm,
        workspace.post_normalized.handle,
        weights.gate_weight,
        workspace.gate.handle,
        weights.up_weight,
        workspace.up.handle,
        workspace.activated.handle,
        weights.down_weight,
        workspace.mlp_output.handle,
        workspace.hidden[output_slot].handle,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        position,
        workspace.capacity,
        cfg.rms_norm_eps,
        1.0 / (cfg.head_dim ** 0.5),
    )
    return output_slot


@torch.no_grad()
def resident_qwen3_model_step(model, workspace: ResidentQwen3Workspace, token_id: int, position: int) -> torch.Tensor:
    """Run one fully resident Qwen3 token and return CPU FP32 logits."""
    from .rope import compute_rope

    resident = model.resident_weights
    if resident is None or resident.embed_tokens is None or resident.final_norm is None:
        raise RuntimeError("pin Qwen3 weights before resident decode")
    if len(resident.layers) != model.config.num_hidden_layers:
        raise RuntimeError("not all Qwen3 layers are resident")
    embedding = model.embed_tokens[int(token_id)].float().reshape(1, -1)
    cos, sin = compute_rope(model.config, torch.tensor([position]))
    slot = workspace.upload_input(embedding, cos, sin)
    for layer_idx, weights in enumerate(resident.layers):
        slot = resident_qwen3_layer(
            model.ext, workspace, weights, layer_idx, slot, position
        )
    workspace.ext.rmsnorm_resident(
        workspace.hidden[slot].handle,
        resident.final_norm,
        workspace.final_normalized.handle,
        1,
        model.config.hidden_size,
        model.config.rms_norm_eps,
    )
    if resident.lm_head is None:
        raise RuntimeError("resident LM head is not pinned")
    if resident.lm_head_format == "fp8":
        if resident.lm_head_scale is None:
            raise RuntimeError("resident FP8 LM head scale is missing")
        workspace.ext.linear_fp8e4m3_resident_io(
            workspace.final_normalized.handle,
            resident.lm_head,
            resident.lm_head_scale,
            workspace.final_logits.handle,
            1,
            model.config.vocab_size,
            model.config.hidden_size,
            (model.config.hidden_size + 127) // 128,
        )
    else:
        workspace.ext.linear_bf16_resident_io(
            workspace.final_normalized.handle,
            resident.lm_head,
            0,
            workspace.final_logits.handle,
            1,
            model.config.vocab_size,
            model.config.hidden_size,
            False,
        )
    return workspace.final_logits.download()
