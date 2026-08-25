"""rope.py — YARN-scaled RoPE cos/sin computation for gpt-oss.

Delegates to transformers' `_compute_yarn_parameters` for the frequency
math (well-tested, matches the model exactly) then duplicates cos/sin so
they can be consumed by our `rope_partial` shader which expects
[S, rotary_dim] (not [S, rotary_dim/2] like transformers).

Later this can be reimplemented from scratch to remove the transformers
dependency; keeping it as a thin wrapper for now to guarantee correctness.
"""
from __future__ import annotations
import torch


def compute_cos_sin_for_positions(
    hf_config,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cos, sin for the given positions using the model's YARN
    scaling. Returns tensors of shape [S, rotary_dim] where rotary_dim
    equals head_dim (gpt-oss has no partial rotary).

    positions: [S] int64.
    hf_config: a transformers AutoConfig for the model (has .head_dim etc).
    """
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

    rot = GptOssRotaryEmbedding(hf_config)
    S = positions.shape[0]
    # rot.forward wants (x, position_ids). x is only used for its device, so
    # any dummy tensor works.
    dummy = torch.zeros(1, S, hf_config.hidden_size)
    pos_ids = positions.unsqueeze(0)   # [1, S]
    cos, sin = rot(dummy, pos_ids)      # each [1, S, head_dim/2]

    # Duplicate to full head_dim: cos_full[d < D/2] = cos[d], cos_full[d >= D/2] = cos[d - D/2].
    # Same for sin. This is what transformers'
    # `_apply_rotary_emb` implicitly does via torch.chunk + broadcast, and it's
    # what our rope_partial shader explicitly indexes.
    cos = cos.squeeze(0)                # [S, head_dim/2]
    sin = sin.squeeze(0)                # [S, head_dim/2]
    cos_full = torch.cat([cos, cos], dim=-1)   # [S, head_dim]
    sin_full = torch.cat([sin, sin], dim=-1)   # [S, head_dim]
    return cos_full.contiguous(), sin_full.contiguous()
