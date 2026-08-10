"""Opt-in tensor capture and deferred-RoPE helpers for Paper 1.5 probes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pra_torch.model import PositionAwareSelfAttention
from pra_torch.positions import RotaryPositionEncoding

from .position_policies import materialization_positions


@dataclass(frozen=True)
class AttentionCapture:
    """Intermediate tensors from one explicit self-attention operation.

    Q/K/V tensors are ``[B,H,T,Dh]``; logits and probabilities are
    ``[B,H,T,T]``; ``output`` is projected back to ``[B,T,D]``.
    """

    hidden: torch.Tensor
    q_raw: torch.Tensor
    q_positioned: torch.Tensor
    k_raw: torch.Tensor
    k_positioned: torch.Tensor
    value: torch.Tensor
    position_ids: torch.Tensor
    attention_logits: torch.Tensor
    attention_probabilities: torch.Tensor
    output: torch.Tensor


def capture_self_attention(
    attention: PositionAwareSelfAttention,
    hidden: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> AttentionCapture:
    """Run the explicit attention math while retaining research intermediates."""
    q_raw = attention._split_heads(attention.q_proj(hidden))
    k_raw = attention._split_heads(attention.k_proj(hidden))
    value = attention._split_heads(attention.v_proj(hidden))
    query, key = attention.position_encoding.transform_qk(q_raw, k_raw, position_ids)
    logits = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dim)
    tokens = hidden.shape[1]
    future = torch.triu(
        torch.ones(tokens, tokens, dtype=torch.bool, device=hidden.device),
        diagonal=1,
    )
    logits = logits.masked_fill(future, float("-inf"))
    if attention_mask is not None:
        logits = logits.masked_fill(
            ~attention_mask.to(device=hidden.device, dtype=torch.bool)[:, None, None, :],
            float("-inf"),
        )
    probabilities = F.softmax(logits, dim=-1)
    output = attention.o_proj(attention._merge_heads(probabilities @ value))
    return AttentionCapture(
        hidden=hidden,
        q_raw=q_raw,
        q_positioned=query,
        k_raw=k_raw,
        k_positioned=key,
        value=value,
        position_ids=position_ids,
        attention_logits=logits,
        attention_probabilities=probabilities,
        output=output,
    )


def materialize_raw_rope_key(
    raw_key: torch.Tensor,
    source_positions: torch.Tensor,
    query_position: int,
    *,
    policy: str,
    distance_limit: int,
    rope: RotaryPositionEncoding,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate raw ``[B,H,M,Dh]`` K under one experimental chunk policy."""
    assigned = materialization_positions(
        source_positions,
        query_position,
        policy,
        distance_limit=distance_limit,
    ).to(raw_key.device)
    return rope.apply_rotary(raw_key, assigned), assigned
