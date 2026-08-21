"""Model-native query representations for adaptive PRA effort routing.

The helpers in this module keep the serving boundary explicit: a short query
prefill may inform an effort decision, but it may not inspect evaluator labels
or post-search state.  Paper-specific controller training and evaluation live
under ``experiments/``; this module only captures and accounts for observable
representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
from torch import nn

from .factorized_control import FactorizedEffortAction


PoolingMode = Literal["mean", "last", "max"]


@dataclass(frozen=True)
class QueryPrefillAccounting:
    """Token/layer work charged to a query-routing prepass.

    ``processed_token_layers`` is the direct product of prompt tokens and
    executed decoder layers.  ``normalized_cost`` expresses that work in the
    same 64-token/full-depth units used by the Paper-3.5 abstract cost model.
    Reused work remains visible but contributes no *additional* execution cost.
    """

    prompt_tokens: int
    query_tokens: int
    layers_executed: int
    total_layers: int
    reused: bool = False

    def __post_init__(self) -> None:
        if self.prompt_tokens <= 0 or self.query_tokens <= 0:
            raise ValueError("Prompt and query token counts must be positive.")
        if self.query_tokens > self.prompt_tokens:
            raise ValueError("The query region cannot exceed the encoded prompt.")
        if not 0 <= self.layers_executed <= self.total_layers or self.total_layers <= 0:
            raise ValueError("Executed depth must lie within the backbone depth.")

    @property
    def processed_token_layers(self) -> int:
        return self.prompt_tokens * self.layers_executed

    @property
    def recomputed_query_tokens(self) -> int:
        return 0 if self.reused else self.prompt_tokens

    @property
    def normalized_cost(self) -> float:
        if self.reused:
            return 0.0
        # Embedding-only routing still reads the prompt once.  Treat that read
        # as one layer-equivalent rather than incorrectly assigning zero cost.
        depth = max(1, self.layers_executed)
        return self.prompt_tokens * depth / (64.0 * self.total_layers)


@dataclass(frozen=True)
class QwenPrefixState:
    """Intermediate Qwen decoder state that can continue without recomputation."""

    hidden_states: torch.Tensor
    attention_masks: Mapping[str, torch.Tensor | None]
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    depth: int


def query_span_mask(
    sequence_length: int,
    start: int,
    end: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a ``[1, T]`` mask selecting the half-open query span ``[start,end)``."""

    if sequence_length <= 0 or not 0 <= start < end <= sequence_length:
        raise ValueError("Query span must be non-empty and lie inside the sequence.")
    mask = torch.zeros((1, sequence_length), dtype=torch.bool, device=device)
    mask[:, start:end] = True
    return mask


def pool_query_tokens(
    states: torch.Tensor,
    query_mask: torch.Tensor,
    mode: PoolingMode,
) -> torch.Tensor:
    """Pool query tokens from ``states [B,T,...]`` into ``[B,...]``.

    Native Q/K tensors may include head and head-width axes after ``T``; the
    token mask is expanded across those axes without flattening head geometry.
    """

    if states.ndim < 3 or query_mask.ndim != 2 or states.shape[:2] != query_mask.shape:
        raise ValueError("Expected states [B,T,...] and a matching query mask [B,T].")
    if not torch.all(query_mask.any(dim=1)):
        raise ValueError("Every batch row must select at least one query token.")
    expanded = query_mask.view(*query_mask.shape, *([1] * (states.ndim - 2)))
    if mode == "mean":
        count = query_mask.sum(dim=1).view(states.shape[0], *([1] * (states.ndim - 2)))
        return (states * expanded).sum(dim=1) / count.to(states.dtype)
    if mode == "max":
        minimum = torch.finfo(states.dtype).min
        return states.masked_fill(~expanded, minimum).amax(dim=1)
    if mode == "last":
        indices = query_mask.long().sum(dim=1) - 1
        starts = query_mask.long().argmax(dim=1)
        absolute = starts + indices
        batch = torch.arange(states.shape[0], device=states.device)
        return states[batch, absolute]
    raise ValueError(f"Unsupported query pooling mode: {mode!r}.")


def native_qk_representation(
    causal_lm: nn.Module,
    layer_input: torch.Tensor,
    layer_index: int,
    query_mask: torch.Tensor,
    *,
    kind: Literal["q", "k", "qk"] = "q",
    pooling: PoolingMode = "mean",
) -> torch.Tensor:
    """Project and pool native pre-RoPE Q/K at one decoder layer.

    ``layer_input`` is the residual stream entering ``layer_index`` with shape
    ``[B,T,D]``.  Qwen applies its input layer norm before native projections;
    this function follows that exact path and returns flattened pooled heads.
    """

    decoder = getattr(causal_lm, "model", causal_lm)
    layers = getattr(decoder, "layers", None)
    if layers is None or not 0 <= layer_index < len(layers):
        raise ValueError("Native Q/K capture requires a valid decoder layer index.")
    layer = layers[layer_index]
    attention = getattr(layer, "self_attn", None)
    if attention is None or not hasattr(layer, "input_layernorm"):
        raise TypeError("Decoder layer does not expose Qwen-style native projections.")
    normalized = layer.input_layernorm(layer_input)
    batch, tokens, _ = normalized.shape
    head_dim = int(attention.head_dim)
    q = attention.q_proj(normalized).view(batch, tokens, -1, head_dim)
    k = attention.k_proj(normalized).view(batch, tokens, -1, head_dim)
    # Qwen3 applies per-head RMS normalization before RoPE.  Capturing the raw
    # linear projection would not be the geometry consumed by native attention.
    if hasattr(attention, "q_norm"):
        q = attention.q_norm(q)
    if hasattr(attention, "k_norm"):
        k = attention.k_norm(k)
    pooled_q = pool_query_tokens(q, query_mask, pooling).flatten(1)
    pooled_k = pool_query_tokens(k, query_mask, pooling).flatten(1)
    if kind == "q":
        return pooled_q
    if kind == "k":
        return pooled_k
    if kind == "qk":
        return torch.cat((pooled_q, pooled_k), dim=1)
    raise ValueError(f"Unsupported native representation kind: {kind!r}.")


def reuse_is_semantically_valid(
    *,
    depth: int,
    first_memory_layer: int,
    ordinary_context_precedes_query: bool,
    query_region_changed: bool = False,
) -> bool:
    """Whether a query prefill can seed normal PRA execution exactly.

    Reuse must stop no later than the first layer that consumes PRA memory.  A
    separately encoded query also cannot replace a state that should have
    attended to preceding ordinary context, nor survive query reinterpretation.
    """

    if depth < 0 or first_memory_layer < 0:
        raise ValueError("Layer depths must be non-negative.")
    return (
        depth <= first_memory_layer
        and not ordinary_context_precedes_query
        and not query_region_changed
    )


def qwen_prefill_prefix(
    causal_lm: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    stop_layer: int,
) -> QwenPrefixState:
    """Run Qwen through ``stop_layer`` and retain an exact continuation state.

    This path intentionally disables the autoregressive cache.  It is used for
    query-prefill reuse validation and for implementations that pause before a
    late PRA consumer layer, select memory, and then resume the same forward.
    """

    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    decoder = getattr(causal_lm, "model", causal_lm)
    if not 0 <= stop_layer <= len(decoder.layers):
        raise ValueError("stop_layer lies outside the decoder depth.")
    hidden = decoder.embed_tokens(input_ids)
    cache_position = torch.arange(hidden.shape[1], device=hidden.device)
    position_ids = cache_position.unsqueeze(0)
    mask_kwargs = {
        "config": decoder.config,
        "input_embeds": hidden,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks: dict[str, torch.Tensor | None] = {
        "full_attention": create_causal_mask(**mask_kwargs)
    }
    if decoder.has_sliding_layers:
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    position_embeddings = decoder.rotary_emb(hidden, position_ids)
    for layer in decoder.layers[:stop_layer]:
        hidden = layer(
            hidden,
            attention_mask=masks[layer.attention_type],
            position_ids=position_ids,
            past_key_value=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    return QwenPrefixState(
        hidden,
        masks,
        position_ids,
        cache_position,
        position_embeddings,
        stop_layer,
    )


def qwen_prefill_continue(causal_lm: nn.Module, state: QwenPrefixState) -> torch.Tensor:
    """Resume a state from :func:`qwen_prefill_prefix` through final norm."""

    decoder = getattr(causal_lm, "model", causal_lm)
    if not 0 <= state.depth <= len(decoder.layers):
        raise ValueError("Prefix depth is incompatible with the decoder.")
    hidden = state.hidden_states
    for layer in decoder.layers[state.depth :]:
        hidden = layer(
            hidden,
            attention_mask=state.attention_masks[layer.attention_type],
            position_ids=state.position_ids,
            past_key_value=None,
            use_cache=False,
            cache_position=state.cache_position,
            position_embeddings=state.position_embeddings,
        )
    return decoder.norm(hidden)


def decode_grouped_action(
    interpret: int,
    search: Sequence[int],
    admit: int,
) -> tuple[FactorizedEffortAction, bool]:
    """Compose ``F``, coherent ``(R,K,H,B_search)``, and ``B_KV`` groups."""

    if len(search) != 4:
        raise ValueError("Search profile must contain R, K, H, and B_search.")
    roots, neighbors, hops, search_budget = map(int, search)
    repaired_kv = max(int(admit), roots)
    repaired_search = max(int(search_budget), roots)
    repaired = repaired_kv != int(admit) or repaired_search != int(search_budget)
    return (
        FactorizedEffortAction(
            int(interpret),
            roots,
            neighbors,
            hops,
            repaired_search,
            repaired_kv,
        ),
        repaired,
    )


class ValidationProjector:
    """Validation-fitted PCA projection with an explicit leakage guard."""

    def __init__(self, width: int = 16) -> None:
        if width <= 0:
            raise ValueError("Projection width must be positive.")
        self.width = width
        self.mean: torch.Tensor | None = None
        self.components: torch.Tensor | None = None

    def fit(self, rows: torch.Tensor, partitions: Sequence[str]) -> "ValidationProjector":
        if rows.ndim != 2 or len(rows) != len(partitions):
            raise ValueError("Projection rows and partition labels must align.")
        if not rows.shape[0] or set(partitions) != {"validation"}:
            raise ValueError("Representation projection may fit only validation rows.")
        values = rows.float()
        self.mean = values.mean(dim=0)
        centered = values - self.mean
        _, _, right = torch.linalg.svd(centered, full_matrices=False)
        rank = min(self.width, right.shape[0], max(1, rows.shape[0] - 1))
        self.components = right[:rank]
        return self

    def transform(self, rows: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.components is None:
            raise RuntimeError("ValidationProjector must be fitted before transform.")
        if rows.ndim != 2 or rows.shape[1] != self.mean.numel():
            raise ValueError("Representation width does not match the fitted projector.")
        projected = (rows.float() - self.mean) @ self.components.T
        return torch.nn.functional.normalize(projected, dim=1)
