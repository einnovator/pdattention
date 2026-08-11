"""Thin Hugging Face adapters backed by the shared PRA execution core."""

from .config import (
    ATTENTION_INPUT_HIDDEN_STATE,
    CENTERED_ROPE_KEY,
    PRAHFConfig,
    canonical_routing_representation,
)
from .injection import PRAHFModel, inject_pra
from .query import (
    QUERY_EXPONENTIAL,
    QUERY_LAST,
    QUERY_LINEAR,
    QUERY_QUESTION_EXPONENTIAL,
    QUERY_QUESTION_MEAN,
    QUERY_STRATEGIES,
    QUERY_UNIFORM,
    aggregate_query_states,
    half_life_to_decay,
    streaming_exponential_query,
    token_span_from_offsets,
)

__all__ = [
    "ATTENTION_INPUT_HIDDEN_STATE",
    "CENTERED_ROPE_KEY",
    "PRAHFConfig",
    "PRAHFModel",
    "canonical_routing_representation",
    "inject_pra",
    "QUERY_EXPONENTIAL",
    "QUERY_LAST",
    "QUERY_LINEAR",
    "QUERY_QUESTION_EXPONENTIAL",
    "QUERY_QUESTION_MEAN",
    "QUERY_STRATEGIES",
    "QUERY_UNIFORM",
    "aggregate_query_states",
    "half_life_to_decay",
    "streaming_exponential_query",
    "token_span_from_offsets",
]
