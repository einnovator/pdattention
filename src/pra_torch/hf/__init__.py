"""Thin Hugging Face adapters backed by the shared PRA execution core."""

from .config import (
    ATTENTION_INPUT_HIDDEN_STATE,
    PRAHFConfig,
    canonical_routing_representation,
)
from .injection import PRAHFModel, inject_pra

__all__ = [
    "ATTENTION_INPUT_HIDDEN_STATE",
    "PRAHFConfig",
    "PRAHFModel",
    "canonical_routing_representation",
    "inject_pra",
]
