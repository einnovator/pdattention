"""Decoder-stack discovery shared by Hugging Face PRA family adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HFDecoderAnatomy:
    """Resolved text decoder behind a causal or multimodal HF wrapper."""

    decoder: Any
    layers: Any
    rotary_embedding: Any
    config: Any
    path: str


def resolve_decoder_anatomy(model) -> HFDecoderAnatomy:
    """Locate one supported text decoder and its actual configuration."""

    root = getattr(model, "model", None)
    if root is None:
        raise TypeError("Expected a Hugging Face model exposing a model module.")
    if hasattr(root, "layers") and hasattr(root, "rotary_emb"):
        return HFDecoderAnatomy(root, root.layers, root.rotary_emb, model.config, "model")
    language_model = getattr(root, "language_model", None)
    if (
        language_model is not None
        and hasattr(language_model, "layers")
        and hasattr(language_model, "rotary_emb")
    ):
        text_config = getattr(model.config, "text_config", language_model.config)
        return HFDecoderAnatomy(
            language_model,
            language_model.layers,
            language_model.rotary_emb,
            text_config,
            "model.language_model",
        )
    raise TypeError(
        "Expected a supported Hugging Face text decoder at model.layers or "
        "model.language_model.layers."
    )
