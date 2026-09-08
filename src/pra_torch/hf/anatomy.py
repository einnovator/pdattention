"""Decoder-stack discovery shared by Hugging Face PRA family adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HFDecoderAnatomy:
    """Resolved text-decoder modules behind a causal or multimodal HF wrapper."""

    decoder: Any
    layers: Any
    rotary_embedding: Any
    config: Any
    path: str


def resolve_decoder_anatomy(model) -> HFDecoderAnatomy:
    """Locate the text decoder without confusing vision and language configs.

    Ordinary causal LMs expose ``model.model.layers``. Mistral 3's conditional
    generation wrapper adds a multimodal shell and stores the same Mistral text
    decoder at ``model.model.language_model``. Keeping this lookup centralized
    makes layer selection, RoPE ownership, and expert-isolation checks agree.
    """

    root = getattr(model, "model", None)
    if root is None:
        raise TypeError("Expected a Hugging Face model exposing a model module.")

    if hasattr(root, "layers") and hasattr(root, "rotary_emb"):
        return HFDecoderAnatomy(
            decoder=root,
            layers=root.layers,
            rotary_embedding=root.rotary_emb,
            config=model.config,
            path="model",
        )

    language_model = getattr(root, "language_model", None)
    if (
        language_model is not None
        and hasattr(language_model, "layers")
        and hasattr(language_model, "rotary_emb")
    ):
        text_config = getattr(model.config, "text_config", language_model.config)
        return HFDecoderAnatomy(
            decoder=language_model,
            layers=language_model.layers,
            rotary_embedding=language_model.rotary_emb,
            config=text_config,
            path="model.language_model",
        )

    raise TypeError(
        "Expected a supported Hugging Face text decoder at model.layers or "
        "model.language_model.layers."
    )
