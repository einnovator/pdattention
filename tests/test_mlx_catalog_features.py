from types import SimpleNamespace

import pytest

from experiments.paper4_5_runtime.extract_mlx_catalog_features import _resolve_decoder
from pra_hf.onboarding import KNOWN_STRUCTURAL_MAPPINGS


def test_decoder_resolution_supports_conventional_mlx_models() -> None:
    decoder = SimpleNamespace(layers=[object()])

    assert _resolve_decoder(SimpleNamespace(model=decoder)) is decoder


def test_decoder_resolution_supports_qwen35_nested_language_model() -> None:
    decoder = SimpleNamespace(layers=[object()])
    model = SimpleNamespace(language_model=SimpleNamespace(model=decoder))

    assert _resolve_decoder(model) is decoder


def test_decoder_resolution_rejects_unknown_layout() -> None:
    with pytest.raises(TypeError, match="decoder layer stack"):
        _resolve_decoder(SimpleNamespace())


def test_qwen35_mapping_does_not_overclaim_native_kv() -> None:
    mapping = KNOWN_STRUCTURAL_MAPPINGS["qwen3_5"]

    assert mapping["status"] == "PARTIAL_TOPOLOGY"
    assert mapping["attention"] == "full_attention_only"
    assert mapping["native_kv"] is False
