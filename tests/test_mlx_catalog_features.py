import json
from types import SimpleNamespace

import pytest

from experiments.paper4_5_runtime.extract_mlx_catalog_features import _resolve_decoder
from pra_hf.onboarding import KNOWN_STRUCTURAL_MAPPINGS, ModelInspector


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


def test_inspector_falls_back_to_raw_config_for_new_model_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    config = tmp_path / "snapshots" / "abc123" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "layer_types": ["linear_attention", "full_attention"],
        },
    }), encoding="utf-8")

    def unsupported(*args, **kwargs):
        raise ValueError("Transformers does not recognize this architecture")

    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", unsupported)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(config))

    result = ModelInspector().inspect("example/qwen35")

    assert result["model"]["revision"] == "abc123"
    assert result["attention"] == {
        "layers": 64,
        "query_heads": 24,
        "kv_heads": 4,
        "head_dim": 256,
        "topology": "heterogeneous",
        "position_encoding": "partial_rope",
    }
    assert result["pra"]["native_kv"] is False
