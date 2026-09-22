import argparse

import pytest

from experiments.paper8_5_agent_memory.model_identity import validate_ollama_tags
from experiments.paper8_5_agent_memory.run_pi_swebench import observed_model_identity


DIGEST = "06c1097efce0431c2045fe7b2e5108366e43bee1b4603a7aded8f21689e90bca"


def _tags():
    return {
        "models": [{
            "name": "qwen3-coder:30b",
            "model": "qwen3-coder:30b",
            "digest": DIGEST,
            "details": {
                "format": "gguf",
                "family": "qwen3moe",
                "parameter_size": "30.5B",
                "quantization_level": "Q4_K_M",
                "context_length": 262144,
            },
        }]
    }


def test_observed_endpoint_digest_is_bound() -> None:
    result = validate_ollama_tags(
        _tags(), expected_model="qwen3-coder:30b", expected_revision=DIGEST
    )
    assert result["revision"] == DIGEST
    assert result["quantization_level"] == "Q4_K_M"


def test_alias_with_different_digest_fails_closed() -> None:
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_ollama_tags(
            _tags(), expected_model="qwen3-coder:30b", expected_revision="stale"
        )


def test_missing_model_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        validate_ollama_tags(
            _tags(), expected_model="qwen3-coder:14b", expected_revision=DIGEST
        )


def test_typed_agent_identity_binding_is_opt_in_but_atomic() -> None:
    assert observed_model_identity(argparse.Namespace(model="m")) is None
    with pytest.raises(ValueError, match="must be supplied together"):
        observed_model_identity(argparse.Namespace(
            model="m", ollama_tags_url="http://endpoint/api/tags",
            model_revision=None,
        ))
