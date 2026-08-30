from experiments.paper6_1_sglang.run_live_storage_lifecycle import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    _resolve_revision,
)


def test_revision_defaults_are_model_aware() -> None:
    assert _resolve_revision(DEFAULT_MODEL, None) == DEFAULT_REVISION
    assert _resolve_revision("mlx-community/Qwen3-1.7B-4bit", None) == "main"
    assert _resolve_revision(DEFAULT_MODEL, "custom") == "custom"
