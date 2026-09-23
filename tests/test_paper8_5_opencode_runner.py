import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.run_opencode_swebench import (
    OPENCODE_DATA_PATH,
    PINNED_OPENCODE_VERSION,
    _has_native_stop_event,
    _load_tool_semantics,
    _native_session_id,
    _session_arguments,
    _validate_model_config,
    build_parser,
)


def test_opencode_version_and_controlled_defaults_are_pinned() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--instance-id", "django__django-15277",
        "--reference-trajectory", "trajectory.json",
        "--model-config", "opencode.json",
        "--output", "result",
        "--model-revision", "revision",
        "--tokenizer", "tokenizer",
        "--tokenizer-revision", "revision",
    ])
    assert PINNED_OPENCODE_VERSION == "1.18.31"
    assert args.agent_version == PINNED_OPENCODE_VERSION
    assert args.model == "pra/qwen3-coder:30b"
    assert args.served_model == "qwen3-coder:30b"
    assert args.agent == "build"
    assert args.native_stop_grace_seconds == 30.0


def test_native_stop_detection_requires_terminal_step_finish() -> None:
    payload = b"\n".join((
        b'{"type":"step_finish","part":{"reason":"tool-calls"}}',
        b'{"type":"step_finish","part":{"reason":"stop"}}',
    ))

    assert _has_native_stop_event(payload)
    assert not _has_native_stop_event(
        b'{"type":"step_finish","part":{"reason":"tool-calls"}}'
    )
    assert not _has_native_stop_event(b"not-json")


def test_native_session_identity_is_unique_and_explicit() -> None:
    payload = b"\n".join((
        b'{"type":"step_start","sessionID":"ses-1"}',
        b'{"type":"tool_use","sessionID":"ses-1"}',
    ))
    assert _native_session_id(payload) == "ses-1"
    assert _native_session_id(b'{"type":"text"}') is None
    with pytest.raises(ValueError, match="multiple root session IDs"):
        _native_session_id(b"\n".join((
            b'{"type":"step_start","sessionID":"ses-1"}',
            b'{"type":"step_start","sessionID":"ses-2"}',
        )))


def test_native_session_store_is_required_for_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="session-store"):
        _session_arguments(None, "ses-1")

    docker_args, opencode_args, store = _session_arguments(
        str(tmp_path / "native-session"), "ses-1"
    )
    assert store is not None and store.is_dir()
    assert docker_args == (
        "--mount",
        f"type=bind,source={store},target={OPENCODE_DATA_PATH}",
    )
    assert opencode_args == ("--session", "ses-1")


def test_model_config_requires_same_primary_and_small_model(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({
        "model": "pra/qwen3-coder:30b",
        "small_model": "pra/other-model",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="small_model"):
        _validate_model_config(path, "pra/qwen3-coder:30b")


def test_model_config_requires_explicit_zero_temperature_controls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opencode.json"
    payload = {
        "model": "pra/qwen3-coder:30b",
        "small_model": "pra/qwen3-coder:30b",
        "agent": {
            "build": {
                "model": "pra/qwen3-coder:30b",
                "temperature": 0.55,
                "top_p": 1.0,
                "seed": 0,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="freeze generation controls"):
        _validate_model_config(path, "pra/qwen3-coder:30b")

    payload["agent"]["build"]["temperature"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_model_config(path, "pra/qwen3-coder:30b")["agent"][
        "build"
    ]["temperature"] == 0.0


def test_opencode_tool_semantics_are_canonical_not_agent_policy() -> None:
    path = (
        Path(__file__).parents[1]
        / "experiments" / "paper8_5_agent_memory" / "configs"
        / "opencode_tool_semantics_v1.json"
    )
    semantics = _load_tool_semantics(path)
    assert semantics["read"]["operation_kind"] == "read"
    assert semantics["edit"]["operation_kind"] == "write"
    assert semantics["bash"]["operation_kind"] == "unknown"
    assert "agent_id" not in json.dumps(semantics)


def test_ctx131k_model_config_is_consistent_and_frozen() -> None:
    path = (
        Path(__file__).parents[1]
        / "experiments" / "paper8_5_agent_memory" / "configs"
        / "opencode_qwen3_coder_30b_ctx131k_medium_mac.json"
    )
    model = "pra/qwen3-coder:30b-ctx131k"

    payload = _validate_model_config(path, model)

    assert payload["provider"]["pra"]["models"][
        "qwen3-coder:30b-ctx131k"
    ]["limit"]["context"] == 131072


def test_ctx64k_model_config_is_consistent_and_frozen() -> None:
    path = (
        Path(__file__).parents[1]
        / "experiments" / "paper8_5_agent_memory" / "configs"
        / "opencode_qwen3_coder_30b_ctx64k_medium_mac.json"
    )
    model = "pra/qwen3-coder:30b-ctx64k"

    payload = _validate_model_config(path, model)

    assert payload["provider"]["pra"]["models"][
        "qwen3-coder:30b-ctx64k"
    ]["limit"]["context"] == 65536
