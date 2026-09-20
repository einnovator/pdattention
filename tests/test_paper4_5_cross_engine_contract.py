from copy import deepcopy

import pytest

from experiments.paper4_5_agent.cross_engine_contract import (
    CrossEngineContractError,
    validate_cross_engine_cells,
)


def _cell() -> dict:
    return {
        "identity": {
            "agent_id": "mini-swe-agent",
            "agent_revision": "2.4.6",
            "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "model_revision": "checkpoint-sha256",
            "observed_source_checkpoint_revision": "checkpoint-sha256",
            "precision": "BF16",
            "tokenizer_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "tokenizer_revision": "checkpoint-sha256",
            "chat_template_digest": "template-sha256",
            "task_id": "django__django-15277",
            "task_snapshot_digest": "snapshot-sha256",
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_completion_tokens": 1024,
        },
        "logical_plan": {
            "history_digest": "history-sha256",
            "plan_digest": "plan-sha256",
            "selected_record_ids": ["system", "task", "turn-6"],
            "full_logical_tokens": 10000,
            "selected_logical_tokens": 6000,
        },
        "physical": {"selected_kv_tokens": 6144},
    }


def test_physical_rounding_may_differ_but_logical_saving_cannot() -> None:
    hf = _cell()
    vllm = deepcopy(hf)
    vllm["physical"]["selected_kv_tokens"] = 6400
    result = validate_cross_engine_cells({"hf": hf, "vllm": vllm})
    assert result["logical_saving_fraction"] == pytest.approx(0.4)


def test_model_change_is_not_an_engine_comparison() -> None:
    hf = _cell()
    mlx = deepcopy(hf)
    mlx["identity"]["model_id"] = "Qwen/Qwen3-Coder-30B"
    with pytest.raises(CrossEngineContractError, match="model_id"):
        validate_cross_engine_cells({"hf": hf, "mlx": mlx})


def test_observed_checkpoint_change_is_not_an_engine_comparison() -> None:
    hf = _cell()
    mlx = deepcopy(hf)
    mlx["identity"]["observed_source_checkpoint_revision"] = "converted-alias"
    with pytest.raises(
        CrossEngineContractError, match="observed_source_checkpoint_revision"
    ):
        validate_cross_engine_cells({"hf": hf, "mlx": mlx})


def test_changed_selection_is_not_an_engine_effect() -> None:
    hf = _cell()
    mlx = deepcopy(hf)
    mlx["logical_plan"]["selected_record_ids"] = ["system", "task", "turn-7"]
    with pytest.raises(CrossEngineContractError, match="selected_record_ids"):
        validate_cross_engine_cells({"hf": hf, "mlx": mlx})
