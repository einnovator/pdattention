from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "experiments/paper8_5_agent_memory/configs/cross_agent_model_replication_v1.json"
)
CORRECTED_MINI_SPEC = (
    ROOT
    / "experiments/paper8_5_agent_memory/configs/"
    / "transverse_mini_swe_tail90_v3_h2t4_corrected.json"
)


def test_cross_agent_model_replication_keeps_the_policy_transverse() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    policy = payload["logical_policy"]

    assert payload["status"] == "frozen_before_execution"
    assert policy == {
        "policy_id": "TAIL90_CAUSAL_H2T4_CORRECTED_V3",
        "supersedes": (
            "Legacy matched-tail runs whose labels recorded H2/T4 although the "
            "materializer did not consume those floors."
        ),
        "implementation": "matched_token_tail",
        "budget_fraction": 0.9,
        "protected_head_turns": 2,
        "protected_tail_turns": 4,
        "keep_all_genuine_user_prompts": True,
        "causal_group_atomicity": True,
        "negative_realization": "drop",
        "materialization": "ordinary_text_whole_record_except_natural_boundary_split",
        "agent_id_visible_to_policy": False,
        "task_id_visible_to_policy": False,
    }
    assert payload["generation"] == {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_completion_tokens": 1024,
    }


def test_cross_agent_model_replication_has_fixed_tasks_agents_and_models() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert payload["task_cohort"]["ordered_instance_ids"] == [
        "django__django-15277",
        "django__django-15368",
        "scikit-learn__scikit-learn-13135",
    ]
    assert payload["agent_replication"]["agents"] == [
        "mini-swe-agent@2.4.6",
        "openhands@1.49.2",
        "pi@0.75.3",
        "kilo@7.7.5",
        "pra-agent@locked-revision",
        "opencode@1.18.31",
    ]
    assert [row["id"] for row in payload["model_replication"]["models"]] == [
        "qwen3-coder:30b",
        "mlx-community/Qwen3-14B-4bit",
        "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    ]
    assert payload["agent_replication"]["repeats_per_admitted_arm"] == 2


def test_cross_agent_model_replication_fails_closed() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    admission = payload["agent_replication"]["admission_rule"]
    stop_conditions = set(payload["stop_conditions"])

    assert "FULL first" in admission
    assert "FULL qualification failure" in stop_conditions
    assert "model or tokenizer identity mismatch" in stop_conditions
    assert "incomplete native event trace" in stop_conditions
    assert "lost paired official success" in stop_conditions


def test_corrected_mini_campaign_has_distinct_policy_identity() -> None:
    payload = json.loads(CORRECTED_MINI_SPEC.read_text(encoding="utf-8"))

    assert payload["campaign_id"].endswith("corrected-h2t4")
    assert payload["history"]["head_turns"] == 2
    assert payload["history"]["tail_turns"] == 4
    assert payload["arms"][0]["arm_id"] == "matched_tail90_corrected_h2t4"
    assert "consume both" in payload["qualification"]["implementation_gate"]
