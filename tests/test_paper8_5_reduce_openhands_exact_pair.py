from __future__ import annotations

import json
from pathlib import Path

from experiments.paper8_5_agent_memory.reduce_openhands_exact_pair import reduce_pair


def _arm(
    path: Path, *, resolved: bool, materialized: int,
    instance: str = "x__y-1", agent: str | None = None,
    summary_key: str = "action_count", initial_request: str = "request-1",
):
    path.mkdir()
    manifest = {
        "instance_id": instance, "served_model": "model", "model_revision": "rev",
        "tokenizer_revision": "rev", "reference_trajectory_sha256": "trajectory",
        "source_image": "image", "patch_bytes": 3, "patch_sha256": "patch",
    }
    if agent is not None:
        manifest.update({"agent": agent, "agent_version": "1.0"})
    (path / "run_manifest.json").write_text(json.dumps(manifest))
    (path / "event_summary.json").write_text(json.dumps({summary_key: 2}))
    (path / "official_report.json").write_text(json.dumps({
        "resolved_ids": [instance] if resolved else [],
    }))
    rows = []
    for index in (1, 2):
        rows.append({
            "request_index": index, "tokenizer": "exact@rev", "full_tokens": 100,
            "selected_tokens": materialized, "materialized_tokens": materialized,
            "reported_prompt_tokens": 150,
            "reported_completion_tokens": 7,
            "generation": {"max_completion_tokens": 8},
            "logical_retention_fraction": materialized / 100,
            "requested_budget_tokens": 90,
            "materialized_budget_unused_tokens": max(0, 90 - materialized),
            "request_input_sha256": (
                initial_request if index == 1 else f"request-{index}"
            ),
            "exact_request_passthrough": index == 1,
        })
    (path / "proxy_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )


def test_exact_pair_reducer_separates_own_and_paired_saving(tmp_path: Path):
    full = tmp_path / "full"
    candidate = tmp_path / "candidate"
    _arm(full, resolved=True, materialized=100)
    _arm(candidate, resolved=True, materialized=75)

    result = reduce_pair(
        full, candidate, policy="matched_token_tail", budget_fraction=0.9,
        protected_head_turns=2, protected_tail_turns=4,
    )

    assert result["qualification"] == "pass"
    assert result["candidate"]["own_logical_saving"] == 0.25
    assert result["candidate"]["own_materialized_saving"] == 0.25
    assert result["paired_input_saving"] == 0.25
    assert result["identity_mismatches"] == {}


def test_exact_pair_reducer_fails_identity_mismatch(tmp_path: Path):
    full = tmp_path / "full"
    candidate = tmp_path / "candidate"
    _arm(full, resolved=True, materialized=100)
    _arm(candidate, resolved=True, materialized=75, instance="x__y-2")

    result = reduce_pair(
        full, candidate, policy="matched_token_tail", budget_fraction=0.9,
        protected_head_turns=2, protected_tail_turns=4,
    )

    assert result["qualification"] == "fail"
    assert "instance_id" in result["identity_mismatches"]


def test_exact_pair_reducer_accepts_typed_agent_call_counter(tmp_path: Path):
    full = tmp_path / "full"
    candidate = tmp_path / "candidate"
    _arm(
        full, resolved=True, materialized=100, agent="pi",
        summary_key="assistant_model_calls",
    )
    _arm(
        candidate, resolved=True, materialized=80, agent="pi",
        summary_key="assistant_model_calls",
    )

    result = reduce_pair(
        full, candidate, policy="matched_token_tail", budget_fraction=0.9,
        protected_head_turns=2, protected_tail_turns=4,
    )

    assert result["qualification"] == "pass"
    assert result["agent"] == "pi"
    assert result["agent_version"] == "1.0"
    assert result["full"]["actions"] == 2


def test_exact_pair_reducer_rejects_different_initial_model_input(tmp_path: Path):
    full = tmp_path / "full"
    candidate = tmp_path / "candidate"
    _arm(full, resolved=True, materialized=100, initial_request="full-request")
    _arm(
        candidate, resolved=True, materialized=75,
        initial_request="candidate-request",
    )

    result = reduce_pair(
        full, candidate, policy="matched_token_tail", budget_fraction=0.9,
        protected_head_turns=2, protected_tail_turns=4,
    )

    assert result["qualification"] == "fail"
    assert "initial_request_sha256" in result["identity_mismatches"]
