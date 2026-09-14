import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.run_autonomous_curve_campaign import (
    _retry_path,
    adaptive_gate,
    campaign_cells,
    import_completed_controls,
    validate_campaign_spec,
    write_curve_spec,
)
from experiments.paper8_5_agent_memory.run_autonomous_swebench import _official_report


ROOT = Path(__file__).resolve().parents[1]


def _spec():
    return {
        "campaign_id": "campaign",
        "generation": {"seed": 0, "max_completion_tokens": 1024},
        "controls": {"full_repeats": 2},
        "arms": [
            {
                "arm_id": "dag100",
                "policy": "task_aware_progress_spine_v4",
                "negative_realization": "protocol_stub",
                "negative_fallback": "none",
                "budget_fraction": 1.0,
            }
        ],
    }


def test_campaign_locks_two_controls_then_runs_arm_major():
    cells = campaign_cells(_spec(), ("a", "b"))
    assert [row["cell_id"] for row in cells] == [
        "task01-full-a", "task01-full-b", "task02-full-a", "task02-full-b",
        "task01-dag100", "task02-dag100",
    ]
    assert cells[0]["pair_id"] == cells[4]["pair_id"]


def test_campaign_can_share_a_frozen_pair_id_with_corrected_policy_code():
    spec = _spec()
    spec["campaign_id"] = "corrected"
    spec["baseline_pair_campaign_id"] = "baseline"
    cells = campaign_cells(spec, ("task",))
    assert {row["pair_id"] for row in cells} == {"baseline:task:seed0"}


def test_imported_controls_are_bound_and_emit_explicit_revision_exception(tmp_path):
    spec = _spec()
    spec["campaign_id"] = "corrected"
    spec["baseline_pair_campaign_id"] = "baseline"
    cells = campaign_cells(spec, ("task",))
    source_cells = {}
    for cell in cells[:2]:
        output = tmp_path / cell["cell_id"]
        output.mkdir()
        (output / "run_manifest.json").write_text("{}")
        (output / "autonomous_metrics.json").write_text(json.dumps({
            "official_result": {"resolved": True, "error": False},
            "calls": 1,
            "cumulative_full_tokens": 10,
            "cumulative_materialized_tokens": 10,
        }))
        source_cells[cell["cell_id"]] = {
            **cell, "status": "complete", "output": str(output),
        }
    source_state = tmp_path / "source_state.json"
    source_state.write_text(json.dumps({
        "campaign_id": "baseline", "cells": source_cells,
    }))
    state = {"cells": {}}

    assert import_completed_controls(
        state, cells, source_state, expected_campaign_id="baseline"
    ) == 2
    state["cells"]["task01-dag100"] = {
        **cells[2], "status": "complete", "output": str(tmp_path / "candidate"),
    }
    value = json.loads(write_curve_spec(state, tmp_path / "curve").read_text())
    policy_pair = next(
        row for row in value["autonomous_pairs"]
        if row["candidate"] == "task01-dag100"
    )
    assert policy_pair["allow_legacy_pair"] is True
    assert "implementation revision changed" in policy_pair["pairing_reason"]


def test_campaign_rejects_unbounded_autonomous_generation():
    spec = _spec()
    spec["generation"]["max_completion_tokens"] = None
    with pytest.raises(ValueError, match="positive, frozen max_completion_tokens"):
        validate_campaign_spec(spec)

    validate_campaign_spec(_spec())


def test_adaptive_gate_stops_two_failures():
    result = adaptive_gate(
        [
            {"status": "complete", "official_resolved": False, "gross_saving_fraction": .2},
            {"status": "complete", "official_resolved": False, "gross_saving_fraction": .2},
        ],
        {"stop_after_failures": 2},
    )
    assert result[0] == "stop"


def test_adaptive_gate_counts_definitive_grader_errors_as_task_failures():
    result = adaptive_gate(
        [
            {"status": "complete", "official_resolved": False,
             "official_error": True, "gross_saving_fraction": .2},
            {"status": "complete", "official_resolved": False,
             "official_error": True, "gross_saving_fraction": .2},
        ],
        {"stop_after_failures": 2, "minimum_tasks_before_accuracy_gate": 1},
    )
    assert result[0] == "stop"


def test_official_report_classifies_patch_apply_failure(tmp_path):
    report = {
        "submitted_ids": ["task"],
        "resolved_ids": [],
        "error_ids": ["task"],
    }
    (tmp_path / "model.run.json").write_text(json.dumps(report), encoding="utf-8")
    grader_log = tmp_path / "grader.log"
    grader_log.write_text(
        "task: >>>>> Patch Apply Failed: invalid patch", encoding="utf-8"
    )

    result = _official_report(
        tmp_path, "run", "task", grader_log=grader_log,
    )

    assert result["resolved"] is False
    assert result["error"] is True
    assert result["failure_class"] == "patch_apply_failed"


def test_official_report_recovers_exact_pre_report_patch_apply_failure(tmp_path):
    grader_log = tmp_path / "grader.log"
    grader_log.write_text(
        "task: >>>>> Patch Apply Failed:\n"
        "patch: **** Only garbage was found in the patch input.\n",
        encoding="utf-8",
    )

    result = _official_report(
        tmp_path, "run", "task", grader_log=grader_log,
    )

    assert result == {
        "official_grader": True,
        "instance_id": "task",
        "resolved": False,
        "score": 0.0,
        "error": True,
        "failure_class": "patch_apply_failed",
        "raw_report": str(grader_log),
        "raw_report_kind": "per_instance_failure_log",
    }


def test_adaptive_gate_stops_low_yield_and_low_accuracy():
    thresholds = {
        "stop_after_failures": 5,
        "minimum_tasks_before_yield_gate": 2,
        "minimum_gross_saving_fraction": .02,
        "minimum_tasks_before_accuracy_gate": 3,
        "minimum_task_resolution": .8,
    }
    assert adaptive_gate([
        {"status": "complete", "official_resolved": True, "gross_saving_fraction": .01},
        {"status": "complete", "official_resolved": True, "gross_saving_fraction": .01},
    ], thresholds)[0] == "stop"
    assert adaptive_gate([
        {"status": "complete", "official_resolved": True, "gross_saving_fraction": .2},
        {"status": "complete", "official_resolved": True, "gross_saving_fraction": .2},
        {"status": "complete", "official_resolved": False, "gross_saving_fraction": .2},
    ], thresholds)[0] == "stop"


def test_curve_spec_pairs_candidates_only_after_two_controls(tmp_path):
    state = {"cells": {
        "a": {"status": "complete", "instance_id": "task", "is_control": True,
              "arm_id": "full_a", "output": str(tmp_path / "a")},
        "b": {"status": "complete", "instance_id": "task", "is_control": True,
              "arm_id": "full_b", "output": str(tmp_path / "b")},
        "c": {"status": "complete", "instance_id": "task", "is_control": False,
              "arm_id": "dag", "output": str(tmp_path / "c")},
    }}
    path = write_curve_spec(state, tmp_path)
    value = json.loads(path.read_text())
    assert {row["candidate"] for row in value["autonomous_pairs"]} == {"b", "c"}
    assert all(row["baseline"] == "a" for row in value["autonomous_pairs"])


def test_retry_path_preserves_incomplete_attempt(tmp_path):
    base = tmp_path / "arm"
    assert _retry_path(base) == base
    base.mkdir()
    (base / "run_manifest.json").write_text("{}")
    assert _retry_path(base) == tmp_path / "arm__retry01"
    (tmp_path / "arm__retry01").mkdir()
    assert _retry_path(base) == tmp_path / "arm__retry02"


def test_long5_card_is_the_predeclared_top_five_success_workloads():
    card = json.loads((
        ROOT / "experiments/paper8_5_agent_memory/benchmarks/"
        "easy14_longest_success5.json"
    ).read_text())
    parent = json.loads((
        ROOT / "experiments/paper4_5_agent/benchmarks/"
        "swebench_verified_easy50_baseline_success14.json"
    ).read_text())
    rows = [
        json.loads(line) for line in (
            ROOT / "docs/papers/shared/results/paper4_5_runtime_productization/"
            "coding_agents/swebench_verified_easy50/no_pra/results.jsonl"
        ).read_text().splitlines() if line.strip()
    ]
    success = set(parent["instance_ids"])
    ranked = sorted(
        (row for row in rows if row["instance_id"] in success),
        key=lambda row: int(row["cumulative_prompt_tokens"]), reverse=True,
    )[:5]
    assert card["instance_ids"] == [row["instance_id"] for row in ranked]
    assert card["baseline_ranking"] == [
        {
            "instance_id": row["instance_id"],
            "cumulative_prompt_tokens": row["cumulative_prompt_tokens"],
            "model_call_count": row["model_call_count"],
        }
        for row in ranked
    ]


def test_certified_long5_campaign_uses_distinct_dag_and_combined_policies():
    spec = json.loads((
        ROOT / "experiments/paper8_5_agent_memory/configs/"
        "autonomous_long5_curve_certified_v3.json"
    ).read_text())

    validate_campaign_spec(spec)
    arms = {row["arm_id"]: row for row in spec["arms"]}
    assert arms["dag_certified100"]["policy"] == "dag_certified_exclusion"
    assert arms["matched_tail90"]["policy"] == "matched_token_tail"
    assert arms["whole_record_h1h3"] == {
        "arm_id": "whole_record_h1h3",
        "policy": "safe2_h1_h3",
        "negative_realization": "drop",
        "negative_fallback": "none",
        "budget_fraction": 1.0,
    }
    assert arms["payload_stub_h1h3"]["negative_realization"] == (
        "observation_receipt"
    )
    assert not any("80" in arm_id for arm_id in arms)
