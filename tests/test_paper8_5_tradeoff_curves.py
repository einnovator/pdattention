import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.tradeoff_curves import (
    _strategy_id,
    adaptive_decisions,
    load_autonomous_run,
    pair_autonomous,
    pareto_frontier,
    summarize_autonomous,
)


def _run(
    root: Path, name: str, *, policy: str, calls: int, tokens: int,
    resolved: bool, command_prefix: str | None = None, pair_id: str | None = None,
):
    path = root / name
    path.mkdir()
    manifest = {
        "study": "paper8_5_autonomous_agent_memory",
        "instance_id": "task",
        "model": "model",
        "seed": 0,
        "pair_id": pair_id,
        "selection": {"policy": policy, "materialization_mode": "whole_record"},
    }
    metrics = {
        "calls": calls,
        "actions": calls - 1,
        "cumulative_full_tokens": 100,
        "cumulative_materialized_tokens": tokens,
        "reacquisition_events": 0,
        "official_result": {"resolved": resolved},
    }
    (path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "autonomous_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    rows = [
        {"request_index": index,
         "assistant_command_sha256": f"{command_prefix or policy}-{index}",
         "selected_messages_sha256": f"{policy}-{index}",
         "full_tokens": 100,
         "materialized_tokens": tokens}
        for index in range(1, calls + 1)
    ]
    (path / "request_selection.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def test_autonomous_pair_reports_gross_net_calls_and_divergence(tmp_path):
    full = load_autonomous_run(_run(
        tmp_path, "full", policy="full", calls=10, tokens=100, resolved=True,
        pair_id="pair",
    ))
    candidate = load_autonomous_run(_run(
        tmp_path, "candidate", policy="h3_read_superseded", calls=8,
        tokens=70, resolved=True, pair_id="pair",
    ))
    pair_autonomous(candidate, full)
    assert candidate["saving_fraction"] == pytest.approx(.3)
    assert candidate["paired_net_saving_fraction"] == pytest.approx(.3)
    assert candidate["call_delta"] == -2
    assert candidate["tool_call_delta"] == -2
    assert candidate["efficiency_qualified"] is True
    assert candidate["first_action_divergence"] == 1
    assert candidate["first_action_divergence_kind"] == "command_mismatch"


def test_pair_marks_early_termination_as_divergence_and_failed_efficiency(tmp_path):
    full = load_autonomous_run(_run(
        tmp_path, "full", policy="full", calls=10, tokens=100, resolved=True,
        command_prefix="same", pair_id="pair",
    ))
    candidate = load_autonomous_run(_run(
        tmp_path, "candidate", policy="h3_read_superseded", calls=8,
        tokens=70, resolved=False, command_prefix="same", pair_id="pair",
    ))
    pair_autonomous(candidate, full)
    assert candidate["efficiency_qualified"] is False
    assert candidate["first_action_divergence"] == 9
    assert candidate["first_action_divergence_kind"] == "candidate_terminated"
    assert candidate["saving_through_first_divergence_fraction"] == pytest.approx(.3)


def test_pair_rejects_conflicting_declared_pair_ids(tmp_path):
    full = load_autonomous_run(_run(
        tmp_path, "full", policy="full", calls=2, tokens=100, resolved=True,
        pair_id="a",
    ))
    candidate = load_autonomous_run(_run(
        tmp_path, "candidate", policy="h3_read_superseded", calls=2,
        tokens=90, resolved=True, pair_id="b",
    ))
    with pytest.raises(ValueError, match="pair IDs"):
        pair_autonomous(candidate, full)


def test_pair_rejects_changed_frozen_execution_identity(tmp_path):
    full = load_autonomous_run(_run(
        tmp_path, "full", policy="full", calls=2, tokens=100, resolved=True,
        pair_id="pair",
    ))
    candidate = load_autonomous_run(_run(
        tmp_path, "candidate", policy="h3_read_superseded", calls=2,
        tokens=90, resolved=True, pair_id="pair",
    ))
    candidate["pairing_identity"]["temperature"] = 0.5
    with pytest.raises(ValueError, match="execution identity"):
        pair_autonomous(candidate, full)


def test_loader_rejects_incomplete_or_sparse_autonomous_artifact(tmp_path):
    path = _run(
        tmp_path, "candidate", policy="full", calls=3, tokens=100,
        resolved=True,
    )
    trace_path = path / "request_selection.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    trace_path.write_text(
        json.dumps(rows[0]) + "\n" + json.dumps(rows[2]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sparse request indexes"):
        load_autonomous_run(path)


def test_pareto_and_pruning_do_not_promote_dominated_low_yield_arm():
    rows = [
        {"strategy": "good", "strategy_family": "a", "saving_fraction": .10,
         "quality_proxy": .95},
        {"strategy": "bad", "strategy_family": "b", "saving_fraction": .01,
         "quality_proxy": .60},
        {"strategy": "full", "strategy_family": "full", "saving_fraction": 0,
         "quality_proxy": 1.0},
    ]
    assert {row["strategy"] for row in pareto_frontier(rows, quality_key="quality_proxy")} == {
        "full", "good"
    }
    decisions = {row["strategy"]: row for row in adaptive_decisions(
        rows, low_yield_saving=.02, low_yield_quality=.80,
        minimum_independent_cohorts_for_promotion=1,
    )}
    assert decisions["bad"]["decision"] == "stop"
    assert decisions["good"]["decision"] == "promote"


def test_exceptional_saving_cannot_bypass_absolute_quality_floor():
    rows = [
        {"strategy": "dangerous", "strategy_family": "candidate",
         "saving_fraction": .50, "quality_proxy": .10},
        {"strategy": "full", "strategy_family": "full",
         "saving_fraction": 0, "quality_proxy": 1.0},
    ]
    decision = {row["strategy"]: row for row in adaptive_decisions(
        rows, low_yield_saving=.02, low_yield_quality=.80,
        exceptional_saving=.20, minimum_independent_cohorts_for_promotion=1,
    )}["dangerous"]
    assert decision["decision"] == "stop"


def test_strategy_identity_separates_memory_and_materialization_parameters():
    base = {
        "policy": "h3_read_superseded",
        "materialization_mode": "tool_structured_evidence",
        "same_span_reads_to_keep": 1,
        "protected_head_turns": 1,
        "protected_tail_turns": 2,
        "materialization_threshold_tokens": 512,
        "materialization_head_lines": 20,
        "materialization_tail_lines": 30,
        "materialization_match_context_lines": 4,
        "materialization_max_matched_lines": 32,
    }
    variants = []
    for key, value in (
        ("same_span_reads_to_keep", 2),
        ("protected_head_turns", 2),
        ("protected_tail_turns", 3),
        ("materialization_threshold_tokens", 1024),
        ("materialization_head_lines", 21),
        ("materialization_tail_lines", 31),
        ("materialization_match_context_lines", 5),
        ("materialization_max_matched_lines", 33),
    ):
        variant = dict(base)
        variant[key] = value
        variants.append(_strategy_id(variant))
    assert len(set(variants + [_strategy_id(base)])) == len(variants) + 1


def test_autonomous_summary_macro_averages_tasks_not_repeat_runs():
    rows = [
        {"strategy": "p", "task_id": "a", "official_resolved": True,
         "saving_fraction": .1, "cumulative_full_tokens": 100,
         "cumulative_materialized_tokens": 90},
        {"strategy": "p", "task_id": "a", "official_resolved": False,
         "saving_fraction": .2, "cumulative_full_tokens": 100,
         "cumulative_materialized_tokens": 80},
        {"strategy": "p", "task_id": "b", "official_resolved": True,
         "saving_fraction": .5, "cumulative_full_tokens": 400,
         "cumulative_materialized_tokens": 200},
    ]
    summary = summarize_autonomous(rows)[0]
    assert summary["task_count"] == 2
    assert summary["macro_task_resolution"] == pytest.approx(.75)
    assert summary["workload_gross_saving_ratio_of_sums"] == pytest.approx(1 - 370 / 600)
