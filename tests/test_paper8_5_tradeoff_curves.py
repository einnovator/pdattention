import json
from pathlib import Path

import pytest

from experiments.paper8_5_agent_memory.tradeoff_curves import (
    _cluster_bootstrap_mean_ci,
    _strategy_id,
    adaptive_decisions,
    load_autonomous_run,
    pair_autonomous,
    pareto_frontier,
    summarize_autonomous,
    summarize_full_repeat_controls,
)
from experiments.paper8_5_agent_memory.run_frozen_replay import _command


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
        "agent_command_template": [
            "python", "-m", "minisweagent.run", "-c", "scaffold.yaml",
            "-c", "model.model_kwargs.temperature=0.0",
            "-c", "model.model_kwargs.api_base=http://127.0.0.1:1234/v1",
            "-o", str(path / "agent"),
        ],
    }
    metrics = {
        "calls": calls,
        "actions": calls - 1,
        "cumulative_full_tokens": calls * 100,
        "cumulative_materialized_tokens": calls * tokens,
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
    assert candidate["paired_net_saving_fraction"] == pytest.approx(.44)
    assert candidate["call_delta"] == -2
    assert candidate["tool_call_delta"] == -2
    assert candidate["efficiency_qualified"] is True
    assert candidate["failure_aware_paired_net_saving_fraction"] == pytest.approx(.44)
    assert candidate["first_action_divergence"] == 1
    assert candidate["first_action_divergence_kind"] == "command_mismatch"
    assert candidate["saving_before_first_action_divergence_fraction"] == 0
    assert candidate["official_outcome"] == "resolved"


def test_loader_separates_grader_error_from_ordinary_unresolved(tmp_path):
    path = _run(
        tmp_path, "grader-error", policy="full", calls=2, tokens=100,
        resolved=False, pair_id="pair",
    )
    metrics_path = path / "autonomous_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["official_result"]["error"] = True
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (path / "grader.log").write_text(
        "task: >>>>> Patch Apply Failed: invalid patch", encoding="utf-8"
    )

    loaded = load_autonomous_run(path)

    assert loaded["official_resolved"] is False
    assert loaded["official_error"] is True
    assert loaded["official_score"] is False
    assert loaded["official_failure_class"] == "patch_apply_failed"
    assert loaded["official_outcome"] == "patch_apply_failed"


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
    assert candidate["failure_aware_paired_net_saving_fraction"] == 0
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


def test_agent_behavior_pairing_ignores_paths_but_not_decoding(tmp_path):
    full_path = _run(
        tmp_path, "full", policy="full", calls=2, tokens=100,
        resolved=True, pair_id="pair",
    )
    candidate_path = _run(
        tmp_path, "candidate", policy="h3_read_superseded", calls=2,
        tokens=90, resolved=True, pair_id="pair",
    )
    manifest_path = candidate_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command = manifest["agent_command_template"]
    command[command.index("-o") + 1] = "/different/output"
    api_index = command.index("model.model_kwargs.api_base=http://127.0.0.1:1234/v1")
    command[api_index] = "model.model_kwargs.api_base=http://127.0.0.1:9876/v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pair_autonomous(load_autonomous_run(candidate_path), load_autonomous_run(full_path))

    command[command.index("model.model_kwargs.temperature=0.0")] = (
        "model.model_kwargs.temperature=0.5"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="execution identity"):
        pair_autonomous(load_autonomous_run(candidate_path), load_autonomous_run(full_path))


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


def test_loader_rejects_trace_token_totals_that_disagree_with_metrics(tmp_path):
    path = _run(
        tmp_path, "candidate", policy="full", calls=2, tokens=100,
        resolved=True,
    )
    metrics_path = path / "autonomous_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["cumulative_materialized_tokens"] += 1
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="materialized-token total"):
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
    assert summary["macro_task_gross_saving"] == pytest.approx((.15 + .5) / 2)
    assert summary["macro_task_resolution_ci95_lower"] is not None
    assert summary["macro_task_gross_saving_ci95_upper"] is not None


def test_adaptive_gate_does_not_count_renamed_same_evidence_as_replication():
    rows = [
        {"strategy": "candidate", "strategy_coordinate": "candidate",
         "strategy_family": "candidate", "cohort": "slice-a",
         "independence_key": "same-task-full", "saving_fraction": .2,
         "quality_proxy": .95},
        {"strategy": "candidate", "strategy_coordinate": "candidate",
         "strategy_family": "candidate", "cohort": "slice-b",
         "independence_key": "same-task-full", "saving_fraction": .2,
         "quality_proxy": .95},
        {"strategy": "full", "strategy_coordinate": "full",
         "strategy_family": "full", "cohort": "slice-a",
         "independence_key": "same-task-full", "saving_fraction": 0,
         "quality_proxy": 1.0},
    ]

    decision = adaptive_decisions(
        rows, low_yield_saving=.02, low_yield_quality=.8,
        minimum_independent_cohorts_for_promotion=2,
    )[0]

    assert decision["independent_cohorts"] == 1
    assert decision["decision"] == "replicate_frozen"


def test_adaptive_gate_collapses_overlapping_suffix_to_broadest_coverage():
    rows = [
        {"strategy": "candidate", "strategy_coordinate": "candidate",
         "strategy_family": "candidate", "cohort": "suffix",
         "independence_key": "same-task", "saving_fraction": .50,
         "quality_proxy": .50, "decisions": 7},
        {"strategy": "candidate", "strategy_coordinate": "candidate",
         "strategy_family": "candidate", "cohort": "full",
         "independence_key": "same-task", "saving_fraction": .10,
         "quality_proxy": .95, "decisions": 23},
    ]
    decision = adaptive_decisions(
        rows, low_yield_saving=.02, low_yield_quality=.8,
        minimum_independent_cohorts_for_promotion=1,
    )[0]
    assert decision["overlapping_observations_collapsed"] == 1
    assert decision["mean_saving_fraction"] == pytest.approx(.10)
    assert decision["mean_quality_proxy"] == pytest.approx(.95)


def test_full_repeat_control_summary_exposes_endpoint_and_trajectory_instability(tmp_path):
    full = load_autonomous_run(_run(
        tmp_path, "full", policy="full", calls=2, tokens=100,
        resolved=True, command_prefix="a", pair_id="pair",
    ))
    repeat = load_autonomous_run(_run(
        tmp_path, "repeat", policy="full", calls=3, tokens=100,
        resolved=False, command_prefix="b", pair_id="pair",
    ))
    pair_autonomous(repeat, full)
    summary = summarize_full_repeat_controls([full, repeat])[0]
    assert summary["endpoint_agreement"] is False
    assert summary["exact_action_trajectory"] is False
    assert summary["baseline_calls"] == 2
    assert summary["repeat_calls"] == 3


def test_task_cluster_bootstrap_treats_single_task_as_one_unit():
    assert _cluster_bootstrap_mean_ci([.5]) == (.5, .5)


def test_frozen_replay_prefers_tagged_action_after_explanatory_fence():
    response = (
        "THOUGHT\n```python\nprint('example')\n```\n"
        "text\n```mswea_bash_command\ngit diff > patch.txt\n```"
    )
    assert _command(response) == "git diff > patch.txt"
