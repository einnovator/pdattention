import json
from pathlib import Path

from experiments.paper8_5_agent_memory.run_autonomous_curve_campaign import (
    _retry_path,
    adaptive_gate,
    campaign_cells,
    write_curve_spec,
)


ROOT = Path(__file__).resolve().parents[1]


def _spec():
    return {
        "campaign_id": "campaign",
        "generation": {"seed": 0},
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


def test_adaptive_gate_stops_two_failures():
    result = adaptive_gate(
        [
            {"status": "complete", "official_resolved": False, "gross_saving_fraction": .2},
            {"status": "complete", "official_resolved": False, "gross_saving_fraction": .2},
        ],
        {"stop_after_failures": 2},
    )
    assert result[0] == "stop"


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
