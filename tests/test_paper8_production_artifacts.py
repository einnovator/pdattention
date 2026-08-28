"""Regression contracts for the measured Paper 8 production-PRA artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/papers/shared/results/paper8_tasks/production_pra"
TASK_RESULTS = RESULTS.parent


def _csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_production_scope_and_model_factorization_are_complete() -> None:
    route = _csv("production_pra_scope_results.csv")
    model = _csv("oracle_model_task_results.csv")

    assert len(route) == 90
    assert len({row["case_id"] for row in route}) == 15
    assert {row["confusability"] for row in route} == {"low", "medium", "high"}
    assert {row["seed"] for row in route} == {"11", "23", "37", "53", "71"}
    assert {
        "PRA_SESSION", "PRA_TASK_LOCAL", "PRA_TASK_STRUCT",
    } <= {row["condition"] for row in route}

    assert len(model) == 135
    assert len({row["condition"] for row in model}) == 9
    assert all(row["oracle_task_graph"] == "1" for row in model)


def test_measured_task_scope_improves_evidence_and_visible_consumption() -> None:
    summary = json.loads((RESULTS / "production_summary.json").read_text(encoding="utf-8"))
    route = summary["route"]
    accuracy = summary["model_accuracy"]

    assert route["PRA_TASK_STRUCT"]["evidence_available"] > route["PRA_SESSION"]["evidence_available"]
    assert route["PRA_TASK_STRUCT"]["cross_task_contamination"] == 0
    assert route["PRA_TASK_STRUCT"]["requested_native_tokens"] < route["PRA_SESSION"]["requested_native_tokens"]
    assert accuracy["FULL_TASK_SCOPE"] > accuracy["FULL_SESSION"]
    assert accuracy["PRA_TASK_STRUCT_VISIBLE"] > accuracy["PRA_SESSION_VISIBLE"]
    assert max(accuracy[name] for name in (
        "PRA_SESSION", "PRA_TASK_LOCAL", "PRA_TASK_STRUCT",
    )) == 0


def test_join_dag_and_replay_gates_are_recorded() -> None:
    joins = _csv("join_capacity_curve.csv")
    dags = _csv("dag_shape_results.csv")
    replay = _csv("session_replay_equivalence.csv")

    assert len(joins) == 125
    assert {int(row["fan_in"]) for row in joins} == {2, 4, 8, 16, 32}
    assert {int(row["budget_chunks"]) for row in joins} == {2, 4, 8, 16, 32}
    assert all(float(row["predecessor_recall"]) < 1 for row in joins if row["fan_in"] == "32")

    assert len(dags) == 50
    assert len({row["dag_shape"] for row in dags}) == 5
    assert {row["condition"] for row in dags} == {"PRA_SESSION", "PRA_TASK_STRUCT"}

    assert len(replay) == 15
    assert all(row["selection_equal"] == row["task_scope_equal"] == row["provenance_equal"] == "1" for row in replay)
    assert all(row["model_cache_persisted"] == "0" for row in replay)


def test_generated_paper_values_are_present() -> None:
    macros = (RESULTS / "generated_production_pra_results.tex").read_text(encoding="utf-8")
    for name in (
        "ProductionStructuralEvidence",
        "ProductionSessionContamination",
        "ProductionVisibleStructuralAccuracy",
        "ProductionDirectNativeAccuracy",
    ):
        assert f"\\newcommand{{\\{name}}}" in macros


def test_native_geometry_separates_semantic_coverage_from_consumption() -> None:
    summary = json.loads((RESULTS / "native_geometry_summary.json").read_text(encoding="utf-8"))
    by_condition = {row["condition"]: row for row in summary["rows"]}
    late = by_condition["NATIVE_FULL_SELECTED_RECORD_LATE_ONLY"]
    sparse = by_condition["NATIVE_FULL_SELECTED_RECORD_SPARSE_MULTI"]
    visible = by_condition["VISIBLE_FULL_SELECTED_RECORD"]
    assert late["n_correctly_routed"] == 15
    assert late["semantic_sufficiency_rate"] == 1
    assert late["conditional_consumption_accuracy"] == 0
    assert sparse["conditional_consumption_accuracy"] == 0
    assert visible["conditional_consumption_accuracy"] > 0.7


def test_task_management_roadmap_records_five_seed_acquisition_and_fallback() -> None:
    with (TASK_RESULTS / "task_acquisition_summary.csv").open(encoding="utf-8") as handle:
        acquisition = {row["mode"]: row for row in csv.DictReader(handle)}
    assert set(acquisition) == {"preflight_json", "preflight_markdown", "online_tools", "hybrid"}
    assert all(int(row["cases"]) == 30 for row in acquisition.values())
    assert float(acquisition["preflight_json"]["edge_f1"]) == 1
    assert float(acquisition["hybrid"]["edge_f1"]) == 1
    assert float(acquisition["online_tools"]["edge_f1"]) < 1

    with (TASK_RESULTS / "task_metadata_robustness_summary.csv").open(encoding="utf-8") as handle:
        metadata = {(row["corruption"], row["policy"]): row for row in csv.DictReader(handle)}
    assert float(metadata[("stale_record_tags", "task_structural")]["required_record_recall"]) == 0
    assert float(metadata[("stale_record_tags", "task_adaptive")]["required_record_recall"]) == 1
