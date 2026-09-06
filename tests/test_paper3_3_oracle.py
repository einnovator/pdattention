from pathlib import Path

import pytest

from experiments.paper3_3_sparse_crossdoc.freeze_splits import (
    build_manifest,
    make_splits,
)
from experiments.paper3_3_sparse_crossdoc.run_oracle_sparsity import (
    _percentages,
    _ranking_targets,
    _select_split_cohort,
    interventional_oracle_gate,
    oracle_gate,
    ranking_frontier_diagnostics,
    summarize_rows,
)


def test_frozen_splits_are_deterministic_disjoint_and_exclude_legacy() -> None:
    identities = tuple(f"example:{index}" for index in range(30))
    first = make_splits(
        identities,
        excluded_ids=("example:1", "example:2"),
        seed=33,
        train_size=10,
        validation_size=5,
        test_size=5,
    )
    second = make_splits(
        identities,
        excluded_ids=("example:1", "example:2"),
        seed=33,
        train_size=10,
        validation_size=5,
        test_size=5,
    )
    assert first == second
    assigned = set().union(*(set(values) for values in first.values()))
    assert not assigned & {"example:1", "example:2"}
    assert sum(len(values) for values in first.values()) == len(assigned) == 20


def test_split_manifest_records_provenance_and_digest() -> None:
    metadata = {
        "dataset_revision": "dataset-revision",
        "corpus_revision": "corpus-revision",
    }
    manifest = build_manifest(
        example_ids=tuple(f"example:{index}" for index in range(20)),
        excluded_ids=("example:0",),
        dataset_metadata=metadata,
        seed=33,
        train_size=8,
        validation_size=4,
        test_size=4,
        legacy_manifest=Path("legacy.json"),
    )
    assert manifest["legacy_eval_excluded_from_all_splits"] is True
    assert manifest["counts"] == {"train": 8, "validation": 4, "test": 4}
    assert len(manifest["split_digest"]) == 64


def test_percentage_parser_validates_bounds_and_deduplicates() -> None:
    assert _percentages("0, 0.1, 0.1, 100") == (0.0, 0.1, 100.0)
    with pytest.raises(Exception):
        _percentages("101")


def test_ranking_target_parser_rejects_unknown_modes() -> None:
    assert _ranking_targets("attention,pair_nll,pair_nll") == (
        "attention",
        "pair_nll",
    )
    with pytest.raises(Exception):
        _ranking_targets("attention,magic")


def test_summary_and_oracle_gate_preserve_measured_scope() -> None:
    rows = [
        {
            "condition": "ORACLE_TOP_ATTENTION",
            "target_percentage": 5.0,
            "token_f1": 0.20,
            "official_multihop_rag_score": 0.70,
            "exact_match": 0.0,
            "gold_answer_mean_nll": 2.0,
            "first_step_js_divergence": 0.01,
            "selected_logical_edge_fraction": 0.05,
            "retained_attention_mass": 0.8,
            "selected_physical_head_edges": 20,
            "encode_ms": 10.0,
            "ttft_ms": 20.0,
            "total_latency_ms": 30.0,
        }
    ]
    summary = summarize_rows(rows)
    gate = oracle_gate(summary)
    assert summary[0]["official_score"] == pytest.approx(0.70)
    assert gate["status"] == "PASS_SMOKE"
    assert gate["passing_conditions"][0]["target_percentage"] == 5.0


def test_interventional_gate_requires_power_and_monotonicity() -> None:
    rows = [
        {
            "condition": "ORACLE_PAIR_NLL",
            "target_percentage": percentage,
            "token_f1": score,
            "official_multihop_rag_score": 0.70,
            "exact_match": 0.0,
            "gold_answer_mean_nll": 2.0,
            "first_step_js_divergence": 0.01,
            "selected_logical_edge_fraction": percentage / 100.0,
            "selected_physical_edge_fraction": percentage / 100.0,
            "retained_attention_mass": 0.8,
            "selected_physical_head_edges": 20,
            "encode_ms": 10.0,
            "ttft_ms": 20.0,
            "total_latency_ms": 30.0,
        }
        for percentage, score in ((0.0, 0.10), (0.1, 0.20))
        for _ in range(100)
    ]
    summary = summarize_rows(rows)
    frontiers = ranking_frontier_diagnostics(summary)
    gate = interventional_oracle_gate(summary, frontiers)
    assert gate["status"] == "PASS_POWERED"
    assert gate["learned_selector_training_unlocked"] is True


def test_split_selection_samples_only_declared_ids(tmp_path) -> None:
    class Question:
        def __init__(self, example_id: str) -> None:
            self.example_id = example_id

    questions = tuple(Question(f"example:{index}") for index in range(6))
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        '{"test_ids":["example:1","example:3","example:5"],"split_digest":"abc"}',
        encoding="utf-8",
    )
    selected, metadata = _select_split_cohort(
        questions,
        split_manifest=manifest,
        split_name="test",
        max_examples=2,
        seed=11,
    )
    assert {row.example_id for row in selected} <= {
        "example:1",
        "example:3",
        "example:5",
    }
    assert metadata["split_digest"] == "abc"


def test_split_builder_rejects_insufficient_dataset() -> None:
    with pytest.raises(ValueError, match="only 8 remain"):
        make_splits(
            tuple(f"example:{index}" for index in range(10)),
            excluded_ids=("example:0", "example:1"),
            seed=33,
            train_size=4,
            validation_size=3,
            test_size=2,
        )
