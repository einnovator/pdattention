from __future__ import annotations

from pathlib import Path

from experiments.paper3_5_adaptive_pra.adaptive_experiment import (
    CONTROLLER_FEATURE_NAMES,
    build_examples,
)
from experiments.paper3_5_adaptive_pra.systems_benchmarks import (
    benchmark_kv_baselines,
    benchmark_rag_and_long_context,
    extract_serving_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_controller_training_has_complete_profiles_and_no_oracle_features() -> None:
    examples = build_examples(
        ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/monotonic_adaptive_competition/transition_policy_rows.csv"
    )
    assert {example["partition"] for example in examples} == {"validation", "test"}
    assert all(set(example["attempts"]) == {"E0_low", "E1_medium", "E2_high"} for example in examples)
    assert not any(
        marker in name
        for name in CONTROLLER_FEATURE_NAMES
        for marker in ("oracle", "gold", "correct", "evidence_recall")
    )


def test_rag_and_long_context_baselines_share_corpus_and_account_tokens() -> None:
    rag, long_context = benchmark_rag_and_long_context()
    assert len({row["corpus_documents"] for row in rag}) == 1
    assert all(row["backbone_generation_held_constant"] for row in rag)
    assert all(row["active_kv_tokens"] >= row["prompt_tokens"] for row in rag)
    iterative = next(row for row in rag if row["method"] == "iterative_rag_top4_plus_links")
    single = next(row for row in rag if row["method"] == "single_shot_rag_top4")
    assert iterative["cost_units"] >= single["cost_units"]
    assert iterative["retrieval_recall"] >= single["retrieval_recall"]
    full = next(row for row in long_context if row["method"] == "native_full_context")
    assert full["active_documents"] == full["logical_context_documents"]


def test_kv_baselines_are_compared_at_identical_active_budgets() -> None:
    rows = benchmark_kv_baselines()
    for budget in (32, 64, 128):
        matched = [row for row in rows if row["active_kv_budget"] == budget]
        assert len(matched) == 5
        assert all(row["matched_budget"] for row in matched)


def test_inherited_serving_metrics_have_mandatory_instrumentation() -> None:
    rows = extract_serving_metrics(
        ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation/gate3_generation_rows.csv"
    )
    assert rows
    mandatory = {
        "active_kv_tokens",
        "active_kv_bytes",
        "peak_gpu_allocated_bytes",
        "ttft_seconds",
        "tpot_seconds",
        "total_latency_seconds",
        "routing_latency_seconds",
        "gather_materialization_seconds",
        "h2d_bytes",
        "h2d_seconds",
        "tokens_per_second",
        "throughput_requests_per_second",
        "measurement_scope",
    }
    assert mandatory <= set(rows[0])
    assert all(row["measured"] for row in rows)
