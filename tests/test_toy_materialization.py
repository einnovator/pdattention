"""Causal accounting tests for the toy-first Paper 3 experiment."""

from __future__ import annotations

import pytest
import torch

from pra_torch.controlled_local_sa import ControlledTokenizer, controlled_examples

from experiments.paper3_kv_materialization.toy_materialization import (
    ToyPolicy,
    attention_partition,
    label_metrics,
    materialized_positions,
    minimum_sufficient_radius,
    policy_intervals,
    representation_portability,
    source_and_fact_spans,
)
from experiments.paper3_kv_materialization.run_oracle_frontier import (
    _policies as pretrained_policies,
    _select_radius as select_pretrained_radius,
)


def _example(depth: int = 4):
    tokenizer = ControlledTokenizer()
    examples = controlled_examples(
        tokenizer,
        count=8,
        seed=812,
        depths=(depth,),
        distractors=(8,),
        evidence_gaps=(2,),
        lexical_overlaps=(0.5,),
        relation_types=(8,),
        branchings=(1,),
    )
    return tokenizer, examples[0]


def test_controlled_source_geometry_is_deterministic_and_disjoint():
    _tokenizer, example = _example()
    source, facts, cores = source_and_fact_spans(example)

    assert len(facts) == len(example.references)
    assert all(end - start == 5 for start, end in facts.values())
    assert all(end - start == 3 for start, end in cores.values())
    assert set(facts) == {reference.uri for reference in example.references}
    assert len(source) == 1 + len(example.references) * (5 + example.evidence_gap)


def test_oracle_and_wrong_disclosures_match_exact_physical_budget():
    _tokenizer, example = _example(depth=3)
    source, facts, cores = source_and_fact_spans(example)
    evidence = [cores[uri] for uri in example.target_reference_uris]
    wrong = [
        cores[reference.uri]
        for reference in example.references
        if not reference.is_evidence
    ][: len(evidence)]
    domain = "controlled-parent://test"
    exact = policy_intervals(
        ToyPolicy("T1_radius_0", "logical_intervals"),
        domain=domain,
        source_tokens=len(source),
        evidence_core_spans=evidence,
        evidence_fact_spans=[facts[uri] for uri in example.target_reference_uris],
        wrong_core_spans=wrong,
    )
    distractor = policy_intervals(
        ToyPolicy("T9_wrong_exact", "logical_intervals", wrong_memory=True),
        domain=domain,
        source_tokens=len(source),
        evidence_core_spans=evidence,
        evidence_fact_spans=[facts[uri] for uri in example.target_reference_uris],
        wrong_core_spans=wrong,
    )

    assert sum(interval.token_count for interval in exact) == 9
    assert sum(interval.token_count for interval in distractor) == 9
    assert {(row.start, row.end) for row in exact}.isdisjoint(
        {(row.start, row.end) for row in distractor}
    )


def test_region_grouping_and_fixed_budget_preserve_total_accounting():
    policy = ToyPolicy(
        "dispersion_2",
        "logical_intervals",
        radius=4,
        budget=18,
        allocation="equal",
        region_groups=2,
    )
    intervals = policy_intervals(
        policy,
        domain="mem://parent",
        source_tokens=96,
        evidence_core_spans=((10, 13), (24, 27), (50, 53), (72, 75)),
        evidence_fact_spans=((9, 14), (23, 28), (49, 54), (71, 76)),
        wrong_core_spans=((16, 19), (30, 33), (56, 59), (80, 83)),
    )
    positions = materialized_positions(policy, intervals, source_tokens=96)

    assert len(intervals) == 2
    assert len(positions) == 18
    assert len(set(positions)) == 18


def test_attention_partition_conserves_mass_and_reports_per_head_evidence():
    summary, heads = attention_partition(
        ((0.20, 0.10, 0.05), (0.30, 0.05, 0.05)),
        (4, 5, None),
        evidence_positions={4},
        distractor_positions={5},
    )

    assert summary["evidence_attention_mass"] == pytest.approx(0.25)
    assert summary["distractor_attention_mass"] == pytest.approx(0.075)
    assert summary["surrounding_attention_mass"] == pytest.approx(0.05)
    assert summary["attention_mass_sum"] == pytest.approx(1.0)
    assert len(heads) == 2
    assert heads[0]["evidence_attention_mass"] == pytest.approx(0.20)


def test_margin_and_portability_metrics_have_declared_semantics():
    logits = torch.tensor(
        [
            8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.4, 0.1, 0.4, -0.2, 0.0, 0.2, -0.1, 0.3,
        ]
    )
    metrics = label_metrics(logits, answer_id=8)
    assert metrics["correct_margin"] == pytest.approx(1.0)
    assert metrics["correct"] == 1
    assert metrics["full_vocabulary_correct"] == 0

    contextual = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    same = representation_portability(contextual, contextual.clone())
    changed = representation_portability(contextual, -contextual)
    assert same["representation_change"] == pytest.approx(0.0)
    assert changed["representation_change"] == pytest.approx(2.0)


def test_minimum_radius_uses_parent_gain_and_smallest_feasible_candidate():
    margins = {
        "T0_none": 0.0,
        "T1_radius_0": 0.20,
        "T2_radius_2": 0.80,
        "T3_radius_4": 0.96,
        "T4_radius_8": 1.01,
        "T5_radius_16": 1.00,
        "T7_whole_parent": 1.00,
    }
    rows = [
        {"policy": policy, "correct_margin": margin, "materialized_tokens": index + 3}
        for index, (policy, margin) in enumerate(margins.items())
    ]
    result = minimum_sufficient_radius(rows, target=0.95)

    assert result["status"] == "supported"
    assert result["radius"] == 4
    assert result["retention"] == pytest.approx(0.96)


def test_pretrained_confirmation_policy_is_small_and_validation_frozen():
    validation = pretrained_policies("validation", study="confirmation")
    assert [policy.name for policy in validation] == [
        "M_none",
        "M0_native_gist",
        "M1_whole_parent",
        "M3_radius_0",
        "M3_radius_2",
        "M3_radius_4",
        "M3_radius_8",
    ]
    aggregates = []
    for dataset in ("musique", "2wikimultihopqa"):
        aggregates.extend(
            [
                {
                    "dataset": dataset,
                    "condition": "M1_whole_parent",
                    "gold_mean_token_logprob": -1.0,
                    "materialized_unique_tokens": 100,
                },
                {
                    "dataset": dataset,
                    "condition": "M3_radius_0",
                    "gold_mean_token_logprob": -1.2,
                    "materialized_unique_tokens": 12,
                },
                {
                    "dataset": dataset,
                    "condition": "M3_radius_2",
                    "gold_mean_token_logprob": -1.04,
                    "materialized_unique_tokens": 20,
                },
                {
                    "dataset": dataset,
                    "condition": "M3_radius_4",
                    "gold_mean_token_logprob": -1.01,
                    "materialized_unique_tokens": 28,
                },
                {
                    "dataset": dataset,
                    "condition": "M3_radius_8",
                    "gold_mean_token_logprob": -0.99,
                    "materialized_unique_tokens": 44,
                },
            ]
        )
    selected = select_pretrained_radius(aggregates)
    assert selected["selection_partition"] == "validation"
    assert selected["selected_radius"] == {
        "musique": 2,
        "2wikimultihopqa": 2,
    }
    heldout = pretrained_policies(
        "heldout", selected_radius=2, study="confirmation"
    )
    assert {policy.name for policy in heldout} == {
        "M_none",
        "M0_native_gist",
        "M1_whole_parent",
        "M2_evidence_only",
        "M3_radius_2",
    }
