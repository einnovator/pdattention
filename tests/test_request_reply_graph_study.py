from __future__ import annotations

from experiments.paper3_5_adaptive_pra.request_reply_graph_study import (
    FEATURE_NAMES,
    _best,
    _rrf,
    _split,
)


def test_controller_split_is_deterministic_and_identity_disjoint() -> None:
    identities = [
        (dataset, f"example-{index}")
        for dataset in ("qasper", "hotpotqa", "2wikimultihopqa", "musique")
        for index in range(10)
    ]
    first = _split(identities)
    second = _split(tuple(reversed(identities)))
    assert first == second
    assert first[0].isdisjoint(first[1])
    assert first[0] | first[1] == set(identities)
    assert all("dataset" not in name for name in FEATURE_NAMES)


def test_oracle_tie_break_uses_work_not_wall_clock_jitter() -> None:
    rows = [
        {
            "condition": "more_work",
            "evidence_recall": 0.5,
            "precision": 0.5,
            "mrr": 0.5,
            "pairwise_similarity_evaluations": 20,
            "routing_ms": 1,
        },
        {
            "condition": "less_work",
            "evidence_recall": 0.5,
            "precision": 0.5,
            "mrr": 0.5,
            "pairwise_similarity_evaluations": 10,
            "routing_ms": 100,
        },
    ]
    assert _best(rows)["condition"] == "less_work"


def test_per_facet_fusion_is_globally_bounded_and_deduplicated() -> None:
    selected, positives = _rrf(
        (
            {"selected_chunk_ids": "a|b|c|d", "positive_chunk_ids": "b|e"},
            {"selected_chunk_ids": "b|e|f|g", "positive_chunk_ids": "b|e"},
        ),
        budget=4,
    )
    assert len(selected) == len(set(selected)) == 4
    assert selected[0] == "b"
    assert positives == {"b", "e"}
