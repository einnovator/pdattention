import inspect
from dataclasses import fields

import torch

from pra_hf.grounded_propagation import (
    AssociativeCandidateSet,
    GroundedCandidate,
    GroundedRanking,
    QueryValidation,
    generate_associative_candidates,
    query_validate_candidates,
    rank_grounded_candidates,
)


def _control():
    association = torch.tensor([float("-inf"), 0.9, 0.8, 0.1])
    candidates = generate_associative_candidates(
        association, source_parents={0}, candidate_k=3, comparisons=4
    )
    # Facet 0 nominated root A; facet 1 grounds weaker association B2.
    query_scores = torch.tensor([[0.9, 0.7, 0.2, 0.0], [0.1, 0.1, 0.95, 0.0]])
    return candidates, query_scores


def test_association_proposes_but_query_rerank_selects_grounded_candidate():
    candidates, scores = _control()
    validation = query_validate_candidates(scores, candidates, root_facet=0)
    association = rank_grounded_candidates(
        candidates, validation, mode="association", root_facet=0
    )
    grounded = rank_grounded_candidates(
        candidates, validation, mode="query_rerank", root_facet=0
    )
    assert association.selected == (1,)
    assert grounded.selected == (2,)
    assert grounded.association_comparisons == 4
    assert grounded.validation_comparisons == 6


def test_residual_grounding_excludes_root_nominating_facet_and_keeps_provenance():
    candidates, scores = _control()
    validation = query_validate_candidates(
        scores, candidates, root_facet=0, residual_only=True
    )
    result = rank_grounded_candidates(
        candidates, validation, mode="query_rerank", root_facet=0
    )
    assert result.selected == (2,)
    record = next(row for row in result.candidates if row.parent_index == 2)
    assert record.validating_facet == 1
    assert not record.validating_facet_is_root


def test_threshold_conjunction_stops_ungrounded_drift():
    candidates = generate_associative_candidates(
        torch.tensor([float("-inf"), 0.9, 0.8]),
        source_parents={0},
        candidate_k=2,
    )
    validation = query_validate_candidates(
        torch.tensor([[0.9, 0.2, 0.1]]), candidates, root_facet=0
    )
    result = rank_grounded_candidates(
        candidates,
        validation,
        mode="threshold_conjunction",
        query_threshold=0.5,
        root_facet=0,
    )
    assert result.selected == ()
    assert not any(row.admitted for row in result.candidates)


def test_rank_conjunction_is_invariant_to_raw_association_rescaling():
    scores = torch.tensor([[0.0, 0.4, 0.9]])
    outputs = []
    for association in (
        torch.tensor([float("-inf"), 0.8, 0.7]),
        torch.tensor([float("-inf"), 8000.0, -7000.0]),
    ):
        candidates = generate_associative_candidates(
            association, source_parents={0}, candidate_k=2
        )
        validation = query_validate_candidates(scores, candidates, root_facet=0)
        outputs.append(
            rank_grounded_candidates(
                candidates,
                validation,
                mode="rank_conjunction",
                rank_lambda=2.0,
                root_facet=0,
            ).selected
        )
    assert outputs == [(2,), (2,)]


def test_grounding_deduplicates_source_and_obeys_final_candidate_budget():
    candidates = generate_associative_candidates(
        torch.tensor([1.0, 0.9, 0.8, 0.7]), source_parents={0, 1}, candidate_k=8
    )
    validation = query_validate_candidates(
        torch.tensor([[0.0, 0.0, 0.8, 0.7]]), candidates
    )
    result = rank_grounded_candidates(
        candidates, validation, mode="query_rerank", final_k=1
    )
    assert candidates.parent_indices == (2, 3)
    assert len(result.selected) == 1


def test_grounded_router_contract_has_no_oracle_or_target_label_input():
    forbidden = {"oracle", "target", "label", "evidence"}
    for record_type in (
        AssociativeCandidateSet,
        QueryValidation,
        GroundedCandidate,
        GroundedRanking,
    ):
        assert not forbidden.intersection(field.name for field in fields(record_type))
    for operation in (
        generate_associative_candidates,
        query_validate_candidates,
        rank_grounded_candidates,
    ):
        assert not forbidden.intersection(inspect.signature(operation).parameters)
