import pytest
import torch

from pra_hf.cross_dataset_diagnostics import (
    all_offset_multiscale_facets,
    annotated_paths,
    best_group_facet_rank,
    bounded_multiscale_candidates,
    evidence_token_metrics,
    group_rank,
    product_model_path_survival,
    token_jaccard_parent_scores,
)
from experiments.paper2_5_iterative_pra.run_natural_multiscale_query_audit import (
    _target_roles,
)
from pra_hf.natural_reasoning_graph import AnnotatedEvidenceNode, NaturalReasoningExample


def test_evidence_density_counts_union_and_selected_payload():
    metrics = evidence_token_metrics(
        [(1, 5), (4, 7)],
        [(0, 4), (4, 8), (8, 12)],
        [0, 1],
        [0],
    )
    assert metrics["evidence_tokens"] == 6
    assert metrics["selected_evidence_tokens"] == 6
    assert metrics["selected_parent_tokens"] == 8
    assert metrics["evidence_density"] == pytest.approx(0.75)
    assert metrics["root_evidence_fraction"] == pytest.approx(0.5)


def test_paths_and_independent_edge_product_are_explicit():
    paths = annotated_paths(("a", "b", "c", "d"), (("a", "b"), ("a", "c"), ("c", "d")))
    assert paths == (("a", "b"), ("a", "c", "d"))
    expected = product_model_path_survival(((1,), (1, 2)), {1: 0.8, 2: 0.5})
    assert expected == pytest.approx(0.6)


def test_multiscale_facets_use_every_offset_and_preserve_provenance():
    hidden = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    facets = all_offset_multiscale_facets(hidden, (2, 7), scales=(1, 2, 4))
    assert facets.hidden.shape[0] == 1 + 5 + 4 + 2
    assert facets.provenance[0].kind == "global"
    local = facets.provenance[1:]
    assert {(row.scale, row.token_start, row.token_end) for row in local} >= {
        (1, 2, 3),
        (1, 6, 7),
        (2, 2, 4),
        (2, 5, 7),
        (4, 2, 6),
        (4, 3, 7),
    }


def test_oracle_best_rank_and_lexical_control_are_post_hoc():
    scores = torch.tensor([[0.8, 0.7, 0.1], [0.2, 0.9, 0.3]])
    ranked = best_group_facet_rank(scores, (1,))
    assert ranked.rank == 1
    assert ranked.facet_index == 1
    lexical = token_jaccard_parent_scores([1, 2], [1, 3, 2, 4, 8, 9], [(0, 2), (2, 4), (4, 6)])
    assert group_rank(lexical, (0, 1)) == 1
    assert lexical.tolist() == pytest.approx([1 / 3, 1 / 3, 0.0])


def test_bounded_router_deduplicates_before_one_shared_budget():
    scores = torch.tensor([[0.9, 0.8, 0.1], [0.7, 0.1, 0.95]])
    selected, candidates = bounded_multiscale_candidates(
        scores, proposal_width=2, global_budget=2
    )
    assert candidates == 3
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_terminal_role_requires_ordered_non_root_evidence():
    example = NaturalReasoningExample(
        dataset="test",
        example_id="x",
        question="q",
        answer="a",
        question_type="chain",
        annotated_hops=2,
        graph_type="chain",
        source="",
        nodes=(
            AnnotatedEvidenceNode("a", (0, 1), (), {}),
            AnnotatedEvidenceNode("b", (1, 2), ("a",), {}),
        ),
        raw_annotation={},
    )
    assert _target_roles(example, "a", {"a": 1, "b": 2}) == ("root",)
    assert _target_roles(example, "b", {"a": 1, "b": 2}) == ("terminal",)


def test_runtime_bounded_router_has_no_oracle_argument():
    names = bounded_multiscale_candidates.__code__.co_varnames[
        : bounded_multiscale_candidates.__code__.co_argcount
        + bounded_multiscale_candidates.__code__.co_kwonlyargcount
    ]
    assert not {"oracle", "target", "evidence"}.intersection(names)
