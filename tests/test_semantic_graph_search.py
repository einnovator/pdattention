import inspect
from dataclasses import fields

import pytest
import torch

from pra_hf.semantic_graph_search import (
    SearchDecision,
    SemanticGraphSearchConfig,
    SemanticGraphSearchResult,
    build_native_parent_adjacency,
    search_semantic_graph,
)


def _scores():
    edge = torch.tensor(
        [
            [float("-inf"), 0.9, 0.2, 0.1],
            [0.8, float("-inf"), 0.85, 0.3],
            [0.1, 0.7, float("-inf"), 0.95],
            [0.2, 0.1, 0.8, float("-inf")],
        ]
    )
    goal = torch.tensor(
        [
            [0.9, 0.1, 0.2, 0.3],
            [0.0, 0.2, 0.1, 0.95],
        ]
    )
    return edge, goal


def _config(**overrides):
    values = {
        "successor_k": 1,
        "max_visited_parents": 4,
        "edge_threshold": 0.5,
        "goal_threshold": 0.9,
        "max_hops": 3,
    }
    values.update(overrides)
    return SemanticGraphSearchConfig(**values)


def test_oracle_free_root_initialization_and_native_topk_expansion():
    edge, goal = _scores()
    result = search_semantic_graph(edge, goal, [0], _config(max_hops=1))
    assert result.roots == (0,)
    assert result.visited == (0, 1)
    assert result.decisions[0].native_rank == 1
    assert result.decisions[0].candidate_parent == 1


def test_zero_hops_returns_only_roots_without_proposals():
    edge, goal = _scores()
    result = search_semantic_graph(
        edge,
        goal,
        [0],
        _config(max_hops=0, goal_threshold=float("inf")),
    )
    assert result.visited == (0,)
    assert result.nodes_expanded == 0
    assert result.raw_proposals == 0
    assert result.stop_reason == "max_hops"


def test_native_parent_adjacency_batches_local_pairs_and_reduces_by_parent():
    query = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[1.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]], [[[1.0, 1.0]]]])
    mask = torch.ones(3, 1, dtype=torch.bool)
    result = build_native_parent_adjacency(
        query,
        key,
        mask,
        torch.tensor([0, 0, 1]),
        2,
        token_reduction="max",
        head_reduction="max",
    )
    assert result.scores.shape == (2, 2)
    assert torch.isneginf(result.scores.diagonal()).all()
    assert result.scores[0, 1] > 0
    assert result.scores[1, 0] > 0
    assert result.dot_products == 9
    assert result.local_pair_count == 9


def test_native_topk_resolves_boundary_ties_by_parent_id():
    edge = torch.tensor(
        [
            [float("-inf"), 0.5, 0.5, 0.5],
            [0.1, float("-inf"), 0.1, 0.1],
            [0.1, 0.1, float("-inf"), 0.1],
            [0.1, 0.1, 0.1, float("-inf")],
        ]
    )
    result = search_semantic_graph(
        edge,
        torch.zeros(1, 4),
        [0],
        _config(successor_k=2, max_hops=1, goal_threshold=float("inf")),
    )
    assert result.visited == (0, 1, 2)


def test_edge_threshold_filters_without_reading_goal_scores():
    edge, goal = _scores()
    result = search_semantic_graph(
        edge, goal, [0], _config(edge_threshold=0.95, successor_k=2)
    )
    assert result.visited == (0,)
    assert not result.goal_triggered
    assert all(not row.admitted for row in result.decisions)


def test_any_query_facet_can_satisfy_terminal_goal():
    edge, goal = _scores()
    result = search_semantic_graph(edge, goal, [2], _config(max_hops=1))
    assert result.goal_triggered
    assert result.terminal_parent == 3
    assert result.terminal_facet == 1
    assert result.path == (2, 3)


def test_different_facet_goal_excludes_the_entry_facet_only_at_terminal():
    edge, goal = _scores()
    any_facet = search_semantic_graph(
        edge,
        goal,
        [0],
        _config(goal_threshold=0.05, max_hops=1),
        entry_facets={0: 0},
    )
    different = search_semantic_graph(
        edge,
        goal,
        [0],
        _config(goal_threshold=0.15, max_hops=1, different_facet_goal=True),
        entry_facets={0: 0},
    )
    assert any_facet.terminal_facet == 1 or any_facet.terminal_facet == 0
    assert different.goal_triggered
    assert different.terminal_facet == 1


def test_low_query_similarity_intermediate_is_not_filtered():
    edge, goal = _scores()
    # Parent 2 has weak query scores but remains the required bridge 1->2->3.
    result = search_semantic_graph(
        edge,
        goal,
        [1],
        _config(successor_k=1, max_hops=2, goal_threshold=0.9),
    )
    assert result.goal_triggered
    assert result.path == (1, 2, 3)
    bridge = next(row for row in result.decisions if row.candidate_parent == 2)
    assert bridge.admitted and not bridge.goal_triggered


def test_minimum_path_depth_prevents_trivial_root_closure():
    edge, goal = _scores()
    result = search_semantic_graph(
        edge,
        goal,
        [0],
        _config(goal_threshold=0.85, max_hops=1),
    )
    assert result.terminal_parent != 0
    assert all(row.hop >= 1 for row in result.decisions if row.goal_triggered)


def test_first_goal_stops_before_exhausting_workspace():
    edge, goal = _scores()
    result = search_semantic_graph(
        edge,
        goal,
        [2],
        _config(successor_k=3, max_visited_parents=None, max_hops=3),
    )
    assert result.stop_reason == "goal"
    assert result.visited == (2, 3)
    assert result.nodes_expanded == 1


def test_deduplication_and_cycle_detection_prevent_repeat_expansion():
    edge = torch.tensor(
        [
            [float("-inf"), 1.0, 0.9],
            [1.0, float("-inf"), 0.8],
            [0.9, 0.8, float("-inf")],
        ]
    )
    goal = torch.zeros(1, 3)
    result = search_semantic_graph(
        edge,
        goal,
        [0],
        _config(
            successor_k=2,
            max_visited_parents=None,
            edge_threshold=0.0,
            goal_threshold=float("inf"),
            max_hops=3,
        ),
    )
    assert result.visited == (0, 1, 2)
    assert result.duplicate_proposals > 0
    assert result.cycles_prevented > 0
    assert len(result.decisions) == result.raw_proposals


@pytest.mark.parametrize("strategy", ["breadth_first", "best_first", "beam"])
def test_search_strategies_are_deterministic_and_preserve_path(strategy):
    edge, goal = _scores()
    config = _config(strategy=strategy, beam_width=2)
    first = search_semantic_graph(edge, goal, [1], config)
    second = search_semantic_graph(edge, goal, [1], config)
    assert first.path == second.path == (1, 2, 3)
    assert first.visited == second.visited


def test_finite_budget_is_global_across_multiple_roots():
    edge, goal = _scores()
    result = search_semantic_graph(
        edge,
        goal,
        [0, 2],
        _config(
            successor_k=2,
            max_visited_parents=3,
            goal_threshold=float("inf"),
        ),
        entry_facets={0: 0, 2: 1},
    )
    assert len(result.visited) == 3
    assert set(result.roots) == {0, 2}
    assert result.stop_reason == "visited_budget"
    assert len(result.decisions) == result.raw_proposals


def test_unbounded_budget_still_obeys_expansion_safety_and_hop_cap():
    edge, goal = _scores()
    result = search_semantic_graph(
        edge,
        goal,
        [0],
        _config(
            successor_k=3,
            max_visited_parents=None,
            goal_threshold=float("inf"),
            max_hops=1,
            max_expanded_nodes=1,
        ),
    )
    assert result.nodes_expanded == 1
    assert max(row.hop for row in result.decisions) == 1
    assert len(result.visited) <= edge.shape[0]


def test_cost_accounting_matches_decision_trace():
    edge, goal = _scores()
    result = search_semantic_graph(edge, goal, [1], _config())
    assert result.raw_proposals >= result.edge_admitted_proposals
    assert result.goal_comparisons == result.goal_tests * goal.shape[0]
    assert result.peak_candidate_tensor_bytes > 0
    assert result.search_seconds >= result.cpu_dedup_seconds >= 0
    assert result.search_seconds >= result.goal_test_seconds >= 0


def test_search_contract_has_no_task_label_channel():
    forbidden = {"oracle", "target", "label", "evidence", "correct"}
    for record in (SearchDecision, SemanticGraphSearchResult):
        assert not forbidden.intersection(field.name for field in fields(record))
    assert not forbidden.intersection(inspect.signature(search_semantic_graph).parameters)
