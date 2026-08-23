from __future__ import annotations

import pytest
import torch

from pra_hf.adaptive_facets import build_adaptive_query_facets
from pra_hf.adaptive_search import AdaptiveSearchAction
from pra_hf.factorized_control import FactorizedEffortAction
from pra_hf.facet_routing import LinearPerFacetRouter, route_query_facets


def _tree():
    texts = ["Which", "ZXQ-19", "appeared", "before", "Beta", "?"]
    hidden = torch.eye(len(texts), dtype=torch.float32)
    return build_adaptive_query_facets(hidden, texts, mode="syntactic")


def _action(policy: str = "fixed") -> AdaptiveSearchAction:
    return AdaptiveSearchAction(
        "semantic",
        "native_semantic",
        FactorizedEffortAction.profile(1),
        facet_mode="syntactic",
        per_facet_policy=policy,
    )


def test_fixed_and_type_rule_per_facet_routes_use_the_same_facets() -> None:
    tree = _tree()
    fixed = route_query_facets(tree, _action("fixed"))
    typed = route_query_facets(tree, _action("type_rule"))
    assert [row.facet_id for row in fixed] == [row.facet_id for row in typed]
    assert all(row.root_method == "semantic" for row in fixed)
    assert any(row.root_method in {"exact", "hybrid"} for row in typed)


def test_learned_per_facet_router_uses_only_node_features() -> None:
    tree = _tree()
    nodes = [node for node in tree.nodes if node.facet_id != tree.root_id]
    targets = [("exact", "exact_new_address") for _ in nodes]
    router = LinearPerFacetRouter.fit(nodes, targets)
    routes = route_query_facets(tree, _action("learned"), learned_router=router)
    assert all(row.root_method == "exact" for row in routes)
    assert all(0.0 <= row.confidence <= 1.0 for row in routes)


def test_per_facet_oracle_is_explicitly_measured_and_requires_full_coverage() -> None:
    tree = _tree()
    nodes = [node for node in tree.nodes if node.facet_id != tree.root_id]
    rows = []
    for node in nodes:
        rows.extend(
            (
                {
                    "facet_id": node.facet_id,
                    "root_method": "semantic",
                    "successor_method": "native_semantic",
                    "recall": 0.5,
                    "precision": 1.0,
                },
                {
                    "facet_id": node.facet_id,
                    "root_method": "hybrid",
                    "successor_method": "hybrid_state",
                    "recall": 0.8,
                    "precision": 0.6,
                },
            )
        )
    selected = route_query_facets(tree, _action("oracle"), oracle_rows=rows)
    assert all(row.root_method == "hybrid" for row in selected)
    with pytest.raises(ValueError, match="cover every"):
        route_query_facets(tree, _action("oracle"), oracle_rows=rows[:-2])


def test_expanded_action_rejects_invalid_graph_and_fusion_controls() -> None:
    effort = FactorizedEffortAction.profile(0)
    with pytest.raises(ValueError, match="graph_threshold"):
        AdaptiveSearchAction("semantic", "native_semantic", effort, graph_threshold=1.1)
    with pytest.raises(ValueError, match="fusion_method"):
        AdaptiveSearchAction("semantic", "native_semantic", effort, fusion_method="unknown")
    action = AdaptiveSearchAction(
        "hybrid",
        "native_semantic",
        effort,
        facet_mode="graph",
        graph_similarity_mode="hybrid",
        channel_profile="facet_type",
        per_facet_policy="type_rule",
    )
    assert action.control_vector["F_mode"] == "graph"
    assert action.control_vector["F_graph"]["similarity"] == "hybrid"
