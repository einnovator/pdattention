from __future__ import annotations

import pytest
import torch

from pra_hf.adaptive_facets import (
    FACET_MODES,
    GraphFacetConfig,
    build_adaptive_query_facets,
    normalize_facet_mode,
)


def _query():
    texts = ["Which", "Ada", "worked", "with", "Babbage", "?", "Why", "?"]
    hidden = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.7, 0.3, 0.0],
            [0.0, 0.3, 0.7],
            [0.0, 0.1, 0.9],
            [0.1, 0.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.8, 0.2],
        ]
    )
    return hidden, texts


@pytest.mark.parametrize("mode", FACET_MODES)
def test_all_request_reply_facet_modes_preserve_a_global_root(mode: str) -> None:
    hidden, texts = _query()
    tree = build_adaptive_query_facets(
        hidden,
        texts,
        mode=mode,
        graph_config=GraphFacetConfig(threshold=0.45, top_k=2),
    )
    assert tree.root_id == "query"
    assert tree.nodes[0].kind == "global"
    assert tree.scoring_facets.hidden.shape[1] == hidden.shape[1]
    assert tree.metrics.facet_mode == mode
    assert tree.metrics.facet_count == tree.scoring_facets.hidden.shape[0]
    if "graph" in mode:
        assert tree.metrics.graph_calls >= 1
        assert tree.metrics.graph_nodes == len(texts)
        assert tree.metrics.pairwise_similarity_evaluations > 0
    else:
        assert tree.metrics.graph_calls == 0


def test_syntactic_graph_tree_retains_parent_children_and_non_oracle_metadata() -> None:
    hidden, texts = _query()
    native = hidden[:, None, :]
    tree = build_adaptive_query_facets(
        hidden,
        texts,
        mode="syntactic->graph",
        native_query=native,
        coarse_partition_mode="sentence",
        graph_config=GraphFacetConfig(min_component_size=1, max_component_size=3),
    )
    by_id = {node.facet_id: node for node in tree.nodes}
    coarse = [node for node in tree.nodes if node.kind == "sentence"]
    graph = [node for node in tree.nodes if node.kind == "graph_community"]
    assert coarse and graph
    assert all(by_id[node.parent_id].kind == "sentence" for node in graph)
    assert all(node.token_spans for node in graph)
    assert all("token_count" in node.lexical_features for node in tree.nodes)
    assert tree.scoring_facets.native_query is not None
    assert tree.scoring_facets.native_query.shape[1:] == (1, 3)
    assert tree.metrics.tree_node_count == len(tree.nodes)
    assert tree.metrics.mean_facet_overlap > 0.0


def test_graph_component_constraints_and_mode_aliases_are_validated() -> None:
    assert normalize_facet_mode("last_span") == "global"
    assert normalize_facet_mode("syntactic -> graph".replace(" ", "")) == "syntactic_graph"
    with pytest.raises(ValueError, match="Maximum"):
        GraphFacetConfig(min_component_size=3, max_component_size=2)
    with pytest.raises(ValueError, match="facet_mode"):
        normalize_facet_mode("unknown")
