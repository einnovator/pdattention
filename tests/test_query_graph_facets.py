import pytest
import torch

from pra_hf.query_graph import QueryUnitProvenance, build_query_graph
from pra_hf.query_graph_cluster import ClusterResult
from pra_hf.query_graph_facets import (
    pool_hard_graph_facets,
    pool_soft_graph_facets,
    suppress_graph_facet,
)


def _fixture():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0], [0.0, 3.0]])
    provenance = tuple(
        QueryUnitProvenance(10 + i, 20 + i, 21 + i, f"token-{i}")
        for i in range(4)
    )
    graph = build_query_graph(hidden, provenance=provenance, top_k=1, policy="union")
    labels = ClusterResult(torch.tensor([0, 1, 0, 1]), 1, True, "fixture")
    return hidden, graph, labels


def test_hard_pooling_preserves_noncontiguous_membership_and_shapes():
    hidden, graph, labels = _fixture()
    native = hidden[:, None, :]
    facets = pool_hard_graph_facets(graph, hidden, labels, native_query=native)
    assert facets.hidden.shape == (3, 2)
    assert facets.native_query.shape == (3, 1, 2)
    assert facets.provenance[0].kind == "global"
    assert facets.provenance[1].member_unit_ids == (10, 12)
    assert torch.allclose(facets.hidden[1], torch.tensor([2.0, 0.0]))
    adapted = facets.as_query_facet_set()
    assert adapted.hidden.shape == facets.hidden.shape
    assert adapted.provenance[1].family == "query_graph_fixture"


def test_soft_pooling_normalizes_membership_and_supports_overlap():
    hidden, graph, _ = _fixture()
    membership = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8]]
    )
    facets = pool_soft_graph_facets(graph, hidden, membership)
    assert facets.hidden.shape == (3, 2)
    assert torch.equal(facets.membership[:, 0], torch.ones(4))
    assert facets.provenance[2].member_unit_ids == (11, 12, 13)


def test_discovered_facet_ablation_keeps_global_control_and_provenance():
    hidden, graph, labels = _fixture()
    facets = pool_hard_graph_facets(graph, hidden, labels)
    reduced = suppress_graph_facet(facets, 1)
    assert reduced.hidden.shape[0] == 2
    assert reduced.provenance[0].kind == "global"
    with pytest.raises(ValueError, match="global"):
        suppress_graph_facet(facets, 0)
