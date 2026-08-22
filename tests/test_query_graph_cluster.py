import pytest
import torch

from pra_hf.query_graph import QueryGraph, QueryUnitProvenance, build_query_graph
from pra_hf.query_graph_cluster import (
    connected_components,
    deterministic_kmeans,
    facet_recovery_metrics,
    threshold_filtration,
    weighted_label_propagation,
)


def _graph(count, edges, weights=None, policy="directed", device="cpu"):
    src = torch.tensor([edge[0] for edge in edges], dtype=torch.long, device=device)
    dst = torch.tensor([edge[1] for edge in edges], dtype=torch.long, device=device)
    values = torch.tensor(
        weights or [1.0] * len(edges), dtype=torch.float32, device=device
    )
    provenance = tuple(QueryUnitProvenance(i, i, i + 1) for i in range(count))
    return QueryGraph(
        node_ids=torch.arange(count, device=device),
        token_start=torch.arange(count, device=device),
        token_end=torch.arange(1, count + 1, device=device),
        src=src,
        dst=dst,
        weight=values,
        components={"contextual": values.clone()},
        provenance=provenance,
        policy=policy,
        top_k=2,
        threshold=0.0,
    )


@pytest.mark.parametrize(
    ("count", "edges", "expected"),
    [
        (3, [], 3),
        (4, [(0, 1), (1, 0), (2, 3), (3, 2)], 2),
        (4, [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)], 1),
        (4, [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)], 1),
    ],
)
def test_connected_components_handles_isolates_components_chain_and_star(count, edges, expected):
    result = connected_components(_graph(count, edges, policy="union"))
    assert result.cluster_count == expected
    assert result.converged


def test_components_accept_duplicate_unsorted_edges_and_respect_direction():
    edges = [(2, 1), (1, 0), (2, 1), (1, 0)]
    directed = connected_components(_graph(3, edges))
    assert directed.cluster_count == 3
    symmetric = connected_components(
        _graph(3, edges + [(1, 2), (0, 1)], policy="union")
    )
    assert symmetric.cluster_count == 1


def test_threshold_filtration_splits_and_never_merges():
    graph = _graph(
        4,
        [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)],
        [0.9, 0.9, 0.4, 0.4, 0.8, 0.8],
        policy="union",
    )
    levels = threshold_filtration(graph, [0.0, 0.5, 0.85])
    assert [level.result.cluster_count for level in levels] == [1, 2, 3]
    assert all(level.split_count >= 0 for level in levels)


def test_weighted_label_propagation_is_deterministic_and_weight_sensitive():
    graph = _graph(
        4,
        [(0, 1), (1, 0), (2, 3), (3, 2), (1, 2), (2, 1)],
        [1.0, 1.0, 1.0, 1.0, 0.05, 0.05],
        policy="union",
    )
    first = weighted_label_propagation(graph)
    second = weighted_label_propagation(graph)
    assert torch.equal(first.labels, second.labels)
    assert first.cluster_count == 2


def test_kmeans_and_recovery_metrics_use_no_target_count_inference():
    values = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    result = deterministic_kmeans(values, 2)
    metrics = facet_recovery_metrics(result.labels, torch.tensor([0, 0, 1, 1]))
    assert metrics.ari == pytest.approx(1.0)
    assert metrics.nmi == pytest.approx(1.0)
    assert metrics.pairwise_f1 == pytest.approx(1.0)
    assert metrics.cluster_count_error == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_clustering_cpu_cuda_parity():
    states = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    cpu = build_query_graph(states, top_k=1, policy="union")
    cuda = build_query_graph(states.cuda(), top_k=1, policy="union")
    assert torch.equal(
        connected_components(cpu).labels,
        connected_components(cuda).labels.cpu(),
    )
    assert torch.equal(
        weighted_label_propagation(cpu).labels,
        weighted_label_propagation(cuda).labels.cpu(),
    )
