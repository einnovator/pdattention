import pytest
import torch

from pra_hf.query_graph import (
    QueryUnitProvenance,
    build_query_graph,
    graph_memory_bytes,
    lexical_feature_matrix,
    threshold_query_graph,
)


def _states(device="cpu"):
    return torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], device=device
    )


def test_graph_contract_retains_stable_query_unit_provenance():
    provenance = tuple(
        QueryUnitProvenance(10 + index, 20 + index, 21 + index, f"u{index}", layer=27)
        for index in range(4)
    )
    graph = build_query_graph(
        _states(), provenance=provenance, top_k=1, threshold=0.1, policy="directed"
    )
    assert graph.node_ids.tolist() == [10, 11, 12, 13]
    assert graph.src.shape == graph.dst.shape == graph.weight.shape
    assert all(values.shape == graph.weight.shape for values in graph.components.values())
    assert graph.edge_count == 4
    assert graph_memory_bytes(graph) > 0


def test_graph_sparsification_and_explicit_symmetry_policies():
    directed_attention = torch.tensor(
        [[0.0, 0.9, 0.0], [0.0, 0.0, 0.8], [0.1, 0.0, 0.0]]
    )
    hidden = torch.eye(3)
    directed = build_query_graph(
        hidden,
        attention=directed_attention,
        contextual_weight=0.0,
        attention_weight=1.0,
        top_k=1,
        policy="directed",
    )
    assert set(zip(directed.src.tolist(), directed.dst.tolist())) == {(0, 1), (1, 2), (2, 0)}
    union = build_query_graph(
        hidden,
        attention=directed_attention,
        contextual_weight=0.0,
        attention_weight=1.0,
        top_k=1,
        policy="union",
    )
    edges = set(zip(union.src.tolist(), union.dst.tolist()))
    assert all((right, left) in edges for left, right in edges)


def test_lexical_features_are_stable_and_threshold_filters_fixed_edges():
    left = lexical_feature_matrix(["Mulder", "mulder", "other"])
    right = lexical_feature_matrix(["Mulder", "mulder", "other"])
    assert torch.equal(left, right)
    assert torch.dot(left[0], left[1]) == pytest.approx(1.0)
    graph = build_query_graph(
        _states(), top_k=2, threshold=0.0, policy="union"
    )
    filtered = threshold_query_graph(graph, 0.8)
    assert filtered.edge_count <= graph.edge_count
    assert set(zip(filtered.src.tolist(), filtered.dst.tolist())).issubset(
        set(zip(graph.src.tolist(), graph.dst.tolist()))
    )


def test_graph_rejects_invalid_shapes_and_duplicate_unit_ids():
    with pytest.raises(ValueError, match="shape"):
        build_query_graph(torch.ones(3))
    provenance = tuple(QueryUnitProvenance(0, i, i + 1) for i in range(3))
    with pytest.raises(ValueError, match="unique"):
        build_query_graph(torch.eye(3), provenance=provenance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_construction_cpu_cuda_parity():
    kwargs = dict(top_k=2, threshold=0.1, policy="union")
    cpu = build_query_graph(_states(), **kwargs)
    cuda = build_query_graph(_states("cuda"), **kwargs)
    assert torch.equal(cpu.src, cuda.src.cpu())
    assert torch.equal(cpu.dst, cuda.dst.cpu())
    assert torch.allclose(cpu.weight, cuda.weight.cpu(), atol=1e-6)
