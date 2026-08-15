import math

import torch

from pra_hf.layerwise_graph import (
    bootstrap_mean_ci,
    pearson,
    shortest_distance,
    spearman,
    topk_neighbors,
    topology_metrics,
)


def test_topology_and_shortest_path_use_only_frozen_scores():
    scores = torch.tensor(
        [
            [float("-inf"), 0.9, 0.8, 0.1],
            [0.2, float("-inf"), 0.9, 0.1],
            [0.1, 0.2, float("-inf"), 0.9],
            [0.9, 0.2, 0.1, float("-inf")],
        ]
    )
    neighbors = topk_neighbors(scores, 1)
    assert neighbors == ((1,), (2,), (3,), (0,))
    assert shortest_distance(neighbors, (0,), (3,), max_hops=4) == 3
    metrics = topology_metrics(scores, 1)
    assert metrics["mean_out_degree"] == 1.0
    assert metrics["connected_component_count"] == 1.0
    assert metrics["giant_component_fraction"] == 1.0
    assert metrics["reciprocal_edge_rate"] == 0.0


def test_correlations_handle_ties_and_bootstrap_examples():
    assert math.isclose(pearson([1, 2, 3], [2, 4, 6]), 1.0)
    assert math.isclose(spearman([1, 1, 2, 3], [4, 4, 5, 9]), 1.0)
    first = bootstrap_mean_ci([0, 1, 1], replicates=100)
    second = bootstrap_mean_ci([0, 1, 1], replicates=100)
    assert first == second
    assert first["n"] == 3
