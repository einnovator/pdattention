from __future__ import annotations

import numpy as np
import pytest

from pra_hf.rag_causal_decomposition import (
    DocumentAttentionPolicy,
    build_document_attention_mask,
)
from pra_hf.sparse_crossdoc import (
    CrossDocumentAttentionCollector,
    CrossDocumentOracleGraph,
    cumulative_attention_mass_plan,
    interaction_localization,
    top_attention_edge_plan,
)


def _graph() -> CrossDocumentOracleGraph:
    collector = CrossDocumentAttentionCollector(
        (2, 2),
        record_ids=("D1", "D2"),
        selection_receipt_id="selection-1",
        model_revision="model@revision",
    )
    first = np.zeros((1, 2, 4, 4), dtype=np.float32)
    first[0, :, 2, 0] = (0.40, 0.20)
    first[0, :, 2, 1] = (0.10, 0.10)
    first[0, :, 3, 0] = (0.05, 0.05)
    first[0, :, 3, 1] = (0.05, 0.05)
    second = np.zeros((1, 2, 4, 4), dtype=np.float32)
    second[0, :, 2, 0] = (0.20, 0.20)
    second[0, :, 2, 1] = (0.10, 0.10)
    second[0, :, 3, 0] = (0.05, 0.05)
    second[0, :, 3, 1] = (0.025, 0.025)
    collector.observe(0, first)
    collector.observe(1, second)
    return collector.finalize()


def test_collector_compresses_cross_document_edges_and_localizes_mass() -> None:
    graph = _graph()
    assert graph.edge_scores.shape == (2, 2, 4)
    assert graph.logical_edge_count == 8
    assert graph.physical_edge_count == 16
    assert graph.attention_mass == pytest.approx(1.75)
    localization = interaction_localization(graph)
    assert len(localization["layers"]) == 2
    assert localization["record_pairs"][0]["source_record_id"] == "D1"
    assert localization["record_pairs"][0]["target_record_id"] == "D2"


def test_top_attention_plan_uses_real_budget_and_replays_selected_edges() -> None:
    graph = _graph()
    plan = top_attention_edge_plan(graph, 0.25)
    assert plan.selected_logical_edges == 2
    assert plan.selected_physical_head_edges == 4
    assert plan.selected_logical_edge_fraction == pytest.approx(0.25)
    assert plan.selected_physical_edge_fraction == pytest.approx(0.25)
    assert plan.retained_attention_mass == pytest.approx(1.0 / 1.75)

    blocked, _ = build_document_attention_mask(
        (2, 2), policy=DocumentAttentionPolicy.NO_CROSS_DOC
    )
    first = plan.mask_for_layer(
        0,
        base_mask=blocked,
        source_tokens=graph.source_tokens,
        target_tokens=graph.target_tokens,
    )
    second = plan.mask_for_layer(
        1,
        base_mask=blocked,
        source_tokens=graph.source_tokens,
        target_tokens=graph.target_tokens,
    )
    assert first.shape == (2, 4, 4)
    assert first[:, 2, 0].all()
    assert second[:, 2, 0].all()
    assert not first[:, 3, 1].any()
    assert first[:, 1, 0].all()


def test_cumulative_mass_plan_selects_minimum_ranked_prefix() -> None:
    graph = _graph()
    plan = cumulative_attention_mass_plan(graph, 0.75)
    assert plan.retained_attention_mass >= 0.75
    assert plan.selected_logical_edges == 4
    empty = cumulative_attention_mass_plan(graph, 0.0)
    assert empty.selected_logical_edges == 0
    assert empty.retained_attention_mass == 0.0


def test_oracle_graph_round_trips_without_expanding_edges(tmp_path) -> None:
    graph = _graph()
    path = tmp_path / "teacher_graph.npz"
    graph.save(path)
    loaded = CrossDocumentOracleGraph.load(path)
    assert loaded.graph_digest == graph.graph_digest
    assert loaded.summary() == graph.summary()


def test_plan_validates_budget_bounds() -> None:
    graph = _graph()
    with pytest.raises(ValueError, match="fraction"):
        top_attention_edge_plan(graph, 1.01)
    with pytest.raises(ValueError, match="fraction"):
        cumulative_attention_mass_plan(graph, -0.01)
