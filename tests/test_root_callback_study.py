import numpy as np
import torch

from experiments.paper3_5_adaptive_pra.request_reply_end_to_end import _selected
from experiments.paper3_5_adaptive_pra.root_callback_study import (
    RidgeClassifier,
    _best,
    _refined_initial,
)
from pra_torch.memory import (
    ChunkRoutingGist,
    LayerKV,
    LayerReferenceMemory,
    PRACacheEntry,
    ReferenceChunkMemory,
)


def _row(action: str, recall: float, precision: float = 0.5) -> dict[str, str | float]:
    return {
        "complete_action": action,
        "evidence_recall": recall,
        "complete_recovery": float(recall == 1.0),
        "precision": precision,
        "mrr": precision,
        "active_fraction": 0.1,
        "root_comparisons": 8,
        "successor_comparisons": 16,
        "graph_calls": 0,
    }


def test_quality_oracle_prefers_recall_before_cost() -> None:
    expensive = _row("high", 1.0)
    expensive["active_fraction"] = 0.9
    cheap = _row("low", 0.5, precision=1.0)
    assert _best([cheap, expensive]) is expensive


def test_graph_refinement_preserves_root_method_and_facet_count() -> None:
    available = {"structural_graph.f4.bm25", "graph.f2.semantic"}
    assert _refined_initial("structural.f4.bm25", available) == "structural_graph.f4.bm25"
    assert _refined_initial("global.f1.semantic", available) == "graph.f2.semantic"
    assert _refined_initial("graph.f2.semantic", available) is None


def test_bootstrap_ridge_returns_only_training_labels() -> None:
    features = np.asarray([[0.0], [0.1], [0.9], [1.0]])
    labels = ["left", "left", "right", "right"]
    classifier = RidgeClassifier.fit(features, labels, seed=7)
    assert classifier.predict_one(np.asarray([0.0])) in set(labels)
    assert classifier.predict_one(np.asarray([1.0])) in set(labels)


def test_conceptual_ids_map_to_native_logical_spans() -> None:
    uri = "benchmark://fixture/example"
    chunks = []
    for index, (start, end) in enumerate(((0, 32), (32, 64))):
        payload = torch.zeros(1, 1, end - start, 1)
        chunks.append(
            ReferenceChunkMemory(
                chunk_id=f"{uri}#chunk={start}:{end}",
                source_uri=uri,
                token_start=start,
                token_end=end,
                token_kv=LayerKV(payload, payload),
                routing_gist=ChunkRoutingGist(torch.zeros(4)),
                logical_start=start,
                logical_end=end,
            )
        )
    entry = PRACacheEntry(uri, "fixture", {27: LayerReferenceMemory(chunks)})
    selected = _selected(entry, 27, [f"{uri}#chunk=1"], [(0, 32), (32, 64)])
    assert [row.chunk_id for row in selected] == [f"{uri}#chunk=32:64"]
    assert selected[0].metadata["conceptual_chunk_id"] == f"{uri}#chunk=1"
