import pytest
import torch

from experiments.paper2_5_iterative_pra.analyze_gate3_output_validation import (
    aggregate,
    paired_bootstrap,
)
from pra_hf.output_validation import (
    GENERATION_CONDITIONS,
    MATERIALIZATION_BANDS,
    GenerationCondition,
    deterministic_answer_metrics,
    exact_generation_join,
    fixed_chunks_for_spans,
    native_kv_accounting,
    selected_span_metrics,
    validate_protocol,
)
from pra_torch.memory import (
    ChunkKV,
    ChunkRoutingGist,
    LayerReferenceMemory,
    PRACacheEntry,
    ReferenceChunkMemory,
)


def _entry() -> PRACacheEntry:
    chunks = []
    for index in range(3):
        start = index * 16
        chunks.append(
            ReferenceChunkMemory(
                chunk_id=f"fixture#chunk={index}",
                source_uri="fixture",
                token_start=start,
                token_end=start + 16,
                logical_start=start,
                logical_end=start + 16,
                token_kv=ChunkKV(
                    torch.zeros(1, 2, 16, 4),
                    torch.zeros(1, 2, 16, 4),
                    torch.arange(start, start + 16).unsqueeze(0),
                ),
                routing_gist=ChunkRoutingGist(torch.zeros(1, 8)),
            )
        )
    return PRACacheEntry("fixture", "text", {27: LayerReferenceMemory(chunks)})


def test_exact_generation_condition_and_layer_configuration():
    validate_protocol()
    assert [row.name for row in GENERATION_CONDITIONS] == [
        "native_bounded",
        "one_shot",
        "graph_sparse",
        "graph_balanced",
        "graph_high",
        "oracle_evidence",
        "native_full_context",
    ]
    assert next(row.layers for row in MATERIALIZATION_BANDS if row.name == "all_28") == tuple(
        range(28)
    )


def test_only_oracle_condition_may_expose_oracle_selection():
    bad = list(GENERATION_CONDITIONS)
    bad[1] = GenerationCondition("one_shot", "one_shot", oracle=True)
    with pytest.raises(ValueError, match="only oracle_evidence"):
        validate_protocol(bad)


def test_fixed_evidence_is_identical_across_materialization_layers():
    selected = fixed_chunks_for_spans(
        _entry(), routing_layer=27, selected_spans=((8, 25),), selection_name="graph_balanced"
    )
    assert [row.chunk_id for row in selected] == ["fixture#chunk=0", "fixture#chunk=1"]
    assert all(row.metadata["selection_name"] == "graph_balanced" for row in selected)


def test_fixed_layer_variable_evidence_preserves_distinct_identities():
    entry = _entry()
    sparse = fixed_chunks_for_spans(
        entry, routing_layer=27, selected_spans=((0, 16),), selection_name="graph_sparse"
    )
    broad = fixed_chunks_for_spans(
        entry, routing_layer=27, selected_spans=((0, 48),), selection_name="graph_high"
    )
    assert {row.chunk_id for row in sparse} < {row.chunk_id for row in broad}
    assert {row.layer_id for row in sparse + broad} == {27}


def test_deterministic_metrics_reproduce_standard_qa_normalization():
    exact = deterministic_answer_metrics("The answer is Paris.", "Paris")
    assert exact == {
        "exact_match": 0.0,
        "token_f1": pytest.approx(0.4),
        "answer_contained": 1.0,
        "normalized_answer_accuracy": 1.0,
    }


def test_memory_accounting_separates_unique_tokens_from_layer_states():
    metrics = native_kv_accounting(
        unique_tokens=32,
        materialization_layers=(24, 25, 26, 27),
        kv_heads=8,
        head_dim=128,
        element_size=2,
    )
    assert metrics["materialized_unique_tokens"] == 32
    assert metrics["native_kv_token_states"] == 128
    assert metrics["native_kv_bytes"] == 128 * 2 * 8 * 128 * 2


def test_selected_span_accounting_deduplicates_overlap():
    metrics = selected_span_metrics(((0, 16), (8, 24)), ((10, 20),), 100)
    assert metrics["selected_source_tokens"] == 24
    assert metrics["evidence_kv_tokens"] == 10
    assert metrics["non_evidence_kv_tokens"] == 14


def test_generation_metric_join_rejects_missing_or_duplicate_rows():
    discovery = [{"dataset": "d", "example_id": "e", "selection": "s", "recall": 1.0}]
    generation = [{"dataset": "d", "example_id": "e", "selection": "s", "answer": "x"}]
    assert exact_generation_join(generation, discovery)[0]["recall"] == 1.0
    with pytest.raises(ValueError, match="duplicate generation"):
        exact_generation_join(generation * 2, discovery)
    with pytest.raises(ValueError, match="no frozen discovery"):
        exact_generation_join(
            [{"dataset": "d", "example_id": "missing", "selection": "s"}], discovery
        )


def test_output_aggregate_and_paired_bootstrap_preserve_example_pairing():
    rows = []
    for example_id, baseline, target in (("a", 0.0, 0.5), ("b", 0.5, 1.0)):
        for condition, value in (("one_shot", baseline), ("graph_balanced", target)):
            rows.append(
                {
                    "dataset": "d",
                    "example_id": example_id,
                    "condition": condition,
                    "token_f1": value,
                    "normalized_answer_accuracy": value,
                }
            )
    summary = aggregate(rows, ("dataset", "condition"))
    assert next(row for row in summary if row["condition"] == "graph_balanced")["token_f1"] == 0.75
    paired = paired_bootstrap(rows, "graph_balanced", replicates=100)
    assert paired["n"] == 2
    assert paired["delta_token_f1"] == 0.5
    assert paired["token_f1_wins"] == 2
