from __future__ import annotations

from experiments.paper6_3_openvino.run_distractor_ablation import (
    document_blocks,
    ranked_distractors,
)
from experiments.paper6_3_openvino.summarize_distractor_ablation import (
    render_table,
    summarize,
)


def test_document_blocks_preserve_portable_manifest_documents() -> None:
    blocks = document_blocks("Document: Alpha\nOne.\n\nDocument: Beta\nTwo.")

    assert blocks == ["Document: Alpha\nOne.", "Document: Beta\nTwo."]


def test_ranked_distractors_separate_query_overlap() -> None:
    entry = {
        "question": "Which city hosted the orbital summit?",
        "answer": "Lisbon",
        "distractor_source": (
            "Document: Weather\nRain fell in a village.\n\n"
            "Document: Summit\nThe orbital summit met in Porto."
        ),
    }

    relevant, irrelevant = ranked_distractors(entry)

    assert relevant[0].startswith("Document: Summit")
    assert irrelevant[0].startswith("Document: Weather")


def test_distractor_summary_preserves_quality_and_latency_axes() -> None:
    aggregates = []
    conditions = ["evidence_only"] + [
        f"{mode}_distractors_k{count}"
        for mode in ("relevant", "irrelevant")
        for count in (1, 2, 4, 8)
    ]
    for dataset in ("qasper", "hotpotqa", "2wikimultihopqa"):
        for condition in conditions:
            aggregates.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "sample_count": 2,
                    "token_f1": 0.25,
                    "exact_match": 0.0,
                    "answer_containment": 0.5,
                    "evidence_recall_at_4": 0.75,
                    "mean_source_tokens": 384.0,
                    "mean_distractor_tokens": 0.0,
                    "ttft_ms": {"p50": 10.0, "p95": 12.0},
                    "completion_latency_ms": {"p50": 20.0},
                    "successful_requests_per_second": 1.5,
                }
            )
    result = summarize(
        {
            "schema_version": "1.0",
            "selector_frozen": True,
            "aggregates": aggregates,
            "rows": [
                {"dataset": "qasper", "example_id": "a"},
                {"dataset": "hotpotqa", "example_id": "b"},
                {"dataset": "2wikimultihopqa", "example_id": "c"},
            ],
        }
    )

    assert len(result["rows"]) == 27
    assert result["example_count"] == 3
    table = render_table(result)
    assert "QASPER" in table
    assert "Rel. 8" in table
