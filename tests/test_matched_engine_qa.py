from experiments.engine_serving.matched_qa import (
    manifest_entries_from_rows,
    selected_source,
    source_digest,
)
from experiments.paper6_2_mlx.run_answer_quality_pressure import QADocument, QAExample


def _example() -> QAExample:
    return QAExample(
        dataset="fixture",
        example_id="example-1",
        question="Where is the answer?",
        answer="Second",
        source="",
        source_scope="fixture",
        documents=(
            QADocument("first", "First", "First body."),
            QADocument("second", "Second", "Second body."),
        ),
        evidence_document_ids=frozenset(("second",)),
    )


def test_selected_source_preserves_frozen_rank_order():
    source = selected_source(_example(), ("second", "first"))
    assert source == "Document: Second\nSecond body.\n\nDocument: First\nFirst body."
    assert source_digest(source) == source_digest(source)


def test_manifest_entries_keep_one_native_row_per_example():
    example = _example()
    rows = [
        {
            "condition": "routed_native",
            "seed": 11,
            "example_id": example.example_id,
            "selected_document_ids": ["second", "first"],
            "evidence_recall_at_4": 1.0,
        },
        {
            "condition": "routed_native",
            "seed": 11,
            "example_id": example.example_id,
            "selected_document_ids": ["second", "first"],
            "evidence_recall_at_4": 1.0,
        },
        {
            "condition": "no_memory",
            "seed": 11,
            "example_id": example.example_id,
            "selected_document_ids": ["first"],
            "evidence_recall_at_4": 0.0,
        },
    ]
    [entry] = manifest_entries_from_rows((example,), rows)
    assert entry["selected_document_ids"] == ["second", "first"]
    assert entry["seed"] == 11
    assert entry["selected_source_characters"] > 0


def test_selected_source_rejects_unknown_document():
    try:
        selected_source(_example(), ("missing",))
    except ValueError as error:
        assert "Unknown selected document" in str(error)
    else:
        raise AssertionError("unknown selected document should fail")
