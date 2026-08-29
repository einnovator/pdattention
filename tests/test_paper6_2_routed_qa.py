from experiments.paper6_2_mlx.routed_qa import route_qa_documents
from experiments.paper6_2_mlx.run_answer_quality_pressure import QADocument, QAExample


def _example() -> QAExample:
    documents = (
        QADocument("noise", "Unrelated", "A paragraph about baking bread."),
        QADocument("answer", "Paris", "Paris is the capital city of France."),
        QADocument("other", "Berlin", "Berlin is the capital city of Germany."),
    )
    return QAExample(
        dataset="fixture",
        example_id="one",
        question="What is the capital city of France?",
        answer="Paris",
        source="Paris is the capital city of France.",
        source_scope="fixture",
        documents=documents,
        evidence_document_ids=frozenset(("answer",)),
    )


def test_hybrid_router_recovers_lexically_grounded_evidence():
    result = route_qa_documents(_example(), top_k=2)
    assert result.ranked_document_ids[0] == "answer"
    assert result.evidence_recall_at_1 == 1.0
    assert result.selected_evidence_recall == 1.0
    assert "Paris is the capital" in result.selected_source
    assert result.index_bytes > 0


def test_hybrid_router_validates_top_k_and_documents():
    example = _example()
    try:
        route_qa_documents(example, top_k=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("top_k=0 should fail")

    empty = QAExample("fixture", "empty", "Q?", "A", "", "fixture")
    try:
        route_qa_documents(empty)
    except ValueError as error:
        assert "no routable documents" in str(error)
    else:
        raise AssertionError("an empty candidate set should fail")
