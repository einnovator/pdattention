from pathlib import Path

from experiments.rag_vs_pra.datasets import controlled_fixture, select_cohort


def test_controlled_fixture_is_stable_and_multihop() -> None:
    documents, questions, metadata = controlled_fixture(seed=11, document_count=30)
    assert len(documents) == 30
    assert len(questions) == 15
    assert any(len(question.gold_document_ids) > 1 for question in questions)
    assert len(metadata["corpus_sha256"]) == 64
    assert controlled_fixture(seed=11, document_count=30)[2] == metadata


def test_cohort_selection_is_seeded_and_order_preserving() -> None:
    _, questions, _ = controlled_fixture(seed=11)
    first = select_cohort(questions, max_examples=5, seed=23)
    second = select_cohort(questions, max_examples=5, seed=23)
    assert first == second
    positions = [questions.index(question) for question in first]
    assert positions == sorted(positions)
