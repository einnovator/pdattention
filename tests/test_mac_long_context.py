from experiments.mac_scaling.run_mlx_long_context import (
    build_long_source_tokens,
    unique_seed_cohort,
)
from experiments.paper6_2_mlx.run_answer_quality_pressure import QAExample, QADocument


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [len(token) for token in text.split()]


def _example(identifier: str, text: str) -> QAExample:
    return QAExample(
        dataset="test",
        example_id=identifier,
        question="question",
        answer="answer",
        source=text,
        source_scope="test",
        documents=(QADocument(identifier, identifier, text),),
        evidence_document_ids=frozenset({identifier}),
    )


def test_unique_seed_cohort_never_repeats_an_example() -> None:
    cohort = unique_seed_cohort(
        [_example(str(index), f"document {index}") for index in range(8)], 5
    )

    assert len(cohort) == 5
    assert len({example.example_id for _, example in cohort}) == 5


def test_long_source_keeps_selected_prefix_and_hits_exact_budget() -> None:
    tokenizer = _Tokenizer()
    examples = [_example("d1", "alpha beta"), _example("d2", "gamma delta")]

    tokens, selected_count = build_long_source_tokens(
        tokenizer,
        "answer evidence",
        examples,
        target_tokens=12,
        max_selected_tokens=4,
        seed=11,
    )

    assert tokens[:2] == tokenizer.encode("answer evidence")
    assert selected_count == 2
    assert len(tokens) == 12
