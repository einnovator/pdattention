import inspect

import pytest
import torch

from pra_hf.dynamic_query import (
    build_dynamic_query_facets,
    reconstruction_text,
    render_reconstructed_query,
)


def test_reconstruction_modes_are_deterministic_and_preserve_exact_regions():
    question = "Where was the author born?"
    memory = "The author was Ada Example."
    for mode in (
        "query_then_memory",
        "query_separator_memory",
        "memory_then_query",
    ):
        first = reconstruction_text(question, memory, mode)
        second = reconstruction_text(question, memory, mode)
        assert first == second
        assert first.content[slice(*first.question_char_span)] == question
        assert first.content[slice(*first.memory_char_span)] == memory
    assert "[Retrieved memory]" in reconstruction_text(
        question, memory, "query_separator_memory"
    ).content


def test_reconstruction_rejects_unknown_mode_and_empty_inputs():
    with pytest.raises(ValueError, match="Unsupported"):
        reconstruction_text("q", "a", "unknown")
    with pytest.raises(ValueError, match="non-empty"):
        reconstruction_text("", "a", "query_then_memory")


def test_dynamic_facets_reuse_contextual_states_and_restrict_support():
    hidden = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    question_only = build_dynamic_query_facets(
        hidden,
        question_span=(1, 4),
        memory_span=(5, 9),
        support_mode="question_only",
        window=2,
        stride=1,
    )
    combined = build_dynamic_query_facets(
        hidden,
        question_span=(1, 4),
        memory_span=(5, 9),
        support_mode="question_and_memory",
        window=2,
        stride=1,
    )
    assert torch.equal(question_only.hidden[0], hidden[-1])
    assert {row.family for row in question_only.provenance[1:]} == {
        "question_window_2"
    }
    assert {row.family for row in combined.provenance[1:]} == {
        "question_window_2",
        "memory_a_window_2",
    }
    assert len(combined.provenance) > len(question_only.provenance)


def test_question_only_support_cannot_nominate_memory_tokens_directly():
    hidden = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    facets = build_dynamic_query_facets(
        hidden,
        question_span=(0, 2),
        memory_span=(3, 5),
        support_mode="question_only",
        window=1,
        include_global=False,
    )
    assert all(row.token_end <= 2 for row in facets.provenance)
    assert torch.equal(facets.hidden, hidden[:2])


def test_reconstruction_api_has_no_oracle_target_channel():
    for function in (reconstruction_text, render_reconstructed_query):
        parameters = set(inspect.signature(function).parameters)
        assert not parameters & {
            "target",
            "targets",
            "target_parent",
            "target_parent_ids",
            "oracle_parent",
        }


def test_tokenizer_truncation_side_is_restored_when_encoding_fails():
    class FailingTokenizer:
        chat_template = None
        truncation_side = "right"

        def __call__(self, *args, **kwargs):
            raise RuntimeError("encoding failed")

    tokenizer = FailingTokenizer()
    with pytest.raises(RuntimeError, match="encoding failed"):
        render_reconstructed_query(
            tokenizer,
            "question",
            "active memory",
            "query_then_memory",
            max_tokens=32,
        )
    assert tokenizer.truncation_side == "right"
