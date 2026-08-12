"""Correctness tests for zero-parameter HF routing-query aggregation."""

from __future__ import annotations

import pytest
import torch

from experiments.paper2_hf.routing.run_query_strategies import _prompt_with_question_span
from pra_torch.hf import (
    QUERY_EXPONENTIAL,
    QUERY_LAST,
    QUERY_LINEAR,
    QUERY_QUESTION_EXPONENTIAL,
    QUERY_QUESTION_MEAN,
    QUERY_UNIFORM,
    aggregate_query_states,
    half_life_to_decay,
    streaming_exponential_query,
    token_span_from_offsets,
)


def _states(tokens: int = 4) -> torch.Tensor:
    return torch.arange(1, tokens + 1, dtype=torch.float32).view(1, tokens, 1)


def test_last_query_is_exact_final_state_view():
    states = torch.randn(2, 5, 7)
    assert torch.equal(aggregate_query_states(states, QUERY_LAST), states[:, -1, :])


def test_uniform_query_clips_window_to_available_tokens_and_normalizes_weights():
    states = _states()
    assert torch.allclose(
        aggregate_query_states(states, QUERY_UNIFORM, window=2),
        torch.tensor([[3.5]]),
    )
    assert torch.allclose(
        aggregate_query_states(states, QUERY_UNIFORM, window=99),
        torch.tensor([[2.5]]),
    )


def test_exponential_query_uses_newest_first_half_life_weights():
    states = _states(3)
    actual = aggregate_query_states(
        states,
        QUERY_EXPONENTIAL,
        window=3,
        half_life=1.0,
    )
    expected = torch.tensor([[(1 * 0.25 + 2 * 0.5 + 3) / 1.75]])
    assert half_life_to_decay(1.0) == pytest.approx(0.5)
    assert torch.allclose(actual, expected)


def test_linear_query_weights_newer_states_more_heavily():
    actual = aggregate_query_states(_states(3), QUERY_LINEAR, window=3)
    assert torch.allclose(actual, torch.tensor([[(1 + 4 + 9) / 6]]))


def test_question_span_mean_and_decay_use_only_explicit_span():
    states = _states(5)
    span = [(1, 4)]
    mean = aggregate_query_states(states, QUERY_QUESTION_MEAN, token_spans=span)
    decayed = aggregate_query_states(
        states,
        QUERY_QUESTION_EXPONENTIAL,
        half_life=1.0,
        token_spans=span,
    )
    assert torch.allclose(mean, torch.tensor([[3.0]]))
    assert torch.allclose(decayed, torch.tensor([[(2 * 0.25 + 3 * 0.5 + 4) / 1.75]]))


def test_token_span_extraction_uses_overlap_and_rejects_truncated_span():
    offsets = [(0, 0), (0, 3), (4, 9), (10, 14), (0, 0)]
    assert token_span_from_offsets(offsets, 4, 14) == (2, 4)
    with pytest.raises(ValueError, match="does not overlap"):
        token_span_from_offsets(offsets, 20, 24)


def test_prompt_span_normalizes_dataset_boundary_whitespace():
    class TrimmingTokenizer:
        chat_template = "trim"
        truncation_side = "right"

        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"].strip()

        def __call__(self, text, **_kwargs):
            offsets = torch.tensor([[(i, i + 1) for i in range(len(text))]])
            return {
                "input_ids": torch.arange(len(text)).unsqueeze(0),
                "attention_mask": torch.ones(1, len(text), dtype=torch.long),
                "offset_mapping": offsets,
            }

    encoded, span = _prompt_with_question_span(
        TrimmingTokenizer(), "Does trimming preserve this question?  ", 128
    )

    assert encoded["input_ids"].shape[1] > 0
    assert span[1] - span[0] == len("Does trimming preserve this question?")


def test_streaming_ema_matches_normalized_full_history_exponential_pooling():
    states = torch.randn(3, 11, 5)
    expected = aggregate_query_states(
        states,
        QUERY_EXPONENTIAL,
        window=states.shape[1],
        half_life=4.0,
    )
    actual = streaming_exponential_query(states, half_life=4.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_query_strategy_validation_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="Unsupported"):
        aggregate_query_states(_states(), "median", window=2)
    with pytest.raises(ValueError, match="positive window"):
        aggregate_query_states(_states(), QUERY_UNIFORM, window=0)
    with pytest.raises(ValueError, match="half-life"):
        aggregate_query_states(_states(), QUERY_EXPONENTIAL, window=2)
    with pytest.raises(ValueError, match="one token span"):
        aggregate_query_states(_states(), QUERY_QUESTION_MEAN)
    with pytest.raises(ValueError, match="positive"):
        half_life_to_decay(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_query_aggregation_cpu_gpu_parity():
    states = torch.randn(2, 13, 9)
    cpu = aggregate_query_states(
        states,
        QUERY_EXPONENTIAL,
        window=8,
        half_life=4.0,
    )
    gpu = aggregate_query_states(
        states.cuda(),
        QUERY_EXPONENTIAL,
        window=8,
        half_life=4.0,
    ).cpu()
    assert torch.allclose(gpu, cpu, atol=2e-6, rtol=2e-6)
