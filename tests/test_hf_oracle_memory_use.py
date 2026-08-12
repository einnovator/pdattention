"""Focused portability checks for the HF causal memory-use probe."""

import torch

from experiments.paper2_hf.qa.run_oracle_memory_use import _decoder_hidden_tensor


def test_decoder_hidden_tensor_accepts_tuple_decoder_outputs():
    """Gemma decoder layers return tuples, unlike some wrapped Qwen paths."""
    expected = torch.ones(1, 3, 4)
    actual = _decoder_hidden_tensor((expected, None))

    assert actual is expected
