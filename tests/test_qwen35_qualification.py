from collections import UserDict

import pytest
import torch

from experiments.paper4_5_runtime.run_qwen35_qualification import (
    _prompt,
    select_chunk_indices,
)
from experiments.paper4_5_runtime.build_hf_catalog_bundles import (
    SPECS,
    _structural_adapter,
)
from experiments.paper6_2_mlx.run_matched_e0_e2 import _stop_token_ids


def test_selected_chunks_respect_budget_and_score_order() -> None:
    selected = select_chunk_indices(
        torch.tensor([0.1, 0.9, 0.8]),
        [(0, 4), (4, 8), (8, 12)],
        token_budget=8,
    )

    assert selected == [1, 2]


def test_selected_chunks_keep_one_candidate_for_tiny_budget() -> None:
    assert select_chunk_indices(
        torch.tensor([0.1, 0.9]), [(0, 8), (8, 16)], token_budget=1
    ) == [1]


def test_selected_chunks_validate_shapes_and_budget() -> None:
    with pytest.raises(ValueError, match="same candidates"):
        select_chunk_indices(torch.tensor([1.0]), [(0, 1), (1, 2)], token_budget=1)
    with pytest.raises(ValueError, match="positive"):
        select_chunk_indices(torch.tensor([1.0]), [(0, 1)], token_budget=0)


def test_qwen35_bundle_declares_hybrid_topology_without_native_memory() -> None:
    spec = SPECS["qwen3.5-27b"]
    structural = _structural_adapter(spec)

    assert spec["native_memory_available"] is False
    assert spec["consumer_layers"] == list(range(3, 64, 4))
    assert structural["mapping"]["layers"] == "language_model.model.layers"
    assert structural["position"]["type"] == "partial_rope"


def test_prompt_tokenizes_string_returned_by_mlx_tokenizer_wrapper() -> None:
    class Tokenizer:
        chat_template = "template"

        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return "rendered prompt"

        @staticmethod
        def encode(text, **_kwargs):
            assert text == "rendered prompt"
            return [4, 5, 6]

    assert _prompt(Tokenizer(), "evidence", "question") == [4, 5, 6]


def test_prompt_accepts_batch_encoding_mapping() -> None:
    class Tokenizer:
        chat_template = "template"

        @staticmethod
        def apply_chat_template(*_args, **_kwargs):
            return UserDict({"input_ids": [[7, 8, 9]]})

    assert _prompt(Tokenizer(), "evidence", "question") == [7, 8, 9]


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [({"eos_token_id": 9}, {9}), ({"eos_token_ids": {9, 10}}, {9, 10}), ({}, set())],
)
def test_mlx_generation_normalizes_stop_tokens(attributes, expected) -> None:
    tokenizer = type("Tokenizer", (), attributes)()

    assert _stop_token_ids(tokenizer) == expected
