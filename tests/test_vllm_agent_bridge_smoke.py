from __future__ import annotations

import pytest

from pra_vllm.agent_executor import (
    record_rounded_selected_indices,
    validate_retention_fractions,
)


def test_bridge_smoke_can_run_pra100_without_starting_a_sparse_arm() -> None:
    assert validate_retention_fractions([1.0]) == (1.0,)


@pytest.mark.parametrize("values", ([], [0.9], [1.0, 1.0], [1.0, 0.0]))
def test_bridge_smoke_rejects_invalid_retention_arm_sets(values) -> None:
    with pytest.raises(ValueError):
        validate_retention_fractions(values)


def test_record_rounded_selector_preserves_pairs_and_rounds_up() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old observation"},
        {"role": "assistant", "content": "middle action"},
        {"role": "user", "content": "middle observation"},
        {"role": "assistant", "content": "recent action"},
        {"role": "user", "content": "recent observation"},
    ]
    spans = {index: (index * 16, (index + 1) * 16) for index in range(8)}
    selected, pages = record_rounded_selected_indices(
        messages,
        spans,
        source_tokens=128,
        block_size=16,
        retention_fraction=0.75,
    )
    assert 2 not in selected and 3 not in selected
    assert 4 in selected and 5 in selected
    assert 6 in selected and 7 in selected
    assert len(pages) * 16 == 96


def test_record_rounded_selector_rounds_to_full_when_no_sparse_group_fits() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old observation"},
        {"role": "assistant", "content": "recent action"},
        {"role": "user", "content": "recent observation"},
    ]
    spans = {index: (index * 32, (index + 1) * 32) for index in range(6)}
    selected, pages = record_rounded_selected_indices(
        messages,
        spans,
        source_tokens=192,
        block_size=16,
        retention_fraction=0.9,
    )
    assert selected == tuple(range(6))
    assert len(pages) * 16 == 192


def test_record_rounded_selector_never_drops_logically_selected_groups() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "selected old action"},
        {"role": "user", "content": "selected old observation"},
        {"role": "assistant", "content": "other old action"},
        {"role": "user", "content": "other old observation"},
        {"role": "assistant", "content": "recent action"},
        {"role": "user", "content": "recent observation"},
    ]
    spans = {index: (index * 16, (index + 1) * 16) for index in range(8)}
    selected, pages = record_rounded_selected_indices(
        messages,
        spans,
        source_tokens=128,
        block_size=16,
        retention_fraction=0.75,
        required_message_indices=(0, 1, 2, 3, 6, 7),
    )
    assert {0, 1, 2, 3, 6, 7}.issubset(selected)
    assert len(pages) * 16 >= 96
