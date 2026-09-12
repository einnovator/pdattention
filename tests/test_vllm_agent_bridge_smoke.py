from __future__ import annotations

import pytest

from pra_vllm.agent_executor import (
    record_rounded_selected_indices,
)


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


def test_record_rounded_selector_fails_when_no_whole_group_fits_floor() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old observation"},
        {"role": "assistant", "content": "recent action"},
        {"role": "user", "content": "recent observation"},
    ]
    spans = {index: (index * 32, (index + 1) * 32) for index in range(6)}
    with pytest.raises(RuntimeError, match="No whole old causal-group"):
        record_rounded_selected_indices(
            messages,
            spans,
            source_tokens=192,
            block_size=16,
            retention_fraction=0.9,
        )
