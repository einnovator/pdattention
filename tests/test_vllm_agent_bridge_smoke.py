from __future__ import annotations

import pytest

from pra_vllm.agent_executor import (
    record_rounded_selected_indices,
    validate_retention_fractions,
)
from experiments.paper4_5_agent.run_vllm_cuda_agent_bridge_smoke import _llm_options


def test_bridge_smoke_pins_bounded_vllm_admission_options(tmp_path) -> None:
    args = type(
        "Args",
        (),
        {
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "revision": "immutable-revision",
            "dtype": "bfloat16",
            "max_model_len": 8192,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.86,
            "cpu_offload_gb": 8.0,
            "kv_transfer_buffer_bytes": 268_435_456,
            "kv_cache_memory_bytes": 536_870_912,
        },
    )()
    options = _llm_options(args, tmp_path)
    assert options["revision"] == "immutable-revision"
    assert options["dtype"] == "bfloat16"
    assert options["cpu_offload_gb"] == 8.0
    assert options["max_num_batched_tokens"] == 2048
    assert options["kv_transfer_config"]["kv_buffer_size"] == 268_435_456
    assert options["kv_cache_memory_bytes"] == 536_870_912


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
