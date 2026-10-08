import argparse
import json
from pathlib import Path

import pytest

from experiments.paper4_5_agent.run_vllm_cuda_frozen_sequence import (
    build_command,
    summarize,
)


def _payload(index: int, *, full: bool = False) -> dict:
    source = 1000 + 10 * index
    selected = source if full else 600 + 10 * index
    resident = source + (16 - source % 16) % 16
    selected_pages = selected + (16 - selected % 16) % 16
    suffix = 100
    copies = {
        "selected_history_kv_copy_bytes": 0,
        "receipt_capture_d2h_bytes": 0,
        "receipt_host_pack_copy_bytes": 0,
        "receipt_h2d_bytes": 0,
        "receipt_intermediate_cow_bytes": 0,
        "receipt_final_cow_bytes": 0,
        "terminal_page_copy_bytes_per_request": 0,
    }
    return {
        "qualified": True,
        "qualification_blockers": [],
        "engine": "vllm-cuda",
        "model": "Qwen/Qwen3-0.6B",
        "request_index": index,
        "request_input_sha256": f"digest-{index}",
        "source_policy": "persistent_instruction_epoch_retirement",
        "logical_source_tokens": source,
        "resident_source_tokens": resident,
        "wire_suffix_tokens": suffix,
        "selected_logical_kv_tokens": selected,
        "selected_original_page_tokens": selected_pages,
        "selected_history_page_overhead_tokens": selected_pages - selected,
        "materialized_history_tokens": 0,
        "total_visible_tokens": selected + suffix,
        "full_visible_tokens": source + suffix,
        "selected_history_reencoded_tokens": 0,
        **copies,
        "checks": {
            "all_receipts_materialized": True,
            "all_original_pages_zero_copy": True,
            "selected_history_not_reencoded": True,
            "repeat_token_exact": True,
            "final_alias_installed_twice": True,
            "final_alias_released_twice": True,
            "lifecycle_empty": True,
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_summarize_reports_distinct_logical_resident_and_visible_savings(tmp_path):
    paths = [
        _write(tmp_path / f"request_{index:02d}.json", _payload(index))
        for index in (1, 2)
    ]
    result = summarize(paths, (1, 2), False)
    assert result["sequence_qualified"] is True
    assert result["qualified_requests"] == 2
    assert result["selected_history_reencoded_tokens"] == 0
    assert result["selected_history_kv_copy_bytes"] == 0
    assert 0 < result["logical_history_omission_fraction"] < 1
    assert 0 < result["resident_original_kv_omission_fraction"] < 1
    assert 0 < result["visible_context_omission_fraction"] < 1


def test_full_retention_requires_full_logical_and_visible_context(tmp_path):
    path = _write(tmp_path / "request_01.json", _payload(1, full=True))
    assert summarize([path], (1,), True)["sequence_qualified"] is True
    payload = _payload(1, full=True)
    payload["total_visible_tokens"] -= 1
    path = _write(tmp_path / "request_01_bad.json", payload)
    assert summarize([path], (1,), True)["sequence_qualified"] is False


def test_copy_or_reencoding_fails_sequence(tmp_path):
    payload = _payload(1)
    payload["selected_history_kv_copy_bytes"] = 16
    path = _write(tmp_path / "request_01.json", payload)
    result = summarize([path], (1,), False)
    assert result["sequence_qualified"] is False
    assert result["selected_history_kv_copy_bytes"] == 16


def test_rejects_duplicate_request_identity(tmp_path):
    first = _write(tmp_path / "a.json", _payload(1))
    second_payload = _payload(2)
    second_payload["request_input_sha256"] = "digest-1"
    second = _write(tmp_path / "b.json", second_payload)
    with pytest.raises(ValueError, match="not unique"):
        summarize([first, second], (1, 2), False)


def test_build_command_uses_per_request_storage_and_full_flag(tmp_path):
    args = argparse.Namespace(
        request_replay=tmp_path / "replay.jsonl",
        selection_fixture=tmp_path / "plan.jsonl",
        storage_root=tmp_path / "storage",
        model="Qwen/Qwen3-0.6B",
        max_model_len=32768,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.84,
        kv_transfer_buffer_bytes=200_000_000,
        continuation_tokens=2,
        full_retention=True,
    )
    command = build_command(args, 3, tmp_path / "request_03.json")
    assert "--frozen-full-retention" in command
    assert str(tmp_path / "storage" / "request_03") in command
    assert command[command.index("--request-index") + 1] == "3"
