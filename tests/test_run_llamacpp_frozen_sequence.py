import argparse
import json

import pytest

from experiments.paper4_5_agent.run_llamacpp_frozen_sequence import (
    build_command,
    run_sequence,
    summarize,
)


def _args(tmp_path):
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"request_index": 1}\n', encoding="utf-8")
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        base_url="http://127.0.0.1:18082",
        request_replay=replay,
        selection_fixture=fixture,
        output_dir=tmp_path / "out",
        full_retention=True,
        tokenizer="tokenizer",
        tokenizer_revision=None,
        model_label="model",
        model_fingerprint="model-sha",
        tokenizer_fingerprint="tokenizer-sha",
        engine_revision="engine-sha",
        experiment_revision="experiment-sha",
        source_slot=0,
        request_slot=1,
        continuation_tokens=2,
        seed=0,
        local_files_only=True,
    )


def test_full_command_binds_strict_full_retention(tmp_path):
    args = _args(tmp_path)
    command = build_command(args, 1, tmp_path / "request.json")
    assert "--frozen-full-retention" in command
    assert "--require-strict-resident-subset" in command
    assert "--local-files-only" in command


def test_existing_output_directory_is_never_overwritten(tmp_path):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="Immutable output directory"):
        run_sequence(args)


def test_summary_names_repeat_exact_claim_boundary(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "engine": "llama.cpp-metal",
                "model": "model",
                "request_index": 1,
                "request_input_sha256": "digest",
                "logical_source_tokens": 90,
                "selected_logical_kv_tokens": 50,
                "wire_suffix_tokens": 10,
                "full_visible_tokens": 100,
                "total_visible_tokens": 60,
                "qualified": True,
                "qualification_blockers": [],
                "selected_history_reencoded_tokens": 0,
                "selected_history_kv_copy_bytes": 0,
                "checks": {
                    "repeat_token_exact": True,
                    "selected_history_not_reencoded": True,
                    "selected_original_kv_not_copied": True,
                    "source_positions_preserved": True,
                    "lifecycle_empty": True,
                },
            }
        ),
        encoding="utf-8",
    )
    result = summarize([path], (1,), full=False)
    assert result["sequence_qualified"] is True
    assert result["weighted_realized_retention_fraction"] == 0.6
    assert "not_independent_dense_logit_reference" in result["claim_boundary"]
