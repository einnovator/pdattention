import argparse
import json
from pathlib import Path

import pytest

from experiments.paper4_5_agent.run_frozen_request_sequence import (
    _request_indices,
    build_command,
    run_sequence,
)


def _args(tmp_path: Path, engine: str = "mlx") -> argparse.Namespace:
    replay = tmp_path / "replay.jsonl"
    replay.write_text(
        "\n".join(json.dumps({"request_index": index}) for index in (1, 2)),
        encoding="utf-8",
    )
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("{}\n", encoding="utf-8")
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        engine=engine,
        request_replay=replay,
        selection_fixture=fixture,
        output_dir=tmp_path / "out",
        full_retention=False,
        model="test-model",
        revision="test-revision",
        provenance=provenance,
        continuation_tokens=2,
        prefill_step_size=1,
        source_prefill_step_size=512,
        max_abs_logit_delta=0.01,
        hardware_label="test-host",
        device="cuda",
        dtype="bfloat16",
        local_files_only=True,
    )


def test_request_replay_requires_contiguous_identity(tmp_path):
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"request_index": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous identities"):
        _request_indices(replay)


def test_sglang_command_binds_provenance_and_revision(tmp_path):
    args = _args(tmp_path, engine="sglang")
    command = build_command(args, 2, tmp_path / "request.json")
    assert "experiments.paper4_5_agent.run_sglang_live_agent_kv_lifecycle" in command
    assert command[command.index("--provenance") + 1] == str(args.provenance)
    assert command[command.index("--revision") + 1] == "test-revision"
    assert "--fused-disjoint-attention" not in command


def test_hf_command_binds_cuda_dtype_without_mlx_only_arguments(tmp_path):
    args = _args(tmp_path, engine="hf")
    command = build_command(args, 1, tmp_path / "request.json")
    assert "experiments.paper4_5_agent.run_hf_live_agent_kv_lifecycle" in command
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--local-files-only" in command
    assert "--source-prefill-step-size" not in command
    assert "--materialization-policy" not in command
    assert "--hardware-label" not in command


def test_existing_output_directory_is_never_overwritten(tmp_path):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="Immutable output directory"):
        run_sequence(args)
