import argparse
import json
from pathlib import Path

import pytest

from experiments.paper4_5_agent.run_frozen_request_sequence import (
    _request_indices,
    _selected_request_indices,
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
        request_indices=None,
        full_retention=False,
        model="test-model",
        revision="test-revision",
        provenance=provenance,
        continuation_tokens=2,
        prefill_step_size=1,
        source_prefill_step_size=512,
        max_abs_logit_delta=0.01,
        hardware_label="test-host",
        max_model_len=32768,
        gpu_memory_utilization=0.10,
        engine_source_revision="engine-revision",
        artifact_overlay=False,
        device="cuda",
        dtype="bfloat16",
        local_files_only=True,
        reuse_hf_model=False,
    )


def test_request_replay_requires_contiguous_identity(tmp_path):
    replay = tmp_path / "replay.jsonl"
    replay.write_text('{"request_index": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous identities"):
        _request_indices(replay)


def test_request_index_segment_is_ordered_and_bounded():
    available = (1, 2, 3, 4, 5, 6, 7, 8)
    assert _selected_request_indices(available, "4-6,8") == (4, 5, 6, 8)
    with pytest.raises(ValueError, match="outside"):
        _selected_request_indices(available, "3,9")
    with pytest.raises(ValueError, match="reversed"):
        _selected_request_indices(available, "6-4")


def test_sglang_command_binds_provenance_and_revision(tmp_path):
    args = _args(tmp_path, engine="sglang")
    command = build_command(args, 2, tmp_path / "request.json")
    assert "experiments.paper4_5_agent.run_sglang_live_agent_kv_lifecycle" in command
    assert command[command.index("--provenance") + 1] == str(args.provenance)
    assert command[command.index("--revision") + 1] == "test-revision"
    assert "--fused-disjoint-attention" not in command


def test_hf_command_binds_cuda_dtype_revision_without_mlx_only_arguments(tmp_path):
    args = _args(tmp_path, engine="hf")
    command = build_command(args, 1, tmp_path / "request.json")
    assert "experiments.paper4_5_agent.run_hf_live_agent_kv_lifecycle" in command
    assert command[command.index("--device") + 1] == "cuda"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--revision") + 1] == "test-revision"
    assert "--local-files-only" in command
    assert "--source-prefill-step-size" not in command
    assert "--materialization-policy" not in command
    assert "--hardware-label" not in command


def test_vllm_metal_command_uses_engine_contract_without_tensor_harness_flags(tmp_path):
    args = _args(tmp_path, engine="vllm-metal")
    args.artifact_overlay = True
    command = build_command(args, 2, tmp_path / "request.json")
    assert "experiments.paper4_5_agent.run_vllm_metal_live_kv_lifecycle" in command
    assert command[command.index("--max-model-len") + 1] == "32768"
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.1"
    assert command[command.index("--engine-source-revision") + 1] == "engine-revision"
    assert command[command.index("--hardware-label") + 1] == "test-host"
    assert "--artifact-overlay" in command
    assert "--prefill-step-size" not in command
    assert "--source-prefill-step-size" not in command
    assert "--max-abs-logit-delta" not in command


def test_existing_output_directory_is_never_overwritten(tmp_path):
    args = _args(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="Immutable output directory"):
        run_sequence(args)


def test_shared_hf_mode_loads_once_and_runs_only_requested_segment(
    tmp_path, monkeypatch
):
    args = _args(tmp_path, engine="hf")
    args.request_indices = "2"
    args.reuse_hf_model = True
    replay_rows = []
    fixture_rows = []
    for index in (1, 2):
        digest = f"digest-{index}"
        replay_rows.append(
            {"request_index": index, "request_input_sha256": digest}
        )
        fixture_rows.append(
            {
                "request_index": index,
                "request_input_sha256": digest,
                "source_plan_digest": f"plan-{index}",
                "selected_resource_digest": f"resource-{index}",
                "source_wire_plan_digest": f"wire-{index}",
            }
        )
    args.request_replay.write_text(
        "\n".join(json.dumps(row) for row in replay_rows) + "\n",
        encoding="utf-8",
    )
    args.selection_fixture.write_text(
        "\n".join(json.dumps(row) for row in fixture_rows) + "\n",
        encoding="utf-8",
    )

    import experiments.paper4_5_agent.run_hf_live_agent_kv_lifecycle as hf

    loads = []
    runs = []
    monkeypatch.setattr(
        hf,
        "load_hf_components",
        lambda namespace: loads.append(namespace.model) or ("tokenizer", "model"),
    )

    def fake_run(namespace, *, tokenizer, model):
        runs.append((namespace.request_index, tokenizer, model))
        payload = {
            "engine": "transformers-pytorch",
            "model": namespace.model,
            "request_index": namespace.request_index,
            "request_input_sha256": f"digest-{namespace.request_index}",
            "source_policy": "frozen-policy",
            "source_tokens": 100,
            "wire_suffix_tokens": 10,
            "selected_kv_tokens": 50,
            "realized_retention_fraction": 60 / 110,
            "selected_text_reencoded_tokens": 0,
            "interval_pack_bytes": 0,
            "physical_kv_copy": False,
            "engine_lifecycle_qualified": True,
            "qualification_blockers": [],
            "checks": {"same_subset_token_exact": True},
        }
        namespace.output.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(hf, "run", fake_run)
    result = run_sequence(args)
    assert loads == ["test-model"]
    assert runs == [(2, "tokenizer", "model")]
    assert result["requested_requests"] == [2]
    assert result["available_requests"] == [1, 2]
    assert result["shared_model_process"] is True
    assert result["frozen_ledger_bound"] is True
