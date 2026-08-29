from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from pra_hf.cli import cli, deprecated_cli


def test_canonical_cli_tree_and_deprecated_alias() -> None:
    runner = CliRunner()
    current = runner.invoke(cli, ["--help"])
    legacy = runner.invoke(deprecated_cli, ["doctor", "--json"])

    assert current.exit_code == 0
    for command in ("model", "adapter", "profiles", "bundle", "runtime", "agent", "gateway", "hf", "doctor"):
        assert command in current.output
    assert legacy.exit_code == 0
    assert "`pra-hf` is deprecated; use `pra`." in legacy.output


def test_model_inspect_uses_config_metadata_without_weights(monkeypatch) -> None:
    config = SimpleNamespace(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        num_hidden_layers=28,
        hidden_size=1024,
        vocab_size=1000,
        intermediate_size=3072,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=64,
        rope_theta=1_000_000,
        _commit_hash="abc123",
    )
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: config)

    result = CliRunner().invoke(cli, ["model", "inspect", "org/model", "--json"])

    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["model"]["revision"] == "abc123"
    assert value["pra"]["structural_adapter"]["source"] == "builtin:qwen"
    assert value["attention"]["kv_heads"] == 8


def test_model_adapt_writes_declarative_adapter_and_validation(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(
        model_type="llama", architectures=["LlamaForCausalLM"], num_hidden_layers=4,
        hidden_size=64, vocab_size=128, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rope_theta=10_000, _commit_hash="rev",
    )
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *args, **kwargs: config)
    output = tmp_path / "adapter"

    result = CliRunner().invoke(cli, ["model", "adapt", "org/llama", "-o", str(output), "--json"])

    assert result.exit_code == 0, result.output
    assert (output / "pra_adapter.yaml").is_file()
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    assert validation["stages"][0]["status"] == "PASS"
    assert validation["stages"][1]["status"] == "DEFERRED_WEIGHT_LOAD"


def test_profile_calibration_and_bundle_workflow_is_offline(tmp_path) -> None:
    run = tmp_path / "run"
    calibrated = CliRunner().invoke(
        cli, ["profiles", "calibrate", "unknown/model", "-o", str(run), "--json"]
    )
    bundle = tmp_path / "bundle"
    built = CliRunner().invoke(cli, ["bundle", "build", str(run), "-o", str(bundle), "--json"])
    inspected = CliRunner().invoke(cli, ["bundle", "inspect", str(bundle), "--json"])

    assert calibrated.exit_code == 0, calibrated.output
    assert json.loads(calibrated.output)["profiles"]["QUALITY_MAX_CANDIDATE"]["status"] == "CALIBRATION_PENDING"
    assert built.exit_code == 0, built.output
    assert inspected.exit_code == 0, inspected.output
    assert (bundle / "bundle.yaml").is_file()
    assert (bundle / "README.md").is_file()


def test_hf_push_dry_run_never_requires_hub_dependency(tmp_path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.yaml").write_text("schema_version: 1\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["hf", "push", str(bundle), "owner/repo", "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dry_run"] is True


def test_runtime_cli_forwards_engine_arguments() -> None:
    result = CliRunner().invoke(
        cli,
        ["runtime", "inspect", "org/model", "-e", "vllm", "--engine-arg", "tensor-parallel-size=2", "--json"],
    )

    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["engine"] == "vllm"
    assert value["capabilities"]["native_kv"] is False

