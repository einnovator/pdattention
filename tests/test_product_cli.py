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
    for command in (
        "doctor", "engines", "inspect", "evaluate", "recommend", "report", "serve",
        "qualify", "assess", "model", "adapter", "profiles", "bundle", "runtime",
        "agent", "gateway", "hf",
    ):
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
    (run / "structural_adapter").mkdir()
    (run / "structural_adapter" / "pra_adapter.yaml").write_text(
        "schema_version: 1\nsource: test\n", encoding="utf-8"
    )
    payload = __import__("yaml").safe_load((run / "pra.yaml").read_text(encoding="utf-8"))
    payload["model"] = {
        "id": "unknown/model",
        "revision": "0123456789abcdef",
        "architecture": "TestForCausalLM",
    }
    payload["structural_adapter"] = {"path": "structural_adapter", "status": "candidate"}
    payload["learned_adapters"] = {}
    (run / "pra.yaml").write_text(
        __import__("yaml").safe_dump(payload, sort_keys=False), encoding="utf-8"
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
    assert json.loads(inspected.output)["schema_version"] == 2


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


def test_normal_help_uses_product_terms_only() -> None:
    runner = CliRunner()
    commands = (
        [], ["gateway", "serve", "--help"], ["serve", "--help"],
        ["evaluate", "--help"], ["qualify", "--help"],
    )
    output = "\n".join(runner.invoke(cli, [*args, "--help"] if not args else args).output for args in commands)

    for internal in ("E0", "E1", "E2", "E3", "G10", "G11"):
        assert internal not in output
    assert "selected-context" in output
    assert "native-memory" in output
    assert "native-serving" in output


def test_serve_mode_alias_and_explanation_use_shared_status_axes(monkeypatch) -> None:
    class Handle:
        def to_dict(self):
            return {"engine": "ollama"}

    class Health:
        status = "ready"

    monkeypatch.setattr("pra_hf.cli.RuntimeManager.serve", lambda self, config: Handle())
    monkeypatch.setattr("pra_hf.cli.RuntimeManager.health", lambda self, handle: Health())
    result = CliRunner().invoke(
        cli,
        ["serve", "org/model", "-e", "ollama", "-m", "auto", "--explain"],
    )

    assert result.exit_code == 0, result.output
    assert "Requested: auto" in result.output
    assert "mechanism:" in result.output
    assert "quality:" in result.output
    assert "economics:" in result.output
    assert "recommendation:" in result.output
    assert "Resolved: selected-context" in result.output


def test_explicit_unqualified_native_serving_is_rejected() -> None:
    result = CliRunner().invoke(
        cli,
        ["serve", "org/model", "-e", "ollama", "-m", "native-serving"],
    )

    assert result.exit_code != 0
    assert "not qualified" in result.output


def test_documented_product_workflow_commands_exist() -> None:
    runner = CliRunner()
    commands = (
        ["doctor"], ["engines"], ["inspect"], ["evaluate"], ["recommend"],
        ["report"], ["serve"], ["qualify", "native-memory"],
        ["qualify", "native-serving"], ["assess", "init"], ["assess", "run"],
        ["assess", "report"],
    )

    for command in commands:
        result = runner.invoke(cli, [*command, "--help"])
        assert result.exit_code == 0, f"{' '.join(command)}: {result.output}"
