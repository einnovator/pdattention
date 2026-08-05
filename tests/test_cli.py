from click.testing import CliRunner

from config.model_size import estimate_model_size
from pra_torch.cli import apply_named_model, cli, load_config


def test_config_show_uses_default_yaml():
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "stage0_synthetic_memory" in result.output
    assert "model:" in result.output


def test_missing_config_warns_and_uses_defaults():
    result = CliRunner().invoke(cli, ["config", "show", "-c", "missing.yml"])
    assert result.exit_code == 0
    assert "Warning: config file not found: missing.yml" in result.output
    assert "stage0_synthetic_memory" in result.output


def test_dataset_show_reports_stage_stats():
    result = CliRunner().invoke(cli, ["dataset", "show", "-g", "stage0_synthetic_memory", "-m", "2"])
    assert result.exit_code == 0
    assert "documents: 4" in result.output
    assert "loaded_examples: 2" in result.output


def test_load_config_accepts_yaml_overrides(tmp_path):
    path = tmp_path / "override.yml"
    path.write_text("train:\n  steps: 7\nmodel:\n  d_model: 32\n", encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg["train"]["steps"] == 7
    assert cfg["model"]["d_model"] == 32


def test_load_config_applies_named_model_profile(tmp_path):
    path = tmp_path / "models.yml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  d_model: 128",
                "  n_heads: 4",
                "train:",
                "  batch_size: 8",
                "pra:",
                "  top_k_references: 2",
                "resolver:",
                "  type: in_memory",
                "cache:",
                "  type: simple",
                "models:",
                "  small:",
                "    d_model: 32",
                "    train:",
                "      batch_size: 2",
                "    pra:",
                "      top_k_references: 1",
                "    resolver:",
                "      type: in_memory",
                "    cache:",
                "      type: simple",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(str(path), model_name="small")

    assert cfg["selected_model"] == "small"
    assert cfg["model"]["d_model"] == 32
    assert cfg["model"]["n_heads"] == 4
    assert cfg["train"]["batch_size"] == 2
    assert cfg["pra"]["top_k_references"] == 1


def test_apply_named_model_default_falls_back_to_global_when_missing():
    cfg = {
        "model": {"d_model": 128},
        "train": {"batch_size": 8},
        "pra": {"top_k_references": 2},
        "resolver": {"type": "in_memory"},
        "cache": {"type": "simple"},
        "models": {"small": {"d_model": 32}},
    }

    assert apply_named_model(cfg, "default")["model"]["d_model"] == 128


def test_standalone_tiny_profile_is_in_central_config():
    cfg = load_config(model_name="standalone_tiny")

    assert cfg["selected_model"] == "standalone_tiny"
    assert cfg["model"]["d_model"] == 32
    assert cfg["model"]["n_layers"] == 2
    assert cfg["pra"]["pra_layer_ids"] == [0, 1]
    assert cfg["standalone"]["experiment_name"] == "standalone_tiny"
    assert cfg["standalone"]["batch_size"] == 4
    assert cfg["standalone"]["max_examples"] == 4


def test_named_decoder_variants_are_in_central_config():
    sa = load_config(model_name="td_sa_tiny")
    pra = load_config(model_name="td_pra_tiny")
    mixed = load_config(model_name="tdx_pra_tiny")

    assert sa["model"]["model_variant"] == "td_sa"
    assert sa["model"]["n_vanilla_layers"] == 4
    assert pra["model"]["model_variant"] == "td_pra"
    assert pra["pra"]["pra_layer_ids"] == [0, 1, 2, 3]
    assert mixed["model"]["model_variant"] == "tdx_pra"
    assert mixed["model"]["n_vanilla_layers"] == 2


def test_train_cli_accepts_model_option(tmp_path):
    out_path = tmp_path / "cli_model.pt"
    result = CliRunner().invoke(
        cli,
        [
            "train",
            "--model",
            "tiny",
            "-s",
            "0",
            "-m",
            "1",
            "-b",
            "1",
            "-d",
            "cpu",
            "-o",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()


def test_train_cli_steps_caps_optimizer_steps(tmp_path):
    out_path = tmp_path / "cli_steps.pt"
    result = CliRunner().invoke(
        cli,
        [
            "train",
            "--model",
            "tiny",
            "-s",
            "1",
            "-m",
            "2",
            "-b",
            "1",
            "-d",
            "cpu",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "[train step=1]" in result.output
    assert "[train step=2]" not in result.output


def test_train_cli_writes_metric_artifacts(tmp_path):
    out_path = tmp_path / "artifact_run.pt"
    result = CliRunner().invoke(
        cli,
        [
            "train",
            "--model",
            "tiny",
            "-s",
            "1",
            "-m",
            "1",
            "-b",
            "1",
            "-d",
            "cpu",
            "-o",
            str(out_path),
        ],
    )

    run_dir = tmp_path / "artifact_run"
    assert result.exit_code == 0, result.output
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.md").exists()
    assert list((run_dir / "plots").glob("*.png"))


def test_estimate_model_size_uses_resolved_config():
    estimate = estimate_model_size(load_config(model_name="medium"), vocab_size=256)

    assert estimate["selected_model"] == "medium"
    assert estimate["d_model"] == 256
    assert estimate["n_layers"] == 6
    assert estimate["total_params"] > 0
    assert estimate["training_mib"] > estimate["inference_mib"]


def test_size_cli_reports_named_model_memory():
    result = CliRunner().invoke(cli, ["size", "--model", "large", "-v", "256", "-d", "float16"])

    assert result.exit_code == 0, result.output
    assert "selected_model: large" in result.output
    assert "d_model: 512" in result.output
    assert "training_mib:" in result.output
