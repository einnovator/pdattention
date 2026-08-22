from click.testing import CliRunner
from pathlib import Path

from pra_torch.cli import cli


def test_distributed_cli_defaults_and_validation():
    runner = CliRunner()
    result = runner.invoke(cli, ["config", "validate"])
    assert result.exit_code == 0, result.output
    assert "1 workers" in result.output
    assert runner.invoke(cli, ["worker", "show", "local"]).exit_code == 0
    assert runner.invoke(cli, ["cluster", "show", "local"]).exit_code == 0
    assert runner.invoke(cli, ["storage", "show", "local"]).exit_code == 0


def test_cli_directory_merge_and_experiment_dry_run(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "01-storage.yml").write_text(
        f"storage:\n  tmp:\n    type: local\n    path: {tmp_path.as_posix()}/results\n",
        encoding="utf-8",
    )
    (config_dir / "02-experiment.yml").write_text(
        "experiments:\n  smoke:\n    module: common.experiments.smoke\n    function: scalar\n"
        "    storage: tmp\n    parameters: {seed: 3}\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli, ["experiment", "run", "smoke", "-c", str(config_dir), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "run " in result.output


def test_pra_multiseed_dry_run_uses_five_trials(tmp_path):
    config = tmp_path / "run.yml"
    config.write_text(
        f"storage:\n  local:\n    type: local\n    path: {tmp_path.as_posix()}/runs\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli, ["train", "-c", str(config), "--seeds", "0:5", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    run_dir = Path(result.output.strip().split(": ", 1)[1])
    manifest = __import__("json").loads((run_dir / "run.json").read_text())
    assert len(manifest["trials"]) == 5
