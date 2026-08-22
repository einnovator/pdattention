import json
from pathlib import Path

from common.config import resolve_infrastructure
from common.experiments.models import ExperimentDefinition
from common.experiments.runner import run_experiment
from common.experiments.sweep import expand_trials, parse_seed_spec


def _config(tmp_path, transport="local"):
    return {
        "workers": {"runner": {"transport": transport, "max_jobs": 2}},
        "clusters": {"test": {"workers": ["runner"], "default": True}},
        "storage": {"results": {"type": "local", "path": str(tmp_path)}},
        "experiments": {
            "five": {
                "module": "common.experiments.smoke",
                "function": "scalar",
                "storage": "results",
                "distribution": "seeds",
                "parameters": {"offset": 0.5},
                "sweep": {"seed": [0, 1, 2, 3, 4]},
                "retry": {"max_attempts": 1, "backoff_seconds": 0},
            }
        },
    }


def test_seed_parser_and_trial_ids_are_deterministic(tmp_path):
    assert parse_seed_spec("0:3,7") == [0, 1, 2, 7]
    infra = resolve_infrastructure(_config(tmp_path))
    definition = infra.experiments["five"]
    first = expand_trials(definition, cluster_name="test", distribution=definition.distribution, storage_name="results")
    second = expand_trials(definition, cluster_name="test", distribution=definition.distribution, storage_name="results")
    assert [trial.trial_id for trial in first] == [trial.trial_id for trial in second]
    assert len({trial.fingerprint for trial in first}) == 5


def test_local_five_seed_run_aggregates_and_resumes(tmp_path):
    infra = resolve_infrastructure(_config(tmp_path))
    first = run_experiment(infra.experiments["five"], infra)
    assert first.failures == 0
    assert first.aggregate["successful_trials"] == 5
    group = next(iter(first.aggregate["groups"].values()))
    assert group["score"]["mean"] == 2.5
    assert len(list((first.run_dir / "trials").glob("*/metric.json"))) == 5

    second = run_experiment(infra.experiments["five"], infra, resume=first.run_id)
    assert second.run_id == first.run_id
    assert second.skipped == 5
    assert second.aggregate["successful_trials"] == 5


def test_process_experiment_runs_in_isolated_interpreter(tmp_path):
    infra = resolve_infrastructure(_config(tmp_path, transport="process"))
    definition = infra.experiments["five"]
    result = run_experiment(definition, infra, max_trials=1)
    trial_dir = next((result.run_dir / "trials").iterdir())
    assert json.loads((trial_dir / "status.json").read_text())["state"] == "SUCCEEDED"
    assert json.loads((trial_dir / "metric.json").read_text())["score"] == 0.5
    assert (trial_dir / "artifacts" / "identity.json").exists()


def test_file_function_entrypoint_runs_without_package_install(tmp_path):
    entrypoint = tmp_path / "generated.py"
    entrypoint.write_text(
        "def run(params, context):\n    return {'answer': params['value'] * 2}\n",
        encoding="utf-8",
    )
    config = _config(tmp_path / "results")
    config["experiments"] = {
        "generated": {
            "file": str(entrypoint),
            "function": "run",
            "storage": "results",
            "parameters": {"value": 21},
        }
    }
    infra = resolve_infrastructure(config)
    result = run_experiment(infra.experiments["generated"], infra)
    metric = json.loads(next((result.run_dir / "trials").glob("*/metric.json")).read_text())
    assert metric["answer"] == 42


def test_cooperative_local_runner_launches_two_ranks(tmp_path):
    config = _config(tmp_path)
    config["workers"] = {
        "rank0": {"transport": "local", "device": "cpu"},
        "rank1": {"transport": "local", "device": "cpu"},
    }
    config["clusters"] = {
        "test": {
            "workers": ["rank0", "rank1"],
            "default": True,
            "distribution": "ddp",
            "backend": "gloo",
        }
    }
    config["experiments"]["five"]["distribution"] = "ddp"
    config["experiments"]["five"]["resources"] = {"workers": 2, "device": "cpu"}
    infra = resolve_infrastructure(config)
    result = run_experiment(infra.experiments["five"], infra, max_trials=1)
    trial_dir = next((result.run_dir / "trials").iterdir())
    assert json.loads((trial_dir / "status.json").read_text())["state"] == "SUCCEEDED"
    assert (trial_dir / "ranks" / "rank-1" / "metric.json").exists()
