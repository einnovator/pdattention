"""Click wiring from the PRA CLI to model-independent experiment infrastructure."""

from __future__ import annotations

import copy
import json
from enum import Enum
from dataclasses import asdict, replace
from pathlib import Path

import click
import yaml

from common.config import apply_overrides, parse_override, resolve_infrastructure
from common.distributed.worker import ping_worker
from common.experiments.aggregate import aggregate_metrics
from common.experiments.models import ExperimentDefinition, ExperimentEntrypoint
from common.experiments.runner import run_experiment
from common.experiments.sweep import parse_seed_spec
from common.storage.transfer import get_tree


CONFIG_OPTION = click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(file_okay=True, dir_okay=True),
    multiple=True,
)


def _resolved(load_config, paths):
    cfg = load_config(paths)
    return cfg, resolve_infrastructure(cfg, sources=cfg.get("_config_sources", ()))


def _echo(value) -> None:
    def default(item):
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    plain = json.loads(json.dumps(value, default=default))
    click.echo(yaml.safe_dump(plain, sort_keys=False))


def _find_run(infrastructure, run_id: str) -> Path:
    candidates = []
    for storage in infrastructure.storage.values():
        if storage.type == "local":
            candidates.extend(Path(storage.path).expanduser().glob(f"*/{run_id}"))
    if not candidates:
        raise click.ClickException(f"Run {run_id!r} was not found in configured local storage.")
    if len(candidates) > 1:
        raise click.ClickException(f"Run {run_id!r} is ambiguous across local storage roots.")
    return candidates[0]


def run_training_request(
    cfg: dict,
    *,
    cluster: str | None,
    experiment: str | None,
    distribution: str | None,
    storage: str | None,
    seeds: str | None,
    num_seeds: int | None,
    base_seed: int,
    resume: bool,
    dry_run: bool,
    max_trials: int | None,
    fail_fast: bool,
):
    """Turn PRA train selectors into the same generic experiment definition."""

    infrastructure = resolve_infrastructure(cfg, sources=cfg.get("_config_sources", ()))
    if experiment:
        if experiment not in infrastructure.experiments:
            raise click.ClickException(f"Unknown experiment {experiment!r}.")
        definition = infrastructure.experiments[experiment]
    else:
        seed_values = parse_seed_spec(seeds) if seeds else None
        if num_seeds is not None:
            if num_seeds <= 0:
                raise click.ClickException("--num-seeds must be positive.")
            seed_values = list(range(base_seed, base_seed + num_seeds))
        parameters = {"_resolved_config": copy.deepcopy(cfg)}
        sweep = {"seed": tuple(seed_values)} if seed_values else {}
        name = Path(cfg["train"]["out"]).stem
        definition = ExperimentDefinition(
            name=name,
            entrypoint=ExperimentEntrypoint(
                module="pra_torch.experiment_adapter",
                function="run_pra_training_experiment",
            ),
            parameters=parameters,
            sweep=sweep,
        )
        if seed_values and distribution is None:
            distribution = "seeds"
    result = run_experiment(
        definition,
        infrastructure,
        cluster_name=cluster,
        distribution=distribution,
        storage_name=storage,
        resume=resume,
        dry_run=dry_run,
        max_trials=max_trials,
        fail_fast=fail_fast,
    )
    click.echo(f"run {result.run_id}: {result.run_dir}")
    if result.failures:
        raise click.ClickException(f"{result.failures} trial(s) failed; inspect per-trial stderr.log.")
    return result


def register_distributed_commands(cli, load_config) -> None:
    """Attach reusable infrastructure groups to the established ``pra`` CLI."""

    @cli.group(name="worker")
    def worker_group():
        """Inspect and probe experiment workers."""

    @worker_group.command(name="list")
    @CONFIG_OPTION
    def worker_list(config_path):
        _, infrastructure = _resolved(load_config, config_path)
        _echo({name: asdict(value) for name, value in infrastructure.workers.items()})

    @worker_group.command(name="show")
    @click.argument("name")
    @CONFIG_OPTION
    def worker_show(name, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        if name not in infrastructure.workers:
            raise click.ClickException(f"Unknown worker {name!r}.")
        _echo(asdict(infrastructure.workers[name]))

    @worker_group.command(name="ping")
    @click.argument("name")
    @CONFIG_OPTION
    def worker_ping(name, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        if name not in infrastructure.workers:
            raise click.ClickException(f"Unknown worker {name!r}.")
        result = ping_worker(infrastructure.workers[name])
        _echo(result)
        if not result["ok"]:
            raise click.ClickException(result["error"])

    @cli.group(name="cluster")
    def cluster_group():
        """Inspect named worker clusters."""

    @cluster_group.command(name="list")
    @CONFIG_OPTION
    def cluster_list(config_path):
        _, infrastructure = _resolved(load_config, config_path)
        _echo({name: asdict(value) for name, value in infrastructure.clusters.items()})

    @cluster_group.command(name="show")
    @click.argument("name")
    @CONFIG_OPTION
    def cluster_show(name, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        if name not in infrastructure.clusters:
            raise click.ClickException(f"Unknown cluster {name!r}.")
        _echo(asdict(infrastructure.clusters[name]))

    @cli.group(name="storage")
    def storage_group():
        """Inspect artifact storage backends."""

    @storage_group.command(name="list")
    @CONFIG_OPTION
    def storage_list(config_path):
        _, infrastructure = _resolved(load_config, config_path)
        _echo({name: value.safe_manifest() for name, value in infrastructure.storage.items()})

    @storage_group.command(name="show")
    @click.argument("name")
    @CONFIG_OPTION
    def storage_show(name, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        if name not in infrastructure.storage:
            raise click.ClickException(f"Unknown storage {name!r}.")
        _echo(infrastructure.storage[name].safe_manifest())

    @cli.group(name="experiment")
    def experiment_group():
        """Run, resume, and aggregate independent experiments."""

    @experiment_group.command(name="list")
    @CONFIG_OPTION
    def experiment_list(config_path):
        _, infrastructure = _resolved(load_config, config_path)
        _echo({name: asdict(value) for name, value in infrastructure.experiments.items()})

    @experiment_group.command(name="show")
    @click.argument("name")
    @CONFIG_OPTION
    def experiment_show(name, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        if name not in infrastructure.experiments:
            raise click.ClickException(f"Unknown experiment {name!r}.")
        _echo(asdict(infrastructure.experiments[name]))

    @experiment_group.command(name="run")
    @click.argument("name")
    @CONFIG_OPTION
    @click.option("-C", "--cluster")
    @click.option("--distribution", type=click.Choice(["local", "trials", "seeds", "sweep", "ddp", "fsdp", "pipeline"]))
    @click.option("--storage", "storage_name")
    @click.option("--param", "parameters", multiple=True, help="Trial override PATH=YAML_VALUE.")
    @click.option("--resume", is_flag=True)
    @click.option("--dry-run", is_flag=True)
    @click.option("--max-trials", type=click.IntRange(min=1))
    @click.option("--fail-fast", is_flag=True)
    def experiment_run(
        name, config_path, cluster, distribution, storage_name, parameters, resume, dry_run, max_trials, fail_fast
    ):
        _, infrastructure = _resolved(load_config, config_path)
        if name not in infrastructure.experiments:
            raise click.ClickException(f"Unknown experiment {name!r}.")
        overrides = dict(parse_override(item) for item in parameters)
        result = run_experiment(
            infrastructure.experiments[name],
            infrastructure,
            cluster_name=cluster,
            distribution=distribution,
            storage_name=storage_name,
            overrides=overrides,
            resume=resume,
            dry_run=dry_run,
            max_trials=max_trials,
            fail_fast=fail_fast,
        )
        click.echo(f"run {result.run_id}: {result.run_dir}")
        if result.failures:
            raise click.ClickException(f"{result.failures} trial(s) failed.")

    @experiment_group.command(name="status")
    @click.argument("run_id")
    @CONFIG_OPTION
    def experiment_status(run_id, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        run_dir = _find_run(infrastructure, run_id)
        _echo(json.loads((run_dir / "run.json").read_text(encoding="utf-8")))

    @experiment_group.command(name="aggregate")
    @click.argument("run_id")
    @CONFIG_OPTION
    def experiment_aggregate(run_id, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        _echo(aggregate_metrics(_find_run(infrastructure, run_id)))

    @experiment_group.command(name="pull")
    @click.argument("run_id")
    @click.option("--storage", "storage_name", required=True)
    @click.option("-o", "--output", type=click.Path(file_okay=False), default="out/pulled")
    @CONFIG_OPTION
    def experiment_pull(run_id, storage_name, output, config_path):
        _, infrastructure = _resolved(load_config, config_path)
        backend = infrastructure.storage_registry().get(storage_name)
        destination = Path(output) / run_id
        run_manifests = [
            key for key in backend.list("") if key.endswith(f"/{run_id}/run.json")
        ]
        if len(run_manifests) != 1:
            raise click.ClickException(
                f"Expected one run {run_id!r} in {storage_name!r}, found {len(run_manifests)}."
            )
        prefix = run_manifests[0][: -len("/run.json")]
        get_tree(backend, prefix, destination)
        click.echo(str(destination))
