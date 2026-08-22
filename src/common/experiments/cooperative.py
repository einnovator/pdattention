"""Cooperative local-rank execution for DDP and FSDP experiment callables."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

from common.distributed.context import (
    DistributedContext,
    barrier,
    destroy_process_group,
    init_process_group,
)
from common.distributed.launcher import launch_local
from common.distributed.models import DistributionMode, ResourceRequirements, WorkerConfig

from .loader import invoke_callable
from .models import ExperimentContext, ExperimentEntrypoint, Trial, TrialState
from .state import atomic_write_json, utc_now, write_status
from .trial import normalize_metrics, repository_state, runtime_state


def _rank_run(rank: int, manifest: dict, run_id: str, run_dir: str, worker_values: list[dict], backend: str):
    trial = Trial(
        experiment_name=manifest["experiment"],
        trial_id=manifest["trial_id"],
        parameters=manifest["parameters"],
        entrypoint=ExperimentEntrypoint.from_mapping(manifest["entrypoint"]),
        distribution=DistributionMode.from_value(manifest["distribution"]),
        cluster_name=manifest["cluster"],
        storage_name=manifest.get("storage"),
        resources=ResourceRequirements.from_mapping(manifest.get("resources")),
        fingerprint=manifest["fingerprint"],
        assigned_workers=tuple(manifest.get("workers") or ()),
        attempt=int(manifest.get("attempt", 1)),
    )
    worker = WorkerConfig.from_mapping(trial.assigned_workers[rank], worker_values[rank])
    distributed = init_process_group(
        DistributedContext.from_environment(
            strategy=trial.distribution.value,
            worker_name=worker.name,
            cluster_name=trial.cluster_name,
        ),
        backend=backend,
        device=worker.device,
    )
    trial_dir = Path(run_dir) / "trials" / trial.trial_id
    rank_dir = trial_dir if rank == 0 else trial_dir / "ranks" / f"rank-{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    context = ExperimentContext(
        experiment_name=trial.experiment_name,
        trial_id=trial.trial_id,
        run_id=run_id,
        worker_name=worker.name,
        cluster_name=trial.cluster_name,
        role=worker.role,
        rank=rank,
        local_rank=rank,
        world_size=distributed.world_size,
        output_dir=trial_dir,
        storage_name=trial.storage_name,
        strategy=trial.distribution.value,
    )
    started = time.perf_counter()
    try:
        with (rank_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (
            rank_dir / "stderr.log"
        ).open("a", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            value = invoke_callable(trial.entrypoint, trial.parameters, context)
        metrics = normalize_metrics(value, time.perf_counter() - started)
        atomic_write_json(rank_dir / "metric.json", metrics)
        barrier(distributed)
    finally:
        destroy_process_group()


def execute_cooperative_local(
    trial: Trial,
    *,
    run_id: str,
    run_dir: Path,
    workers: tuple[WorkerConfig, ...],
    backend: str,
) -> dict:
    """Launch a cooperative rank set on one host and finalize rank-zero artifacts."""

    if trial.distribution == DistributionMode.PIPELINE:
        raise NotImplementedError("Pipeline execution is reserved but not implemented.")
    if any(worker.transport == "ssh" for worker in workers):
        raise NotImplementedError(
            "Multi-host launch wiring is configured but requires the same checkout and rendezvous on each host."
        )
    trial.assigned_workers = tuple(worker.name for worker in workers)
    trial_dir = run_dir / "trials" / trial.trial_id
    (trial_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (trial_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    manifest = {
        **trial.manifest(),
        "run_id": run_id,
        "attempt": trial.attempt,
        "git": repository_state(Path.cwd()),
        "runtime": runtime_state(workers[0]),
    }
    atomic_write_json(trial_dir / "experiment.json", manifest)
    write_status(
        trial_dir / "status.json",
        TrialState.RUNNING.value,
        attempt=trial.attempt,
        started_at=utc_now(),
        workers=list(trial.assigned_workers),
    )
    try:
        launch_local(
            _rank_run,
            world_size=len(workers),
            args=(
                manifest,
                run_id,
                str(run_dir.resolve()),
                [
                    {
                        "host": worker.host,
                        "transport": "local",
                        "device": worker.device,
                        "role": worker.role,
                        "tags": list(worker.tags),
                    }
                    for worker in workers
                ],
                backend,
            ),
        )
        metrics = __import__("json").loads((trial_dir / "metric.json").read_text(encoding="utf-8"))
        write_status(
            trial_dir / "status.json",
            TrialState.SUCCEEDED.value,
            attempt=trial.attempt,
            finished_at=utc_now(),
            exit_code=0,
        )
        return metrics
    except BaseException as exc:
        write_status(
            trial_dir / "status.json",
            TrialState.FAILED.value,
            attempt=trial.attempt,
            finished_at=utc_now(),
            exit_code=1,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
