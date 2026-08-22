"""Coordinator-owned experiment expansion, scheduling, resume, and aggregation."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from common.distributed.models import DistributionMode, WorkerConfig
from common.distributed.scheduler import schedule

from .aggregate import aggregate_metrics
from .models import ExperimentDefinition, Trial, TrialState
from .state import atomic_write_json, read_json, utc_now, write_status
from .sweep import expand_trials
from .trial import execute_trial_local, execute_trial_process

if TYPE_CHECKING:
    from common.config import InfrastructureConfig


@dataclass(frozen=True)
class ExperimentRunResult:
    run_id: str
    run_dir: Path
    aggregate: Mapping[str, Any]
    failures: int
    skipped: int


def make_run_id(name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{os.getpid()}"


def _find_resume_dir(root: Path, experiment: str, resume: str | bool) -> Path | None:
    parent = root / experiment
    if isinstance(resume, str) and resume:
        candidate = Path(resume)
        if candidate.is_dir():
            return candidate
        candidate = parent / resume
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Cannot find run {resume!r}.")
    if resume:
        candidates = sorted((item for item in parent.glob("*") if item.is_dir()), reverse=True)
        return candidates[0] if candidates else None
    return None


def run_experiment(
    definition: ExperimentDefinition,
    infrastructure: "InfrastructureConfig",
    *,
    cluster_name: str | None = None,
    distribution: str | DistributionMode | None = None,
    storage_name: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    max_trials: int | None = None,
    resume: str | bool = False,
    dry_run: bool = False,
    fail_fast: bool = False,
) -> ExperimentRunResult:
    """Resolve and execute one experiment definition under a single coordinator."""

    cluster = infrastructure.cluster(cluster_name or definition.cluster)
    mode = DistributionMode.from_value(distribution or definition.distribution or cluster.distribution)
    first_worker = cluster.resolved_workers(infrastructure.workers)[0]
    from common.config import resolve_storage_name

    selected_storage = resolve_storage_name(
        infrastructure,
        cli=storage_name,
        experiment=definition,
        cluster=cluster,
        worker=first_worker,
    )
    storage_config = infrastructure.storage[selected_storage]
    if storage_config.type == "local":
        root = Path(storage_config.path).expanduser().resolve()
    else:
        root = Path("out/experiments/.staging").resolve()
    trials = expand_trials(
        definition,
        cluster_name=cluster.name,
        distribution=mode,
        storage_name=selected_storage,
        cli_overrides=overrides,
        max_trials=max_trials,
    )
    if mode.cooperative and len(cluster.workers) > 1:
        for trial in trials:
            if trial.resources.workers == 1:
                trial.resources = replace(trial.resources, workers=len(cluster.workers))
    resumed_dir = _find_resume_dir(root, definition.name, resume)
    run_id = resumed_dir.name if resumed_dir else make_run_id(definition.name)
    run_dir = resumed_dir or root / definition.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_run = read_json(run_dir / "run.json", {}) or {}
    previous_fingerprints = {
        item["trial_id"]: item["fingerprint"] for item in previous_run.get("trials", ())
    }
    for trial in trials:
        old = previous_fingerprints.get(trial.trial_id)
        if old is not None and old != trial.fingerprint:
            raise ValueError(
                f"Cannot resume trial {trial.trial_id!r}: resolved fingerprint changed."
            )
    run_manifest = {
        "experiment": definition.name,
        "run_id": run_id,
        "started_at": previous_run.get("started_at", utc_now()),
        "updated_at": utc_now(),
        "coordinator": cluster.coordinator or socket.gethostname(),
        "cluster": cluster.name,
        "distribution": mode.value,
        "storage": selected_storage,
        "config_sources": list(infrastructure.sources),
        "trials": [trial.manifest() for trial in trials],
        "dry_run": dry_run,
    }
    atomic_write_json(run_dir / "run.json", run_manifest)
    if dry_run:
        return ExperimentRunResult(run_id, run_dir, {}, 0, 0)

    skipped = 0
    runnable: list[Trial] = []
    for trial in trials:
        status_path = run_dir / "trials" / trial.trial_id / "status.json"
        status = read_json(status_path, {}) or {}
        if resume and status.get("state") == TrialState.SUCCEEDED.value:
            skipped += 1
            continue
        if status.get("state") in {TrialState.RUNNING.value, TrialState.STARTING.value}:
            write_status(status_path, TrialState.INTERRUPTED.value, reason="stale coordinator lease")
        trial.state = TrialState.QUEUED
        runnable.append(trial)

    def execute(trial: Trial, assigned: tuple[WorkerConfig, ...]) -> dict:
        trial.assigned_workers = tuple(worker.name for worker in assigned)
        last_error: BaseException | None = None
        for attempt in range(1, definition.retry.max_attempts + 1):
            trial.attempt = attempt
            worker = assigned[0]
            try:
                if mode.cooperative:
                    from .cooperative import execute_cooperative_local

                    return execute_cooperative_local(
                        trial,
                        run_id=run_id,
                        run_dir=run_dir,
                        workers=assigned,
                        backend=cluster.backend,
                    )
                if worker.transport == "local":
                    return execute_trial_local(
                        trial,
                        run_id=run_id,
                        run_dir=run_dir,
                        worker=worker,
                        resumed=bool(resume),
                    )
                return execute_trial_process(
                    trial,
                    run_id=run_id,
                    run_dir=run_dir,
                    worker=worker,
                    resumed=bool(resume),
                    timeout=definition.timeout_seconds,
                )
            except BaseException as exc:
                last_error = exc
                write_status(
                    run_dir / "trials" / trial.trial_id / "status.json",
                    TrialState.FAILED.value,
                    attempt=attempt,
                    finished_at=utc_now(),
                    exit_code=1,
                    error=f"{type(exc).__name__}: {exc}",
                )
                if attempt < definition.retry.max_attempts:
                    time.sleep(definition.retry.backoff_seconds)
        assert last_error is not None
        raise last_error

    results = schedule(
        runnable,
        cluster=cluster,
        workers=dict(infrastructure.workers),
        resources_for=lambda trial: trial.resources,
        execute=execute,
        fail_fast=fail_fast,
    )
    failures = sum(result.error is not None for result in results)
    aggregate = aggregate_metrics(run_dir)
    run_manifest.update(
        finished_at=utc_now(),
        failures=failures,
        skipped=skipped,
        aggregate=aggregate,
        trials=[trial.manifest() for trial in trials],
    )
    atomic_write_json(run_dir / "run.json", run_manifest)

    if storage_config.type != "local":
        backend = infrastructure.storage_registry().get(selected_storage)
        from common.storage.transfer import put_tree

        put_tree(backend, run_dir, f"{definition.name}/{run_id}")
    return ExperimentRunResult(run_id, run_dir, aggregate, failures, skipped)
