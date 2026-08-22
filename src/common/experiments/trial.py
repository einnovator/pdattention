"""Execution of one resolved experiment trial."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import posixpath
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from common.distributed.models import WorkerConfig
from common.distributed.transport import transport_for

from .loader import invoke_callable, run_script
from .models import ExperimentContext, Trial, TrialState
from .state import atomic_write_json, read_json, utc_now, write_status


_IN_PROCESS_EXECUTION_LOCK = threading.Lock()


def repository_state(workdir: Path) -> dict:
    """Capture a lightweight reproducibility fingerprint without failing outside Git."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workdir, text=True, capture_output=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workdir,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def runtime_state(worker: WorkerConfig) -> dict:
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": worker.device,
    }


def normalize_metrics(value: Any, elapsed: float) -> dict:
    if value is None:
        metrics: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        metrics = dict(value)
    elif isinstance(value, (int, float)):
        metrics = {"value": float(value)}
    else:
        raise TypeError("Experiment callable must return a mapping, number, or None.")
    metrics.setdefault("elapsed_seconds", elapsed)
    return metrics


def execute_trial_local(
    trial: Trial,
    *,
    run_id: str,
    run_dir: Path,
    worker: WorkerConfig,
    resumed: bool,
) -> dict:
    """Execute a callable in-process and persist the complete trial contract."""

    trial_dir = run_dir / "trials" / trial.trial_id
    (trial_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (trial_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    context = ExperimentContext(
        experiment_name=trial.experiment_name,
        trial_id=trial.trial_id,
        run_id=run_id,
        worker_name=worker.name,
        cluster_name=trial.cluster_name,
        role=worker.role,
        rank=0,
        local_rank=0,
        world_size=1,
        output_dir=trial_dir,
        storage_name=trial.storage_name,
        resumed=resumed,
    )
    manifest = {
        **trial.manifest(),
        "run_id": run_id,
        "git": repository_state(Path.cwd()),
        "runtime": runtime_state(worker),
    }
    atomic_write_json(trial_dir / "experiment.json", manifest)
    started = utc_now()
    write_status(
        trial_dir / "status.json",
        TrialState.RUNNING.value,
        attempt=trial.attempt,
        started_at=started,
        worker=worker.name,
    )
    stdout_path = trial_dir / "stdout.log"
    stderr_path = trial_dir / "stderr.log"
    before = time.perf_counter()
    try:
        # stdout redirection and temporary script environment mutate process globals.
        with _IN_PROCESS_EXECUTION_LOCK, stdout_path.open(
            "a", encoding="utf-8"
        ) as stdout, stderr_path.open("a", encoding="utf-8") as stderr, contextlib.redirect_stdout(
            stdout
        ), contextlib.redirect_stderr(stderr):
            if trial.entrypoint.script_only:
                previous = os.environ.copy()
                os.environ.update(
                    PRA_EXPERIMENT_PARAMS=json.dumps(trial.parameters),
                    PRA_EXPERIMENT_CONTEXT=json.dumps(context.as_dict()),
                )
                try:
                    run_script(trial.entrypoint)
                    value = read_json(trial_dir / "metric.json", {})
                finally:
                    os.environ.clear()
                    os.environ.update(previous)
            else:
                value = invoke_callable(trial.entrypoint, trial.parameters, context)
        metrics = normalize_metrics(value, time.perf_counter() - before)
        atomic_write_json(trial_dir / "metric.json", metrics)
        write_status(
            trial_dir / "status.json",
            TrialState.SUCCEEDED.value,
            attempt=trial.attempt,
            started_at=started,
            finished_at=utc_now(),
            exit_code=0,
        )
        return metrics
    except BaseException as exc:
        with stderr_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{type(exc).__name__}: {exc}\n")
        write_status(
            trial_dir / "status.json",
            TrialState.FAILED.value,
            attempt=trial.attempt,
            started_at=started,
            finished_at=utc_now(),
            exit_code=1,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def execute_trial_process(
    trial: Trial,
    *,
    run_id: str,
    run_dir: Path,
    worker: WorkerConfig,
    resumed: bool,
    timeout: int | None,
) -> dict:
    """Execute a trial in an isolated local or SSH Python interpreter."""

    trial_dir = run_dir / "trials" / trial.trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    remote = worker.transport == "ssh"
    remote_root = worker.staging_dir
    remote_trial_dir = posixpath.join(remote_root, run_id, trial.trial_id)
    execution_run_dir = str(run_dir.resolve())
    manifest = trial.manifest()
    if remote:
        execution_run_dir = posixpath.join(remote_root, run_id)
        if trial.entrypoint.file:
            remote_entrypoint = posixpath.join(remote_trial_dir, Path(trial.entrypoint.file).name)
            manifest["entrypoint"]["file"] = remote_entrypoint
    invocation = {
        "trial": manifest,
        "run_id": run_id,
        "run_dir": execution_run_dir,
        "worker": worker.name,
        "worker_config": {
            "host": worker.host,
            "transport": worker.transport,
            "device": worker.device,
            "role": worker.role,
            "tags": list(worker.tags),
        },
        "resumed": resumed,
        "attempt": trial.attempt,
    }
    invocation_path = trial_dir / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    executable = worker.python_executable if worker.transport == "ssh" else sys.executable
    invocation_argument = str(invocation_path.resolve())
    transport = transport_for(worker)
    if remote:
        from common.distributed.ssh import SSHTransport

        assert isinstance(transport, SSHTransport)
        created = transport.run(["mkdir", "-p", remote_trial_dir], cwd=worker.workdir)
        if created.returncode:
            raise RuntimeError(f"Could not create remote trial directory: {created.stderr.strip()}")
        invocation_argument = posixpath.join(remote_trial_dir, "invocation.json")
        transport.put(invocation_path, invocation_argument)
        if trial.entrypoint.file:
            transport.put(trial.entrypoint.file, manifest["entrypoint"]["file"])
    result = transport.run(
        [executable, "-m", "common.experiments.process_entry", invocation_argument],
        cwd=worker.workdir or Path.cwd(),
        timeout=timeout,
    )
    with (trial_dir / "stdout.log").open("a", encoding="utf-8") as stream:
        stream.write(result.stdout)
    with (trial_dir / "stderr.log").open("a", encoding="utf-8") as stream:
        stream.write(result.stderr)
    if remote:
        transport.get_tree(remote_trial_dir, trial_dir)
    if result.returncode:
        raise RuntimeError(f"Worker {worker.name!r} exited with code {result.returncode}.")
    metrics = read_json(trial_dir / "metric.json")
    if metrics is None:
        raise RuntimeError(f"Worker {worker.name!r} produced no metric.json.")
    return metrics
