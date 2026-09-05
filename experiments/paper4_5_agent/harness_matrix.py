"""Crash-resumable no-PRA coding-agent matrix over official Harbor tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import Field, model_validator

from experiments.agents.runner import import_harbor_job, load_runs
from experiments.agents.schema import (
    BenchmarkManifest,
    PRAProfile,
    PRAMode,
    StrictModel,
)


class MatrixModel(StrictModel):
    """One directly served model; credentials remain in environment variables."""

    model_id: str
    served_model: str
    model_revision: str | None = None
    engine: str
    engine_version: str
    quantization: str | None = None
    base_url_env: str
    api_key_env: str = "PRA_AGENT_API_KEY"
    enabled: bool = True


class MatrixHarness(StrictModel):
    """A Harbor agent implementation and its pinned constructor arguments."""

    harness_id: str
    agent: str
    version: str
    kwargs: Mapping[str, str | int | float | bool] = Field(default_factory=dict)
    enabled: bool = True


class HarnessMatrixConfig(StrictModel):
    """Frozen no-PRA matrix used to qualify models and agent harnesses."""

    schema_version: int = 1
    campaign_id: str
    manifest: str
    output_directory: str
    agent_host: str
    hardware: Mapping[str, Any] = Field(default_factory=dict)
    minimum_runs: int = Field(default=15, ge=1)
    minimum_success_rate: float = Field(default=0.10, ge=0, le=1)
    maximum_success_rate: float = Field(default=0.90, ge=0, le=1)
    models: tuple[MatrixModel, ...]
    harnesses: tuple[MatrixHarness, ...]

    @model_validator(mode="after")
    def unique_matrix_ids(self) -> "HarnessMatrixConfig":
        for name, values in (
            ("model_id", [row.model_id for row in self.models]),
            ("harness_id", [row.harness_id for row in self.harnesses]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} values must be unique")
        if not any(row.enabled for row in self.models):
            raise ValueError("at least one model must be enabled")
        if not any(row.enabled for row in self.harnesses):
            raise ValueError("at least one harness must be enabled")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "HarnessMatrixConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def matrix_cells(
    config: HarnessMatrixConfig, manifest: BenchmarkManifest,
) -> list[tuple[str, MatrixModel, MatrixHarness, str, int]]:
    """Enumerate the frozen matrix in task-major order for early cross-harness data."""

    cells = []
    for model in config.models:
        if not model.enabled:
            continue
        for repeat in range(manifest.repeats):
            for task_id in manifest.task_ids:
                for harness in config.harnesses:
                    if not harness.enabled:
                        continue
                    cell_id = f"{model.model_id}__{task_id}__{harness.harness_id}__r{repeat}"
                    cells.append((cell_id, model, harness, task_id, repeat))
    return cells


def harbor_command(
    *, harbor: str, manifest: BenchmarkManifest, model: MatrixModel,
    harness: MatrixHarness, task_id: str, job_directory: Path,
    base_url: str, api_key: str,
) -> list[str]:
    """Build one isolated official-Harbor trial command."""

    qualified_task = task_id if task_id.startswith("terminal-bench/") else f"terminal-bench/{task_id}"
    command = [
        harbor, "run", "-d", manifest.dataset, "-a", harness.agent,
        "-m", f"openai/{model.served_model}",
    ]
    kwargs = {"version": harness.version, **harness.kwargs}
    for name, value in kwargs.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        command.extend(("--ak", f"{name}={rendered}"))
    command.extend((
        "-i", qualified_task,
        "--agent-env", f"OPENAI_API_KEY={api_key}",
        "--agent-env", f"OPENAI_BASE_URL={base_url.rstrip('/')}",
        "--allow-agent-host", _endpoint_host(base_url),
        "--jobs-dir", str(job_directory), "-n", "1", "-y",
    ))
    return command


def run_matrix(
    config_path: Path, *, resume: bool, dry_run: bool = False,
    max_cells: int | None = None, harbor: str = "harbor",
) -> dict[str, Any]:
    """Run and checkpoint a frozen matrix, preserving failed cells for retry."""

    config = HarnessMatrixConfig.load(config_path)
    repository = _repository_root(config_path.resolve())
    manifest = BenchmarkManifest.load(repository / config.manifest)
    if manifest.benchmark != "terminal-bench":
        raise ValueError("the Harbor matrix requires a terminal-bench manifest")
    output = (repository / config.output_directory).resolve()
    state_path = output / "matrix_state.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": config.campaign_id,
        "pra_enabled": False,
        "manifest": config.manifest,
        "cells": {},
    }
    if resume and state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    launched = 0

    for cell_id, model, harness_spec, task_id, repeat in matrix_cells(config, manifest):
        record = state.setdefault("cells", {}).setdefault(cell_id, {"state": "PENDING"})
        if record.get("state") == "COMPLETED":
            continue
        if max_cells is not None and launched >= max_cells:
            break
        base_url = os.environ.get(model.base_url_env)
        api_key = os.environ.get(model.api_key_env, "pra-local")
        if not base_url and not dry_run:
            record.update(state="BLOCKED", reason=f"missing environment variable {model.base_url_env}")
            _persist(config, manifest, state, output, state_path)
            continue
        cell_dir = output / "cells" / cell_id
        attempt = int(record.get("attempts", 0)) + 1
        jobs_dir = cell_dir / "attempts" / f"a{attempt:03d}"
        normalized_dir = cell_dir / "normalized"
        command = harbor_command(
            harbor=harbor, manifest=manifest, model=model, harness=harness_spec,
            task_id=task_id, job_directory=jobs_dir,
            base_url=base_url or "http://HOST_REQUIRED/v1", api_key=api_key,
        )
        record.update(
            state="PENDING" if dry_run else "RUNNING",
            model=model.model_id, task_id=task_id, harness=harness_spec.harness_id,
            repeat=repeat, pra_enabled=False, attempts=attempt,
            active_attempt=f"a{attempt:03d}", command=_redact(command),
        )
        _persist(config, manifest, state, output, state_path)
        if dry_run:
            launched += 1
            continue

        launched += 1
        record["started_at"] = _now()
        cell_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(repository), environment.get("PYTHONPATH")) if value
        )
        completed = subprocess.run(
            command, cwd=repository, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        (cell_dir / "launcher.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            record.update(
                state="FAILED", finished_at=_now(), returncode=completed.returncode,
                reason="Harbor trial failed; inspect launcher.log and resume to retry.",
            )
            _persist(config, manifest, state, output, state_path)
            continue
        try:
            rows = import_harbor_job(
                jobs_dir, manifest, output=normalized_dir,
                engine=model.engine, engine_version=model.engine_version,
                host=config.agent_host, hardware=config.hardware,
                model=model.served_model, model_revision=model.model_revision,
                quantization=model.quantization, pra_mode=PRAMode.NONE,
                pra_profile=PRAProfile.NONE, connection="direct",
                protocol="openai-chat-completions",
            )
            if len(rows) != 1 or rows[0].identity.task_id != task_id:
                raise ValueError(f"expected one normalized row for {task_id}, found {len(rows)}")
            record.update(
                state="COMPLETED", finished_at=_now(),
                success=rows[0].outcome.success,
                official_score=rows[0].outcome.official_score,
                normalized_result=str((normalized_dir / "runs.jsonl").relative_to(output)),
            )
        except (OSError, ValueError) as exc:
            record.update(state="FAILED", finished_at=_now(), reason=str(exc))
        _persist(config, manifest, state, output, state_path)
    return state


def _persist(
    config: HarnessMatrixConfig, manifest: BenchmarkManifest,
    state: dict[str, Any], output: Path, state_path: Path,
) -> None:
    """Atomically persist state and rebuild aggregate rows from completed cells."""

    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(state_path)
    rows = []
    for record in state.get("cells", {}).values():
        relative = record.get("normalized_result")
        if record.get("state") == "COMPLETED" and relative:
            rows.extend(load_runs(output / relative))
    aggregate = output / "runs.jsonl"
    aggregate.write_text("".join(row.json_line() + "\n" for row in rows), encoding="utf-8")
    summary = _summarize(rows)
    gate = _admission_gate(config, rows)
    (output / "summary.json").write_text(
        json.dumps({
            "campaign_id": config.campaign_id,
            "pra_enabled": False,
            "expected_runs": len(matrix_cells(config, manifest)),
            "completed_runs": len(rows),
            "summary": summary,
            "admission_gate": gate,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def _summarize(rows: list[Any]) -> dict[str, Any]:
    """Summarize official rows without importing the PRA runtime package."""

    by_harness: dict[str, dict[str, int | float]] = {}
    for row in rows:
        bucket = by_harness.setdefault(
            row.identity.agent,
            {"runs": 0, "successes": 0, "input_tokens": 0, "output_tokens": 0,
             "model_calls": 0, "tool_calls": 0, "wall_ms": 0.0},
        )
        bucket["runs"] += 1
        bucket["successes"] += int(row.outcome.success)
        bucket["input_tokens"] += row.tokens.input_tokens
        bucket["output_tokens"] += row.tokens.output_tokens
        bucket["model_calls"] += row.behavior.model_calls
        bucket["tool_calls"] += row.behavior.tool_calls
        bucket["wall_ms"] += row.timings.task_wall_ms
    for bucket in by_harness.values():
        runs = int(bucket["runs"])
        bucket["success_rate"] = bucket["successes"] / runs if runs else 0.0
    successes = sum(row.outcome.success for row in rows)
    return {
        "runs": len(rows), "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "input_tokens": sum(row.tokens.input_tokens for row in rows),
        "output_tokens": sum(row.tokens.output_tokens for row in rows),
        "by_harness": by_harness,
    }


def _admission_gate(config: HarnessMatrixConfig, rows: list[Any]) -> dict[str, Any]:
    """Apply the preregistered floor, ceiling, and full-matrix requirements."""

    successes = sum(row.outcome.success for row in rows)
    rate = successes / len(rows) if rows else None
    if len(rows) < config.minimum_runs:
        status = "BLOCKED"
        reason = f"Only {len(rows)} completed runs; all {config.minimum_runs} are required."
    elif successes == 0:
        status = "BLOCKED"
        reason = "No-PRA official success is zero; PRA efficacy comparisons are floor-confounded."
    elif rate is not None and rate < config.minimum_success_rate:
        status = "BLOCKED"
        reason = f"No-PRA success {rate:.1%} is below the promotion floor."
    elif rate is not None and rate > config.maximum_success_rate:
        status = "BLOCKED"
        reason = f"No-PRA success {rate:.1%} exceeds the promotion ceiling."
    else:
        status = "ELIGIBLE"
        reason = "The complete no-PRA matrix is inside the preregistered comparison band."
    return {
        "status": status, "eligible": status == "ELIGIBLE", "runs": len(rows),
        "successes": successes, "official_success_rate": rate,
        "target_range": [config.minimum_success_rate, config.maximum_success_rate],
        "reason": reason,
    }


def _endpoint_host(url: str) -> str:
    from urllib.parse import urlparse

    host = urlparse(url).hostname
    if not host:
        raise ValueError(f"base URL has no host: {url!r}")
    return host


def _redact(command: list[str]) -> list[str]:
    return [
        "OPENAI_API_KEY=***" if value.startswith("OPENAI_API_KEY=") else value
        for value in command
    ]


def _repository_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "experiments").is_dir():
            return parent
    return Path.cwd().resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--harbor", default="harbor")
    args = parser.parse_args()
    state = run_matrix(
        args.config, resume=args.resume, dry_run=args.dry_run,
        max_cells=args.max_cells, harbor=args.harbor,
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
