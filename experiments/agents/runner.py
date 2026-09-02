"""Isolated execution and normalized artifact writing for coding-agent tasks."""

from __future__ import annotations

import codecs
import gzip
import json
import platform
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from .adapters import AgentTask, CodingAgentAdapter
from .benchmarks import benchmark_adapter
from .schema import (
    AgentBehaviorMetrics,
    BenchmarkManifest,
    CacheState,
    CodingAgentRun,
    CostMetrics,
    OutcomeMetrics,
    PRAProfile,
    PRAMode,
    ResourceMetrics,
    RunIdentity,
    TimingMetrics,
    TokenMetrics,
)


FIXTURE_TASKS: dict[str, dict[str, str]] = {
    "write-marker": {
        "instruction": "Create marker.txt containing exactly one line with the text PRA and a trailing newline.",
        "file": "marker.txt", "content": "PRA\n",
    },
    "nested-marker": {
        "instruction": "Create out/result.txt containing exactly one line with the text ok and a trailing newline.",
        "file": "out/result.txt", "content": "ok\n",
    },
}


def run_manifest(
    manifest: BenchmarkManifest,
    adapter: CodingAgentAdapter,
    *,
    output: Path,
    agent: str,
    engine: str,
    model: str,
    pra_mode: PRAMode,
    pra_profile: PRAProfile,
    connection: str = "fixture",
    protocol: str = "fixture",
    cache_state: CacheState = CacheState.COLD,
    pra_version: str = "0.2.0rc1",
) -> list[CodingAgentRun]:
    """Run a frozen fixture cohort with fresh workspace and session per task."""

    if manifest.benchmark != "fixture":
        raise RuntimeError(
            "External benchmark execution must be delegated to its official harness; "
            "use the generated plan rather than treating task IDs as local fixtures."
        )
    output.mkdir(parents=True, exist_ok=True)
    raw_output = output / "raw"
    raw_output.mkdir(exist_ok=True)
    rows: list[CodingAgentRun] = []
    for repeat in range(manifest.repeats):
        for task_id in manifest.task_ids:
            fixture = FIXTURE_TASKS[task_id]
            with tempfile.TemporaryDirectory(prefix=f"pra-agent-{task_id}-") as directory:
                workspace = Path(directory)
                adapter.configure_workspace(workspace)
                execution = adapter.run_task(AgentTask(
                    task_id, fixture["instruction"], workspace,
                    manifest.timeout_seconds, fixture,
                ))
                target = workspace / fixture["file"]
                success = (
                    execution.exit_code == 0
                    and target.is_file()
                    and _read_fixture_text(target) == fixture["content"]
                )
                usage = adapter.collect_usage(execution)
                behavior = dict(execution.behavior)
                artifact_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{task_id}-{repeat}")
                stdout_path = _write_text_artifact(
                    raw_output / f"{artifact_stem}.stdout", execution.stdout
                )
                stderr_path = _write_text_artifact(
                    raw_output / f"{artifact_stem}.stderr", execution.stderr
                )
                rows.append(CodingAgentRun(
                    identity=RunIdentity(
                        run_id=str(uuid.uuid4()), agent=agent, agent_version=adapter.version(),
                        benchmark=manifest.dataset, benchmark_revision=manifest.revision,
                        task_id=task_id, repeat=repeat, engine=engine, host=platform.node(),
                        hardware={"system": platform.system(), "machine": platform.machine()},
                        model=model, pra_version=pra_version, pra_mode=pra_mode,
                        pra_profile=pra_profile, connection=connection, protocol=protocol,
                        cache_state=cache_state,
                    ),
                    outcome=OutcomeMetrics(
                        success=success, official_score=float(success), patch_correct=success,
                        failure_kind="timeout" if execution.timed_out else None if success else "fixture_failed",
                    ),
                    behavior=AgentBehaviorMetrics.model_validate(behavior),
                    tokens=TokenMetrics(
                        input_tokens=int(usage.get("input_tokens", 0)),
                        output_tokens=int(usage.get("output_tokens", 0)),
                        cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
                        cumulative_context_sent=int(usage.get("input_tokens", 0)),
                        max_context_tokens=int(usage.get("input_tokens", 0)),
                    ),
                    timings=TimingMetrics(task_wall_ms=execution.wall_ms),
                    resources=ResourceMetrics(), cost=CostMetrics(),
                    artifacts={
                        "stdout": str(stdout_path.relative_to(output)),
                        "stderr": str(stderr_path.relative_to(output)),
                    },
                    metadata={"harness_validation_only": True},
                ))
                adapter.cleanup()
    path = output / "runs.jsonl"
    path.write_text("".join(row.json_line() + "\n" for row in rows), encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    return rows


def _read_fixture_text(path: Path) -> str | None:
    """Decode ordinary CLI-created text without making encoding a task outcome."""

    content = path.read_bytes()
    encodings = (
        "utf-8-sig",
        "utf-16" if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else None,
    )
    for encoding in encodings:
        if encoding is None:
            continue
        try:
            return content.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    return None


def _write_text_artifact(path: Path, text: str, *, compress_at: int = 524_288) -> Path:
    """Write a transcript, compressing verbose token streams transparently."""

    encoded = text.encode("utf-8")
    compressed_path = path.with_suffix(path.suffix + ".gz")
    if len(encoded) >= compress_at:
        with gzip.open(compressed_path, "wb", compresslevel=6) as output:
            output.write(encoded)
        path.unlink(missing_ok=True)
        return compressed_path
    path.write_bytes(encoded)
    compressed_path.unlink(missing_ok=True)
    return path


def load_runs(path: str | Path) -> list[CodingAgentRun]:
    return [
        CodingAgentRun.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def external_plan(
    manifest: BenchmarkManifest, *, agent: str, model: str, condition: str,
) -> Mapping[str, Any]:
    """Describe official-harness work without pretending it has executed."""

    if manifest.benchmark == "fixture":
        command = ["python", "-m", "experiments.agents", "run", "--manifest", manifest.name]
    else:
        command = benchmark_adapter(manifest).command(
            manifest, agent=agent, model=model, condition=condition,
        )
    return {
        "manifest": manifest.name, "benchmark": manifest.benchmark,
        "dataset": manifest.dataset, "revision": manifest.revision,
        "tasks": list(manifest.task_ids), "repeats": manifest.repeats,
        "agent": agent, "model": model, "condition": condition,
        "official_harness_command": command,
        "status": "QUALIFICATION_PENDING",
    }
