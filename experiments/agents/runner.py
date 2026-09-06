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
from collections import defaultdict
from datetime import datetime
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


def import_harbor_job(
    job_dir: str | Path,
    manifest: BenchmarkManifest,
    *,
    output: Path,
    engine: str,
    engine_version: str | None = None,
    host: str,
    hardware: Mapping[str, Any] | None = None,
    model: str,
    model_revision: str | None = None,
    quantization: str | None = None,
    pra_mode: PRAMode,
    pra_profile: PRAProfile,
    connection: str,
    protocol: str,
    pra_version: str = "0.2.0rc1",
    engine_pra_enabled: bool | None = None,
    gateway_pra_enabled: bool | None = None,
    gateway_mode: str | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> list[CodingAgentRun]:
    """Normalize official Harbor trial results without re-grading them."""

    job_path = Path(job_dir)
    allowed_tasks = set(manifest.task_ids)
    repeats: defaultdict[str, int] = defaultdict(int)
    rows: list[CodingAgentRun] = []
    trial_results = []
    candidates = {
        *job_path.glob("*/result.json"),
        *job_path.glob("*/*/result.json"),
    }
    for path in sorted(candidates):
        if path.parent == job_path:
            continue
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if "task_name" in candidate:
            trial_results.append(path)
    if not trial_results:
        raise ValueError(f"no Harbor trial results found under {job_path}")

    for result_path in trial_results:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
        qualified_task = str(raw["task_name"])
        task_id = qualified_task.removeprefix("terminal-bench/")
        if task_id not in allowed_tasks:
            raise ValueError(f"Harbor result contains unfrozen task {qualified_task!r}")
        repeat = repeats[task_id]
        repeats[task_id] += 1
        agent_info = raw.get("agent_info") or {}
        agent_result = raw.get("agent_result") or {}
        verifier = raw.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        official_score = _harbor_reward(rewards)
        exception = raw.get("exception_info") or {}
        success = exception == {} and official_score is not None and official_score >= 1.0
        trajectory_path = result_path.parent / "agent" / "trajectory.json"
        agent_log = next(
            iter(sorted((result_path.parent / "agent").glob("*.txt"))), None
        )
        behavior = _harbor_behavior(trajectory_path, agent_log=agent_log)
        tests_passed, tests_total = _harbor_test_counts(
            result_path.parent / "verifier" / "ctrf.json"
        )
        artifacts = {
            "harbor_trial": str(result_path.parent.relative_to(job_path)),
            "harbor_result": str(result_path.relative_to(job_path)),
        }
        if trajectory_path.is_file():
            artifacts["trajectory"] = str(trajectory_path.relative_to(job_path))
        if agent_log is not None:
            artifacts["agent_log"] = str(agent_log.relative_to(job_path))
        rows.append(CodingAgentRun(
            identity=RunIdentity(
                run_id=str(raw.get("id") or uuid.uuid4()),
                agent=str(agent_info.get("name") or "unknown"),
                agent_version=str(agent_info.get("version") or "unknown"),
                benchmark=manifest.dataset,
                benchmark_revision=manifest.revision,
                task_id=task_id,
                repeat=repeat,
                engine=engine,
                engine_version=engine_version,
                host=host,
                hardware=dict(hardware or {}),
                model=model,
                model_revision=model_revision,
                quantization=quantization,
                pra_version=pra_version,
                pra_mode=pra_mode,
                pra_profile=pra_profile,
                connection=connection,
                engine_pra_enabled=engine_pra_enabled,
                gateway_pra_enabled=gateway_pra_enabled,
                gateway_mode=gateway_mode,
                protocol=protocol,
                cache_state=CacheState.COLD,
                started_at=str(raw.get("started_at") or datetime.now().isoformat()),
            ),
            outcome=OutcomeMetrics(
                success=success,
                official_score=official_score,
                tests_passed=tests_passed,
                tests_total=tests_total,
                patch_correct=success,
                failure_kind=(
                    str(exception.get("exception_type")) if exception
                    else None if success else "official_score_below_success"
                ),
            ),
            behavior=behavior,
            tokens=TokenMetrics(
                input_tokens=int(agent_result.get("n_input_tokens") or 0),
                output_tokens=int(agent_result.get("n_output_tokens") or 0),
                cached_input_tokens=int(agent_result.get("n_cache_tokens") or 0),
                cumulative_context_sent=int(agent_result.get("n_input_tokens") or 0),
                max_context_tokens=_harbor_max_context(
                    trajectory_path, agent_log=agent_log
                ),
            ),
            timings=TimingMetrics(
                task_wall_ms=_harbor_duration_ms(raw),
                inference_ms=_harbor_duration_ms(raw.get("agent_execution") or {}),
                completion_ms=_harbor_duration_ms(raw.get("verifier") or {}),
            ),
            resources=ResourceMetrics(),
            cost=CostMetrics(total=agent_result.get("cost_usd")),
            artifacts=artifacts,
            metadata={
                "official_harness": "harbor",
                "official_harness_version": "0.22.0",
                "harbor_job_id": raw.get("config", {}).get("job_id"),
                "task_checksum": raw.get("task_checksum"),
                "verifier_environment_mode": raw.get("verifier_environment_mode"),
                "reward_components": rewards,
                **dict(run_metadata or {}),
            },
        ))

    output.mkdir(parents=True, exist_ok=True)
    (output / "runs.jsonl").write_text(
        "".join(row.json_line() + "\n" for row in rows), encoding="utf-8"
    )
    return rows


def _harbor_reward(rewards: Mapping[str, Any]) -> float | None:
    for name in ("reward", "task_success"):
        value = rewards.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    numeric = [float(value) for value in rewards.values() if isinstance(value, (int, float))]
    return numeric[0] if len(numeric) == 1 else None


def _harbor_duration_ms(value: Mapping[str, Any]) -> float:
    started = value.get("started_at")
    finished = value.get("finished_at")
    if not started or not finished:
        return 0.0
    parse = lambda item: datetime.fromisoformat(str(item).replace("Z", "+00:00"))
    return max(0.0, (parse(finished) - parse(started)).total_seconds() * 1000)


def _harbor_trajectory(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _harbor_jsonl_events(path: Path | None) -> list[Mapping[str, Any]]:
    """Load structured events from agent logs such as Pi's JSONL transcript."""

    if path is None or not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            events.append(value)
    return events


def _harbor_max_context(path: Path, *, agent_log: Path | None = None) -> int:
    prompts = [
        int((step.get("metrics") or {}).get("prompt_tokens") or 0)
        for step in _harbor_trajectory(path).get("steps", ())
    ]
    prompts.extend(
        int((event.get("message", {}).get("usage") or {}).get("input") or 0)
        for event in _harbor_jsonl_events(agent_log)
        if event.get("type") == "message_end"
        and event.get("message", {}).get("role") == "assistant"
    )
    return max(prompts, default=0)


def _harbor_behavior(
    path: Path, *, agent_log: Path | None = None
) -> AgentBehaviorMetrics:
    steps = _harbor_trajectory(path).get("steps", ())
    agent_steps = [step for step in steps if step.get("source") == "agent"]
    calls = [call for step in agent_steps for call in (step.get("tool_calls") or ())]
    names = [str(call.get("function_name") or "").lower() for call in calls]
    events = _harbor_jsonl_events(agent_log)
    if not agent_steps and events:
        assistant_messages = [
            event for event in events
            if event.get("type") == "message_end"
            and event.get("message", {}).get("role") == "assistant"
        ]
        tool_events = [
            event for event in events if event.get("type") == "tool_execution_start"
        ]
        names = [str(event.get("toolName") or "").lower() for event in tool_events]
        return AgentBehaviorMetrics(
            turns=len(assistant_messages),
            model_calls=len(assistant_messages),
            tool_calls=len(tool_events),
            shell_calls=sum(
                any(token in name for token in ("bash", "shell", "terminal", "exec"))
                for name in names
            ),
            file_reads=sum(
                any(token in name for token in ("read", "view", "grep", "find"))
                for name in names
            ),
            file_writes=sum(
                any(token in name for token in ("write", "edit", "patch"))
                for name in names
            ),
            tests=sum("test" in name for name in names),
            context_compactions=sum(
                "compact" in str(event.get("type") or "").lower()
                for event in events
            ),
        )
    return AgentBehaviorMetrics(
        turns=len(agent_steps),
        model_calls=sum(int(step.get("llm_call_count") or 0) for step in agent_steps),
        tool_calls=len(calls),
        shell_calls=sum(any(token in name for token in ("bash", "shell", "terminal", "exec")) for name in names),
        file_reads=sum(any(token in name for token in ("read", "view", "grep", "glob")) for name in names),
        file_writes=sum(any(token in name for token in ("write", "edit", "patch")) for name in names),
        tests=sum("test" in name for name in names),
    )


def _harbor_test_counts(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    summary = (
        json.loads(path.read_text(encoding="utf-8"))
        .get("results", {})
        .get("summary", {})
    )
    passed = summary.get("passed")
    total = summary.get("tests")
    return (
        int(passed) if isinstance(passed, (int, float)) else None,
        int(total) if isinstance(total, (int, float)) else None,
    )


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
