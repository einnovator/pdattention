"""Run one locked SWE-bench task through the engine-neutral Paper 8.5 proxy.

The model endpoint receives an ordinary OpenAI chat request.  This runner does
not import a PRA engine, gateway, cache adapter, or K/V implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .auxiliary_workspace_state import (
    AUXILIARY_WORKSPACE_STATE_LABEL,
    create_auxiliary_workspace_state_prediction,
)
from .autonomous_proxy import (
    AUTONOMOUS_POLICIES,
    AutonomousSelectionConfig,
    AutonomousSelectionProxy,
    join_instrumentation_sidecars,
)
from .materialization import MaterializationMode
from .negative_receipts import NegativeRealizationMode
from .selectors import whitespace_tokens


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_persistent_prefix(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load completed episodes without inferring session boundaries."""

    if path is None:
        return [], {"path": None, "sha256": None, "episode_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("persistent prefix requires schema_version=1")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("persistent prefix requires completed episodes")
    result: list[dict[str, Any]] = []
    instance_ids: list[str] = []
    for index, row in enumerate(episodes, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"persistent prefix episode {index} is not an object")
        trajectory = row.get("trajectory", row)
        if not isinstance(trajectory, Mapping):
            raise ValueError(f"persistent prefix episode {index} has no trajectory")
        instance_id = str(trajectory.get("instance_id") or "")
        messages = trajectory.get("messages")
        if not instance_id or not isinstance(messages, list) or not messages:
            raise ValueError(f"persistent prefix episode {index} is incomplete")
        instance_ids.append(instance_id)
        result.append(dict(trajectory))
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("persistent prefix episode instance IDs must be distinct")
    return result, {
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(path.read_bytes()),
        "episode_count": len(result),
        "instance_ids": instance_ids,
        "session_id": payload.get("session_id"),
    }


def export_persistent_episode(
    *, agent_output: Path, instrumentation_root: Path, instance_id: str,
    output: Path,
) -> dict[str, Any]:
    """Export one completed trajectory with selector-only observation metadata."""

    paths = sorted(agent_output.rglob("*.traj.json"))
    if len(paths) != 1:
        raise ValueError(
            f"persistent episode export expected one trajectory, found {len(paths)}"
        )
    trajectory = json.loads(paths[0].read_text(encoding="utf-8"))
    if (
        not isinstance(trajectory, Mapping)
        or trajectory.get("instance_id") != instance_id
        or not isinstance(trajectory.get("messages"), list)
    ):
        raise ValueError("persistent episode trajectory identity is invalid")
    enriched_messages, sidecar_join = join_instrumentation_sidecars(
        [dict(row) for row in trajectory["messages"]], instrumentation_root
    )
    exported_trajectory = dict(trajectory)
    exported_trajectory["messages"] = enriched_messages
    artifact = {
        "schema_version": 1,
        "study": "paper8_5_persistent_episode_export",
        "trajectory": exported_trajectory,
        "trajectory_source": str(paths[0].resolve()),
        "trajectory_source_sha256": _sha256_bytes(paths[0].read_bytes()),
        "instrumentation_sidecar_join": sidecar_join,
    }
    _write_json(output, artifact)
    return artifact


def load_locked_task(
    path: Path,
    *,
    task_index: int | None,
    instance_id: str | None,
) -> tuple[dict[str, Any], str, int]:
    """Select exactly one predeclared task without outcome-based resampling."""

    card = json.loads(path.read_text(encoding="utf-8"))
    ids = card.get("instance_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(row, str) for row in ids):
        raise ValueError("benchmark card must contain non-empty string instance_ids")
    if len(ids) != len(set(ids)) or len(ids) != int(card.get("expected_count", len(ids))):
        raise ValueError("benchmark card instance count or uniqueness is invalid")
    digest = _sha256_bytes(("\n".join(ids) + "\n").encode("utf-8"))
    if digest != str(card.get("canonical_ids_sha256", "")).lower():
        raise ValueError("benchmark card canonical_ids_sha256 does not match ordered IDs")
    if (task_index is None) == (instance_id is None):
        raise ValueError("select exactly one task with --task-index or --instance-id")
    if task_index is not None:
        if task_index < 1 or task_index > len(ids):
            raise ValueError(f"task index must be in 1..{len(ids)}")
        selected = ids[task_index - 1]
        selected_index = task_index
    else:
        if instance_id not in ids:
            raise ValueError(f"instance {instance_id!r} is not in the locked card")
        selected = str(instance_id)
        selected_index = ids.index(selected) + 1
    return card, selected, selected_index


def swebench_image(instance_id: str) -> str:
    docker_instance_id = instance_id.replace("__", "_1776_").lower()
    return "docker.io/swebench/sweb.eval.x86_64." + docker_instance_id + ":latest"


def build_agent_command(
    args: argparse.Namespace,
    *,
    proxy_base_url: str,
    instance_id: str,
    agent_output: Path,
) -> list[str]:
    task_image = swebench_image(instance_id)
    command = [
        sys.executable,
        "-m",
        "minisweagent.run.benchmarks.swebench",
        "--subset",
        "verified",
        "--split",
        str(args.split),
        "--filter",
        f"({re.escape(instance_id)})",
        "-m",
        f"openai/{args.served_model}",
        "--model-class",
        "litellm_textbased",
        "-c",
        str(args.scaffold),
        "-c",
        f"model.model_kwargs.api_base={proxy_base_url}",
        "-c",
        f"model.model_kwargs.temperature={float(args.temperature)}",
        "-c",
        f"model.model_kwargs.top_p={float(args.top_p)}",
        "-c",
        f"model.model_kwargs.seed={int(args.seed)}",
        "-c",
        "model.model_kwargs.stream=false",
    ]
    if args.max_completion_tokens is not None:
        command.extend((
            "-c", f"model.model_kwargs.max_tokens={args.max_completion_tokens}"
        ))
    if args.docker_executable is not None:
        command.extend((
            "-c", f"environment.executable={Path(args.docker_executable).resolve()}"
        ))
    if args.instrument_observations:
        command.extend((
            "-c",
            "environment.environment_class="
            "experiments.paper8_5_agent_memory.miniswe_environment."
            "InstrumentedDockerEnvironment",
            "-c",
            f"environment.image={task_image}",
            "-c",
            f"environment.instrumentation_output_root={args.instrumentation_output_root}",
            "-c",
            "environment.capture_workspace_checkpoints=true",
        ))
    command.extend((
        "-c",
        f"agent.step_limit={args.max_calls}",
        "-w",
        "1",
        "-o",
        str(agent_output),
    ))
    return command


def agent_behavior_digest(command: Sequence[str]) -> str:
    """Hash behavioral CLI state without run-local transport/output paths."""

    if not command:
        raise ValueError("agent command cannot be empty")
    normalized: list[str] = ["<python>"]
    index = 1
    while index < len(command):
        argument = str(command[index])
        if argument == "-o":
            index += 2
            continue
        if argument == "-c" and index + 1 < len(command):
            value = str(command[index + 1])
            key = value.split("=", 1)[0]
            if key == "model.model_kwargs.api_base":
                value = key + "=<proxy>"
            elif key == "environment.executable":
                value = key + "=<docker-executable>"
            elif key == "environment.instrumentation_output_root":
                value = key + "=<instrumentation-output>"
            normalized.extend((argument, value))
            index += 2
            continue
        normalized.append(argument)
        index += 1
    return _sha256_bytes(json.dumps(
        normalized, separators=(",", ":"), ensure_ascii=False,
    ).encode())


def build_grader_command(
    args: argparse.Namespace,
    *,
    dataset: str,
    instance_id: str,
    predictions: Path,
    output: Path,
    run_id: str | None = None,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.paper8_5_agent_memory.swebench_grader_entrypoint",
        "-d",
        dataset,
        "-s",
        str(args.split),
        "-p",
        str(predictions),
        "--instance_ids",
        instance_id,
        "--run_id",
        str(run_id or args.run_id),
        "--max_workers",
        str(args.grader_workers),
        "--cache_level",
        "base",
        "--clean",
        "True",
        "--report_dir",
        str(output),
    ]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _scaffold_identity(scaffold: str, harness_version: str | None) -> dict[str, Any]:
    """Bind the agent prompt scaffold to bytes when possible.

    mini-swe-agent accepts either a filesystem path or the basename of a
    packaged scaffold.  A symbolic name alone is not a sufficient pairing
    identity: the same name can denote different prompts across installs.
    The package-version-bound fallback remains explicit for unusual package
    layouts, rather than silently presenting a null content hash as equality.
    """

    requested = str(scaffold)
    direct = Path(requested).expanduser()
    candidates: list[Path] = []
    if direct.is_file():
        candidates.append(direct.resolve())
    else:
        spec = importlib.util.find_spec("minisweagent")
        for root in (spec.submodule_search_locations or ()) if spec else ():
            candidates.extend(Path(root).rglob(Path(requested).name))
    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(unique) == 1:
        path = unique[0]
        return {
            "requested": requested,
            "resolution": "content",
            "resolved_path": str(path),
            "content_sha256": _sha256_bytes(path.read_bytes()),
            "identity_sha256": _sha256_bytes(path.read_bytes()),
        }
    symbolic = {
        "requested": requested,
        "harness_version": harness_version,
        "resolution": "package_version_bound_symbolic",
    }
    return {
        **symbolic,
        "resolved_path": None,
        "content_sha256": None,
        "identity_sha256": _sha256_bytes(json.dumps(
            symbolic, sort_keys=True, separators=(",", ":")
        ).encode()),
    }


def _docker_image_identity(
    args: argparse.Namespace, *, image: str, environment: Mapping[str, str]
) -> dict[str, Any]:
    """Return immutable image identity after the agent has materialized it."""

    executable = str(args.docker_executable or "docker")
    completed = subprocess.run(
        [executable, "image", "inspect", image],
        capture_output=True,
        text=True,
        env=dict(environment),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "cannot establish immutable workspace image identity after the agent run: "
            + completed.stderr.strip()
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeError("Docker image inspect returned an ambiguous identity")
    row = payload[0]
    image_id = row.get("Id")
    if not isinstance(image_id, str) or not image_id:
        raise RuntimeError("Docker image inspect omitted the immutable image ID")
    return {
        "environment_image_id": image_id,
        "environment_image_repo_digests": sorted(row.get("RepoDigests") or ()),
        "environment_image_os": row.get("Os"),
        "environment_image_architecture": row.get("Architecture"),
    }


def _exact_token_counter(
    tokenizer_name: str,
    tokenizer_revision: str,
    *,
    allow_whitespace: bool,
):
    if tokenizer_name == "whitespace":
        if not allow_whitespace:
            raise ValueError(
                "whitespace token counting requires --allow-whitespace-tokenizer"
            )
        return whitespace_tokens, "whitespace_v1_diagnostic_only"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
    )
    return (
        lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        f"{tokenizer_name}@{tokenizer_revision}",
    )


def _git_revision() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _run(
    command: Sequence[str],
    *,
    log: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
    cwd: Path | None = None,
) -> float:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        env=dict(environment),
        cwd=cwd,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log.write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}; see {log}")
    return elapsed


def _active_agent_containers(
    args: argparse.Namespace, environment: Mapping[str, str]
) -> set[str]:
    executable = str(args.docker_executable or "docker")
    result = subprocess.run(
        [
            executable, "ps", "--filter", "name=^/minisweagent-",
            "--format", "{{.ID}}",
        ],
        capture_output=True, text=True, env=dict(environment), check=False,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _cleanup_owned_agent_containers(
    args: argparse.Namespace, environment: Mapping[str, str], before: set[str],
) -> list[str]:
    owned = sorted(_active_agent_containers(args, environment) - before)
    if owned:
        executable = str(args.docker_executable or "docker")
        subprocess.run(
            [executable, "rm", "-f", *owned], capture_output=True, text=True,
            env=dict(environment), check=False,
        )
    return owned


def _run_agent_fail_closed(
    command: Sequence[str], *, log: Path, environment: Mapping[str, str],
    timeout_seconds: int, proxy: AutonomousSelectionProxy,
    args: argparse.Namespace,
) -> float:
    """Terminate the owned agent after the first upstream transport failure."""

    started = time.perf_counter()
    containers_before = _active_agent_containers(args, environment)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            list(command), stdout=stream, stderr=subprocess.STDOUT,
            text=True, env=dict(environment),
        )
        failure: str | None = None
        try:
            while process.poll() is None:
                if proxy.upstream_failed:
                    failure = (
                        "upstream transport failed during agent execution: "
                        f"{proxy.upstream_failure_detail}"
                    )
                    process.terminate()
                    break
                if time.perf_counter() - started > timeout_seconds:
                    failure = f"agent exceeded {timeout_seconds}s execution timeout"
                    process.terminate()
                    break
                time.sleep(0.25)
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if failure is not None:
                removed = _cleanup_owned_agent_containers(
                    args, environment, containers_before
                )
                stream.write(f"\n--- FAIL-CLOSED ---\n{failure}\n")
                stream.write(f"owned_containers_removed={removed!r}\n")
                raise RuntimeError(failure)
            if process.returncode:
                raise RuntimeError(
                    f"command exited {process.returncode}; see {log}"
                )
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            raise
    return time.perf_counter() - started


def _execution_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("OPENAI_API_KEY", "paper8-5-local-proxy")
    environment.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    root_path = Path(__file__).resolve().parents[2]
    # The repository uses a ``src`` layout for ``pra_hf`` while the Paper 8.5
    # harness lives at repository root.  Graders run with the episode output as
    # their cwd, so relying on the launch cwd makes the wrapper importable but
    # leaves its ``pra_hf`` dependency unavailable.  Bind both roots explicitly
    # for every child process.
    python_paths = [
        str(root_path),
        str(root_path / "src"),
        *(str(Path(row).resolve()) for row in args.pythonpath),
    ]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    if args.docker_executable is not None:
        executable = Path(args.docker_executable).resolve()
        if not executable.is_file():
            raise ValueError(f"docker executable does not exist: {executable}")
        environment["PATH"] = os.pathsep.join((
            str(executable.parent), environment.get("PATH", "")
        ))
        environment["PAPER8_5_DOCKER_EXECUTABLE"] = str(executable)
    if args.docker_platform:
        environment["DOCKER_DEFAULT_PLATFORM"] = args.docker_platform
    return environment


def _ensure_evaluation_image(
    args: argparse.Namespace,
    *,
    instance_id: str,
    environment: Mapping[str, str],
    log: Path,
) -> None:
    """Ensure the locked architecture exists before a grading pass.

    SWE-bench's ``--clean True`` can remove the image after the primary grade.
    On Apple Silicon the Python Docker client may then ignore
    ``DOCKER_DEFAULT_PLATFORM`` while rebuilding the auxiliary grade and try an
    unavailable arm64 manifest.  Reuse a local image only after checking its
    platform; otherwise pull with the same explicit Docker CLI/platform
    contract as the agent environment.  This avoids making every grade depend
    on Docker Hub when the exact image is already resident.
    """

    if not args.docker_platform:
        return
    executable = str(args.docker_executable or "docker")
    image = swebench_image(instance_id)
    inspect = subprocess.run(
        [
            executable,
            "image",
            "inspect",
            "--format",
            "{{.Os}}/{{.Architecture}}",
            image,
        ],
        capture_output=True,
        text=True,
        env=dict(environment),
        check=False,
    )
    observed_platform = inspect.stdout.strip()
    if inspect.returncode == 0 and observed_platform == str(args.docker_platform):
        log.write_text(
            f"Reused resident image {image} ({observed_platform}).\n",
            encoding="utf-8",
        )
        return
    _run(
        [
            executable,
            "pull",
            "--platform",
            str(args.docker_platform),
            image,
        ],
        log=log,
        environment=environment,
        timeout_seconds=args.timeout_seconds,
    )


def summarize_trace(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if path.is_file() else []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    repeated = {"search": 0, "read": 0, "test": 0}
    action_counts = {"search": 0, "read": 0, "test": 0}
    for row in rows:
        category = (
            "search" if row.get("assistant_is_search") else
            "read" if row.get("assistant_is_read") else
            "test" if row.get("assistant_is_test") else None
        )
        if category is None:
            continue
        action_counts[category] += 1
        resources = tuple(sorted(str(value) for value in row.get("assistant_resource_ids") or ()))
        signature = (category, resources or (str(row.get("assistant_command_sha256")),))
        if signature in seen:
            repeated[category] += 1
        seen.add(signature)
    total_full = sum(int(row.get("full_tokens") or 0) for row in rows)
    total_selected = sum(int(row.get("selected_tokens") or 0) for row in rows)
    total_materialized = sum(int(row.get("materialized_tokens") or 0) for row in rows)
    usage_rows = [
        row for row in rows if row.get("reported_completion_tokens") is not None
    ]
    reported_completion = sum(
        int(row["reported_completion_tokens"]) for row in usage_rows
    )
    return {
        "schema_version": 1,
        "evidence_class": "autonomous_agent_logical_selection",
        "calls": len(rows),
        "actions": sum(row.get("assistant_command_sha256") is not None for row in rows),
        "action_counts": action_counts,
        "repeated_same_operation_resource_counts": repeated,
        "cumulative_full_tokens": total_full,
        "cumulative_selected_tokens": total_selected,
        "cumulative_materialized_tokens": total_materialized,
        "reported_usage_coverage_calls": len(usage_rows),
        "cumulative_reported_completion_tokens": reported_completion,
        "cumulative_materialized_plus_reported_completion_tokens": (
            total_materialized + reported_completion
        ),
        "cumulative_logical_retention_fraction": (
            total_selected / total_full if total_full else 1.0
        ),
        "cumulative_materialized_retention_fraction": (
            total_materialized / total_full if total_full else 1.0
        ),
        "excluded_causal_groups": sum(
            int(row.get("excluded_causal_group_count") or 0) for row in rows
        ),
        "excluded_tokens": sum(int(row.get("excluded_tokens") or 0) for row in rows),
        "reacquisition_events": sum(
            int(row.get("reacquisition_count") or 0) for row in rows
        ),
        "reacquired_observation_tokens": sum(
            int(row.get("reacquired_observation_tokens_from_previous_action") or 0)
            for row in rows
        ),
        "requests_over_budget": sum(not bool(row.get("budget_satisfied")) for row in rows),
        "upstream_error_calls": sum(
            int(row.get("upstream_status") or 0) < 100
            or int(row.get("upstream_status") or 0) >= 400
            for row in rows
        ),
        "upstream_transport_error_calls": sum(
            bool(row.get("upstream_transport_error")) for row in rows
        ),
        "metric_note": (
            "token counts cover message content and exclude chat-template tokens; repeated "
            "counts mean a repeated operation/resource signature, not redundant intent; "
            "completion tokens use endpoint-reported usage with explicit coverage"
        ),
    }


def _official_report(
    output: Path, run_id: str, instance_id: str, *, grader_log: Path | None = None,
) -> dict[str, Any]:
    matches = sorted(output.glob(f"*.{run_id}.json"))
    report_kind = "aggregate"
    if len(matches) == 1:
        raw = json.loads(matches[0].read_text(encoding="utf-8"))
        submitted = set(raw.get("submitted_ids") or ())
        if submitted != {instance_id}:
            raise RuntimeError("official grader report does not match the locked single task")
        resolved = instance_id in set(raw.get("resolved_ids") or ())
        error = instance_id in set(raw.get("error_ids") or ())
    elif not matches:
        # SWE-bench can finish the locked instance and persist its signed
        # per-instance report, then fail while enumerating already-removed
        # Docker containers during aggregate-report cleanup.  The task result
        # is still auditable; prefer it to parsing progress-bar text.
        instance_matches = sorted(
            output.glob(
                f"logs/run_evaluation/{run_id}/**/{instance_id}/report.json"
            )
        )
        if len(instance_matches) != 1:
            grader_text = (
                grader_log.read_text(encoding="utf-8", errors="replace")
                if grader_log is not None and grader_log.is_file() else ""
            )
            # A malformed submission can fail before SWE-bench writes either
            # an aggregate or per-instance report.  This line is emitted by
            # the locked instance worker itself, so it is exact task evidence
            # rather than a progress-bar inference.  Count the attempt as an
            # unresolved submission-protocol failure instead of losing it as
            # an indeterminate grader error.
            patch_apply_marker = f"{instance_id}: >>>>> Patch Apply Failed:"
            if patch_apply_marker in grader_text:
                return {
                    "official_grader": True,
                    "instance_id": instance_id,
                    "resolved": False,
                    "score": 0.0,
                    "error": True,
                    "failure_class": "patch_apply_failed",
                    "raw_report": str(grader_log),
                    "raw_report_kind": "per_instance_failure_log",
                }
            raise RuntimeError(
                f"expected one official report for {run_id}, found 0 aggregate "
                f"and {len(instance_matches)} per-instance reports"
            )
        matches = instance_matches
        raw = json.loads(matches[0].read_text(encoding="utf-8"))
        instance = raw.get(instance_id)
        if not isinstance(instance, Mapping):
            raise RuntimeError("per-instance grader report does not match the locked task")
        resolved = bool(instance.get("resolved"))
        error = not bool(instance.get("patch_successfully_applied", True))
        report_kind = "per_instance"
    else:
        raise RuntimeError(f"expected one official report for {run_id}, found {len(matches)}")
    failure_class = None
    if error:
        grader_text = (
            grader_log.read_text(encoding="utf-8", errors="replace")
            if grader_log is not None and grader_log.is_file() else ""
        )
        failure_class = (
            "patch_apply_failed" if "Patch Apply Failed" in grader_text
            else "grader_reported_error"
        )
    return {
        "official_grader": True,
        "instance_id": instance_id,
        "resolved": resolved,
        "score": 1.0 if resolved else 0.0,
        "error": error,
        "failure_class": failure_class,
        "raw_report": str(matches[0]),
        "raw_report_kind": report_kind,
    }


def _unreported_submission_result(
    predictions: Path,
    *,
    instance_id: str,
    error: Exception,
    grader_log: Path,
) -> dict[str, Any]:
    """Classify exact empty submissions while preserving other uncertainty."""

    prediction_payload = json.loads(predictions.read_text(encoding="utf-8"))
    prediction = prediction_payload.get(instance_id) or {}
    submitted_patch_empty = not bool(str(prediction.get("model_patch") or "").strip())
    return {
        "official_grader": True,
        "instance_id": instance_id,
        "resolved": False if submitted_patch_empty else None,
        "score": 0.0 if submitted_patch_empty else None,
        "error": True,
        "failure_class": "empty_submission" if submitted_patch_empty else None,
        "error_detail": str(error),
        "submitted_patch_empty": submitted_patch_empty,
        "grader_log": str(grader_log),
    }


def run(args: argparse.Namespace) -> Path:
    card, instance_id, task_index = load_locked_task(
        args.benchmark_card,
        task_index=args.task_index,
        instance_id=args.instance_id,
    )
    if args.split != card.get("split"):
        raise ValueError(
            f"split mismatch: CLI {args.split!r}, card {card.get('split')!r}"
        )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    args.instrumentation_output_root = (
        args.instrumentation_output_root.resolve()
        if args.instrumentation_output_root is not None else output / "instrumentation"
    )
    count_tokens, tokenizer_identity = _exact_token_counter(
        args.tokenizer,
        args.tokenizer_revision,
        allow_whitespace=args.allow_whitespace_tokenizer,
    )
    prior_episodes, persistent_prefix_identity = load_persistent_prefix(
        args.persistent_prefix
    )
    expected_episode_index = len(prior_episodes) + 1
    if args.episode_index != expected_episode_index:
        raise ValueError(
            "episode-index must equal completed prefix episode count plus one: "
            f"expected {expected_episode_index}, got {args.episode_index}"
        )
    if prior_episodes and not args.session_id:
        raise ValueError("persistent-prefix requires an explicit session-id")
    prefix_session_id = persistent_prefix_identity.get("session_id")
    if (
        prior_episodes
        and prefix_session_id
        and prefix_session_id != args.session_id
    ):
        raise ValueError("persistent-prefix session-id does not match the CLI session-id")
    config = AutonomousSelectionConfig(
        policy=args.policy,
        budget_fraction=args.budget_fraction,
        protected_head_turns=args.head,
        protected_tail_turns=args.tail,
        search_delay_turns=args.search_delay_turns,
        write_delay_turns=args.write_delay_turns,
        same_span_reads_to_keep=args.same_span_reads_to_keep,
        working_set_resources=args.working_set_resources,
        h2b_allow_workspace_verification=args.h2b_allow_workspace_verification,
        materialization_mode=MaterializationMode(args.materialization_mode),
        materialization_threshold_tokens=args.materialization_threshold_tokens,
        materialization_head_lines=args.materialization_head_lines,
        materialization_tail_lines=args.materialization_tail_lines,
        materialization_match_context_lines=args.materialization_match_context_lines,
        materialization_max_matched_lines=args.materialization_max_matched_lines,
        expected_model=args.served_model,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        max_calls=args.max_calls,
        max_completion_tokens=args.max_completion_tokens,
        tokenizer_identity=tokenizer_identity,
        task_id=instance_id,
        session_id=args.session_id,
        episode_index=args.episode_index,
        completed_recent_turns=args.completed_recent_turns,
        completed_mutation_turns=args.completed_mutation_turns,
        completed_verification_turns=args.completed_verification_turns,
        completed_protocol_turns=args.completed_protocol_turns,
        completed_finalization_turns=args.completed_finalization_turns,
        completed_instruction_epochs=args.completed_instruction_epochs,
        keep_completed_task_statements=args.keep_completed_task_statements,
        boundary_mode=args.boundary_mode,
        require_exact_sidecars=args.require_exact_sidecars,
        negative_realization=NegativeRealizationMode(args.negative_realization),
        negative_fallback=args.negative_fallback,
    )
    trace_path = output / "request_selection.jsonl"
    agent_output = output / "agent"
    placeholder_proxy = "http://127.0.0.1:PORT/v1"
    agent_command = build_agent_command(
        args,
        proxy_base_url=placeholder_proxy,
        instance_id=instance_id,
        agent_output=agent_output,
    )
    grader_command = build_grader_command(
        args,
        dataset=str(card["dataset"]),
        instance_id=instance_id,
        predictions=agent_output / "preds.json",
        output=output,
    )
    auxiliary_run_id = f"{args.run_id}-aux-workspace-state"
    auxiliary_predictions = output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}_preds.json"
    auxiliary_grader_command = build_grader_command(
        args,
        dataset=str(card["dataset"]),
        instance_id=instance_id,
        predictions=auxiliary_predictions,
        output=output,
        run_id=auxiliary_run_id,
    )
    observed_harness_version = _package_version("mini-swe-agent")
    scaffold_identity = _scaffold_identity(args.scaffold, observed_harness_version)
    manifest = {
        "schema_version": 2,
        "study": "paper8_5_autonomous_agent_memory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": _git_revision(),
        "benchmark_card": str(args.benchmark_card.resolve()),
        "benchmark_card_sha256": _sha256_bytes(args.benchmark_card.read_bytes()),
        "benchmark_ids_sha256": card["canonical_ids_sha256"],
        "dataset": card["dataset"],
        "dataset_revision": card.get("execution_dataset_revision"),
        "split": card["split"],
        "instance_id": instance_id,
        "task_index": task_index,
        "pair_id": args.pair_id,
        "persistent_session": {
            "session_id": args.session_id or instance_id,
            "episode_index": args.episode_index,
            "prefix": persistent_prefix_identity,
            "logical_session_continues_across_agent_processes": bool(prior_episodes),
            "workspace_continuity_claimed": False,
        },
        "selection": {
            **asdict(config),
            "materialization_mode": config.materialization_mode.value,
            "negative_realization": config.negative_realization.value,
        },
        "model": args.model,
        "served_model": args.served_model,
        "model_revision": args.model_revision,
        "tokenizer": tokenizer_identity,
        "tokenizer_revision": args.tokenizer_revision,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_calls": args.max_calls,
        "max_completion_tokens": args.max_completion_tokens,
        "harness": "mini-swe-agent",
        "harness_version_requested": args.harness_version,
        "harness_version_observed": observed_harness_version,
        "grader": "official SWE-bench Docker harness",
        "grader_version_requested": args.grader_version,
        "grader_version_observed": _package_version("swebench"),
        "upstream_base_url": args.upstream_base_url,
        "upstream_api_key_environment": args.upstream_api_key_env,
        "upstream_qualification_path": args.upstream_qualification_path,
        "upstream_connect_attempts": args.upstream_connect_attempts,
        "upstream_connect_retry_seconds": args.upstream_connect_retry_seconds,
        "upstream_curl_executable": args.upstream_curl_executable,
        "upstream_relay_target": args.upstream_relay_target,
        "docker_executable": str(args.docker_executable) if args.docker_executable else None,
        "docker_platform": args.docker_platform,
        "environment_image": swebench_image(instance_id),
        "environment_image_id": None,
        "workspace_source_identity_sha256": None,
        "pythonpath": args.pythonpath,
        "instrument_observations": args.instrument_observations,
        "instrumentation_output_root": str(args.instrumentation_output_root),
        "agent_command_template": agent_command,
        "agent_behavior_sha256": agent_behavior_digest(agent_command),
        "scaffold_identity": scaffold_identity,
        "scaffold_identity_sha256": scaffold_identity["identity_sha256"],
        "grader_command": grader_command,
        "official_grading_enabled": not args.skip_grading,
        "auxiliary_workspace_state": {
            "outcome_label": AUXILIARY_WORKSPACE_STATE_LABEL,
            "evidence_role": "auxiliary_only",
            "replaces_official_submission": False,
            "extraction_enabled": args.instrument_observations,
            "official_grading_requested": args.grade_auxiliary_workspace_state,
            "official_grading_enabled": (
                args.grade_auxiliary_workspace_state and not args.skip_grading
            ),
            "predictions": str(auxiliary_predictions),
            "grader_run_id": auxiliary_run_id,
            "grader_command": auxiliary_grader_command,
        },
        "preflight_only": args.preflight_only,
        "engine_or_kv_metrics_claimed": False,
    }
    _write_json(output / "run_manifest.json", manifest)
    if args.preflight_only:
        return output / "run_manifest.json"

    environment = _execution_environment(args)
    upstream_key = os.environ.get(args.upstream_api_key_env)
    proxy = AutonomousSelectionProxy(
        args.upstream_base_url,
        config=config,
        trace_path=trace_path,
        count_tokens=count_tokens,
        upstream_api_key=upstream_key,
        timeout_seconds=args.upstream_timeout_seconds,
        instrumentation_root=args.instrumentation_output_root,
        prior_episodes=prior_episodes,
        upstream_qualification_path=args.upstream_qualification_path,
        upstream_connect_attempts=args.upstream_connect_attempts,
        upstream_connect_retry_seconds=args.upstream_connect_retry_seconds,
        upstream_curl_executable=args.upstream_curl_executable,
    )
    proxy_url = proxy.start()
    try:
        agent_command = build_agent_command(
            args,
            proxy_base_url=proxy_url,
            instance_id=instance_id,
            agent_output=agent_output,
        )
        agent_output.mkdir(parents=True, exist_ok=True)
        agent_wall_time = _run_agent_fail_closed(
            agent_command,
            log=output / "agent.log",
            environment=environment,
            timeout_seconds=args.timeout_seconds,
            proxy=proxy,
            args=args,
        )
    finally:
        proxy.close()

    image_identity = _docker_image_identity(
        args,
        image=swebench_image(instance_id),
        environment=environment,
    )
    manifest.update(image_identity)
    workspace_source = {
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["dataset_revision"],
        "split": manifest["split"],
        "instance_id": manifest["instance_id"],
        "environment_image_id": manifest["environment_image_id"],
        "docker_platform": manifest["docker_platform"],
    }
    manifest["workspace_source_identity_sha256"] = _sha256_bytes(json.dumps(
        workspace_source, sort_keys=True, separators=(",", ":")
    ).encode())
    _write_json(output / "run_manifest.json", manifest)

    predictions = agent_output / "preds.json"
    if not predictions.is_file():
        raise RuntimeError(f"mini-swe-agent did not produce {predictions}")
    auxiliary = create_auxiliary_workspace_state_prediction(
        instrumentation_root=args.instrumentation_output_root,
        output=output,
        primary_predictions=predictions,
        instance_id=instance_id,
    )
    persistent_episode = export_persistent_episode(
        agent_output=agent_output,
        instrumentation_root=args.instrumentation_output_root,
        instance_id=instance_id,
        output=output / "persistent_episode_export.json",
    )
    auxiliary["official_grading_requested"] = bool(
        args.grade_auxiliary_workspace_state and not args.skip_grading
    )
    _write_json(output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.json", auxiliary)
    metrics = summarize_trace(trace_path)
    metrics["agent_wall_time_seconds"] = agent_wall_time
    metrics["instance_id"] = instance_id
    metrics["auxiliary_workspace_state"] = auxiliary
    metrics["persistent_episode_export"] = {
        "path": str(output / "persistent_episode_export.json"),
        "sha256": _sha256_bytes(
            (output / "persistent_episode_export.json").read_bytes()
        ),
        "sidecar_join": persistent_episode["instrumentation_sidecar_join"],
    }
    _write_json(output / "autonomous_metrics.json", metrics)
    if metrics["calls"] == 0:
        raise RuntimeError(
            "mini-swe-agent completed without a model request; this is an "
            "infrastructure failure, not an unresolved task result (see agent.log)"
        )
    if args.skip_grading:
        return output / "autonomous_metrics.json"

    official_result_path = output / "official_result.json"
    try:
        _ensure_evaluation_image(
            args,
            instance_id=instance_id,
            environment=environment,
            log=output / "grader_image_pull.log",
        )
        grader_wall_time = _run(
            grader_command,
            log=output / "grader.log",
            environment=environment,
            timeout_seconds=args.timeout_seconds,
            cwd=output,
        )
        result = _official_report(
            output, args.run_id, instance_id, grader_log=output / "grader.log",
        )
        result["grader_wall_time_seconds"] = grader_wall_time
    except Exception as error:
        try:
            result = _official_report(
                output, args.run_id, instance_id, grader_log=output / "grader.log",
            )
            result["grader_process_error"] = True
            result["grader_process_error_detail"] = str(error)
        except Exception:
            result = _unreported_submission_result(
                predictions,
                instance_id=instance_id,
                error=error,
                grader_log=output / "grader.log",
            )
    result["autonomous_metrics"] = str(output / "autonomous_metrics.json")
    _write_json(official_result_path, result)
    auxiliary["primary_official_result"] = str(official_result_path)
    auxiliary["primary_official_result_sha256"] = _sha256_bytes(
        official_result_path.read_bytes()
    )
    metrics["official_result"] = result
    if (
        args.grade_auxiliary_workspace_state
        and auxiliary["status"] == "available"
    ):
        try:
            _ensure_evaluation_image(
                args,
                instance_id=instance_id,
                environment=environment,
                log=output / "auxiliary_workspace_state_grader_image_pull.log",
            )
            auxiliary_grader_wall_time = _run(
                auxiliary_grader_command,
                log=output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}_grader.log",
                environment=environment,
                timeout_seconds=args.timeout_seconds,
                cwd=output,
            )
            auxiliary_result = _official_report(
                output, auxiliary_run_id, instance_id,
                grader_log=output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}_grader.log",
            )
            auxiliary_result.update({
                "outcome_label": AUXILIARY_WORKSPACE_STATE_LABEL,
                "evidence_role": "auxiliary_only",
                "replaces_official_submission": False,
                "grader_wall_time_seconds": auxiliary_grader_wall_time,
                "predictions": str(auxiliary_predictions),
                "workspace_state_provenance": str(
                    output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.json"
                ),
            })
            auxiliary_result_path = (
                output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}_official_result.json"
            )
            _write_json(auxiliary_result_path, auxiliary_result)
            auxiliary["official_grading_completed"] = True
            auxiliary["official_result"] = str(auxiliary_result_path)
            auxiliary["official_result_summary"] = auxiliary_result
        except Exception as error:  # Auxiliary grading cannot replace primary outcome.
            auxiliary["official_grading_error"] = str(error)
    _write_json(output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.json", auxiliary)
    metrics["auxiliary_workspace_state"] = auxiliary
    _write_json(output / "autonomous_metrics.json", metrics)
    return output / "official_result.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-card", type=Path, required=True)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument("--task-index", type=int)
    task.add_argument("--instance-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--pair-id",
        help=(
            "Stable identifier shared by a FULL control and its candidate arm; "
            "required by multi-task trade-off reducers for paired deltas."
        ),
    )
    parser.add_argument(
        "--persistent-prefix",
        type=Path,
        help=(
            "Schema-v1 completed-episode prefix to prepend logically to every "
            "request in this issue. The source trajectories are never inferred."
        ),
    )
    parser.add_argument(
        "--session-id",
        help="Stable logical session identity shared by an ordered issue sequence.",
    )
    parser.add_argument("--episode-index", type=int, default=1)
    parser.add_argument("--completed-recent-turns", type=int, default=1)
    parser.add_argument("--completed-mutation-turns", type=int, default=1)
    parser.add_argument("--completed-verification-turns", type=int, default=1)
    parser.add_argument("--completed-protocol-turns", type=int, default=0)
    parser.add_argument(
        "--completed-finalization-turns",
        type=int,
        default=0,
        help=(
            "Keep this many terminal submission/finalization turns from each "
            "retired instruction epoch. This prevents preserved user prompts "
            "from appearing unanswered after their interaction detail is retired."
        ),
    )
    parser.add_argument(
        "--completed-instruction-epochs",
        type=int,
        default=0,
        help=(
            "Keep this many immediately preceding genuine-user instruction "
            "epochs whole under instruction-epoch retirement."
        ),
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("explicit", "boundary_free"),
        default="explicit",
        help=(
            "Whether persistent issue transitions are exposed to the model/policy "
            "or retained only in the evaluator ledger."
        ),
    )
    parser.add_argument(
        "--keep-completed-task-statements",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--upstream-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", required=True, help="Published model identity.")
    parser.add_argument("--served-model", required=True, help="Exact OpenAI request model value.")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--policy", choices=AUTONOMOUS_POLICIES, default="full")
    parser.add_argument(
        "--negative-realization",
        choices=tuple(mode.value for mode in NegativeRealizationMode),
        default=NegativeRealizationMode.DROP.value,
    )
    parser.add_argument(
        "--negative-fallback",
        choices=("none", "recency"),
        default="none",
        help="Apply a progress-spine recency budget after certified exclusions.",
    )
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--tail", type=int, default=1)
    parser.add_argument("--search-delay-turns", type=int, default=0)
    parser.add_argument("--write-delay-turns", type=int, default=1)
    parser.add_argument("--same-span-reads-to-keep", type=int, default=1)
    parser.add_argument("--working-set-resources", type=int, default=4)
    parser.add_argument("--h2b-allow-workspace-verification", action="store_true")
    parser.add_argument(
        "--materialization-mode",
        choices=(
            MaterializationMode.WHOLE_RECORD.value,
            MaterializationMode.TOOL_HEAD_TAIL.value,
            MaterializationMode.TOOL_MATCHED_SPAN.value,
            MaterializationMode.TOOL_STRUCTURED_EVIDENCE.value,
        ),
        default=MaterializationMode.WHOLE_RECORD.value,
    )
    parser.add_argument("--materialization-threshold-tokens", type=int, default=512)
    parser.add_argument("--materialization-head-lines", type=int, default=20)
    parser.add_argument("--materialization-tail-lines", type=int, default=30)
    parser.add_argument("--materialization-match-context-lines", type=int, default=4)
    parser.add_argument("--materialization-max-matched-lines", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--scaffold", default="swebench_backticks.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--harness-version", default="2.4.6")
    parser.add_argument("--grader-version", default="4.1.0")
    parser.add_argument("--grader-workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    parser.add_argument("--upstream-timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--upstream-qualification-path",
        help=(
            "Optional non-generating GET path used to qualify and retain one "
            "upstream connection before any model POST."
        ),
    )
    parser.add_argument("--upstream-connect-attempts", type=int, default=1)
    parser.add_argument("--upstream-connect-retry-seconds", type=float, default=1.0)
    parser.add_argument("--upstream-curl-executable")
    parser.add_argument(
        "--upstream-relay-target",
        help="Audit-only destination of a byte-transparent local TCP relay.",
    )
    parser.add_argument("--docker-executable", type=Path)
    parser.add_argument("--docker-platform")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--instrumentation-output-root", type=Path)
    parser.add_argument(
        "--instrument-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-exact-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use FULL for a decision instead of applying a negative policy when the "
            "ordered execution-receipt join is not exact."
        ),
    )
    parser.add_argument("--skip-grading", action="store_true")
    parser.add_argument(
        "--grade-auxiliary-workspace-state",
        action="store_true",
        help=(
            "Separately grade a fail-closed final workspace patch when one is "
            "available; never replaces the official submitted-patch outcome."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-whitespace-tokenizer",
        action="store_true",
        help="Explicitly permit structural-only whitespace accounting; never paper evidence.",
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(result)


if __name__ == "__main__":
    main()
