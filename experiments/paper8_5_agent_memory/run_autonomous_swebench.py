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
)
from .materialization import MaterializationMode
from .negative_receipts import NegativeRealizationMode
from .selectors import whitespace_tokens


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        "swebench.harness.run_evaluation",
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


def _execution_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("OPENAI_API_KEY", "paper8-5-local-proxy")
    environment.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    root = str(Path(__file__).resolve().parents[2])
    python_paths = [root, *(str(Path(row).resolve()) for row in args.pythonpath)]
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
        "upstream_error_calls": sum(int(row.get("upstream_status") or 0) >= 400 for row in rows),
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
    manifest = {
        "schema_version": 1,
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
        "harness_version_observed": _package_version("mini-swe-agent"),
        "grader": "official SWE-bench Docker harness",
        "grader_version_requested": args.grader_version,
        "grader_version_observed": _package_version("swebench"),
        "upstream_base_url": args.upstream_base_url,
        "upstream_api_key_environment": args.upstream_api_key_env,
        "docker_executable": str(args.docker_executable) if args.docker_executable else None,
        "docker_platform": args.docker_platform,
        "environment_image": swebench_image(instance_id),
        "pythonpath": args.pythonpath,
        "instrument_observations": args.instrument_observations,
        "instrumentation_output_root": str(args.instrumentation_output_root),
        "agent_command_template": agent_command,
        "agent_behavior_sha256": agent_behavior_digest(agent_command),
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
        agent_wall_time = _run(
            agent_command,
            log=output / "agent.log",
            environment=environment,
            timeout_seconds=args.timeout_seconds,
        )
    finally:
        proxy.close()

    predictions = agent_output / "preds.json"
    if not predictions.is_file():
        raise RuntimeError(f"mini-swe-agent did not produce {predictions}")
    auxiliary = create_auxiliary_workspace_state_prediction(
        instrumentation_root=args.instrumentation_output_root,
        output=output,
        primary_predictions=predictions,
        instance_id=instance_id,
    )
    auxiliary["official_grading_requested"] = bool(
        args.grade_auxiliary_workspace_state and not args.skip_grading
    )
    _write_json(output / f"{AUXILIARY_WORKSPACE_STATE_LABEL}.json", auxiliary)
    metrics = summarize_trace(trace_path)
    metrics["agent_wall_time_seconds"] = agent_wall_time
    metrics["instance_id"] = instance_id
    metrics["auxiliary_workspace_state"] = auxiliary
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
