"""Run one locked SWE-bench task with condenser-free OpenHands SDK tools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence

from .run_pi_swebench import (
    _run,
    _sha256,
    _write_json,
    load_locked_task,
    swebench_image,
    task_prompt,
)


def _event_inventory(path: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        rows += 1
        name = str(value.get("type") or "unknown")
        counts[name] = counts.get(name, 0) + 1
    return rows, counts


def _event_summary(path: Path) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    observation_counts: dict[str, int] = {}
    response_ids: set[str] = set()
    action_count = observation_count = observation_errors = 0
    receipt_count = effect_complete_receipts = receipt_delivery_failures = 0
    run_summary: Mapping[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping):
            continue
        row_type = str(row.get("type") or "unknown")
        event = row.get("event")
        event = event if isinstance(event, Mapping) else {}
        if row_type == "ActionEvent":
            action_count += 1
            action = event.get("action")
            action = action if isinstance(action, Mapping) else {}
            kind = str(action.get("kind") or "unknown")
            action_counts[kind] = action_counts.get(kind, 0) + 1
            response_id = event.get("llm_response_id")
            if response_id:
                response_ids.add(str(response_id))
        elif row_type == "ObservationEvent":
            observation_count += 1
            observation = event.get("observation")
            observation = observation if isinstance(observation, Mapping) else {}
            kind = str(observation.get("kind") or "unknown")
            observation_counts[kind] = observation_counts.get(kind, 0) + 1
            exit_code = observation.get("exit_code")
            if bool(observation.get("is_error")) or (
                isinstance(exit_code, int) and exit_code != 0
            ):
                observation_errors += 1
        elif row_type == "paper85_run_summary":
            run_summary = row
        elif row_type == "paper85_execution_receipt":
            receipt = row.get("receipt")
            receipt = receipt if isinstance(receipt, Mapping) else {}
            receipt_count += 1
            effect_complete_receipts += int(bool(receipt.get("effect_trace_complete")))
            receipt_delivery_failures += int(
                receipt.get("sidechannel_delivery") != "accepted"
            )
    return {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "agent": "openhands-sdk",
        "source": str(path),
        "source_sha256": _sha256(path.read_bytes()),
        "action_count": action_count,
        "observation_count": observation_count,
        "distinct_llm_response_count": len(response_ids),
        "action_counts": dict(sorted(action_counts.items())),
        "observation_counts": dict(sorted(observation_counts.items())),
        "semantic_failure_observations": observation_errors,
        "execution_receipt_count": receipt_count,
        "effect_trace_complete_receipts": effect_complete_receipts,
        "execution_receipt_delivery_failures": receipt_delivery_failures,
        "run_summary": dict(run_summary or {}),
        "token_note": (
            "Provider request and usage totals are authoritative in the paired "
            "logical-proxy trace, not inferred from SDK events."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark_card).resolve()
    _, instance_id, task_index = load_locked_task(benchmark, args.instance_id)
    trajectory = Path(args.reference_trajectory).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt = task_prompt(trajectory, instance_id)
    prompt_path = output / "task_prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    repository = Path(__file__).resolve().parents[2]
    dockerfile = (
        repository
        / "experiments"
        / "paper8_5_agent_memory"
        / "docker"
        / "openhands-swebench.Dockerfile"
    )
    source_image = swebench_image(instance_id)
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    image = args.image or f"paper85-openhands-{slug}:{args.agent_version}"
    container = args.container or f"paper85-openhands-{slug}-{int(time.time())}"
    build_command = (
        args.docker,
        "build",
        "--platform", "linux/amd64",
        "--build-arg", f"BASE_IMAGE={source_image}",
        "--file", str(dockerfile),
        "--tag", image,
        str(repository),
    )
    build = (
        subprocess.CompletedProcess(build_command, 0, stdout=b"skipped\n", stderr=b"")
        if args.skip_build
        else _run(build_command, timeout=args.build_timeout_seconds, check=False)
    )
    (output / "docker_build.stdout.log").write_bytes(build.stdout)
    (output / "docker_build.stderr.log").write_bytes(build.stderr)
    if build.returncode:
        raise RuntimeError(f"OpenHands task image build failed with {build.returncode}")

    command: tuple[str, ...] = (
        args.docker,
        "run",
        "--name", container,
        "--platform", "linux/amd64",
        "--volume", f"{prompt_path}:/paper85/task_prompt.txt:ro",
        "--env", "OPENHANDS_SUPPRESS_BANNER=1",
        "--env", "OTEL_SDK_DISABLED=true",
        "--env", "DO_NOT_TRACK=1",
        "--env", "ANONYMIZED_TELEMETRY=false",
        "--entrypoint", "/opt/openhands/bin/python",
        image,
        "/opt/paper85/openhands_swebench_entry.py",
        "--prompt-file", "/paper85/task_prompt.txt",
        "--session-id", instance_id,
        "--base-url", args.base_url,
        "--model", args.model,
        "--max-iterations", str(args.max_iterations),
        "--max-output-tokens", str(args.max_completion_tokens),
    )
    started = datetime.now(timezone.utc)
    timed_out = False
    try:
        execution = _run(command, timeout=args.agent_timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        _run((args.docker, "stop", "--time", "5", container), check=False)
        execution = subprocess.CompletedProcess(
            command,
            124,
            stdout=error.stdout or b"",
            stderr=error.stderr or b"",
        )
    finished = datetime.now(timezone.utc)
    events = output / "openhands_events.jsonl"
    events.write_bytes(execution.stdout)
    (output / "openhands_stderr.log").write_bytes(execution.stderr)

    snapshot = f"{image.rsplit(':', 1)[0]}-snapshot:{int(time.time())}"
    committed = _run((args.docker, "commit", container, snapshot), check=False)
    if committed.returncode:
        raise RuntimeError("failed to snapshot the stopped OpenHands task container")
    status = _run((
        args.docker, "run", "--rm", "--platform", "linux/amd64",
        "--entrypoint", "git", snapshot,
        "-C", "/testbed", "status", "--short",
    ), check=False)
    (output / "workspace_status.txt").write_bytes(status.stdout)
    patch = _run((
        args.docker, "run", "--rm", "--platform", "linux/amd64",
        "--entrypoint", "git", snapshot,
        "-C", "/testbed", "diff", "--binary", "--",
    ), check=False)
    (output / "model.patch").write_bytes(patch.stdout)
    patch_text = patch.stdout.decode("utf-8", errors="replace")
    arm_slug = re.sub(r"[^a-z0-9]+", "-", args.arm.lower()).strip("-")
    _write_json(output / "preds.json", {
        instance_id: {
            "model_name_or_path": (
                f"openhands-{args.agent_version}-{args.model.replace('/', '_')}-{arm_slug}"
            ),
            "model_patch": patch_text,
            "instance_id": instance_id,
        }
    })

    event_count, event_types = _event_inventory(events)
    _write_json(output / "event_summary.json", _event_summary(events))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "arm": args.arm,
        "agent": "openhands-sdk",
        "agent_version": args.agent_version,
        "agent_protocol": "openai_tools",
        "native_context_management": "disabled_condenser_none",
        "instance_id": instance_id,
        "task_index": task_index,
        "benchmark_card": str(benchmark),
        "benchmark_card_sha256": _sha256(benchmark.read_bytes()),
        "reference_trajectory": str(trajectory),
        "reference_trajectory_sha256": _sha256(trajectory.read_bytes()),
        "model": args.model,
        "base_url": args.base_url,
        "source_image": source_image,
        "derived_image": image,
        "image_build_skipped": bool(args.skip_build),
        "capture_snapshot_image": snapshot,
        "container": container,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "exit_code": execution.returncode,
        "timed_out": timed_out,
        "max_iterations": args.max_iterations,
        "event_count": event_count,
        "event_types": event_types,
        "workspace_status_sha256": _sha256(status.stdout),
        "patch_sha256": _sha256(patch.stdout),
        "patch_bytes": len(patch.stdout),
        "official_resolution": None,
        "notes": [
            "OpenHands uses its native terminal, file-editor, and task-tracker tools.",
            "The OpenHands condenser is explicitly disabled for the FULL control.",
            "External telemetry is disabled for hermetic execution.",
            "Absent optional cache-creation usage counters are zero-defaulted in telemetry only.",
            "Official resolution remains unset until the common SWE-bench grader consumes model.patch.",
        ],
    }
    _write_json(output / "run_manifest.json", manifest)
    if not args.keep_snapshot:
        _run((args.docker, "image", "rm", snapshot), check=False)
    if not args.keep_container:
        _run((args.docker, "rm", "--force", container), check=False)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--reference-trajectory", required=True)
    parser.add_argument(
        "--benchmark-card",
        default=str(
            Path(__file__).with_name("benchmarks")
            / "easy14_baseline_success14.json"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="FULL")
    parser.add_argument("--base-url", default="http://host.docker.internal:18185/v1")
    parser.add_argument("--model", default="openai/qwen3-coder:30b")
    parser.add_argument("--agent-version", default="1.49.2")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--container")
    parser.add_argument("--build-timeout-seconds", type=int, default=2400)
    parser.add_argument("--agent-timeout-seconds", type=int, default=3600)
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--keep-snapshot", action="store_true")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
