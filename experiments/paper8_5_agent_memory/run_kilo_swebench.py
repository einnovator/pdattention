"""Run one locked SWE-bench task through Kilo's native typed-tool loop.

This admission runner preserves Kilo's own prompt and tools.  It captures the
native JSON event stream and exports the source patch for the common official
grader; it does not infer task success from Kilo's process exit status.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping, Sequence


KILO_DATA_PATH = "/root/.local/share/kilo"


def _session_arguments(
    session_store: str | None,
    continue_session: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], Path | None]:
    """Build an isolated native Kilo session shared by fresh task sandboxes."""

    if continue_session and not session_store:
        raise ValueError("--continue-session requires --session-store")
    if not session_store:
        return (), ("--env", "KILO_DB=:memory:"), (), None
    store = Path(session_store).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    return (
        ("--mount", f"type=bind,source={store},target={KILO_DATA_PATH}"),
        ("--env", f"KILO_DB={KILO_DATA_PATH}/kilo.db"),
        (("--continue",) if continue_session else ()),
        store,
    )

from .reduce_kilo_events import reduce_events
from .run_pi_swebench import (
    _run,
    _sha256,
    _write_json,
    load_locked_task,
    observed_model_identity,
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


def _native_completion_observed(path: Path) -> bool:
    """Require Kilo's terminal non-tool step, not merely transport exit zero."""
    last: Mapping[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            last = value
    part = None if last is None else last.get("part")
    return bool(
        last
        and last.get("type") == "step_finish"
        and isinstance(part, Mapping)
        and part.get("reason") not in {None, "tool-calls"}
    )


def run(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark_card).resolve()
    _, instance_id, task_index = load_locked_task(benchmark, args.instance_id)
    trajectory = Path(args.reference_trajectory).resolve()
    model_identity = observed_model_identity(args)
    model_config = Path(args.model_config).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (
        session_docker_args,
        session_env_args,
        session_kilo_args,
        session_store,
    ) = _session_arguments(args.session_store, args.continue_session)

    prompt = task_prompt(trajectory, instance_id)
    (output / "task_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    repository = Path(__file__).resolve().parents[2]
    dockerfile = (
        repository
        / "experiments"
        / "paper8_5_agent_memory"
        / "docker"
        / "kilo-swebench.Dockerfile"
    )
    source_image = swebench_image(instance_id)
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    image = args.image or f"paper85-kilo-{slug}:{args.agent_version}"
    container = args.container or f"paper85-kilo-{slug}-{int(time.time())}"

    build_command = (
        args.docker,
        "build",
        "--platform", "linux/amd64",
        "--build-arg", f"BASE_IMAGE={source_image}",
        "--build-arg", f"KILO_VERSION={args.agent_version}",
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
        raise RuntimeError(f"Kilo task image build failed with {build.returncode}")

    command: tuple[str, ...] = (
        args.docker,
        "run",
        "--name", container,
        "--platform", "linux/amd64",
        *session_docker_args,
        "--volume", f"{model_config}:/root/.config/kilo/kilo.json:ro",
        "--env", "KILO_TELEMETRY_LEVEL=off",
        "--env", "KILO_DISABLE_DEFAULT_PLUGINS=true",
        "--env", "KILO_DISABLE_PROJECT_CONFIG=true",
        *session_env_args,
        "--entrypoint", "kilo",
        image,
        "--pure",
        "run",
        "--format", "json",
        "--auto",
        *session_kilo_args,
        "--model", args.model,
        "--dir", "/testbed",
        prompt,
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
    events = output / "kilo_events.jsonl"
    events.write_bytes(execution.stdout)
    (output / "kilo_stderr.log").write_bytes(execution.stderr)

    snapshot = f"{image.rsplit(':', 1)[0]}-snapshot:{int(time.time())}"
    committed = _run((args.docker, "commit", container, snapshot), check=False)
    if committed.returncode:
        raise RuntimeError("failed to snapshot the stopped Kilo task container")
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
                f"kilo-{args.agent_version}-{args.model.replace('/', '_')}-{arm_slug}"
            ),
            "model_patch": patch_text,
            "instance_id": instance_id,
        }
    })

    event_count, event_types = _event_inventory(events)
    native_completion_observed = _native_completion_observed(events)
    if event_count:
        reduce_events(events, output)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "arm": args.arm,
        "agent": "kilo",
        "agent_version": args.agent_version,
        "agent_protocol": "openai_tools",
        "instance_id": instance_id,
        "task_index": task_index,
        "benchmark_card": str(benchmark),
        "benchmark_card_sha256": _sha256(benchmark.read_bytes()),
        "reference_trajectory": str(trajectory),
        "reference_trajectory_sha256": _sha256(trajectory.read_bytes()),
        "model": args.model,
        "served_model": getattr(args, "served_model", None) or args.model,
        "model_revision": getattr(args, "model_revision", None),
        "observed_model_identity": model_identity,
        "model_config_sha256": _sha256(model_config.read_bytes()),
        "native_session": {
            "data_path": KILO_DATA_PATH,
            "host_store": None if session_store is None else str(session_store),
            "continued": bool(args.continue_session),
            "continuity_mode": (
                "dedicated-persistent-database"
                if session_store is not None else "ephemeral-memory-database"
            ),
        },
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
        "event_count": event_count,
        "event_types": event_types,
        "native_completion_observed": native_completion_observed,
        "workspace_status_sha256": _sha256(status.stdout),
        "patch_sha256": _sha256(patch.stdout),
        "patch_bytes": len(patch.stdout),
        "official_resolution": None,
        "notes": [
            "This admission runner preserves Kilo's native system prompt and typed tools.",
            "External telemetry, default plugins, project config, and persistent session storage are disabled for hermetic execution.",
            "Kilo transport completion is not treated as semantic tool success.",
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
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="FULL")
    parser.add_argument("--model", default="openai-compatible/qwen3-coder:30b")
    parser.add_argument("--served-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--ollama-tags-url")
    parser.add_argument("--agent-version", default="7.7.5")
    parser.add_argument(
        "--session-store",
        help=(
            "Dedicated host directory mounted as Kilo's native data store. "
            "Reuse it across fresh task containers for a persistent session."
        ),
    )
    parser.add_argument(
        "--continue-session",
        action="store_true",
        help="Continue the sole native session in --session-store.",
    )
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--container")
    parser.add_argument("--build-timeout-seconds", type=int, default=1200)
    parser.add_argument("--agent-timeout-seconds", type=int, default=3600)
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--keep-snapshot", action="store_true")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
