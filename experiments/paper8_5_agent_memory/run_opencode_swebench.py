"""Run one locked SWE-bench task with OpenCode's native typed-tool loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Mapping

from .model_identity import fetch_ollama_model_identity
from .reduce_opencode_events import reduce_events
from .run_pi_swebench import (
    _run,
    _sha256,
    _write_json,
    load_locked_task,
    swebench_image,
    task_prompt,
)


PINNED_OPENCODE_VERSION = "1.18.31"
OPENCODE_DATA_PATH = "/root/.local/share/opencode"


def _has_native_stop_event(payload: bytes) -> bool:
    """Return whether an OpenCode JSONL trace contains semantic completion."""

    for line in payload.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "step_finish":
            continue
        part = event.get("part")
        if isinstance(part, Mapping) and part.get("reason") == "stop":
            return True
    return False


def _native_session_id(payload: bytes) -> str | None:
    """Return the unique native OpenCode session ID in a JSONL trace.

    Persistent-session evidence must not infer continuity from task order or a
    shared output directory.  OpenCode emits ``sessionID`` on every root event;
    require those events to agree before a later task may resume the session.
    """

    values: set[str] = set()
    for line in payload.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        value = event.get("sessionID")
        if isinstance(value, str) and value:
            values.add(value)
    if len(values) > 1:
        raise ValueError(
            "OpenCode event trace contains multiple root session IDs: "
            + ", ".join(sorted(values))
        )
    return next(iter(values), None)


def _session_arguments(
    session_store: str | None,
    session_id: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...], Path | None]:
    """Build Docker and OpenCode arguments for native session continuation."""

    if session_id and not session_store:
        raise ValueError("--session-id requires --session-store")
    if not session_store:
        return (), (), None
    store = Path(session_store).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    docker_args = (
        "--mount",
        f"type=bind,source={store},target={OPENCODE_DATA_PATH}",
    )
    opencode_args = () if not session_id else ("--session", session_id)
    return docker_args, opencode_args, store


def _native_stop_watchdog(
    docker: str,
    container: str,
    *,
    done: threading.Event,
    grace_seconds: float,
    state: dict[str, bool],
) -> None:
    """Stop a CLI that remains alive after its native terminal event."""

    while not done.wait(1.0):
        logs = _run((docker, "logs", "--tail", "32", container), check=False)
        if not _has_native_stop_event(logs.stdout):
            continue
        state["native_stop_observed"] = True
        if done.wait(grace_seconds):
            return
        running = _run((
            docker,
            "inspect",
            "--format",
            "{{.State.Running}}",
            container,
        ), check=False)
        if running.returncode == 0 and running.stdout.strip() == b"true":
            _run((docker, "stop", "--timeout", "10", container), check=False)
            state["forced_stop_after_native_stop"] = True
        return


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


def _load_tool_semantics(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("tool_semantics_by_name", payload)
    if not isinstance(values, Mapping):
        raise ValueError("tool semantics must be an object keyed by tool name")
    return {
        str(name): dict(value)
        for name, value in values.items()
        if isinstance(value, Mapping)
    }


def _validate_model_config(
    path: Path,
    expected_model: str,
    *,
    agent_name: str = "build",
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != expected_model:
        raise ValueError(
            f"OpenCode config model {payload.get('model')!r} does not match "
            f"--model {expected_model!r}"
        )
    if payload.get("small_model") != expected_model:
        raise ValueError(
            "OpenCode small_model must equal the controlled primary model"
        )
    agent = payload.get("agent", {}).get(agent_name, {})
    expected_controls = {"temperature": 0.0, "top_p": 1.0, "seed": 0}
    observed_controls = {
        name: agent.get(name) for name in expected_controls
    }
    if observed_controls != expected_controls:
        raise ValueError(
            f"OpenCode agent {agent_name!r} must freeze generation controls "
            f"as {expected_controls}, observed {observed_controls}"
        )
    if agent.get("model") != expected_model:
        raise ValueError(
            f"OpenCode agent {agent_name!r} model must equal {expected_model!r}"
        )
    return payload


def run(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark_card).resolve()
    _, instance_id, task_index = load_locked_task(benchmark, args.instance_id)
    trajectory = Path(args.reference_trajectory).resolve()
    model_config = Path(args.model_config).resolve()
    tool_semantics_path = Path(args.tool_semantics).resolve()
    validated_config = _validate_model_config(
        model_config, args.model, agent_name=args.agent
    )
    generation = {
        name: validated_config["agent"][args.agent][name]
        for name in ("temperature", "top_p", "seed")
    }
    generation["max_completion_tokens"] = 1024
    tool_semantics = _load_tool_semantics(tool_semantics_path)
    observed_model_identity = (
        fetch_ollama_model_identity(
            args.ollama_tags_url,
            expected_model=args.served_model,
            expected_revision=args.model_revision,
        )
        if args.ollama_tags_url else None
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    session_docker_args, session_opencode_args, session_store = (
        _session_arguments(args.session_store, args.session_id)
    )

    prompt = task_prompt(trajectory, instance_id)
    (output / "task_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    repository = Path(__file__).resolve().parents[2]
    dockerfile = repository / "experiments" / "paper8_5_agent_memory" / "docker" / "opencode-swebench.Dockerfile"
    source_image = swebench_image(instance_id)
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    image = args.image or f"paper85-opencode-{slug}:{args.agent_version}"
    container = args.container or f"paper85-opencode-{slug}-{int(time.time())}"

    build_command = (
        args.docker, "build", "--platform", "linux/amd64",
        "--build-arg", f"BASE_IMAGE={source_image}",
        "--build-arg", f"OPENCODE_VERSION={args.agent_version}",
        "--file", str(dockerfile), "--tag", image, str(repository),
    )
    build = (
        subprocess.CompletedProcess(build_command, 0, stdout=b"skipped\n", stderr=b"")
        if args.skip_build
        else _run(build_command, timeout=args.build_timeout_seconds, check=False)
    )
    (output / "docker_build.stdout.log").write_bytes(build.stdout)
    (output / "docker_build.stderr.log").write_bytes(build.stderr)
    if build.returncode:
        raise RuntimeError(f"OpenCode task image build failed with {build.returncode}")

    command: tuple[str, ...] = (
        args.docker, "run", "--name", container, "--platform", "linux/amd64",
        *session_docker_args,
        "--volume", f"{model_config}:/root/.config/opencode/opencode.json:ro",
        "--env", "OPENCODE_CONFIG=/root/.config/opencode/opencode.json",
        "--env", "OPENCODE_DISABLE_AUTOUPDATE=true",
        "--env", "OPENCODE_DISABLE_DEFAULT_PLUGINS=true",
        "--env", "OPENCODE_DISABLE_AUTOCOMPACT=true",
        "--env", "OPENCODE_DISABLE_CLAUDE_CODE=true",
        "--env", "OPENCODE_DISABLE_LSP_DOWNLOAD=true",
        "--env", "DO_NOT_TRACK=1",
        "--entrypoint", "opencode", image,
        "--pure", "run", "--format", "json", "--auto",
        *session_opencode_args,
        "--agent", args.agent, "--model", args.model, "--dir", "/testbed", prompt,
    )
    started = datetime.now(timezone.utc)
    timed_out = False
    watchdog_done = threading.Event()
    watchdog_state = {
        "native_stop_observed": False,
        "forced_stop_after_native_stop": False,
    }
    watchdog = threading.Thread(
        target=_native_stop_watchdog,
        args=(args.docker, container),
        kwargs={
            "done": watchdog_done,
            "grace_seconds": args.native_stop_grace_seconds,
            "state": watchdog_state,
        },
        daemon=True,
    )
    watchdog.start()
    try:
        execution = _run(command, timeout=args.agent_timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        _run((args.docker, "stop", "--time", "5", container), check=False)
        execution = subprocess.CompletedProcess(
            command, 124, stdout=error.stdout or b"", stderr=error.stderr or b""
        )
    finally:
        watchdog_done.set()
        watchdog.join(timeout=args.native_stop_grace_seconds + 2)
    finished = datetime.now(timezone.utc)
    events = output / "opencode_events.jsonl"
    events.write_bytes(execution.stdout)
    observed_session_id = _native_session_id(execution.stdout)
    if args.session_store and observed_session_id is None:
        raise RuntimeError(
            "persistent OpenCode run emitted no native session identity"
        )
    if args.session_id and observed_session_id != args.session_id:
        raise RuntimeError(
            "OpenCode session continuity failed: requested "
            f"{args.session_id!r}, observed {observed_session_id!r}"
        )
    (output / "opencode_stderr.log").write_bytes(execution.stderr)

    snapshot = f"{image.rsplit(':', 1)[0]}-snapshot:{int(time.time())}"
    committed = _run((args.docker, "commit", container, snapshot), check=False)
    if committed.returncode:
        raise RuntimeError("failed to snapshot the stopped OpenCode task container")
    status = _run((
        args.docker, "run", "--rm", "--platform", "linux/amd64",
        "--entrypoint", "git", snapshot, "-C", "/testbed", "status", "--short",
    ), check=False)
    (output / "workspace_status.txt").write_bytes(status.stdout)
    patch = _run((
        args.docker, "run", "--rm", "--platform", "linux/amd64",
        "--entrypoint", "git", snapshot, "-C", "/testbed", "diff", "--binary", "--",
    ), check=False)
    (output / "model.patch").write_bytes(patch.stdout)
    patch_text = patch.stdout.decode("utf-8", errors="replace")
    arm_slug = re.sub(r"[^a-z0-9]+", "-", args.arm.lower()).strip("-")
    _write_json(output / "preds.json", {
        instance_id: {
            "model_name_or_path": f"opencode-{args.agent_version}-{args.model.replace('/', '_')}-{arm_slug}",
            "model_patch": patch_text,
            "instance_id": instance_id,
        }
    })

    event_count, event_types = _event_inventory(events)
    summary = reduce_events(events, output, tool_semantics) if event_count else {}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "arm": args.arm,
        "agent": "opencode",
        "agent_mode": args.agent,
        "agent_version": args.agent_version,
        "agent_protocol": "openai_tools",
        "instance_id": instance_id,
        "task_index": task_index,
        "benchmark_card": str(benchmark),
        "benchmark_card_sha256": _sha256(benchmark.read_bytes()),
        "reference_trajectory": str(trajectory),
        "reference_trajectory_sha256": _sha256(trajectory.read_bytes()),
        "model": args.model,
        "served_model": args.served_model,
        "model_revision": args.model_revision,
        "observed_model_identity": observed_model_identity,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "generation": generation,
        "model_config_sha256": _sha256(model_config.read_bytes()),
        "tool_semantics_sha256": _sha256(tool_semantics_path.read_bytes()),
        "native_context_management": "disabled_OPENCODE_DISABLE_AUTOCOMPACT",
        "native_session": {
            "data_path": OPENCODE_DATA_PATH,
            "host_store": None if session_store is None else str(session_store),
            "requested_session_id": args.session_id,
            "observed_session_id": observed_session_id,
            "continuity_validated": bool(
                observed_session_id
                and (args.session_id is None or observed_session_id == args.session_id)
            ),
        },
        "source_image": source_image,
        "derived_image": image,
        "capture_snapshot_image": snapshot,
        "container": container,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "exit_code": execution.returncode,
        "timed_out": timed_out,
        "native_stop_observed": watchdog_state["native_stop_observed"],
        "forced_stop_after_native_stop": watchdog_state[
            "forced_stop_after_native_stop"
        ],
        "native_stop_grace_seconds": args.native_stop_grace_seconds,
        "event_count": event_count,
        "event_types": event_types,
        "root_event_trace_complete": summary.get("root_event_trace_complete"),
        "workspace_status_sha256": _sha256(status.stdout),
        "patch_sha256": _sha256(patch.stdout),
        "patch_bytes": len(patch.stdout),
        "official_resolution": None,
        "notes": [
            "OpenCode uses its native prompt and typed tools; the PRA policy consumes only normalized causal records.",
            "Native auto-compaction, external plugins, Claude-Code imports, and LSP downloads are disabled.",
            "A task/subagent call makes the root stdout trace incomplete until child-session export is joined.",
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
    parser.add_argument("--benchmark-card", default=str(Path(__file__).with_name("benchmarks") / "easy14_baseline_success14.json"))
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tool-semantics", default=str(Path(__file__).with_name("configs") / "opencode_tool_semantics_v1.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="FULL")
    parser.add_argument("--model", default="pra/qwen3-coder:30b")
    parser.add_argument("--served-model", default="qwen3-coder:30b")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--ollama-tags-url")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--agent", default="build")
    parser.add_argument("--agent-version", default=PINNED_OPENCODE_VERSION)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--container")
    parser.add_argument("--build-timeout-seconds", type=int, default=1800)
    parser.add_argument("--agent-timeout-seconds", type=int, default=3600)
    parser.add_argument("--native-stop-grace-seconds", type=float, default=30.0)
    parser.add_argument(
        "--session-store",
        help=(
            "Dedicated host directory mounted at OpenCode's native data path. "
            "Reuse it across fresh task containers for a persistent session."
        ),
    )
    parser.add_argument(
        "--session-id",
        help="Native session ID emitted by the preceding task in this session.",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--keep-snapshot", action="store_true")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
