"""Run one locked SWE-bench task with OpenCode's native typed-tool loop."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
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
        "--agent", args.agent, "--model", args.model, "--dir", "/testbed", prompt,
    )
    started = datetime.now(timezone.utc)
    timed_out = False
    try:
        execution = _run(command, timeout=args.agent_timeout_seconds, check=False)
    except subprocess.TimeoutExpired as error:
        timed_out = True
        _run((args.docker, "stop", "--time", "5", container), check=False)
        execution = subprocess.CompletedProcess(
            command, 124, stdout=error.stdout or b"", stderr=error.stderr or b""
        )
    finished = datetime.now(timezone.utc)
    events = output / "opencode_events.jsonl"
    events.write_bytes(execution.stdout)
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
        "source_image": source_image,
        "derived_image": image,
        "capture_snapshot_image": snapshot,
        "container": container,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "exit_code": execution.returncode,
        "timed_out": timed_out,
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
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--keep-snapshot", action="store_true")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
