"""Run one predeclared SWE-bench task with Pi's native tool protocol.

This is an admission runner, not a replacement for the official grader.  It
keeps Pi's own system prompt and tools, captures the native JSON event stream,
and exports a source patch that can be graded by the same SWE-bench harness as
the mini-swe-agent controls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

# Direct script execution would otherwise let this package's ``selectors.py``
# shadow Python's standard-library module imported by ``subprocess``.
_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == _SCRIPT_DIRECTORY:
    sys.path.pop(0)

import subprocess
import time
from typing import Any, Mapping, Sequence


PI_SESSION_PATH = "/root/.pi/agent/sessions"


def _session_arguments(
    session_store: str | None,
    continue_session: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], Path | None]:
    """Return isolated native-session mounts and CLI arguments.

    A dedicated store contains exactly one campaign session, so ``--continue``
    cannot silently select history from another experiment.  The first episode
    creates that session; later fresh task containers mount the same store and
    continue it while retaining a fresh ``/testbed`` workspace.
    """

    if continue_session and not session_store:
        raise ValueError("--continue-session requires --session-store")
    if not session_store:
        return (), ("--no-session",), None
    store = Path(session_store).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    docker_args = (
        "--mount", f"type=bind,source={store},target={PI_SESSION_PATH}",
    )
    pi_args = ("--session-dir", PI_SESSION_PATH)
    if continue_session:
        pi_args += ("--continue",)
    return docker_args, pi_args, store

from .model_identity import fetch_ollama_model_identity

_PR_DESCRIPTION = re.compile(
    r"<pr_description>\s*(.*?)\s*</pr_description>", re.DOTALL | re.IGNORECASE
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def observed_model_identity(args: argparse.Namespace) -> dict[str, Any] | None:
    """Bind a typed-agent run to an observed endpoint digest when requested."""

    url = getattr(args, "ollama_tags_url", None)
    revision = getattr(args, "model_revision", None)
    if bool(url) != bool(revision):
        raise ValueError(
            "--ollama-tags-url and --model-revision must be supplied together"
        )
    if not url:
        return None
    return fetch_ollama_model_identity(
        url,
        expected_model=getattr(args, "served_model", None) or args.model,
        expected_revision=revision,
        curl_executable=getattr(args, "curl_executable", None),
    )


def load_locked_task(path: Path, instance_id: str) -> tuple[dict[str, Any], str, int]:
    card = json.loads(path.read_text(encoding="utf-8"))
    ids = card.get("instance_ids")
    if not isinstance(ids, list) or not all(isinstance(row, str) for row in ids):
        raise ValueError("benchmark card must contain string instance_ids")
    if len(ids) != len(set(ids)) or len(ids) != int(card.get("expected_count", len(ids))):
        raise ValueError("benchmark card instance count or uniqueness is invalid")
    digest = _sha256(("\n".join(ids) + "\n").encode("utf-8"))
    if digest != str(card.get("canonical_ids_sha256", "")).lower():
        raise ValueError("benchmark card canonical_ids_sha256 does not match")
    if instance_id not in ids:
        raise ValueError(f"instance {instance_id!r} is not in the locked card")
    return card, instance_id, ids.index(instance_id) + 1


def swebench_image(instance_id: str) -> str:
    docker_instance_id = instance_id.replace("__", "_1776_").lower()
    return "docker.io/swebench/sweb.eval.x86_64." + docker_instance_id + ":latest"


def task_prompt(trajectory_path: Path, instance_id: str) -> str:
    artifact = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory = artifact.get("trajectory", artifact)
    if not isinstance(trajectory, Mapping):
        raise ValueError("reference artifact lacks a trajectory object")
    if str(trajectory.get("instance_id")) != instance_id:
        raise ValueError("reference trajectory identity does not match the task")
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError("reference trajectory lacks messages")
    first_user = next(
        (
            str(row.get("content") or "")
            for row in messages
            if isinstance(row, Mapping) and row.get("role") == "user"
        ),
        "",
    )
    match = _PR_DESCRIPTION.search(first_user)
    problem = match.group(1).strip() if match else first_user.strip()
    if not problem:
        raise ValueError("reference trajectory has no task description")
    return (
        "Solve the following software issue in the repository at /testbed. "
        "Inspect the code, make the smallest general source-code fix, and run "
        "focused verification. Do not modify tests, dependency files, build "
        "configuration, or benchmark metadata. Do not commit. Stop when the "
        "working tree contains the verified source fix.\n\n"
        f"Issue ({instance_id}):\n{problem}"
    )


def _run(
    command: Sequence[str],
    *,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command), capture_output=True, timeout=timeout, check=check
    )


def run(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark_card).resolve()
    _, instance_id, task_index = load_locked_task(benchmark, args.instance_id)
    trajectory = Path(args.reference_trajectory).resolve()
    model_identity = observed_model_identity(args)
    model_config = Path(args.model_config).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    session_docker_args, session_pi_args, session_store = _session_arguments(
        args.session_store, args.continue_session
    )
    prompt = task_prompt(trajectory, instance_id)
    (output / "task_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    repository = Path(__file__).resolve().parents[2]
    dockerfile = repository / "experiments" / "paper8_5_agent_memory" / "docker" / "pi-swebench.Dockerfile"
    source_image = swebench_image(instance_id)
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    image = args.image or f"paper85-pi-{slug}:0.75.3"
    container = args.container or f"paper85-pi-{slug}-{int(time.time())}"

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
        else _run(
            build_command,
            timeout=args.build_timeout_seconds,
            check=False,
        )
    )
    (output / "docker_build.stdout.log").write_bytes(build.stdout)
    (output / "docker_build.stderr.log").write_bytes(build.stderr)
    if build.returncode:
        raise RuntimeError(f"Pi task image build failed with {build.returncode}")

    command = (
        args.docker,
        "run",
        "--name", container,
        "--platform", "linux/amd64",
        *session_docker_args,
        "--volume", f"{model_config}:/root/.pi/agent/models.json:ro",
        "--env", "PI_OFFLINE=1",
        image,
        "pi",
        "--provider", args.provider,
        "--model", args.model,
        "--mode", "json",
        "--print",
        *session_pi_args,
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--offline",
        prompt,
    )
    started = datetime.now(timezone.utc)
    timed_out = False
    try:
        execution = _run(
            command, timeout=args.agent_timeout_seconds, check=False
        )
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
    (output / "pi_events.jsonl").write_bytes(execution.stdout)
    (output / "pi_stderr.log").write_bytes(execution.stderr)

    # ``docker run`` leaves a stopped container.  Snapshot it before
    # inspection; ``docker exec`` cannot inspect an exited container, while
    # restarting it would execute the agent a second time and contaminate the
    # trajectory.
    snapshot = f"{image.rsplit(':', 1)[0]}-snapshot:{int(time.time())}"
    committed = _run((
        args.docker, "commit", container, snapshot,
    ), check=False)
    if committed.returncode:
        raise RuntimeError("failed to snapshot the stopped Pi task container")
    status = _run((
        args.docker, "run", "--rm", "--platform", "linux/amd64",
        "--entrypoint", "git", snapshot,
        "-C", "/testbed", "status", "--short",
    ), check=False)
    (output / "workspace_status.txt").write_bytes(status.stdout)
    # The primary patch contains tracked repository changes only.  Scratch
    # reproducers and editor backup files are inventoried by status but never
    # silently submitted.  A task whose intended fix requires a new source file
    # must be reviewed and explicitly promoted before common grading.
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
                f"pi-0.75.3-{args.model.replace('/', '_')}-{arm_slug}"
            ),
            "model_patch": patch_text,
            "instance_id": instance_id,
        }
    })

    event_rows = []
    for line in execution.stdout.decode("utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            event_rows.append(dict(value))
    event_types: dict[str, int] = {}
    for row in event_rows:
        name = str(row.get("type") or "unknown")
        event_types[name] = event_types.get(name, 0) + 1

    manifest = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "arm": args.arm,
        "agent": "pi",
        "agent_version": "0.75.3",
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
        "provider": args.provider,
        "native_session": {
            "data_path": PI_SESSION_PATH,
            "host_store": None if session_store is None else str(session_store),
            "continued": bool(args.continue_session),
            "continuity_mode": (
                "dedicated_single-session-store"
                if session_store is not None else "ephemeral"
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
        "event_count": len(event_rows),
        "event_types": event_types,
        "workspace_status_sha256": _sha256(status.stdout),
        "patch_sha256": _sha256(patch.stdout),
        "patch_bytes": len(patch.stdout),
        "official_resolution": None,
        "notes": [
            "This admission runner preserves Pi's native system prompt and tools.",
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
            Path(__file__).with_name("benchmarks") / "easy14_baseline_success14.json"
        ),
    )
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="FULL")
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--served-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--ollama-tags-url")
    parser.add_argument("--provider", default="paper85-ollama")
    parser.add_argument(
        "--session-store",
        help=(
            "Dedicated host directory mounted as Pi's native session store. "
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
    output = run(build_parser().parse_args())
    print(output)


if __name__ == "__main__":
    main()
