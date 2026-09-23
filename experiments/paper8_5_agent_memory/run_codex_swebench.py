"""Run one locked SWE-bench task with Codex CLI and a native Responses API."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping

from .responses_audit_proxy import ResponsesAuditConfig, ResponsesAuditProxy
from .run_pi_swebench import (
    _run,
    _sha256,
    _write_json,
    load_locked_task,
    swebench_image,
    task_prompt,
)


def _event_summary(path: Path) -> dict[str, Any]:
    types: dict[str, int] = {}
    item_types: dict[str, int] = {}
    usage: Mapping[str, Any] = {}
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, Mapping):
            continue
        rows += 1
        row_type = str(row.get("type") or "unknown")
        types[row_type] = types.get(row_type, 0) + 1
        item = row.get("item")
        if isinstance(item, Mapping):
            item_type = str(item.get("type") or "unknown")
            item_types[item_type] = item_types.get(item_type, 0) + 1
        if row_type == "turn.completed" and isinstance(row.get("usage"), Mapping):
            usage = row["usage"]
    return {
        "schema_version": 1,
        "agent": "codex-cli",
        "event_rows": rows,
        "event_types": dict(sorted(types.items())),
        "item_types": dict(sorted(item_types.items())),
        "final_turn_usage": dict(usage),
    }


def _trace_summary(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if path.exists() else []
    return {
        "request_count": len(rows),
        "all_success": bool(rows) and all(int(row.get("status", 0)) < 400 for row in rows),
        "uses_previous_response_id": any(
            bool(row.get("uses_previous_response_id")) for row in rows
        ),
        "input_item_counts": [row.get("input_item_count") for row in rows],
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
    dockerfile = repository / "experiments" / "paper8_5_agent_memory" / "docker" / "codex-swebench.Dockerfile"
    source_image = swebench_image(instance_id)
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    image = args.image or f"paper85-codex-{slug}:{args.agent_version}"
    container = args.container or f"paper85-codex-{slug}-{int(time.time())}"
    build_command = (
        args.docker, "build", "--platform", "linux/amd64",
        "--build-arg", f"BASE_IMAGE={source_image}",
        "--build-arg", f"CODEX_VERSION={args.agent_version}",
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
        raise RuntimeError(f"Codex task image build failed with {build.returncode}")

    trace = output / "responses_proxy_trace.jsonl"
    proxy: ResponsesAuditProxy | None = None
    provider_args: tuple[str, ...]
    provider_environment: tuple[str, ...]
    if args.provider_mode == "responses_audit":
        proxy = ResponsesAuditProxy(ResponsesAuditConfig(
            upstream_base_url=args.upstream_base_url,
            expected_model=args.model,
            trace_path=trace,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_output_tokens=args.max_completion_tokens,
            timeout_seconds=args.upstream_timeout_seconds,
        ))
        proxy_url = proxy.start()
        provider_environment = ()
        provider_args = (
            "--config", "model_provider=paper85",
            "--config", "model_providers.paper85.name=Paper85",
            "--config", f"model_providers.paper85.base_url={proxy_url}",
            "--config", "model_providers.paper85.env_key=OPENAI_API_KEY",
            "--config", "model_providers.paper85.wire_api=responses",
        )
    else:
        # Let Codex own the open-model adaptation. This avoids attributing a
        # Responses-to-Chat translation layer to the PRA memory policy.
        provider_environment = (
            "--env", f"OLLAMA_HOST={args.upstream_base_url.rstrip('/')}",
        )
        provider_args = ("--oss", "--local-provider", "ollama")
    command = (
        args.docker, "run", "--name", container, "--platform", "linux/amd64",
        "--env", "OPENAI_API_KEY=paper85-local",
        "--env", "OTEL_SDK_DISABLED=true",
        "--env", "DO_NOT_TRACK=1",
        *provider_environment,
        "--entrypoint", "codex", image,
        "exec", "--json", "--ephemeral", "--ignore-user-config", "--ignore-rules",
        "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
        "--model", args.model,
        *provider_args,
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
            command, 124, stdout=error.stdout or b"", stderr=error.stderr or b""
        )
    finally:
        if proxy is not None:
            proxy.close()
    finished = datetime.now(timezone.utc)
    events = output / "codex_events.jsonl"
    events.write_bytes(execution.stdout)
    (output / "codex_stderr.log").write_bytes(execution.stderr)

    snapshot = f"{image.rsplit(':', 1)[0]}-snapshot:{int(time.time())}"
    committed = _run((args.docker, "commit", container, snapshot), check=False)
    if committed.returncode:
        raise RuntimeError("failed to snapshot the stopped Codex task container")
    status = _run((args.docker, "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "git", snapshot, "-C", "/testbed", "status", "--short"), check=False)
    (output / "workspace_status.txt").write_bytes(status.stdout)
    patch = _run((args.docker, "run", "--rm", "--platform", "linux/amd64", "--entrypoint", "git", snapshot, "-C", "/testbed", "diff", "--binary", "--"), check=False)
    (output / "model.patch").write_bytes(patch.stdout)
    arm_slug = re.sub(r"[^a-z0-9]+", "-", args.arm.lower()).strip("-")
    _write_json(output / "preds.json", {instance_id: {
        "model_name_or_path": f"codex-{args.agent_version}-{args.model.replace('/', '_')}-{arm_slug}",
        "model_patch": patch.stdout.decode("utf-8", errors="replace"),
        "instance_id": instance_id,
    }})
    event_summary = _event_summary(events)
    trace_summary = _trace_summary(trace)
    _write_json(output / "event_summary.json", event_summary)
    _write_json(output / "run_manifest.json", {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "arm": args.arm,
        "agent": "codex-cli",
        "agent_version": args.agent_version,
        "agent_protocol": "openai_responses",
        "native_context_management": "ephemeral_session_no_compaction_claim",
        "instance_id": instance_id,
        "task_index": task_index,
        "benchmark_card": str(benchmark),
        "benchmark_card_sha256": _sha256(benchmark.read_bytes()),
        "reference_trajectory": str(trajectory),
        "reference_trajectory_sha256": _sha256(trajectory.read_bytes()),
        "model": args.model,
        "provider_mode": args.provider_mode,
        "upstream_base_url": args.upstream_base_url,
        "source_image": source_image,
        "derived_image": image,
        "container": container,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "exit_code": execution.returncode,
        "timed_out": timed_out,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_output_tokens": args.max_completion_tokens,
        },
        "responses_trace": trace_summary,
        "event_summary": event_summary,
        "workspace_status_sha256": _sha256(status.stdout),
        "patch_sha256": _sha256(patch.stdout),
        "patch_bytes": len(patch.stdout),
        "official_resolution": None,
        "notes": [
            "Codex uses its native prompt, shell/file tools, and Responses protocol.",
            (
                "The audit proxy does not translate protocol or select history; "
                "it only fills frozen controls and records the request envelope."
                if args.provider_mode == "responses_audit" else
                "Codex uses its built-in Ollama OSS provider; no external "
                "Responses-to-Chat translation is attributed to PRA."
            ),
            "Policy transfer is blocked until the Responses history representation is inventoried from this FULL control.",
        ],
    })
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="FULL")
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--agent-version", default="0.153.0")
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument(
        "--provider-mode",
        choices=("responses_audit", "ollama_oss"),
        default="responses_audit",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--upstream-timeout-seconds", type=int, default=3600)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--container")
    parser.add_argument("--build-timeout-seconds", type=int, default=1200)
    parser.add_argument("--agent-timeout-seconds", type=int, default=7200)
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--keep-snapshot", action="store_true")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
