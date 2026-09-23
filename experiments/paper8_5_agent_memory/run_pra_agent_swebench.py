"""Admit the record-native PRA Agent on one locked SWE-bench task.

The agent runs on the host while every workspace operation is executed inside
the official SWE-bench image.  This keeps the PRA SDK's typed tool records and
authorization boundary intact without copying the repository out of the
benchmark environment.  The exported patch is graded by the same official
harness used by the other Paper 8.5 agents.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from pra_hf import (
    AgentConfig,
    CapabilitySDK,
    InMemorySessionService,
    NegotiatedRemoteBackend,
    PRAAgent,
    PRAAgentConfig,
    PRARuntime,
    PRARuntimeConfig,
    SideEffectClass,
    Tool,
    Toolset,
)

from .run_pi_swebench import (
    _sha256,
    _write_json,
    load_locked_task,
    observed_model_identity,
    swebench_image,
    task_prompt,
)


def _run(
    command: Sequence[str], *, input_bytes: bytes | None = None,
    timeout: int | None = None, check: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command), input=input_bytes, capture_output=True,
        timeout=timeout, check=check,
    )


class DockerWorkspaceTools:
    """Typed workspace tools backed by one official task container."""

    def __init__(self, docker: str, container: str) -> None:
        self.docker = docker
        self.container = container

    @staticmethod
    def _relative_path(value: str) -> str:
        path = PurePosixPath(value or ".")
        if path.is_absolute() and path.parts[:2] == ("/", "testbed"):
            path = (
                PurePosixPath(*path.parts[2:])
                if len(path.parts) > 2 else PurePosixPath(".")
            )
        if path.is_absolute() or ".." in path.parts:
            raise PermissionError(f"Path escapes /testbed: {value}")
        return str(path)

    def _shell(
        self, command: str, *, timeout_seconds: int = 120,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        return _run(
            (
                self.docker, "exec", "-i", "-w", "/testbed",
                self.container, "bash", "-lc", command,
            ),
            input_bytes=input_bytes,
            timeout=timeout_seconds,
        )

    @staticmethod
    def _text(value: bytes) -> str:
        return value.decode("utf-8", errors="replace")

    def list_files(self, path: str = ".", pattern: str = "*") -> dict[str, object]:
        """List at most 1000 files below one workspace-relative path."""

        root = self._relative_path(path)
        command = (
            f"find {shlex.quote(root)} -type f -name {shlex.quote(pattern)} "
            "-print | LC_ALL=C sort | head -1001"
        )
        result = self._shell(command)
        rows = self._text(result.stdout).splitlines()
        return {
            "path": root,
            "files": rows[:1000],
            "truncated": len(rows) > 1000,
            "exit_code": result.returncode,
            "stderr": self._text(result.stderr)[-4000:],
        }

    def read_file(
        self, path: str, start_line: int = 1, end_line: int = 400,
    ) -> dict[str, object]:
        """Read a bounded inclusive line range from a workspace file."""

        source = self._relative_path(path)
        if start_line < 1 or end_line < start_line or end_line - start_line > 2000:
            raise ValueError("Expected a positive range of at most 2001 lines.")
        quoted = shlex.quote(source)
        result = self._shell(
            f"sed -n '{start_line},{end_line}p' {quoted}; "
            f"printf '\\n__PRA_TOTAL_LINES__='; wc -l < {quoted}"
        )
        text = self._text(result.stdout)
        marker = "\n__PRA_TOTAL_LINES__="
        body, _, total = text.rpartition(marker)
        return {
            "path": source,
            "start_line": start_line,
            "end_line": end_line,
            "text": body if marker in text else text,
            "total_lines": int(total.strip()) if total.strip().isdigit() else None,
            "exit_code": result.returncode,
            "stderr": self._text(result.stderr)[-4000:],
        }

    def search_text(
        self, query: str, path: str = ".", pattern: str = "*.py",
    ) -> dict[str, object]:
        """Search literal text in bounded workspace files."""

        if not query:
            raise ValueError("query cannot be empty")
        root = self._relative_path(path)
        command = (
            "grep -RIn --binary-files=without-match "
            f"--include={shlex.quote(pattern)} -F -- {shlex.quote(query)} "
            f"{shlex.quote(root)} | head -501"
        )
        result = self._shell(command)
        rows = self._text(result.stdout).splitlines()
        return {
            "query": query,
            "matches": rows[:500],
            "truncated": len(rows) > 500,
            "exit_code": result.returncode,
            "stderr": self._text(result.stderr)[-4000:],
        }

    def write_file(
        self, path: str, content: str, overwrite: bool = False,
    ) -> dict[str, object]:
        """Write one UTF-8 file inside /testbed."""

        target = self._relative_path(path)
        script = (
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "exists=p.exists(); overwrite=sys.argv[2]=='1'; "
            "(_ for _ in ()).throw(FileExistsError(str(p))) "
            "if exists and not overwrite else None; "
            "p.parent.mkdir(parents=True,exist_ok=True); "
            "data=sys.stdin.buffer.read(); p.write_bytes(data); "
            "print(int(exists),len(data))"
        )
        result = _run(
            (
                self.docker, "exec", "-i", "-w", "/testbed", self.container,
                "python", "-c", script, target, "1" if overwrite else "0",
            ),
            input_bytes=content.encode("utf-8"), timeout=120,
        )
        if result.returncode:
            raise RuntimeError(self._text(result.stderr)[-4000:])
        existed, size = self._text(result.stdout).strip().split()
        return {"path": target, "created": existed == "0", "bytes": int(size)}

    def replace_text(
        self, path: str, old: str, new: str, count: int = 1,
    ) -> dict[str, object]:
        """Replace an exact text fragment in one workspace file."""

        target = self._relative_path(path)
        if not old or count < 1:
            raise ValueError("old text and a positive count are required")
        payload = json.dumps({"old": old, "new": new, "count": count}).encode()
        script = (
            "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "v=json.load(sys.stdin); s=p.read_text(); n=s.count(v['old']); "
            "(_ for _ in ()).throw(ValueError('exact text not found')) if n==0 else None; "
            "p.write_text(s.replace(v['old'],v['new'],v['count'])); "
            "print(n,min(v['count'],n))"
        )
        result = _run(
            (
                self.docker, "exec", "-i", "-w", "/testbed", self.container,
                "python", "-c", script, target,
            ),
            input_bytes=payload, timeout=120,
        )
        if result.returncode:
            raise RuntimeError(self._text(result.stderr)[-4000:])
        available, replaced = self._text(result.stdout).strip().split()
        return {
            "path": target,
            "available_occurrences": int(available),
            "replaced": int(replaced),
        }

    def run_command(
        self, command: str, cwd: str = ".", timeout_seconds: int = 120,
    ) -> dict[str, object]:
        """Run a shell command in the task container."""

        if not command or timeout_seconds < 1 or timeout_seconds > 1800:
            raise ValueError("command and timeout in 1..1800 are required")
        directory = self._relative_path(cwd)
        result = self._shell(
            f"cd {shlex.quote(directory)} && {command}",
            timeout_seconds=timeout_seconds,
        )
        return {
            "command": command,
            "cwd": directory,
            "exit_code": result.returncode,
            "stdout": self._text(result.stdout)[-100_000:],
            "stderr": self._text(result.stderr)[-100_000:],
        }

    def git_status(self) -> dict[str, object]:
        """Return the concise Git status of the task workspace."""

        result = self._shell("git status --short --branch")
        return {
            "exit_code": result.returncode,
            "status": self._text(result.stdout),
            "stderr": self._text(result.stderr)[-4000:],
        }

    def toolset(self, tenant_id: str) -> Toolset:
        definitions = (
            (self.list_files, SideEffectClass.READ, ("files", "workspace")),
            (self.read_file, SideEffectClass.READ, ("files", "read")),
            (self.search_text, SideEffectClass.READ, ("search", "files")),
            (self.git_status, SideEffectClass.READ, ("git", "status")),
            (self.write_file, SideEffectClass.WRITE, ("files", "write")),
            (self.replace_text, SideEffectClass.WRITE, ("files", "edit")),
            (self.run_command, SideEffectClass.WRITE, ("shell", "command")),
        )
        return Toolset(
            (
                Tool(
                    function,
                    side_effect=effect,
                    namespace="pra-swebench",
                    tenant_id=tenant_id,
                    tags=frozenset(tags),
                    metadata={"workspace": "official_swebench_container"},
                )
                for function, effect, tags in definitions
            ),
            name="pra-swebench",
        )


def run(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark_card).resolve()
    _, instance_id, task_index = load_locked_task(benchmark, args.instance_id)
    trajectory = Path(args.reference_trajectory).resolve()
    model_identity = observed_model_identity(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt = task_prompt(trajectory, instance_id)
    (output / "task_prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    image = swebench_image(instance_id)
    slug = re.sub(r"[^a-z0-9]+", "-", instance_id.lower()).strip("-")
    container = args.container or f"paper85-pra-agent-{slug}-{int(time.time())}"
    start = _run((
        args.docker, "run", "-d", "--name", container,
        "--platform", "linux/amd64", "-w", "/testbed", image, "sleep", "2h",
    ), timeout=args.image_timeout_seconds)
    (output / "container_start.stdout.log").write_bytes(start.stdout)
    (output / "container_start.stderr.log").write_bytes(start.stderr)
    if start.returncode:
        raise RuntimeError(f"task container failed to start: {start.returncode}")

    tenant_id = "paper8-5"
    tools = DockerWorkspaceTools(args.docker, container).toolset(tenant_id)
    capabilities = CapabilitySDK(AgentConfig(
        tools=tools.records,
        namespace="pra-agent",
        tenant_id=tenant_id,
        max_candidates=len(tools.records),
    ))
    backend = NegotiatedRemoteBackend(
        args.endpoint,
        args.model,
        transport="text",
        timeout_seconds=args.model_timeout_seconds,
        curl_executable=args.curl_executable,
        openai_fields={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
    )
    runtime = PRARuntime(
        config=PRARuntimeConfig(),
        backend=backend,
        capability_sdk=capabilities,
        executor=tools.executor(),
        session_service=InMemorySessionService(),
    )
    agent = PRAAgent(
        runtime,
        config=PRAAgentConfig(
            user_id="paper8-5",
            tenant_id=tenant_id,
            context_records=args.context_records,
            tool_candidates=len(tools.records),
            max_tool_rounds=args.max_tool_rounds,
            allow_writes=True,
            max_new_tokens=args.max_completion_tokens,
        ),
        toolset=tools,
    )

    started = datetime.now(timezone.utc)
    error: Exception | None = None
    turn = None
    try:
        agent.start_session(
            f"paper85-pra-agent-{slug}", task_description=prompt,
        )
        turn = agent.run_turn(prompt)
    except Exception as observed:
        error = observed
        (output / "agent_error.txt").write_text(
            f"{type(observed).__name__}: {observed}\n", encoding="utf-8"
        )
    finally:
        if agent.session is not None:
            agent.export_session(output / "session.json")
        agent.close()
    finished = datetime.now(timezone.utc)

    status = _run((
        args.docker, "exec", "-w", "/testbed", container,
        "git", "status", "--short",
    ))
    patch = _run((
        args.docker, "exec", "-w", "/testbed", container,
        "git", "diff", "--binary", "--",
    ))
    (output / "workspace_status.txt").write_bytes(status.stdout)
    (output / "model.patch").write_bytes(patch.stdout)

    events = []
    if turn is not None:
        (output / "final_response.txt").write_text(turn.text, encoding="utf-8")
        for index, execution in enumerate(turn.tool_executions, start=1):
            result = execution.execution
            events.append({
                "index": index,
                "accepted": result.accepted,
                "executed": result.executed,
                "reason": result.reason,
                "resource_uri": result.resource_uri,
                "tool_name": None if result.call is None else result.call.name,
                "arguments": None if result.call is None else dict(result.call.arguments),
                "output": dict(result.output),
                "record": execution.record.compact_view(),
            })
    (output / "tool_events.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in events),
        encoding="utf-8",
    )
    predictions = {
        instance_id: {
            "model_name_or_path": f"pra-agent-{args.model}-full",
            "model_patch": patch.stdout.decode("utf-8", errors="replace"),
            "instance_id": instance_id,
        }
    }
    _write_json(output / "preds.json", predictions)

    transport = backend.inspect()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "arm": "FULL",
        "agent": "pra-agent",
        "agent_protocol": "typed_records_with_openai_text_fallback",
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
        "endpoint": args.endpoint,
        "source_image": image,
        "container": container,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "max_tool_rounds": args.max_tool_rounds,
        "context_records": args.context_records,
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_completion_tokens": args.max_completion_tokens,
        },
        "tool_event_count": len(events),
        "patch_bytes": len(patch.stdout),
        "patch_sha256": _sha256(patch.stdout),
        "transport": transport,
        "error_type": None if error is None else type(error).__name__,
        "error_detail": None if error is None else str(error),
        "official_resolution": None,
    }
    _write_json(output / "run_manifest.json", manifest)

    _run((args.docker, "rm", "-f", container))
    if error is not None:
        raise RuntimeError(f"PRA Agent failed: {type(error).__name__}: {error}")
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--served-model")
    parser.add_argument("--model-revision")
    parser.add_argument("--ollama-tags-url")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--container")
    parser.add_argument("--max-tool-rounds", type=int, default=80)
    parser.add_argument("--context-records", type=int, default=512)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-timeout-seconds", type=int, default=3600)
    parser.add_argument("--image-timeout-seconds", type=int, default=900)
    parser.add_argument("--curl-executable")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
