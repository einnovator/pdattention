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
    LocalSessionService,
    NegotiatedRemoteBackend,
    PRAAgent,
    PRAAgentConfig,
    PRARuntime,
    PRARuntimeConfig,
    SideEffectClass,
    Tool,
    Toolset,
    ToolCallGuardDecision,
)
from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.agent_transport import TEXT_TOOL_OBSERVATION_PROJECTION

from .run_pi_swebench import (
    _sha256,
    _write_json,
    load_locked_task,
    observed_model_identity,
    swebench_image,
    task_prompt,
)
from .pra_agent_policy import PRAAgentMatchedTailSelector
from .run_autonomous_swebench import _exact_token_counter
from .miniswe_semantics import classify_bash_operation
from pra_hf.tool_semantics import OperationKind


PRA_SWE_AGENT_BEHAVIOR = (
    "Treat each successful tool observation as authoritative. Do not repeat an "
    "identical tool call unless its observation was incomplete, failed, or the "
    "workspace changed. When the issue names a file, function, or symbol, "
    "inspect that target directly. Do not inspect repository history merely "
    "because the issue mentions an earlier commit; use history only when the "
    "current source leaves a concrete unresolved question. Prefer targeted "
    "search and bounded reads over broad repository enumeration, and require "
    "each inspection to answer a specific unresolved question. Use at most "
    "eight discovery/read-only actions before the first source mutation; then "
    "make the best small general edit or report a concrete blocker. Take the "
    "requested action rather than only describing or promising it. Inspect the "
    "resulting diff, run focused verification when feasible, and only then "
    "finish."
)


class SWEInspectionBudgetGuard:
    """Require progress after bounded pre-mutation repository inspection."""

    def __init__(
        self, workspace: "DockerWorkspaceTools", *, max_inspections: int = 8
    ) -> None:
        if max_inspections < 1:
            raise ValueError("max_inspections must be positive")
        self.workspace = workspace
        self.max_inspections = max_inspections

    @staticmethod
    def _protected_mutation_path(value: object) -> bool:
        text = str(value or "").replace("\\", "/")
        path = PurePosixPath(text)
        parts = {part.lower() for part in path.parts}
        name = path.name.lower()
        if parts & {"test", "tests", "benchmarks", ".github"}:
            return True
        if name.startswith("test_") or name.endswith(("_test.py", ".test.js")):
            return True
        return name in {
            "package.json",
            "package-lock.json",
            "pyproject.toml",
            "requirements.txt",
            "setup.cfg",
            "setup.py",
            "tox.ini",
        }

    def __call__(self, _state, call, _resource, executions) -> str | None:
        if call.name in {"write_file", "replace_text"}:
            path = call.arguments.get("path")
            if self._protected_mutation_path(path):
                return (
                    "protected_path. This benchmark permits source fixes but "
                    f"forbids modifying tests, dependency/build files, or "
                    f"benchmark metadata; {path!r} was not written."
                )
        if len(executions) < self.max_inspections:
            return None
        if self.workspace.has_tracked_source_patch():
            return None
        if call.name in {"write_file", "replace_text"}:
            return None
        if call.name == "run_command":
            command = str(call.arguments.get("command") or "")
            if classify_bash_operation(command) == OperationKind.WRITE:
                return None
        return ToolCallGuardDecision(
            reason=(
                "inspection_budget_exhausted. No tracked source patch exists after "
                f"{self.max_inspections} executed tools. Read-only discovery is "
                "now undisclosed. The next tool must mutate the source using "
                "write_file, replace_text, or a clearly mutating run_command; "
                "otherwise report a concrete blocker."
            ),
            suppress_tool_names=(
                "list_files", "read_file", "search_text", "git_status",
            ),
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

    def bind_container(self, container: str) -> None:
        """Retarget stable tool identities to another isolated workspace.

        Persistent agent sessions keep one capability/tool schema while each
        SWE-bench issue executes in its own official container.  The bound
        methods held by :class:`Toolset` resolve ``self.container`` at call
        time, so switching this pointer does not rewrite tool identities or
        the model-visible protocol.
        """

        if not container.strip():
            raise ValueError("container cannot be empty")
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
        """List at most 500 non-Git files below a workspace-relative path."""

        root = self._relative_path(path)
        command = (
            f"find {shlex.quote(root)} -path '*/.git' -prune -o "
            f"-type f -name {shlex.quote(pattern)} "
            "-print | LC_ALL=C sort | head -501"
        )
        result = self._shell(command)
        rows = self._text(result.stdout).splitlines()
        return {
            "path": root,
            "files": rows[:500],
            "truncated": len(rows) > 500,
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
            # Preserve the middleware's observed operation class in the
            # durable tool result.  The history policy must not have to parse
            # rendered prose later to distinguish an edit, verification, or
            # source inspection performed through the generic shell tool.
            "operation_kind": classify_bash_operation(command).value,
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

    def completion_rejection(self) -> str | None:
        """Reject apparent completion until a tracked source patch exists."""

        result = self._shell("git diff --quiet --")
        if result.returncode == 0:
            return (
                "No tracked source patch exists. Continue working and use an "
                "authorized tool to implement the fix before finishing."
            )
        if result.returncode == 1:
            return None
        return (
            "Workspace mutation status could not be verified; inspect git diff "
            "before finishing."
        )

    def has_tracked_source_patch(self) -> bool:
        """Return whether tracked workspace content differs from the base."""

        return self._shell("git diff --quiet --").returncode == 1

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


def _durable_tool_events(
    records: Sequence[ContextRecord],
) -> list[dict[str, object]]:
    """Recover committed tool observations even when a turn raises later."""

    rows: list[dict[str, object]] = []
    for record in records:
        if record.record_type != RecordType.TOOL_RESPONSE:
            continue
        payload = record.payload if isinstance(record.payload, Mapping) else {}
        rows.append({
            "index": len(rows) + 1,
            "record_id": record.record_id,
            "producer_tool_uri": payload.get("producer_tool_uri"),
            "call_id": payload.get("call_id"),
            "payload": dict(payload),
        })
    return rows


def run(args: argparse.Namespace) -> Path:
    benchmark = Path(args.benchmark_card).resolve()
    _, instance_id, task_index = load_locked_task(benchmark, args.instance_id)
    trajectory = Path(args.reference_trajectory).resolve()
    model_identity = observed_model_identity(args)
    count_tokens, tokenizer_identity = _exact_token_counter(
        args.tokenizer,
        args.tokenizer_revision,
        allow_whitespace=(
            args.allow_whitespace_tokenizer or args.tokenizer == "whitespace"
        ),
    )
    retention_fraction = (
        1.0 if args.history_policy == "full" else args.retention_fraction
    )
    history_selector = PRAAgentMatchedTailSelector(
        retention_fraction=retention_fraction,
        count_tokens=count_tokens,
        tokenizer_identity=tokenizer_identity,
        protected_head_turns=args.protected_head_turns,
        protected_tail_turns=args.protected_tail_turns,
    )
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
    workspace = DockerWorkspaceTools(args.docker, container)
    tools = workspace.toolset(tenant_id)
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
        session_service=LocalSessionService(output / "live_session"),
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
            behavior_instructions=PRA_SWE_AGENT_BEHAVIOR,
        ),
        toolset=tools,
        history_selector=history_selector,
        tool_call_guard=SWEInspectionBudgetGuard(workspace),
        completion_guard=lambda _state, _text, _executions: (
            workspace.completion_rejection()
        ),
    )

    started = datetime.now(timezone.utc)
    error: Exception | None = None
    turn = None
    durable_records: tuple[ContextRecord, ...] = ()
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
            durable_records = tuple(agent.state.records)
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
                # Rejected or schema-invalid actions intentionally have no
                # committed tool-result record.  Export the decision without
                # inventing one; durable executed observations remain
                # recoverable from the session record stream below.
                "record": (
                    None if execution.record is None
                    else execution.record.compact_view()
                ),
            })
    (output / "tool_events.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in events),
        encoding="utf-8",
    )
    durable_events = _durable_tool_events(durable_records)
    (output / "durable_tool_events.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in durable_events),
        encoding="utf-8",
    )
    _write_json(output / "selection_trace.json", {
        "schema_version": 1,
        "requests": history_selector.traces,
    })
    cumulative_full_tokens = sum(
        int(row["full_history_tokens"]) for row in history_selector.traces
    )
    cumulative_selected_tokens = sum(
        int(row["selected_history_tokens"]) for row in history_selector.traces
    )
    cumulative_materialized_tokens = sum(
        int(row["materialized_history_tokens"]) for row in history_selector.traces
    )
    policy_label = (
        "full" if args.history_policy == "full"
        else f"matched-tail-h{args.protected_head_turns}-t{args.protected_tail_turns}"
    )
    predictions = {
        instance_id: {
            "model_name_or_path": f"pra-agent-{args.model}-{policy_label}",
            "model_patch": patch.stdout.decode("utf-8", errors="replace"),
            "instance_id": instance_id,
        }
    }
    _write_json(output / "preds.json", predictions)

    transport = backend.inspect()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "arm": "FULL" if args.history_policy == "full" else "MATCHED_RECENCY",
        "agent": "pra-agent",
        "agent_protocol": "typed_records_with_openai_text_fallback",
        "text_tool_observation_projection": TEXT_TOOL_OBSERVATION_PROJECTION,
        "pre_mutation_inspection_budget": 8,
        "session_persistence": "atomic_local_json_per_record",
        "behavior_instructions": PRA_SWE_AGENT_BEHAVIOR,
        "behavior_instructions_sha256": _sha256(
            PRA_SWE_AGENT_BEHAVIOR.encode("utf-8")
        ),
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
        "history_selection": {
            "policy": args.history_policy,
            "retention_fraction": retention_fraction,
            "protected_head_turns": args.protected_head_turns,
            "protected_tail_turns": args.protected_tail_turns,
            "tokenizer": tokenizer_identity,
            "tokenizer_revision": args.tokenizer_revision,
            "request_count": len(history_selector.traces),
            "cumulative_full_history_tokens": cumulative_full_tokens,
            "cumulative_selected_history_tokens": cumulative_selected_tokens,
            "cumulative_materialized_history_tokens": cumulative_materialized_tokens,
            "own_saving_fraction": (
                0.0 if not cumulative_full_tokens else
                1.0 - cumulative_materialized_tokens / cumulative_full_tokens
            ),
        },
        "generation": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_completion_tokens": args.max_completion_tokens,
        },
        "tool_event_count": len(durable_events),
        "returned_turn_tool_event_count": len(events),
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
    parser.add_argument(
        "--history-policy", choices=("full", "matched_token_tail"), default="full"
    )
    parser.add_argument("--retention-fraction", type=float, default=0.9)
    parser.add_argument("--protected-head-turns", type=int, default=2)
    parser.add_argument("--protected-tail-turns", type=int, default=4)
    parser.add_argument("--tokenizer", default="whitespace")
    parser.add_argument("--tokenizer-revision", default="diagnostic")
    parser.add_argument("--allow-whitespace-tokenizer", action="store_true")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
