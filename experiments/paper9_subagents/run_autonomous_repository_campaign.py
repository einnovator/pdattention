"""Run a real model-driven sequential/parallel Paper 9 subagent campaign.

Unlike the deterministic tracked-source cohort, each child decides whether and
how to inspect the repository through read-only tools. The factorial conditions
separate scheduling from completed-peer context consumption:

* sequential/parallel with hard peer isolation;
* sequential/parallel with completed-peer visibility and fielded BM25 routing.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.subagent_context import (
    ContextVisibilityPolicy,
    EffectType,
    ResourceIdentity,
    ToolEffectDescriptor,
)
from pra_hf.subagent_harness import DeclarativeTool, SubagentHarness, SubagentSpec
from pra_hf.subagent_routing import (
    DescendantRecordRouter,
    DescendantRoutingMode,
    visible_routing_candidates,
)


@dataclass(frozen=True)
class CampaignTask:
    task_id: str
    question: str
    expected_paths: tuple[str, ...]


@dataclass(frozen=True)
class CampaignCondition:
    name: str
    parallel: bool
    completed_peer_context: bool


@dataclass
class AgentOutcome:
    task_id: str
    question: str
    expected_paths: tuple[str, ...]
    answer: str
    path_hit: bool
    failed: bool
    error: str | None
    model_calls: int
    prompt_tokens: int
    generated_tokens: int
    model_seconds: float
    logical_tool_calls: int
    physical_tool_calls: int
    reused_tool_calls: int
    routed_peer_records: int
    routed_peer_tokens: int
    trace: list[dict[str, object]]


TASKS = (
    CampaignTask(
        "unknown_effect",
        "Where and how are UNKNOWN tool side effects rejected from cross-agent reuse?",
        ("src/pra_hf/subagent_context.py",),
    ),
    CampaignTask(
        "resource_invalidation",
        "Which implementation performs resource-level write invalidation for reusable results?",
        ("src/pra_hf/subagent_context.py",),
    ),
    CampaignTask(
        "external_validation",
        "Where is external resource validation required before a cached result can be reused?",
        ("src/pra_hf/subagent_context.py",),
    ),
    CampaignTask(
        "parallel_scheduler",
        "Which implementation owns bounded sequential and parallel child callback scheduling?",
        ("src/pra_hf/subagent_harness.py",),
    ),
    CampaignTask(
        "duplicate_read_coalescing",
        "Where are concurrent identical tool-call misses coalesced into one physical execution?",
        ("src/pra_hf/subagent_harness.py",),
    ),
    CampaignTask(
        "descendant_router",
        "Where is field-aware BM25 ranking of completed-child evidence implemented?",
        ("src/pra_hf/subagent_routing.py",),
    ),
    CampaignTask(
        "completed_visibility",
        "Which implementation enforces completed-only descendant or peer record visibility?",
        ("src/pra_hf/subagent_context.py",),
    ),
    CampaignTask(
        "mlx_native_cache",
        "Where is exact split-prefill MLX state materialized for a child request?",
        ("src/pra_hf/subagent_mlx_native.py",),
    ),
)

CONDITIONS = (
    CampaignCondition("sequential_isolated", False, False),
    CampaignCondition("parallel_isolated", True, False),
    CampaignCondition("sequential_completed", False, True),
    CampaignCondition("parallel_completed", True, True),
)

TOOL_SCHEMAS = (
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "Read one UTF-8 repository file by relative path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search tracked source-like files for a literal text fragment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "description": "Optional relative subtree."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List source-like files below a relative repository path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
)

_JSON_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_QWEN_FUNCTION = re.compile(
    r"<function=([^>]+)>(.*?)</function>", re.DOTALL | re.IGNORECASE
)
_QWEN_PARAMETER = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE
)


def _textual_tool_calls(content: object) -> list[dict[str, object]]:
    """Normalize JSON and Qwen textual tool syntax into Ollama call objects."""

    text = str(content or "")
    calls: list[dict[str, object]] = []
    for match in _JSON_TOOL_CALL.finditer(text):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and isinstance(value.get("name"), str):
            arguments = value.get("arguments", {})
            if isinstance(arguments, Mapping):
                calls.append(
                    {"function": {"name": value["name"], "arguments": dict(arguments)}}
                )
    if calls:
        return calls
    for match in _QWEN_FUNCTION.finditer(text):
        arguments = {
            parameter.group(1).strip(): parameter.group(2).strip()
            for parameter in _QWEN_PARAMETER.finditer(match.group(2))
        }
        calls.append(
            {"function": {"name": match.group(1).strip(), "arguments": arguments}}
        )
    return calls


class ChatClient(Protocol):
    def chat(
        self, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]]
    ) -> tuple[dict[str, object], dict[str, float | int]]:
        """Return one assistant message and token/timing metrics."""


class OllamaChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        max_new_tokens: int,
        timeout_seconds: int,
    ) -> None:
        self.url = base_url.rstrip("/") + "/api/chat"
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds

    def chat(
        self, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]]
    ) -> tuple[dict[str, object], dict[str, float | int]]:
        payload = {
            "model": self.model,
            "stream": False,
            "keep_alive": "30m",
            "messages": list(messages),
            "tools": list(tools),
            "options": {
                "temperature": 0,
                "num_predict": self.max_new_tokens,
                "num_ctx": 32768,
            },
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        error_text = ""
        for attempt in range(3):
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read())
                message = result.get("message", {})
                if not isinstance(message, dict):
                    raise RuntimeError("Ollama response omitted an assistant message.")
                return message, {
                    "prompt_tokens": int(result.get("prompt_eval_count", 0)),
                    "generated_tokens": int(result.get("eval_count", 0)),
                    "model_seconds": time.perf_counter() - started,
                }
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                error_text = f"{type(error).__name__}: {error}"
                time.sleep(2**attempt)
        raise RuntimeError(f"Ollama generation failed after retries: {error_text}")


class RepositoryToolSet:
    """Read-only autonomous tools with harness-visible effect descriptors."""

    _SUFFIXES = frozenset({".py", ".md", ".toml", ".yaml", ".yml", ".json"})

    def __init__(self, repo: Path, *, max_output_chars: int = 16_000) -> None:
        self.repo = repo.resolve()
        self.max_output_chars = max_output_chars
        self._physical_calls = 0
        self._lock = threading.Lock()
        self.tools = {
            "read_text": DeclarativeTool(
                "tool://repository/read_text", self._read_text, self._read_effect
            ),
            "search_text": DeclarativeTool(
                "tool://repository/search_text", self._search_text, self._search_effect
            ),
            "list_files": DeclarativeTool(
                "tool://repository/list_files", self._list_files, self._list_effect
            ),
        }

    @property
    def physical_calls(self) -> int:
        with self._lock:
            return self._physical_calls

    def _count(self) -> None:
        with self._lock:
            self._physical_calls += 1

    def _resolve(self, relative: object) -> Path:
        value = str(relative or ".").replace("\\", "/")
        target = (self.repo / value).resolve()
        if target != self.repo and self.repo not in target.parents:
            raise ValueError("Path escapes the repository root.")
        return target

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.repo).as_posix()

    def _bounded(self, value: str) -> str:
        if len(value) <= self.max_output_chars:
            return value
        return value[: self.max_output_chars] + "\n...[truncated by campaign harness]"

    def _read_text(self, arguments: Mapping[str, object]) -> str:
        self._count()
        path = self._resolve(arguments.get("path"))
        if not path.is_file():
            return f"ERROR: file not found: {self._relative(path)}"
        return self._bounded(path.read_text(encoding="utf-8", errors="replace"))

    def _search_text(self, arguments: Mapping[str, object]) -> str:
        self._count()
        query = str(arguments.get("query", "")).strip().lower()
        if not query:
            return "ERROR: query is required"
        root = self._resolve(arguments.get("path", "."))
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        matches: list[str] = []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in self._SUFFIXES:
                continue
            if any(part in {".git", ".venv", "__pycache__"} for part in path.parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, 1):
                if query in line.lower():
                    matches.append(f"{self._relative(path)}:{line_number}:{line.strip()}")
                    if len(matches) >= 80:
                        return self._bounded("\n".join(matches))
        return self._bounded("\n".join(matches) if matches else "NO MATCHES")

    def _list_files(self, arguments: Mapping[str, object]) -> str:
        self._count()
        root = self._resolve(arguments.get("path", "."))
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        values = [
            self._relative(path)
            for path in paths
            if path.is_file()
            and path.suffix.lower() in self._SUFFIXES
            and not any(part in {".git", ".venv", "__pycache__"} for part in path.parts)
        ]
        return self._bounded("\n".join(values[:500]))

    def _read_effect(self, arguments: Mapping[str, object]) -> ToolEffectDescriptor:
        path = self._relative(self._resolve(arguments.get("path")))
        return ToolEffectDescriptor(
            EffectType.READ,
            (ResourceIdentity("os.FILE", path),),
            reuse_enabled=True,
        )

    def _search_effect(self, arguments: Mapping[str, object]) -> ToolEffectDescriptor:
        path = self._relative(self._resolve(arguments.get("path", ".")))
        return ToolEffectDescriptor(
            EffectType.READ,
            (ResourceIdentity("os.DIRECTORY", path),),
            reuse_enabled=True,
        )

    def _list_effect(self, arguments: Mapping[str, object]) -> ToolEffectDescriptor:
        return self._search_effect(arguments)


def _peer_context(
    harness: SubagentHarness, agent_uuid: str, question: str
) -> tuple[str, int, int]:
    candidates = visible_routing_candidates(harness.graph, agent_uuid)
    if not candidates:
        return "", 0, 0
    route = DescendantRecordRouter().route(
        question,
        candidates,
        mode=DescendantRoutingMode.FIELDED_BM25,
        top_k=1,
    )
    selected = route.selected[0]
    identity = selected.record.record_id
    payload = selected.record.payload
    if isinstance(payload, Mapping):
        arguments = payload.get("arguments")
        if isinstance(arguments, Mapping) and isinstance(arguments.get("path"), str):
            identity = str(arguments["path"])
        output = payload.get("output", selected.text)
    else:
        output = selected.text
    text = output if isinstance(output, str) else json.dumps(output, sort_keys=True)
    text = text[:16_000]
    context = f"\nA completed peer exposed this routed record ({identity}):\n{text}\n"
    return context, 1, selected.token_count


def run_agent(
    client: ChatClient,
    harness: SubagentHarness,
    tool_set: RepositoryToolSet,
    agent_uuid: str,
    task: CampaignTask,
    *,
    use_peer_context: bool,
    max_steps: int,
) -> AgentOutcome:
    for descriptor in harness.graph.agents:
        if descriptor.agent_uuid not in {"root", agent_uuid}:
            harness.link_subagent_peers(agent_uuid, descriptor.agent_uuid)

    peer_text, peer_records, peer_tokens = (
        _peer_context(harness, agent_uuid, task.question)
        if use_peer_context
        else ("", 0, 0)
    )
    messages: list[dict[str, object]] = [
        {
            "role": "system",
            "content": (
                "You are an autonomous repository investigator. Use the read-only tools "
                "to verify the answer. Finish with a concise answer naming the exact relative "
                "source path and the relevant class or function. Do not guess."
            ),
        },
        {"role": "user", "content": "/no_think " + task.question + peer_text},
    ]
    trace: list[dict[str, object]] = []
    totals = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "generated_tokens": 0,
        "model_seconds": 0.0,
        "logical_tool_calls": 0,
        "physical_tool_calls": 0,
        "reused_tool_calls": 0,
    }
    answer = ""
    error_text: str | None = None
    try:
        for step in range(max_steps):
            assistant, metrics = client.chat(messages, TOOL_SCHEMAS)
            totals["model_calls"] += 1
            totals["prompt_tokens"] += int(metrics.get("prompt_tokens", 0))
            totals["generated_tokens"] += int(metrics.get("generated_tokens", 0))
            totals["model_seconds"] += float(metrics.get("model_seconds", 0.0))
            calls = assistant.get("tool_calls", ())
            if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
                calls = ()
            if not calls:
                calls = _textual_tool_calls(assistant.get("content", ""))
            if calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": calls})
            else:
                messages.append(dict(assistant))
            if not calls:
                answer = str(assistant.get("content", "")).strip()
                break
            for call in calls:
                function = call.get("function", {}) if isinstance(call, Mapping) else {}
                name = str(function.get("name", ""))
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                if not isinstance(arguments, Mapping) or name not in tool_set.tools:
                    output = f"ERROR: invalid or unavailable tool call: {name}"
                    executed = False
                    reused = False
                    record_uuid = ""
                else:
                    result = harness.execute_tool(
                        agent_uuid, tool_set.tools[name], dict(arguments)
                    )
                    output = str(result.output)
                    executed = result.executed
                    reused = not result.executed
                    record_uuid = result.record.record_id
                    totals["logical_tool_calls"] += 1
                    totals["physical_tool_calls"] += int(executed)
                    totals["reused_tool_calls"] += int(reused)
                trace.append(
                    {
                        "step": step,
                        "tool": name,
                        "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
                        "executed": executed,
                        "reused": reused,
                        "record_uuid": record_uuid,
                        "output_chars": len(output),
                    }
                )
                messages.append({"role": "tool", "tool_name": name, "content": output})
        else:
            error_text = "max_steps_exhausted"
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"

    normalized_answer = answer.replace("\\", "/").lower()
    path_hit = any(path.lower() in normalized_answer for path in task.expected_paths)
    harness.graph.append_record(
        agent_uuid,
        ContextRecord(
            f"answer:{agent_uuid}",
            RecordType.GENERIC_TEXT,
            json.dumps(
                {"task_id": task.task_id, "answer": answer, "path_hit": path_hit},
                sort_keys=True,
            ),
        ),
    )
    return AgentOutcome(
        task_id=task.task_id,
        question=task.question,
        expected_paths=task.expected_paths,
        answer=answer,
        path_hit=path_hit,
        failed=error_text is not None,
        error=error_text,
        routed_peer_records=peer_records,
        routed_peer_tokens=peer_tokens,
        trace=trace,
        **totals,
    )


def run_condition(
    client: ChatClient,
    repo: Path,
    tasks: Sequence[CampaignTask],
    condition: CampaignCondition,
    *,
    seed: int,
    max_steps: int,
    max_workers: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ordered = list(tasks)
    random.Random(seed).shuffle(ordered)
    visibility = "completed_only" if condition.completed_peer_context else "none"
    harness = SubagentHarness(f"autonomous-{condition.name}-{seed}", root_agent_uuid="root")
    tool_set = RepositoryToolSet(repo)
    task_by_id = {task.task_id: task for task in ordered}
    specs = tuple(
        SubagentSpec(
            parent_task_uuid=task.task_id,
            context_policy=ContextVisibilityPolicy(peer=visibility),
        )
        for task in ordered
    )

    def runner(child, active_harness):
        task = task_by_id[str(child.parent_task_uuid)]
        return run_agent(
            client,
            active_harness,
            tool_set,
            child.agent_uuid,
            task,
            use_peer_context=condition.completed_peer_context,
            max_steps=max_steps,
        )

    started = time.perf_counter()
    results = harness.run_subagents(
        "root",
        specs,
        runner,
        parallel=condition.parallel,
        max_workers=max_workers,
    )
    wall_seconds = time.perf_counter() - started
    rows: list[dict[str, object]] = []
    for result in results:
        if result.value is None:
            task_id = str(harness.graph.descriptor(result.agent_uuid).parent_task_uuid)
            outcome = AgentOutcome(
                task_id,
                task_by_id[task_id].question,
                task_by_id[task_id].expected_paths,
                "",
                False,
                True,
                result.error,
                0,
                0,
                0,
                0.0,
                0,
                0,
                0,
                0,
                0,
                [],
            )
        else:
            outcome = result.value
        row = asdict(outcome)
        row.update(
            {
                "condition": condition.name,
                "parallel": condition.parallel,
                "completed_peer_context": condition.completed_peer_context,
                "seed": seed,
                "agent_uuid": result.agent_uuid,
                "agent_elapsed_ms": result.elapsed_ms,
            }
        )
        rows.append(row)
    summary = {
        "condition": condition.name,
        "seed": seed,
        "tasks": len(rows),
        "wall_seconds": wall_seconds,
        "path_accuracy": sum(bool(row["path_hit"]) for row in rows) / max(1, len(rows)),
        "failures": sum(bool(row["failed"]) for row in rows),
        "model_calls": sum(int(row["model_calls"]) for row in rows),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in rows),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "model_seconds_sum": sum(float(row["model_seconds"]) for row in rows),
        "logical_tool_calls": sum(int(row["logical_tool_calls"]) for row in rows),
        "physical_tool_calls": tool_set.physical_calls,
        "reused_tool_calls": sum(int(row["reused_tool_calls"]) for row in rows),
        "routed_peer_records": sum(int(row["routed_peer_records"]) for row in rows),
        "routed_peer_tokens": sum(int(row["routed_peer_tokens"]) for row in rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--seeds", default="11")
    parser.add_argument("--max-tasks", type=int, default=len(TASKS))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--conditions",
        default=",".join(condition.name for condition in CONDITIONS),
    )
    args = parser.parse_args()
    selected_names = {value.strip() for value in args.conditions.split(",") if value.strip()}
    conditions = [condition for condition in CONDITIONS if condition.name in selected_names]
    if selected_names != {condition.name for condition in conditions}:
        unknown = selected_names - {condition.name for condition in conditions}
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    tasks = TASKS[: args.max_tasks]
    if not tasks or not conditions or not seeds:
        raise ValueError("At least one task, condition, and seed are required.")

    client = OllamaChatClient(
        args.base_url,
        args.model,
        max_new_tokens=args.max_new_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    for seed in seeds:
        for condition in conditions:
            rows, summary = run_condition(
                client,
                args.repo.resolve(),
                tasks,
                condition,
                seed=seed,
                max_steps=args.max_steps,
                max_workers=args.max_workers,
            )
            all_rows.extend(rows)
            condition_rows.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)

    with (args.output / "agent_rows.jsonl").open("w", encoding="utf-8") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    final = {
        "protocol": "paper9-autonomous-repository-v1",
        "evidence_scope": (
            "real autonomous read-only model loops; not a patch-generation benchmark"
        ),
        "model": args.model,
        "base_url": args.base_url,
        "seeds": seeds,
        "task_ids": [task.task_id for task in tasks],
        "max_steps": args.max_steps,
        "max_workers": args.max_workers,
        "conditions": condition_rows,
    }
    (args.output / "summary.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
