"""Safe subprocess adapter for CLIs with verified noninteractive commands."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .base import AgentExecution, AgentTask, CodingAgentAdapter


_SECRET_NAMES = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


class CommandAgentAdapter(CodingAgentAdapter):
    """Execute one pinned CLI using argv, never a shell command string."""

    def __init__(
        self,
        *,
        executable: str,
        version_args: Sequence[str],
        run_args: Sequence[str],
        extra_args: Sequence[str] = (),
        prompt_arg: str | None = None,
        model_arg: str | None = None,
    ) -> None:
        self.executable = executable
        self.version_args = tuple(version_args)
        self.run_args = tuple(run_args)
        self.extra_args = tuple(extra_args)
        self.prompt_arg = prompt_arg
        self.model_arg = model_arg
        self.workspace = Path.cwd()
        self.provider: dict[str, Any] = {}

    def install_or_verify(self) -> Mapping[str, Any]:
        path = shutil.which(self.executable)
        return {"installed": path is not None, "executable": path, "version": self.version() if path else None}

    def version(self) -> str:
        completed = subprocess.run(
            [self.executable, *self.version_args], capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=15, check=False,
        )
        text = (completed.stdout or completed.stderr).strip().splitlines()
        return text[0] if text else "unknown"

    def configure_provider(self, **settings: Any) -> None:
        self.provider = {key: value for key, value in settings.items() if value is not None}

    def configure_workspace(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def run_task(self, task: AgentTask) -> AgentExecution:
        argv = [self.executable, *self.run_args, *self.extra_args]
        if self.model_arg and self.provider.get("model"):
            argv.extend((self.model_arg, str(self.provider["model"])))
        if self.prompt_arg:
            argv.extend((self.prompt_arg, task.instruction))
        else:
            argv.append(task.instruction)
        env = os.environ.copy()
        for key, value in dict(self.provider.get("environment", {})).items():
            env[str(key)] = str(value)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                argv, cwd=task.workspace, env=env, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=task.timeout_seconds, check=False,
            )
            wall_ms = (time.perf_counter() - started) * 1000
            usage, behavior = _parse_json_lines(completed.stdout)
            return AgentExecution(
                completed.returncode, completed.stdout, completed.stderr, wall_ms,
                usage=usage, behavior=behavior,
            )
        except subprocess.TimeoutExpired as error:
            stdout = _as_text(error.stdout)
            stderr = _as_text(error.stderr)
            usage, behavior = _parse_json_lines(stdout)
            return AgentExecution(
                124, stdout, stderr, (time.perf_counter() - started) * 1000,
                timed_out=True, usage=usage, behavior=behavior,
            )

    def collect_usage(self, execution: AgentExecution) -> Mapping[str, Any]:
        return dict(execution.usage)

    def cleanup(self) -> None:
        # Provider settings describe the benchmark condition and must survive
        # the per-task workspace/session cleanup performed by run_manifest().
        return None


def redacted_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Keep useful non-secret provenance while removing credential values."""

    return {key: "<redacted>" if _SECRET_NAMES.search(key) else value for key, value in environment.items()}


def _as_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _parse_json_lines(text: str) -> tuple[dict[str, int], dict[str, int]]:
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    behavior = {
        "turns": 0, "model_calls": 0, "tool_calls": 0, "shell_calls": 0,
        "file_reads": 0, "file_writes": 0, "tests": 0,
        "retries": 0, "context_compactions": 0,
    }
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        kind = str(value.get("type", ""))
        item = value.get("item") if isinstance(value.get("item"), Mapping) else {}
        part = value.get("part") if isinstance(value.get("part"), Mapping) else {}
        message = value.get("message") if isinstance(value.get("message"), Mapping) else {}
        if kind == "step_finish" and isinstance(part.get("tokens"), Mapping):
            payload = part["tokens"]
        elif (
            kind == "message_end"
            and message.get("role") == "assistant"
            and isinstance(message.get("usage"), Mapping)
        ):
            payload = message["usage"]
        else:
            payload = value.get("usage", value)
        if isinstance(payload, Mapping):
            aliases = {
                "input_tokens": ("input_tokens", "prompt_tokens", "inputTokens", "input"),
                "output_tokens": (
                    "output_tokens", "completion_tokens", "outputTokens", "output"
                ),
                "cached_input_tokens": (
                    "cached_input_tokens", "cache_read_input_tokens", "cacheRead"
                ),
            }
            for target, names in aliases.items():
                for name in names:
                    if isinstance(payload.get(name), (int, float)):
                        usage[target] += int(payload[name])
                        break
            cache = payload.get("cache")
            if (
                isinstance(cache, Mapping)
                and isinstance(cache.get("read"), (int, float))
                and not any(name in payload for name in aliases["cached_input_tokens"])
            ):
                usage["cached_input_tokens"] += int(cache["read"])
        item_kind = str(item.get("type", ""))
        completed_item = kind.endswith("completed")
        command = str(item.get("command", "")).lower()
        if completed_item and item_kind == "command_execution":
            behavior["tool_calls"] += 1
            behavior["shell_calls"] += 1
            behavior["tests"] += int(any(token in command for token in ("pytest", "npm test", "cargo test", "go test")))
        elif completed_item and item_kind == "file_change":
            behavior["tool_calls"] += 1
            behavior["file_writes"] += len(item.get("changes", ()))
        elif kind == "tool_use" and part.get("type") == "tool":
            tool = str(part.get("tool", ""))
            behavior["tool_calls"] += 1
            behavior["shell_calls"] += int(tool == "bash")
            behavior["file_reads"] += int(tool == "read")
            behavior["file_writes"] += int(tool in {"edit", "write"})
        elif kind == "message_end" and message.get("role") == "assistant":
            for content in message.get("content", ()):
                if not isinstance(content, Mapping) or content.get("type") != "toolCall":
                    continue
                tool = str(content.get("name", ""))
                behavior["tool_calls"] += 1
                behavior["shell_calls"] += int(tool == "bash")
                behavior["file_reads"] += int(tool == "read")
                behavior["file_writes"] += int(tool in {"edit", "write"})
        else:
            behavior["tool_calls"] += int("tool" in kind and ("call" in kind or "use" in kind))
        behavior["model_calls"] += int(
            kind in {"turn.completed", "step_finish", "result"}
            or (kind == "message_end" and message.get("role") == "assistant")
        )
        behavior["context_compactions"] += int("compact" in kind)
    behavior["turns"] = behavior["model_calls"]
    return usage, behavior
