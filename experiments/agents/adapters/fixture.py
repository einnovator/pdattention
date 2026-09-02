"""Deterministic adapter that validates orchestration without claiming model quality."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from .base import AgentExecution, AgentTask, CodingAgentAdapter


class FixtureAgentAdapter(CodingAgentAdapter):
    def __init__(self) -> None:
        self.workspace = Path.cwd()
        self.provider: dict[str, Any] = {}

    def install_or_verify(self) -> Mapping[str, Any]:
        return {"installed": True, "executable": "in-process", "version": self.version()}

    def version(self) -> str:
        return "fixture-1"

    def configure_provider(self, **settings: Any) -> None:
        self.provider = dict(settings)

    def configure_workspace(self, workspace: Path) -> None:
        self.workspace = workspace

    def run_task(self, task: AgentTask) -> AgentExecution:
        started = time.perf_counter()
        target = task.workspace / str(task.metadata.get("file", "answer.txt"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(task.metadata.get("content", task.task_id)), encoding="utf-8")
        input_tokens = len(task.instruction.split())
        return AgentExecution(
            0,
            '{"type":"result","usage":{"input_tokens":%d,"output_tokens":2}}\n' % input_tokens,
            "",
            (time.perf_counter() - started) * 1000,
            usage={"input_tokens": input_tokens, "output_tokens": 2, "cached_input_tokens": 0},
            behavior={"turns": 1, "model_calls": 1, "tool_calls": 1, "file_writes": 1},
        )

    def collect_usage(self, execution: AgentExecution) -> Mapping[str, Any]:
        return execution.usage

    def cleanup(self) -> None:
        self.provider.clear()
