"""Common contract between agent harnesses and benchmark orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    instruction: str
    workspace: Path
    timeout_seconds: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentExecution:
    exit_code: int
    stdout: str
    stderr: str
    wall_ms: float
    timed_out: bool = False
    usage: Mapping[str, Any] = field(default_factory=dict)
    behavior: Mapping[str, int] = field(default_factory=dict)


class CodingAgentAdapter(ABC):
    """Lifecycle required for deterministic, unattended coding-agent runs."""

    @abstractmethod
    def install_or_verify(self) -> Mapping[str, Any]:
        """Verify an exact agent installation without silently upgrading it."""

    @abstractmethod
    def version(self) -> str:
        """Return the executable version used by result provenance."""

    @abstractmethod
    def configure_provider(self, **settings: Any) -> None:
        """Apply endpoint/model settings while keeping credentials out of artifacts."""

    @abstractmethod
    def configure_workspace(self, workspace: Path) -> None:
        """Prepare one isolated benchmark checkout."""

    @abstractmethod
    def run_task(self, task: AgentTask) -> AgentExecution:
        """Run one task noninteractively and return raw evidence plus usage."""

    @abstractmethod
    def collect_usage(self, execution: AgentExecution) -> Mapping[str, Any]:
        """Normalize provider- or harness-reported usage counters."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release process/session state after every task."""
