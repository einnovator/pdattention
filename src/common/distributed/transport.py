"""Command transport contracts shared by experiment workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .models import WorkerConfig


@dataclass(frozen=True)
class CommandResult:
    """Completed command output independent of the underlying transport."""

    returncode: int
    stdout: str
    stderr: str


class WorkerTransport(Protocol):
    """Minimal interface required by the coordinator."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> CommandResult: ...


def transport_for(worker: WorkerConfig) -> WorkerTransport:
    """Build a transport lazily so SSH stays an optional execution path."""

    if worker.transport == "ssh":
        from .ssh import SSHTransport

        return SSHTransport(worker)
    if worker.transport == "process":
        from .process import ProcessTransport

        return ProcessTransport(worker)
    from .local import LocalTransport

    return LocalTransport(worker)
