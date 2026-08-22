"""Same-host command transport."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .models import WorkerConfig
from .transport import CommandResult


class LocalTransport:
    """Run commands synchronously on the coordinator host."""

    def __init__(self, worker: WorkerConfig):
        self.worker = worker

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> CommandResult:
        process_env = os.environ.copy()
        process_env.update(self.worker.env)
        process_env.update(env or {})
        completed = subprocess.run(
            list(command),
            cwd=str(cwd or self.worker.workdir or Path.cwd()),
            env=process_env,
            text=True,
            capture_output=True,
            timeout=timeout or self.worker.timeout_seconds,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)
