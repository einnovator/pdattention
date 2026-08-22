"""OpenSSH-based remote worker transport without an extra Python dependency."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .models import WorkerConfig
from .transport import CommandResult


class SSHTransport:
    """Execute one safely quoted command through the system OpenSSH client."""

    def __init__(self, worker: WorkerConfig):
        self.worker = worker

    def _destination(self) -> str:
        return f"{self.worker.user}@{self.worker.host}" if self.worker.user else self.worker.host

    def _ssh_options(self) -> list[str]:
        options = ["-o", "BatchMode=yes"]
        if self.worker.port:
            options.extend(["-p", str(self.worker.port)])
        if self.worker.ssh_identity_file:
            options.extend(["-i", self.worker.ssh_identity_file])
        return options

    def _scp_options(self) -> list[str]:
        options = ["-o", "BatchMode=yes"]
        if self.worker.port:
            options.extend(["-P", str(self.worker.port)])
        if self.worker.ssh_identity_file:
            options.extend(["-i", self.worker.ssh_identity_file])
        return options

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> CommandResult:
        ssh = ["ssh", *self._ssh_options()]
        remote_env = {**self.worker.env, **(env or {})}
        pieces = [f"{key}={shlex.quote(value)}" for key, value in remote_env.items()]
        workdir = str(cwd or self.worker.workdir or "")
        if workdir:
            pieces.extend(["cd", shlex.quote(workdir), "&&"])
        if self.worker.setup_command:
            pieces.extend([self.worker.setup_command, "&&"])
        pieces.extend(shlex.quote(str(item)) for item in command)
        ssh.extend([self._destination(), " ".join(pieces)])
        completed = subprocess.run(
            ssh,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            timeout=timeout or self.worker.timeout_seconds,
            check=False,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    def put(self, local_path: str | Path, remote_path: str) -> None:
        """Copy one coordinator file to an already-created remote directory."""

        command = [
            "scp",
            *self._scp_options(),
            str(Path(local_path).resolve()),
            f"{self._destination()}:{shlex.quote(remote_path)}",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"SCP upload failed: {completed.stderr.strip()}")

    def get_tree(self, remote_path: str, local_path: str | Path) -> None:
        """Recursively collect one remote trial directory."""

        target = Path(local_path)
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "scp",
            *self._scp_options(),
            "-r",
            f"{self._destination()}:{shlex.quote(remote_path.rstrip('/') + '/.')}" ,
            str(target.resolve()),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(f"SCP download failed: {completed.stderr.strip()}")
