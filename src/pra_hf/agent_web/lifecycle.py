"""Safe foreground/detached lifecycle for the experimental agent web UI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..product_config import pra_home


@dataclass(frozen=True)
class WebServerState:
    pid: int
    host: str
    port: int
    start_time: float
    process_create_time: float | None
    profile: str | None
    pra_override: str | None
    config_path: str | None
    log_path: str
    command: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "command": list(self.command), "url": self.url}


class AgentWebLifecycle:
    """Start and stop the optional UI while guarding against PID reuse."""

    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(home) if home else pra_home()
        self.state_path = self.home / "agent-web.json"
        self.pid_path = self.home / "agent-web.pid"

    def command(
        self,
        *,
        host: str,
        port: int,
        profile: str | None,
        pra_override: str | None,
        config_path: str | None,
    ) -> list[str]:
        value = [
            sys.executable,
            "-m",
            "pra_hf.agent_web.server",
            "--host",
            host,
            "--port",
            str(port),
        ]
        if profile:
            value += ["--profile", profile]
        if pra_override:
            value += ["--pra", pra_override]
        if config_path:
            value += ["--config", config_path]
        return value

    def start(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        profile: str | None = None,
        pra_override: str | None = None,
        config_path: str | None = None,
        detach: bool = False,
        open_browser: bool = False,
    ) -> WebServerState | None:
        if host not in {"127.0.0.1", "localhost", "::1"}:
            import warnings

            warnings.warn("The experimental PRA web UI is exposed beyond loopback and has no production authentication.")
        command = self.command(
            host=host,
            port=port,
            profile=profile,
            pra_override=pra_override,
            config_path=config_path,
        )
        if not detach:
            if open_browser:
                webbrowser.open(f"http://{host}:{port}")
            from .server import run_server

            run_server(
                host=host,
                port=port,
                profile=profile,
                pra_override=pra_override,
                config_path=config_path,
            )
            return None
        self.home.mkdir(parents=True, exist_ok=True)
        log_path = self.home / "agent-web.log"
        log = log_path.open("ab")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            process = subprocess.Popen(command, stdout=log, stderr=log, creationflags=creationflags)
        finally:
            log.close()
        try:
            import psutil

            create_time = psutil.Process(process.pid).create_time()
        except ImportError:
            create_time = None
        state = WebServerState(
            process.pid,
            host,
            port,
            time.time(),
            create_time,
            profile,
            pra_override,
            config_path,
            str(log_path),
            tuple(command),
        )
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        self.pid_path.write_text(str(process.pid), encoding="ascii")
        self._wait_ready(state)
        if open_browser:
            webbrowser.open(state.url)
        return state

    def read(self) -> WebServerState | None:
        if not self.state_path.is_file():
            return None
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        value.pop("url", None)
        value.setdefault("pra_override", None)
        value["command"] = tuple(value.get("command", ()))
        return WebServerState(**value)

    def stop(self) -> str:
        state = self.read()
        if state is None:
            self._clean()
            return "NOT_RUNNING"
        try:
            import psutil
        except ImportError as error:
            raise RuntimeError("Safe detached web shutdown requires the 'web' extra with psutil.") from error
        try:
            process = psutil.Process(state.pid)
        except psutil.NoSuchProcess:
            self._clean()
            return "STALE_STATE_CLEANED"
        if state.process_create_time is None or abs(process.create_time() - state.process_create_time) > 0.01:
            self._clean()
            return "STALE_STATE_CLEANED"
        current = tuple(process.cmdline())
        if "pra_hf.agent_web.server" not in " ".join(current):
            raise RuntimeError("PID belongs to another process; state was retained for inspection.")
        process.terminate()
        try:
            process.wait(timeout=10)
        except psutil.TimeoutExpired:
            process.kill()
        self._clean()
        return "STOPPED"

    def _wait_ready(self, state: WebServerState) -> None:
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(state.url + "/health", timeout=1) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError(f"PRA Agent Web UI did not become ready; inspect {state.log_path}")

    def _clean(self) -> None:
        self.state_path.unlink(missing_ok=True)
        self.pid_path.unlink(missing_ok=True)
