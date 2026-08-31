"""Managed and remote llama-server provider for the PRA runtime SDK."""

from __future__ import annotations

import shutil
from typing import Any

from pra_hf.deployment import PRAEngineCapabilities
from pra_hf.engine_profiles import EngineType, PrefixCacheMode
from pra_hf.runtime_providers import RuntimeConfig, RuntimeDoctorReport, RuntimeProvider


class LlamaCppRuntimeProvider(RuntimeProvider):
    engine_type = "llama_cpp"
    launch_mode = "both"

    def _executable(self, config: RuntimeConfig) -> str | None:
        configured = config.engine_options.get("executable")
        return str(configured) if configured else shutil.which("llama-server")

    def capabilities(self, config: RuntimeConfig) -> PRAEngineCapabilities:
        del config
        return PRAEngineCapabilities(
            adapter="llama_cpp_http",
            engine_type=EngineType.LLAMA_CPP,
            integration_level="E0",
            prefix_cache_mode=PrefixCacheMode.EXPLICIT_PREFIX_HANDLE,
            automatic_prefix_cache=True,
            explicit_prefix_cache=True,
            prefix_cache_handle=True,
            session_state=True,
            cache_affinity=True,
            text_fallback=True,
            streaming=False,
        )

    def doctor(self, config: RuntimeConfig) -> RuntimeDoctorReport:
        executable = self._executable(config)
        checks: list[dict[str, Any]] = [
            {"name": "provider", "status": "READY", "detail": type(self).__name__},
            {
                "name": "llama-server",
                "status": "READY" if executable else "NOT_INSTALLED",
                "detail": executable or "Set engine_options.executable or install llama-server.",
            },
            {
                "name": "native-pra",
                "status": "NOT_INSTALLED",
                "detail": "Upstream slot state is sequential prefix state, not detached E2 K/V.",
            },
        ]
        return RuntimeDoctorReport(
            self.engine_type,
            "AVAILABLE" if executable else "NOT_INSTALLED",
            tuple(checks),
        )

    def build_command(self, config: RuntimeConfig) -> list[str]:
        executable = self._executable(config)
        if executable is None:
            raise RuntimeError("llama-server executable was not found.")
        if not config.model:
            raise ValueError("A GGUF model path or -hf repository is required.")
        command = [
            executable,
            "--model",
            config.model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--alias",
            str(config.engine_options.get("alias", config.model)),
            "--slots",
        ]
        reserved = {"executable", "alias"}
        for key, value in config.engine_options.items():
            if key in reserved:
                continue
            option = "--" + str(key).replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(option)
            else:
                command.extend((option, str(value)))
        return command
