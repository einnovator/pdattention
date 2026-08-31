"""Remote Ollama runtime provider for CLI discovery and diagnostics."""

from __future__ import annotations

from typing import Any

from pra_hf.deployment import PRAEngineCapabilities
from pra_hf.engine_profiles import EngineType, PrefixCacheMode
from pra_hf.runtime_providers import RuntimeConfig, RuntimeDoctorReport, RuntimeProvider

from .adapter import OllamaEngineAdapter


class OllamaRuntimeProvider(RuntimeProvider):
    """Connect to an Ollama daemon; model process lifecycle remains Ollama-owned."""

    engine_type = "ollama"
    launch_mode = "connect_remote"

    def capabilities(self, config: RuntimeConfig) -> PRAEngineCapabilities:
        del config
        return PRAEngineCapabilities(
            adapter="ollama_gateway",
            engine_type=EngineType.OLLAMA,
            integration_level="E0",
            prefix_cache_mode=PrefixCacheMode.SESSION_STATE,
            session_state=True,
            cache_affinity=True,
            text_fallback=True,
            streaming=False,
        )

    def doctor(self, config: RuntimeConfig) -> RuntimeDoctorReport:
        checks: list[dict[str, Any]] = [
            {"name": "provider", "status": "READY", "detail": type(self).__name__}
        ]
        try:
            info = OllamaEngineAdapter(config.resolved_endpoint).inspect_endpoint()
        except Exception as exc:
            checks.append({"name": "endpoint", "status": "UNREACHABLE", "detail": str(exc)})
            return RuntimeDoctorReport(self.engine_type, "MISCONFIGURED", tuple(checks))
        checks.extend(
            (
                {"name": "endpoint", "status": "READY", "detail": info.version},
                {
                    "name": "models",
                    "status": "READY" if info.installed_models else "EMPTY",
                    "detail": ", ".join(info.installed_models),
                },
                {
                    "name": "native-pra",
                    "status": "FALLBACK",
                    "detail": "No backend PRA handshake; AUTO remains E0.",
                },
            )
        )
        return RuntimeDoctorReport(self.engine_type, "AVAILABLE", tuple(checks))
