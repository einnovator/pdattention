"""Remote Ollama runtime provider for CLI discovery and diagnostics."""

from __future__ import annotations

from typing import Any

from pra_hf.deployment import PRAEngineCapabilities
from pra_hf.engine_profiles import EngineType, PrefixCacheMode
from pra_hf.runtime_providers import RuntimeConfig, RuntimeDoctorReport, RuntimeProvider

from .adapter import OllamaEngineAdapter, OllamaLlamaCppBackendExecutor


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
            )
        )
        native_url = config.engine_options.get("native_backend_url")
        artifact_digest = config.engine_options.get("model_artifact_digest")
        backend_revision = config.engine_options.get("backend_revision")
        if native_url and artifact_digest and backend_revision:
            try:
                backend = OllamaLlamaCppBackendExecutor(
                    str(native_url),
                    backend_revision=str(backend_revision),
                    model_artifact_digest=str(artifact_digest),
                    resource_slot=int(config.engine_options.get("resource_slot", 0)),
                    request_slot=int(config.engine_options.get("request_slot", 1)),
                )
                checks.append(
                    {
                        "name": "native-pra",
                        "status": "READY",
                        "detail": (
                            f"{backend.native.protocol}; artifact "
                            f"{str(artifact_digest)[:12]}...; AUTO may negotiate E2."
                        ),
                    }
                )
            except Exception as exc:
                checks.append(
                    {"name": "native-pra", "status": "UNREACHABLE", "detail": str(exc)}
                )
        else:
            checks.append(
                {
                    "name": "native-pra",
                    "status": "FALLBACK",
                    "detail": (
                        "Configure native_backend_url, backend_revision, and "
                        "model_artifact_digest to negotiate E2; AUTO remains E0."
                    ),
                }
            )
        return RuntimeDoctorReport(self.engine_type, "AVAILABLE", tuple(checks))
