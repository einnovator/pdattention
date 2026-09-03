"""Control Plane discovery client and unified inference-target inventory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from .agent_config import ControlPlaneClientConfig, ProviderConfig


@dataclass(frozen=True)
class InferenceTarget:
    target_id: str
    engine_instance: str
    runtime_model_id: str
    model_id: str
    provider_type: str
    endpoint: str | None = None
    credentials_ref: str | None = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    qualification: str | None = None
    recommendation: Mapping[str, Any] = field(default_factory=dict)
    status: str = "UNKNOWN"
    source: str = "static"


class AmbiguousTargetError(LookupError):
    pass


class ControlPlaneClient:
    """Small REST client kept separate from the Control Plane MCP surface."""

    def __init__(self, config: ControlPlaneClientConfig) -> None:
        self.config = config
        self.status = "DISCONNECTED"
        self.error: str | None = None
        self._fleet: tuple[dict[str, Any], ...] = ()
        self._fetched_at = 0.0

    async def _get(self, path: str, **params: Any) -> Any:
        if not self.config.url:
            raise ValueError("Control Plane URL is required when enabled.")
        import httpx
        certificate = (
            (self.config.auth.cert_file, self.config.auth.key_file)
            if self.config.auth.cert_file and self.config.auth.key_file
            else self.config.auth.cert_file
        )
        async with httpx.AsyncClient(
            base_url=self.config.url, headers=self.config.auth.resolved_headers(),
            timeout=self.config.timeout_seconds, cert=certificate,
        ) as client:
            response = await client.get(path, params={k: v for k, v in params.items() if v is not None})
            response.raise_for_status()
            return response.json()

    async def list_engines(self, *, refresh: bool = False) -> tuple[dict[str, Any], ...]:
        if not refresh and self._fleet and time.monotonic() - self._fetched_at < self.config.cache_ttl_seconds:
            return self._fleet
        try:
            payload = await self._get("/api/fleet")
            self._fleet = tuple(payload.get("items", payload) if isinstance(payload, dict) else payload)
            self._fetched_at, self.status, self.error = time.monotonic(), "CONNECTED", None
        except Exception as error:
            self.status, self.error = "DEGRADED", f"{type(error).__name__}: {error}"
            if self.config.required:
                raise
        return self._fleet

    async def inspect_engine(self, name: str, section: str = "summary") -> Any:
        return await self._get(f"/api/engines/{name}/{section}")

    async def list_models(self, *, refresh: bool = False) -> tuple[dict[str, Any], ...]:
        engines = await self.list_engines(refresh=refresh)
        return tuple(dict(model, engine_instance=row.get("name")) for row in engines for model in row.get("models", ()))

    async def find_models(self, query: str = "", *, engine: str | None = None) -> tuple[dict[str, Any], ...]:
        needle = query.casefold()
        return tuple(row for row in await self.list_models()
                     if (not engine or str(row.get("engine_instance", "")).casefold() == engine.casefold())
                     and (not needle or needle in str(row).casefold()))

    async def qualifications(self, *, model: str | None = None, engine: str | None = None) -> tuple[dict[str, Any], ...]:
        value = await self._get("/api/qualifications", model=model, engine=engine)
        return tuple(value.get("items", ()))

    async def recommendations(self) -> tuple[dict[str, Any], ...]:
        value = await self._get("/api/recommendations")
        return tuple(value.get("items", ()))

    async def targets(self) -> tuple[InferenceTarget, ...]:
        rows = []
        for engine in await self.list_engines():
            for model in engine.get("models", ()):
                instance = str(engine.get("name"))
                runtime_id = str(model.get("runtime_model_id", "default"))
                rows.append(InferenceTarget(
                    f"{instance}/{runtime_id}", instance, runtime_id,
                    str(model.get("model_id") or model.get("model") or runtime_id),
                    str(model.get("provider") or engine.get("engine") or "openai"),
                    model.get("endpoint") or engine.get("endpoint") or engine.get("management_url"),
                    capabilities={**engine.get("capabilities", {}), **model.get("capabilities", {})},
                    qualification=model.get("qualification"), status=str(engine.get("status", "UNKNOWN")),
                    source="control-plane",
                ))
        return tuple(rows)


class InferenceTargetManager:
    """Merge static and discovered targets and enforce unambiguous selection."""

    def __init__(self, providers: Mapping[str, ProviderConfig] | None = None,
                 control_plane: ControlPlaneClient | None = None) -> None:
        self.providers = dict(providers or {})
        self.control_plane = control_plane
        self.active: InferenceTarget | None = None

    def static_targets(self) -> tuple[InferenceTarget, ...]:
        return tuple(InferenceTarget(
            f"{provider.engine_instance or name}/{provider.runtime_model_id}",
            provider.engine_instance or name, provider.runtime_model_id,
            provider.model or provider.runtime_model_id, provider.type, provider.base_url,
            provider.api_key_env, metadata_to_capabilities(provider.metadata), source="static",
        ) for name, provider in self.providers.items() if provider.enabled)

    async def list(self, *, refresh: bool = False) -> tuple[InferenceTarget, ...]:
        merged = {row.target_id: row for row in self.static_targets()}
        if self.control_plane:
            if refresh:
                await self.control_plane.list_engines(refresh=True)
            for row in await self.control_plane.targets():
                prior = merged.get(row.target_id)
                merged[row.target_id] = row if prior is None else replace(
                    row, endpoint=prior.endpoint or row.endpoint,
                    credentials_ref=prior.credentials_ref or row.credentials_ref,
                    capabilities={**row.capabilities, **prior.capabilities}, source="static+control-plane",
                )
        return tuple(sorted(merged.values(), key=lambda row: row.target_id))

    async def resolve(self, value: str) -> InferenceTarget:
        targets = await self.list()
        exact = [row for row in targets if row.target_id.casefold() == value.casefold()]
        matches = exact or [row for row in targets if value.casefold() in {
            row.runtime_model_id.casefold(), row.model_id.casefold()
        }]
        if not matches:
            raise LookupError(f"Unknown inference target: {value}")
        if len(matches) > 1:
            raise AmbiguousTargetError(f"Ambiguous target {value!r}: {', '.join(row.target_id for row in matches)}")
        return matches[0]

    async def switch(self, value: str) -> InferenceTarget:
        self.active = await self.resolve(value)
        return self.active


def metadata_to_capabilities(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return dict(value.get("capabilities", value))
