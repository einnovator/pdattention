"""Fleet aggregation, desired-state comparison, and safe action dispatch."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping

from .clients import EngineClient, RegistryClient, ServiceClientError
from .config import ControlPlaneConfig, EngineTargetConfig
from .fleet_policy import alerts as _alerts
from .fleet_policy import compare_desired_observed
from .fleet_policy import light_metrics as _light_metrics
from .fleet_policy import match_desired
from .persistence import ControlStore


# Compatibility metadata only. Action policy is owned by ActionManager.
SAFE_ACTIONS = frozenset({
    "prefetch", "promote", "demote", "evict", "maintenance",
    "load-model", "unload-model",
})
HIGH_IMPACT_ACTIONS = frozenset({"evict", "demote", "unload-model"})
ENGINE_SECTIONS = frozenset({"summary", "capabilities", "config", "models", "sessions", "resources", "storage", "observability", "audit"})


class FleetService:
    def __init__(
        self, config: ControlPlaneConfig, store: ControlStore,
        *, engine_factory: Callable[..., EngineClient] = EngineClient,
        registry_client: RegistryClient | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.engine_factory = engine_factory
        self.registry = registry_client or (
            RegistryClient(config.registry.url, config.registry.token()) if config.registry.url else None
        )

    async def targets(self) -> list[EngineTargetConfig]:
        values: dict[str, EngineTargetConfig] = (
            {row.name: row for row in self.config.fleet.engines}
            if self.config.fleet.discovery_mode in {"static", "combined"} else {}
        )
        if self.config.fleet.discovery_mode in {"manual", "combined"}:
            for row in self.store.manual_engines():
                metadata = dict(row.get("metadata_payload") or {})
                values[row["name"]] = EngineTargetConfig(
                    name=row["name"], management_url=row["management_url"],
                    token_env=row.get("token_env"), **metadata,
                )
        if self.registry and self.config.fleet.discovery_mode in {"registry", "combined"}:
            try:
                for instance in await self.registry.instances(instance_type="ENGINE"):
                    if instance.get("management_url") and instance.get("status") != "OFFLINE":
                        name = str(instance.get("name") or instance["instance_id"])
                        values.setdefault(name, EngineTargetConfig(
                            name=name, management_url=instance["management_url"],
                            inference_url=instance.get("inference_url"),
                            environment=instance.get("environment", "unknown"),
                            region=instance.get("region", "unknown"),
                            cluster=instance.get("cluster", "unknown"),
                            namespace=instance.get("namespace", "default"),
                            host=instance.get("host"), labels=dict(instance.get("labels") or {}),
                        ))
            except ServiceClientError:
                pass
        return [values[name] for name in sorted(values)]

    async def target(self, name: str) -> EngineTargetConfig:
        for target in await self.targets():
            if target.name == name:
                return target
        raise KeyError(name)

    def client(self, target: EngineTargetConfig) -> EngineClient:
        return self.engine_factory(target.name, target.management_url, target.token())

    async def overview(self) -> dict[str, Any]:
        """Compatibility view; ControlManager uses ``collect_instances`` directly."""
        targets = await self.targets()
        desired = await self._desired()
        rows = await asyncio.gather(*(self._inspect(target, desired) for target in targets))
        return {
            "items": rows,
            "summary": {
                "total": len(rows), "healthy": sum(row["status"] == "IN_SYNC" for row in rows),
                "drift": sum(row["status"] == "DRIFT" for row in rows),
                "offline": sum(row["status"] == "OFFLINE" for row in rows),
                "unknown": sum(row["status"] == "UNKNOWN" for row in rows),
            },
        }

    async def collect_instances(self) -> list[dict[str, Any]]:
        """Collect raw backend observations without applying drift or alert policy."""
        targets = await self.targets()
        desired = await self._desired()
        return await asyncio.gather(*(self._collect(target, desired) for target in targets))

    async def _collect(self, target: EngineTargetConfig, desired_rows: list[dict[str, Any]]) -> dict[str, Any]:
        client = self.client(target)
        try:
            return {
                "target": target.model_dump(mode="json"),
                "desired": match_desired(target, desired_rows),
                "snapshot": await client.snapshot(),
            }
        except Exception as error:
            return {
                "target": target.model_dump(mode="json"), "desired": match_desired(target, desired_rows),
                "snapshot": None, "error": f"{type(error).__name__}: {error}",
            }
        finally:
            await client.close()

    async def _desired(self) -> list[dict[str, Any]]:
        if not self.registry:
            return []
        try:
            return await self.registry.deployments()
        except ServiceClientError:
            return []

    async def _inspect(self, target: EngineTargetConfig, desired_rows: list[dict[str, Any]]) -> dict[str, Any]:
        client = self.client(target)
        try:
            snapshot = await client.snapshot()
            desired = match_desired(target, desired_rows)
            drift = compare_desired_observed(desired, snapshot)
            info = snapshot.get("info", {})
            models = snapshot.get("models", {}).get("items", snapshot.get("models", []))
            model = next(
                (row for row in models if row.get("runtime_model_id") == "default"),
                next(iter(models), {}),
            )
            return {
                "name": target.name, "status": drift["status"],
                "inference_url": target.inference_url,
                "environment": target.environment, "region": target.region,
                "cluster": target.cluster, "namespace": target.namespace,
                "host": target.host or info.get("host"), "engine": info.get("engine"),
                "engine_version": info.get("engine_version"), "pra_version": info.get("pra_version"),
                "health": info.get("health", "healthy"), "model": model.get("model_id"),
                "bundle": model.get("pra_bundle_id"), "profile": model.get("profile"),
                "mode": model.get("execution_mode"), "drift": drift,
                "models": models, "model_count": len(models),
                "capabilities": snapshot.get("capabilities", {}),
                "metrics": _light_metrics(snapshot), "alerts": _alerts(snapshot, drift),
            }
        except Exception as error:
            return {
                "name": target.name, "status": "OFFLINE", "health": "offline",
                "inference_url": target.inference_url,
                "environment": target.environment, "region": target.region,
                "cluster": target.cluster, "namespace": target.namespace,
                "host": target.host, "error": f"{type(error).__name__}: {error}",
                "drift": {"status": "OFFLINE", "differences": []}, "metrics": {},
                "alerts": ["engine offline"],
            }
        finally:
            await client.close()

    async def engine_section(self, name: str, section: str) -> Any:
        if section not in ENGINE_SECTIONS:
            raise KeyError(section)
        target = await self.target(name)
        client = self.client(target)
        try:
            return await client.endpoint(section)
        finally:
            await client.close()

    async def action(self, name: str, action: str, body: Mapping[str, Any]) -> Any:
        target = await self.target(name)
        client = self.client(target)
        try:
            return await client.action(action, body)
        finally:
            await client.close()

    async def patch_config(self, name: str, body: Mapping[str, Any]) -> Any:
        target = await self.target(name)
        client = self.client(target)
        try:
            return await client.patch_config(body)
        finally:
            await client.close()

    async def close(self) -> None:
        if self.registry:
            await self.registry.close()
