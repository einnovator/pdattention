"""Fleet aggregation, desired-state comparison, and safe action dispatch."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping

from .clients import EngineClient, RegistryClient, ServiceClientError
from .config import ControlPlaneConfig, EngineTargetConfig
from .persistence import ControlStore


SAFE_ACTIONS = frozenset({"prefetch", "promote", "demote", "evict", "maintenance"})
HIGH_IMPACT_ACTIONS = frozenset({"evict", "demote"})
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
        values: dict[str, EngineTargetConfig] = {row.name: row for row in self.config.fleet.engines}
        if self.config.fleet.discovery_mode in {"manual", "combined"}:
            for row in self.store.manual_engines():
                metadata = dict(row.get("metadata_payload") or {})
                values[row["name"]] = EngineTargetConfig(
                    name=row["name"], management_url=row["management_url"],
                    token_env=row.get("token_env"), **metadata,
                )
        if self.registry and self.config.fleet.discovery_mode in {"registry", "combined"}:
            try:
                for deployment in await self.registry.deployments():
                    selector = dict(deployment.get("engine_instance_selector") or {})
                    if selector.get("management_url"):
                        name = str(selector.get("name") or deployment["id"])
                        values.setdefault(name, EngineTargetConfig(
                            name=name, management_url=selector["management_url"],
                            token_env=selector.get("token_env"), environment=deployment.get("environment", "unknown"),
                            cluster=deployment.get("cluster", "unknown"), labels=dict(selector.get("labels") or {}),
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
            desired = self._match_desired(target, desired_rows)
            drift = compare_desired_observed(desired, snapshot)
            info = snapshot.get("info", {})
            models = snapshot.get("models", {}).get("items", snapshot.get("models", []))
            model = models[0] if models else {}
            return {
                "name": target.name, "status": drift["status"],
                "environment": target.environment, "region": target.region,
                "cluster": target.cluster, "namespace": target.namespace,
                "host": target.host or info.get("host"), "engine": info.get("engine"),
                "engine_version": info.get("engine_version"), "pra_version": info.get("pra_version"),
                "health": info.get("health", "healthy"), "model": model.get("model_id"),
                "bundle": model.get("pra_bundle_id"), "profile": model.get("profile"),
                "mode": model.get("execution_mode"), "drift": drift,
                "metrics": _light_metrics(snapshot), "alerts": _alerts(snapshot, drift),
            }
        except Exception as error:
            return {
                "name": target.name, "status": "OFFLINE", "health": "offline",
                "environment": target.environment, "region": target.region,
                "cluster": target.cluster, "namespace": target.namespace,
                "host": target.host, "error": f"{type(error).__name__}: {error}",
                "drift": {"status": "OFFLINE", "differences": []}, "metrics": {},
                "alerts": ["engine offline"],
            }
        finally:
            await client.close()

    @staticmethod
    def _match_desired(target: EngineTargetConfig, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        matches = []
        for row in rows:
            if row.get("environment") != target.environment or row.get("cluster") != target.cluster:
                continue
            selector = dict(row.get("engine_instance_selector") or {})
            if selector.get("name") and selector["name"] != target.name:
                continue
            if selector.get("host") and selector["host"] != target.host:
                continue
            if selector.get("namespace") and selector["namespace"] != target.namespace:
                continue
            labels = dict(selector.get("labels") or {})
            if any(target.labels.get(key) != value for key, value in labels.items()):
                continue
            matches.append(row)
        return sorted(matches, key=lambda row: (-int(row.get("desired_revision", 0)), str(row.get("id"))))[0] if matches else None

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
        if action not in SAFE_ACTIONS:
            raise ValueError(f"unsupported safe action: {action}")
        if action in HIGH_IMPACT_ACTIONS and not body.get("confirmed"):
            raise ValueError(f"{action} requires confirmed=true")
        target = await self.target(name)
        client = self.client(target)
        try:
            payload = {key: value for key, value in body.items() if key != "confirmed"}
            return await client.action(action, payload)
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


def compare_desired_observed(desired: Mapping[str, Any] | None, snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    if snapshot is None:
        return {"status": "OFFLINE", "differences": []}
    if desired is None:
        return {"status": "UNKNOWN", "differences": []}
    models = snapshot.get("models", {}).get("items", snapshot.get("models", []))
    observed = models[0] if models else {}
    pairs = {
        "model": (desired.get("desired_model_id"), observed.get("model_id")),
        "bundle": (desired.get("desired_bundle_id"), observed.get("pra_bundle_id")),
        "profile": (desired.get("desired_profile_id"), observed.get("profile")),
        "mode": (desired.get("desired_mode"), observed.get("execution_mode")),
    }
    differences = [
        {"field": field, "desired": expected, "observed": actual}
        for field, (expected, actual) in pairs.items()
        if expected is not None and expected != actual
    ]
    return {"status": "DRIFT" if differences else "IN_SYNC", "differences": differences, "desired_revision": desired.get("desired_revision")}


def _light_metrics(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    storage = snapshot.get("storage", {})
    metrics = storage.get("metrics", {})
    return {
        "selected_full_token_ratio": metrics.get("selected_full_token_ratio"),
        "visible_reuse": metrics.get("visible_reuse"), "native_reuse": metrics.get("native_reuse"),
        "storage_reloads": metrics.get("reloads"), "request_rate": metrics.get("request_rate"),
        "ttft_p95_ms": metrics.get("ttft_p95_ms"), "error_rate": metrics.get("error_rate"),
    }


def _alerts(snapshot: Mapping[str, Any], drift: Mapping[str, Any]) -> list[str]:
    alerts = []
    if drift["status"] == "DRIFT":
        alerts.append("desired state drift")
    metrics = snapshot.get("storage", {}).get("metrics", {})
    if float(metrics.get("reload_rate", 0) or 0) > 0.25:
        alerts.append("high storage reload rate")
    return alerts
