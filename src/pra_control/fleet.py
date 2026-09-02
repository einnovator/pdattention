"""Fleet aggregation, desired-state comparison, and safe action dispatch."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping

from .clients import EngineClient, RegistryClient, ServiceClientError
from .config import ControlPlaneConfig, EngineTargetConfig
from .persistence import ControlStore


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
            model = next(
                (row for row in models if row.get("runtime_model_id") == "default"),
                next(iter(models), {}),
            )
            return {
                "name": target.name, "status": drift["status"],
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
    desired_models = desired.get("desired_models")
    legacy = not isinstance(desired_models, list) or not desired_models
    if legacy:
        desired_models = [{
            "runtime_model_id": "default",
            "model_id": desired.get("desired_model_id"),
            "bundle_id": desired.get("desired_bundle_id"),
            "profile_id": desired.get("desired_profile_id"),
            "mode": desired.get("desired_mode"),
        }]
    observed_by_id = {
        str(row.get("runtime_model_id") or ("default" if len(models) == 1 else row.get("model_id"))): row
        for row in models
    }
    differences: list[dict[str, Any]] = []
    per_model: dict[str, Any] = {}
    expected_ids: set[str] = set()
    for expected_model in desired_models:
        runtime_id = str(expected_model.get("runtime_model_id") or "default")
        expected_ids.add(runtime_id)
        observed = observed_by_id.get(runtime_id, {})
        pairs = {
            "model": (expected_model.get("model_id"), observed.get("model_id")),
            "bundle": (expected_model.get("bundle_id"), observed.get("pra_bundle_id")),
            "profile": (expected_model.get("profile_id"), observed.get("profile")),
            "mode": (expected_model.get("mode"), observed.get("execution_mode")),
        }
        model_differences = [
            {
                "field": field,
                "desired": expected,
                "observed": actual,
                **({} if legacy else {"runtime_model_id": runtime_id}),
            }
            for field, (expected, actual) in pairs.items()
            if expected is not None and expected != actual
        ]
        if not observed:
            model_differences.insert(0, {
                "field": "MODEL_NOT_LOADED", "desired": runtime_id, "observed": None,
                **({} if legacy else {"runtime_model_id": runtime_id}),
            })
        per_model[runtime_id] = {
            "status": "DRIFT" if model_differences else "IN_SYNC",
            "differences": model_differences,
        }
        differences.extend(model_differences)
    if not bool(desired.get("allow_extra_models", True)):
        for runtime_id in observed_by_id.keys() - expected_ids:
            difference = {
                "field": "UNAPPROVED_MODEL_LOADED", "desired": None,
                "observed": runtime_id, "runtime_model_id": runtime_id,
            }
            differences.append(difference)
            per_model[runtime_id] = {"status": "DRIFT", "differences": [difference]}
    return {
        "status": "DRIFT" if differences else "IN_SYNC",
        "differences": differences,
        "models": per_model,
        "desired_revision": desired.get("desired_revision"),
    }


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
