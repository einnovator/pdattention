"""Compilers from canonical PRA routes to external router configurations.

Adapters intentionally compile only deployment/pool intent. LiteLLM,
agentgateway, Kubernetes endpoint pickers, Bifrost, or the reference router
retain ownership of request transport and replica selection.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import yaml

from .controller import ReconcilePlan, RouterDesiredState, stable_digest


FORWARDED_HEADERS = (
    "traceparent", "tracestate", "baggage", "x-pra-tenant-id",
    "x-pra-session-id", "x-pra-mode", "x-pra-profile",
    "x-pra-route-id", "x-pra-experiment-id",
)


def stable_backend_name(router_id: str, backend: dict[str, Any]) -> str:
    """Build a DNS/config-safe name from immutable PRA endpoint identity."""

    identity = ":".join((
        router_id,
        str(backend.get("engine_instance_id") or backend["id"]),
        str(backend.get("runtime_model_id") or "default"),
    ))
    slug = re.sub(r"[^a-z0-9-]+", "-", identity.casefold()).strip("-")
    return f"pra-{slug[:48]}-{stable_digest(identity)[:8]}"


def _route_backends(route: dict[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pool in route.get("pools", []):
        for backend in pool.get("backends", []):
            if backend["id"] not in seen:
                values.append({**backend, "pool_id": pool["id"], "fallback": bool(pool.get("fallback"))})
                seen.add(backend["id"])
    return values


def _public_metadata(route: dict[str, Any], backend: dict[str, Any]) -> dict[str, Any]:
    return {
        "pra_route_id": route["id"],
        "pra_pool_id": backend["pool_id"],
        "pra_backend_id": backend["id"],
        "pra_mode": list(backend.get("modes") or []),
        "pra_profile": backend.get("profile"),
        "bundle_id": backend.get("bundle_id"),
        "bundle_revision": backend.get("bundle_revision"),
        "qualification_tier": backend.get("qualification_tier"),
        "model_revision": backend.get("model_revision"),
        "engine": backend.get("engine"),
        "engine_version": backend.get("engine_version"),
        "hardware": backend.get("metadata", {}).get("hardware"),
        "region": backend.get("region"),
    }


class RouterTransport(Protocol):
    async def read(self, router: dict[str, Any]) -> dict[str, Any]: ...

    async def apply(
        self, router: dict[str, Any], config: dict[str, Any], plan: ReconcilePlan,
    ) -> None: ...


class MemoryRouterTransport:
    """Deterministic transport for tests and embedded controller deployments."""

    def __init__(self) -> None:
        self.states: dict[str, dict[str, Any]] = {}

    async def read(self, router: dict[str, Any]) -> dict[str, Any]:
        return self.states.get(router["id"], {"revision": 0, "config": {}})

    async def apply(self, router: dict[str, Any], config: dict[str, Any], plan: ReconcilePlan) -> None:
        self.states[router["id"]] = {"revision": plan.desired_revision, "config": config}


class HTTPRouterTransport:
    """JSON management transport using a secret named by Registry metadata."""

    def __init__(self, *, read_path: str = "/v1/router/config", apply_path: str = "/v1/router/config") -> None:
        self.read_path = read_path
        self.apply_path = apply_path

    async def read(self, router: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._request, router, "GET", self.read_path, None)

    async def apply(self, router: dict[str, Any], config: dict[str, Any], plan: ReconcilePlan) -> None:
        await asyncio.to_thread(self._request, router, "PUT", self.apply_path, {
            "revision": plan.desired_revision, "config": config,
        })

    @staticmethod
    def _request(
        router: dict[str, Any], method: str, path: str, body: dict[str, Any] | None,
    ) -> dict[str, Any]:
        base = str(router["management_url"]).rstrip("/")
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        secret_ref = router.get("credential_reference")
        token = os.environ.get(str(secret_ref)) if secret_ref else None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(f"{base}{path}", data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"router management API returned HTTP {error.code}") from error


class AtomicFileTransport:
    """Atomic last-good configuration for agentgateway file-watch mode."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.revision_path = self.path.with_suffix(self.path.suffix + ".pra-revision")

    async def read(self, router: dict[str, Any]) -> dict[str, Any]:
        if not self.path.is_file():
            return {"revision": 0, "config": {}}
        value = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        revision = int(self.revision_path.read_text(encoding="ascii")) if self.revision_path.is_file() else 0
        return {"revision": revision, "config": value}

    async def apply(self, router: dict[str, Any], config: dict[str, Any], plan: ReconcilePlan) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        os.replace(temporary, self.path)
        revision_temporary = self.revision_path.with_suffix(self.revision_path.suffix + ".tmp")
        revision_temporary.write_text(str(plan.desired_revision), encoding="ascii")
        os.replace(revision_temporary, self.revision_path)


class AutoRouterTransport:
    """Select HTTP management or atomic file hot reload per router instance."""

    async def read(self, router: dict[str, Any]) -> dict[str, Any]:
        transport = self._transport(router)
        return await transport.read(router)

    async def apply(self, router: dict[str, Any], config: dict[str, Any], plan: ReconcilePlan) -> None:
        transport = self._transport(router)
        await transport.apply(router, config, plan)

    @staticmethod
    def _transport(router: dict[str, Any]) -> RouterTransport:
        url = str(router["management_url"])
        if url.startswith("file://"):
            return AtomicFileTransport(url.removeprefix("file://"))
        return HTTPRouterTransport()


class BaseRouterAdapter:
    kind = "base"
    features: tuple[str, ...] = ("full-replace", "last-good")

    def __init__(self, transport: RouterTransport | None = None) -> None:
        self.transport = transport or AutoRouterTransport()

    async def discover_capabilities(self) -> list[str]:
        return list(self.features)

    async def read_observed(self, desired: RouterDesiredState) -> dict[str, Any]:
        return await self.transport.read(desired.router)

    async def apply(self, desired: RouterDesiredState, compiled: dict[str, Any], plan: ReconcilePlan) -> None:
        await self.transport.apply(desired.router, compiled, plan)

    async def verify(self, desired: RouterDesiredState, compiled: dict[str, Any]) -> bool:
        observed = await self.read_observed(desired)
        return (
            int(observed.get("revision", -1)) == desired.desired_revision
            and stable_digest(observed.get("config", observed)) == stable_digest(compiled)
        )


class LiteLLMRouterAdapter(BaseRouterAdapter):
    kind = "litellm"
    features = ("openai", "streaming", "weighted", "fallback", "dynamic-config", "last-good")

    def compile(self, desired: RouterDesiredState) -> dict[str, Any]:
        model_list = []
        fallbacks = []
        for route in desired.routes:
            fallback_models = []
            for pool in route.get("pools", []):
                is_fallback = bool(pool.get("fallback"))
                model_group = (
                    f"pra-internal/{route['id']}/{pool['id']}"
                    if is_fallback else route["public_model"]
                )
                if is_fallback and pool.get("backends"):
                    fallback_models.append(model_group)
                for backend in pool.get("backends", []):
                    backend = {**backend, "pool_id": pool["id"], "fallback": is_fallback}
                    deployment_id = stable_backend_name(desired.router["id"], backend)
                    model_list.append({
                        "model_name": model_group,
                        "litellm_params": {
                            "model": f"openai/{backend.get('runtime_model_id') or backend['model_id']}",
                            "api_base": backend["inference_url"],
                            "weight": backend.get("weight", 1.0),
                        },
                        "model_info": {"id": deployment_id, **_public_metadata(route, backend)},
                    })
            if fallback_models:
                fallbacks.append({route["public_model"]: fallback_models})
        strategies = {route["policy"]["strategy"] for route in desired.routes}
        strategy = next(iter(strategies)) if len(strategies) == 1 else "simple-shuffle"
        return {
            "model_list": model_list,
            "router_settings": {"routing_strategy": _litellm_strategy(strategy), "fallbacks": fallbacks},
            "general_settings": {"forward_client_headers_to_llm_api": False},
        }


class AgentGatewayAdapter(BaseRouterAdapter):
    kind = "agentgateway"
    features = ("openai", "mcp", "a2a", "streaming", "file-watch", "xds-ready", "last-good")

    def compile(self, desired: RouterDesiredState) -> dict[str, Any]:
        backends: list[dict[str, Any]] = []
        routes: list[dict[str, Any]] = []
        for route in desired.routes:
            targets = []
            for backend in _route_backends(route):
                name = stable_backend_name(desired.router["id"], backend)
                backends.append({"name": name, "host": backend["inference_url"]})
                targets.append({
                    "backend": name,
                    "weight": max(1, round(float(backend.get("weight", 1.0)) * 100)),
                })
            routes.append({
                "name": route["id"],
                "gateways": ["pra"],
                "matches": [{"headers": [{"name": "x-pra-model", "value": {"exact": route["public_model"]}}]}],
                "backends": targets,
            })
        return {
            "gateways": {"pra": {"port": int(desired.router.get("metadata", {}).get("port", 3000))}},
            "backends": backends,
            "routes": routes,
        }


class KubernetesGAIEAdapter(BaseRouterAdapter):
    kind = "kubernetes-gaie"
    features = ("gateway-api", "inference-pool-v1", "llm-d", "cluster-local-replica-selection", "last-good")

    def compile(self, desired: RouterDesiredState) -> dict[str, Any]:
        namespace = str(desired.router.get("metadata", {}).get("namespace", "default"))
        gateway_name = str(desired.router.get("metadata", {}).get("gateway", "pra-gateway"))
        resources: list[dict[str, Any]] = []
        for route in desired.routes:
            pool_names = []
            for pool in route.get("pools", []):
                name = _k8s_name(pool["id"])
                pool_names.append(name)
                selector = dict(pool.get("selectors", {}).get("labels") or {})
                selector.setdefault("pra.einnovator.dev/model-id", _k8s_label(pool["model_id"]))
                resources.append({
                    "apiVersion": "inference.networking.k8s.io/v1",
                    "kind": "InferencePool",
                    "metadata": {"name": name, "namespace": namespace, "labels": {"pra.einnovator.dev/managed": "true"}},
                    "spec": {
                        "selector": {"matchLabels": selector},
                        "targetPorts": [{"number": int(pool.get("metadata", {}).get("port", 8000))}],
                        "endpointPickerRef": {
                            "name": str(pool.get("metadata", {}).get("endpoint_picker", f"{name}-epp")),
                            "group": "", "kind": "Service", "port": {"number": 9002},
                        },
                    },
                })
            rules = [{
                "matches": [{"headers": [{"name": "x-pra-model", "value": route["public_model"]}]}],
                "backendRefs": [{
                    "name": name, "group": "inference.networking.k8s.io", "kind": "InferencePool", "port": 8000,
                } for name in pool_names],
            }]
            resources.append({
                "apiVersion": "gateway.networking.k8s.io/v1",
                "kind": "HTTPRoute",
                "metadata": {"name": _k8s_name(route["id"]), "namespace": namespace},
                "spec": {"parentRefs": [{"name": gateway_name}], "rules": rules},
            })
        return {"apiVersion": "v1", "kind": "List", "items": resources}


class ReferenceRouterAdapter(BaseRouterAdapter):
    kind = "pra-reference"
    features = ("openai", "streaming", "round-robin", "weighted", "least-active", "lowest-recent-ttft", "last-good")

    def compile(self, desired: RouterDesiredState) -> dict[str, Any]:
        return {
            "revision": desired.desired_revision,
            "routes": [{
                "id": route["id"], "public_model": route["public_model"],
                "route_kind": route.get("route_kind", "llm"),
                "tenant_ids": list(route.get("tenant_ids") or []),
                "strategy": route["policy"]["strategy"],
                "fallback": route["policy"].get("fallback", []),
                "backends": [{
                    "id": backend["id"], "url": backend["inference_url"],
                    "model": backend.get("runtime_model_id") or backend["model_id"],
                    "fallback": bool(backend.get("fallback")),
                    "weight": backend.get("weight", 1.0), "region": backend.get("region"),
                    "labels": backend.get("labels", {}), "metadata": _public_metadata(route, backend),
                } for backend in _route_backends(route)],
            } for route in desired.routes],
            "forwarded_headers": list(FORWARDED_HEADERS),
        }


class BifrostRouterAdapter(BaseRouterAdapter):
    kind = "bifrost"
    features = ("openai", "streaming", "custom-provider", "weighted", "routing-rules", "fallback", "last-good")

    def compile(self, desired: RouterDesiredState) -> dict[str, Any]:
        providers: dict[str, dict[str, Any]] = {}
        rules: list[dict[str, Any]] = []
        for index, route in enumerate(desired.routes):
            targets = []
            fallbacks = []
            for backend in _route_backends(route):
                provider = stable_backend_name(desired.router["id"], backend)
                providers[provider] = {
                    "keys": [],
                    "network_config": {
                        "base_url": backend["inference_url"],
                        "default_request_timeout_in_seconds": 120,
                    },
                    "custom_provider_config": {
                        "is_key_less": True,
                        "base_provider_type": "openai",
                        "allowed_requests": {
                            "list_models": True,
                            "chat_completion": True,
                            "chat_completion_stream": True,
                            "responses": True,
                            "responses_stream": True,
                        },
                    },
                }
                target = {"provider": provider, "model": backend.get("runtime_model_id") or backend["model_id"], "weight": backend.get("weight", 1.0)}
                (fallbacks if backend.get("fallback") else targets).append(target)
            targets = _normalized_targets(targets)
            rules.append({
                "id": f"pra-{_k8s_name(route['id'])}", "name": route["id"], "enabled": True,
                "cel_expression": _bifrost_expression(route),
                "targets": targets,
                "fallbacks": [f"{item['provider']}/{item['model']}" for item in fallbacks],
                "scope": "global", "scope_id": None, "priority": index + 10,
            })
        return {"providers": providers, "governance": {"routing_rules": rules}}


def adapter_for(kind: str, transport: RouterTransport | None = None) -> BaseRouterAdapter:
    classes = {
        "litellm": LiteLLMRouterAdapter,
        "agentgateway": AgentGatewayAdapter,
        "kubernetes-gaie": KubernetesGAIEAdapter,
        "pra-reference": ReferenceRouterAdapter,
        "bifrost": BifrostRouterAdapter,
    }
    try:
        return classes[kind](transport)
    except KeyError as error:
        raise ValueError(f"unsupported router kind {kind!r}") from error


def _litellm_strategy(strategy: str) -> str:
    return {
        "round-robin": "simple-shuffle",
        "default": "simple-shuffle",
        "weighted": "simple-shuffle",
        "least-busy": "least-busy",
        "lowest-cost": "cost-based-routing",
        "cost-preferred": "cost-based-routing",
    }.get(strategy, "simple-shuffle")


def _k8s_name(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.casefold()).strip("-")[:63]


def _k8s_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)[:63].strip("-_.")


def _normalized_targets(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not values:
        return []
    total = sum(float(value["weight"]) for value in values)
    return [{**value, "weight": float(value["weight"]) / total} for value in values]


def _bifrost_expression(route: dict[str, Any]) -> str:
    expression = f'model == {json.dumps(route["public_model"])}'
    tenants = list(route.get("tenant_ids") or [])
    if tenants:
        expression += f' && headers["x-pra-tenant-id"] in {json.dumps(tenants)}'
    return expression
