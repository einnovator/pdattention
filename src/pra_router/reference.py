"""Small async OpenAI-compatible router for labs and integration tests."""

import asyncio
import random
import time
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .adapters import FORWARDED_HEADERS


class ReferenceBackend(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    url: str
    model: str
    fallback: bool = False
    weight: float = Field(default=1.0, gt=0)
    region: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReferenceRoute(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    public_model: str
    route_kind: str = "llm"
    strategy: str = "round-robin"
    fallback: list[str] = Field(default_factory=list)
    tenant_ids: list[str] = Field(default_factory=list)
    backends: list[ReferenceBackend] = Field(min_length=1)


class ReferenceRouterConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    revision: int = Field(default=0, ge=0)
    routes: list[ReferenceRoute] = Field(default_factory=list)
    forwarded_headers: list[str] = Field(default_factory=lambda: list(FORWARDED_HEADERS))
    max_attempts: int = Field(default=2, ge=1, le=10)
    request_timeout_seconds: float = Field(default=120.0, gt=0)


class BackendClient(Protocol):
    async def request(
        self, backend: ReferenceBackend, payload: dict[str, Any], headers: dict[str, str], timeout: float,
    ) -> dict[str, Any]: ...

    def stream(
        self, backend: ReferenceBackend, payload: dict[str, Any], headers: dict[str, str], timeout: float,
    ) -> AsyncIterator[bytes]: ...


class HttpxBackendClient:
    """Optional HTTPX transport; importing the module does not require HTTPX."""

    async def request(
        self, backend: ReferenceBackend, payload: dict[str, Any], headers: dict[str, str], timeout: float,
    ) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{backend.url.rstrip('/')}/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def stream(
        self, backend: ReferenceBackend, payload: dict[str, Any], headers: dict[str, str], timeout: float,
    ) -> AsyncIterator[bytes]:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", f"{backend.url.rstrip('/')}/v1/chat/completions", json=payload, headers=headers,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk


class ReferenceRouter:
    """Inspectably select 1-20 backends and proxy without Registry hot-path calls."""

    def __init__(
        self, config: ReferenceRouterConfig | Mapping[str, Any],
        client: BackendClient | None = None, *, seed: int = 0,
    ) -> None:
        self.config = config if isinstance(config, ReferenceRouterConfig) else ReferenceRouterConfig.model_validate(config)
        self.applied_config = self.config.model_dump(mode="json")
        self.client = client or HttpxBackendClient()
        self.random = random.Random(seed)
        self.active: Counter[str] = Counter()
        self.recent_ttft_ms: dict[str, float] = {}
        self.metrics: Counter[tuple[str, str]] = Counter()
        self.latency_ms: defaultdict[str, list[float]] = defaultdict(list)
        self.route_decision_ms: defaultdict[str, list[float]] = defaultdict(list)
        self._round_robin: Counter[str] = Counter()
        self._smooth_weight: defaultdict[str, dict[str, float]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def reload(self, config: ReferenceRouterConfig | Mapping[str, Any]) -> None:
        value = config if isinstance(config, ReferenceRouterConfig) else ReferenceRouterConfig.model_validate(config)
        async with self._lock:
            self.config = value
            self.applied_config = dict(config) if isinstance(config, Mapping) else config.model_dump(mode="json")

    def route(self, public_model: str, headers: Mapping[str, str] | None = None) -> ReferenceRoute:
        try:
            route = next(route for route in self.config.routes if route.public_model == public_model)
        except StopIteration as error:
            raise KeyError(f"logical model {public_model!r} is not configured") from error
        tenant = next((value for key, value in (headers or {}).items() if key.casefold() == "x-pra-tenant-id"), None)
        if route.tenant_ids and tenant not in route.tenant_ids:
            raise PermissionError(f"tenant is not allowed to use route {route.id!r}")
        return route

    async def choose(self, route: ReferenceRoute, headers: Mapping[str, str] | None = None) -> list[ReferenceBackend]:
        primary = [row for row in route.backends if not row.fallback]
        fallback = [row for row in route.backends if row.fallback]
        candidates = primary or fallback
        preferred_region = (headers or {}).get("x-pra-region")
        if preferred_region:
            local = [row for row in candidates if row.region == preferred_region]
            candidates = local + [row for row in candidates if row.region != preferred_region]
        strategy = route.strategy
        async with self._lock:
            if strategy in {"least-active", "least-busy"}:
                candidates.sort(key=lambda row: (self.active[row.id], row.id))
            elif strategy == "lowest-recent-ttft":
                candidates.sort(key=lambda row: (self.recent_ttft_ms.get(row.id, float("inf")), row.id))
            elif strategy == "random":
                self.random.shuffle(candidates)
            elif strategy == "weighted":
                first = self._smooth_weighted(route, candidates)
                candidates = [first] + [row for row in candidates if row.id != first.id]
            else:
                offset = self._round_robin[route.id] % len(candidates)
                self._round_robin[route.id] += 1
                candidates = candidates[offset:] + candidates[:offset]
        return candidates + fallback if primary else candidates

    async def complete(self, payload: dict[str, Any], headers: Mapping[str, str] | None = None) -> dict[str, Any]:
        route = self.route(str(payload.get("model", "")), headers)
        decision_started = time.perf_counter()
        candidates = await self.choose(route, headers)
        self.route_decision_ms[route.id].append((time.perf_counter() - decision_started) * 1000)
        attempts = min(self.config.max_attempts, len(candidates))
        last_error: Exception | None = None
        for attempt, backend in enumerate(candidates[:attempts]):
            request_payload = {**payload, "model": backend.model, "stream": False}
            forwarded = self.forward_headers(headers or {}, route.id)
            started = time.perf_counter()
            self.active[backend.id] += 1
            try:
                result = await self.client.request(
                    backend, request_payload, forwarded, self.config.request_timeout_seconds,
                )
                elapsed = (time.perf_counter() - started) * 1000
                self._observe(route.id, backend.id, elapsed, success=True, retry=attempt > 0)
                result.setdefault("model", route.public_model)
                result.setdefault("pra_router", {"route": route.id, "backend": backend.id})
                return result
            except Exception as error:
                last_error = error
                elapsed = (time.perf_counter() - started) * 1000
                self._observe(route.id, backend.id, elapsed, success=False, retry=attempt > 0)
            finally:
                self.active[backend.id] -= 1
        raise RuntimeError(f"all backends failed for route {route.id!r}: {last_error}")

    async def stream(self, payload: dict[str, Any], headers: Mapping[str, str] | None = None) -> AsyncIterator[bytes]:
        route = self.route(str(payload.get("model", "")), headers)
        decision_started = time.perf_counter()
        candidates = await self.choose(route, headers)
        self.route_decision_ms[route.id].append((time.perf_counter() - decision_started) * 1000)
        attempts = min(self.config.max_attempts, len(candidates))
        last_error: Exception | None = None
        for attempt, backend in enumerate(candidates[:attempts]):
            request_payload = {**payload, "model": backend.model, "stream": True}
            forwarded = self.forward_headers(headers or {}, route.id)
            started = time.perf_counter()
            first = True
            emitted = False
            self.active[backend.id] += 1
            try:
                async for chunk in self.client.stream(
                    backend, request_payload, forwarded, self.config.request_timeout_seconds,
                ):
                    if first:
                        self.recent_ttft_ms[backend.id] = (time.perf_counter() - started) * 1000
                        first = False
                    emitted = True
                    yield chunk
                elapsed = (time.perf_counter() - started) * 1000
                self._observe(route.id, backend.id, elapsed, success=True, retry=attempt > 0)
                return
            except Exception as error:
                last_error = error
                elapsed = (time.perf_counter() - started) * 1000
                self._observe(route.id, backend.id, elapsed, success=False, retry=attempt > 0)
                if emitted:
                    raise RuntimeError("stream failed after response bytes were emitted; retry is unsafe") from error
            finally:
                self.active[backend.id] -= 1
        raise RuntimeError(f"all streaming backends failed for route {route.id!r}: {last_error}")

    def forward_headers(self, headers: Mapping[str, str], route_id: str) -> dict[str, str]:
        allowed = {value.casefold() for value in self.config.forwarded_headers}
        result = {key: value for key, value in headers.items() if key.casefold() in allowed}
        result["x-pra-route-id"] = route_id
        return result

    def inspect(self) -> dict[str, Any]:
        return {
            "kind": "pra-reference", "revision": self.config.revision,
            "scope": "local/reference; not the preferred large-fleet data plane",
            "routes": [route.model_dump(mode="json") for route in self.config.routes],
            "active": dict(self.active), "recent_ttft_ms": dict(self.recent_ttft_ms),
        }

    def metric_snapshot(self) -> dict[str, Any]:
        return {
            "counters": {f"{route}:{metric}": value for (route, metric), value in sorted(self.metrics.items())},
            "latency_ms": {
                backend: {"count": len(values), "mean": sum(values) / len(values)}
                for backend, values in self.latency_ms.items() if values
            },
            "route_decision_ms": {
                route: {"count": len(values), "mean": sum(values) / len(values)}
                for route, values in self.route_decision_ms.items() if values
            },
            "active": dict(self.active),
        }

    def _observe(self, route: str, backend: str, elapsed_ms: float, *, success: bool, retry: bool) -> None:
        self.metrics[(route, "requests")] += 1
        self.metrics[(route, "success" if success else "failures")] += 1
        self.metrics[(route, "retries")] += int(retry)
        self.latency_ms[backend].append(elapsed_ms)

    def _smooth_weighted(self, route: ReferenceRoute, candidates: list[ReferenceBackend]) -> ReferenceBackend:
        current = self._smooth_weight[route.id]
        total = sum(row.weight for row in candidates)
        for row in candidates:
            current[row.id] = current.get(row.id, 0.0) + row.weight
        selected = max(candidates, key=lambda row: (current[row.id], row.id))
        current[selected.id] -= total
        return selected


def create_reference_router_app(router: ReferenceRouter, *, reload_token: str | None = None):
    """Expose inference and management APIs without coupling to Registry calls."""

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="PRA Reference Router", version="1.0.0")
    app.state.pra_router = router

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "kind": "pra-reference", "revision": router.config.revision}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [
            {"id": route.public_model, "object": "model", "owned_by": "pra-router"}
            for route in router.config.routes
        ]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        try:
            if payload.get("stream"):
                # Authorize before StreamingResponse sends headers. Exceptions
                # raised later by an async generator can no longer become a 403.
                router.route(str(payload.get("model", "")), request.headers)
                return StreamingResponse(router.stream(payload, request.headers), media_type="text/event-stream")
            return JSONResponse(await router.complete(payload, request.headers))
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except PermissionError as error:
            raise HTTPException(403, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(502, str(error)) from error

    @app.get("/v1/router/info")
    async def info() -> dict[str, Any]:
        return router.inspect()

    @app.get("/v1/router/routes")
    async def routes() -> dict[str, Any]:
        return {"items": [route.model_dump(mode="json") for route in router.config.routes]}

    @app.get("/v1/router/backends")
    async def backends() -> dict[str, Any]:
        return {"items": [backend.model_dump(mode="json") for route in router.config.routes for backend in route.backends]}

    @app.get("/v1/router/metrics")
    async def metrics() -> dict[str, Any]:
        return router.metric_snapshot()

    @app.get("/v1/router/config")
    async def config() -> dict[str, Any]:
        return {"revision": router.config.revision, "config": router.applied_config}

    @app.put("/v1/router/config")
    async def reload(request: Request) -> dict[str, Any]:
        if reload_token and request.headers.get("authorization") != f"Bearer {reload_token}":
            raise HTTPException(401, "invalid router management token")
        body = await request.json()
        config_value = dict(body.get("config") or {})
        config_value["revision"] = int(body.get("revision", config_value.get("revision", 0)))
        await router.reload(config_value)
        return {"status": "reloaded", "revision": router.config.revision}

    return app
