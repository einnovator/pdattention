"""Async clients for the open Registry and Engine Management contracts."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import httpx


class ServiceClientError(RuntimeError):
    def __init__(self, service: str, status_code: int, detail: str) -> None:
        super().__init__(f"{service} returned {status_code}: {detail}")
        self.service = service
        self.status_code = status_code
        self.detail = detail


class AsyncServiceClient:
    def __init__(self, name: str, url: str, token: str | None = None, *, timeout: float = 8.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self.token = token
        self.client = httpx.AsyncClient(base_url=self.url, timeout=timeout, transport=transport)

    async def request(self, method: str, path: str, body: Mapping[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json", "User-Agent": "pra-control/1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = await self.client.request(method, path, json=body, headers=headers)
        except httpx.HTTPError as error:
            # Normalize network failures so fleet discovery can degrade without
            # taking the Control Plane UI and agent model picker down with it.
            raise ServiceClientError(self.name, 503, str(error)) from error
        if response.status_code >= 400:
            try:
                payload = response.json()
                detail = payload.get("error", payload).get("detail", response.text)
            except Exception:
                detail = response.text
            raise ServiceClientError(self.name, response.status_code, str(detail))
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()


class RegistryClient(AsyncServiceClient):
    def __init__(self, url: str, token: str | None = None, **kwargs: Any) -> None:
        super().__init__("registry", url, token, **kwargs)

    async def list(self, resource: str, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        return await self.request("GET", f"/v1/{resource}?limit={limit}&offset={offset}")

    async def deployments(self) -> list[dict[str, Any]]:
        return (await self.list("deployments", limit=500))["items"]

    async def instances(self, *, instance_type: str | None = None) -> list[dict[str, Any]]:
        suffix = f"&instance_type={instance_type}" if instance_type else ""
        return (await self.request("GET", f"/v1/instances?limit=500&offset=0{suffix}"))["items"]


class EngineClient(AsyncServiceClient):
    def __init__(self, name: str, url: str, token: str | None = None, **kwargs: Any) -> None:
        super().__init__(name, url, token, **kwargs)

    async def snapshot(self) -> dict[str, Any]:
        paths = {
            "info": "/v1/pra/info", "state": "/v1/pra/state",
            "capabilities": "/v1/pra/capabilities", "models": "/v1/pra/models",
            "profiles": "/v1/pra/profiles", "storage": "/v1/pra/storage",
            "observability": "/v1/pra/observability",
        }
        results = await asyncio.gather(*(self.request("GET", path) for path in paths.values()))
        return dict(zip(paths, results))

    async def endpoint(self, section: str) -> Any:
        allowed = {"summary": "info", "capabilities": "capabilities", "config": "config", "models": "models", "sessions": "sessions", "resources": "resources", "storage": "storage", "observability": "observability", "audit": "audit"}
        if section not in allowed:
            raise KeyError(section)
        return await self.request("GET", f"/v1/pra/{allowed[section]}")

    async def action(self, action: str, body: Mapping[str, Any]) -> Any:
        return await self.request("POST", f"/v1/pra/actions/{action}", body)

    async def patch_config(self, body: Mapping[str, Any]) -> Any:
        return await self.request("PATCH", "/v1/pra/config", body)
