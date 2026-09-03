"""Async Registry client used by the router reconciliation loop."""

from __future__ import annotations

import asyncio
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any


class RegistryRouterSource:
    """Read desired state and report compact observed state over Registry REST."""

    def __init__(self, base_url: str, token: str | None = None, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    async def list_router_ids(self) -> list[str]:
        value = await asyncio.to_thread(self._request, "GET", "/v1/routers?limit=500&offset=0", None)
        return [str(row["id"]) for row in value.get("items", [])]

    async def desired_state(self, router_id: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(router_id, safe="")
        return await asyncio.to_thread(self._request, "GET", f"/v1/routers/{quoted}/desired", None)

    async def report_observed(
        self, router_id: str, *, observed_revision: int, health: str,
        last_error: str | None = None, supported_features: list[str] | None = None,
    ) -> None:
        quoted = urllib.parse.quote(router_id, safe="")
        body: dict[str, Any] = {
            "observed_revision": observed_revision,
            "health": health,
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "last_error": last_error,
        }
        if supported_features is not None:
            body["supported_features"] = supported_features
        await asyncio.to_thread(self._request, "PATCH", f"/v1/routers/{quoted}", body)

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        headers = {"Accept": "application/json", "User-Agent": "pra-router-controller/1"}
        payload = None
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=payload, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


class InMemoryRouterSource:
    """Small source used by tests and embedded demonstrations."""

    def __init__(self, states: dict[str, dict[str, Any]]) -> None:
        self.states = states
        self.reports: list[dict[str, Any]] = []

    async def list_router_ids(self) -> list[str]:
        return sorted(self.states)

    async def desired_state(self, router_id: str) -> dict[str, Any]:
        return self.states[router_id]

    async def report_observed(self, router_id: str, **values: Any) -> None:
        self.reports.append({"router_id": router_id, **values})
        self.states[router_id]["router"].update(values)
