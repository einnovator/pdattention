"""RBAC-governed Control Plane assistant with resumable event streams."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
from collections import deque
from typing import Any, Awaitable, Callable

from pra_hf.toolsets import Toolset

from .auth import Identity
from .fleet import FleetService
from .persistence import ControlStore
from .rbac import Permission, permits


class ControlPlaneAgent:
    """Expose fleet tools through PRA SDK schemas and one authorization path."""

    def __init__(self, fleet: FleetService, store: ControlStore, grafana_url: str | None = None) -> None:
        self.fleet = fleet
        self.store = store
        self.grafana_url = grafana_url
        # PRA Toolset produces the same typed capability records used by PRAAgent.
        self.toolset = Toolset.from_callables(
            [self.list_fleet_tool, self.show_drift_tool, self.open_grafana_tool],
            name="pra-control-plane", namespace="control-plane",
        )

    def list_fleet_tool(self, model: str = "") -> dict[str, Any]:
        """List fleet engines, optionally filtering by model name."""
        return {"model_filter": model, "execution": "async-control-service"}

    def show_drift_tool(self) -> dict[str, Any]:
        """Show deployment differences between Registry intent and engine state."""
        return {"execution": "async-control-service"}

    def open_grafana_tool(self, engine: str = "") -> dict[str, Any]:
        """Return the configured Grafana dashboard for an engine."""
        return {"engine": engine, "grafana_url": self.grafana_url}

    async def answer(self, identity: Identity, prompt: str, emit: Callable[[str, dict[str, Any]], Awaitable[None]]) -> str:
        if not permits(identity.role, Permission.FLEET_READ):
            raise PermissionError("fleet:read is required")
        lower = prompt.casefold()
        await emit("tool.started", {"tool": "fleet_overview", "side_effect": "none"})
        fleet = await self.fleet.overview()
        await emit("tool.completed", {"tool": "fleet_overview", "result_count": len(fleet["items"])})
        if "grafana" in lower:
            engine = next((row["name"] for row in fleet["items"] if row["name"].casefold() in lower), None)
            suffix = f"?var-engine={engine}" if engine else ""
            return f"Open the fleet dashboard: {self.grafana_url}{suffix}" if self.grafana_url else "Grafana is not configured."
        if "drift" in lower or "approved" in lower or "bundle revision" in lower:
            rows = [row for row in fleet["items"] if row["status"] == "DRIFT"]
            if not rows:
                return "No known engine currently differs from its resolved Registry desired state."
            return "Drifted engines:\n" + "\n".join(
                f"- {row['name']}: " + ", ".join(diff["field"] for diff in row["drift"]["differences"])
                for row in rows
            )
        if "running" in lower or "which engine" in lower or "which engines" in lower:
            terms = [word.strip("?,.") for word in prompt.split() if "/" in word or any(char.isdigit() for char in word)]
            rows = fleet["items"]
            if terms:
                rows = [row for row in rows if any(term.casefold() in str(row.get("model", "")).casefold() for term in terms)]
            return "Matching engines:\n" + ("\n".join(f"- {row['name']}: {row.get('model') or 'no model'} ({row['status']})" for row in rows) or "- none")
        if "reload" in lower or "warm" in lower:
            rows = sorted(fleet["items"], key=lambda row: float(row.get("metrics", {}).get("storage_reloads") or 0), reverse=True)
            return "Storage reload summary:\n" + "\n".join(f"- {row['name']}: {row.get('metrics', {}).get('storage_reloads', 'not measured')}" for row in rows)
        summary = fleet["summary"]
        return f"Fleet has {summary['total']} engines: {summary['healthy']} in sync, {summary['drift']} drifted, {summary['offline']} offline, and {summary['unknown']} without desired state."


class AgentReplayService:
    """Persist message IDs and bounded events for reconnect and replay."""

    def __init__(self, store: ControlStore, limit: int = 250) -> None:
        self.store = store
        self.limit = limit
        self.locks: dict[str, asyncio.Lock] = {}

    def open(self, identity: Identity, token: str | None = None) -> dict[str, Any]:
        if token:
            existing = self.store.get_agent_session(token)
            if existing and existing["actor"] == identity.subject:
                return existing
        token = secrets.token_urlsafe(32)
        value = {"resume_token": token, "actor": identity.subject, "role": identity.role.value, "events": [], "seen_message_ids": []}
        self.store.put_agent_session(value)
        return value

    def append(self, session: dict[str, Any], event_type: str, payload: dict[str, Any], message_id: str | None = None) -> dict[str, Any]:
        events = deque(session.get("events", []), maxlen=self.limit)
        event = {
            "sequence": (events[-1]["sequence"] + 1) if events else 1,
            "message_id": message_id or secrets.token_hex(12), "type": event_type,
            "timestamp": time.time(), **payload,
        }
        events.append(event)
        session["events"] = list(events)
        session["seen_message_ids"] = list(deque(session.get("seen_message_ids", []), maxlen=self.limit))
        self.store.put_agent_session(session)
        return event

    def seen(self, session: dict[str, Any], message_id: str) -> bool:
        if message_id in session.get("seen_message_ids", []):
            return True
        seen = deque(session.get("seen_message_ids", []), maxlen=self.limit)
        seen.append(message_id)
        session["seen_message_ids"] = list(seen)
        self.store.put_agent_session(session)
        return False

    @staticmethod
    def replay(session: dict[str, Any], after: int) -> list[dict[str, Any]]:
        return [event for event in session.get("events", []) if int(event["sequence"]) > after]
