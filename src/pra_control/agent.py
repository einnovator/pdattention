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
from .managers import ControlManager
from .persistence import ControlStore


class ControlPlaneAgent:
    """Render manager-owned domain facts for the built-in conversational UI."""

    def __init__(self, manager: ControlManager, config: Any | None = None) -> None:
        self.manager = manager
        self.config = config
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
        return {"engine": engine, "execution": "async-control-manager"}

    async def answer(
        self, identity: Identity, prompt: str,
        emit: Callable[[str, dict[str, Any]], Awaitable[None]],
        *, target: dict[str, Any] | None = None,
    ) -> str:
        caller = identity.caller(transport="agent")
        lower = prompt.casefold()
        await emit("tool.started", {"tool": "fleet_overview", "side_effect": "none"})
        fleet = (await self.manager.fleet.list(caller)).model_dump(mode="json")
        await emit("tool.completed", {"tool": "fleet_overview", "result_count": len(fleet["items"])})
        if (
            target and target.get("inference_url")
            and bool(getattr(getattr(self.config, "agent", None), "model_enabled", True))
        ):
            try:
                return await self._model_answer(prompt, fleet, target)
            except Exception as error:
                await emit("tool.completed", {
                    "tool": "agent_model", "status": "fallback",
                    "detail": f"{type(error).__name__}: {error}",
                })
        if "grafana" in lower:
            engine = next((row["name"] for row in fleet["items"] if row["name"].casefold() in lower), None)
            url = self.manager.observability.links(caller, engine=engine).get("grafana")
            return f"Open the fleet dashboard: {url}" if url else "Grafana is not configured."
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

    async def _model_answer(
        self, prompt: str, fleet: dict[str, Any], target: dict[str, Any],
    ) -> str:
        """Use a configured OpenAI-compatible model without granting mutation access."""

        import httpx

        agent_config = getattr(self.config, "agent", None)
        timeout = float(getattr(agent_config, "request_timeout_seconds", 60.0))
        headers = {"Content-Type": "application/json"}
        api_key = agent_config.api_key() if agent_config else None
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {
            "model": target["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the read-only PRA Control Plane assistant. Answer from the "
                        "provided fleet snapshot. Never claim that an operational change was applied."
                    ),
                },
                {"role": "system", "content": "Fleet snapshot:\n" + json.dumps(fleet, default=str)},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(getattr(agent_config, "temperature", 0.1)),
            "max_tokens": int(getattr(agent_config, "max_tokens", 512)),
            "stream": False,
        }
        await asyncio.sleep(0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{str(target['inference_url']).rstrip('/')}/v1/chat/completions",
                headers=headers, json=body,
            )
            response.raise_for_status()
            value = response.json()
        return str(value["choices"][0]["message"]["content"])


class AgentReplayService:
    """Persist message IDs and bounded events for reconnect and replay."""

    def __init__(self, store: ControlStore, limit: int = 250) -> None:
        self.store = store
        self.limit = limit
        self.locks: dict[str, asyncio.Lock] = {}

    def open(
        self, identity: Identity, token: str | None = None,
        *, settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if token:
            existing = self.store.get_agent_session(token)
            if existing and existing["actor"] == identity.subject:
                return existing
        token = secrets.token_urlsafe(32)
        value = {
            "resume_token": token, "actor": identity.subject, "role": identity.role.value,
            "events": [], "seen_message_ids": [], "settings": dict(settings or {}),
        }
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

    def update_settings(self, session: dict[str, Any], **changes: Any) -> dict[str, Any]:
        settings = dict(session.get("settings") or {})
        for key, value in changes.items():
            if value is None:
                settings.pop(key, None)
            else:
                settings[key] = value
        session["settings"] = settings
        self.store.put_agent_session(session)
        return settings

    def list(self, identity: Identity) -> list[dict[str, Any]]:
        rows = self.store.list_agent_sessions(identity.subject)
        return [{
            "resume_token": row["resume_token"],
            "updated_at": row["updated_at"],
            "event_count": len(row.get("events") or []),
            "settings": dict(row.get("settings") or {}),
        } for row in rows]

    @staticmethod
    def replay(session: dict[str, Any], after: int) -> list[dict[str, Any]]:
        return [event for event in session.get("events", []) if int(event["sequence"]) > after]
