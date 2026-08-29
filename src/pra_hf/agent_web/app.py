"""FastAPI/WebSocket transport over the shared PRA agent services."""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ..agent_profiles import AgentLauncher, AgentProfileRegistry


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: str | None = None
    user_id: str = "local-user"
    session_id: str | None = None
    task: str | None = None


class MessageCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: str
    approved: bool


class AgentWebService:
    """Own active presentation sessions while durable state stays in SessionService."""

    def __init__(
        self,
        *,
        registry: AgentProfileRegistry | None = None,
        launcher: AgentLauncher | None = None,
        config_path: str | Path | None = None,
        default_profile: str | None = None,
        pra_override: str | None = None,
    ) -> None:
        self.registry = registry or AgentProfileRegistry()
        self.launcher = launcher or AgentLauncher()
        self.config_path = config_path
        self.default_profile = default_profile
        self.pra_override = pra_override
        self.agents: dict[str, Any] = {}
        self.profile_names: dict[str, str] = {}
        self.events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._sequences = defaultdict(lambda: itertools.count(1))
        self._locks = defaultdict(threading.RLock)
        self._approval_condition = threading.Condition()
        self._approval_results: dict[str, bool] = {}

    def emit(self, session_id: str, event_type: str, **payload: Any) -> dict[str, Any]:
        event = {
            "sequence": next(self._sequences[session_id]),
            "type": event_type,
            "session_id": session_id,
            "timestamp": time.time(),
            **payload,
        }
        self.events[session_id].append(event)
        return event

    def profiles(self) -> dict[str, Any]:
        document = self.registry.load(config_path=self.config_path)
        return {
            "default_profile": self.default_profile or document.default_profile,
            "profiles": [profile.redacted_dict() for profile in document.profiles.values()],
        }

    def create_session(self, request: SessionCreate) -> dict[str, Any]:
        selected_profile = request.profile or self.default_profile
        profile, _ = self.registry.resolve(profile_name=selected_profile, config_path=self.config_path)
        if self.pra_override:
            profile = replace(profile, pra=self.pra_override)
        launch = self.launcher.launch(profile)
        launch.agent.config = replace(launch.agent.config, user_id=request.user_id)
        state = launch.agent.start_session(request.session_id, task_description=request.task)
        session_id = state.session_id
        self.agents[session_id] = launch.agent
        self.profile_names[session_id] = profile.name
        if profile.tools.approval == "ask":
            launch.agent.authorization_callback = (
                lambda resource, call, sid=session_id: self.request_approval(sid, resource, call)
            )
        self.emit(session_id, "session.created", summary=dict(launch.summary))
        return self.session(session_id)

    def request_approval(self, session_id: str, resource: Any, call: Any) -> bool:
        """Emit one typed approval request and wait without granting a lasting rule."""

        approval_id = uuid.uuid4().hex
        self.emit(
            session_id,
            "tool.approval_requested",
            approval_id=approval_id,
            tool=resource.name,
            arguments=dict(call.arguments),
            side_effect_class=resource.side_effect_class.value,
            reason="The selected agent profile requires approval for this side effect.",
        )
        deadline = time.time() + 120
        with self._approval_condition:
            while approval_id not in self._approval_results and time.time() < deadline:
                self._approval_condition.wait(timeout=max(0.01, min(1.0, deadline - time.time())))
            approved = self._approval_results.pop(approval_id, False)
        self.emit(session_id, "tool.approval_resolved", approval_id=approval_id, approved=approved)
        return approved

    def approve(self, session_id: str, approval_id: str, approved: bool) -> dict[str, Any]:
        if session_id not in self.agents:
            raise KeyError(session_id)
        requested = any(
            event.get("approval_id") == approval_id and event["type"] == "tool.approval_requested"
            for event in self.events[session_id]
        )
        if not requested:
            raise KeyError(approval_id)
        with self._approval_condition:
            self._approval_results[approval_id] = bool(approved)
            self._approval_condition.notify_all()
        return {"approval_id": approval_id, "approved": bool(approved)}

    def session(self, session_id: str) -> dict[str, Any]:
        try:
            agent = self.agents[session_id]
        except KeyError as error:
            raise KeyError(session_id) from error
        state = agent.state
        return {
            "session_id": state.session_id,
            "user_id": state.user_id,
            "profile": self.profile_names[session_id],
            "version": state.version,
            "active_task_id": state.active_task_id,
            "tasks": state.tasks.to_dict(),
            "records": [
                {"id": row.record_id, "type": row.record_type.value, "views": [name.value for name in row.views]}
                for row in state.records
            ],
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        return [self.session(session_id) for session_id in sorted(self.agents)]

    def message(self, session_id: str, text: str) -> dict[str, Any]:
        if session_id not in self.agents:
            raise KeyError(session_id)
        with self._locks[session_id]:
            self.emit(session_id, "message.user", text=text)
            self.emit(session_id, "generation.started")
            turn = self.agents[session_id].run_turn(text)
            event = self.emit(
                session_id,
                "message.assistant",
                text=turn.text,
                selected_record_ids=list(turn.selected_record_ids),
                tool_executions=len(turn.tool_executions),
            )
            self.emit(session_id, "generation.completed", version=turn.session.version)
            return event

    def close(self, session_id: str) -> None:
        agent = self.agents.pop(session_id, None)
        self.profile_names.pop(session_id, None)
        if agent is not None:
            agent.close()


def create_app(
    *,
    service: AgentWebService | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    """Create the optional web app without introducing another agent runtime."""

    web = service or AgentWebService(config_path=config_path)
    app = FastAPI(title="PRA Agent Web UI", version="0.1.0", docs_url="/api/docs")
    app.state.pra_web = web
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/profiles")
    def profiles() -> dict[str, Any]:
        return web.profiles()

    @app.get("/api/sessions")
    def sessions() -> list[dict[str, Any]]:
        return web.list_sessions()

    @app.post("/api/sessions")
    def create_session(request: SessionCreate) -> dict[str, Any]:
        try:
            return web.create_session(request)
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str) -> dict[str, Any]:
        try:
            return web.session(session_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error

    @app.post("/api/sessions/{session_id}/messages")
    async def message(session_id: str, request: MessageCreate) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(web.message, session_id, request.text)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session not found") from error

    @app.post("/api/sessions/{session_id}/cancel")
    def cancel(session_id: str) -> dict[str, Any]:
        if session_id not in web.agents:
            raise HTTPException(status_code=404, detail="Session not found")
        return web.emit(session_id, "generation.cancel_requested")

    @app.post("/api/sessions/{session_id}/approvals")
    def approve(session_id: str, request: ApprovalDecision) -> dict[str, Any]:
        try:
            return web.approve(session_id, request.approval_id, request.approved)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Session or approval not found") from error

    @app.delete("/api/sessions/{session_id}")
    def close(session_id: str) -> dict[str, bool]:
        web.close(session_id)
        return {"closed": True}

    @app.websocket("/ws/sessions/{session_id}")
    async def websocket(websocket: WebSocket, session_id: str, after: int = 0) -> None:
        if session_id not in web.agents:
            await websocket.close(code=4404)
            return
        await websocket.accept()
        cursor = int(after)
        try:
            while True:
                pending = [event for event in web.events[session_id] if event["sequence"] > cursor]
                for event in pending:
                    await websocket.send_json(event)
                    cursor = event["sequence"]
                try:
                    incoming = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                    if incoming.get("type") == "resume":
                        cursor = int(incoming.get("after", cursor))
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "heartbeat", "sequence": cursor})
        except WebSocketDisconnect:
            return

    return app
