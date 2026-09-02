"""FastAPI application for fleet governance, Registry workflows, and PRA chat."""

from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentReplayService, ControlPlaneAgent
from .auth import AuthService, Identity
from .clients import ServiceClientError
from .config import ControlPlaneConfig, EngineTargetConfig
from .fleet import HIGH_IMPACT_ACTIONS, FleetService
from .persistence import ControlStore
from .rbac import Permission, permits
from .saml import SAMLService, SAMLUnavailable


COOKIE = "pra_control_session"
OAUTH_COOKIE = "pra_control_oauth"
REGISTRY_RESOURCES = frozenset({
    "models", "bundles", "profiles", "qualifications", "compatibility",
    "deployments", "policies", "approvals", "audit", "instances",
})
APPROVAL_TRANSITIONS = frozenset({"approve", "deprecate", "revoke", "promote"})


class LocalLogin(BaseModel):
    username: str
    password: str


class ManualEngineBody(BaseModel):
    name: str
    management_url: str
    token_env: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)


class MutationBody(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    confirmed: bool = False


class RegistryMutationBody(BaseModel):
    values: dict[str, Any]
    reason: str = Field(min_length=1)


class ControlRuntime:
    def __init__(self, config: ControlPlaneConfig) -> None:
        self.config = config
        self.store = ControlStore(config.database_url)
        self.auth = AuthService(config.auth, config.public_url)
        self.fleet = FleetService(config, self.store)
        self.replay = AgentReplayService(self.store, config.replay_limit)
        self.agent = ControlPlaneAgent(self.fleet, self.store, config.grafana.url)


def create_app(config: ControlPlaneConfig | None = None, *, runtime: ControlRuntime | None = None) -> FastAPI:
    config = config or ControlPlaneConfig.load()
    runtime = runtime or ControlRuntime(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await runtime.fleet.close()

    app = FastAPI(title="eInnovator PRA Control Plane", version="0.1.0", lifespan=lifespan)
    app.state.control = runtime
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")

    def identity(request: Request) -> Identity:
        token = request.cookies.get(COOKIE)
        value = runtime.auth.codec.decode(token) if token else runtime.auth.development_identity()
        if not value:
            raise HTTPException(401, "authentication required")
        return value

    def require(permission: Permission) -> Callable[[Identity], Identity]:
        def dependency(value: Identity = Depends(identity)) -> Identity:
            if not permits(value.role, permission):
                raise HTTPException(403, f"{permission.value} is required")
            return value
        return dependency

    def csrf(request: Request, value: Identity = Depends(identity)) -> Identity:
        if runtime.auth.development_identity() == value:
            return value
        supplied = request.headers.get("X-CSRF-Token", "")
        if not supplied or not secrets.compare_digest(supplied, value.csrf_token):
            raise HTTPException(403, "invalid CSRF token")
        return value

    def csrf_permission(permission: Permission) -> Callable[[Identity], Identity]:
        def dependency(value: Identity = Depends(csrf)) -> Identity:
            if not permits(value.role, permission):
                raise HTTPException(403, f"{permission.value} is required")
            return value
        return dependency

    def audit(actor: Identity, action: str, target: str, reason: str, result: str, *, before: Any = None, after: Any = None, request: Request | None = None) -> None:
        runtime.store.audit(
            actor=actor.subject, role=actor.role.value, action=action, target=target,
            before=before, after=after, reason=reason,
            trace_id=request.headers.get("traceparent") if request else None, result=result,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "protocol": "pra-control/1"}

    @app.get("/api/auth/providers")
    async def auth_providers() -> dict[str, Any]:
        return {"items": runtime.auth.providers()}

    @app.get("/api/auth/me")
    async def auth_me(value: Identity = Depends(identity)) -> dict[str, Any]:
        return {**value.public(), "csrf_token": value.csrf_token}

    @app.post("/api/auth/login/local")
    async def auth_local(body: LocalLogin, response: Response) -> dict[str, Any]:
        value = runtime.auth.local_login(body.username, body.password)
        if not value:
            raise HTTPException(401, "invalid credentials")
        response.set_cookie(COOKIE, runtime.auth.codec.encode(value), httponly=True, secure=config.auth.cookie_secure, samesite="lax", max_age=config.auth.session_ttl_seconds)
        return {**value.public(), "csrf_token": value.csrf_token}

    @app.get("/api/auth/login/{provider_name}")
    async def auth_begin(provider_name: str, request: Request) -> Response:
        try:
            provider = runtime.auth.provider(provider_name)
            if provider.kind == "saml":
                url = SAMLService(provider, config.public_url).begin(_saml_request(request), config.public_url)
                return RedirectResponse(url)
            url, transaction = runtime.auth.begin(provider_name)
        except (KeyError, ValueError, SAMLUnavailable) as error:
            raise HTTPException(400, str(error)) from error
        response = RedirectResponse(url)
        response.set_cookie(OAUTH_COOKIE, transaction, httponly=True, secure=config.auth.cookie_secure, samesite="lax", max_age=600)
        return response

    @app.get("/auth/callback/{provider_name}")
    async def auth_callback(provider_name: str, request: Request, code: str, state: str) -> Response:
        transaction = request.cookies.get(OAUTH_COOKIE, "")
        try:
            value = await runtime.auth.callback(provider_name, code, state, transaction)
        except (KeyError, ValueError, ServiceClientError) as error:
            raise HTTPException(400, str(error)) from error
        response = RedirectResponse("/")
        response.delete_cookie(OAUTH_COOKIE)
        response.set_cookie(COOKIE, runtime.auth.codec.encode(value), httponly=True, secure=config.auth.cookie_secure, samesite="lax", max_age=config.auth.session_ttl_seconds)
        return response

    @app.post("/auth/callback/{provider_name}")
    async def saml_callback(provider_name: str, request: Request) -> Response:
        provider = runtime.auth.provider(provider_name)
        if provider.kind != "saml":
            raise HTTPException(400, "not a SAML provider")
        form = dict(await request.form())
        try:
            value = SAMLService(provider, config.public_url).callback(_saml_request(request, form))
        except (ValueError, SAMLUnavailable) as error:
            raise HTTPException(400, str(error)) from error
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE, runtime.auth.codec.encode(value), httponly=True, secure=config.auth.cookie_secure, samesite="lax", max_age=config.auth.session_ttl_seconds)
        return response

    @app.post("/api/auth/logout")
    async def auth_logout(response: Response, _: Identity = Depends(csrf)) -> dict[str, bool]:
        response.delete_cookie(COOKIE)
        return {"logged_out": True}

    @app.get("/api/fleet")
    async def fleet(_: Identity = Depends(require(Permission.FLEET_READ))) -> dict[str, Any]:
        return await runtime.fleet.overview()

    @app.get("/api/engines/{name}/{section}")
    async def engine_section(name: str, section: str, _: Identity = Depends(require(Permission.FLEET_READ))) -> Any:
        try:
            return await runtime.fleet.engine_section(name, section)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ServiceClientError as error:
            raise HTTPException(502, str(error)) from error

    @app.post("/api/engines", status_code=201)
    async def engine_put(body: ManualEngineBody, request: Request, actor: Identity = Depends(csrf_permission(Permission.FLEET_ADMIN))) -> dict[str, Any]:
        try:
            EngineTargetConfig(name=body.name, management_url=body.management_url, token_env=body.token_env, **body.metadata)
        except (TypeError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        values = {"name": body.name, "management_url": body.management_url, "token_env": body.token_env, "metadata_payload": body.metadata}
        before, after = runtime.store.put_engine(values)
        audit(actor, "engine.register", body.name, body.reason, "success", before=before, after=after, request=request)
        return after

    @app.delete("/api/engines/{name}")
    async def engine_delete(name: str, body: MutationBody, request: Request, actor: Identity = Depends(csrf_permission(Permission.FLEET_ADMIN))) -> dict[str, bool]:
        if not body.confirmed:
            raise HTTPException(409, "engine removal requires confirmed=true")
        before = runtime.store.delete_engine(name)
        if not before:
            raise HTTPException(404, "manual engine not found")
        audit(actor, "engine.remove", name, body.reason, "success", before=before, request=request)
        return {"removed": True}

    @app.post("/api/engines/{name}/actions/{action}")
    async def engine_action(name: str, action: str, body: MutationBody, request: Request, actor: Identity = Depends(csrf_permission(Permission.ENGINE_ACTION))) -> Any:
        if action in HIGH_IMPACT_ACTIONS and not body.confirmed:
            raise HTTPException(409, f"{action} requires confirmation")
        try:
            result = await runtime.fleet.action(name, action, {**body.values, "confirmed": body.confirmed})
            audit(actor, f"engine.{action}", name, body.reason, "success", after=result, request=request)
            return result
        except Exception as error:
            audit(actor, f"engine.{action}", name, body.reason, "failure", after={"error": str(error)}, request=request)
            raise HTTPException(502, str(error)) from error

    @app.patch("/api/engines/{name}/config")
    async def engine_config(name: str, body: MutationBody, request: Request, actor: Identity = Depends(csrf_permission(Permission.ENGINE_CONFIGURE))) -> Any:
        try:
            before = await runtime.fleet.engine_section(name, "config")
            result = await runtime.fleet.patch_config(name, body.values)
            audit(actor, "engine.config.patch", name, body.reason, "success", before=before, after=result, request=request)
            return result
        except Exception as error:
            audit(actor, "engine.config.patch", name, body.reason, "failure", after={"error": str(error)}, request=request)
            raise HTTPException(502, str(error)) from error

    @app.get("/api/registry/{resource}")
    async def registry_list(resource: str, limit: int = 200, offset: int = 0, _: Identity = Depends(require(Permission.REGISTRY_READ))) -> Any:
        _registry_resource(resource)
        if not runtime.fleet.registry:
            raise HTTPException(503, "Registry is not configured")
        return await runtime.fleet.registry.list(resource, limit=min(limit, 500), offset=offset)

    @app.post("/api/registry/{resource}", status_code=201)
    async def registry_create(resource: str, body: RegistryMutationBody, request: Request, actor: Identity = Depends(csrf_permission(Permission.REGISTRY_WRITE))) -> Any:
        _registry_resource(resource, mutable=True)
        return await _registry_mutation(runtime, actor, request, "POST", f"/v1/{resource}", body.values, body.reason, f"registry.{resource}.create")

    @app.patch("/api/registry/{resource}/{resource_id}")
    async def registry_patch(resource: str, resource_id: str, body: RegistryMutationBody, request: Request, actor: Identity = Depends(csrf_permission(Permission.REGISTRY_WRITE))) -> Any:
        _registry_resource(resource, mutable=True)
        path = f"/v1/{resource}/{quote(resource_id, safe='')}"
        before = await runtime.fleet.registry.request("GET", path) if runtime.fleet.registry else None
        return await _registry_mutation(runtime, actor, request, "PATCH", path, body.values, body.reason, f"registry.{resource}.patch", before=before)

    @app.post("/api/registry/{resource}/{resource_id}/{transition}")
    async def registry_transition(resource: str, resource_id: str, transition: str, body: RegistryMutationBody, request: Request, actor: Identity = Depends(csrf_permission(Permission.APPROVE))) -> Any:
        _registry_resource(resource, mutable=True)
        if transition not in APPROVAL_TRANSITIONS:
            raise HTTPException(404, "unsupported transition")
        states = {"approve": "APPROVED", "promote": "APPROVED", "deprecate": "DEPRECATED", "revoke": "REVOKED"}
        singular = {"compatibility": "compatibility", "policies": "policy", "qualifications": "qualification"}.get(resource, resource.rstrip("s"))
        values = {
            "resource_type": singular, "resource_id": resource_id,
            "version": str(body.values.get("version", "current")), "state": states[transition],
            "approver": actor.subject, "reason": body.reason,
        }
        return await _registry_mutation(runtime, actor, request, "POST", "/v1/approvals", values, body.reason, f"registry.{resource}.{transition}")

    @app.get("/api/audit")
    async def audit_events(limit: int = 200, offset: int = 0, _: Identity = Depends(require(Permission.AUDIT_READ))) -> dict[str, Any]:
        return runtime.store.audit_events(limit=min(limit, 500), offset=offset)

    @app.get("/api/observability/links")
    async def observability_links(engine: str | None = None, trace_id: str | None = None, _: Identity = Depends(require(Permission.OBSERVABILITY_READ))) -> dict[str, Any]:
        suffix = f"?var-engine={quote(engine)}" if engine else ""
        trace = f"/explore?traceId={quote(trace_id)}" if trace_id else ""
        return {
            "grafana": f"{config.grafana.url}{suffix}" if config.grafana.url else None,
            "tempo": f"{config.tempo.url}{trace}" if config.tempo.url else None,
            "prometheus": config.prometheus.url,
        }

    @app.get("/api/recommendations")
    async def recommendations(_: Identity = Depends(require(Permission.FLEET_READ))) -> dict[str, Any]:
        overview = await runtime.fleet.overview()
        items = []
        for row in overview["items"]:
            if row["status"] == "DRIFT":
                items.append({"engine": row["name"], "kind": "reconcile", "approval_required": True, "reason": "observed state differs from Registry intent"})
            if row["status"] == "OFFLINE":
                items.append({"engine": row["name"], "kind": "investigate", "approval_required": True, "reason": "management endpoint is offline"})
            if float(row.get("metrics", {}).get("storage_reloads") or 0) > 10:
                items.append({"engine": row["name"], "kind": "warm-quota", "approval_required": True, "reason": "high storage reload count"})
        return {"items": items, "mode": "recommendation-only"}

    @app.websocket("/ws/agent")
    async def agent_socket(websocket: WebSocket) -> None:
        value = _websocket_identity(websocket, runtime)
        if not value or not permits(value.role, Permission.FLEET_READ):
            await websocket.close(code=4401)
            return
        session = runtime.replay.open(value, websocket.query_params.get("resume_token"))
        after = int(websocket.query_params.get("after", "0") or 0)
        await websocket.accept()
        await websocket.send_json({"type": "session", "resume_token": session["resume_token"]})
        for event in runtime.replay.replay(session, after):
            await websocket.send_json(event)

        async def emit(kind: str, payload: dict[str, Any]) -> None:
            event = runtime.replay.append(session, kind, payload)
            await websocket.send_json(event)

        heartbeat = asyncio.create_task(_heartbeat(websocket))
        try:
            while True:
                packet = await websocket.receive_json()
                if packet.get("type") == "pong":
                    continue
                if packet.get("type") != "message":
                    await emit("error", {"detail": "unsupported message type"})
                    continue
                message_id = str(packet.get("message_id") or secrets.token_hex(12))
                if runtime.replay.seen(session, message_id):
                    await websocket.send_json({"type": "ack", "message_id": message_id, "duplicate": True})
                    continue
                await websocket.send_json({"type": "ack", "message_id": message_id, "duplicate": False})
                try:
                    answer = await runtime.agent.answer(value, str(packet.get("text", "")), emit)
                    words = answer.split(" ")
                    for index, word in enumerate(words):
                        await emit("message.delta", {"request_message_id": message_id, "text": word + (" " if index < len(words) - 1 else "")})
                    await emit("message.completed", {"request_message_id": message_id, "text": answer})
                except Exception as error:
                    await emit("error", {"request_message_id": message_id, "detail": str(error)})
        except WebSocketDisconnect:
            pass
        finally:
            heartbeat.cancel()

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static / "index.html")

    return app


async def _registry_mutation(runtime: ControlRuntime, actor: Identity, request: Request, method: str, path: str, values: dict[str, Any], reason: str, action: str, *, before: Any = None) -> Any:
    if not runtime.fleet.registry:
        raise HTTPException(503, "Registry is not configured")
    try:
        result = await runtime.fleet.registry.request(method, path, values)
        runtime.store.audit(actor=actor.subject, role=actor.role.value, action=action, target=path, before=before, after=result, reason=reason, trace_id=request.headers.get("traceparent"), result="success")
        return result
    except Exception as error:
        runtime.store.audit(actor=actor.subject, role=actor.role.value, action=action, target=path, before=before, after={"error": str(error)}, reason=reason, trace_id=request.headers.get("traceparent"), result="failure")
        raise HTTPException(502, str(error)) from error


def _registry_resource(resource: str, *, mutable: bool = False) -> None:
    if resource not in REGISTRY_RESOURCES or (mutable and resource in {"audit", "instances"}):
        raise HTTPException(404, "unknown Registry resource")


def _websocket_identity(websocket: WebSocket, runtime: ControlRuntime) -> Identity | None:
    token = websocket.cookies.get(COOKIE)
    return runtime.auth.codec.decode(token) if token else runtime.auth.development_identity()


async def _heartbeat(websocket: WebSocket) -> None:
    try:
        while True:
            await asyncio.sleep(20)
            await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, RuntimeError):
        return


def _saml_request(request: Request, form: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "https": "on" if request.url.scheme == "https" else "off",
        "http_host": request.url.hostname, "server_port": request.url.port,
        "script_name": request.url.path, "get_data": dict(request.query_params),
        "post_data": form or {},
    }
