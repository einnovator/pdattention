"""FastAPI application for fleet governance, Registry workflows, and PRA chat."""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import AgentReplayService, ControlPlaneAgent
from .auth import AuthService, Identity
from .clients import ServiceClientError
from .config import ControlPlaneConfig
from .domain import ControlError, domain_payload
from .fleet import FleetService
from .managers import ControlManager
from .operations import OPERATION_CATALOG, TOOL_CATALOG, allowed
from .persistence import ControlStore
from .rbac import Permission, permits
from .saml import SAMLService, SAMLUnavailable


COOKIE = "pra_control_session"
OAUTH_COOKIE = "pra_control_oauth"
LOGGER = logging.getLogger(__name__)


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
        self.manager = ControlManager.build(config, self.store, self.fleet)
        self.replay = AgentReplayService(self.store, config.replay_limit)
        self.agent = ControlPlaneAgent(self.manager)

    def bind_manager(self) -> None:
        """Rebind after tests or embedders replace a physical fleet backend."""
        if self.manager.fleet.backend is not self.fleet:
            self.manager = ControlManager.build(self.config, self.store, self.fleet)
            self.agent = ControlPlaneAgent(self.manager)


def create_app(config: ControlPlaneConfig | None = None, *, runtime: ControlRuntime | None = None) -> FastAPI:
    config = config or ControlPlaneConfig.load()
    runtime = runtime or ControlRuntime(config)
    runtime.bind_manager()
    mcp_http_app = None
    if config.mcp.enabled and config.mcp.transports.http.enabled:
        from .mcp import build_http_app
        mcp_http_app = build_http_app(runtime.manager, config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with AsyncExitStack() as stack:
            inner_mcp = getattr(mcp_http_app, "app", None)
            if inner_mcp is not None and hasattr(inner_mcp, "router"):
                await stack.enter_async_context(inner_mcp.router.lifespan_context(inner_mcp))
            summary = app.state.capability_summary
            LOGGER.info(
                "PRA Control capabilities: REST=%d operations; MCP=%d tools, %d resource templates",
                summary["rest_operations"], summary["mcp_tools"], summary["mcp_resource_templates"],
            )
            yield
            await runtime.fleet.close()

    app = FastAPI(title="eInnovator PRA Control Plane", version="0.1.0", lifespan=lifespan)
    app.state.control = runtime
    static = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static), name="static")
    if mcp_http_app is not None:
        app.mount(config.mcp.transports.http.path, mcp_http_app, name="mcp")
    app.state.capability_summary = {
        "rest_operations": sum(
            config.rest.enabled and allowed(item.id, config.rest.allow, config.rest.deny)
            for item in OPERATION_CATALOG
        ),
        "mcp_tools": sum(
            config.mcp.enabled and allowed(item.name, config.mcp.tools.allow, config.mcp.tools.deny)
            for item in TOOL_CATALOG
        ),
        "mcp_resource_templates": len(config.mcp.resources.allow) if config.mcp.enabled and config.mcp.resources.enabled else 0,
    }

    @app.exception_handler(ControlError)
    async def control_error(_request: Request, error: ControlError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": str(error), "details": error.details}},
        )

    def rest(operation: str | tuple[str, ...], method: str, path: str, **kwargs: Any):
        """Register only operations enabled by the semantic exposure policy."""
        def decorate(function: Callable[..., Any]):
            operations = (operation,) if isinstance(operation, str) else operation
            if config.rest.enabled and any(allowed(item, config.rest.allow, config.rest.deny) for item in operations):
                getattr(app, method)(path, **kwargs)(function)
            return function
        return decorate

    def identity(request: Request) -> Identity:
        token = request.cookies.get(COOKIE)
        value = runtime.auth.codec.decode(token) if token else runtime.auth.development_identity()
        if not value:
            raise HTTPException(401, "authentication required")
        return value

    def csrf(request: Request, value: Identity = Depends(identity)) -> Identity:
        if runtime.auth.development_identity() == value:
            return value
        supplied = request.headers.get("X-CSRF-Token", "")
        if not supplied or not secrets.compare_digest(supplied, value.csrf_token):
            raise HTTPException(403, "invalid CSRF token")
        return value

    def caller(actor: Identity, request: Request, *, transport: str = "rest"):
        return actor.caller(
            transport=transport,
            request_id=request.headers.get("X-Request-ID") or secrets.token_hex(12),
            trace_id=request.headers.get("traceparent"),
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

    @rest("fleet.list", "get", "/api/fleet")
    async def fleet(request: Request, actor: Identity = Depends(identity)) -> dict[str, Any]:
        return domain_payload(await runtime.manager.fleet.list(caller(actor, request)))

    @rest("engine.inspect", "get", "/api/engines/{name}/{section}")
    async def engine_section(name: str, section: str, request: Request, actor: Identity = Depends(identity)) -> Any:
        result = await runtime.manager.fleet.inspect(caller(actor, request), name, section)
        return domain_payload(result.value)

    @rest("engine.register", "post", "/api/engines", status_code=201)
    async def engine_put(body: ManualEngineBody, request: Request, actor: Identity = Depends(csrf)) -> dict[str, Any]:
        return await runtime.manager.fleet.register(caller(actor, request), body.model_dump(), reason=body.reason)

    @rest("engine.remove", "delete", "/api/engines/{name}")
    async def engine_delete(name: str, body: MutationBody, request: Request, actor: Identity = Depends(csrf)) -> dict[str, bool]:
        return await runtime.manager.fleet.remove(caller(actor, request), name, reason=body.reason, confirmed=body.confirmed)

    @rest("action.apply", "post", "/api/engines/{name}/actions/{action}")
    async def engine_action(name: str, action: str, body: MutationBody, request: Request, actor: Identity = Depends(csrf)) -> Any:
        result = await runtime.manager.actions.execute(
            caller(actor, request), action, name, body.values, reason=body.reason,
            confirmed=body.confirmed, idempotency_key=request.headers.get("Idempotency-Key"),
        )
        return domain_payload(result.result)

    @rest("engine.config.patch", "patch", "/api/engines/{name}/config")
    async def engine_config(name: str, body: MutationBody, request: Request, actor: Identity = Depends(csrf)) -> Any:
        result = await runtime.manager.actions.execute(
            caller(actor, request), "config-patch", name, body.values, reason=body.reason,
            confirmed=body.confirmed, idempotency_key=request.headers.get("Idempotency-Key"),
        )
        return domain_payload(result.result)

    @rest("registry.list", "get", "/api/registry/{resource}")
    async def registry_list(resource: str, request: Request, limit: int = 200, offset: int = 0, actor: Identity = Depends(identity)) -> Any:
        return await runtime.manager.registry.list(caller(actor, request), resource, limit=limit, offset=offset)

    @rest("registry.write", "post", "/api/registry/{resource}", status_code=201)
    async def registry_create(resource: str, body: RegistryMutationBody, request: Request, actor: Identity = Depends(csrf)) -> Any:
        return await runtime.manager.registry.create(caller(actor, request), resource, body.values, reason=body.reason)

    @rest("registry.write", "patch", "/api/registry/{resource}/{resource_id}")
    async def registry_patch(resource: str, resource_id: str, body: RegistryMutationBody, request: Request, actor: Identity = Depends(csrf)) -> Any:
        return await runtime.manager.registry.patch(caller(actor, request), resource, resource_id, body.values, reason=body.reason)

    @rest(("registry.write", "qualification.approve"), "post", "/api/registry/{resource}/{resource_id}/{transition}")
    async def registry_transition(resource: str, resource_id: str, transition: str, body: RegistryMutationBody, request: Request, actor: Identity = Depends(csrf)) -> Any:
        semantic_operation = "qualification.approve" if resource == "qualifications" else "registry.write"
        if not allowed(semantic_operation, config.rest.allow, config.rest.deny):
            raise HTTPException(404, "operation disabled")
        return await runtime.manager.registry.transition(
            caller(actor, request), resource, resource_id, transition, body.values, reason=body.reason,
        )

    @rest("audit.read", "get", "/api/audit")
    async def audit_events(request: Request, limit: int = 200, offset: int = 0, actor: Identity = Depends(identity)) -> dict[str, Any]:
        return runtime.manager.audit.list(caller(actor, request), limit=limit, offset=offset)

    @rest("observability.read", "get", "/api/observability/links")
    async def observability_links(request: Request, engine: str | None = None, trace_id: str | None = None, actor: Identity = Depends(identity)) -> dict[str, Any]:
        return runtime.manager.observability.links(caller(actor, request), engine=engine, trace_id=trace_id)

    @rest("fleet.list", "get", "/api/recommendations")
    async def recommendations(request: Request, actor: Identity = Depends(identity)) -> dict[str, Any]:
        return await runtime.manager.fleet.recommendations(caller(actor, request))

    @rest("action.plan", "post", "/api/actions/plan")
    async def action_plan(body: MutationBody, request: Request, action: str, target: str, actor: Identity = Depends(csrf)) -> Any:
        return domain_payload(await runtime.manager.actions.plan(
            caller(actor, request), action, target, body.values,
            idempotency_key=request.headers.get("Idempotency-Key"),
        ))

    @rest("action.apply", "post", "/api/actions/plans/{plan_id}/apply")
    async def action_apply(plan_id: str, body: MutationBody, request: Request, actor: Identity = Depends(csrf)) -> Any:
        return domain_payload(await runtime.manager.actions.apply(
            caller(actor, request), plan_id, confirmation=body.confirmed, reason=body.reason,
            idempotency_key=request.headers.get("Idempotency-Key"),
        ))

    @rest("deployment.read", "get", "/api/deployments")
    async def deployments(request: Request, actor: Identity = Depends(identity)) -> Any:
        return {"items": await runtime.manager.deployments.list(caller(actor, request))}

    @rest("deployment.read", "get", "/api/deployments/{deployment_id}")
    async def deployment(deployment_id: str, request: Request, actor: Identity = Depends(identity)) -> Any:
        return await runtime.manager.deployments.get(caller(actor, request), deployment_id)

    @rest("qualification.read", "get", "/api/qualifications")
    async def qualifications(request: Request, model: str | None = None, engine: str | None = None, actor: Identity = Depends(identity)) -> Any:
        return {"items": await runtime.manager.qualifications.list_evidence(caller(actor, request), model=model, engine=engine)}

    @rest("context.read", "post", "/api/context")
    async def task_context(body: dict[str, Any], request: Request, actor: Identity = Depends(identity)) -> Any:
        return domain_payload(await runtime.manager.context.assemble(
            caller(actor, request), task=str(body.get("task", "")), repository=body.get("repository"),
        ))

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
