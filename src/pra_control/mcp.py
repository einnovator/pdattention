"""MCP presentation adapter over the canonical PRA ControlManager."""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
from typing import Any, Awaitable, Callable, Mapping
from urllib.parse import unquote, urlparse

from .config import ControlAuthProfile, ControlPlaneConfig
from .domain import CallerContext, ControlError, Forbidden, InvalidRequest, NotFound, domain_payload
from .managers import ControlManager
from .operations import TOOL_CATALOG, ToolSpec, allowed
from .rbac import ROLE_PERMISSIONS, Role


RESOURCE_TEMPLATES: tuple[str, ...] = (
    "pra://fleet",
    "pra://engines/{instance_id}",
    "pra://engines/{instance_id}/models/{runtime_model_id}",
    "pra://models/{model_id}",
    "pra://bundles/{bundle_id}",
    "pra://qualifications/{qualification_id}",
    "pra://deployments/{deployment_id}",
)

_ACTIVE_CALLER: contextvars.ContextVar[CallerContext | None] = contextvars.ContextVar(
    "pra_control_mcp_caller", default=None,
)


def caller_from_profile(
    profile: ControlAuthProfile, *, transport: str, supplied_token: str | None = None,
    request_id: str | None = None, trace_id: str | None = None,
) -> CallerContext:
    """Resolve a secret-referencing auth profile without exposing its credentials."""
    if profile.type in {"bearer_token", "client_credentials"}:
        expected = profile.token()
        if not expected or not supplied_token or not hmac.compare_digest(expected, supplied_token):
            raise Forbidden("invalid MCP bearer credential")
    elif profile.type == "oidc":
        raise Forbidden("OIDC profiles require the MCP HTTP verifier")
    permissions = {
        permission.value
        for role in profile.roles
        for permission in ROLE_PERMISSIONS[role]
    }
    return CallerContext(
        subject=profile.subject or "mcp-service", roles=[role.value for role in profile.roles],
        permissions=permissions, auth_source=profile.type, tenant=profile.tenant,
        request_id=request_id, trace_id=trace_id, transport=transport,
    )


class MCPPresentation:
    """Protocol-independent MCP semantics used by both official SDK transports."""

    def __init__(
        self, manager: ControlManager, config: ControlPlaneConfig,
        caller_provider: Callable[[], CallerContext],
    ) -> None:
        self.manager = manager
        self.config = config
        self.caller_provider = caller_provider

    @property
    def tools(self) -> list[ToolSpec]:
        return [
            spec for spec in TOOL_CATALOG
            if allowed(spec.name, self.config.mcp.tools.allow, self.config.mcp.tools.deny)
        ]

    @property
    def resources(self) -> list[str]:
        if not self.config.mcp.resources.enabled:
            return []
        return [
            uri for uri in RESOURCE_TEMPLATES
            if any(_resource_pattern_matches(pattern, uri) for pattern in self.config.mcp.resources.allow)
        ]

    def discovery(self) -> dict[str, Any]:
        return {
            "tools": [spec.model_dump(mode="json") for spec in self.tools],
            "resource_templates": self.resources,
            "read_only_default": "pra_apply" not in {spec.name for spec in self.tools},
        }

    async def call(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if name not in {spec.name for spec in self.tools}:
            return _error(NotFound(f"MCP tool is disabled or unknown: {name}"))
        caller = self.caller_provider()
        args = dict(arguments or {})
        try:
            if name == "pra_fleet":
                result = await self.manager.fleet.find(
                    caller, query=str(args.get("query", "")), engine=args.get("engine"), model=args.get("model"),
                )
            elif name in {"pra_engine", "pra_gateway"}:
                result = await self.manager.fleet.inspect(
                    caller, str(args["instance_id"]), str(args.get("section", "summary")),
                )
            elif name == "pra_catalog":
                result = await self.manager.registry.list(
                    caller, str(args.get("resource", "models")),
                    limit=int(args.get("limit", 100)), offset=int(args.get("offset", 0)),
                )
            elif name == "pra_qualification":
                result = await self.manager.qualifications.get_support_status(
                    caller, str(args["model"]), str(args["engine"]), hardware=args.get("hardware"),
                )
            elif name == "pra_deployment":
                deployment_id = args.get("deployment_id")
                result = (
                    await self.manager.deployments.get(caller, str(deployment_id))
                    if deployment_id else {"items": await self.manager.deployments.list(caller)}
                )
            elif name == "pra_metrics":
                result = await self.manager.observability.summary(
                    caller, engine=args.get("engine"), period=str(args.get("period", "15m")),
                )
            elif name == "pra_context":
                result = await self.manager.context.assemble(
                    caller, task=str(args["task"]), repository=args.get("repository"),
                )
            elif name == "pra_plan":
                result = await self.manager.actions.plan(
                    caller, str(args["action"]), str(args["target"]),
                    dict(args.get("requested_change") or {}), idempotency_key=args.get("idempotency_key"),
                )
            elif name == "pra_apply":
                result = await self.manager.actions.apply(
                    caller, str(args["plan_id"]), confirmation=bool(args.get("confirmation", False)),
                    reason=str(args.get("reason", "MCP-approved action")),
                    idempotency_key=args.get("idempotency_key"),
                )
            elif name == "pra_experiment":
                if args.get("submit"):
                    result = await self.manager.experiments.submit(caller, dict(args.get("request") or {}))
                else:
                    result = {"items": await self.manager.experiments.list(caller)}
            else:
                raise NotFound(f"unknown MCP tool: {name}")
            return {"ok": True, "result": domain_payload(result)}
        except ControlError as error:
            return _error(error)
        except (KeyError, TypeError, ValueError) as error:
            return _error(InvalidRequest(f"invalid {name} arguments: {error}"))

    async def read(self, uri: str) -> dict[str, Any]:
        if not any(_template_matches(template, uri) for template in self.resources):
            return _error(NotFound(f"MCP resource is disabled or unknown: {uri}"))
        caller = self.caller_provider()
        parsed = urlparse(uri)
        parts = [parsed.netloc, *[unquote(part) for part in parsed.path.split("/") if part]]
        try:
            if parts == ["fleet"]:
                result = await self.manager.fleet.list(caller)
            elif parts and parts[0] == "engines" and len(parts) == 2:
                result = await self.manager.fleet.inspect(caller, parts[1], "summary")
            elif parts and parts[0] == "engines" and len(parts) == 4 and parts[2] == "models":
                models = await self.manager.fleet.list_models(caller, parts[1])
                result = next((row for row in models if str(row.get("runtime_model_id")) == parts[3]), None)
                if result is None:
                    raise NotFound(f"runtime model not found: {parts[3]}")
            elif parts and parts[0] in {"models", "bundles", "qualifications", "deployments"} and len(parts) == 2:
                page = await self.manager.registry.list(caller, parts[0], limit=500)
                result = next((row for row in page.get("items", []) if str(row.get("id")) == parts[1]), None)
                if result is None:
                    raise NotFound(f"resource not found: {uri}")
            else:
                raise NotFound(f"unknown MCP resource: {uri}")
            return {"ok": True, "result": domain_payload(result)}
        except ControlError as error:
            return _error(error)


def build_fastmcp(
    presentation: MCPPresentation, *, host: str = "127.0.0.1", port: int = 9301,
    streamable_http_path: str = "/mcp", max_request_body_size: int = 4_194_304,
):
    """Build an official Python MCP SDK server with filtered discovery."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError("Install the 'control-mcp' optional dependency") from error

    server = FastMCP(
        "PRA Control Manager", instructions="Read and operate PRA through the canonical ControlManager.",
        host=host, port=port, streamable_http_path=streamable_http_path,
        stateless_http=True, json_response=True, max_request_body_size=max_request_body_size,
    )
    enabled = {spec.name for spec in presentation.tools}

    if "pra_fleet" in enabled:
        @server.tool(name="pra_fleet", description="List and filter managed PRA instances", structured_output=True)
        async def pra_fleet(query: str = "", engine: str | None = None, model: str | None = None) -> dict[str, Any]:
            return await presentation.call("pra_fleet", {"query": query, "engine": engine, "model": model})

    if "pra_engine" in enabled:
        @server.tool(name="pra_engine", description="Inspect one PRA engine", structured_output=True)
        async def pra_engine(instance_id: str, section: str = "summary") -> dict[str, Any]:
            return await presentation.call("pra_engine", {"instance_id": instance_id, "section": section})

    if "pra_gateway" in enabled:
        @server.tool(name="pra_gateway", description="Inspect one PRA gateway", structured_output=True)
        async def pra_gateway(instance_id: str, section: str = "summary") -> dict[str, Any]:
            return await presentation.call("pra_gateway", {"instance_id": instance_id, "section": section})

    if "pra_catalog" in enabled:
        @server.tool(name="pra_catalog", description="Read PRA Registry catalog resources", structured_output=True)
        async def pra_catalog(resource: str = "models", limit: int = 100, offset: int = 0) -> dict[str, Any]:
            return await presentation.call("pra_catalog", {"resource": resource, "limit": limit, "offset": offset})

    if "pra_qualification" in enabled:
        @server.tool(name="pra_qualification", description="Read model-engine support evidence", structured_output=True)
        async def pra_qualification(model: str, engine: str, hardware: str | None = None) -> dict[str, Any]:
            return await presentation.call("pra_qualification", {"model": model, "engine": engine, "hardware": hardware})

    if "pra_deployment" in enabled:
        @server.tool(name="pra_deployment", description="Read desired deployment state", structured_output=True)
        async def pra_deployment(deployment_id: str | None = None) -> dict[str, Any]:
            return await presentation.call("pra_deployment", {"deployment_id": deployment_id})

    if "pra_metrics" in enabled:
        @server.tool(name="pra_metrics", description="Read semantic metrics and observability links", structured_output=True)
        async def pra_metrics(engine: str | None = None, period: str = "15m") -> dict[str, Any]:
            return await presentation.call("pra_metrics", {"engine": engine, "period": period})

    if "pra_context" in enabled:
        @server.tool(name="pra_context", description="Assemble deterministic task context", structured_output=True)
        async def pra_context(task: str, repository: str | None = None) -> dict[str, Any]:
            return await presentation.call("pra_context", {"task": task, "repository": repository})

    if "pra_plan" in enabled:
        @server.tool(name="pra_plan", description="Plan an operational action without applying it", structured_output=True)
        async def pra_plan(action: str, target: str, requested_change: dict[str, Any] | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
            return await presentation.call("pra_plan", {"action": action, "target": target, "requested_change": requested_change or {}, "idempotency_key": idempotency_key})

    if "pra_apply" in enabled:
        @server.tool(name="pra_apply", description="Apply a confirmed action plan", structured_output=True)
        async def pra_apply(plan_id: str, confirmation: bool, reason: str, idempotency_key: str | None = None) -> dict[str, Any]:
            return await presentation.call("pra_apply", {"plan_id": plan_id, "confirmation": confirmation, "reason": reason, "idempotency_key": idempotency_key})

    if "pra_experiment" in enabled:
        @server.tool(name="pra_experiment", description="Inspect or submit enabled research experiments", structured_output=True)
        async def pra_experiment(submit: bool = False, request: dict[str, Any] | None = None) -> dict[str, Any]:
            return await presentation.call("pra_experiment", {"submit": submit, "request": request or {}})

    for template in presentation.resources:
        _register_resource(server, presentation, template)
    return server


def build_http_app(manager: ControlManager, config: ControlPlaneConfig):
    """Return authenticated streamable-HTTP MCP ASGI application for mounting."""
    settings = config.mcp.transports.http
    profile = config.auth_profiles[settings.auth_profile or ""]

    def active() -> CallerContext:
        caller = _ACTIVE_CALLER.get()
        if caller is None:
            raise Forbidden("MCP request has no authenticated caller")
        return caller

    presentation = MCPPresentation(manager, config, active)
    server = build_fastmcp(
        presentation, host=settings.host, port=settings.port, streamable_http_path="/",
        max_request_body_size=settings.max_request_bytes,
    )
    return MCPAuthMiddleware(server.streamable_http_app(), profile)


class MCPAuthMiddleware:
    """Authenticate MCP HTTP before tool code receives a CallerContext."""

    def __init__(self, app: Any, profile: ControlAuthProfile) -> None:
        self.app = app
        self.profile = profile

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[Any]]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        bearer = headers.get("authorization", "")
        supplied = bearer[7:] if bearer.lower().startswith("bearer ") else None
        try:
            caller = await _http_caller(
                self.profile, headers=headers, supplied_token=supplied,
                request_id=headers.get("x-request-id"), trace_id=headers.get("traceparent"),
            )
        except ControlError as error:
            body = json.dumps(_error(error)).encode("utf-8")
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        token = _ACTIVE_CALLER.set(caller)
        try:
            await self.app(scope, receive, send)
        finally:
            _ACTIVE_CALLER.reset(token)


def stdio_presentation(manager: ControlManager, config: ControlPlaneConfig) -> tuple[MCPPresentation, Any]:
    settings = config.mcp.transports.stdio
    profile = config.auth_profiles[settings.auth_profile or ""]
    caller = caller_from_profile(profile, transport="mcp-stdio", supplied_token=profile.token())
    presentation = MCPPresentation(manager, config, lambda: caller)
    return presentation, build_fastmcp(presentation)


def _register_resource(server: Any, presentation: MCPPresentation, template: str) -> None:
    if template == "pra://fleet":
        @server.resource(template, name="PRA fleet", mime_type="application/json")
        async def fleet_resource() -> str:
            return json.dumps(await presentation.read("pra://fleet"), sort_keys=True)
        return
    if template == "pra://engines/{instance_id}":
        @server.resource(template, name="PRA engine", mime_type="application/json")
        async def engine_resource(instance_id: str) -> str:
            return json.dumps(await presentation.read(f"pra://engines/{instance_id}"), sort_keys=True)
    elif template == "pra://engines/{instance_id}/models/{runtime_model_id}":
        @server.resource(template, name="PRA runtime model", mime_type="application/json")
        async def runtime_model_resource(instance_id: str, runtime_model_id: str) -> str:
            return json.dumps(await presentation.read(f"pra://engines/{instance_id}/models/{runtime_model_id}"), sort_keys=True)
    elif template == "pra://models/{model_id}":
        @server.resource(template, name="PRA model", mime_type="application/json")
        async def model_resource(model_id: str) -> str:
            return json.dumps(await presentation.read(f"pra://models/{model_id}"), sort_keys=True)
    elif template == "pra://bundles/{bundle_id}":
        @server.resource(template, name="PRA bundle", mime_type="application/json")
        async def bundle_resource(bundle_id: str) -> str:
            return json.dumps(await presentation.read(f"pra://bundles/{bundle_id}"), sort_keys=True)
    elif template == "pra://qualifications/{qualification_id}":
        @server.resource(template, name="PRA qualification", mime_type="application/json")
        async def qualification_resource(qualification_id: str) -> str:
            return json.dumps(await presentation.read(f"pra://qualifications/{qualification_id}"), sort_keys=True)
    elif template == "pra://deployments/{deployment_id}":
        @server.resource(template, name="PRA deployment", mime_type="application/json")
        async def deployment_resource(deployment_id: str) -> str:
            return json.dumps(await presentation.read(f"pra://deployments/{deployment_id}"), sort_keys=True)


def _error(error: ControlError) -> dict[str, Any]:
    return {"ok": False, "error": {"code": error.code, "message": str(error), "details": error.details}}


def _resource_pattern_matches(pattern: str, template: str) -> bool:
    normalized = template
    while "{" in normalized:
        start = normalized.index("{")
        end = normalized.index("}", start)
        normalized = normalized[:start] + "item" + normalized[end + 1:]
    return allowed(normalized, [pattern], [])


def _template_matches(template: str, uri: str) -> bool:
    left = template.split("/")
    right = uri.split("/")
    return len(left) == len(right) and all(a == b or (a.startswith("{") and a.endswith("}")) for a, b in zip(left, right))


async def _http_caller(
    profile: ControlAuthProfile, *, headers: Mapping[str, str], supplied_token: str | None,
    request_id: str | None, trace_id: str | None,
) -> CallerContext:
    if profile.type == "oidc":
        if not supplied_token:
            raise Forbidden("MCP OIDC bearer token is required")
        claims = await asyncio.to_thread(_decode_oidc, profile, supplied_token)
        role_value = claims.get(profile.role_claim)
        values = role_value if isinstance(role_value, list) else [role_value] if role_value else []
        roles = []
        for value in values:
            try:
                roles.append(Role(str(value)))
            except ValueError:
                continue
        roles = roles or profile.roles
        permissions = {permission.value for role in roles for permission in ROLE_PERMISSIONS[role]}
        return CallerContext(
            subject=str(claims.get("sub") or profile.subject or "oidc-service"),
            roles=[role.value for role in roles], permissions=permissions, auth_source="oidc",
            tenant=str(claims.get(profile.tenant_claim)) if claims.get(profile.tenant_claim) is not None else profile.tenant,
            request_id=request_id, trace_id=trace_id, transport="mcp-http",
        )
    if profile.type == "mtls":
        subject = headers.get(profile.mtls_subject_header.lower())
        if not subject or (profile.subject and not hmac.compare_digest(profile.subject, subject)):
            raise Forbidden("trusted MCP client-certificate subject is required")
        effective = profile.model_copy(update={"subject": subject})
        return caller_from_profile(effective, transport="mcp-http", request_id=request_id, trace_id=trace_id)
    return caller_from_profile(
        profile, transport="mcp-http", supplied_token=supplied_token,
        request_id=request_id, trace_id=trace_id,
    )


def _decode_oidc(profile: ControlAuthProfile, encoded: str) -> dict[str, Any]:
    try:
        import jwt
        client = jwt.PyJWKClient(str(profile.jwks_url))
        key = client.get_signing_key_from_jwt(encoded)
        return dict(jwt.decode(encoded, key.key, algorithms=["RS256", "ES256"], audience=profile.audience, issuer=profile.issuer))
    except Exception as error:
        raise Forbidden("invalid MCP OIDC token") from error
