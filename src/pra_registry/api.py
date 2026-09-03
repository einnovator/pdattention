"""FastAPI application for the versioned PRA Registry REST contract."""

from __future__ import annotations

import hmac
import re
import time
from collections import Counter
from contextlib import asynccontextmanager, nullcontext
from typing import Any, Callable

from fastapi import Depends, FastAPI, Query, Request, Security
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import REGISTRY_PROTOCOL
from .config import RegistryConfig
from .contracts import (
    ApprovalCreate,
    ApprovalState,
    BundleCreate,
    BundlePatch,
    BundleResolveRequest,
    CompatibilityCreate,
    DeploymentCreate,
    DeploymentPatch,
    DeploymentResolveRequest,
    ModelCreate,
    ModelPatch,
    HuggingFaceCollectionSyncRequest,
    HuggingFaceImportRequest,
    ManagedInstanceHeartbeat,
    ManagedInstanceObservedPatch,
    ManagedInstanceRegister,
    PolicyCreate,
    PolicyPatch,
    ProfileCreate,
    ProfilePatch,
    ProfileResolveRequest,
    QualificationCreate,
    BackendEndpointCreate,
    BackendEndpointPatch,
    ModelPoolCreate,
    ModelPoolPatch,
    RouteBindingCreate,
    RouteBindingPatch,
    RouteCreate,
    RoutePatch,
    RouterInstanceCreate,
    RouterInstancePatch,
    RoutingPolicyCreate,
    RoutingPolicyPatch,
)
from .database import RegistryDatabase
from .service import RegistryError, RegistryService


bearer = HTTPBearer(auto_error=False)


class RegistryMetrics:
    """Dependency-free default-off counters; Prometheus text is opt-in."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.counters: Counter[tuple[str, str]] = Counter()
        self.latencies: Counter[tuple[str, str]] = Counter()
        self.errors: Counter[str] = Counter()
        self.operations: Counter[str] = Counter()
        self.operation_latencies: Counter[str] = Counter()
        self.instance_events: Counter[tuple[str, str]] = Counter()
        self.instance_status: Counter[str] = Counter()
        self.heartbeat_age_seconds: dict[str, float] = {}

    def observe(self, method: str, route: str, status: int, elapsed: float) -> None:
        if not self.enabled:
            return
        labels = (method, route)
        self.counters[labels] += 1
        self.latencies[labels] += int(elapsed * 1_000_000)
        if status >= 400:
            self.errors[str(status)] += 1

    def operation(self, name: str, elapsed: float) -> None:
        if self.enabled:
            self.operations[name] += 1
            self.operation_latencies[name] += int(elapsed * 1_000_000)

    def instance_event(self, operation: str, result: str) -> None:
        if self.enabled:
            self.instance_events[(operation, result)] += 1

    def instance_snapshot(self, rows: list[dict[str, Any]]) -> None:
        if not self.enabled:
            return
        self.instance_status = Counter(str(row.get("status", "UNKNOWN")) for row in rows)
        now = time.time()
        self.heartbeat_age_seconds = {
            str(row["instance_id"]): max(
                0.0,
                now - float(row["last_heartbeat"].timestamp()),
            )
            for row in rows if row.get("last_heartbeat") is not None
        }

    def render(self) -> str:
        lines = ["# HELP pra_registry_requests_total Registry HTTP requests.", "# TYPE pra_registry_requests_total counter"]
        for (method, route), value in sorted(self.counters.items()):
            lines.append(f'pra_registry_requests_total{{method="{method}",route="{route}"}} {value}')
        lines.extend(["# HELP pra_registry_errors_total Registry HTTP errors.", "# TYPE pra_registry_errors_total counter"])
        for status, value in sorted(self.errors.items()):
            lines.append(f'pra_registry_errors_total{{status="{status}"}} {value}')
        lines.extend(["# HELP pra_registry_request_latency_microseconds_total Accumulated request latency.", "# TYPE pra_registry_request_latency_microseconds_total counter"])
        for (method, route), value in sorted(self.latencies.items()):
            lines.append(f'pra_registry_request_latency_microseconds_total{{method="{method}",route="{route}"}} {value}')
        lines.extend(["# HELP pra_registry_operations_total Registry DB, resolver, approval, and sync operations.", "# TYPE pra_registry_operations_total counter"])
        for operation, value in sorted(self.operations.items()):
            lines.append(f'pra_registry_operations_total{{operation="{operation}"}} {value}')
        lines.extend(["# HELP pra_registry_operation_latency_microseconds_total Accumulated operation latency.", "# TYPE pra_registry_operation_latency_microseconds_total counter"])
        for operation, value in sorted(self.operation_latencies.items()):
            lines.append(f'pra_registry_operation_latency_microseconds_total{{operation="{operation}"}} {value}')
        lines.extend(["# HELP pra_registry_instances_online Managed runtimes currently online.", "# TYPE pra_registry_instances_online gauge"])
        lines.append(f'pra_registry_instances_online {self.instance_status.get("ONLINE", 0)}')
        lines.extend(["# HELP pra_registry_instances_offline Managed runtimes currently offline.", "# TYPE pra_registry_instances_offline gauge"])
        lines.append(f'pra_registry_instances_offline {self.instance_status.get("OFFLINE", 0)}')
        lines.extend(["# HELP pra_registry_registration_total Runtime registration attempts.", "# TYPE pra_registry_registration_total counter"])
        for (operation, result), value in sorted(self.instance_events.items()):
            if operation == "registration":
                lines.append(f'pra_registry_registration_total{{result="{result}"}} {value}')
        lines.extend(["# HELP pra_registry_registration_failures_total Failed runtime registrations.", "# TYPE pra_registry_registration_failures_total counter"])
        lines.append(f'pra_registry_registration_failures_total {self.instance_events.get(("registration", "failure"), 0)}')
        lines.extend(["# HELP pra_registry_heartbeat_failures_total Failed runtime heartbeats.", "# TYPE pra_registry_heartbeat_failures_total counter"])
        lines.append(f'pra_registry_heartbeat_failures_total {self.instance_events.get(("heartbeat", "failure"), 0)}')
        lines.extend(["# HELP pra_registry_heartbeat_age_seconds Age of the newest heartbeat per instance.", "# TYPE pra_registry_heartbeat_age_seconds gauge"])
        for instance_id, value in sorted(self.heartbeat_age_seconds.items()):
            lines.append(f'pra_registry_heartbeat_age_seconds{{instance_id="{instance_id}"}} {value:.6f}')
        return "\n".join(lines) + "\n"


def _identity(config: RegistryConfig, credentials: HTTPAuthorizationCredentials | None) -> tuple[str, set[str]]:
    if config.auth.mode == "none":
        return "local-dev", {"registry:read", "registry:write", "registry:approve", "registry:admin"}
    if credentials is None:
        raise RegistryAuthError("missing bearer token", 401)
    token = credentials.credentials
    if config.auth.mode == "static_token":
        expected = config.auth.token()
        if not expected or not hmac.compare_digest(token, expected):
            raise RegistryAuthError("invalid bearer token", 401)
        return "static-token", {"registry:read", "registry:write", "registry:approve", "registry:admin"}
    for service, secret in config.auth.service_credentials.items():
        if hmac.compare_digest(token, secret.get_secret_value()):
            return service, {"registry:read", "registry:write"}
    if config.auth.mode in {"oidc", "jwt", "oidc_jwt"}:
        try:
            import jwt
            key: Any = jwt.PyJWKClient(config.auth.oidc_jwks_url).get_signing_key_from_jwt(token).key
            payload = jwt.decode(
                token, key=key, algorithms=["RS256", "ES256"],
                audience=config.auth.oidc_audience, issuer=config.auth.oidc_issuer,
            )
        except Exception as error:
            raise RegistryAuthError("invalid OIDC/JWT token", 401) from error
        raw_scopes = payload.get("scope", payload.get("scp", ""))
        scopes = set(raw_scopes.split() if isinstance(raw_scopes, str) else raw_scopes)
        return str(payload.get("sub", "oidc-user")), scopes
    raise RegistryAuthError("unsupported authentication mode", 401)


class RegistryAuthError(RuntimeError):
    def __init__(self, detail: str, status_code: int) -> None:
        super().__init__(detail)
        self.status_code = status_code


def create_registry_app(config: RegistryConfig | None = None, database: RegistryDatabase | None = None) -> FastAPI:
    settings = config or RegistryConfig()
    settings.validate_binding()
    db = database or RegistryDatabase(settings.database_url, create_schema=settings.database_url.startswith("sqlite"))
    metrics = RegistryMetrics(settings.observability.enabled and settings.observability.prometheus_enabled)
    tracer, tracer_provider = _configure_tracing(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        if tracer_provider is not None:
            tracer_provider.shutdown()
        if database is None:
            db.close()

    app = FastAPI(
        title="PRA Registry API", version="1.0.0",
        description="Open metadata, qualification, approval, and desired-state registry for PRA.",
        lifespan=lifespan,
    )
    app.state.registry_database = db
    app.state.registry_config = settings

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[..., Any]):
        started = time.perf_counter()
        span_context = tracer.start_as_current_span(f"registry.http.{request.method.lower()}") if tracer else nullcontext()
        with span_context as span:
            response = await call_next(request)
            route = getattr(request.scope.get("route"), "path", "unmatched")
            if tracer:
                span.set_attribute("http.request.method", request.method)
                span.set_attribute("http.route", route)
                span.set_attribute("http.response.status_code", response.status_code)
        elapsed = time.perf_counter() - started
        metrics.observe(request.method, route, response.status_code, elapsed)
        if request.url.path.startswith("/v1/"):
            metrics.operation("db", elapsed)
        if request.url.path.startswith("/v1/resolve"):
            metrics.operation("resolver", elapsed)
        if "approve" in request.url.path or "deprecate" in request.url.path or request.url.path == "/v1/approvals":
            metrics.operation("approval", elapsed)
        if request.url.path.startswith("/v1/import") or request.url.path.startswith("/v1/sync"):
            metrics.operation("artifact_sync", elapsed)
        if request.url.path == "/v1/instances/register" and response.status_code >= 400:
            metrics.instance_event("registration", "failure")
        if request.url.path.endswith("/heartbeat") and response.status_code >= 400:
            metrics.instance_event("heartbeat", "failure")
        return response

    @app.exception_handler(RegistryError)
    async def registry_error(_request: Request, error: RegistryError):
        return JSONResponse({"error": {"detail": str(error)}}, status_code=error.status_code)

    @app.exception_handler(RegistryAuthError)
    async def auth_error(_request: Request, error: RegistryAuthError):
        return JSONResponse({"error": {"detail": str(error)}}, status_code=error.status_code)

    def scoped(required: str):
        def dependency(credentials: HTTPAuthorizationCredentials | None = Security(bearer)) -> str:
            actor, scopes = _identity(settings, credentials)
            if required not in scopes and "registry:admin" not in scopes:
                raise RegistryAuthError(f"scope {required!r} is required", 403)
            return actor
        return dependency

    def run(actor: str, operation: Callable[[RegistryService], Any]) -> Any:
        with db.session_factory() as session:
            return operation(RegistryService(session, actor=actor))

    read = scoped("registry:read")
    write = scoped("registry:write")
    approve_scope = scoped("registry:approve")

    def admit_instance(actor: str, value: ManagedInstanceRegister) -> None:
        policy = settings.instance_registration
        if policy.allowed_identities and actor not in policy.allowed_identities:
            raise RegistryAuthError("service identity is not allowed to register instances", 403)
        if policy.allowed_environments and value.environment not in policy.allowed_environments:
            raise RegistryAuthError("instance environment is not allowed", 403)
        if policy.allowed_clusters and value.cluster not in policy.allowed_clusters:
            raise RegistryAuthError("instance cluster is not allowed", 403)
        if policy.instance_name_pattern and not re.fullmatch(policy.instance_name_pattern, value.name):
            raise RegistryAuthError("instance name does not satisfy registration policy", 403)

    @app.get("/health", tags=["service"])
    def health() -> dict[str, Any]:
        return {"status": "ok", "protocol": REGISTRY_PROTOCOL, "database": db.engine.dialect.name}

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> PlainTextResponse:
        if not metrics.enabled:
            return PlainTextResponse("metrics disabled\n", status_code=404)
        with db.session_factory() as session:
            snapshot = RegistryService(session, actor="metrics").list_instances(
                100_000, 0,
                offline_after_seconds=settings.instance_registration.offline_after_seconds,
            )
        metrics.instance_snapshot(snapshot["items"])
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/import/huggingface", tags=["artifacts"])
    def import_huggingface(value: HuggingFaceImportRequest, actor: str = Depends(write)):
        from .connectors import HuggingFaceConnector
        model, bundle = HuggingFaceConnector().inspect(value.repo_id, revision=value.revision)
        return run(actor, lambda service: service.import_bundle_metadata(model, bundle))

    @app.post("/v1/sync/huggingface-collection", tags=["artifacts"])
    def sync_huggingface_collection(value: HuggingFaceCollectionSyncRequest, actor: str = Depends(write)):
        from .connectors import HuggingFaceConnector
        connector = HuggingFaceConnector()
        results = []
        for repo_id in connector.collection_items(value.collection):
            try:
                model, bundle = connector.inspect(repo_id)
                imported = run(actor, lambda service, m=model, b=bundle: service.import_bundle_metadata(m, b))
                results.append({"repo_id": repo_id, "status": "IMPORTED", **imported})
            except Exception as error:
                results.append({"repo_id": repo_id, "status": "FAILED", "error": f"{type(error).__name__}: {error}"})
        return {"collection": value.collection, "results": results}

    def page(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)) -> tuple[int, int]:
        return limit, offset

    # Managed runtime discovery. Registration is idempotent by stable instance ID.
    @app.post("/v1/instances/register", tags=["instances"])
    def register_instance(value: ManagedInstanceRegister, actor: str = Depends(write)):
        admit_instance(actor, value)
        result = run(actor, lambda service: service.register_instance(value, credential_identity=actor))
        metrics.instance_event("registration", "success")
        return result

    @app.post("/v1/instances/{instance_id}/heartbeat", tags=["instances"])
    def heartbeat_instance(instance_id: str, value: ManagedInstanceHeartbeat, actor: str = Depends(write)):
        result = run(actor, lambda service: service.heartbeat_instance(instance_id, value))
        metrics.instance_event("heartbeat", "success")
        return result

    @app.patch("/v1/instances/{instance_id}/observed", tags=["instances"])
    def patch_instance(instance_id: str, value: ManagedInstanceObservedPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_instance_observed(instance_id, value))

    @app.post("/v1/instances/{instance_id}/deregister", tags=["instances"])
    def deregister_instance(instance_id: str, actor: str = Depends(write)):
        return run(actor, lambda service: service.deregister_instance(instance_id))

    @app.get("/v1/instances", tags=["instances"])
    def list_instances(
        paging: tuple[int, int] = Depends(page), instance_type: str | None = None,
        environment: str | None = None, cluster: str | None = None,
        status: str | None = None, actor: str = Depends(read),
    ):
        result = run(actor, lambda service: service.list_instances(
            *paging,
            offline_after_seconds=settings.instance_registration.offline_after_seconds,
            instance_type=instance_type, environment=environment, cluster=cluster, status=status,
        ))
        metrics.instance_snapshot(result["items"])
        return result

    @app.get("/v1/instances/{instance_id}", tags=["instances"])
    def get_instance(instance_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_instance(
            instance_id,
            offline_after_seconds=settings.instance_registration.offline_after_seconds,
        ))

    @app.get("/v1/instances/{instance_id}/desired", tags=["instances"])
    def desired_instance(instance_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.desired_instance(instance_id))

    # Routing control state. External routers consume these resources; the
    # Registry is deliberately absent from their inference hot path.
    @app.get("/v1/routers", tags=["routing"])
    def list_routers(paging: tuple[int, int] = Depends(page), kind: str | None = None, region: str | None = None, health: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_routers(*paging, kind=kind, region=region, health=health))

    @app.post("/v1/routers", status_code=201, tags=["routing"])
    def create_router(value: RouterInstanceCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_router(value))

    @app.get("/v1/routers/{resource_id}", tags=["routing"])
    def get_router(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_router(resource_id))

    @app.patch("/v1/routers/{resource_id}", tags=["routing"])
    def patch_router(resource_id: str, value: RouterInstancePatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_router(resource_id, value))

    @app.get("/v1/routers/{resource_id}/desired", tags=["routing"])
    def router_desired(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.router_desired_state(resource_id))

    @app.get("/v1/routes", tags=["routing"])
    def list_routes(paging: tuple[int, int] = Depends(page), enabled: bool | None = None, route_kind: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_routes(*paging, enabled=enabled, route_kind=route_kind))

    @app.post("/v1/routes", status_code=201, tags=["routing"])
    def create_route(value: RouteCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_route(value))

    @app.get("/v1/routes/{resource_id}", tags=["routing"])
    def get_route(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_route(resource_id))

    @app.patch("/v1/routes/{resource_id}", tags=["routing"])
    def patch_route(resource_id: str, value: RoutePatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_route(resource_id, value))

    @app.get("/v1/model-pools", tags=["routing"])
    def list_model_pools(paging: tuple[int, int] = Depends(page), model_id: str | None = None, enabled: bool | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_model_pools(*paging, model_id=model_id, enabled=enabled))

    @app.post("/v1/model-pools", status_code=201, tags=["routing"])
    def create_model_pool(value: ModelPoolCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_model_pool(value))

    @app.get("/v1/model-pools/{resource_id}", tags=["routing"])
    def get_model_pool(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_model_pool(resource_id))

    @app.patch("/v1/model-pools/{resource_id}", tags=["routing"])
    def patch_model_pool(resource_id: str, value: ModelPoolPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_model_pool(resource_id, value))

    @app.get("/v1/backend-endpoints", tags=["routing"])
    def list_backend_endpoints(paging: tuple[int, int] = Depends(page), engine: str | None = None, model_id: str | None = None, health: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_backend_endpoints(*paging, engine=engine, model_id=model_id, health=health))

    @app.post("/v1/backend-endpoints", status_code=201, tags=["routing"])
    def create_backend_endpoint(value: BackendEndpointCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_backend_endpoint(value))

    @app.get("/v1/backend-endpoints/{resource_id}", tags=["routing"])
    def get_backend_endpoint(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_backend_endpoint(resource_id))

    @app.patch("/v1/backend-endpoints/{resource_id}", tags=["routing"])
    def patch_backend_endpoint(resource_id: str, value: BackendEndpointPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_backend_endpoint(resource_id, value))

    @app.get("/v1/routing-policies", tags=["routing"])
    def list_routing_policies(paging: tuple[int, int] = Depends(page), strategy: str | None = None, enabled: bool | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_routing_policies(*paging, strategy=strategy, enabled=enabled))

    @app.post("/v1/routing-policies", status_code=201, tags=["routing"])
    def create_routing_policy(value: RoutingPolicyCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_routing_policy(value))

    @app.get("/v1/routing-policies/{resource_id}", tags=["routing"])
    def get_routing_policy(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_routing_policy(resource_id))

    @app.patch("/v1/routing-policies/{resource_id}", tags=["routing"])
    def patch_routing_policy(resource_id: str, value: RoutingPolicyPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_routing_policy(resource_id, value))

    @app.get("/v1/route-bindings", tags=["routing"])
    def list_route_bindings(paging: tuple[int, int] = Depends(page), router_id: str | None = None, route_id: str | None = None, enabled: bool | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_route_bindings(*paging, router_id=router_id, route_id=route_id, enabled=enabled))

    @app.post("/v1/route-bindings", status_code=201, tags=["routing"])
    def create_route_binding(value: RouteBindingCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_route_binding(value))

    @app.get("/v1/route-bindings/{resource_id}", tags=["routing"])
    def get_route_binding(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_route_binding(resource_id))

    @app.patch("/v1/route-bindings/{resource_id}", tags=["routing"])
    def patch_route_binding(resource_id: str, value: RouteBindingPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_route_binding(resource_id, value))

    # Model resources.
    @app.get("/v1/models", tags=["models"])
    def list_models(paging: tuple[int, int] = Depends(page), provider: str | None = None, state: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_models(*paging, provider=provider, state=state))

    @app.post("/v1/models", status_code=201, tags=["models"])
    def create_model(value: ModelCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_model(value))

    @app.get("/v1/models/{resource_id}", tags=["models"])
    def get_model(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_model(resource_id))

    @app.patch("/v1/models/{resource_id}", tags=["models"])
    def patch_model(resource_id: str, value: ModelPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_model(resource_id, value))

    @app.delete("/v1/models/{resource_id}", tags=["models"])
    def delete_model(resource_id: str, actor: str = Depends(write)):
        return run(actor, lambda service: service.delete_model(resource_id))

    # Bundles.
    @app.get("/v1/bundles", tags=["bundles"])
    def list_bundles(paging: tuple[int, int] = Depends(page), base_model_id: str | None = None, trust: str | None = None, approval_state: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_bundles(*paging, base_model_id=base_model_id, trust=trust, approval_state=approval_state))

    @app.post("/v1/bundles", status_code=201, tags=["bundles"])
    def create_bundle(value: BundleCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_bundle(value))

    @app.get("/v1/bundles/{resource_id}", tags=["bundles"])
    def get_bundle(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_bundle(resource_id))

    @app.patch("/v1/bundles/{resource_id}", tags=["bundles"])
    def patch_bundle(resource_id: str, value: BundlePatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_bundle(resource_id, value))

    def transition(resource_type: str, resource_id: str, state: ApprovalState, actor: str, reason: str):
        return run(actor, lambda service: service.approve(ApprovalCreate(
            resource_type=resource_type, resource_id=resource_id, version="current",
            state=state, approver=actor, reason=reason,
        )))

    @app.post("/v1/bundles/{resource_id}/approve", tags=["bundles"])
    def approve_bundle(resource_id: str, reason: str = "approved through API", actor: str = Depends(approve_scope)):
        return transition("bundle", resource_id, ApprovalState.APPROVED, actor, reason)

    @app.post("/v1/bundles/{resource_id}/deprecate", tags=["bundles"])
    def deprecate_bundle(resource_id: str, reason: str = "deprecated through API", actor: str = Depends(approve_scope)):
        return transition("bundle", resource_id, ApprovalState.DEPRECATED, actor, reason)

    # Profiles.
    @app.get("/v1/profiles", tags=["profiles"])
    def list_profiles(paging: tuple[int, int] = Depends(page), approval_state: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_profiles(*paging, approval_state=approval_state))

    @app.post("/v1/profiles", status_code=201, tags=["profiles"])
    def create_profile(value: ProfileCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_profile(value))

    @app.get("/v1/profiles/{resource_id}", tags=["profiles"])
    def get_profile(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_profile(resource_id))

    @app.patch("/v1/profiles/{resource_id}", tags=["profiles"])
    def patch_profile(resource_id: str, value: ProfilePatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_profile(resource_id, value))

    @app.post("/v1/profiles/{resource_id}/approve", tags=["profiles"])
    def approve_profile(resource_id: str, reason: str = "approved through API", actor: str = Depends(approve_scope)):
        return transition("profile", resource_id, ApprovalState.APPROVED, actor, reason)

    # Compatibility and qualification.
    @app.get("/v1/compatibility", tags=["compatibility"])
    def list_compatibility(paging: tuple[int, int] = Depends(page), model_id: str | None = None, bundle_id: str | None = None, engine: str | None = None, execution_mode: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_compatibility(*paging, model_id=model_id, bundle_id=bundle_id, engine=engine, execution_mode=execution_mode))

    @app.post("/v1/compatibility", status_code=201, tags=["compatibility"])
    def create_compatibility(value: CompatibilityCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_compatibility(value))

    @app.get("/v1/compatibility/resolve", tags=["compatibility"])
    def resolve_compatibility(model_id: str | None = None, model_revision: str | None = None, bundle_id: str | None = None, engine: str | None = None, engine_version: str | None = None, hardware_class: str | None = None, execution_mode: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.resolve_compatibility(
            model_id=model_id, model_revision=model_revision, bundle_id=bundle_id,
            engine=engine, engine_version=engine_version, hardware_class=hardware_class,
            execution_mode=execution_mode,
        ))

    @app.get("/v1/qualifications", tags=["qualifications"])
    def list_qualifications(paging: tuple[int, int] = Depends(page), model_id: str | None = None, bundle_id: str | None = None, engine: str | None = None, workload: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_qualifications(*paging, model_id=model_id, bundle_id=bundle_id, engine=engine, workload=workload))

    @app.post("/v1/qualifications", status_code=201, tags=["qualifications"])
    def create_qualification(value: QualificationCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_qualification(value))

    @app.get("/v1/qualifications/{resource_id}", tags=["qualifications"])
    def get_qualification(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_qualification(resource_id))

    # Deployment desired state.
    @app.get("/v1/deployments", tags=["deployments"])
    def list_deployments(paging: tuple[int, int] = Depends(page), environment: str | None = None, cluster: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_deployments(*paging, environment=environment, cluster=cluster))

    @app.post("/v1/deployments", status_code=201, tags=["deployments"])
    def create_deployment(value: DeploymentCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_deployment(value))

    @app.get("/v1/deployments/{resource_id}", tags=["deployments"])
    def get_deployment(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_deployment(resource_id))

    @app.patch("/v1/deployments/{resource_id}", tags=["deployments"])
    def patch_deployment(resource_id: str, value: DeploymentPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_deployment(resource_id, value))

    @app.get("/v1/deployments/{resource_id}/desired", tags=["deployments"])
    def desired_deployment(resource_id: str, actor: str = Depends(read)):
        value = run(actor, lambda service: service.get_deployment(resource_id))
        return {"deployment_id": resource_id, "desired_revision": value["desired_revision"], "desired": value}

    # Policies and approvals.
    @app.get("/v1/policies", tags=["policies"])
    def list_policies(paging: tuple[int, int] = Depends(page), scope: str | None = None, approval_state: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_policies(*paging, scope=scope, approval_state=approval_state))

    @app.post("/v1/policies", status_code=201, tags=["policies"])
    def create_policy(value: PolicyCreate, actor: str = Depends(write)):
        return run(actor, lambda service: service.create_policy(value))

    @app.get("/v1/policies/{resource_id}", tags=["policies"])
    def get_policy(resource_id: str, actor: str = Depends(read)):
        return run(actor, lambda service: service.get_policy(resource_id))

    @app.patch("/v1/policies/{resource_id}", tags=["policies"])
    def patch_policy(resource_id: str, value: PolicyPatch, actor: str = Depends(write)):
        return run(actor, lambda service: service.patch_policy(resource_id, value))

    @app.post("/v1/policies/{resource_id}/approve", tags=["policies"])
    def approve_policy(resource_id: str, reason: str = "approved through API", actor: str = Depends(approve_scope)):
        return transition("policy", resource_id, ApprovalState.APPROVED, actor, reason)

    @app.get("/v1/approvals", tags=["governance"])
    def list_approvals(paging: tuple[int, int] = Depends(page), resource_type: str | None = None, resource_id: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_approvals(*paging, resource_type=resource_type, resource_id=resource_id))

    @app.post("/v1/approvals", status_code=201, tags=["governance"])
    def create_approval(value: ApprovalCreate, actor: str = Depends(approve_scope)):
        return run(actor, lambda service: service.approve(value.model_copy(update={"approver": actor})))

    @app.get("/v1/audit", tags=["governance"])
    def list_audit(paging: tuple[int, int] = Depends(page), resource_type: str | None = None, resource_id: str | None = None, actor: str = Depends(read)):
        return run(actor, lambda service: service.list_audit(*paging, resource_type=resource_type, resource_id=resource_id))

    # High-value deterministic resolvers.
    @app.post("/v1/resolve/bundle", tags=["resolution"])
    def resolve_bundle(value: BundleResolveRequest, actor: str = Depends(read)):
        return run(actor, lambda service: service.resolve_bundle(value))

    @app.post("/v1/resolve/profile", tags=["resolution"])
    def resolve_profile(value: ProfileResolveRequest, actor: str = Depends(read)):
        return run(actor, lambda service: service.resolve_profile(value))

    @app.post("/v1/resolve/deployment", tags=["resolution"])
    def resolve_deployment(value: DeploymentResolveRequest, actor: str = Depends(read)):
        return run(actor, lambda service: service.resolve_deployment(value))

    return app


def _configure_tracing(settings: RegistryConfig) -> tuple[Any | None, Any | None]:
    if not (settings.observability.enabled and settings.observability.otel_enabled):
        return None, None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except ImportError as error:
        raise ImportError("OTel registry tracing requires the 'registry' extra") from error
    provider = TracerProvider(resource=Resource.create({"service.name": "pra-registry"}))
    endpoint = settings.observability.otel_endpoint
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://")) if endpoint else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider.get_tracer("pra.registry"), provider
