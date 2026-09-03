"""Canonical Control Plane application layer shared by REST, MCP, agent, and CLI."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
from urllib.parse import quote

from .backends import ActionBackend, ObservedStateBackend, RegistryBackend
from .config import ControlPlaneConfig, EngineTargetConfig
from .domain import (
    ActionPlan,
    ActionResult,
    ApprovalRequired,
    CallerContext,
    Conflict,
    ContextSummary,
    EngineDetails,
    FleetSummary,
    Forbidden,
    InvalidRequest,
    MetricsSummary,
    NotFound,
    NotSupported,
    Offline,
    QualificationSummary,
)
from .fleet_policy import alerts, compare_desired_observed, light_metrics
from .operations import OPERATIONS
from .persistence import ControlStore
from pra_router.adapters import adapter_for
from pra_router.controller import RouterController


REGISTRY_RESOURCES = frozenset({
    "models", "bundles", "profiles", "qualifications", "compatibility",
    "deployments", "policies", "approvals", "audit", "instances",
    "routers", "routes", "model-pools", "backend-endpoints", "routing-policies", "route-bindings",
})
APPROVAL_TRANSITIONS = frozenset({"approve", "deprecate", "revoke", "promote"})
ENGINE_SECTIONS = frozenset({
    "summary", "capabilities", "config", "models", "sessions", "resources",
    "storage", "observability", "audit",
})


def authorize(caller: CallerContext, operation_id: str, *, permission: str | None = None) -> str:
    """Enforce manager-layer authorization even when adapters are bypassed."""
    operation = OPERATIONS[operation_id]
    required = permission or operation.permission
    if "admin" not in caller.permissions and required not in caller.permissions:
        raise Forbidden(f"{required} is required", details={"operation": operation_id, "permission": required})
    return required


class AuditManager:
    def __init__(self, store: ControlStore, *, enabled: bool = True) -> None:
        self.store = store
        self.enabled = enabled

    def record(
        self, caller: CallerContext, operation: str, target: str, reason: str, result: str,
        *, permission: str, before: Any = None, after: Any = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        return self.store.audit(
            actor=caller.subject, role=caller.roles[0] if caller.roles else "unknown",
            roles=caller.roles, permission=permission, action=operation, target=target,
            before=before, after=after, reason=reason, request_id=caller.request_id,
            trace_id=caller.trace_id, transport=caller.transport,
            idempotency_key=idempotency_key, result=result,
        )

    def list(self, caller: CallerContext, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        authorize(caller, "audit.read")
        return self.store.audit_events(limit=min(limit, 500), offset=max(0, offset))


class FleetManager:
    def __init__(self, backend: ObservedStateBackend, store: ControlStore, audit: AuditManager) -> None:
        self.backend = backend
        self.store = store
        self.audit = audit

    async def list(self, caller: CallerContext) -> FleetSummary:
        authorize(caller, "fleet.list")
        collector = getattr(self.backend, "collect_instances", None)
        if collector is not None:
            observations = await collector()
            rows = [self._resolve_observation(row) for row in observations]
            return FleetSummary(items=rows, summary={
                "total": len(rows), "healthy": sum(row["status"] == "IN_SYNC" for row in rows),
                "drift": sum(row["status"] == "DRIFT" for row in rows),
                "offline": sum(row["status"] == "OFFLINE" for row in rows),
                "unknown": sum(row["status"] == "UNKNOWN" for row in rows),
            })
        return FleetSummary.model_validate(await self.backend.overview())

    @staticmethod
    def _resolve_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        target = dict(observation["target"])
        snapshot = observation.get("snapshot")
        if snapshot is None:
            return {
                "name": target["name"], "status": "OFFLINE", "health": "offline",
                "environment": target.get("environment"), "region": target.get("region"),
                "cluster": target.get("cluster"), "namespace": target.get("namespace"),
                "host": target.get("host"), "error": observation.get("error"),
                "drift": {"status": "OFFLINE", "differences": []}, "metrics": {},
                "alerts": ["engine offline"],
            }
        desired = observation.get("desired")
        drift = compare_desired_observed(desired, snapshot)
        info = snapshot.get("info", {})
        models = snapshot.get("models", {}).get("items", snapshot.get("models", []))
        model = next((row for row in models if row.get("runtime_model_id") == "default"), next(iter(models), {}))
        return {
            "name": target["name"], "status": drift["status"],
            "environment": target.get("environment"), "region": target.get("region"),
            "cluster": target.get("cluster"), "namespace": target.get("namespace"),
            "host": target.get("host") or info.get("host"), "engine": info.get("engine"),
            "engine_version": info.get("engine_version"), "pra_version": info.get("pra_version"),
            "health": info.get("health", "healthy"), "model": model.get("model_id"),
            "bundle": model.get("pra_bundle_id"), "profile": model.get("profile"),
            "mode": model.get("execution_mode"), "drift": drift, "models": models,
            "model_count": len(models), "capabilities": snapshot.get("capabilities", {}),
            "metrics": light_metrics(snapshot), "alerts": alerts(snapshot, drift),
        }

    async def find(self, caller: CallerContext, *, query: str = "", engine: str | None = None, model: str | None = None) -> FleetSummary:
        result = await self.list(caller)
        if not query and not engine and not model:
            return result
        terms = query.casefold().split()
        result.items = [
            row for row in result.items
            if (not engine or str(row.get("engine", "")).casefold() == engine.casefold())
            and (not model or model.casefold() in str(row.get("model", "")).casefold())
            and (not terms or all(term in str(row).casefold() for term in terms))
        ]
        result.summary = {"total": len(result.items)}
        return result

    async def list_instances(self, caller: CallerContext) -> FleetSummary:
        return await self.list(caller)

    async def find_instances(self, caller: CallerContext, **filters: Any) -> FleetSummary:
        return await self.find(caller, **filters)

    async def inspect_instance(self, caller: CallerContext, instance_id: str, section: str = "summary") -> EngineDetails:
        return await self.inspect(caller, instance_id, section)

    async def inspect(self, caller: CallerContext, name: str, section: str = "summary") -> EngineDetails:
        authorize(caller, "engine.inspect")
        if section not in ENGINE_SECTIONS:
            raise NotFound(f"unknown engine section: {section}")
        try:
            value = await self.backend.engine_section(name, section)
        except KeyError as error:
            raise NotFound(f"engine not found: {name}") from error
        except Exception as error:
            raise Offline(f"engine {name} is unavailable", details={"cause": str(error)}) from error
        return EngineDetails(instance_id=name, section=section, value=value)

    async def list_models(self, caller: CallerContext, name: str | None = None) -> list[dict[str, Any]]:
        if name:
            details = await self.inspect(caller, name, "models")
            value = details.value
            return list(value.get("items", value) if isinstance(value, dict) else value)
        fleet = await self.list(caller)
        return [dict(model, instance_id=row.get("name")) for row in fleet.items for model in row.get("models", [])]

    async def inspect_model(self, caller: CallerContext, instance_id: str, runtime_model_id: str) -> dict[str, Any]:
        rows = await self.list_models(caller, instance_id)
        row = next((item for item in rows if str(item.get("runtime_model_id")) == runtime_model_id), None)
        if row is None:
            raise NotFound(f"runtime model not found: {runtime_model_id}")
        return row

    async def recommendations(self, caller: CallerContext) -> dict[str, Any]:
        fleet = await self.list(caller)
        items: list[dict[str, Any]] = []
        for row in fleet.items:
            if row.get("status") == "DRIFT":
                items.append({"engine": row["name"], "kind": "reconcile", "approval_required": True, "reason": "observed state differs from Registry intent"})
            if row.get("status") == "OFFLINE":
                items.append({"engine": row["name"], "kind": "investigate", "approval_required": True, "reason": "management endpoint is offline"})
            if float(row.get("metrics", {}).get("storage_reloads") or 0) > 10:
                items.append({"engine": row["name"], "kind": "warm-quota", "approval_required": True, "reason": "high storage reload count"})
        return {"items": items, "mode": "recommendation-only"}

    async def register(
        self, caller: CallerContext, values: Mapping[str, Any], *, reason: str,
    ) -> dict[str, Any]:
        permission = authorize(caller, "engine.register")
        metadata = dict(values.get("metadata") or values.get("metadata_payload") or {})
        try:
            EngineTargetConfig(
                name=str(values["name"]), management_url=str(values["management_url"]),
                token_env=values.get("token_env"), **metadata,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InvalidRequest(str(error)) from error
        stored = {
            "name": str(values["name"]), "management_url": str(values["management_url"]),
            "token_env": values.get("token_env"), "metadata_payload": metadata,
        }
        before, after = self.store.put_engine(stored)
        self.audit.record(caller, "engine.register", stored["name"], reason, "success", permission=permission, before=before, after=after)
        return after

    async def remove(self, caller: CallerContext, name: str, *, reason: str, confirmed: bool) -> dict[str, bool]:
        permission = authorize(caller, "engine.remove")
        if not confirmed:
            raise ApprovalRequired("engine removal requires confirmation")
        before = self.store.delete_engine(name)
        if not before:
            raise NotFound("manual engine not found")
        self.audit.record(caller, "engine.remove", name, reason, "success", permission=permission, before=before)
        return {"removed": True}


class RegistryManager:
    def __init__(self, backend: RegistryBackend | None, audit: AuditManager) -> None:
        self.backend = backend
        self.audit = audit

    def _available(self) -> RegistryBackend:
        if self.backend is None:
            raise Offline("Registry is not configured")
        return self.backend

    @staticmethod
    def _resource(resource: str, *, mutable: bool = False) -> None:
        if resource not in REGISTRY_RESOURCES or (mutable and resource in {"audit", "instances"}):
            raise NotFound(f"unknown Registry resource: {resource}")

    async def list(self, caller: CallerContext, resource: str, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        authorize(caller, "registry.list")
        self._resource(resource)
        try:
            return await self._available().list(resource, limit=min(limit, 500), offset=max(offset, 0))
        except Exception as error:
            raise Offline("Registry read failed", details={"cause": str(error)}) from error

    async def create(self, caller: CallerContext, resource: str, values: Mapping[str, Any], *, reason: str) -> Any:
        self._resource(resource, mutable=True)
        return await self._mutate(caller, "registry.write", "POST", f"/v1/{resource}", values, reason, f"registry.{resource}.create")

    async def patch(self, caller: CallerContext, resource: str, resource_id: str, values: Mapping[str, Any], *, reason: str) -> Any:
        self._resource(resource, mutable=True)
        path = f"/v1/{resource}/{quote(resource_id, safe='')}"
        backend = self._available()
        before = await backend.request("GET", path)
        return await self._mutate(caller, "registry.write", "PATCH", path, values, reason, f"registry.{resource}.patch", before=before)

    async def transition(
        self, caller: CallerContext, resource: str, resource_id: str, transition: str,
        values: Mapping[str, Any], *, reason: str,
    ) -> Any:
        self._resource(resource, mutable=True)
        if transition not in APPROVAL_TRANSITIONS:
            raise NotFound(f"unsupported transition: {transition}")
        operation = "qualification.approve" if resource == "qualifications" else "registry.write"
        states = {"approve": "APPROVED", "promote": "APPROVED", "deprecate": "DEPRECATED", "revoke": "REVOKED"}
        singular = {"compatibility": "compatibility", "policies": "policy", "qualifications": "qualification"}.get(resource, resource.rstrip("s"))
        payload = {
            "resource_type": singular, "resource_id": resource_id,
            "version": str(values.get("version", "current")), "state": states[transition],
            "approver": caller.subject, "reason": reason,
        }
        return await self._mutate(caller, operation, "POST", "/v1/approvals", payload, reason, f"registry.{resource}.{transition}")

    async def _mutate(
        self, caller: CallerContext, operation_id: str, method: str, path: str,
        values: Mapping[str, Any], reason: str, operation: str, *, before: Any = None,
    ) -> Any:
        permission = authorize(caller, operation_id)
        try:
            result = await self._available().request(method, path, values)
            self.audit.record(caller, operation, path, reason, "success", permission=permission, before=before, after=result)
            return result
        except Exception as error:
            self.audit.record(caller, operation, path, reason, "failure", permission=permission, before=before, after={"error": str(error)})
            raise Offline("Registry mutation failed", details={"cause": str(error)}) from error


class _ControlRouterSource:
    """Adapt the configured Control Plane Registry client to RouterController."""

    def __init__(self, registry: RegistryManager) -> None:
        self.registry = registry

    async def list_router_ids(self) -> list[str]:
        rows = await self.registry._available().list("routers", limit=500, offset=0)
        return [str(row["id"]) for row in rows.get("items", [])]

    async def desired_state(self, router_id: str) -> dict[str, Any]:
        return await self.registry._available().request(
            "GET", f"/v1/routers/{quote(router_id, safe='')}/desired",
        )

    async def report_observed(self, router_id: str, **values: Any) -> None:
        await self.registry._available().request(
            "PATCH", f"/v1/routers/{quote(router_id, safe='')}", values,
        )


class RouterManager:
    """Govern router reconciliation while leaving inference traffic external."""

    def __init__(self, registry: RegistryManager, audit: AuditManager) -> None:
        self.registry = registry
        self.audit = audit
        self.controller = RouterController(
            _ControlRouterSource(registry),
            {kind: adapter_for(kind) for kind in (
                "litellm", "agentgateway", "kubernetes-gaie", "pra-reference", "bifrost",
            )},
        )

    async def list(self, caller: CallerContext) -> list[dict[str, Any]]:
        authorize(caller, "router.list")
        return list((await self.registry.list(caller, "routers", limit=500)).get("items", []))

    async def inspect(self, caller: CallerContext, router_id: str) -> dict[str, Any]:
        authorize(caller, "router.list")
        try:
            return await self.controller.inspect(router_id)
        except Exception as error:
            raise Offline("router inspection failed", details={"cause": str(error)}) from error

    async def routes(self, caller: CallerContext, route_id: str | None = None) -> list[dict[str, Any]]:
        authorize(caller, "route.list")
        rows = list((await self.registry.list(caller, "routes", limit=500)).get("items", []))
        return [row for row in rows if route_id is None or str(row.get("id")) == route_id]

    async def preview(self, caller: CallerContext, router_id: str) -> dict[str, Any]:
        authorize(caller, "route.plan")
        try:
            return (await self.controller.preview(router_id)).model_dump(mode="json")
        except Exception as error:
            raise Offline("router preview failed", details={"cause": str(error)}) from error

    async def apply(self, caller: CallerContext, router_id: str, *, reason: str, confirmed: bool) -> dict[str, Any]:
        permission = authorize(caller, "route.apply")
        if not confirmed:
            raise ApprovalRequired("router reconciliation requires confirmation")
        before = await self.preview(caller, router_id)
        result = await self.controller.reconcile(router_id)
        payload = result.model_dump(mode="json")
        self.audit.record(
            caller, "router.reconcile", router_id, reason,
            "success" if result.verified else "failure", permission=permission,
            before=before, after=payload,
        )
        if not result.verified:
            raise Offline("router reconciliation failed", details={"result": payload})
        return payload


class DeploymentManager:
    def __init__(self, registry: RegistryManager, fleet: FleetManager) -> None:
        self.registry = registry
        self.fleet = fleet
        self.actions: ActionManager | None = None

    async def list(self, caller: CallerContext) -> list[dict[str, Any]]:
        authorize(caller, "deployment.read")
        page = await self.registry.list(caller, "deployments")
        return list(page.get("items", []))

    async def get(self, caller: CallerContext, deployment_id: str) -> dict[str, Any]:
        for row in await self.list(caller):
            if str(row.get("id")) == deployment_id:
                return row
        raise NotFound(f"deployment not found: {deployment_id}")

    async def drift(self, caller: CallerContext, instance_id: str) -> dict[str, Any]:
        authorize(caller, "deployment.read")
        fleet = await self.fleet.list(caller)
        row = next((item for item in fleet.items if item.get("name") == instance_id), None)
        if row is None:
            raise NotFound(f"engine not found: {instance_id}")
        return dict(row.get("drift") or {})

    async def get_deployment(self, caller: CallerContext, deployment_id: str) -> dict[str, Any]:
        return await self.get(caller, deployment_id)

    async def get_drift(self, caller: CallerContext, instance_id: str) -> dict[str, Any]:
        return await self.drift(caller, instance_id)

    async def plan_reconciliation(
        self, caller: CallerContext, instance_id: str, deployment_id: str,
        *, idempotency_key: str | None = None,
    ) -> ActionPlan:
        authorize(caller, "deployment.read")
        deployment = await self.get(caller, deployment_id)
        if self.actions is None:
            raise NotSupported("reconciliation action manager is unavailable")
        return await self.actions.plan(
            caller, "reconcile", instance_id, {"deployment_id": deployment_id, "desired": deployment},
            idempotency_key=idempotency_key,
        )


ACTION_POLICY: dict[str, dict[str, Any]] = {
    "prefetch": {"permission": "engine:action", "impact": "low", "reversible": True},
    "promote": {"permission": "engine:action", "impact": "low", "reversible": True},
    "maintenance": {"permission": "engine:action", "impact": "medium", "reversible": True},
    "load-model": {"permission": "engine:action", "impact": "medium", "reversible": True},
    "config-patch": {"permission": "engine:configure", "impact": "medium", "reversible": True},
    "demote": {"permission": "engine:high-impact", "impact": "high", "reversible": True},
    "evict": {"permission": "engine:high-impact", "impact": "high", "reversible": False},
    "unload-model": {"permission": "engine:high-impact", "impact": "high", "reversible": True},
    "reconcile": {"permission": "deployment:apply", "impact": "high", "reversible": False},
}


class ActionManager:
    def __init__(
        self, backend: ActionBackend, fleet: FleetManager, audit: AuditManager,
        *, ttl_seconds: int = 900,
    ) -> None:
        self.backend = backend
        self.fleet = fleet
        self.audit = audit
        self.store = audit.store
        self.ttl_seconds = ttl_seconds
        self.plans: dict[str, ActionPlan] = {}
        self.applied: dict[str, ActionResult] = {}

    async def plan(
        self, caller: CallerContext, action: str, target: str,
        requested_change: Mapping[str, Any] | None = None, *, idempotency_key: str | None = None,
    ) -> ActionPlan:
        authorize(caller, "action.plan")
        policy = ACTION_POLICY.get(action)
        if policy is None:
            raise NotSupported(f"unsupported action: {action}")
        try:
            section = "config" if action == "config-patch" else "summary"
            current = (await self.fleet.inspect(caller, target, section)).value
        except Offline:
            current = {"status": "offline"}
        try:
            capabilities = (await self.fleet.inspect(caller, target, "capabilities")).value
        except (NotFound, Offline):
            capabilities = {}
        advertised = (
            capabilities.get("management_actions", capabilities.get("actions", []))
            if isinstance(capabilities, dict) else []
        )
        if advertised and action not in advertised:
            raise NotSupported(f"engine {target} does not advertise action {action}")
        plan = ActionPlan(
            plan_id=secrets.token_urlsafe(18), action=action, target=target,
            requested_change=dict(requested_change or {}),
            current_state=current if isinstance(current, dict) else {"value": current},
            projected_state={"action": action, **dict(requested_change or {})},
            reversible=bool(policy["reversible"]), restart_required=action in {"load-model", "unload-model"},
            impact=str(policy["impact"]), required_permission=str(policy["permission"]),
            requires_confirmation=policy["impact"] == "high",
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds),
            idempotency_key=idempotency_key, created_by=caller.subject, tenant=caller.tenant,
        )
        self.plans[plan.plan_id] = plan
        self.store.put_action_plan(plan.plan_id, plan.model_dump(mode="json"), idempotency_key)
        return plan

    async def apply(
        self, caller: CallerContext, plan_id: str, *, confirmation: bool = False,
        reason: str, idempotency_key: str | None = None,
    ) -> ActionResult:
        plan = self.plans.get(plan_id)
        if plan is None:
            stored = self.store.get_action_plan(plan_id)
            if stored:
                plan = ActionPlan.model_validate(stored["payload"])
                self.plans[plan_id] = plan
        if plan is None:
            raise NotFound(f"action plan not found: {plan_id}")
        if plan.tenant and caller.tenant != plan.tenant and "admin" not in caller.permissions:
            raise Forbidden("action plan belongs to another tenant")
        key = idempotency_key or plan.idempotency_key
        if key:
            stored_result = self.store.find_action_result(key)
            if stored_result and stored_result.get("result_payload"):
                previous = ActionPlan.model_validate(stored_result["payload"])
                if (
                    previous.action != plan.action
                    or previous.target != plan.target
                    or previous.requested_change != plan.requested_change
                ):
                    raise Conflict("idempotency key was already used for a different action intent")
                return ActionResult.model_validate(stored_result["result_payload"]).model_copy(update={"idempotent_replay": True})
        if plan_id in self.applied:
            return self.applied[plan_id].model_copy(update={"idempotent_replay": True})
        stored_plan = self.store.get_action_plan(plan_id)
        if stored_plan and stored_plan.get("result_payload"):
            return ActionResult.model_validate(stored_plan["result_payload"]).model_copy(update={"idempotent_replay": True})
        if plan.expires_at and datetime.now(timezone.utc) > plan.expires_at:
            raise Conflict("action plan has expired")
        if plan.requires_confirmation and not confirmation:
            raise ApprovalRequired(f"{plan.action} requires confirmation")
        authorize(caller, "action.apply", permission=plan.required_permission)
        try:
            if plan.action == "config-patch":
                value = await self.backend.patch_config(plan.target, plan.requested_change)
            else:
                value = await self.backend.action(plan.target, plan.action, plan.requested_change)
            result = ActionResult(plan_id=plan_id, action=plan.action, target=plan.target, status="applied", result=value)
            self.audit.record(
                caller, f"engine.{plan.action}", plan.target, reason, "success",
                permission=plan.required_permission, before=plan.current_state, after=value, idempotency_key=key,
            )
        except Exception as error:
            self.audit.record(
                caller, f"engine.{plan.action}", plan.target, reason, "failure",
                permission=plan.required_permission, before=plan.current_state,
                after={"error": str(error)}, idempotency_key=key,
            )
            raise Offline(f"action {plan.action} failed", details={"cause": str(error)}) from error
        self.applied[plan_id] = result
        self.store.complete_action_plan(plan_id, result.model_dump(mode="json"))
        return result

    async def execute(
        self, caller: CallerContext, action: str, target: str, values: Mapping[str, Any],
        *, reason: str, confirmed: bool = False, idempotency_key: str | None = None,
    ) -> ActionResult:
        plan = await self.plan(caller, action, target, values, idempotency_key=idempotency_key)
        return await self.apply(caller, plan.plan_id, confirmation=confirmed, reason=reason, idempotency_key=idempotency_key)


class QualificationManager:
    def __init__(self, registry: RegistryManager) -> None:
        self.registry = registry

    async def list_evidence(
        self, caller: CallerContext, *, model: str | None = None, engine: str | None = None,
    ) -> list[dict[str, Any]]:
        authorize(caller, "qualification.read")
        page = await self.registry.list(caller, "qualifications")
        rows = list(page.get("items", []))
        return [
            row for row in rows
            if (not model or model.casefold() in str(row.get("model_id", row.get("model", ""))).casefold())
            and (not engine or engine.casefold() == str(row.get("engine", "")).casefold())
        ]

    async def get_support_status(
        self, caller: CallerContext, model: str, engine: str, *, hardware: str | None = None,
    ) -> QualificationSummary:
        rows = await self.list_evidence(caller, model=model, engine=engine)
        if hardware:
            rows = [row for row in rows if not row.get("hardware") or hardware.casefold() in str(row.get("hardware")).casefold()]
        statuses = [str(row.get("status", row.get("qualification", "NOT_MEASURED"))).upper() for row in rows]
        order = ["RECOMMENDED", "QUALIFIED", "AVAILABLE", "NOT_MEASURED", "BLOCKED"]
        status = next((value for value in order if value in statuses), "NOT_MEASURED")
        recommendation = next((row.get("recommendation") for row in rows if row.get("recommendation")), None)
        return QualificationSummary(model_id=model, engine=engine, hardware=hardware, status=status, recommendation=recommendation, evidence=rows)

    async def get_recommendation(
        self, caller: CallerContext, model: str, engine: str, *, hardware: str | None = None,
    ) -> dict[str, Any]:
        status = await self.get_support_status(caller, model, engine, hardware=hardware)
        return {"status": status.status, "recommendation": status.recommendation, "evidence_count": len(status.evidence)}


class ObservabilityManager:
    def __init__(self, config: ControlPlaneConfig, fleet: FleetManager) -> None:
        self.config = config
        self.fleet = fleet

    async def summary(self, caller: CallerContext, *, engine: str | None = None, period: str = "15m") -> MetricsSummary:
        authorize(caller, "observability.read")
        metrics: dict[str, Any] = {}
        if engine:
            try:
                value = (await self.fleet.inspect(caller, engine, "observability")).value
                metrics = value if isinstance(value, dict) else {"value": value}
            except (NotFound, Offline):
                fleet = await self.fleet.list(caller)
                row = next((item for item in fleet.items if item.get("name") == engine), None)
                metrics = dict(row.get("metrics") or {}) if row else {}
        return MetricsSummary(engine=engine, period=period, metrics=metrics, links=self.links(caller, engine=engine))

    def links(self, caller: CallerContext, *, engine: str | None = None, trace_id: str | None = None) -> dict[str, str | None]:
        authorize(caller, "observability.read")
        suffix = f"?var-engine={quote(engine)}" if engine else ""
        trace = f"/explore?traceId={quote(trace_id)}" if trace_id else ""
        return {
            "grafana": f"{self.config.grafana.url}{suffix}" if self.config.grafana.url else None,
            "tempo": f"{self.config.tempo.url}{trace}" if self.config.tempo.url else None,
            "prometheus": self.config.prometheus.url,
        }

    async def get_metrics_summary(self, caller: CallerContext, engine: str | None = None, period: str = "15m") -> MetricsSummary:
        return await self.summary(caller, engine=engine, period=period)

    def get_trace_link(self, caller: CallerContext, trace_id: str) -> str | None:
        return self.links(caller, trace_id=trace_id)["tempo"]

    def get_dashboard_link(self, caller: CallerContext, engine: str | None = None) -> str | None:
        return self.links(caller, engine=engine)["grafana"]


class ContextManager:
    def __init__(self, fleet: FleetManager, registry: RegistryManager, qualifications: QualificationManager) -> None:
        self.fleet = fleet
        self.registry = registry
        self.qualifications = qualifications

    async def assemble(self, caller: CallerContext, *, task: str, repository: str | None = None) -> ContextSummary:
        authorize(caller, "context.read")
        fleet = await self.fleet.list(caller)
        bundles = await self._rows(caller, "bundles")
        qualifications = await self._rows(caller, "qualifications")
        deployments = await self._rows(caller, "deployments")
        terms = {part.casefold() for part in task.replace("/", " ").replace("-", " ").split() if len(part) > 2}

        def relevant(row: Mapping[str, Any]) -> bool:
            return not terms or any(term in str(row).casefold() for term in terms)

        limitations = sorted({
            str(item) for row in qualifications for item in row.get("limitations", []) if relevant(row)
        })
        evidence = [row for row in qualifications if relevant(row)][:20]
        return ContextSummary(
            task=task, repository=repository, fleet=fleet.model_dump(mode="json"),
            bundles=[row for row in bundles if relevant(row)][:20],
            qualifications=[row for row in qualifications if relevant(row)][:20],
            deployments=[row for row in deployments if relevant(row)][:20],
            evidence=evidence, limitations=limitations,
        )

    async def _rows(self, caller: CallerContext, resource: str) -> list[dict[str, Any]]:
        try:
            return list((await self.registry.list(caller, resource, limit=100)).get("items", []))
        except Offline:
            return []


class ExperimentManager:
    def __init__(self, audit: AuditManager, enabled: bool = False) -> None:
        self.audit = audit
        self.enabled = enabled

    async def list(self, caller: CallerContext) -> list[dict[str, Any]]:
        authorize(caller, "experiment.read")
        return []

    async def submit(self, caller: CallerContext, request: Mapping[str, Any]) -> dict[str, Any]:
        permission = authorize(caller, "experiment.run")
        if not self.enabled:
            raise NotSupported("experiment submission is disabled")
        result = {"status": "accepted", "request": dict(request)}
        self.audit.record(
            caller, "experiment.submit", str(request.get("name", "experiment")),
            str(request.get("reason", "experiment submission")), "success",
            permission=permission, after=result,
            idempotency_key=request.get("idempotency_key"),
        )
        return result


class ControlManager:
    """Top-level semantic facade consumed by every presentation adapter."""

    def __init__(
        self, *, fleet: FleetManager, registry: RegistryManager, deployments: DeploymentManager,
        actions: ActionManager, qualifications: QualificationManager,
        observability: ObservabilityManager, audit: AuditManager, context: ContextManager,
        experiments: ExperimentManager | None = None, routers: RouterManager | None = None,
    ) -> None:
        self.fleet = fleet
        self.registry = registry
        self.deployments = deployments
        self.actions = actions
        self.qualifications = qualifications
        self.observability = observability
        self.audit = audit
        self.context = context
        self.experiments = experiments
        self.routers = routers or RouterManager(registry, audit)

    @classmethod
    def build(
        cls, config: ControlPlaneConfig, store: ControlStore,
        fleet_backend: ObservedStateBackend,
    ) -> "ControlManager":
        audit = AuditManager(store, enabled=config.manager.audit)
        fleet = FleetManager(fleet_backend, store, audit)
        registry = RegistryManager(getattr(fleet_backend, "registry", None), audit)
        deployments = DeploymentManager(registry, fleet)
        actions = ActionManager(
            fleet_backend, fleet, audit, ttl_seconds=config.manager.action_plan_ttl_seconds,
        )
        deployments.actions = actions
        qualifications = QualificationManager(registry)
        observability = ObservabilityManager(config, fleet)
        context = ContextManager(fleet, registry, qualifications)
        experiments = ExperimentManager(audit, config.manager.experiments_enabled)
        routers = RouterManager(registry, audit)
        return cls(
            fleet=fleet, registry=registry, deployments=deployments, actions=actions,
            qualifications=qualifications, observability=observability, audit=audit,
            context=context, experiments=experiments, routers=routers,
        )
