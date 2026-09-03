"""Transport-neutral reconciliation of Registry intent into router state."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class RouterControllerError(RuntimeError):
    """A desired-state read, compile, apply, or verification operation failed."""


class RouterDesiredState(BaseModel):
    """Canonical state consumed by every router adapter."""

    model_config = ConfigDict(extra="allow")

    router: dict[str, Any]
    desired_revision: int = Field(ge=0)
    routes: list[dict[str, Any]] = Field(default_factory=list)
    in_sync: bool = False


class ReconcileOperation(BaseModel):
    action: str
    resource_type: str
    resource_id: str
    before: Any = None
    after: Any = None


class ReconcilePlan(BaseModel):
    router_id: str
    router_kind: str
    desired_revision: int
    observed_revision: int
    desired_digest: str
    observed_digest: str
    operations: list[ReconcileOperation] = Field(default_factory=list)
    restart_required: bool = False
    serving_continues: bool = True

    @property
    def drifted(self) -> bool:
        return bool(self.operations or self.desired_revision != self.observed_revision)


class ReconcileResult(BaseModel):
    router_id: str
    status: str
    desired_revision: int
    observed_revision: int
    operations_applied: int
    verified: bool
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None


class RouterStateSource(Protocol):
    async def list_router_ids(self) -> list[str]: ...

    async def desired_state(self, router_id: str) -> dict[str, Any]: ...

    async def report_observed(
        self, router_id: str, *, observed_revision: int, health: str,
        last_error: str | None = None, supported_features: list[str] | None = None,
    ) -> None: ...


class RouterAdapter(Protocol):
    kind: str

    async def discover_capabilities(self) -> list[str]: ...

    async def read_observed(self, desired: RouterDesiredState) -> dict[str, Any]: ...

    def compile(self, desired: RouterDesiredState) -> dict[str, Any]: ...

    async def apply(self, desired: RouterDesiredState, compiled: dict[str, Any], plan: ReconcilePlan) -> None: ...

    async def verify(self, desired: RouterDesiredState, compiled: dict[str, Any]) -> bool: ...


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def structural_diff(before: Any, after: Any, *, prefix: str = "config") -> list[ReconcileOperation]:
    """Return stable top-level operations suitable for preview and audit."""

    if before == after:
        return []
    if not isinstance(before, dict) or not isinstance(after, dict):
        return [ReconcileOperation(
            action="replace", resource_type="configuration", resource_id=prefix,
            before=before, after=after,
        )]
    operations: list[ReconcileOperation] = []
    for key in sorted(set(before) | set(after)):
        if key not in before:
            action = "add"
        elif key not in after:
            action = "remove"
        elif before[key] != after[key]:
            action = "update"
        else:
            continue
        operations.append(ReconcileOperation(
            action=action, resource_type="configuration", resource_id=f"{prefix}.{key}",
            before=before.get(key), after=after.get(key),
        ))
    return operations


class RouterController:
    """Reconcile Registry state without entering the inference request path."""

    def __init__(self, source: RouterStateSource, adapters: dict[str, RouterAdapter]) -> None:
        self.source = source
        self.adapters = adapters
        self._locks: dict[str, asyncio.Lock] = {}

    async def preview(self, router_id: str) -> ReconcilePlan:
        desired = RouterDesiredState.model_validate(await self.source.desired_state(router_id))
        adapter = self._adapter(desired)
        observed = await adapter.read_observed(desired)
        compiled = adapter.compile(desired)
        observed_config = observed.get("config", observed)
        return ReconcilePlan(
            router_id=router_id,
            router_kind=adapter.kind,
            desired_revision=desired.desired_revision,
            observed_revision=int(observed.get("revision", desired.router.get("observed_revision", 0))),
            desired_digest=stable_digest(compiled),
            observed_digest=stable_digest(observed_config),
            operations=structural_diff(observed_config, compiled),
            restart_required=bool(observed.get("restart_required", False)),
        )

    async def inspect(self, router_id: str) -> dict[str, Any]:
        desired = RouterDesiredState.model_validate(await self.source.desired_state(router_id))
        adapter = self._adapter(desired)
        observed = await adapter.read_observed(desired)
        plan = await self.preview(router_id)
        return {
            "router": desired.router,
            "desired_revision": desired.desired_revision,
            "observed": observed,
            "capabilities": await adapter.discover_capabilities(),
            "drift": plan.model_dump(mode="json"),
        }

    async def reconcile(self, router_id: str) -> ReconcileResult:
        lock = self._locks.setdefault(router_id, asyncio.Lock())
        async with lock:
            desired = RouterDesiredState.model_validate(await self.source.desired_state(router_id))
            adapter = self._adapter(desired)
            plan = await self.preview(router_id)
            compiled = adapter.compile(desired)
            try:
                if plan.drifted:
                    await adapter.apply(desired, compiled, plan)
                verified = await adapter.verify(desired, compiled)
                if not verified:
                    raise RouterControllerError("router read-back did not match desired configuration")
                await self.source.report_observed(
                    router_id, observed_revision=desired.desired_revision, health="READY",
                    supported_features=await adapter.discover_capabilities(),
                )
                return ReconcileResult(
                    router_id=router_id, status="IN_SYNC",
                    desired_revision=desired.desired_revision,
                    observed_revision=desired.desired_revision,
                    operations_applied=len(plan.operations), verified=True,
                )
            except Exception as error:
                await self.source.report_observed(
                    router_id, observed_revision=plan.observed_revision,
                    health="DEGRADED", last_error=f"{type(error).__name__}: {error}",
                )
                return ReconcileResult(
                    router_id=router_id, status="FAILED",
                    desired_revision=desired.desired_revision,
                    observed_revision=plan.observed_revision,
                    operations_applied=0, verified=False,
                    error=f"{type(error).__name__}: {error}",
                )

    async def reconcile_all(self) -> list[ReconcileResult]:
        return [await self.reconcile(router_id) for router_id in await self.source.list_router_ids()]

    def _adapter(self, desired: RouterDesiredState) -> RouterAdapter:
        kind = str(desired.router.get("kind", ""))
        try:
            return self.adapters[kind]
        except KeyError as error:
            raise RouterControllerError(f"no adapter is registered for router kind {kind!r}") from error
