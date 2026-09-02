"""Transport-neutral identities, results, and errors for PRA Control Plane."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ControlError(Exception):
    """Base class for failures that presentation adapters translate to protocols."""

    code = "control_error"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class NotFound(ControlError):
    code = "not_found"
    status_code = 404


class Unauthorized(ControlError):
    code = "unauthorized"
    status_code = 401


class Forbidden(ControlError):
    code = "forbidden"
    status_code = 403


class Conflict(ControlError):
    code = "conflict"
    status_code = 409


class NotSupported(ControlError):
    code = "not_supported"
    status_code = 422


class RestartRequired(ControlError):
    code = "restart_required"
    status_code = 409


class Offline(ControlError):
    code = "offline"
    status_code = 503


class InvalidRequest(ControlError):
    code = "validation_error"
    status_code = 422


class ApprovalRequired(ControlError):
    code = "approval_required"
    status_code = 409


class CallerContext(BaseModel):
    """Identity and trace metadata propagated through every presentation."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    roles: list[str] = Field(default_factory=list)
    permissions: set[str] = Field(default_factory=set)
    auth_source: str
    tenant: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    transport: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FleetSummary(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


class EngineDetails(BaseModel):
    instance_id: str
    section: str
    value: Any


class DriftResult(BaseModel):
    status: str
    differences: list[dict[str, Any]] = Field(default_factory=list)
    models: dict[str, Any] = Field(default_factory=dict)
    desired_revision: int | None = None


class QualificationSummary(BaseModel):
    model_id: str
    engine: str
    hardware: str | None = None
    status: str = "NOT_MEASURED"
    recommendation: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class ActionPlan(BaseModel):
    """Immutable inspection result used by every mutation presentation."""

    plan_id: str
    action: str
    target: str
    requested_change: dict[str, Any] = Field(default_factory=dict)
    current_state: dict[str, Any] = Field(default_factory=dict)
    projected_state: dict[str, Any] = Field(default_factory=dict)
    reversible: bool
    restart_required: bool
    impact: str
    required_permission: str
    requires_confirmation: bool
    expires_at: datetime | None = None
    idempotency_key: str | None = None
    created_by: str | None = None
    tenant: str | None = None


class ActionResult(BaseModel):
    plan_id: str
    action: str
    target: str
    status: str
    result: Any = None
    applied_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    idempotent_replay: bool = False


class MetricsSummary(BaseModel):
    engine: str | None = None
    period: str = "15m"
    metrics: dict[str, Any] = Field(default_factory=dict)
    links: dict[str, str | None] = Field(default_factory=dict)


class ContextSummary(BaseModel):
    task: str
    repository: str | None = None
    fleet: dict[str, Any] = Field(default_factory=dict)
    bundles: list[dict[str, Any]] = Field(default_factory=list)
    qualifications: list[dict[str, Any]] = Field(default_factory=list)
    deployments: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


def domain_payload(value: Any) -> Any:
    """Convert domain models recursively to JSON-ready protocol payloads."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [domain_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): domain_payload(item) for key, item in value.items()}
    return value
