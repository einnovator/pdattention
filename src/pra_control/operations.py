"""Canonical operation and MCP-tool catalog for all presentations."""

from __future__ import annotations

from fnmatch import fnmatchcase

from pydantic import BaseModel, ConfigDict


class OperationSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    permission: str
    side_effect: str = "none"
    risk: str = "read"
    description: str


class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    operations: tuple[str, ...]
    read_only: bool = True
    default_enabled: bool = True
    description: str


OPERATION_CATALOG: tuple[OperationSpec, ...] = (
    OperationSpec(id="fleet.list", permission="fleet:read", description="List fleet state and drift."),
    OperationSpec(id="engine.inspect", permission="engine:read", description="Inspect an engine state section."),
    OperationSpec(id="engine.register", permission="engine:configure", side_effect="write", risk="write", description="Register a manually managed engine."),
    OperationSpec(id="engine.remove", permission="engine:high-impact", side_effect="write", risk="high", description="Remove a manually managed engine."),
    OperationSpec(id="engine.action", permission="engine:action", side_effect="write", risk="write", description="Apply a supported operational engine action."),
    OperationSpec(id="engine.config.patch", permission="engine:configure", side_effect="write", risk="write", description="Patch engine configuration."),
    OperationSpec(id="registry.list", permission="registry:read", description="List Registry resources."),
    OperationSpec(id="registry.write", permission="registry:write", side_effect="write", risk="write", description="Create or update Registry resources."),
    OperationSpec(id="qualification.read", permission="qualification:read", description="Read support status and evidence."),
    OperationSpec(id="qualification.approve", permission="qualification:approve", side_effect="write", risk="high", description="Approve or revoke governed evidence."),
    OperationSpec(id="deployment.read", permission="deployment:read", description="Read desired state and drift."),
    OperationSpec(id="deployment.write", permission="deployment:write", side_effect="write", risk="write", description="Change desired deployment state."),
    OperationSpec(id="deployment.apply", permission="deployment:apply", side_effect="write", risk="high", description="Reconcile desired state."),
    OperationSpec(id="router.list", permission="registry:read", description="List and inspect router instances and drift."),
    OperationSpec(id="route.list", permission="registry:read", description="List and inspect logical routes and pool membership."),
    OperationSpec(id="route.plan", permission="registry:read", description="Preview router reconciliation without applying it."),
    OperationSpec(id="route.apply", permission="deployment:apply", side_effect="write", risk="high", description="Apply and verify router desired state."),
    OperationSpec(id="action.plan", permission="engine:read", description="Plan and classify an operational action."),
    OperationSpec(id="action.apply", permission="engine:action", side_effect="write", risk="write", description="Apply a previously inspected action plan."),
    OperationSpec(id="observability.read", permission="observability:read", description="Read semantic metrics and observability links."),
    OperationSpec(id="audit.read", permission="audit:read", description="Read central mutation audit records."),
    OperationSpec(id="context.read", permission="fleet:read", description="Assemble deterministic task context."),
    OperationSpec(id="experiment.read", permission="experiment:read", description="Read experiment evidence."),
    OperationSpec(id="experiment.run", permission="experiment:run", side_effect="write", risk="high", description="Submit an enabled research experiment."),
)

OPERATIONS = {item.id: item for item in OPERATION_CATALOG}

TOOL_CATALOG: tuple[ToolSpec, ...] = (
    ToolSpec(name="pra_fleet", operations=("fleet.list",), description="List and filter managed PRA instances."),
    ToolSpec(name="pra_engine", operations=("engine.inspect",), description="Inspect one engine or gateway."),
    ToolSpec(name="pra_gateway", operations=("engine.inspect",), description="Inspect gateway state."),
    ToolSpec(name="pra_catalog", operations=("registry.list",), description="Read models, bundles, profiles, and compatibility."),
    ToolSpec(name="pra_qualification", operations=("qualification.read",), description="Read model-engine support and evidence."),
    ToolSpec(name="pra_deployment", operations=("deployment.read",), description="Read desired state and drift."),
    ToolSpec(name="pra_router", operations=("router.list", "route.plan"), description="Inspect router instances and desired-state drift."),
    ToolSpec(name="pra_route", operations=("route.list",), description="Inspect logical routes, pools, and eligible backends."),
    ToolSpec(name="pra_metrics", operations=("observability.read",), description="Read semantic metrics and dashboard links."),
    ToolSpec(name="pra_context", operations=("context.read",), description="Assemble compact task-relevant control context."),
    ToolSpec(name="pra_plan", operations=("action.plan",), description="Plan an operational change without applying it."),
    ToolSpec(name="pra_apply", operations=("action.apply",), read_only=False, default_enabled=False, description="Apply an approved action plan."),
    ToolSpec(name="pra_experiment", operations=("experiment.read", "experiment.run"), read_only=False, default_enabled=False, description="Inspect or submit research experiments."),
)

TOOLS = {item.name: item for item in TOOL_CATALOG}


def allowed(name: str, allow: list[str], deny: list[str]) -> bool:
    """Apply deny-overrides-allow glob filtering to an operation or tool ID."""
    if any(fnmatchcase(name, pattern) for pattern in deny):
        return False
    return any(fnmatchcase(name, pattern) for pattern in allow)


def operation_documentation() -> str:
    rows = ["| Operation | Permission | Side effect | Risk |", "|---|---|---|---|"]
    rows.extend(f"| `{item.id}` | `{item.permission}` | {item.side_effect} | {item.risk} |" for item in OPERATION_CATALOG)
    return "\n".join(rows)


def tool_documentation() -> str:
    rows = ["| Tool | Operations | Default |", "|---|---|---|"]
    rows.extend(
        f"| `{item.name}` | {', '.join(f'`{op}`' for op in item.operations)} | {'read-only' if item.default_enabled else 'disabled'} |"
        for item in TOOL_CATALOG
    )
    return "\n".join(rows)
