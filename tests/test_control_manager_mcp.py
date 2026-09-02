from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pra_control.app import COOKIE, ControlRuntime, create_app
from pra_control.auth import Identity
from pra_control.config import (
    ControlAuthProfile,
    ControlPlaneConfig,
    MCPStdioTransportConfig,
    MCPTransportsConfig,
    MCPPresentationConfig,
    MCPToolsConfig,
    RESTPresentationConfig,
    ServiceLinkConfig,
)
from pra_control.domain import ApprovalRequired, Conflict, Forbidden
from pra_control.managers import ControlManager
from pra_control.mcp import MCPPresentation, build_fastmcp
from pra_control.persistence import ControlStore
from pra_control.rbac import Role
from pra_control.operations import OPERATION_CATALOG, TOOL_CATALOG


class FakeControlBackend:
    def __init__(self) -> None:
        self.registry = self
        self.actions: list[tuple[str, str, dict[str, Any]]] = []

    async def overview(self) -> dict[str, Any]:
        return {
            "items": [{
                "name": "mlx-01", "engine": "mlx", "status": "IN_SYNC",
                "model": "mlx-community/Qwen3-8B-4bit", "models": [{"runtime_model_id": "qwen", "model_id": "Qwen3-8B"}],
                "metrics": {"ttft_p95_ms": 42}, "drift": {"status": "IN_SYNC", "differences": []},
            }],
            "summary": {"total": 1, "healthy": 1, "drift": 0, "offline": 0, "unknown": 0},
        }

    async def engine_section(self, name: str, section: str) -> Any:
        if name != "mlx-01":
            raise KeyError(name)
        if section == "models":
            return {"items": [{"runtime_model_id": "qwen", "model_id": "Qwen3-8B"}]}
        return {"name": name, "section": section, "engine": "mlx", "credentials": None}

    async def targets(self) -> list[Any]:
        return []

    async def action(self, name: str, action: str, body: dict[str, Any]) -> Any:
        self.actions.append((name, action, dict(body)))
        return {"status": "accepted", "action": action}

    async def patch_config(self, name: str, body: dict[str, Any]) -> Any:
        self.actions.append((name, "config-patch", dict(body)))
        return {"status": "accepted", "effective": dict(body)}

    async def list(self, resource: str, *, limit: int = 200, offset: int = 0) -> dict[str, Any]:
        rows = {
            "models": [{"id": "model-1", "model_id": "Qwen3-8B"}],
            "bundles": [{"id": "bundle-1", "model_id": "Qwen3-8B"}],
            "qualifications": [{"id": "qualification-1", "model_id": "Qwen3-8B", "engine": "mlx", "status": "QUALIFIED"}],
            "deployments": [{"id": "deployment-1", "desired_model_id": "Qwen3-8B"}],
        }.get(resource, [])
        return {"items": rows[offset:offset + limit], "limit": limit, "offset": offset}

    async def request(self, method: str, path: str, body=None) -> Any:
        return {"method": method, "path": path, **dict(body or {})}

    async def deployments(self) -> list[dict[str, Any]]:
        return (await self.list("deployments"))["items"]

    async def close(self) -> None:
        return None


def caller(role: Role, *, transport: str = "test"):
    identity = Identity(f"test:{role.value}", role.value, None, role, "test", "csrf")
    return identity.caller(transport=transport, request_id="request-1", trace_id="trace-1")


@pytest.fixture
def manager(tmp_path):
    config = ControlPlaneConfig(database_url=f"sqlite:///{tmp_path / 'manager.db'}", registry=ServiceLinkConfig(url=None))
    store = ControlStore(config.database_url)
    backend = FakeControlBackend()
    return ControlManager.build(config, store, backend), backend, store, config


def test_manager_enforces_authorization_when_adapter_is_bypassed(manager):
    control, _, _, _ = manager

    async def exercise():
        assert (await control.fleet.list(caller(Role.VIEWER))).summary["total"] == 1
        plan = await control.actions.plan(caller(Role.VIEWER), "prefetch", "mlx-01", {"resource_id": "r1"})
        with pytest.raises(Forbidden, match="engine:action"):
            await control.actions.apply(caller(Role.VIEWER), plan.plan_id, reason="bypass")

    asyncio.run(exercise())


def test_plan_apply_is_idempotent_and_manager_audits_transport_metadata(manager):
    control, backend, store, _ = manager

    async def exercise():
        operator = caller(Role.OPERATOR, transport="mcp-stdio")
        plan = await control.actions.plan(operator, "prefetch", "mlx-01", {"resource_id": "r1"}, idempotency_key="same-request")
        first = await control.actions.apply(operator, plan.plan_id, reason="prepare launch")
        second = await control.actions.apply(operator, plan.plan_id, reason="retry")
        return first, second

    first, second = asyncio.run(exercise())
    assert first.status == "applied"
    assert second.idempotent_replay is True
    assert len(backend.actions) == 1
    event = store.audit_events()["items"][0]
    assert event["permission"] == "engine:action"
    assert event["transport"] == "mcp-stdio"
    assert event["request_id"] == "request-1"
    assert event["trace_id"] == "trace-1"
    assert event["idempotency_key"] == "same-request"


def test_idempotency_key_cannot_replay_a_different_action_intent(manager):
    control, _, _, _ = manager

    async def exercise():
        operator = caller(Role.OPERATOR)
        first = await control.actions.plan(
            operator, "prefetch", "mlx-01", {"resource_id": "r1"},
            idempotency_key="shared-key",
        )
        await control.actions.apply(operator, first.plan_id, reason="first")
        second = await control.actions.plan(
            operator, "promote", "mlx-01", {"resource_id": "r2"},
            idempotency_key="shared-key",
        )
        with pytest.raises(Conflict, match="different action intent"):
            await control.actions.apply(operator, second.plan_id, reason="second")

    asyncio.run(exercise())


def test_high_impact_action_requires_confirmation_and_permission(manager):
    control, _, _, _ = manager

    async def exercise():
        operator = caller(Role.OPERATOR)
        plan = await control.actions.plan(operator, "evict", "mlx-01", {"resource_id": "r1"})
        with pytest.raises(ApprovalRequired):
            await control.actions.apply(operator, plan.plan_id, reason="budget")
        with pytest.raises(Forbidden, match="engine:high-impact"):
            await control.actions.apply(operator, plan.plan_id, confirmation=True, reason="budget")

    asyncio.run(exercise())


def test_mcp_defaults_are_read_only_and_return_structured_domain_facts(manager):
    control, _, _, config = manager
    config.mcp = MCPPresentationConfig(enabled=True)
    presentation = MCPPresentation(control, config, lambda: caller(Role.VIEWER, transport="mcp-stdio"))
    discovery = presentation.discovery()
    names = {row["name"] for row in discovery["tools"]}
    assert "pra_context" in names
    assert "pra_plan" in names
    assert "pra_apply" not in names
    assert discovery["read_only_default"] is True

    fleet = asyncio.run(presentation.call("pra_fleet", {}))
    resource = asyncio.run(presentation.read("pra://engines/mlx-01/models/qwen"))
    disabled = asyncio.run(presentation.call("pra_apply", {"plan_id": "missing"}))
    assert fleet["ok"] and fleet["result"]["summary"]["total"] == 1
    assert resource["ok"] and resource["result"]["model_id"] == "Qwen3-8B"
    assert disabled["error"]["code"] == "not_found"


def test_mcp_tool_filtering_controls_official_sdk_discovery(manager):
    control, _, _, config = manager
    config.mcp = MCPPresentationConfig(
        enabled=True,
        tools=MCPToolsConfig(allow=["pra_fleet", "pra_metrics"], deny=["pra_metrics"]),
    )
    presentation = MCPPresentation(control, config, lambda: caller(Role.VIEWER))
    server = build_fastmcp(presentation)
    tools = asyncio.run(server.list_tools())
    templates = asyncio.run(server.list_resource_templates())
    assert [tool.name for tool in tools] == ["pra_fleet"]
    assert {str(item.uriTemplate) for item in templates} >= {"pra://engines/{instance_id}", "pra://models/{model_id}"}


def test_rest_exposure_hides_disabled_operations_from_routes_and_openapi(monkeypatch, tmp_path):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "test-cookie-secret")
    config = ControlPlaneConfig(
        database_url=f"sqlite:///{tmp_path / 'rest.db'}", registry=ServiceLinkConfig(url=None),
        rest=RESTPresentationConfig(allow=["fleet.list"], deny=[]),
    )
    runtime = ControlRuntime(config)
    runtime.fleet = FakeControlBackend()
    app = create_app(config, runtime=runtime)
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/fleet" in paths
        assert "/api/audit" not in paths
        assert client.get("/api/audit").status_code == 404


def test_shared_transition_route_cannot_bypass_operation_exposure(monkeypatch, tmp_path):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "test-cookie-secret")
    config = ControlPlaneConfig(
        database_url=f"sqlite:///{tmp_path / 'transition.db'}", registry=ServiceLinkConfig(url=None),
        rest=RESTPresentationConfig(allow=["qualification.approve"], deny=["registry.write"]),
    )
    app = create_app(config)
    with TestClient(app) as client:
        response = client.post(
            "/api/registry/models/model-1/deprecate",
            json={"values": {}, "reason": "must remain hidden"},
        )
        assert response.status_code == 404


def test_rest_and_mcp_share_the_same_manager_result(monkeypatch, tmp_path):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "test-cookie-secret")
    config = ControlPlaneConfig(database_url=f"sqlite:///{tmp_path / 'parity.db'}", registry=ServiceLinkConfig(url=None))
    runtime = ControlRuntime(config)
    runtime.fleet = FakeControlBackend()
    app = create_app(config, runtime=runtime)
    presentation = MCPPresentation(runtime.manager, config, lambda: caller(Role.VIEWER, transport="mcp-stdio"))
    with TestClient(app) as client:
        rest = client.get("/api/fleet").json()
    mcp = asyncio.run(presentation.call("pra_fleet", {}))["result"]
    assert rest == mcp


def test_mcp_security_configuration_and_secret_references(monkeypatch):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "cookie-secret")
    monkeypatch.setenv("PRA_MCP_TEST_SECRET", "must-not-be-serialized")
    config = ControlPlaneConfig(
        mcp=MCPPresentationConfig(
            enabled=True,
            transports=MCPTransportsConfig(stdio=MCPStdioTransportConfig(enabled=True, auth_profile="local")),
        ),
        auth_profiles={
            "local": ControlAuthProfile(
                type="bearer_token", subject="automation", roles=[Role.VIEWER], token_env="PRA_MCP_TEST_SECRET",
            ),
        },
    )
    config.validate_security()
    serialized = config.model_dump_json()
    assert "PRA_MCP_TEST_SECRET" in serialized
    assert "must-not-be-serialized" not in serialized

    config.mcp.transports.stdio.auth_profile = "missing"
    with pytest.raises(ValueError, match="unknown MCP auth profile"):
        config.validate_security()


def test_generated_operation_catalog_covers_runtime_metadata():
    page = (Path(__file__).parents[1] / "docs/site/control-plane/operation-catalog.md").read_text(encoding="utf-8")
    assert all(f"`{item.id}`" in page for item in OPERATION_CATALOG)
    assert all(f"`{item.name}`" in page for item in TOOL_CATALOG)
