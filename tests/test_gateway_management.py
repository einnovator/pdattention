from __future__ import annotations

import json
import socket
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
from fastapi.testclient import TestClient

from pra_hf.deployment import (
    PRAEngineCapabilities,
    PRAEngineIntegrationLevel,
    PRAEngineResult,
    PRAWireRequest,
    PRAWireResource,
)
from pra_hf.gateway import PRAGateway
from pra_hf.gateway_cli import gateway_cli
from pra_hf.gateway_management import (
    GATEWAY_API_PREFIX,
    GATEWAY_MANAGEMENT_PROTOCOL,
    GatewayManagementAPIConfig,
    GatewayManagementAuthConfig,
    GatewayManagementProvider,
    GatewayMetricRecorder,
    GatewayPolicy,
    GatewayRegistryConfig,
    GatewayRegistryReporter,
    GatewayUpstreamRouter,
    UpstreamCreate,
    create_gateway_management_app,
    start_gateway_management_api,
    stop_gateway_management_api,
)
from pra_hf.management import AuthMode
from pra_hf.management_cli import ManagementClient


class _Metrics:
    tracing_enabled = False
    metrics_enabled = False
    config = None

    def increment(self, *_args, **_kwargs):
        pass

    def observe(self, *_args, **_kwargs):
        pass

    def set_gauge(self, *_args, **_kwargs):
        pass

    @contextmanager
    def span(self, *_args, **_kwargs):
        yield None

    @staticmethod
    def hash_id(value):
        return "none" if value is None else f"hash-{value}"


class _Adapter:
    def __init__(self, name: str = "test", *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.requests = []
        self.closed = []

    def capabilities(self):
        return PRAEngineCapabilities(
            adapter=self.name,
            integration_level=PRAEngineIntegrationLevel.E1_LOGICAL_PRA,
            session_state=True,
            incremental_messages=True,
            resource_delta=True,
            logical_refs=True,
            typed_records=True,
            streaming=True,
        )

    def prepare_session(self, request):
        return request.session_id

    def generate(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("upstream unavailable")
        return PRAEngineResult("ok", {"engine_session_id": request.session_id})

    def stream(self, request):
        self.requests.append(request)
        yield {"text": "ok"}

    def close_session(self, session_id):
        self.closed.append(session_id)


def _provider(*, settings=None, policy=None, adapter=None):
    settings = settings or GatewayManagementAPIConfig()
    adapter = adapter or _Adapter()
    initial = UpstreamCreate(
        upstream_id="primary",
        name="Primary",
        base_url="http://127.0.0.1:9999",
        models=("test/model",),
        priority=0,
    )
    router = GatewayUpstreamRouter(
        initial, adapter, policy or GatewayPolicy(default_upstream_id="primary")
    )
    metrics = GatewayMetricRecorder(_Metrics())
    gateway = PRAGateway(router, mode="G11", observability=metrics)
    return GatewayManagementProvider(gateway, router, settings, metrics), adapter


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait(client: ManagementClient) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if client.get(f"{GATEWAY_API_PREFIX}/health")["status"] == "healthy":
                return
        except Exception:
            time.sleep(0.05)
    raise AssertionError("gateway management listener did not become ready")


def test_gateway_management_is_disabled_by_default(monkeypatch) -> None:
    for name in tuple(key for key in __import__("os").environ if key.startswith("PRA_GATEWAY_")):
        monkeypatch.delenv(name, raising=False)
    settings = GatewayManagementAPIConfig.from_mapping()
    provider, _ = _provider(settings=settings)
    assert settings.enabled is False
    assert settings.port == 9150
    assert start_gateway_management_api(provider, settings) is None


def test_openapi_contains_versioned_gateway_surface() -> None:
    provider, _ = _provider()
    client = TestClient(create_gateway_management_app(provider, provider.settings))
    schema = client.get("/openapi.json").json()
    expected = {
        f"{GATEWAY_API_PREFIX}/{path}"
        for path in (
            "health", "info", "capabilities", "config", "state", "upstreams",
            "sessions", "resources", "transport", "observability", "audit",
            "actions/clear-capability-cache", "actions/reload-policy",
        )
    }
    assert expected.issubset(schema["paths"])
    assert client.get(f"{GATEWAY_API_PREFIX}/health").json()["protocol"] == GATEWAY_MANAGEMENT_PROTOCOL


def test_checked_in_gateway_openapi_matches_runtime_surface() -> None:
    provider, _ = _provider()
    generated = create_gateway_management_app(provider, provider.settings).openapi()
    checked_in = json.loads(Path(
        "docs/site/api/openapi/pra-gateway-management-v1.json"
    ).read_text(encoding="utf-8"))
    assert checked_in == generated


def test_static_auth_and_gateway_admin_scope() -> None:
    read_only = GatewayManagementAPIConfig(auth=GatewayManagementAuthConfig(
        mode=AuthMode.STATIC_BEARER, token="secret", scopes=("pra-gateway:read",)
    ))
    provider, _ = _provider(settings=read_only)
    client = TestClient(create_gateway_management_app(provider, read_only))
    assert client.get(f"{GATEWAY_API_PREFIX}/info").status_code == 401
    headers = {"Authorization": "Bearer secret"}
    assert client.get(f"{GATEWAY_API_PREFIX}/info", headers=headers).status_code == 200
    assert client.patch(f"{GATEWAY_API_PREFIX}/config", headers=headers, json={"default_profile": "x"}).status_code == 403

    admin = GatewayManagementAPIConfig(auth=GatewayManagementAuthConfig(
        mode=AuthMode.STATIC_BEARER, token="admin", scopes=("pra-gateway:admin",)
    ))
    provider, _ = _provider(settings=admin)
    client = TestClient(create_gateway_management_app(provider, admin))
    assert client.get(
        f"{GATEWAY_API_PREFIX}/info", headers={"Authorization": "Bearer admin"}
    ).status_code == 200


def test_upstream_crud_routing_and_secret_redaction(monkeypatch) -> None:
    monkeypatch.setenv("PRIVATE_UPSTREAM_TOKEN", "do-not-return")
    provider, primary = _provider(policy=GatewayPolicy(
        default_upstream_id="primary", upstream_selection="model"
    ))
    client = TestClient(create_gateway_management_app(provider, provider.settings))
    created = client.post(f"{GATEWAY_API_PREFIX}/upstreams", json={
        "upstream_id": "secondary", "name": "Secondary", "base_url": "http://127.0.0.1:9998",
        "models": ["other/model"], "auth_reference": "PRIVATE_UPSTREAM_TOKEN", "priority": 2,
    })
    assert created.status_code == 201
    assert "do-not-return" not in created.text
    patched = client.patch(
        f"{GATEWAY_API_PREFIX}/upstreams/secondary", json={"weight": 2.0, "enabled": False}
    )
    assert patched.status_code == 200
    assert patched.json()["weight"] == 2.0
    assert provider.upstreams.generate(PRAWireRequest(
        model="test/model", messages=({"role": "user", "content": "hello"},)
    )).text == "ok"
    assert len(primary.requests) == 1
    assert client.delete(f"{GATEWAY_API_PREFIX}/upstreams/secondary").status_code == 200
    assert [row.event for row in provider.audit][-3:] == [
        "UPSTREAM_ADDED", "UPSTREAM_CHANGED", "UPSTREAM_REMOVED"
    ]


def test_capability_negotiation_and_policy_patch(monkeypatch) -> None:
    provider, _ = _provider()

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self):
            return json.dumps({
                "protocol": "pra-engine/1",
                "effective_capabilities": {"typed_records": True, "native_kv": True, "native_serving": True},
            }).encode()

    monkeypatch.setattr("pra_hf.gateway_management.urllib.request.urlopen", lambda *_args, **_kwargs: _Response())
    client = TestClient(create_gateway_management_app(provider, provider.settings))
    response = client.post(
        f"{GATEWAY_API_PREFIX}/actions/renegotiate/primary", json={"reason": "refresh"}
    )
    assert response.status_code == 200
    assert response.json()["detail"]["native_memory"] == "validated"
    changed = client.patch(f"{GATEWAY_API_PREFIX}/config", json={
        "policy": {"upstream_selection": "failover", "default_upstream_id": "primary"}
    })
    assert changed.status_code == 200
    assert changed.json()["policy"]["upstream_selection"] == "failover"


def test_session_resources_are_useful_but_privacy_safe() -> None:
    provider, _ = _provider()
    provider.gateway.generate(PRAWireRequest(
        model="test/model",
        messages=({"role": "user", "content": "private question"},),
        tenant_id="customer-secret",
        session_id="session-secret",
        resources=(PRAWireResource(
            resource_id="resource-secret", uri="memory://private", text="one two three",
            authorization_scope="tenant:private", record_type="tool_result",
        ),),
    ))
    client = TestClient(create_gateway_management_app(provider, provider.settings))
    sessions = client.get(f"{GATEWAY_API_PREFIX}/sessions").json()["items"]
    resources = client.get(f"{GATEWAY_API_PREFIX}/resources").json()["items"]
    serialized = json.dumps({"sessions": sessions, "resources": resources})
    assert "private question" not in serialized
    assert "customer-secret" not in serialized
    assert "session-secret" not in serialized
    assert "resource-secret" not in serialized
    assert sessions[0]["canonical_message_count"] >= 1
    assert sessions[0]["serialized_message_count"] >= sessions[0]["canonical_message_count"]
    assert sessions[0]["upstream_id"] == "primary"
    assert resources[0]["record_type"] == "tool_result"
    assert resources[0]["body_known"] is True
    assert resources[0]["token_count"] == 3

    safe_id = sessions[0]["session_id"]
    resync = client.post(
        f"{GATEWAY_API_PREFIX}/actions/resync-session/{safe_id}", json={"reason": "recover"}
    )
    assert resync.status_code == 200
    dropped = client.post(
        f"{GATEWAY_API_PREFIX}/actions/drop-session/{safe_id}", json={"reason": "operator request"}
    )
    assert dropped.status_code == 200
    assert client.get(f"{GATEWAY_API_PREFIX}/sessions").json()["total"] == 0


def test_failed_action_is_audited_without_sensitive_detail() -> None:
    provider, _ = _provider()
    client = TestClient(create_gateway_management_app(provider, provider.settings))
    response = client.post(
        f"{GATEWAY_API_PREFIX}/actions/drop-session/missing",
        headers={"x-request-id": "request-1", "x-trace-id": "trace-1"},
        json={"reason": "cleanup"},
    )
    assert response.status_code == 404
    event = client.get(f"{GATEWAY_API_PREFIX}/audit").json()["items"][0]
    assert event["event"] == "SESSION_DROPPED"
    assert event["result"] == "failure"
    assert event["request_id"] == "request-1"
    assert event["trace_id"] == "trace-1"


def test_registry_reporter_registers_heartbeats_and_offline() -> None:
    settings = GatewayManagementAPIConfig(registry=GatewayRegistryConfig(
        enabled=True, url="http://registry", deployment_id="gateway-1", model_id="model-1"
    ))
    provider, _ = _provider(settings=settings)
    reporter = GatewayRegistryReporter(provider)
    calls = []
    reporter._request = lambda method, path, body: calls.append((method, path, body)) or {}
    reporter._publish("healthy")
    reporter._publish("offline")
    assert calls[0][0:2] == ("PATCH", "/v1/deployments/gateway-1")
    assert calls[0][2]["engine_instance_selector"]["protocol"] == GATEWAY_MANAGEMENT_PROTOCOL
    assert calls[1][2]["engine_instance_selector"]["health"] == "offline"


def test_gateway_remote_cli_end_to_end() -> None:
    port = _free_port()
    settings = GatewayManagementAPIConfig(enabled=True, port=port)
    provider, _ = _provider(settings=settings)
    server = start_gateway_management_api(provider, settings)
    url = f"http://127.0.0.1:{port}"
    _wait(ManagementClient(url))
    runner = CliRunner()
    try:
        for command in ("health", "upstreams", "sessions", "transport", "config", "inspect"):
            result = runner.invoke(gateway_cli, [command, "--management-url", url, "--json"])
            assert result.exit_code == 0, (command, result.output, result.exception)
        result = runner.invoke(gateway_cli, [
            "renegotiate", "primary", "--reason", "cli test", "--management-url", url, "--json",
        ])
        assert result.exit_code == 0, (result.output, result.exception)
    finally:
        stop_gateway_management_api(server)
