from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pra_control.app import COOKIE, ControlRuntime, create_app
from pra_control.auth import AuthService, Identity
from pra_control.clients import AsyncServiceClient
from pra_control.config import (
    ControlAuthConfig,
    ControlAuthProfile,
    ControlAgentConfig,
    ControlPlaneConfig,
    EngineTargetConfig,
    FleetConfig,
    IdentityProviderConfig,
    ServiceLinkConfig,
)
from pra_control.fleet import FleetService, compare_desired_observed
from pra_control.managers import ControlManager
from pra_control.rbac import Permission, Role, permits


class FakeRegistry:
    def __init__(self) -> None:
        self.calls = []

    async def list(self, resource, *, limit=200, offset=0):
        return {"items": [{"id": f"{resource}-1"}], "limit": limit, "offset": offset}

    async def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        return {"id": body.get("id", "changed"), "state": "APPROVED"}

    async def close(self):
        pass


class FakeFleet:
    def __init__(self) -> None:
        self.registry = FakeRegistry()
        self.actions = []

    async def overview(self):
        return {
            "items": [{
                "name": "mlx-01", "status": "DRIFT", "engine": "mlx", "model": "Qwen/Qwen3-4B",
                "health": "healthy", "inference_url": "http://mlx.test:8000",
                "cluster": "mac-lab", "environment": "test", "alerts": ["desired state drift"],
                "models": [{
                    "runtime_model_id": "qwen-4b", "model_id": "Qwen/Qwen3-4B",
                    "profile": "BALANCED", "execution_mode": "native-memory",
                }],
                "model_count": 1,
                "metrics": {"storage_reloads": 12},
                "drift": {"status": "DRIFT", "differences": [{"field": "profile", "desired": "BALANCED", "observed": "ECONOMY"}]},
            }],
            "summary": {"total": 1, "healthy": 0, "drift": 1, "offline": 0, "unknown": 0},
        }

    async def engine_section(self, name, section):
        if name != "mlx-01":
            raise KeyError(name)
        return {"engine": "mlx", "section": section, "secret": None}

    async def action(self, name, action, body):
        self.actions.append((name, action, body))
        return {"status": "accepted", "action": action}

    async def patch_config(self, name, body):
        return {"name": name, "effective": body}

    async def close(self):
        await self.registry.close()


@pytest.fixture
def control(monkeypatch, tmp_path):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "test-cookie-secret-that-is-not-for-production")
    config = ControlPlaneConfig(
        database_url=f"sqlite:///{tmp_path / 'control.db'}",
        registry=ServiceLinkConfig(url=None),
        grafana=ServiceLinkConfig(url="https://grafana.test/d/pra"),
        tempo=ServiceLinkConfig(url="https://tempo.test"),
        prometheus=ServiceLinkConfig(url="https://prometheus.test"),
    )
    runtime = ControlRuntime(config)
    runtime.fleet = FakeFleet()
    app = create_app(config, runtime=runtime)
    with TestClient(app) as client:
        yield client, runtime


def set_identity(client: TestClient, runtime: ControlRuntime, role: Role) -> Identity:
    identity = Identity(f"test:{role.value}", role.value, None, role, "test", f"csrf-{role.value}")
    client.cookies.set(COOKIE, runtime.auth.codec.encode(identity))
    return identity


def test_control_plane_accepts_configured_agent_bearer(control, monkeypatch):
    client, runtime = control
    monkeypatch.setenv("PRA_AGENT_CONTROL_TOKEN", "agent-secret")
    runtime.config.auth_profiles["agent"] = ControlAuthProfile(
        type="bearer_token", subject="pra-agent", roles=[Role.VIEWER],
        token_env="PRA_AGENT_CONTROL_TOKEN",
    )
    response = client.get("/api/fleet", headers={"Authorization": "Bearer agent-secret"})
    assert response.status_code == 200
    denied = client.get("/api/fleet", headers={"Authorization": "Bearer wrong"})
    assert denied.status_code == 401


def test_config_precedence_and_non_loopback_guard(monkeypatch, tmp_path):
    path = tmp_path / "control.yaml"
    path.write_text("control_plane:\n  host: 127.0.0.1\n  port: 9301\n", encoding="utf-8")
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "secret")
    monkeypatch.setenv("PRA_CONTROL_PORT", "9302")
    config = ControlPlaneConfig.load(path, overrides={"port": 9303})
    assert config.port == 9303
    config.host = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        config.validate_security()
    config.auth.allow_local_auth_non_loopback = True
    config.validate_security()


def test_control_agent_and_inference_target_configuration(monkeypatch, tmp_path):
    path = tmp_path / "control.yaml"
    path.write_text(
        "control_plane:\n"
        "  agent:\n"
        "    engine: mlx-01\n"
        "    model: qwen-4b\n"
        "    api_key_env: PRA_CHAT_KEY\n"
        "  fleet:\n"
        "    engines:\n"
        "      - name: mlx-01\n"
        "        management_url: http://mlx.test:9101\n"
        "        inference_url: http://mlx.test:8000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PRA_CHAT_KEY", "private-chat-key")
    config = ControlPlaneConfig.load(path)
    assert config.agent == ControlAgentConfig(engine="mlx-01", model="qwen-4b", api_key_env="PRA_CHAT_KEY")
    assert config.agent.api_key() == "private-chat-key"
    assert config.fleet.engines[0].inference_url == "http://mlx.test:8000"
    assert "private-chat-key" not in config.model_dump_json()


def test_rbac_and_drift_comparison():
    assert permits(Role.VIEWER, Permission.FLEET_READ)
    assert not permits(Role.VIEWER, Permission.ENGINE_ACTION)
    assert permits(Role.APPROVER, Permission.APPROVE)
    desired = {"desired_model_id": "model-a", "desired_profile_id": "BALANCED", "desired_revision": 4}
    observed = {"models": {"items": [{"model_id": "model-a", "profile": "ECONOMY"}]}}
    result = compare_desired_observed(desired, observed)
    assert result["status"] == "DRIFT"
    assert result["differences"] == [{"field": "profile", "desired": "BALANCED", "observed": "ECONOMY"}]


def test_multi_model_drift_matches_runtime_identity_and_rejects_extras():
    desired = {
        "desired_revision": 9,
        "allow_extra_models": False,
        "desired_models": [
            {"runtime_model_id": "qwen", "model_id": "org/qwen", "profile_id": "BALANCED"},
            {"runtime_model_id": "gemma", "model_id": "org/gemma", "profile_id": "QUALITY"},
        ],
    }
    observed = {"models": {"items": [
        {"runtime_model_id": "qwen", "model_id": "org/qwen", "profile": "BALANCED"},
        {"runtime_model_id": "gemma", "model_id": "org/gemma", "profile": "ECONOMY"},
        {"runtime_model_id": "extra", "model_id": "org/extra"},
    ]}}
    result = compare_desired_observed(desired, observed)
    assert result["status"] == "DRIFT"
    assert result["models"]["qwen"]["status"] == "IN_SYNC"
    assert result["models"]["gemma"]["differences"][0]["field"] == "profile"
    assert result["models"]["extra"]["differences"][0]["field"] == "UNAPPROVED_MODEL_LOADED"


def test_fleet_aggregates_two_engines_against_one_registry(control):
    _, runtime = control

    class Registry:
        async def deployments(self):
            return [{
                "id": "desired", "environment": "test", "cluster": "lab", "desired_revision": 2,
                "desired_model_id": "model-a", "desired_profile_id": "BALANCED", "desired_mode": "native-memory",
            }]

        async def close(self):
            pass

    class Engine:
        def __init__(self, name, *_):
            self.name = name

        async def snapshot(self):
            profile = "BALANCED" if self.name == "one" else "ECONOMY"
            return {
                "info": {"engine": "test", "health": "healthy"},
                "models": {"items": [{"model_id": "model-a", "profile": profile, "execution_mode": "native-memory"}]},
                "storage": {"metrics": {}},
            }

        async def close(self):
            pass

    config = runtime.config.model_copy(update={"fleet": FleetConfig(
        discovery_mode="static",
        engines=[
            EngineTargetConfig(name="one", management_url="http://one", environment="test", cluster="lab"),
            EngineTargetConfig(name="two", management_url="http://two", environment="test", cluster="lab"),
        ],
    )})
    service = FleetService(config, runtime.store, engine_factory=Engine, registry_client=Registry())
    manager = ControlManager.build(config, runtime.store, service)

    async def inspect():
        try:
            identity = Identity("test:viewer", "Viewer", None, Role.VIEWER, "test", "csrf")
            return (await manager.fleet.list(identity.caller(transport="test"))).model_dump(mode="json")
        finally:
            await service.close()

    import asyncio
    result = asyncio.run(inspect())
    assert result["summary"] == {"total": 2, "healthy": 1, "drift": 1, "offline": 0, "unknown": 0}
    assert [row["status"] for row in result["items"]] == ["IN_SYNC", "DRIFT"]


def test_service_token_is_backend_header_not_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    async def exercise():
        client = AsyncServiceClient("engine", "https://engine.test", "backend-secret", transport=httpx.MockTransport(handler))
        try:
            assert await client.request("POST", "/action", {"resource": "r1"}) == {"ok": True}
        finally:
            await client.close()

    import asyncio
    asyncio.run(exercise())
    assert captured == {"authorization": "Bearer backend-secret", "body": {"resource": "r1"}}


def test_oauth_transaction_rejects_changed_state(monkeypatch):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "secret")
    monkeypatch.setenv("GITHUB_SECRET", "provider-secret")
    auth = AuthService(ControlAuthConfig(providers=[IdentityProviderConfig(
        name="github", kind="github", client_id="client", client_secret_env="GITHUB_SECRET",
    )]), "https://control.test")
    url, transaction = auth.begin("github")
    assert "code_challenge=" in url and "nonce=" in url

    async def changed_state():
        await auth.callback("github", "code", "wrong", transaction)

    with pytest.raises(ValueError, match="state mismatch"):
        import asyncio
        asyncio.run(changed_state())


def test_fleet_read_observability_and_recommendations(control):
    client, runtime = control
    assert client.get("/index.html").status_code == 200
    response = client.get("/api/fleet")
    assert response.status_code == 200
    assert response.json()["items"][0]["name"] == "mlx-01"
    assert client.get("/api/engines/mlx-01/storage").json()["section"] == "storage"
    links = client.get("/api/observability/links?engine=mlx-01&trace_id=abc").json()
    assert "var-engine=mlx-01" in links["grafana"]
    assert "traceId=abc" in links["tempo"]
    recommendations = client.get("/api/recommendations").json()
    assert {row["kind"] for row in recommendations["items"]} == {"reconcile", "warm-quota"}


def test_viewer_cannot_mutate_and_operator_action_is_audited(control):
    client, runtime = control
    viewer = set_identity(client, runtime, Role.VIEWER)
    denied = client.post(
        "/api/engines/mlx-01/actions/prefetch",
        headers={"X-CSRF-Token": viewer.csrf_token},
        json={"values": {}, "reason": "test", "confirmed": False},
    )
    assert denied.status_code == 403
    operator = set_identity(client, runtime, Role.OPERATOR)
    accepted = client.post(
        "/api/engines/mlx-01/actions/prefetch",
        headers={"X-CSRF-Token": operator.csrf_token},
        json={"values": {"resource_id": "r1"}, "reason": "warm before launch", "confirmed": False},
    )
    assert accepted.status_code == 200
    events = client.get("/api/audit").json()["items"]
    assert events[0]["action"] == "engine.prefetch"
    assert events[0]["reason"] == "warm before launch"


def test_csrf_confirmation_and_registry_approval(control):
    client, runtime = control
    operator = set_identity(client, runtime, Role.OPERATOR)
    no_csrf = client.patch("/api/engines/mlx-01/config", json={"values": {}, "reason": "x"})
    assert no_csrf.status_code == 403
    evict = client.post(
        "/api/engines/mlx-01/actions/evict", headers={"X-CSRF-Token": operator.csrf_token},
        json={"values": {}, "reason": "budget", "confirmed": False},
    )
    assert evict.status_code == 409
    approver = set_identity(client, runtime, Role.APPROVER)
    response = client.post(
        "/api/registry/bundles/bundle-1/approve", headers={"X-CSRF-Token": approver.csrf_token},
        json={"values": {"comment": "qualified"}, "reason": "review complete"},
    )
    assert response.status_code == 200
    assert runtime.fleet.registry.calls[-1][1] == "/v1/approvals"
    assert runtime.fleet.registry.calls[-1][2]["resource_id"] == "bundle-1"


def test_manual_engine_requires_administrator(control):
    client, runtime = control
    admin = set_identity(client, runtime, Role.ADMINISTRATOR)
    response = client.post(
        "/api/engines", headers={"X-CSRF-Token": admin.csrf_token},
        json={"name": "vllm-02", "management_url": "http://vllm:9101", "token_env": "VLLM_TOKEN", "metadata": {"cluster": "gpu"}, "reason": "capacity"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["token_env"] == "VLLM_TOKEN"
    assert "backend-secret" not in response.text


def test_agent_websocket_resume_replay_and_duplicate_suppression(control):
    client, runtime = control
    with client.websocket_connect("/ws/agent") as socket:
        session = socket.receive_json()
        assert session["type"] == "session"
        token = session["resume_token"]
        socket.send_json({"type": "message", "message_id": "msg-1", "text": "Which engines are running Qwen3-4B?"})
        assert socket.receive_json() == {"type": "ack", "message_id": "msg-1", "duplicate": False}
        event = socket.receive_json()
        assert event["type"] == "tool.started"
        while event["type"] != "message.completed":
            event = socket.receive_json()
        last_sequence = event["sequence"]
        socket.send_json({"type": "message", "message_id": "msg-1", "text": "duplicate"})
        assert socket.receive_json()["duplicate"] is True
    with client.websocket_connect(f"/ws/agent?resume_token={token}&after={last_sequence - 1}") as resumed:
        assert resumed.receive_json()["resume_token"] == token
        replayed = resumed.receive_json()
        assert replayed["sequence"] == last_sequence
        assert replayed["type"] == "message.completed"


def test_agent_model_discovery_selection_commands_and_session_listing(control):
    client, runtime = control
    models = client.get("/api/agent/models").json()["items"]
    assert models == [{
        "target_id": "mlx-01:qwen-4b", "engine": "mlx-01", "engine_type": "mlx",
        "model": "qwen-4b", "model_id": "Qwen/Qwen3-4B", "reachable": True,
        "health": "healthy", "status": "DRIFT", "inference_url": "http://mlx.test:8000",
    }]
    identity = runtime.auth.development_identity()
    created = client.post(
        "/api/agent/sessions", headers={"X-CSRF-Token": identity.csrf_token},
        json={"target_id": "mlx-01:qwen-4b"},
    )
    assert created.status_code == 201
    assert created.json()["settings"]["target_id"] == "mlx-01:qwen-4b"
    assert created.json()["settings"]["inference_url"] == "http://mlx.test:8000"
    assert client.get("/api/agent/sessions").json()["items"][0]["event_count"] == 0

    with client.websocket_connect(f"/ws/agent?resume_token={created.json()['resume_token']}") as socket:
        assert socket.receive_json()["settings"]["target_id"] == "mlx-01:qwen-4b"
        socket.send_json({"type": "model.select", "target_id": "mlx-01:qwen-4b"})
        updated = socket.receive_json()
        assert updated["type"] == "session.updated"
        socket.send_json({"type": "model.select", "target_id": None})
        cleared = socket.receive_json()
        assert cleared["type"] == "session.updated"
        assert "target_id" not in cleared["settings"]
        socket.send_json({"type": "message", "message_id": "command-1", "text": "/models"})
        assert socket.receive_json()["type"] == "ack"
        event = socket.receive_json()
        while event["type"] != "message.completed":
            event = socket.receive_json()
        assert "mlx-01:qwen-4b: reachable" in event["text"]


def test_control_plane_agent_uses_selected_model_with_read_only_fallback_context(control, monkeypatch):
    _, runtime = control
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "Model-backed fleet answer."}}]}

    class Client:
        def __init__(self, **kwargs):
            captured["timeout"] = kwargs["timeout"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def post(self, url, *, headers, json):
            captured.update({"url": url, "headers": headers, "body": json})
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    emitted = []

    async def exercise():
        async def emit(kind, payload):
            emitted.append((kind, payload))

        return await runtime.agent.answer(
            runtime.auth.development_identity(), "Summarize the fleet", emit,
            target={
                "engine": "mlx-01", "model": "qwen-4b",
                "target_id": "mlx-01:qwen-4b", "inference_url": "http://mlx.test:8000",
            },
        )

    import asyncio
    assert asyncio.run(exercise()) == "Model-backed fleet answer."
    assert captured["url"] == "http://mlx.test:8000/v1/chat/completions"
    assert captured["body"]["model"] == "qwen-4b"
    assert "read-only PRA Control Plane assistant" in captured["body"]["messages"][0]["content"]
    assert emitted[-1][0] == "tool.completed"


def test_frontend_contains_required_stack_and_reconnect_protocol():
    static = Path(__file__).parents[1] / "src" / "pra_control" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    help_script = (static / "help.js").read_text(encoding="utf-8")
    styles = (static / "styles.css").read_text(encoding="utf-8")
    help_styles = (static / "help.css").read_text(encoding="utf-8")
    assert all(name in html for name in ("jquery", "bootstrap", "dockview", "lucide", "tippy", "marked", "purify"))
    assert '/static/app.js?v=' in html
    assert "localStorage" in script
    assert "resume_token" in script
    assert "Math.min(state.retry * 2, 30000)" in script
    assert "globalThis['dockview-core'].createDockview" in script
    assert "dockview.createDockview" not in script
    assert "init(params)" in script
    assert 'data-default-layout="20-50-30"' in html
    assert 'id="left-pane"' in html
    assert 'id="chat-pane"' in html
    assert 'id="chat-resize"' in html
    assert 'id="events-template"' not in html
    assert 'id="agent-template"' not in html
    assert script.count("dv.addPanel(") == 1
    assert "dv.getPanel(id)" in script
    assert "direction: 'within'" in script
    assert "pra-control-central-tabs-v3" in script
    assert "pra-control-chat-ratio-v3" in script
    assert "renderActivity" in script
    assert "fleet-filters" in script
    assert "data-fleet-filter" in script
    assert "data-sort" in script
    assert "data-open-engine" in script
    assert "field-info" in script
    assert "tippy.delegate" in script
    assert "placement: 'right-start'" in script
    assert 'id="info-modal"' not in html
    assert "loading-state" in script and "loading-spinner" in styles
    assert "host.className = 'workspace-host'" in script
    assert ".workspace-host { width: 100%; height: 100%; min-width: 0; }" in styles
    assert ".workspace-content, .engine-section { width: 100%; min-width: 0; }" in styles
    assert ".loading-state { width: 100%; min-width: 0;" in styles
    assert "Loading remote fleet state" in script
    assert "Engine state" in script and "Desired state" in script
    assert "runtime_model_id" in script
    assert "dynamic_model_unload" in script
    assert "--chat-width: 30%" in styles
    assert ".left-collapsed .left-pane" in styles
    assert ".sort-button:hover .sort-icon" in styles
    assert ".table-link" in styles and "cursor: pointer" in styles
    assert ".detail-field" in styles and "border: 0" in styles
    assert "pra-control-theme" in html + script
    assert 'id="user-menu-toggle"' in html
    assert 'id="help-drawer"' in html
    assert 'id="agent-sessions"' in html
    assert 'id="agent-model"' in html
    assert 'id="about-modal"' in html
    assert "DOMPurify" in help_script
    assert "marked.parse" in help_script
    assert "translateX(100%)" in help_styles
    for slug in ("index", "fleet", "registry", "agent", "activity"):
        packaged_help = static / "help" / f"{slug}.md"
        docs_help = Path(__file__).parents[1] / "docs" / "help" / "control-plane" / f"{slug}.md"
        assert packaged_help.read_text(encoding="utf-8") == docs_help.read_text(encoding="utf-8")
    assert "backend-secret" not in html + script + help_script
