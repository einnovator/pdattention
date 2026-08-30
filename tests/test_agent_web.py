from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from pra_hf.agent import AgentTurn, PRAAgentConfig
from pra_hf.agent_profiles import AgentLaunch, AgentProfile, AgentProfileDocument
from pra_hf.agent_web.app import AgentWebService, create_app
from pra_hf.agent_web.lifecycle import AgentWebLifecycle
from pra_hf.session_service import InMemorySessionService


class FakeRegistry:
    def __init__(self):
        self.profile = AgentProfile(name="default", model="fake/model")

    def load(self, **kwargs):
        return AgentProfileDocument(1, "default", {"default": self.profile}, ("fake",))

    def resolve(self, **kwargs):
        return self.profile, ("fake",)


class FakeAgent:
    def __init__(self):
        self.config = PRAAgentConfig()
        self.service = InMemorySessionService()
        self._state = None
        self.authorization_callback = None

    def start_session(self, session_id=None, **kwargs):
        self._state = self.service.create_session(
            self.config.user_id,
            session_id,
            task_description=kwargs.get("task_description"),
        )
        return self._state

    @property
    def state(self):
        return self._state

    def run_turn(self, text):
        return AgentTurn("reply: " + text, self._state)

    def close(self):
        return None


class FakeLauncher:
    def __init__(self):
        self.last_profile = None

    def launch(self, profile):
        self.last_profile = profile
        return AgentLaunch(FakeAgent(), profile, {"agent_profile": profile.name, "model": profile.model})


def test_web_profiles_sessions_messages_and_event_replay() -> None:
    service = AgentWebService(registry=FakeRegistry(), launcher=FakeLauncher())
    client = TestClient(create_app(service=service))

    assert client.get("/health").json() == {"status": "ok"}
    created = client.post("/api/sessions", json={"session_id": "one"}).json()
    assert created["session_id"] == "one"
    assert len(client.get("/api/sessions").json()) == 1
    reply = client.post("/api/sessions/one/messages", json={"text": "hello"}).json()
    assert reply["text"] == "reply: hello"
    with client.websocket_connect("/ws/sessions/one?after=0") as socket:
        assert socket.receive_json()["type"] == "session.created"
        assert socket.receive_json()["type"] == "message.user"


def test_web_approval_cannot_be_invented_by_client() -> None:
    service = AgentWebService(registry=FakeRegistry(), launcher=FakeLauncher())
    client = TestClient(create_app(service=service))
    client.post("/api/sessions", json={"session_id": "one"})

    result = client.post(
        "/api/sessions/one/approvals", json={"approval_id": "not-requested", "approved": True}
    )

    assert result.status_code == 404


def test_web_applies_user_and_server_pra_override() -> None:
    launcher = FakeLauncher()
    service = AgentWebService(
        registry=FakeRegistry(),
        launcher=launcher,
        default_profile="default",
        pra_override="ECONOMY",
    )
    client = TestClient(create_app(service=service))

    created = client.post(
        "/api/sessions",
        json={"session_id": "owned", "user_id": "alice"},
    ).json()

    assert created["user_id"] == "alice"
    assert service.profiles()["default_profile"] == "default"
    assert service.agents["owned"].config.user_id == "alice"
    assert launcher.last_profile.pra == "ECONOMY"


def test_web_applies_session_transport_override() -> None:
    launcher = FakeLauncher()
    service = AgentWebService(registry=FakeRegistry(), launcher=launcher)
    client = TestClient(create_app(service=service))

    response = client.post(
        "/api/sessions",
        json={
            "session_id": "native-required",
            "context_transport": "pra",
            "allow_text_fallback": False,
        },
    )

    assert response.status_code == 200
    assert launcher.last_profile.context_transport.value == "pra"
    assert launcher.last_profile.allow_text_fallback is False


def test_web_lifecycle_command_preserves_profile_and_pra_override(tmp_path) -> None:
    command = AgentWebLifecycle(tmp_path).command(
        host="127.0.0.1",
        port=8765,
        profile="work",
        pra_override="ECONOMY",
        config_path="agents.yaml",
    )

    assert command[command.index("--profile") + 1] == "work"
    assert command[command.index("--pra") + 1] == "ECONOMY"


def test_web_lifecycle_cleans_absent_state(tmp_path) -> None:
    lifecycle = AgentWebLifecycle(tmp_path)
    assert lifecycle.stop() == "NOT_RUNNING"
    assert not lifecycle.state_path.exists()
    assert not lifecycle.pid_path.exists()
