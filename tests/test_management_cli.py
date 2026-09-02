from __future__ import annotations

import json
import importlib
import socket
import time

import pytest
from click.testing import CliRunner

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from pra_hf.management import (
    ManagementAPIConfig,
    ManagementProvider,
    start_management_api,
    stop_management_api,
)
from pra_hf.management_cli import ManagementClient, _registered_token, engine_cli
from pra_hf.runtime_providers import HFRuntimeProvider, RuntimeConfig
from pra_hf.deployment import PRAEngineCapabilities


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait(client: ManagementClient) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if client.get("/v1/pra/health")["status"] == "healthy":
                return
        except Exception:
            time.sleep(0.05)
    raise AssertionError("management server did not become ready")


def test_remote_engine_cli_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PRA_HOME", str(tmp_path / ".pra"))
    port = _free_port()
    provider = ManagementProvider(
        engine="test-engine",
        capabilities={"text_fallback": True},
        config_patch_handler=lambda values: values,
        action_handlers={"maintenance": lambda _request: {"maintained": True}},
    )
    settings = ManagementAPIConfig(enabled=True, port=port)
    server = start_management_api(provider, settings)
    url = f"http://127.0.0.1:{port}"
    _wait(ManagementClient(url))
    runner = CliRunner()
    try:
        connected = runner.invoke(engine_cli, ["connect", url, "--name", "lab", "--json"])
        assert connected.exit_code == 0, connected.output
        assert json.loads(connected.output)["stored_secret"] is False

        for command in (
            "health", "config", "storage", "sessions", "resources", "models",
            "profiles", "capabilities", "audit",
            "registry-status",
        ):
            result = runner.invoke(engine_cli, [command, "lab", "--json"])
            assert result.exit_code == 0, (command, result.output)

        inspected = runner.invoke(
            engine_cli, ["inspect", "--management-url", url, "--json"]
        )
        assert inspected.exit_code == 0, inspected.output
        assert json.loads(inspected.output)["info"]["engine"] == "test-engine"

        patch = tmp_path / "patch.yaml"
        patch.write_text("profile: ECONOMY\n", encoding="utf-8")
        changed = runner.invoke(
            engine_cli, ["patch-config", "lab", "--patch", str(patch), "--json"]
        )
        assert changed.exit_code == 0, changed.output
        assert json.loads(changed.output)["effective"]["profile"] == "ECONOMY"

        action = runner.invoke(
            engine_cli,
            ["action", "maintenance", "lab", "--idempotency-key", "m1", "--json"],
        )
        assert action.exit_code == 0, action.output
        assert json.loads(action.output)["detail"]["maintained"] is True
    finally:
        stop_management_api(server)


def test_hf_managed_runtime_forwards_management_listener_options() -> None:
    command = HFRuntimeProvider().build_command(RuntimeConfig(
        engine="hf",
        model="org/model",
        management_api=True,
        management_host="127.0.0.2",
        management_port=9191,
        management_auth_mode="static_bearer",
        management_token_env="PRA_TEST_TOKEN",
        management_metrics_url="http://metrics",
        management_trace_url="http://tempo",
        management_grafana_url="http://grafana",
    ))
    assert "--management-api" in command
    assert command[command.index("--management-port") + 1] == "9191"
    assert command[command.index("--management-auth-mode") + 1] == "static_bearer"
    assert command[command.index("--management-grafana-url") + 1] == "http://grafana"


def test_management_serve_advertises_external_registry_address(monkeypatch) -> None:
    module = importlib.import_module("pra_hf.management_cli")
    captured = {}

    def serve(provider, settings):
        captured["provider"] = provider
        captured["settings"] = settings

    monkeypatch.setattr("pra_hf.management.serve_management_api", serve)
    result = CliRunner().invoke(module.engine_cli, [
        "serve",
        "--engine", "vllm",
        "--host", "0.0.0.0",
        "--port", "9101",
        "--inference-url", "http://192.168.1.86:8000",
        "--registry-url", "http://192.168.1.102:9200",
        "--registry-instance-id", "nvidia-vllm-cuda",
        "--registry-instance-host", "192.168.1.86",
        "--registry-management-url", "http://192.168.1.86:9101",
    ])

    assert result.exit_code == 0, result.output
    identity = captured["settings"].registry.instance
    assert identity.host == "192.168.1.86"
    assert identity.management_url == "http://192.168.1.86:9101"
    payload = captured["provider"].registry_payload(
        captured["settings"], "nvidia-vllm-cuda"
    )
    assert payload["host"] == "192.168.1.86"
    assert payload["management_url"] == "http://192.168.1.86:9101"


def test_one_off_management_url_never_inherits_saved_connection_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PRA_HOME", str(tmp_path / ".pra"))
    monkeypatch.setenv("SAVED_ENGINE_TOKEN", "must-not-leak")
    registry = tmp_path / ".pra" / "engines" / "connections.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({
        "version": 1,
        "default": "production",
        "connections": {
            "production": {
                "url": "https://trusted.example",
                "token_env": "SAVED_ENGINE_TOKEN",
            }
        },
    }), encoding="utf-8")
    assert _registered_token(None, None) == "must-not-leak"
    assert _registered_token(None, None, "https://other.example") is None
    assert _registered_token("https://other.example", None) is None


def test_gateway_explicitly_attaches_live_management_provider(monkeypatch) -> None:
    module = importlib.import_module("pra_hf.gateway_cli")
    captured = {}

    class Adapter:
        storage_manager = SimpleStorage = object()

        def __init__(self, *_args, **_kwargs):
            pass

        def capabilities(self):
            return PRAEngineCapabilities(adapter="openai", engine_type="openai_generic")

    def start(provider, settings):
        captured["provider"] = provider
        captured["settings"] = settings
        return object()

    monkeypatch.setattr(module, "OpenAICompatibleEngineAdapter", Adapter)
    monkeypatch.setattr(module, "serve_gateway", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("pra_hf.gateway_management.start_gateway_management_api", start)
    monkeypatch.setattr("pra_hf.gateway_management.stop_gateway_management_api", lambda server: captured.setdefault("stopped", server))

    result = CliRunner().invoke(module.gateway_cli, [
        "serve", "--backend-url", "http://127.0.0.1:9999",
        "--management-api", "--management-port", "9192",
    ])
    assert result.exit_code == 0, result.output
    assert captured["settings"].enabled is True
    assert captured["settings"].port == 9192
    assert captured["provider"].gateway.sessions is not None
    assert captured["provider"].upstreams.row("default").base_url == "http://127.0.0.1:9999"
    assert "Gateway Management API: http://127.0.0.1:9192/v1/pra/gateway/info" in result.output


def test_gateway_does_not_initialize_management_when_disabled(monkeypatch) -> None:
    module = importlib.import_module("pra_hf.gateway_cli")

    class Adapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def capabilities(self):
            return PRAEngineCapabilities(adapter="openai", engine_type="openai_generic")

    monkeypatch.setattr(module, "OpenAICompatibleEngineAdapter", Adapter)
    monkeypatch.setattr(module, "serve_gateway", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "pra_hf.gateway_management.start_gateway_management_api",
        lambda *_args, **_kwargs: pytest.fail("disabled management must not initialize"),
    )
    result = CliRunner().invoke(module.gateway_cli, [
        "serve", "--backend-url", "http://127.0.0.1:9999",
    ])
    assert result.exit_code == 0, result.output
    assert "Gateway Management API:" not in result.output
