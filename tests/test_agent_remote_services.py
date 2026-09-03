import asyncio
import json
import sys

import pytest
from click.testing import CliRunner

from pra_hf.agent_config import (
    ControlPlaneClientConfig,
    MCPAgentConfig,
    MCPServerConfig,
    PRAAgentSettings,
    ProviderConfig,
)
from pra_hf.agent_control_plane import (
    AmbiguousTargetError,
    ControlPlaneClient,
    InferenceTargetManager,
)
from pra_hf.agent_mcp import MCPClientManager, MCPConnectionState
from pra_hf.cli import cli, _resolve_agent_profile


def test_agent_config_precedence_env_and_explicit_save(tmp_path, monkeypatch):
    config = tmp_path / "agent.yaml"
    config.write_text("agent:\n  model: file-model\n  context_records: 9\n", encoding="utf-8")
    monkeypatch.setenv("PRA_AGENT_MODEL", "env-model")
    settings = PRAAgentSettings.compose(
        config_file=config,
        config={"agent": {"model": "object-model"}},
        overrides={"agent": {"model": "explicit-model"}},
    )
    assert settings.agent.model == "explicit-model"
    assert settings.agent.context_records == 9
    settings.save()
    assert "explicit-model" in config.read_text(encoding="utf-8")


def test_application_config_resolves_through_existing_cli_profile_boundary(tmp_path):
    config = tmp_path / "agent.yaml"
    config.write_text("""
agent:
  model: model-a
  provider: remote
providers:
  remote:
    type: openai
    base_url: http://runtime/v1
    model: model-a
""", encoding="utf-8")
    profile, sources = _resolve_agent_profile(None, config, None, None, None, None, None, (), None, None)
    assert profile.runtime.endpoint == "http://runtime/v1"
    assert profile.application_settings.source_file == str(config.resolve())
    assert sources == (f"application:{config}",)
    result = CliRunner().invoke(cli, ["agent", "inspect", "--config", str(config), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["profile"]["application_settings"]["agent"]["model"] == "model-a"


def test_agent_mcp_cli_requires_explicit_save(tmp_path):
    path = tmp_path / "agent.yaml"
    runner = CliRunner()
    dry = runner.invoke(cli, ["agent", "mcp", "add", "docs", "http://mcp", "--config", str(path)])
    assert dry.exit_code == 0 and not path.exists()
    saved = runner.invoke(cli, ["agent", "mcp", "add", "docs", "http://mcp", "--config", str(path), "--save"])
    assert saved.exit_code == 0 and "docs:" in path.read_text(encoding="utf-8")


class _FakeControlPlane(ControlPlaneClient):
    async def list_engines(self, *, refresh=False):
        self.status = "CONNECTED"
        return ({"name": "mac", "engine": "mlx", "status": "READY", "models": [
            {"runtime_model_id": "qwen", "model_id": "Qwen/Qwen3-4B", "endpoint": "http://mac"},
            {"runtime_model_id": "gemma", "model_id": "google/gemma-3-4b", "endpoint": "http://mac"},
        ]},)


def test_static_and_control_plane_target_merge_and_ambiguity():
    client = _FakeControlPlane(ControlPlaneClientConfig(enabled=True, url="http://control"))
    targets = InferenceTargetManager({"local": ProviderConfig(
        model="Qwen/Qwen3-4B", engine_instance="local", runtime_model_id="qwen",
        base_url="http://local",
    )}, client)
    rows = asyncio.run(targets.list())
    assert {row.target_id for row in rows} == {"local/qwen", "mac/qwen", "mac/gemma"}
    assert asyncio.run(targets.resolve("gemma")).target_id == "mac/gemma"
    with pytest.raises(AmbiguousTargetError):
        asyncio.run(targets.resolve("Qwen/Qwen3-4B"))


def test_optional_mcp_failure_degrades_without_aborting():
    manager = MCPClientManager(MCPAgentConfig(servers={
        "offline": MCPServerConfig(url="http://127.0.0.1:1/mcp", timeout_seconds=0.2, retries=0),
    }))
    statuses = asyncio.run(manager.connect_all())
    assert statuses[0].state == MCPConnectionState.DEGRADED
    asyncio.run(manager.disconnect_all())


def test_required_mcp_failure_aborts():
    manager = MCPClientManager(MCPAgentConfig(servers={
        "required": MCPServerConfig(url="http://127.0.0.1:1/mcp", timeout_seconds=0.2, retries=0, required=True),
    }))
    with pytest.raises(Exception):
        asyncio.run(manager.connect_all())
    asyncio.run(manager.disconnect_all())


def test_stdio_mcp_tool_resource_and_namespaced_call():
    source = '''
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("fixture")
@mcp.tool()
def echo(value: str) -> str:
    return value
@mcp.resource("memo://one")
def memo() -> str:
    return "remember me"
mcp.run(transport="stdio")
'''
    manager = MCPClientManager(MCPAgentConfig(servers={
        "fixture": MCPServerConfig(transport="stdio", command=sys.executable, args=["-c", source]),
    }))
    statuses = asyncio.run(manager.connect_all())
    assert statuses[0].state == MCPConnectionState.CONNECTED
    tools = asyncio.run(manager.list_tools())
    assert tools[0]["runtime_name"] == "mcp:fixture:echo"
    result = asyncio.run(manager.call_tool("fixture", "echo", {"value": "hello"}))
    assert "hello" in json.dumps(result.model_dump(mode="json"))
    resources = asyncio.run(manager.list_resources())
    assert resources[0]["uri"] == "memo://one"
    asyncio.run(manager.disconnect_all())
