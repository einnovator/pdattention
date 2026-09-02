from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from pra_control.app import ControlRuntime, create_app
from pra_control.config import (
    ControlAuthProfile,
    ControlPlaneConfig,
    MCPHTTPTransportConfig,
    MCPPresentationConfig,
    MCPStdioTransportConfig,
    MCPTransportsConfig,
    ServiceLinkConfig,
)
from pra_control.rbac import Role


@pytest.mark.e2e
def test_official_mcp_client_discovers_and_calls_stdio(tmp_path, monkeypatch):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "mcp-test-cookie-secret")
    config_path = tmp_path / "stdio-control.yaml"
    config_path.write_text(
        "control_plane:\n"
        f"  database_url: sqlite:///{(tmp_path / 'stdio.db').as_posix()}\n"
        "  registry:\n    url: null\n"
        "  mcp:\n"
        "    enabled: true\n"
        "    transports:\n"
        "      stdio:\n        enabled: true\n        auth_profile: local-agent\n"
        "  auth_profiles:\n"
        "    local-agent:\n"
        "      type: service_identity\n"
        "      subject: test-coding-agent\n"
        "      roles: [Viewer]\n",
        encoding="utf-8",
    )

    async def exercise() -> None:
        environment = dict(os.environ)
        root = str(Path(__file__).parents[1] / "src")
        environment["PYTHONPATH"] = root + os.pathsep + environment.get("PYTHONPATH", "")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pra_control.cli", "mcp", "--config", str(config_path), "--transport", "stdio"],
            env=environment,
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert "pra_fleet" in {tool.name for tool in tools.tools}
                assert "pra_apply" not in {tool.name for tool in tools.tools}
                result = await session.call_tool("pra_fleet", {})
                payload = result.structuredContent or json.loads(result.content[0].text)
                assert payload["ok"] is True
                assert payload["result"]["summary"]["total"] == 0

    import asyncio
    asyncio.run(exercise())


@pytest.mark.e2e
def test_official_mcp_client_calls_authenticated_streamable_http(tmp_path, monkeypatch):
    monkeypatch.setenv("PRA_CONTROL_COOKIE_SECRET", "mcp-test-cookie-secret")
    monkeypatch.setenv("PRA_TEST_MCP_TOKEN", "remote-test-token")
    config = ControlPlaneConfig(
        database_url=f"sqlite:///{tmp_path / 'http.db'}",
        registry=ServiceLinkConfig(url=None),
        mcp=MCPPresentationConfig(
            enabled=True,
            transports=MCPTransportsConfig(
                http=MCPHTTPTransportConfig(enabled=True, path="/mcp", auth_profile="remote-agent"),
                stdio=MCPStdioTransportConfig(enabled=False),
            ),
        ),
        auth_profiles={
            "remote-agent": ControlAuthProfile(
                type="bearer_token", subject="remote-test", roles=[Role.VIEWER], token_env="PRA_TEST_MCP_TOKEN",
            ),
        },
    )
    config.validate_security()
    runtime = ControlRuntime(config)
    app = create_app(config, runtime=runtime)

    async def exercise() -> None:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://localhost:9300") as anonymous:
                denied = await anonymous.post("/mcp/", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
                assert denied.status_code == 401
            async with httpx.AsyncClient(
                transport=transport, base_url="http://localhost:9300",
                headers={"Authorization": "Bearer remote-test-token", "X-Request-ID": "mcp-http-1"},
            ) as client:
                async with streamable_http_client("http://localhost:9300/mcp/", http_client=client) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        assert "pra_context" in {tool.name for tool in tools.tools}
                        result = await session.call_tool("pra_fleet", {})
                        payload = result.structuredContent or json.loads(result.content[0].text)
                        assert payload["ok"] is True

    import asyncio
    asyncio.run(exercise())
