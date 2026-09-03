"""Multi-server MCP client and PRA capability adapters for the Agent SDK."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import re
import threading
import concurrent.futures
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .agent_config import MCPAgentConfig, MCPServerConfig
from .agent_resources import AgentResource, SideEffectClass, resource_uri
from .tool_records import ParameterSchema, ReturnSchema, ToolRecord, ToolSchema


class MCPConnectionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass
class MCPServerStatus:
    name: str
    state: MCPConnectionState = MCPConnectionState.DISCONNECTED
    error: str | None = None
    tool_count: int = 0
    resource_count: int = 0


@dataclass
class _Connection:
    name: str
    config: MCPServerConfig
    stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    session: Any = None
    tools: tuple[Any, ...] = ()
    resources: tuple[Any, ...] = ()


def _allowed(name: str, allow: Sequence[str], deny: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in allow) and not any(
        fnmatch.fnmatchcase(name, pattern) for pattern in deny
    )


class MCPClientManager:
    """Own MCP transports on one event loop and isolate per-server failures."""

    def __init__(self, config: MCPAgentConfig | None = None) -> None:
        self.config = config or MCPAgentConfig()
        self._connections: dict[str, _Connection] = {}
        self._status = {name: MCPServerStatus(name) for name in self.config.servers}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._queue: asyncio.Queue[Any] | None = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop and self._loop.is_running():
            return self._loop
        ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, args=(self._loop, ready), daemon=True, name="pra-mcp"
        )
        self._thread.start()
        ready.wait(5)
        return self._loop

    def _run_loop(self, loop: asyncio.AbstractEventLoop, ready: threading.Event) -> None:
        asyncio.set_event_loop(loop)
        self._queue = asyncio.Queue()
        loop.create_task(self._worker())
        ready.set()
        loop.run_forever()

    async def _worker(self) -> None:
        """Enter, use, and exit AnyIO transports from one owning task."""
        assert self._queue is not None
        while True:
            coroutine, future = await self._queue.get()
            if coroutine is None:
                future.set_result(None)
                return
            try:
                future.set_result(await coroutine)
            except BaseException as error:
                future.set_exception(error)

    async def _submit(self, coroutine: Any) -> Any:
        loop = self._ensure_loop()
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        assert self._queue is not None
        loop.call_soon_threadsafe(self._queue.put_nowait, (coroutine, future))
        return await asyncio.wrap_future(future)

    async def connect_all(self) -> tuple[MCPServerStatus, ...]:
        results = []
        for name, config in self.config.servers.items():
            if config.enabled:
                try:
                    results.append(await self.connect(name))
                except Exception:
                    if config.required:
                        raise
                    results.append(self._status[name])
        return tuple(results)

    async def connect(self, name: str) -> MCPServerStatus:
        config = self.config.servers[name]
        status = self._status.setdefault(name, MCPServerStatus(name))
        for attempt in range(config.retries + 1):
            status = await self._submit(self._connect(name))
            if status.state == MCPConnectionState.CONNECTED:
                return status
            if attempt < config.retries:
                await asyncio.sleep(config.backoff_seconds * (2 ** attempt))
        if config.required:
            raise RuntimeError(f"Required MCP server {name!r} failed: {status.error}")
        return status

    async def _connect(self, name: str) -> MCPServerStatus:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        config = self.config.servers[name]
        status = self._status.setdefault(name, MCPServerStatus(name))
        if status.state == MCPConnectionState.CONNECTED:
            return status
        status.state, status.error = MCPConnectionState.CONNECTING, None
        connection = _Connection(name, config)
        try:
            if config.transport == "stdio":
                if not config.command:
                    raise ValueError(f"MCP server {name!r} requires command for stdio transport.")
                streams = await connection.stack.enter_async_context(stdio_client(
                    StdioServerParameters(command=config.command, args=config.args, env=config.env or None)
                ))
                read, write = streams
            else:
                if not config.url:
                    raise ValueError(f"MCP server {name!r} requires url for HTTP transport.")
                import httpx
                certificate = (
                    (config.auth.cert_file, config.auth.key_file)
                    if config.auth.cert_file and config.auth.key_file
                    else config.auth.cert_file
                )
                client = await connection.stack.enter_async_context(httpx.AsyncClient(
                    headers=config.auth.resolved_headers(), timeout=config.timeout_seconds,
                    cert=certificate,
                ))
                read, write, _ = await connection.stack.enter_async_context(
                    streamable_http_client(config.url, http_client=client)
                )
            connection.session = await connection.stack.enter_async_context(
                ClientSession(read, write)
            )
            await connection.session.initialize()
            tools = await connection.session.list_tools()
            resources = await connection.session.list_resources()
            connection.tools = tuple(
                tool for tool in tools.tools
                if _allowed(tool.name, config.tool_allow, config.tool_deny)
            )
            connection.resources = tuple(
                item for item in resources.resources
                if _allowed(str(item.uri), config.resource_allow, config.resource_deny)
            )
            self._connections[name] = connection
            status.state = MCPConnectionState.CONNECTED
            status.tool_count, status.resource_count = len(connection.tools), len(connection.resources)
        except BaseException as error:
            try:
                await connection.stack.aclose()
            except BaseException:
                pass
            status.state = MCPConnectionState.FAILED if config.required else MCPConnectionState.DEGRADED
            status.error = f"{type(error).__name__}: {error}"
        return status

    async def disconnect(self, name: str) -> MCPServerStatus:
        return await self._submit(self._disconnect(name))

    async def _disconnect(self, name: str) -> MCPServerStatus:
        connection = self._connections.pop(name, None)
        if connection:
            await connection.stack.aclose()
        status = self._status.setdefault(name, MCPServerStatus(name))
        status.state, status.error = MCPConnectionState.DISCONNECTED, None
        status.tool_count = status.resource_count = 0
        return status

    async def disconnect_all(self) -> None:
        if not self._loop:
            return
        await self._submit(self._disconnect_all())
        assert self._loop is not None
        stopped: concurrent.futures.Future[Any] = concurrent.futures.Future()
        assert self._queue is not None
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (None, stopped))
        await asyncio.wrap_future(stopped)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._loop = self._thread = None

    async def _disconnect_all(self) -> None:
        for name in tuple(self._connections):
            await self._disconnect(name)

    async def list_servers(self) -> tuple[MCPServerStatus, ...]:
        return tuple(self._status[name] for name in sorted(self._status))

    async def server_status(self, name: str) -> MCPServerStatus:
        return self._status[name]

    async def list_tools(self, server: str | None = None) -> tuple[dict[str, Any], ...]:
        rows = []
        for name, connection in self._connections.items():
            if server and name != server:
                continue
            for tool in connection.tools:
                rows.append({
                    "server": name, "name": tool.name,
                    "runtime_name": f"mcp:{name}:{tool.name}",
                    "description": tool.description or "",
                    "input_schema": dict(tool.inputSchema or {}),
                    "annotations": tool.annotations.model_dump() if tool.annotations else {},
                })
        return tuple(rows)

    async def list_resources(self, server: str | None = None) -> tuple[dict[str, Any], ...]:
        rows = []
        for name, connection in self._connections.items():
            if server and name != server:
                continue
            for item in connection.resources:
                rows.append({"server": name, "uri": str(item.uri), "name": item.name,
                             "title": item.title, "mime_type": item.mimeType})
        return tuple(rows)

    async def call_tool(self, server: str, tool: str, arguments: Mapping[str, Any]) -> Any:
        return await self._submit(self._connections[server].session.call_tool(tool, dict(arguments)))

    async def read_resource(self, server: str, uri: str) -> Any:
        return await self._submit(self._connections[server].session.read_resource(uri))

    def call_tool_sync(self, server: str, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        loop = self._ensure_loop()
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        assert self._queue is not None
        loop.call_soon_threadsafe(
            self._queue.put_nowait,
            (self._connections[server].session.call_tool(tool, dict(arguments)), future),
        )
        result = future.result(timeout=self.config.servers[server].timeout_seconds)
        return result.model_dump(mode="json") if hasattr(result, "model_dump") else {"result": result}

    async def tool_records(self, tenant_id: str = "default") -> tuple[ToolRecord, ...]:
        return tuple(_tool_record(row, self.config.servers[row["server"]], tenant_id)
                     for row in await self.list_tools())

    async def resource_records(self, tenant_id: str = "default") -> tuple[AgentResource, ...]:
        return tuple(_resource_record(row, tenant_id) for row in await self.list_resources())

    def tool_handlers(self) -> dict[str, Any]:
        handlers = {}
        for name, connection in self._connections.items():
            namespace = connection.config.namespace or f"mcp-{name}"
            for tool in connection.tools:
                uri = resource_uri("tool", namespace, f"mcp:{name}:{tool.name}", "v1")
                handlers[uri] = lambda args, _prior, s=name, t=tool.name: self.call_tool_sync(s, t, args)
        return handlers


def _schema(value: Mapping[str, Any]) -> ToolSchema:
    properties = value.get("properties", {}) if isinstance(value, Mapping) else {}
    required = set(value.get("required", ())) if isinstance(value, Mapping) else set()
    inputs = tuple(ParameterSchema(
        str(name), str(spec.get("type", "unknown")), name in required,
        str(spec.get("description", "")), json_schema=dict(spec)
    ) for name, spec in properties.items() if isinstance(spec, Mapping))
    return ToolSchema(inputs, ReturnSchema(json_schema={"type": "object"}))


def _tool_record(row: Mapping[str, Any], config: MCPServerConfig, tenant_id: str) -> ToolRecord:
    annotations = {**row.get("annotations", {}), **config.annotations.get(str(row["name"]), {})}
    read_only = bool(annotations.get("readOnlyHint", annotations.get("read_only", False)))
    destructive = bool(annotations.get("destructiveHint", annotations.get("high_impact", False)))
    side_effect = "read" if read_only else "destructive" if destructive else "write"
    runtime_name = str(row["runtime_name"])
    display_name = re.sub(r"[^A-Za-z0-9_-]", "_", runtime_name)
    return ToolRecord(
        name=display_name, qualified_name=runtime_name, module=f"mcp.{row['server']}",
        description=str(row.get("description", "")), signature=json.dumps(row.get("input_schema", {}), sort_keys=True),
        schema=_schema(row.get("input_schema", {})), is_async=True, accepts_varargs=False,
        accepts_varkw=False, namespace=config.namespace or f"mcp-{row['server']}",
        tenant_id=tenant_id, version="v1", aliases=(runtime_name,),
        keywords=frozenset(str(row["name"]).replace("_", " ").split()),
        metadata={"mcp_server": row["server"], "mcp_tool_name": row["name"],
                  "transport": config.transport, "annotations": annotations,
                  "runtime_identity": runtime_name, "display_name": display_name,
                  "side_effect_class": side_effect, "trust": "external_mcp"},
    )


def _resource_record(row: Mapping[str, Any], tenant_id: str) -> AgentResource:
    return AgentResource(
        uri=resource_uri("mcp-resource", f"mcp-{row['server']}", str(row["uri"]), "v1"),
        kind="mcp-resource", namespace=f"mcp-{row['server']}", name=str(row.get("name") or row["uri"]),
        version="v1", description=str(row.get("title") or row.get("name") or row["uri"]),
        tenant_id=tenant_id, metadata={"source_server": row["server"], "original_uri": row["uri"],
                                       "mime_type": row.get("mime_type")},
    )
