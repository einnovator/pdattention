"""Composable terminal client for PRA Agent sessions and remote services."""

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import click

from .agent import PRAAgent
from .agent_config import MCPServerConfig, PRAAgentSettings
from .agent_execution import ToolCall
from .agent_resources import AgentResource


@dataclass(frozen=True)
class CommandSpec:
    """One command used by dispatch, completion, help, and generated docs."""

    name: str
    description: str
    handler: Callable[[list[str]], bool | None]
    aliases: tuple[str, ...] = ()
    subcommands: tuple[str, ...] = ()
    sensitive: bool = False


class CommandRegistry:
    """Declarative slash-command index with nested completion."""

    def __init__(self) -> None:
        self._commands: dict[str, CommandSpec] = {}

    def register(self, spec: CommandSpec) -> None:
        for name in (spec.name, *spec.aliases):
            key = name.lstrip("/").casefold()
            if key in self._commands:
                raise ValueError(f"Duplicate slash command: {name}")
            self._commands[key] = spec

    def resolve(self, name: str) -> CommandSpec | None:
        return self._commands.get(name.lstrip("/").casefold())

    def commands(self) -> tuple[CommandSpec, ...]:
        return tuple(sorted(set(self._commands.values()), key=lambda row: row.name))

    def complete(self, text: str) -> tuple[str, ...]:
        raw = text.lstrip()
        if not raw.startswith("/"):
            return ()
        parts, trailing = raw[1:].split(), raw.endswith(" ")
        if len(parts) <= 1 and not trailing:
            prefix = parts[0].casefold() if parts else ""
            return tuple(f"/{row.name}" for row in self.commands() if row.name.startswith(prefix))
        spec = self.resolve(parts[0]) if parts else None
        prefix = "" if trailing else parts[-1].casefold()
        return tuple(value for value in (spec.subcommands if spec else ()) if value.startswith(prefix))

    def markdown(self) -> str:
        lines = ["# PRA Agent Slash Commands", "", "Generated from `CommandRegistry`.", ""]
        for row in self.commands():
            lines.extend((f"## `/{row.name}`", "", row.description, ""))
            if row.aliases:
                lines.extend((f"Aliases: {', '.join(f'`/{name}`' for name in row.aliases)}", ""))
            if row.subcommands:
                lines.extend((f"Subcommands: {', '.join(f'`{name}`' for name in row.subcommands)}", ""))
        return "\n".join(lines)


@dataclass
class AgentShell:
    """Interactive shell assembled from independent Agent SDK services."""

    agent: PRAAgent
    output: Callable[[str], None] = click.echo
    registry: CommandRegistry = field(init=False)
    last_output: str = ""

    def __post_init__(self) -> None:
        self.registry = CommandRegistry()
        definitions = (
            ("help", "Show available commands.", self._help, (), ()),
            ("history", "Inspect or clear input history.", self._history, (), ("search", "clear")),
            ("clear", "Clear the terminal.", self._clear, (), ()),
            ("session", "Inspect or export the transcript.", self._session, (), ("export",)),
            ("sessions", "List retained sessions for the current user.", self._sessions, (), ()),
            ("new", "Start a new durable session.", self._new, (), ()),
            ("tips", "Show common interactive workflows.", self._tips, (), ()),
            ("model", "Inspect or switch inference target.", self._model, (), ("use",)),
            ("models", "List static and discovered models.", self._models, (), ("all", "engine")),
            ("engine", "Inspect or select an engine.", self._engine, ("runtime",), ("use",)),
            ("mcp", "Manage MCP servers, tools, and resources.", self._mcp, (), ("list", "add", "remove", "connect", "disconnect", "tools", "resources", "attach", "status")),
            ("tools", "List local and MCP tools.", self._tools, (), ()),
            ("resources", "List session and MCP resources.", self._resources, (), ()),
            ("attach", "Attach a local file.", self._attach, (), ()),
            ("attachments", "List active attachments.", self._attachments, (), ()),
            ("detach", "Detach a session resource.", self._detach, (), ()),
            ("context", "Inspect logical PRA context.", self._context, (), ()),
            ("status", "Show runtime and service status.", self._status, (), ()),
            ("config", "Inspect, edit, reload, or save configuration.", self._config, (), ("show", "get", "set", "reload", "save")),
            ("save", "Explicitly persist configuration.", self._save, (), ()),
            ("tasks", "List tasks.", self._tasks, (), ()),
            ("task", "Create, activate, or complete a task.", self._task, (), ("new", "use", "done")),
            ("skills", "List progressively disclosed skills.", self._skills, (), ()),
            ("metrics", "Inspect runtime metrics.", self._metrics, (), ()),
            ("pra", "Inspect PRA mode/profile/accounting.", self._pra, (), ("mode", "profile", "stats")),
            ("search", "Search the transcript.", self._search, (), ()),
            ("last", "Show the last result.", self._last, (), ()),
            ("block", "Enter an explicit multiline block.", self._block, (), ()),
            ("quit", "Exit and retain the durable session.", self._quit, ("exit",), ()),
        )
        for name, description, handler, aliases, children in definitions:
            self.registry.register(CommandSpec(name, description, handler, aliases, children))

    def emit(self, value: object) -> None:
        self.last_output = str(value)
        self.output(self.last_output)

    def authorize(self, resource: AgentResource, call: ToolCall) -> bool:
        return click.confirm(
            f"Authorize {resource.side_effect_class.value} tool {resource.name}({dict(call.arguments)!r})?",
            default=False,
        )

    def status(self) -> str:
        state = self.agent.state
        mcp = tuple(self.agent.mcp._status.values())
        connected = sum(row.state.value == "CONNECTED" for row in mcp)
        degraded = sum(row.state.value in {"DEGRADED", "FAILED"} for row in mcp)
        target = self.agent.targets.active
        cp = self.agent.control_plane.status if self.agent.control_plane else "disabled"
        return (f"session={state.session_id} task={state.active_task_id or '-'} records={len(state.records)} "
                f"target={target.target_id if target else '-'} mcp={connected} connected/{degraded} degraded "
                f"attachments={len(self.agent.attachments.list())} control_plane={cp}")

    def dispatch(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as error:
            self.emit(f"Invalid command: {error}")
            return True
        spec = self.registry.resolve(parts[0]) if parts else None
        if not spec:
            self.emit("Unknown command. Use /help.")
            return True
        self.agent.history.add(line, sensitive=spec.sensitive)
        try:
            result = spec.handler(parts[1:])
            return True if result is None else result
        except Exception as error:
            self.emit(f"{type(error).__name__}: {error}")
            return True

    def _help(self, _: list[str]) -> None:
        self.emit("\n".join(f"/{row.name:<12} {row.description}" for row in self.registry.commands()))

    def _history(self, args: list[str]) -> None:
        if args[:1] == ["clear"]:
            self.agent.history.clear(); self.emit("Input history cleared."); return
        rows = self.agent.history.search(" ".join(args[1:])) if args[:1] == ["search"] else tuple(self.agent.history.entries)
        if args and args[0].isdigit(): rows = rows[-int(args[0]):]
        self.emit("\n".join(rows) or "No input history.")

    def _clear(self, _: list[str]) -> None: click.clear()
    def _quit(self, _: list[str]) -> bool: return False

    def _session(self, args: list[str]) -> None:
        if args[:1] == ["export"]:
            target = args[1] if len(args) > 1 else f"{self.agent.state.session_id}.md"
            self.emit(f"Exported {self.agent.export_session(target)}")
        else: self.emit(self.status())

    def _new(self, _: list[str]) -> None:
        if self.agent.session is not None: self.agent.runtime.close_session(self.agent.session)
        self.emit(f"New session: {self.agent.start_session().session_id}")

    def _sessions(self, _: list[str]) -> None:
        rows = self.agent.sessions.list_sessions(self.agent.config.user_id)
        self.emit("\n".join(
            f"{row.session_id}\t{row.updated_at}\trecords={len(row.records)}"
            for row in rows
        ) or "No retained sessions.")

    def _tips(self, _: list[str]) -> None:
        self.emit(
            "Use /models before /model use <target>; /sessions and /new manage durable "
            "work; /status shows the active target; /help lists every command."
        )

    def _models(self, args: list[str]) -> None:
        rows = asyncio.run(self.agent.targets.list(refresh=True))
        if len(args) >= 2 and args[0] == "engine": rows = tuple(row for row in rows if row.engine_instance == args[1])
        self.emit("\n".join(f"{row.target_id}\t{row.status}\t{row.qualification or '-'}\t{row.model_id}" for row in rows) or "No targets.")

    def _model(self, args: list[str]) -> None:
        if not args: self.emit(self.agent.targets.active.target_id if self.agent.targets.active else "No active target."); return
        value = args[1] if args[0] == "use" and len(args) > 1 else args[0]
        target = asyncio.run(self.agent.switch_target(value))
        self.emit(f"Model switched -> {target.target_id}; native state invalidated.")

    def _engine(self, args: list[str]) -> None:
        if args[:1] == ["use"] and len(args) > 1:
            rows = asyncio.run(self.agent.targets.list()); matches = [row for row in rows if row.engine_instance == args[1]]
            if len(matches) != 1: raise ValueError(f"Engine has {len(matches)} models; use /model use <target>.")
            self._model([matches[0].target_id]); return
        self._models(["engine", args[0]]) if args else self._models([])

    def _mcp(self, args: list[str]) -> None:
        action = args[0] if args else "status"
        if action in {"status", "list"}:
            rows = asyncio.run(self.agent.mcp.list_servers())
            self.emit("\n".join(f"{row.name}\t{row.state.value}\ttools={row.tool_count}\tresources={row.resource_count}\t{row.error or ''}" for row in rows) or "No MCP servers.")
        elif action == "connect": self.emit(asyncio.run(self.agent.mcp.connect(args[1])))
        elif action == "disconnect": self.emit(asyncio.run(self.agent.mcp.disconnect(args[1])))
        elif action == "tools": self.emit(json.dumps(asyncio.run(self.agent.mcp.list_tools(args[1] if len(args)>1 else None)), indent=2))
        elif action == "resources": self.emit(json.dumps(asyncio.run(self.agent.mcp.list_resources(args[1] if len(args)>1 else None)), indent=2))
        elif action == "attach":
            row = asyncio.run(self.agent.attach_mcp_resource(args[1], args[2]))
            self.emit(f"Attached MCP resource #{row.attachment_id} -> {row.uri}")
        elif action == "add":
            self.agent.settings.mcp.servers[args[1]] = MCPServerConfig(url=args[2]); self.agent.mcp.config = self.agent.settings.mcp
            self.emit(f"Added MCP server {args[1]} in memory. Use /save to persist.")
        elif action == "remove":
            self.agent.settings.mcp.servers.pop(args[1]); self.agent.mcp.config = self.agent.settings.mcp
            self.emit(f"Removed MCP server {args[1]} in memory. Use /save to persist.")
        else: raise ValueError("Use /mcp status|add|remove|connect|disconnect|tools|resources")

    def _tools(self, _: list[str]) -> None:
        rows = self.agent.runtime.capability_resources(kinds=("tool",)) if self.agent.runtime.capabilities else ()
        self.emit("\n".join(f"{row.name}\t{row.side_effect_class.value}\t{row.metadata.get('mcp_server', 'local')}" for row in rows) or "No tools.")

    def _resources(self, _: list[str]) -> None:
        local = [f"#{row.attachment_id}\t{row.name}\t{row.mime_type}\t{row.size_bytes} bytes" for row in self.agent.attachments.list()]
        remote = [f"{row['server']}\t{row['uri']}" for row in asyncio.run(self.agent.mcp.list_resources())]
        self.emit("\n".join((*local, *remote)) or "No resources.")

    def _attach(self, args: list[str]) -> None:
        row = self.agent.attachments.add(" ".join(args)); self.emit(f"Attached #{row.attachment_id} -> {row.name}")
    def _attachments(self, _: list[str]) -> None:
        rows = self.agent.attachments.list(); self.emit("\n".join(f"#{row.attachment_id} {row.name} {row.mime_type} {row.size_bytes} bytes ACTIVE" for row in rows) or "No attachments.")
    def _detach(self, args: list[str]) -> None: self.agent.attachments.detach(args[0].lstrip("#")); self.emit(f"Detached {args[0]}")

    def _context(self, _: list[str]) -> None:
        state = self.agent.state; counts: dict[str, int] = {}
        for row in state.records: counts[row.record_type.value] = counts.get(row.record_type.value, 0) + 1
        self.emit(json.dumps({"session": state.session_id, "records": len(state.records), "types": counts,
                              "attachments": len(self.agent.attachments.list()),
                              "target": self.agent.targets.active.target_id if self.agent.targets.active else None}, indent=2))
    def _status(self, _: list[str]) -> None: self.emit(self.status())

    def _config(self, args: list[str]) -> None:
        action = args[0] if args else "show"
        if action == "show": self.emit(json.dumps(self.agent.settings.redacted(), indent=2))
        elif action == "get":
            value: object = self.agent.settings.redacted()
            for name in args[1].split("."): value = value[name]  # type: ignore[index]
            self.emit(json.dumps(value, indent=2))
        elif action == "set":
            payload = self.agent.settings.model_dump(exclude={"source_file"}); node = payload
            names = args[1].split(".")
            for name in names[:-1]: node = node[name]
            try: node[names[-1]] = json.loads(" ".join(args[2:]))
            except json.JSONDecodeError: node[names[-1]] = " ".join(args[2:])
            self.agent.settings = PRAAgentSettings.model_validate(payload); self.emit(f"Set {args[1]} in memory.")
        elif action == "reload":
            if not self.agent.settings.source_file: raise ValueError("No source config file.")
            self.agent.settings = PRAAgentSettings.from_file(self.agent.settings.source_file); self.emit("Configuration reloaded.")
        elif action == "save": self._save(args[1:])
        else: raise ValueError("Use /config show|get|set|reload|save")

    def _save(self, args: list[str]) -> None: self.emit(f"Saved {self.agent.settings.save(args[0] if args else None)}")
    def _tasks(self, _: list[str]) -> None: self.emit("\n".join(f"{row.task_id}\t{row.status.value}\t{row.description}" for row in self.agent.state.tasks.tasks) or "No tasks.")
    def _task(self, args: list[str]) -> None:
        if args[0] == "new": self.emit(f"Active task: {self.agent.create_task(' '.join(args[1:])).active_task_id}")
        elif args[0] == "use": self.agent.activate_task(args[1]); self.emit(f"Active task: {args[1]}")
        elif args[0] == "done": self.agent.complete_task(args[1]); self.emit(f"Completed task: {args[1]}")
    def _skills(self, _: list[str]) -> None:
        rows = self.agent.runtime.capability_resources(kinds=("skill",)) if self.agent.runtime.capabilities else (); self.emit("\n".join(f"{row.name}\t{row.uri}" for row in rows) or "No skills.")
    def _metrics(self, _: list[str]) -> None: self.emit(json.dumps(self.agent.runtime.inspect(), indent=2, default=str))
    def _pra(self, args: list[str]) -> None: self._metrics(args)
    def _search(self, args: list[str]) -> None:
        needle = " ".join(args).casefold(); rows = [row for row in self.agent.state.records if needle in str(row.payload).casefold()]
        self.emit("\n".join(f"{row.record_id}\t{row.record_type.value}" for row in rows) or "No matches.")
    def _last(self, _: list[str]) -> None: self.output(self.last_output)

    def _block(self, _: list[str]) -> None:
        self.emit("Enter text; finish with a line containing only '.'."); lines = []
        while True:
            line = click.prompt("...", prompt_suffix="> ")
            if line == ".": break
            lines.append(line)
        self._run_message("\n".join(lines))

    def _run_message(self, line: str) -> None:
        paste = self.agent.settings.tui.paste
        if len(line) >= paste.block_threshold_chars or len(line.splitlines()) >= paste.block_threshold_lines:
            row = self.agent.attachments.add_paste(line)
            self.emit(f"[pasted block #{row.attachment_id} · {row.line_count or 0} lines · {row.size_bytes} bytes]")
        self.agent.history.add(line)
        self.emit(f"assistant> {self.agent.run_turn(line).text}")

    def run(self) -> None:
        """Use prompt-toolkit editing when installed, with a Click fallback."""

        self.agent.authorization_callback = self.authorize
        self.output("PRA Agent"); self.output(self.status()); self.output("Use /help for commands.")
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import Completer, Completion
            from prompt_toolkit.history import FileHistory
            shell = self
            class SlashCompleter(Completer):
                def get_completions(self, document, complete_event):
                    word = document.get_word_before_cursor()
                    for value in shell.registry.complete(document.text_before_cursor):
                        yield Completion(value, start_position=-len(word))
            options = {"completer": SlashCompleter(), "multiline": False}
            if self.agent.history.path: options["history"] = FileHistory(str(self.agent.history.path))
            session = PromptSession(**options)
            read = lambda: session.prompt("you> ")
        except ImportError:
            read = lambda: click.prompt("you", prompt_suffix="> ")
        while True:
            try: line = read()
            except EOFError: break
            except KeyboardInterrupt:
                self.output("Cancelled current input. Press Ctrl+D or /quit to exit."); continue
            if line.startswith("/"):
                if not self.dispatch(line): break
            else:
                try: self._run_message(line)
                except KeyboardInterrupt: self.output("Cancelled current turn; session retained.")


def write_command_reference(path: str | Path, registry: CommandRegistry) -> Path:
    """Generate slash-command documentation from the live registry."""

    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(registry.markdown(), encoding="utf-8")
    return target
