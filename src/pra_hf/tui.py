"""Small coding-agent terminal UI for the PRA SDK."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Callable

import click

from .agent import PRAAgent
from .agent_execution import ToolCall
from .agent_resources import AgentResource


@dataclass
class AgentShell:
    """Stateful slash-command dispatcher used by the interactive CLI and tests."""

    agent: PRAAgent
    output: Callable[[str], None] = click.echo

    def authorize(self, resource: AgentResource, call: ToolCall) -> bool:
        """Ask for one dangerous call without granting future calls."""

        level = resource.side_effect_class.value
        rendered = f"{resource.name}({dict(call.arguments)!r})"
        return click.confirm(f"Authorize {level} tool {rendered}?", default=False)

    def status(self) -> str:
        state = self.agent.state
        return (
            f"session={state.session_id} version={state.version} "
            f"task={state.active_task_id or '-'} records={len(state.records)}"
        )

    def dispatch(self, line: str) -> bool:
        """Handle one slash command; return false when the shell should exit."""

        parts = shlex.split(line)
        command = parts[0].lower() if parts else ""
        if command in {"/exit", "/quit"}:
            return False
        if command == "/help":
            self.output(
                "/status  /profile  /model  /runtime  /session  /sessions  /tasks  "
                "/task new <text>  /task use <id>  /task done <id>  /context  "
                "/tools  /skills  /metrics  /clear  /exit"
            )
        elif command == "/status":
            self.output(self.status())
        elif command == "/sessions":
            service = self.agent.runtime.session_service
            rows = service.list_sessions(self.agent.config.user_id) if service else ()
            self.output("\n".join(f"{row.session_id}\t{row.active_task_id or '-'}\t{len(row.records)} records" for row in rows) or "No sessions.")
        elif command in {"/profile", "/model", "/runtime"}:
            summary = getattr(self.agent, "product_summary", {})
            key = command[1:]
            if key == "profile":
                self.output(f"agent={summary.get('agent_profile', '-')} pra={summary.get('pra_profile', '-')}")
            elif key == "model":
                self.output(f"model={summary.get('model', '-')} revision={summary.get('revision', '-')}")
            else:
                self.output(
                    f"mode={summary.get('runtime_mode', '-')} engine={summary.get('engine', '-')} "
                    f"endpoint={summary.get('endpoint', '-')}"
                )
        elif command == "/session":
            self.output(self.status())
        elif command == "/tasks":
            rows = self.agent.state.tasks.tasks
            self.output("\n".join(f"{row.task_id}\t{row.status.value}\t{row.description}" for row in rows) or "No tasks.")
        elif command == "/task" and len(parts) >= 3 and parts[1] == "new":
            state = self.agent.create_task(" ".join(parts[2:]))
            self.output(f"Active task: {state.active_task_id}")
        elif command == "/task" and len(parts) == 3 and parts[1] == "use":
            self.agent.activate_task(parts[2])
            self.output(f"Active task: {parts[2]}")
        elif command == "/task" and len(parts) == 3 and parts[1] == "done":
            self.agent.complete_task(parts[2])
            self.output(f"Completed task: {parts[2]}")
        elif command == "/context":
            state = self.agent.state
            if state.active_task_id and state.records:
                active = next(
                    row for row in state.tasks.tasks
                    if row.task_id == state.active_task_id
                )
                selected = self.agent.runtime.select_task_context(
                    self.agent.session,
                    active.description,
                    policy=self.agent.config.task_scope,
                    max_records=self.agent.config.context_records,
                )
                self.output("\n".join(selected.selected_record_ids) or "No selected records.")
            else:
                self.output("No active task context.")
        elif command == "/tools":
            resources = self.agent.runtime.capability_resources(kinds=("tool",)) if self.agent.runtime.capabilities else ()
            self.output("\n".join(f"{row.name}\t{row.side_effect_class.value}" for row in resources) or "No tools.")
        elif command == "/skills":
            resources = self.agent.runtime.capability_resources(kinds=("skill",)) if self.agent.runtime.capabilities else ()
            self.output("\n".join(f"{row.name}\t{row.uri}" for row in resources) or "No skills.")
        elif command == "/metrics":
            self.output(str(self.agent.runtime.inspect()))
        elif command == "/clear":
            click.clear()
        else:
            self.output("Unknown command. Use /help.")
        return True

    def run(self) -> None:
        """Run a resumable prompt loop with slash commands and multiline input."""

        self.agent.authorization_callback = self.authorize
        self.output("PRA Agent")
        self.output(self.status())
        self.output("Use /help for commands. End a line with \\ for multiline input.")
        while True:
            try:
                line = click.prompt("you", prompt_suffix="> ")
                while line.endswith("\\"):
                    line = line[:-1] + "\n" + click.prompt("...", prompt_suffix="> ")
            except (EOFError, KeyboardInterrupt):
                self.output("")
                break
            if line.startswith("/"):
                if not self.dispatch(line):
                    break
                continue
            turn = self.agent.run_turn(line)
            self.output(f"assistant> {turn.text}")
