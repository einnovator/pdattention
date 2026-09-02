"""Load audited command specs and construct generic CLI adapters."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..schema import AgentCommandManifest, AgentCommandSpec
from .command import CommandAgentAdapter


COMMANDS_PATH = Path(__file__).parents[1] / "agent_commands.yaml"


def load_command_manifest(path: str | Path = COMMANDS_PATH) -> AgentCommandManifest:
    return AgentCommandManifest.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def command_spec(slug: str, path: str | Path = COMMANDS_PATH) -> AgentCommandSpec:
    for row in load_command_manifest(path).commands:
        if row.slug == slug:
            return row
    raise KeyError(f"no command spec for coding agent {slug!r}")


def command_adapter(
    slug: str,
    path: str | Path = COMMANDS_PATH,
    *,
    executable: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> CommandAgentAdapter:
    row = command_spec(slug, path)
    if not row.verified:
        raise RuntimeError(f"{slug} command is recorded but not verified for unattended execution")
    return CommandAgentAdapter(
        executable=executable or row.executable,
        version_args=row.version_args,
        run_args=row.run_args,
        extra_args=extra_args,
        prompt_arg=row.prompt_arg,
        model_arg=row.model_arg,
    )
