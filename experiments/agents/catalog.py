"""Agent catalog loading and non-invasive local version auditing."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .schema import AgentCatalog, AgentCatalogEntry


CATALOG_PATH = Path(__file__).with_name("agent_catalog.yaml")


def load_catalog(path: str | Path = CATALOG_PATH) -> AgentCatalog:
    """Load the machine-readable source of truth used by plans and docs."""

    return AgentCatalog.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def find_agent(slug: str, path: str | Path = CATALOG_PATH) -> AgentCatalogEntry:
    for agent in load_catalog(path).agents:
        if agent.slug == slug:
            return agent
    raise KeyError(f"unknown coding agent: {slug}")


def audit_local_agents(
    path: str | Path = CATALOG_PATH, timeout: float = 10, *, include_paths: bool = False,
) -> list[dict[str, Any]]:
    """Report installed versions without installing tools or reading credentials."""

    rows = []
    for agent in load_catalog(path).agents:
        executable = shutil.which(agent.executable)
        row: dict[str, Any] = {
            "agent": agent.slug,
            "catalog_version": agent.version,
            "installed": executable is not None,
            "executable": executable if include_paths else Path(executable).name if executable else None,
            "installed_version": None,
            "version_matches_catalog": None,
        }
        if executable:
            command = [executable, *agent.version_command[1:]]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=timeout, check=False,
                )
                output = (completed.stdout or completed.stderr).strip().splitlines()
                first = output[0] if output else ""
                match = re.search(r"\d+(?:\.\d+){1,3}(?:[-+][\w.-]+)?", first)
                row["installed_version"] = match.group(0) if match else first or None
                row["version_matches_catalog"] = row["installed_version"] == agent.version
                row["version_exit_code"] = completed.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                row["audit_error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows
