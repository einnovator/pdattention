"""Shared configuration and presentation helpers for the PRA product CLI.

The research modules deliberately keep their own experiment configuration.
This module owns only product-level precedence, deterministic paths, and
human/machine rendering used by CLI, notebooks, and the experimental web UI.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


def pra_home(workspace: str | Path | None = None) -> Path:
    """Return the configured PRA home, defaulting to ``WORKSPACE/.pra``."""

    configured = os.environ.get("PRA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(workspace or ".").expanduser().resolve() / ".pra")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and sequence values."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML mapping and reject ambiguous non-mapping roots."""

    resolved = Path(path).expanduser()
    value = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {resolved}")
    return dict(value)


def discover_product_configs(workspace: str | Path | None = None) -> tuple[Path, ...]:
    """Return existing user-to-project config files in increasing precedence."""

    root = Path(workspace or ".").expanduser().resolve()
    candidates = (
        Path.home() / ".config" / "pra" / "config.yaml",
        root / ".pra" / "config.yaml",
        root / "pra.yaml",
    )
    return tuple(path for path in candidates if path.is_file())


def load_product_config(
    *,
    workspace: str | Path | None = None,
    command_config: str | Path | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Resolve package, user, project, and command YAML configuration layers."""

    value = deepcopy(dict(defaults or {}))
    trace = ["package defaults"]
    for path in discover_product_configs(workspace):
        value = deep_merge(value, read_yaml(path))
        trace.append(str(path))
    if command_config is not None:
        path = Path(command_config).expanduser().resolve()
        value = deep_merge(value, read_yaml(path))
        trace.append(str(path))
    return value, tuple(trace)


def dump_data(value: Any, format_name: str = "human") -> str:
    """Render product results as concise text, JSON, or YAML."""

    if format_name == "json":
        return json.dumps(value, indent=2, sort_keys=True, default=str)
    if format_name == "yaml":
        return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).rstrip()
    return _human(value)


def _human(value: Any, *, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, Mapping):
        rows: list[str] = []
        width = max((len(str(key)) for key in value), default=0)
        for key, item in value.items():
            label = str(key).replace("_", " ").title()
            if isinstance(item, (Mapping, list, tuple)):
                rows.append(f"{prefix}{label}:")
                rows.append(_human(item, indent=indent + 2))
            else:
                rows.append(f"{prefix}{label:<{width}}  {item}")
        return "\n".join(row for row in rows if row)
    if isinstance(value, (list, tuple)):
        rows = []
        for item in value:
            if isinstance(item, Mapping):
                rendered = _human(item, indent=indent + 2)
                first, *rest = rendered.splitlines()
                rows.append(f"{prefix}- {first.strip()}")
                rows.extend(rest)
            else:
                rows.append(f"{prefix}- {item}")
        return "\n".join(rows)
    return f"{prefix}{value}"

