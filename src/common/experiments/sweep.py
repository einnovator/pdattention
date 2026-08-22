"""Deterministic nested parameter overrides and sweep expansion."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from typing import Any, Iterable, Mapping

from common.distributed.models import DistributionMode

from .models import ExperimentDefinition, Trial


def set_dotted(mapping: dict, path: str, value: Any) -> None:
    """Set a dotted nested path, creating intermediate dictionaries."""

    parts = [part for part in path.split(".") if part]
    if not parts:
        raise ValueError("Parameter path cannot be empty.")
    cursor = mapping
    for part in parts[:-1]:
        current = cursor.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot descend through non-mapping parameter {part!r}.")
        cursor = current
    cursor[parts[-1]] = value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_fingerprint(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _flatten(mapping: Mapping[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    for key in sorted(mapping):
        path = f"{prefix}.{key}" if prefix else str(key)
        value = mapping[key]
        if isinstance(value, Mapping):
            yield from _flatten(value, path)
        else:
            yield path, value


def trial_id(parameters: Mapping[str, Any], varied: Iterable[str]) -> str:
    flattened = dict(_flatten(parameters))
    labels = []
    for key in varied:
        if key not in flattened:
            continue
        short = key.split(".")[-1].replace("references", "refs")
        value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(flattened[key])).strip("-")
        labels.append(f"{short}={value}"[:36])
    prefix = "_".join(labels[:4]) or "trial"
    return f"{prefix}__{stable_fingerprint(parameters, 10)}"


def expand_trials(
    definition: ExperimentDefinition,
    *,
    cluster_name: str,
    distribution: DistributionMode,
    storage_name: str | None,
    cli_overrides: Mapping[str, Any] | None = None,
    max_trials: int | None = None,
) -> list[Trial]:
    """Expand Cartesian or explicit trials with CLI overrides applied last."""

    variants: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    if definition.sweep:
        keys = tuple(sorted(definition.sweep))
        for choices in itertools.product(*(definition.sweep[key] for key in keys)):
            variants.append((dict(zip(keys, choices)), keys))
    elif definition.trials:
        variants.extend((dict(values), tuple(sorted(values))) for values in definition.trials)
    else:
        variants.append(({}, ()))

    output = []
    for trial_values, varied in variants:
        parameters = copy.deepcopy(dict(definition.parameters))
        for path, value in trial_values.items():
            set_dotted(parameters, path, value)
        for path, value in (cli_overrides or {}).items():
            set_dotted(parameters, path, value)
        identity = {
            "experiment": definition.name,
            "entrypoint": definition.entrypoint.as_dict(),
            "parameters": parameters,
            "cluster": cluster_name,
            "distribution": distribution.value,
            "storage": storage_name,
        }
        output.append(
            Trial(
                experiment_name=definition.name,
                trial_id=trial_id(parameters, (*varied, *(cli_overrides or {}).keys())),
                parameters=parameters,
                entrypoint=definition.entrypoint,
                distribution=distribution,
                cluster_name=cluster_name,
                storage_name=storage_name,
                resources=definition.resources,
                fingerprint=stable_fingerprint(identity),
            )
        )
    return output[:max_trials] if max_trials is not None else output


def parse_seed_spec(value: str) -> list[int]:
    """Parse comma lists and Python-style half-open ranges such as ``0:5``."""

    seeds = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(item) for item in part.split(":")]
            if len(pieces) not in {2, 3}:
                raise ValueError(f"Invalid seed range {part!r}.")
            seeds.extend(range(*pieces))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("Seed specification did not contain any seeds.")
    return list(dict.fromkeys(seeds))
