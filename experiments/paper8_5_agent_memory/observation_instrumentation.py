"""Build generic tool-effect metadata for mini-swe-agent Bash observations."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .dag import (
    EffectKind,
    bash_semantics_are_certifiable,
    classify_bash_effect,
)
from .recordizer import extract_resource_ids


def stable_environment_fingerprint(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(values), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_bash_observation_metadata(
    *,
    command: str,
    cwd: str,
    raw_output: str,
    return_code: int,
    exception_info: str,
    pre_state: Mapping[str, Any],
    post_state: Mapping[str, Any],
    environment_fingerprint: str,
    visible_output_limit: int = 10_000,
) -> dict[str, Any]:
    """Describe one Bash result without claiming more than tracing establishes."""

    kind = classify_bash_effect(command)
    semantics_complete = bash_semantics_are_certifiable(command)
    timed_out = "TimeoutExpired" in exception_info or "timed out" in exception_info.lower()
    output_truncated = len(raw_output) >= visible_output_limit
    resources = extract_resource_ids(command, "")
    pre_versions = dict(pre_state.get("resource_version_fingerprints") or {})
    post_versions = dict(post_state.get("resource_version_fingerprints") or {})
    workspace_pre = pre_state.get("workspace_version_fingerprint")
    workspace_post = post_state.get("workspace_version_fingerprint")

    effects = []
    for resource in resources:
        fingerprint = (
            post_versions.get(resource)
            if kind == EffectKind.WRITE
            else pre_versions.get(resource)
        )
        effects.append({
            "kind": kind.value,
            "resource_id": resource,
            "resource_version_fingerprint": fingerprint,
        })
    if not effects and kind in {EffectKind.PURE, EffectKind.VERIFY}:
        effects.append({
            "kind": kind.value,
            "resource_id": "workspace",
            "resource_version_fingerprint": workspace_pre,
        })

    snapshots_complete = bool(pre_state.get("complete") and post_state.get("complete"))
    versions_complete = bool(effects) and all(
        row["resource_version_fingerprint"] for row in effects
    )
    complete = bool(
        semantics_complete
        and kind != EffectKind.UNKNOWN
        and snapshots_complete
        and versions_complete
        and not timed_out
        and not output_truncated
    )
    return {
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "cwd": cwd,
        "return_code": return_code,
        "raw_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
        "environment_fingerprint": environment_fingerprint,
        "resource_version_fingerprints": pre_versions,
        "post_resource_version_fingerprints": post_versions,
        "workspace_version_fingerprint": workspace_pre,
        "post_workspace_version_fingerprint": workspace_post,
        "output_complete": not timed_out and not output_truncated,
        "timed_out": timed_out,
        "output_truncated": output_truncated,
        "tool_semantics": {
            "schema_version": 1,
            "category": "bash",
            "provenance": "runtime_traced" if complete else "static_heuristic",
            "complete": complete,
            "unknown_barrier": not complete,
            "effects": effects,
        },
    }
