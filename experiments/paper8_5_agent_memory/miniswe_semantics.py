"""Evaluation-only mini-swe-agent Bash compatibility adapter.

Nothing in ``src/pra_hf`` imports this module.  It translates mini-swe-agent's
single generic Bash action into PRA's portable operation/resource vocabulary.
Other agents provide another adapter or native ``pra_record`` declarations.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from pra_hf.tool_semantics import EffectKind, OperationKind, ResourceAccess


# Compatibility name retained for existing Paper 8.5 analysis scripts.
BashOperation = OperationKind

_PATH = re.compile(
    r"(?<![\w.-])(?:\.?\.?/)?(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9_+-]+"
)
_SEARCH = re.compile(
    r"(?:^|[;&|]\s*)(?:find\b|fd\b|rg\s+--files\b|"
    r"(?:grep|rg)\b[^\n]*(?:\s-(?:[^\s]*l[^\s]*|files-with-matches)\b))"
)
_READ = re.compile(r"(?:^|[;&|]\s*)(?:cat|head|tail|less|more|sed\s+-n|grep|rg)\b")
_READ_ONLY = re.compile(
    r"^\s*(?:cat|head|tail|sed\s+-n|rg|grep|find|ls|pwd|git\s+(?:status|diff|show))\b"
)
_WRITE = re.compile(
    r"(?:apply_patch|sed\s+-i|perl\s+-pi|git\s+apply|patch\s+-p|"
    r"(?:write_text|write_bytes|open\([^)]*,\s*['\"]w)|(?:^|[;&|]\s*)rm\b)"
)
_OUTPUT_REDIRECT = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:\d+|&)?>{1,2}\s*(['\"]?[^\s;&|]+)"
)
_DIFF = re.compile(r"(?:^|[;&|]\s*)git\s+(?:diff|show)\b")
_VERIFY = re.compile(
    r"(?:^|[;&|]\s*)(?:pytest|tox|nox|python\s+-m\s+(?:pytest|unittest)|"
    r"make\s+(?:test|check|lint)|ruff|mypy|npm\s+test|cargo\s+test)\b"
)
_SED_SPAN = re.compile(r"sed\s+-n\s+['\"]?(\d+)\s*,\s*(\d+)p")
_HEAD_SPAN = re.compile(r"head(?:\s+-n)?\s+(\d+)\b")
_GREP_PATTERN = re.compile(r"(?:grep|rg)\s+(?:-[^\s]+\s+)*(['\"]?[^\s'\"]+['\"]?)")


def normalize_resource(value: str) -> str:
    value = value.strip("'\"`[](){}:,;").replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value


def extract_resource_ids(command: str | None, content: str) -> tuple[str, ...]:
    values: list[str] = []
    for value in _PATH.findall("\n".join(part for part in (command, content) if part)):
        normalized = value.rstrip(":,;)")
        if normalized not in values:
            values.append(normalized)
    return tuple(values[:32])


def classify_bash_operation(command: str | None) -> OperationKind:
    command = command or ""
    if _writes_workspace(command):
        return OperationKind.WRITE
    if _DIFF.search(command):
        return OperationKind.DIFF
    if _VERIFY.search(command):
        return OperationKind.VERIFY
    if _SEARCH.search(command):
        return OperationKind.SEARCH_DISCOVERY
    if _READ.search(command):
        return OperationKind.READ
    return OperationKind.OTHER


def classify_bash_effect(command: str | None) -> EffectKind:
    if not command:
        return EffectKind.UNKNOWN
    if _writes_workspace(command):
        return EffectKind.WRITE
    if _VERIFY.search(command):
        return EffectKind.VERIFY
    if _READ_ONLY.search(command):
        return EffectKind.READ
    if command.strip() in {"true", ":"}:
        return EffectKind.PURE
    return EffectKind.UNKNOWN


def _writes_workspace(command: str) -> bool:
    """Detect mutations without treating diagnostic sink redirects as writes."""

    if _WRITE.search(command):
        return True
    for match in _OUTPUT_REDIRECT.finditer(command):
        target = match.group(1).strip("'\"")
        if target not in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
            return True
    return False


def bash_semantics_are_certifiable(command: str | None) -> bool:
    if not command or "\n" in command or "\r" in command:
        return False
    if any(token in command for token in ("&&", "||", ";", "|", "`", "$(", ">", "<")):
        return False
    return classify_bash_effect(command) in {EffectKind.READ, EffectKind.PURE}


def search_signature(command: str) -> str:
    normalized = " ".join(command.split())
    return "search:" + hashlib.sha256(normalized.encode()).hexdigest()[:12]


def resource_span(command: str, resource: str) -> ResourceAccess:
    if re.search(r"(?:^|[;&|]\s*)cat\b", command):
        return ResourceAccess(resource, "unknown", "whole")
    if match := _SED_SPAN.search(command):
        return ResourceAccess(resource, "unknown", "lines", int(match.group(1)), int(match.group(2)))
    if match := _HEAD_SPAN.search(command):
        return ResourceAccess(resource, "unknown", "lines", 1, int(match.group(1)))
    if re.search(r"(?:^|[;&|]\s*)tail\b", command):
        digest = hashlib.sha256(command.encode()).hexdigest()[:12]
        return ResourceAccess(resource, "unknown", "query", signature=f"tail:{resource}:{digest}")
    if re.search(r"(?:^|[;&|]\s*)(?:grep|rg)\b", command):
        match = _GREP_PATTERN.search(command)
        query = match.group(1).strip("'\"") if match else command
        digest = hashlib.sha256(query.encode()).hexdigest()[:12]
        return ResourceAccess(resource, "unknown", "query", signature=f"grep:{resource}:{digest}")
    return ResourceAccess(resource, "unknown")


def declared_turn_metadata(
    *,
    command: str | None,
    observation_content: str = "",
    observation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate one Bash turn to portable fields consumed by policies."""

    metadata = dict(observation_metadata or {})
    operation = classify_bash_operation(command)
    resources = tuple(normalize_resource(row) for row in extract_resource_ids(command, ""))
    pre = metadata.get("resource_version_fingerprints")
    pre = {normalize_resource(str(k)): str(v) for k, v in pre.items()} if isinstance(pre, Mapping) else {}
    post = metadata.get("post_resource_version_fingerprints")
    post = {normalize_resource(str(k)): str(v) for k, v in post.items()} if isinstance(post, Mapping) else {}
    accesses = []
    changed = []
    for resource in resources:
        base = resource_span(command or "", resource)
        version = post.get(resource) if operation == OperationKind.WRITE else pre.get(resource)
        accesses.append({
            "resource_id": resource,
            "version": version or "unknown",
            "span_kind": base.span_kind,
            "span_start": base.span_start,
            "span_end": base.span_end,
            "signature": base.signature,
        })
        if operation == OperationKind.WRITE and version not in (None, "missing", pre.get(resource)):
            changed.append(resource)
    discovered: list[str] = []
    if operation == OperationKind.SEARCH_DISCOVERY:
        discovered.append(search_signature(command or ""))
        discovered.extend(
            normalize_resource(row)
            for row in extract_resource_ids(None, observation_content)
        )
    return {
        "tool_category": "bash",
        "operation_kind": operation.value,
        "resource_accesses": accesses,
        "discovered_resource_ids": list(dict.fromkeys(row for row in discovered if row)),
        "changed_resource_ids": list(dict.fromkeys(changed)),
    }


__all__ = [
    "BashOperation",
    "bash_semantics_are_certifiable",
    "classify_bash_effect",
    "classify_bash_operation",
    "declared_turn_metadata",
    "extract_resource_ids",
    "normalize_resource",
    "resource_span",
    "search_signature",
]
