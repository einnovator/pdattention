"""Serialize a logical memory plan to an ordinary chat request."""

from __future__ import annotations

from typing import Mapping, Sequence

from .materialization import MaterializedMemoryPlan
from .model import CanonicalAgentHistory


def serialize_materialized_messages(
    history: CanonicalAgentHistory,
    materialized: MaterializedMemoryPlan,
) -> list[dict[str, str]]:
    """Return selected records in canonical order with their original chat roles."""

    source = history.record_by_id
    messages = [
        {"role": source[row.record_id].role, "content": row.content}
        for row in materialized.records
    ]
    validate_minisweagent_chat(messages)
    return messages


def validate_minisweagent_chat(messages: Sequence[Mapping[str, str]]) -> None:
    """Enforce the protocol invariants needed by mini-swe-agent templates.

    A turn can contain multiple observation records, so adjacent user/tool roles
    are permitted. Adjacent assistant actions are not: they indicate that a
    causal observation was orphaned by selection.
    """

    if len(messages) < 2:
        raise ValueError("selected chat must contain the system prompt and task")
    if messages[0].get("role") != "system":
        raise ValueError("selected chat must begin with the system prompt")
    if messages[1].get("role") != "user":
        raise ValueError("selected chat must retain the original user task")
    previous = None
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role at selected message {index}: {role!r}")
        if role == "system" and index != 0:
            raise ValueError("system records may appear only at the start")
        if role == previous == "assistant":
            raise ValueError("selected chat contains adjacent assistant actions")
        previous = role
    if messages[-1].get("role") == "assistant":
        raise ValueError("selected history ends with an action lacking its observation")
