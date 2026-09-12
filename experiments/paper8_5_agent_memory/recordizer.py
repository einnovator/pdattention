"""Normalize mini-swe-agent's linear message list into typed logical records."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .model import AgentRecord, AgentRecordRole, AgentTurn, CanonicalAgentHistory


_COMMAND_BLOCK = re.compile(r"```(?:mswea_bash_command|bash)?\s*\n(.*?)\n```", re.DOTALL)
_RETURN_CODE = re.compile(r"<returncode>(-?\d+)</returncode>")
_PATH = re.compile(
    r"(?<![\w.-])(?:\.?\.?/)?(?:[\w.-]+/)*[\w.-]+\.[A-Za-z0-9_+-]+"
)
_SOURCE_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|head|tail|less|more|sed\s+-n|rg|grep)\b"
)
_MUTATION_COMMAND = re.compile(
    r"(?:apply_patch|sed\s+-i|perl\s+-pi|git\s+apply|patch\s+-p|"
    r"(?:python\d*|ruby)\b.*(?:write_text|open\([^)]*,\s*['\"]w|replace\()|"
    r">{1,2}\s*[^&|]+)"
)
_VERIFICATION_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:pytest|tox|nox|make\s+(?:test|check|lint)|"
    r"python\s+-m\s+(?:pytest|unittest)|git\s+(?:diff|status)|"
    r"ruff|mypy|npm\s+test|cargo\s+test)\b"
)
_FINAL_COMMAND = re.compile(r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    return value if isinstance(value, str) else str(value)


def _command(content: str) -> str | None:
    match = _COMMAND_BLOCK.search(content)
    return match.group(1).strip() if match else None


def _return_code(content: str, message: Mapping[str, Any]) -> int | None:
    extra = message.get("extra")
    if isinstance(extra, Mapping) and isinstance(extra.get("returncode"), int):
        return int(extra["returncode"])
    match = _RETURN_CODE.search(content)
    return int(match.group(1)) if match else None


def extract_resource_ids(command: str | None, content: str) -> tuple[str, ...]:
    values: list[str] = []
    for value in _PATH.findall("\n".join(part for part in (command, content) if part)):
        normalized = value.rstrip(":,;)")
        if normalized not in values:
            values.append(normalized)
    return tuple(values[:32])


def _assistant_roles(command: str | None, content: str) -> tuple[AgentRecordRole, ...]:
    roles = [AgentRecordRole.ASSISTANT_ACTION]
    if content.partition("```")[0].strip():
        roles.append(AgentRecordRole.PROGRESS)
    if command and _MUTATION_COMMAND.search(command):
        roles.append(AgentRecordRole.MUTATION)
    if command and _VERIFICATION_COMMAND.search(command):
        roles.append(AgentRecordRole.VERIFICATION)
    if command and _FINAL_COMMAND.search(command):
        roles.append(AgentRecordRole.FINALIZATION)
    return tuple(dict.fromkeys(roles))


def _observation_roles(
    command: str | None,
    content: str,
    return_code: int | None,
) -> tuple[AgentRecordRole, ...]:
    roles = [AgentRecordRole.TOOL_OBSERVATION]
    if return_code not in (None, 0) or "<exception>" in content or "Format error" in content:
        roles.append(AgentRecordRole.ERROR_OR_REJECTION)
    if command and _SOURCE_COMMAND.search(command):
        roles.append(AgentRecordRole.SOURCE_VIEW)
    if command and _MUTATION_COMMAND.search(command):
        roles.append(AgentRecordRole.MUTATION)
    if command and _VERIFICATION_COMMAND.search(command):
        roles.append(AgentRecordRole.VERIFICATION)
    return tuple(dict.fromkeys(roles))


def recordize_minisweagent_messages(
    messages: Sequence[Mapping[str, Any]],
) -> CanonicalAgentHistory:
    """Create stable records and complete action--observation causal turns.

    The system prompt and original task are immutable records outside the
    selectable turn sequence.  Each assistant action starts a turn; subsequent
    user/tool observations belong to that turn until the next assistant action.
    A standalone format-error observation receives its own error turn instead
    of being silently attached to unrelated history.
    """

    records: list[AgentRecord] = []
    turn_rows: list[dict[str, Any]] = []
    current_turn: dict[str, Any] | None = None
    task_seen = False
    last_command: str | None = None
    turn_counter = 0

    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        content = _content(message)
        record_id = f"m{index}"

        if role == "system":
            records.append(AgentRecord(
                record_id, "system", "system", index, role, content,
                AgentRecordRole.SYSTEM, (AgentRecordRole.SYSTEM,),
            ))
            continue

        if role == "user" and not task_seen:
            task_seen = True
            records.append(AgentRecord(
                record_id, "task", "task", index, role, content,
                AgentRecordRole.TASK, (AgentRecordRole.TASK,),
                resource_ids=extract_resource_ids(None, content),
            ))
            continue

        if role == "assistant":
            if current_turn is not None:
                turn_rows.append(current_turn)
            turn_id = f"t{turn_counter:04d}"
            turn_counter += 1
            group_id = f"turn:{turn_id}"
            current_turn = {
                "turn_id": turn_id,
                "causal_group_id": group_id,
                "record_ids": [record_id],
                "first_message_index": index,
                "has_assistant": True,
                "has_observation": False,
            }
            last_command = _command(content)
            roles = _assistant_roles(last_command, content)
            records.append(AgentRecord(
                record_id, turn_id, group_id, index, role, content,
                AgentRecordRole.ASSISTANT_ACTION, roles,
                command=last_command,
                resource_ids=extract_resource_ids(last_command, content),
            ))
            continue

        if current_turn is None:
            turn_id = f"t{turn_counter:04d}"
            turn_counter += 1
            group_id = f"error:{turn_id}"
            current_turn = {
                "turn_id": turn_id,
                "causal_group_id": group_id,
                "record_ids": [],
                "first_message_index": index,
                "has_assistant": False,
                "has_observation": False,
            }
            last_command = None

        current_turn["record_ids"].append(record_id)
        current_turn["has_observation"] = True
        return_code = _return_code(content, message)
        roles = _observation_roles(last_command, content, return_code)
        extra = message.get("extra")
        extra = extra if isinstance(extra, Mapping) else {}
        primary = (
            AgentRecordRole.ERROR_OR_REJECTION
            if AgentRecordRole.ERROR_OR_REJECTION in roles
            else AgentRecordRole.SOURCE_VIEW
            if AgentRecordRole.SOURCE_VIEW in roles
            else AgentRecordRole.VERIFICATION
            if AgentRecordRole.VERIFICATION in roles
            else AgentRecordRole.TOOL_OBSERVATION
        )
        records.append(AgentRecord(
            record_id,
            current_turn["turn_id"],
            current_turn["causal_group_id"],
            index,
            role,
            content,
            primary,
            roles,
            command=last_command,
            return_code=return_code,
            resource_ids=extract_resource_ids(last_command, content),
            metadata={
                "observation_for_command": last_command,
                "cwd": extra.get("cwd"),
                "environment_fingerprint": extra.get("environment_fingerprint"),
                "resource_version_fingerprints": extra.get(
                    "resource_version_fingerprints"
                ),
                "post_resource_version_fingerprints": extra.get(
                    "post_resource_version_fingerprints"
                ),
                "verification_resource_ids": extra.get("verification_resource_ids"),
                "dependency_resource_ids": extra.get("dependency_resource_ids"),
                "output_complete": extra.get("output_complete"),
                "timed_out": extra.get("timed_out"),
                "output_truncated": extra.get("output_truncated"),
                "tool_semantics": extra.get("tool_semantics"),
            },
        ))

    if current_turn is not None:
        turn_rows.append(current_turn)

    turns = tuple(
        AgentTurn(
            turn_id=row["turn_id"],
            causal_group_id=row["causal_group_id"],
            record_ids=tuple(row["record_ids"]),
            first_message_index=row["first_message_index"],
            complete=bool(row["has_assistant"] and row["has_observation"]),
        )
        for row in turn_rows
    )
    return CanonicalAgentHistory(tuple(records), turns)
