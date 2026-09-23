"""Normalize mini-swe-agent's linear message list into typed logical records."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from pra_hf.agent_history import OpenAIRecordizer

from .model import AgentRecord, AgentRecordRole, AgentTurn, CanonicalAgentHistory
from .miniswe_semantics import (
    declared_turn_metadata,
    extract_resource_ids,
)


_MINISWE_COMMAND_BLOCK = re.compile(
    r"^```mswea_bash_command[ \t]*\r?\n(.*?)\r?\n^```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_COMMAND_BLOCK = re.compile(
    r"^```(?:bash)?[ \t]*\r?\n(.*?)\r?\n^```[ \t]*$",
    re.DOTALL | re.MULTILINE,
)
_RETURN_CODE = re.compile(r"<returncode>(-?\d+)</returncode>")
_SOURCE_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:cat|head|tail|less|more|sed\s+-n|rg|grep)\b"
)
_MUTATION_COMMAND = re.compile(
    r"(?:apply_patch|sed\s+-i|perl\s+-pi|git\s+apply|patch\s+-p|"
    r"(?:python\d*|ruby)\b.*(?:write_text|open\([^)]*,\s*['\"]w|replace\()|"
    r">{1,2}\s*[^&|]+)"
)
_DIRECT_MUTATION_COMMAND = re.compile(
    r"(?:apply_patch|sed\s+-i|perl\s+-pi|git\s+apply|patch\s+-p|"
    r"(?:python\d*|ruby)\b.*(?:write_text|open\([^)]*,\s*['\"]w|replace\())"
)
_VERIFICATION_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)(?:pytest|tox|nox|make\s+(?:test|check|lint)|"
    r"python\s+-m\s+(?:pytest|unittest)|git\s+(?:diff|status)|"
    r"ruff|mypy|npm\s+test|cargo\s+test)\b"
)
_FINAL_COMMAND = re.compile(r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
_PR_DESCRIPTION = re.compile(
    r"<pr_description>\s*(.*?)\s*</pr_description>", re.DOTALL | re.IGNORECASE
)
_TASK_RESOURCE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".json", ".jsx", ".md", ".php", ".py", ".rb", ".rs",
    ".rst", ".sh", ".sql", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content", "")
    return value if isinstance(value, str) else str(value)


def _command(content: str) -> str | None:
    # mini-swe-agent executes only its tagged action block.  Prefer that exact
    # protocol marker so an explanatory Python/Bash fence earlier in the
    # response cannot be mistaken for the command that changed the workspace.
    match = _MINISWE_COMMAND_BLOCK.search(content) or _COMMAND_BLOCK.search(content)
    return match.group(1).strip() if match else None


def _return_code(content: str, message: Mapping[str, Any]) -> int | None:
    extra = message.get("extra")
    if isinstance(extra, Mapping) and isinstance(extra.get("returncode"), int):
        return int(extra["returncode"])
    match = _RETURN_CODE.search(content)
    return int(match.group(1)) if match else None


def _task_resource_ids(content: str) -> tuple[str, ...]:
    """Extract task-local paths without treating agent instructions as state.

    mini-swe-agent wraps the actual issue in ``<pr_description>`` and appends a
    long, repeated operating protocol containing example names such as
    ``patch.txt`` and ``pyproject.toml``.  Those examples are not task resource
    dependencies.  The adapter exposes only path-like values from the issue
    envelope; generic PRA code consumes the resulting typed metadata.
    """

    match = _PR_DESCRIPTION.search(content)
    task_text = match.group(1) if match else content
    values = []
    for value in extract_resource_ids(None, task_text):
        normalized = value.lower().split("?", 1)[0].split("#", 1)[0]
        suffix = "." + normalized.rsplit(".", 1)[-1] if "." in normalized else ""
        if "/" not in normalized and "\\" not in value and suffix not in _TASK_RESOURCE_EXTENSIONS:
            continue
        if normalized.startswith(("http://", "https://")):
            continue
        values.append(value)
    return tuple(dict.fromkeys(values))


def _assistant_roles(command: str | None, content: str) -> tuple[AgentRecordRole, ...]:
    roles = [AgentRecordRole.ASSISTANT_ACTION]
    if content.partition("```")[0].strip():
        roles.append(AgentRecordRole.PROGRESS)
    if command and _is_solution_mutation(command):
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
    if command and _is_solution_mutation(command):
        roles.append(AgentRecordRole.MUTATION)
    if command and _VERIFICATION_COMMAND.search(command):
        roles.append(AgentRecordRole.VERIFICATION)
    return tuple(dict.fromkeys(roles))


def _is_solution_mutation(command: str) -> bool:
    """Exclude derived verification output from the source-mutation floor.

    A command such as ``git diff > patch.txt`` does write an artifact, but it
    does not change the solution state that later reasoning must remember.  A
    direct edit remains a mutation even when the same compound command also
    performs verification.
    """

    if _DIRECT_MUTATION_COMMAND.search(command):
        return True
    return bool(
        _MUTATION_COMMAND.search(command)
        and not _VERIFICATION_COMMAND.search(command)
    )


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
                resource_ids=_task_resource_ids(content),
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
            portable = declared_turn_metadata(command=last_command)
            records.append(AgentRecord(
                record_id, turn_id, group_id, index, role, content,
                AgentRecordRole.ASSISTANT_ACTION, roles,
                command=last_command,
                resource_ids=extract_resource_ids(last_command, content),
                metadata=portable,
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
        portable = declared_turn_metadata(
            command=last_command,
            observation_content=content,
            observation_metadata=extra,
        )
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
                **portable,
                "observation_for_command": last_command,
                "cwd": extra.get("cwd"),
                "environment_fingerprint": extra.get("environment_fingerprint"),
                "workspace_lineage_id": extra.get("workspace_lineage_id"),
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


def annotate_minisweagent_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project mini-swe-agent traffic into the portable ``pra_record`` schema.

    This compatibility adapter is evaluation-side code.  The common PRA
    mediator subsequently consumes the same typed schema used by standard
    OpenAI tools or any other agent; no Bash or mini-swe syntax enters the
    selector, gateway, runtime, or engine.
    """

    history = recordize_minisweagent_messages(messages)
    by_index = {record.message_index: record for record in history.records}
    annotated: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        copied = dict(message)
        record = by_index[index]
        metadata = dict(copied.get("metadata") or {})
        metadata["pra_record"] = {
            "schema_version": 1,
            "record_id": record.record_id,
            "turn_id": record.turn_id,
            "causal_group_id": record.causal_group_id,
            "primary_role": record.primary_role.value,
            "semantic_roles": [role.value for role in record.semantic_roles],
            "command": record.command,
            "return_code": record.return_code,
            "resource_ids": list(record.resource_ids),
            "metadata": dict(record.metadata),
        }
        copied["metadata"] = metadata
        annotated.append(copied)
    return annotated


def recordize_replay_messages(
    messages: Sequence[Mapping[str, Any]],
) -> CanonicalAgentHistory:
    """Recordize either one mini-swe trace or a fully typed session trace.

    Multi-issue sessions have more than one task record and may span workspace
    boundaries. Their composer declares every record explicitly. Mixing typed
    and inferred records is rejected because an episode boundary must never be
    guessed from ordinary user text.
    """

    declared = []
    for message in messages:
        metadata = message.get("metadata")
        declared.append(bool(
            isinstance(message.get("pra_record"), Mapping)
            or isinstance(metadata, Mapping)
            and isinstance(metadata.get("pra_record"), Mapping)
        ))
    if any(declared):
        if not all(declared):
            raise ValueError("multi-issue replay cannot mix typed and inferred records")
        result = OpenAIRecordizer().recordize(
            messages, request_metadata={"session_id": "paper8_5_frozen_session"}
        )
        if not result.exact:
            raise ValueError(
                "typed multi-issue replay is ambiguous: "
                + ", ".join(result.ambiguity_reasons)
            )
        return result.history
    return recordize_minisweagent_messages(messages)


def active_task_content(messages: Sequence[Mapping[str, Any]]) -> str:
    """Return the newest explicitly declared task, with legacy fallback.

    A persistent developer session can contain several issue statements.  The
    current issue is therefore the last TASK record, not the first user
    message.  Ordinary one-issue mini-swe-agent traces remain unchanged.
    """

    tasks: list[str] = []
    for message in messages:
        metadata = message.get("metadata")
        declaration = message.get("pra_record")
        if not isinstance(declaration, Mapping) and isinstance(metadata, Mapping):
            declaration = metadata.get("pra_record")
        if isinstance(declaration, Mapping) and str(
            declaration.get("primary_role") or ""
        ).lower() == AgentRecordRole.TASK.value:
            tasks.append(str(message.get("content") or ""))
    if tasks:
        return tasks[-1]
    return next(
        (str(row.get("content", "")) for row in messages if row.get("role") == "user"),
        "",
    )
