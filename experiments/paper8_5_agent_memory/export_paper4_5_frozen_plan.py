"""Export a Paper 8.5 logical selection as a Paper 4.5 engine fixture.

The exporter never reruns the selector.  It reconstructs each canonical
pre-selection request from the immutable persistent prefix and current
mini-swe-agent trajectory, verifies every recorded request and selected-message
hash, then emits Paper 4.5's ordered resource fixture.  Physical record
segmentation is the only translation performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from pra_hf.agent_history import OpenAIRecordizer

from .multi_issue_session import BoundaryMode, compose_multi_issue_session
from .run_autonomous_swebench import load_persistent_prefix


_TOKEN = re.compile(r"\S+")
_NATURAL_OBSERVATION_BOUNDARY = re.compile(
    r"^(?:diff --git |@@ |Traceback \(most recent call last\):|"
    r"\s*File \"|(?:FAILED|ERROR|PASSED)\s+|={3,}\s*(?:FAILURES|ERRORS)|"
    r"\*{3,}\s*(?:FAILURES|ERRORS)|---\s+a/|\+\+\+\s+b/)"
)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _message_content_digest(content: Any) -> str:
    if isinstance(content, str):
        return _content_digest(content)
    return _digest(content)


def _selection_digest(rows: Sequence[tuple[str, str]]) -> str:
    material = [
        {"resource_id": resource_id, "content_sha256": _content_digest(text)}
        for resource_id, text in rows
    ]
    return _digest(material)


def _split_record_text(content: str, segment_tokens: int, *, natural: bool) -> list[str]:
    words = list(_TOKEN.finditer(content))
    if not words:
        return [content] if content else []
    if len(words) <= segment_tokens:
        return [content]
    region_starts = [0]
    if natural:
        cursor = 0
        for line in content.splitlines(keepends=True):
            if cursor and _NATURAL_OBSERVATION_BOUNDARY.match(line):
                region_starts.append(cursor)
            cursor += len(line)
    region_starts.append(len(content))
    result: list[str] = []
    for region_start, region_end in zip(region_starts, region_starts[1:]):
        region = content[region_start:region_end]
        region_words = list(_TOKEN.finditer(region))
        for offset in range(0, len(region_words), segment_tokens):
            end = (
                region_words[offset + segment_tokens].start()
                if offset + segment_tokens < len(region_words)
                else len(region)
            )
            start = 0 if offset == 0 else region_words[offset].start()
            text = region[start:end]
            if text:
                result.append(text)
    if "".join(result) != content:
        raise AssertionError("Paper 4.5 segmentation changed record text")
    return result


def _mandatory_indices(
    messages: Sequence[Mapping[str, Any]], *, current_episode_start: int,
) -> set[int]:
    system = {index for index, row in enumerate(messages) if row.get("role") == "system"}
    latest_observation = next(
        (
            index for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") != "system"
        ),
        len(messages) - 1,
    )
    latest_action = next(
        (
            index
            for index in range(latest_observation - 1, current_episode_start - 1, -1)
            if messages[index].get("role") == "assistant"
        ),
        None,
    )
    tail_start = latest_action if latest_action is not None else latest_observation
    return {
        *system,
        *(
            index for index in range(tail_start, latest_observation + 1)
            if messages[index].get("role") != "system"
        ),
    }


def _selected_indices(
    messages: Sequence[Mapping[str, Any]], selected_hashes: Sequence[str],
) -> list[int]:
    """Resolve the recorded selected subsequence without assuming unique text."""

    indices: list[int] = []
    cursor = 0
    for expected in selected_hashes:
        for index in range(cursor, len(messages)):
            content = str(messages[index].get("content", ""))
            if _content_digest(content) == expected:
                indices.append(index)
                cursor = index + 1
                break
        else:
            raise ValueError(
                "recorded selected-message hashes are not an ordered subset of "
                "the reconstructed request"
            )
    return indices


def _wire_record_message_indices(
    messages: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
    record_ids: Sequence[str],
) -> dict[str, int]:
    """Resolve legacy positional IDs and native stable record IDs.

    Native OpenAI-tool campaigns use the common runtime recordizer's stable
    ``r-*`` identities. Re-running recordization is deterministic and does not
    rerun selection: the frozen wire plan remains authoritative.
    """

    positional: dict[str, int] = {}
    native: list[str] = []
    for record_id in record_ids:
        match = re.fullmatch(r"record-(\d+)", record_id)
        if match is None:
            native.append(record_id)
            continue
        index = int(match.group(1))
        if index >= len(messages):
            raise ValueError(f"selected record is outside the request: {record_id}")
        positional[record_id] = index
    if not native:
        return positional
    if positional:
        raise ValueError("wire plan mixes positional and stable record identities")
    session_id = str(trace.get("session_id") or "")
    if not session_id:
        raise ValueError("stable record identities require a session identity")
    result = OpenAIRecordizer().recordize(
        messages,
        request_metadata={"session_id": session_id},
    )
    if not result.exact:
        raise ValueError(
            "captured request cannot be recordized exactly: "
            + ", ".join(result.ambiguity_reasons)
        )
    mapping = {row.record_id: row.message_index for row in result.history.records}
    missing = [record_id for record_id in native if record_id not in mapping]
    if missing:
        raise ValueError(
            "wire plan contains records absent from the captured request: "
            + ", ".join(missing)
        )
    return mapping


def _selected_materialized_messages(
    messages: Sequence[Mapping[str, Any]], trace: Mapping[str, Any],
) -> tuple[list[int], list[dict[str, Any]]]:
    """Resolve an exact selected subsequence, including wire replacements.

    Older fixtures selected unchanged records and can be recovered from their
    content hashes alone.  Newer policies may replace a selected record's
    payload with a compact receipt.  In that case the wire plan supplies both
    the stable record identity and replacement text; treating the replacement
    hash as if it belonged to the source request would either fail export or,
    worse, silently approximate a different policy.
    """

    selected_hashes = [
        str(value) for value in trace.get("selected_message_content_sha256", ())
    ]
    wire_plan = trace.get("wire_plan")
    if not isinstance(wire_plan, Mapping):
        indices = _selected_indices(messages, selected_hashes)
        return indices, [
            dict(messages[index])
            for index in indices
        ]

    record_ids = wire_plan.get("selected_record_ids")
    replacements = wire_plan.get("record_replacements", {})
    if not isinstance(record_ids, list) or len(record_ids) != len(selected_hashes):
        raise ValueError(
            "wire plan selected-record identities do not match recorded "
            "selected-message hashes"
        )
    if not isinstance(replacements, Mapping):
        raise ValueError("wire plan record replacements are not an object")

    record_id_values = [str(value) for value in record_ids]
    replacement_ids = [str(value) for value in replacements]
    record_to_index = _wire_record_message_indices(
        messages,
        trace,
        (*record_id_values, *replacement_ids),
    )

    indices: list[int] = []
    selected_messages: list[dict[str, Any]] = []
    previous = -1
    for record_id, expected_hash in zip(record_id_values, selected_hashes):
        index = record_to_index[record_id]
        if index <= previous or index >= len(messages):
            raise ValueError("wire plan selected records are not an ordered request subset")
        source = messages[index]
        content: Any = (
            replacements[record_id]
            if record_id in replacements
            else source.get("content", "")
        )
        stable_content_hashes = (
            trace.get("message_content_digest_scheme")
            == "raw-string-or-canonical-json-v1"
        )
        if (
            (stable_content_hashes or isinstance(content, str))
            and _message_content_digest(content) != expected_hash
        ):
            raise ValueError(
                f"wire materialization for {record_id} does not match the "
                "recorded selected-message hash"
            )
        indices.append(index)
        selected = dict(source)
        selected["content"] = content
        selected_messages.append(selected)
        previous = index
    return indices, selected_messages


def _assistant_text_from_event(message: Mapping[str, Any]) -> str:
    """Project a native agent assistant event to OpenAI ``content`` text."""

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(item.get("text", ""))
        for item in content
        if isinstance(item, Mapping) and item.get("type") == "text"
    )


def _expected_assistant_message(
    native_events: Path,
    *,
    native_request_index: int,
) -> dict[str, Any]:
    if native_request_index <= 0:
        raise ValueError("native request index must be positive")
    by_type: dict[str, list[Mapping[str, Any]]] = {
        "message_end": [],
        "message": [],
    }
    for line in native_events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_type = str(event.get("type") or "")
        if event_type not in by_type:
            continue
        message = event.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        by_type[event_type].append(message)
    events = by_type["message_end"] or by_type["message"]
    if native_request_index > len(events):
        raise ValueError(
            "native event stream has fewer assistant actions than the captured "
            f"request index: {len(events)} < {native_request_index}"
        )
    source = events[native_request_index - 1]
    result: dict[str, Any] = {
        "role": "assistant",
        "content": _assistant_text_from_event(source),
    }
    parts = source.get("content")
    if isinstance(parts, list):
        tool_calls = []
        for part in parts:
            if not isinstance(part, Mapping) or part.get("type") != "toolCall":
                continue
            tool_calls.append({
                "id": str(part.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(part.get("name") or ""),
                    "arguments": json.dumps(
                        part.get("arguments", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            })
        if tool_calls:
            result["tool_calls"] = tool_calls
    return result


def export_captured_request_fixture(
    *,
    captured_request: Path,
    request_selection: Path,
    native_events: Path,
    output: Path,
    request_replay_output: Path,
    segment_tokens: int = 256,
) -> dict[str, Any]:
    """Export one agent-neutral captured request without rerunning selection.

    Native OpenAI-tool agents already carry their persistent history in the
    request itself.  Their first request after an audited campaign continuation
    is therefore a complete, immutable handoff point; unlike mini-swe-agent it
    needs no harness-specific episode reconstruction.  Tool schemas and native
    message fields remain part of the logical payload.
    """

    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    payload = json.loads(captured_request.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("captured request is not an object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not all(
        isinstance(message, Mapping) for message in messages
    ):
        raise ValueError("captured request has no canonical message list")
    request_digest = _digest(messages)
    trace_rows = [
        json.loads(line)
        for line in request_selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        row for row in trace_rows
        if row.get("request_input_sha256") == request_digest
    ]
    if len(matches) != 1:
        raise ValueError(
            "captured request identity is not unique in selection trace: "
            f"observed {len(matches)} matches"
        )
    trace = matches[0]
    roles = [str(message.get("role", "")) for message in messages]
    content_hashes = [
        _message_content_digest(message.get("content", "")) for message in messages
    ]
    if roles != trace.get("request_message_roles"):
        raise ValueError("captured request role sequence disagrees with trace")
    recorded_content_hashes = trace.get("request_message_content_sha256")
    stable_content_hashes = (
        trace.get("message_content_digest_scheme")
        == "raw-string-or-canonical-json-v1"
    )
    if stable_content_hashes:
        if content_hashes != recorded_content_hashes:
            raise ValueError("captured request content identities disagree with trace")
    else:
        # Legacy traces used ``str(value)`` for structured content, which is
        # sensitive to JSON key insertion order. The canonical full-request
        # digest above remains authoritative; retain scalar checks here.
        for message, observed, expected in zip(
            messages, content_hashes, recorded_content_hashes or ()
        ):
            if isinstance(message.get("content", ""), str) and observed != expected:
                raise ValueError(
                    "captured scalar-content identity disagrees with trace"
                )

    selected_indices, selected_messages = _selected_materialized_messages(
        messages, trace,
    )
    if _digest(selected_messages) != trace.get("selected_messages_sha256"):
        raise ValueError("captured selected-message identity disagrees with trace")
    mandatory = _mandatory_indices(messages, current_episode_start=0)
    if not mandatory.issubset(selected_indices):
        raise ValueError("captured plan omits mandatory native current state")

    selected_by_index = dict(zip(selected_indices, selected_messages))
    resources: list[tuple[str, str]] = []
    for index in selected_indices:
        if index in mandatory:
            continue
        role = str(selected_by_index[index].get("role", ""))
        content = str(selected_by_index[index].get("content", ""))
        for child, text in enumerate(_split_record_text(
            content,
            segment_tokens,
            natural=role in {"tool", "user"},
        )):
            resources.append((f"m{index}-{child}-{role}", text))

    wire_plan = trace.get("wire_plan")
    replacements = (
        wire_plan.get("record_replacements", {})
        if isinstance(wire_plan, Mapping) else {}
    )
    if not isinstance(replacements, Mapping):
        raise ValueError("captured wire replacements are not an object")
    record_to_index = _wire_record_message_indices(
        messages,
        trace,
        [str(value) for value in replacements],
    )
    materialized_replacements: list[dict[str, Any]] = []
    for record_id_value, replacement_value in replacements.items():
        record_id = str(record_id_value)
        index = record_to_index[record_id]
        if index not in selected_by_index:
            raise ValueError(f"replacement record is not selected: {record_id}")
        content = str(replacement_value)
        if str(selected_by_index[index].get("content", "")) != content:
            raise ValueError(f"replacement content disagrees for {record_id}")
        materialized_replacements.append({
            "record_id": record_id,
            "message_index": index,
            "role": str(selected_by_index[index].get("role", "")),
            "content": content,
            "content_sha256": _content_digest(content),
        })

    fixture = {
        "schema_version": 1,
        "contract": "frozen-agent-memory-plan-v1",
        "source_policy": str(trace.get("plan_policy") or trace.get("policy")),
        "source_plan_digest": trace.get("plan_digest"),
        "source_wire_plan_digest": trace.get("wire_plan_digest"),
        "request_index": 1,
        "source_request_index": int(trace.get("request_index") or 0),
        "request_input_sha256": request_digest,
        "session_id": str(trace.get("session_id") or ""),
        "segment_tokens": segment_tokens,
        "mandatory_message_indices": sorted(mandatory),
        "selected_message_indices": selected_indices,
        "materialized_message_replacements": materialized_replacements,
        "selected_resource_digest": _selection_digest(resources),
        "resources": [
            {"resource_id": resource_id, "text": text}
            for resource_id, text in resources
        ],
    }
    logical_payload = dict(payload)
    logical_payload["messages"] = [dict(message) for message in messages]
    logical_payload["stream"] = False
    logical_payload.pop("stream_options", None)
    logical_payload.pop("store", None)
    native_request_index = int(
        trace.get("native_request_index") or trace.get("request_index") or 0
    )
    expected_message = _expected_assistant_message(
        native_events,
        native_request_index=native_request_index,
    )
    expected = str(expected_message.get("content") or "")
    replay = {
        "schema_version": 1,
        "contract": "paper8.5-frozen-agent-request-replay-v1",
        "request_index": 1,
        "source_request_index": int(trace.get("request_index") or 0),
        "request_input_sha256": request_digest,
        "logical_payload_sha256": _digest(logical_payload),
        "session_id": str(trace.get("session_id") or ""),
        "logical_payload": logical_payload,
        "expected_assistant_content": expected,
        "expected_assistant_content_sha256": _content_digest(expected),
        "expected_assistant_message": expected_message,
        "expected_assistant_message_sha256": _digest(expected_message),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, sort_keys=True) + "\n", encoding="utf-8")
    request_replay_output.parent.mkdir(parents=True, exist_ok=True)
    request_replay_output.write_text(
        json.dumps(replay, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "schema_version": 1,
        "contract": "paper8.5-to-paper4.5-captured-request-export-v1",
        "captured_request_sha256": hashlib.sha256(
            captured_request.read_bytes()
        ).hexdigest(),
        "request_selection_sha256": hashlib.sha256(
            request_selection.read_bytes()
        ).hexdigest(),
        "native_events_sha256": hashlib.sha256(native_events.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "request_replay_sha256": hashlib.sha256(
            request_replay_output.read_bytes()
        ).hexdigest(),
        "source_request_index": int(trace.get("request_index") or 0),
        "source_policy": fixture["source_policy"],
    }


def _request_messages(
    prior_episodes: Sequence[Mapping[str, Any]],
    current_trajectory: Mapping[str, Any],
    assistant_ordinal: int,
    *,
    session_id: str,
) -> tuple[list[dict[str, str]], int]:
    current_messages = current_trajectory.get("messages")
    if not isinstance(current_messages, list):
        raise ValueError("current trajectory has no messages")
    seen = 0
    stop = None
    for index, message in enumerate(current_messages):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            seen += 1
            if seen == assistant_ordinal:
                stop = index
                break
    if stop is None:
        raise ValueError(f"trajectory has no assistant decision {assistant_ordinal}")
    current = dict(current_trajectory)
    current["messages"] = [
        dict(row) for row in current_messages[:stop] if isinstance(row, Mapping)
    ]
    composed = compose_multi_issue_session(
        (*prior_episodes, current),
        session_id=session_id,
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    messages = [
        {"role": str(row.get("role", "")), "content": str(row.get("content", ""))}
        for row in composed["messages"]
    ]
    # Every episode after the first drops its repeated system message.  The
    # current-episode floor prevents Paper 4.5's active-tail rule from reaching
    # into the preceding issue on the first request of a new task.
    current_visible_count = len(current["messages"])
    if prior_episodes and current["messages"] and current["messages"][0].get("role") == "system":
        current_visible_count -= 1
    return messages, len(messages) - current_visible_count


def _assistant_content(
    trajectory: Mapping[str, Any], assistant_ordinal: int,
) -> str:
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError("current trajectory has no messages")
    seen = 0
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            seen += 1
            if seen == assistant_ordinal:
                return str(message.get("content", ""))
    raise ValueError(f"trajectory has no assistant decision {assistant_ordinal}")


def export_fixture(
    *,
    persistent_prefix: Path,
    trajectory: Path,
    request_selection: Path,
    output: Path,
    segment_tokens: int = 256,
    request_replay_output: Path | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 0,
    max_completion_tokens: int = 1024,
) -> dict[str, Any]:
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    prior_episodes, prefix_identity = load_persistent_prefix(persistent_prefix)
    source_trajectory = json.loads(trajectory.read_text(encoding="utf-8"))
    if not isinstance(source_trajectory, Mapping):
        raise ValueError("current trajectory is not an object")
    # Autonomous campaign artifacts store the canonical mini-swe-agent
    # trajectory inside an evidence envelope.  Accept that immutable envelope
    # directly so the Paper 4.5 handoff does not depend on an untracked
    # extract/re-serialization step.
    current = source_trajectory.get("trajectory", source_trajectory)
    if not isinstance(current, Mapping):
        raise ValueError("current trajectory envelope has no trajectory object")
    trace_rows = [
        json.loads(line)
        for line in request_selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not trace_rows:
        raise ValueError("request selection trace is empty")
    if request_replay_output is not None and not model:
        raise ValueError("request replay export requires a model identity")
    session_ids = {str(row.get("session_id") or "") for row in trace_rows}
    if len(session_ids) != 1 or not next(iter(session_ids)):
        raise ValueError("request trace has ambiguous session identity")
    session_id = next(iter(session_ids))

    fixture_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for ordinal, trace in enumerate(trace_rows, 1):
        if int(trace.get("request_index", -1)) != ordinal:
            raise ValueError("request trace is not a contiguous one-based sequence")
        messages, current_episode_start = _request_messages(
            prior_episodes, current, ordinal, session_id=session_id,
        )
        if _digest(messages) != trace.get("request_input_sha256"):
            raise ValueError(f"request {ordinal} failed canonical input validation")
        request_hashes = [_content_digest(row["content"]) for row in messages]
        if request_hashes != trace.get("request_message_content_sha256"):
            raise ValueError(f"request {ordinal} failed message-content validation")
        selected_indices, selected_messages = _selected_materialized_messages(
            messages, trace,
        )
        if _digest(selected_messages) != trace.get("selected_messages_sha256"):
            raise ValueError(f"request {ordinal} failed selected-message validation")

        mandatory = _mandatory_indices(
            messages, current_episode_start=current_episode_start,
        )
        if not mandatory.issubset(selected_indices):
            raise ValueError(
                f"request {ordinal} selected plan omits mandatory current-episode state"
            )
        resource_indices = [index for index in selected_indices if index not in mandatory]
        resources: list[tuple[str, str]] = []
        selected_by_index = dict(zip(selected_indices, selected_messages))
        for index in resource_indices:
            role = selected_by_index[index]["role"]
            content = selected_by_index[index]["content"]
            for child, text in enumerate(
                _split_record_text(
                    content,
                    segment_tokens,
                    natural=role in {"tool", "user"},
                )
            ):
                resources.append((f"m{index}-{child}-{role}", text))
        wire_plan = trace.get("wire_plan")
        raw_replacements = (
            wire_plan.get("record_replacements", {})
            if isinstance(wire_plan, Mapping)
            else {}
        )
        materialized_replacements: list[dict[str, Any]] = []
        for record_id_value, replacement_value in raw_replacements.items():
            record_id = str(record_id_value)
            match = re.fullmatch(r"record-(\d+)", record_id)
            if match is None:
                raise ValueError(f"unsupported replacement record identity: {record_id}")
            index = int(match.group(1))
            if index not in selected_by_index:
                raise ValueError(f"replacement record is not selected: {record_id}")
            content = str(replacement_value)
            if selected_by_index[index]["content"] != content:
                raise ValueError(f"replacement content disagrees for {record_id}")
            materialized_replacements.append({
                "record_id": record_id,
                "message_index": index,
                "role": selected_by_index[index]["role"],
                "content": content,
                "content_sha256": _content_digest(content),
            })
        fixture_rows.append({
            "schema_version": 1,
            "contract": "frozen-agent-memory-plan-v1",
            "source_policy": str(trace.get("plan_policy") or trace.get("policy")),
            "source_plan_digest": trace.get("plan_digest"),
            "source_wire_plan_digest": trace.get("wire_plan_digest"),
            "request_index": ordinal,
            "request_input_sha256": trace["request_input_sha256"],
            "session_id": session_id,
            "segment_tokens": segment_tokens,
            "mandatory_message_indices": sorted(mandatory),
            "selected_message_indices": selected_indices,
            "materialized_message_replacements": materialized_replacements,
            "selected_resource_digest": _selection_digest(resources),
            "resources": [
                {"resource_id": resource_id, "text": text}
                for resource_id, text in resources
            ],
        })
        if request_replay_output is not None:
            expected_content = _assistant_content(current, ordinal)
            replay_rows.append({
                "schema_version": 1,
                "contract": "paper8.5-frozen-agent-request-replay-v1",
                "request_index": ordinal,
                "request_input_sha256": trace["request_input_sha256"],
                "session_id": session_id,
                "logical_payload": {
                    "model": model,
                    "messages": messages,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "seed": int(seed),
                    "stream": False,
                    "max_tokens": int(max_completion_tokens),
                    "session_id": session_id,
                },
                "expected_assistant_content": expected_content,
                "expected_assistant_content_sha256": _content_digest(
                    expected_content
                ),
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fixture_rows),
        encoding="utf-8",
    )
    if request_replay_output is not None:
        request_replay_output.parent.mkdir(parents=True, exist_ok=True)
        request_replay_output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in replay_rows),
            encoding="utf-8",
        )
    result = {
        "schema_version": 1,
        "contract": "paper8.5-to-paper4.5-frozen-plan-export-v1",
        "prefix_sha256": prefix_identity["sha256"],
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "request_selection_sha256": hashlib.sha256(request_selection.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "requests": len(fixture_rows),
        "segment_tokens": segment_tokens,
        "source_policy": fixture_rows[0]["source_policy"],
    }
    if request_replay_output is not None:
        result.update({
            "request_replay_sha256": hashlib.sha256(
                request_replay_output.read_bytes()
            ).hexdigest(),
            "model": model,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "seed": int(seed),
            "max_completion_tokens": int(max_completion_tokens),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-prefix", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--captured-request", type=Path)
    parser.add_argument("--native-events", type=Path)
    parser.add_argument("--request-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--request-replay-output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--segment-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.captured_request is not None:
        if args.persistent_prefix is not None or args.trajectory is not None:
            parser.error(
                "--captured-request cannot be combined with "
                "--persistent-prefix/--trajectory"
            )
        if args.native_events is None or args.request_replay_output is None:
            parser.error(
                "captured-request export requires --native-events and "
                "--request-replay-output"
            )
        result = export_captured_request_fixture(
            captured_request=args.captured_request,
            request_selection=args.request_selection,
            native_events=args.native_events,
            output=args.output,
            request_replay_output=args.request_replay_output,
            segment_tokens=args.segment_tokens,
        )
    else:
        if args.persistent_prefix is None or args.trajectory is None:
            parser.error(
                "trajectory export requires --persistent-prefix and --trajectory"
            )
        result = export_fixture(
            persistent_prefix=args.persistent_prefix,
            trajectory=args.trajectory,
            request_selection=args.request_selection,
            output=args.output,
            segment_tokens=args.segment_tokens,
            request_replay_output=args.request_replay_output,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            max_completion_tokens=args.max_completion_tokens,
        )
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
