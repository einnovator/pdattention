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
    current = json.loads(trajectory.read_text(encoding="utf-8"))
    if not isinstance(current, Mapping):
        raise ValueError("current trajectory is not an object")
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
        selected_hashes = [str(value) for value in trace.get("selected_message_content_sha256", ())]
        selected_indices = _selected_indices(messages, selected_hashes)
        selected_messages = [messages[index] for index in selected_indices]
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
        for index in resource_indices:
            role = messages[index]["role"]
            content = messages[index]["content"]
            for child, text in enumerate(
                _split_record_text(
                    content,
                    segment_tokens,
                    natural=role in {"tool", "user"},
                )
            ):
                resources.append((f"m{index}-{child}-{role}", text))
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
    parser.add_argument("--persistent-prefix", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
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
