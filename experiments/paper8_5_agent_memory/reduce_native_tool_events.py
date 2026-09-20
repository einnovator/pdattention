"""Normalize typed coding-agent events without embedding agent policy.

The reducer consumes the small JSONL vocabulary emitted by headless agents
such as Kilo and OpenCode.  Agent adapters declare tool semantics; the output
uses the same causal resource/effect vocabulary consumed by Paper 8.5.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


_FAILURE_SIGNAL = re.compile(
    r"(?:traceback \(most recent call last\)|no module named|can't open file|"
    r"command (?:failed|exited)|error:|permission denied|not found)",
    re.IGNORECASE,
)


def _operation(
    tool: str,
    inputs: Mapping[str, Any],
    semantics: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, tuple[str, ...]]:
    declaration = semantics.get(tool, {})
    category = str(declaration.get("category") or "unknown")
    operation = declaration.get("operation_kind")
    operation_argument = declaration.get("operation_argument")
    if operation_argument:
        raw = str(inputs.get(str(operation_argument)) or "")
        operation = (declaration.get("operation_map") or {}).get(
            raw, declaration.get("default_operation_kind")
        )
    resources = [str(value) for value in declaration.get("resource_ids", ())]
    for argument in declaration.get("resource_arguments", ()):
        value = inputs.get(str(argument))
        if isinstance(value, str) and value:
            resources.append(value)
        elif isinstance(value, list):
            resources.extend(str(item) for item in value if item)
    return category, str(operation or "unknown"), tuple(dict.fromkeys(resources))


def reduce_native_tool_events(
    source: Path,
    output: Path,
    *,
    agent: str,
    tool_semantics: Mapping[str, Mapping[str, Any]],
    token_note: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    invalid_lines = 0
    event_types: Counter[str] = Counter()
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if not isinstance(value, Mapping):
            invalid_lines += 1
            continue
        row = dict(value)
        event_types[str(row.get("type") or "unknown")] += 1
        rows.append(row)

    finishes = [
        row.get("part") for row in rows if row.get("type") == "step_finish"
    ]
    finishes = [part for part in finishes if isinstance(part, Mapping)]
    tools = [row.get("part") for row in rows if row.get("type") == "tool_use"]
    tools = [part for part in tools if isinstance(part, Mapping)]
    token_rows = [
        part.get("tokens") for part in finishes
        if isinstance(part.get("tokens"), Mapping)
    ]
    logical_inputs = [
        int(tokens.get("input") or 0)
        + int((tokens.get("cache") or {}).get("read") or 0)
        for tokens in token_rows
    ]

    canonical: list[dict[str, Any]] = []
    failure_results = 0
    status_errors = 0
    for index, part in enumerate(tools, start=1):
        tool = str(part.get("tool") or "unknown")
        state = part.get("state") if isinstance(part.get("state"), Mapping) else {}
        inputs = state.get("input") if isinstance(state.get("input"), Mapping) else {}
        status = str(state.get("status") or "unknown")
        output_text = str(state.get("output") or state.get("error") or "")
        failure_signal = bool(_FAILURE_SIGNAL.search(output_text))
        status_error = status != "completed"
        failure_results += int(failure_signal)
        status_errors += int(status_error)
        category, operation, resources = _operation(
            tool, inputs, tool_semantics
        )
        canonical.append({
            "schema_version": 1,
            "record_id": str(part.get("callID") or part.get("id") or f"tool-{index}"),
            "native_tool": tool,
            "category": category,
            "operation_kind": operation,
            "resource_ids": list(resources),
            "transport_status": status,
            "semantic_success": not status_error and not failure_signal,
            "failure_signal": failure_signal,
            "input": dict(inputs),
            "output_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
            "output_bytes": len(output_text.encode()),
        })

    task_calls = sum(
        str(part.get("tool") or "").lower() in {"task", "subagent"}
        for part in tools
    )
    summary = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "agent": agent,
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_event_count": len(rows),
        "invalid_json_lines": invalid_lines,
        "raw_event_types": dict(sorted(event_types.items())),
        "assistant_model_calls": len(finishes),
        "native_tool_calls": len(tools),
        "tool_call_counts": dict(sorted(Counter(
            str(part.get("tool") or "unknown") for part in tools
        ).items())),
        "tool_status_errors": status_errors,
        "tool_failure_signal_results": failure_results,
        "nested_agent_tool_calls": task_calls,
        "root_event_trace_complete": task_calls == 0 and invalid_lines == 0,
        "cumulative_logical_input_tokens": sum(logical_inputs),
        "cumulative_uncached_input_tokens": sum(
            int(tokens.get("input") or 0) for tokens in token_rows
        ),
        "cumulative_cache_read_tokens": sum(
            int((tokens.get("cache") or {}).get("read") or 0)
            for tokens in token_rows
        ),
        "cumulative_output_tokens": sum(
            int(tokens.get("output") or 0) for tokens in token_rows
        ),
        "maximum_logical_request_tokens": max(logical_inputs, default=0),
        "token_note": token_note,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversation_events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "canonical_tool_events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in canonical),
        encoding="utf-8",
    )
    (output / "event_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
