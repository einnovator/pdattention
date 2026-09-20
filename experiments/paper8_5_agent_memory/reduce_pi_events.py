"""Reduce Pi's verbose streaming JSONL to auditable causal events and metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_KEPT_TYPES = {
    "session",
    "message_end",
    "tool_execution_start",
    "tool_execution_end",
    "agent_end",
}


def reduce_events(source: Path, output: Path) -> dict[str, Any]:
    rows = []
    event_types: Counter[str] = Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        event_types[str(row.get("type") or "unknown")] += 1
        if row.get("type") in _KEPT_TYPES:
            rows.append(row)

    assistants = [
        row["message"]
        for row in rows
        if row.get("type") == "message_end"
        and isinstance(row.get("message"), Mapping)
        and row["message"].get("role") == "assistant"
    ]
    results = [
        row["message"]
        for row in rows
        if row.get("type") == "message_end"
        and isinstance(row.get("message"), Mapping)
        and row["message"].get("role") == "toolResult"
    ]
    starts = [row for row in rows if row.get("type") == "tool_execution_start"]
    usages = [
        row.get("usage")
        for row in assistants
        if isinstance(row.get("usage"), Mapping)
    ]
    tool_names = Counter(str(row.get("toolName") or "unknown") for row in starts)
    logical_inputs = [
        int(row.get("input") or 0) + int(row.get("cacheRead") or 0)
        for row in usages
    ]
    summary = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_event_count": sum(event_types.values()),
        "raw_event_types": dict(sorted(event_types.items())),
        "retained_event_count": len(rows),
        "assistant_model_calls": len(assistants),
        "native_tool_calls": len(starts),
        "tool_call_counts": dict(sorted(tool_names.items())),
        "tool_result_errors": sum(bool(row.get("isError")) for row in results),
        "cumulative_logical_input_tokens": sum(logical_inputs),
        "cumulative_uncached_input_tokens": sum(
            int(row.get("input") or 0) for row in usages
        ),
        "cumulative_cache_read_tokens": sum(
            int(row.get("cacheRead") or 0) for row in usages
        ),
        "cumulative_output_tokens": sum(
            int(row.get("output") or 0) for row in usages
        ),
        "maximum_logical_request_tokens": max(logical_inputs, default=0),
        "token_note": (
            "Pi provider usage: logical input is input + cacheRead. These "
            "include the agent system prompt and tool schemas and are not "
            "pooled with mini-swe message-content-only token counts."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "conversation_events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "event_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(
        reduce_events(Path(args.input), Path(args.output)),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
