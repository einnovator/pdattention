"""Reduce Kilo's JSON event stream to auditable causal events and metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_FAILURE_SIGNAL = re.compile(
    r"(?:traceback \(most recent call last\)|no module named|can't open file|"
    r"command (?:failed|exited)|error:)",
    re.IGNORECASE,
)


def reduce_events(source: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    event_types: Counter[str] = Counter()
    for line in source.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        event_types[str(row.get("type") or "unknown")] += 1
        rows.append(row)

    finishes = [
        row.get("part") for row in rows if row.get("type") == "step_finish"
    ]
    finishes = [row for row in finishes if isinstance(row, Mapping)]
    tools = [row.get("part") for row in rows if row.get("type") == "tool_use"]
    tools = [row for row in tools if isinstance(row, Mapping)]
    token_rows = [
        row.get("tokens") for row in finishes
        if isinstance(row.get("tokens"), Mapping)
    ]
    logical_inputs = [
        int(row.get("input") or 0)
        + int((row.get("cache") or {}).get("read") or 0)
        for row in token_rows
    ]
    outputs = [
        str((row.get("state") or {}).get("output") or "") for row in tools
    ]
    summary = {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_transfer",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "raw_event_count": len(rows),
        "raw_event_types": dict(sorted(event_types.items())),
        "assistant_model_calls": len(finishes),
        "native_tool_calls": len(tools),
        "tool_call_counts": dict(sorted(Counter(
            str(row.get("tool") or "unknown") for row in tools
        ).items())),
        "tool_status_errors": sum(
            str((row.get("state") or {}).get("status")) != "completed"
            for row in tools
        ),
        "tool_failure_signal_results": sum(
            bool(_FAILURE_SIGNAL.search(value)) for value in outputs
        ),
        "cumulative_logical_input_tokens": sum(logical_inputs),
        "cumulative_uncached_input_tokens": sum(
            int(row.get("input") or 0) for row in token_rows
        ),
        "cumulative_cache_read_tokens": sum(
            int((row.get("cache") or {}).get("read") or 0)
            for row in token_rows
        ),
        "cumulative_output_tokens": sum(
            int(row.get("output") or 0) for row in token_rows
        ),
        "maximum_logical_request_tokens": max(logical_inputs, default=0),
        "token_note": (
            "Kilo usage includes the agent system prompt and tool schemas. "
            "Compare policy savings within Kilo; do not pool these totals "
            "with mini-swe message-content-only counters."
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
