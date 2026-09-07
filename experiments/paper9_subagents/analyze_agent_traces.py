"""Measure exact repeated tool operations in existing coding-agent traces.

The source traces are single-agent Paper 4.5 runs, so this is an opportunity
audit rather than evidence of cross-agent reuse.  It estimates how frequently
ordinary trajectories repeat exact read/search calls that a decomposed harness
could expose to descendants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Mapping


FENCE = re.compile(r"```(?:mswea_bash_command|bash|sh)?\s*\n(.*?)```", re.DOTALL)
READ_PREFIXES = ("cat ", "sed ", "head ", "tail ", "rg ", "grep ", "find ", "ls ")
READ_TOOL_NAMES = {"read", "grep", "glob", "search", "list", "view", "cat"}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _read_like_command(command: str) -> bool:
    normalized = " ".join(command.strip().split()).lower()
    return normalized.startswith(READ_PREFIXES) and not any(marker in normalized for marker in (">", "tee "))


def _from_miniswe(data: Mapping[str, object]) -> list[tuple[str, bool]]:
    calls = []
    for message in data.get("messages", ()):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        for command in FENCE.findall(content):
            normalized = " ".join(command.strip().split())
            calls.append((f"shell:{normalized}", _read_like_command(normalized)))
    return calls


def _from_atif(data: Mapping[str, object]) -> list[tuple[str, bool]]:
    calls = []
    for step in data.get("steps", ()):
        if not isinstance(step, Mapping):
            continue
        for call in step.get("tool_calls", ()) or ():
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("function_name", "unknown")).lower()
            arguments = call.get("arguments", {})
            signature = f"{name}:{_canonical(arguments)}"
            read_like = name in READ_TOOL_NAMES
            if name in {"bash", "shell", "execute"} and isinstance(arguments, Mapping):
                command = str(arguments.get("command", arguments.get("cmd", "")))
                read_like = _read_like_command(command)
            calls.append((signature, read_like))
    return calls


def analyze(path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, Mapping):
        return None
    calls = _from_miniswe(data) if "messages" in data else _from_atif(data)
    read_calls = [signature for signature, read_like in calls if read_like]
    counts = Counter(read_calls)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "trace": str(path),
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tool_calls": len(calls),
        "read_search_calls": len(read_calls),
        "exact_repeated_read_search_calls": repeated,
        "has_exact_repeat": repeated > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for root, directories, names in os.walk(args.input, onerror=lambda _: None):
        # Agent sandboxes may contain broken snapshot links. They are unrelated
        # to the exported trajectories and can be skipped safely.
        directories[:] = [name for name in directories if name not in {"xdg-data", ".git"}]
        for name in names:
            if name.endswith(".traj.json") or name == "trajectory.json":
                files.append(Path(root) / name)
    files = sorted(set(files))
    rows = [row for path in files if (row := analyze(path)) is not None]
    total_reads = sum(int(row["read_search_calls"]) for row in rows)
    total_repeats = sum(int(row["exact_repeated_read_search_calls"]) for row in rows)
    summary = {
        "scope": "single-agent Paper 4.5 trace opportunity audit; not observed cross-agent reuse",
        "trace_count": len(rows),
        "tool_calls": sum(int(row["tool_calls"]) for row in rows),
        "read_search_calls": total_reads,
        "exact_repeated_read_search_calls": total_repeats,
        "exact_repeat_fraction": total_repeats / max(total_reads, 1),
        "traces_with_exact_repeat": sum(bool(row["has_exact_repeat"]) for row in rows),
        "mean_repeats_per_trace": mean(int(row["exact_repeated_read_search_calls"]) for row in rows) if rows else 0,
        "traces": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "traces"}, indent=2))


if __name__ == "__main__":
    main()
