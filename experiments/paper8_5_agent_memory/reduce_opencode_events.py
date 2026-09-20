"""Reduce OpenCode's native JSONL into the shared causal-event schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reduce_native_tool_events import reduce_native_tool_events


def reduce_events(
    source: Path,
    output: Path,
    tool_semantics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return reduce_native_tool_events(
        source,
        output,
        agent="opencode",
        tool_semantics=tool_semantics,
        token_note=(
            "OpenCode usage includes its system prompt and native tool schemas. "
            "Compare policy savings within OpenCode; cross-agent comparisons use "
            "fractions and normalized causal actions, not pooled absolute tokens."
        ),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tool-semantics", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.tool_semantics).read_text(encoding="utf-8"))
    semantics = payload.get("tool_semantics_by_name", payload)
    print(json.dumps(
        reduce_events(Path(args.input), Path(args.output), semantics),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
