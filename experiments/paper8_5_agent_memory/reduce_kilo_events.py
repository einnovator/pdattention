"""Reduce Kilo's JSON event stream to auditable causal events and metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reduce_native_tool_events import reduce_native_tool_events
from .typed_tool_semantics import DEFAULT_TOOL_SEMANTICS


def reduce_events(source: Path, output: Path) -> dict[str, Any]:
    return reduce_native_tool_events(
        source,
        output,
        agent="kilo",
        tool_semantics=DEFAULT_TOOL_SEMANTICS,
        token_note=(
            "Kilo usage includes the agent system prompt and tool schemas. "
            "Compare policy savings within Kilo; do not pool these totals "
            "with mini-swe message-content-only counters."
        ),
    )


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
