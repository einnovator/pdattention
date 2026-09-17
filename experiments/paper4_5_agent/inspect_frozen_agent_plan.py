"""Render and audit a frozen Paper 8.5 plan with an engine tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .frozen_agent_plan import (
    frozen_live_kv_geometry,
    load_frozen_agent_decisions,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    decisions = load_frozen_agent_decisions(
        args.request_replay, args.selection_fixture
    )
    rows = []
    prefix_ids_cache: dict[str, tuple[int, ...]] = {}
    for decision in decisions[: args.requests or None]:
        geometry = frozen_live_kv_geometry(
            tokenizer,
            decision,
            full_retention=args.full_retention,
            prefix_ids_cache=prefix_ids_cache,
        )
        rows.append({
            "request_index": decision.request_index,
            "request_input_sha256": decision.request_input_sha256,
            "prompt_tokens": len(geometry.prompt_ids),
            "resident_source_tokens": len(geometry.source_ids),
            "wire_tail_tokens": len(geometry.wire_tail_ids),
            "selected_kv_tokens": geometry.plan.selected_tokens,
            "realized_retention_fraction": geometry.realized_retention_fraction,
            "has_holes": geometry.plan.has_holes,
            "selection_plan": geometry.plan.to_dict(),
        })
    result = {
        "schema_version": 1,
        "contract": "paper4.5-frozen-agent-plan-tokenizer-audit-v1",
        "tokenizer": args.tokenizer,
        "revision": args.revision,
        "full_retention": args.full_retention,
        "requests": len(rows),
        "all_geometry_valid": len(rows) == min(
            len(decisions), args.requests or len(decisions)
        ),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-replay", type=Path, required=True)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--full-retention", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "tokenizer": result["tokenizer"],
        "requests": result["requests"],
        "all_geometry_valid": result["all_geometry_valid"],
        "retention": [
            row["realized_retention_fraction"] for row in result["rows"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
