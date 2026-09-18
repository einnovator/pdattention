"""Reduce an exact-prefix baseline-conditional persistent-session campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .run_autonomous_multi_issue_campaign import _aggregate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reduce_state(state: Mapping[str, Any]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for cell_id, cell in (state.get("cells") or {}).items():
        episodes = list((cell.get("episodes") or {}).values())
        if cell.get("status") != "complete" or not episodes:
            continue
        if not any(
            bool((episode.get("same_prefix_full_control") or {}).get("required"))
            for episode in episodes
        ):
            continue
        summary = _aggregate(cell, episodes)
        cells.append({
            "cell_id": cell_id,
            "sequence_id": cell.get("sequence_id"),
            "strategy_id": cell.get("strategy_id"),
            "strategy_config_id": cell.get("strategy_config_id"),
            **summary,
        })
    if not cells:
        raise ValueError("no complete embedded exact-prefix campaign cell found")
    return {
        "schema_version": 1,
        "study": "paper8_5_baseline_conditional_campaign_reduction",
        "campaign_id": state.get("campaign_id"),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.campaign_state.read_text(encoding="utf-8"))
    reduced = reduce_state(state)
    reduced["campaign_state_sha256"] = _sha256(args.campaign_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reduced, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "campaign_id": reduced["campaign_id"],
        "cells": len(reduced["cells"]),
    }))


if __name__ == "__main__":
    main()
