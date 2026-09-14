"""Quality-first gate for repeat-qualified persistent-spine experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .adaptive_spine_bracket import paired_point


def evaluate(
    state: Mapping[str, Any], *, sequence_id: str, strategy_config_id: str,
    required_repeats: int = 2,
) -> dict[str, Any]:
    cells = state.get("cells") or {}
    pairs: list[dict[str, Any]] = []
    missing: list[str] = []
    for repeat in range(1, required_repeats + 1):
        control_id = f"{sequence_id}__S01_persistent_full__r{repeat:02d}"
        candidate_id = (
            f"{sequence_id}__S03_completed_episode_spine-"
            f"{strategy_config_id}__r{repeat:02d}"
        )
        control = cells.get(control_id)
        candidate = cells.get(candidate_id)
        if not control or control.get("status") != "complete":
            missing.append(control_id)
            continue
        if not candidate or candidate.get("status") != "complete":
            missing.append(candidate_id)
            continue
        pairs.append({
            "repeat": repeat,
            "control_cell_id": control_id,
            "candidate_cell_id": candidate_id,
            "control_tokens": int(control["cumulative_full_tokens"]),
            "candidate_selected_tokens": int(
                candidate["cumulative_materialized_tokens"]
            ),
            "control_calls": int(control["calls"]),
            "candidate_calls": int(candidate["calls"]),
            **paired_point(candidate, control),
        })
    if missing:
        return {
            "decision": "wait", "sequence_id": sequence_id,
            "strategy_config_id": strategy_config_id,
            "required_repeats": required_repeats, "missing_cells": missing,
        }
    lost = sum(int(row["lost_persistent_full_successes"]) for row in pairs)
    resolution_delta = sum(
        int(row["resolution_delta_vs_persistent_full"]) for row in pairs
    )
    control_tokens = sum(int(row["control_tokens"]) for row in pairs)
    candidate_tokens = sum(int(row["candidate_selected_tokens"]) for row in pairs)
    aggregate_saving = 1 - candidate_tokens / control_tokens
    call_delta = sum(
        int(row["candidate_calls"]) - int(row["control_calls"]) for row in pairs
    )
    advance = (
        lost == 0 and resolution_delta >= 0
        and 0.30 <= aggregate_saving <= 0.50 and call_delta <= 0
    )
    return {
        "decision": "advance" if advance else "stop_or_rebracket",
        "sequence_id": sequence_id,
        "strategy_config_id": strategy_config_id,
        "required_repeats": required_repeats,
        "pair_count": len(pairs),
        "aggregate_failure_aware_saving_vs_persistent_full": (
            aggregate_saving if lost == 0 else 0.0
        ),
        "lost_persistent_full_successes": lost,
        "resolution_delta_vs_persistent_full": resolution_delta,
        "calls_delta_vs_persistent_full": call_delta,
        "pairs": pairs,
        "scope_note": (
            "repeat-qualified on one ordered sequence; repeats are not "
            "independent task clusters and do not establish population accuracy"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--strategy-config-id", required=True)
    parser.add_argument("--required-repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.state.read_text(encoding="utf-8")),
        sequence_id=args.sequence_id,
        strategy_config_id=args.strategy_config_id,
        required_repeats=args.required_repeats,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if result["decision"] == "advance" else 2)


if __name__ == "__main__":
    main()
