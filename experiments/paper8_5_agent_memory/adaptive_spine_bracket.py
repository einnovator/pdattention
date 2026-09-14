"""Recommend the next completed-episode-spine point in the frozen bracket.

The bracket is deliberately quality first.  A candidate is charged against the
paired persistent-FULL token total, and any lost FULL success makes that point
ineligible regardless of its nominal token saving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


CONFIGS = ("r1m1v1_tasks1", "r2m2v2_tasks1", "r4m2v2_tasks1",
           "r6m2v2_tasks1", "r8m2v2_tasks1")


def _cell_id(sequence_id: str, config_id: str) -> str:
    return (
        f"{sequence_id}__S03_completed_episode_spine-{config_id}__r01"
    )


def _lost_successes(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> int:
    candidate_episodes = list((candidate.get("episodes") or {}).values())
    control_episodes = list((control.get("episodes") or {}).values())
    if len(candidate_episodes) != len(control_episodes):
        raise ValueError("candidate/control episode count differs")
    return sum(
        bool(base.get("official_resolved"))
        and not bool(test.get("official_resolved"))
        for test, base in zip(candidate_episodes, control_episodes)
    )


def paired_point(
    candidate: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the discovery metrics used by the adaptive bracket."""

    if candidate.get("status") != "complete" or control.get("status") != "complete":
        raise ValueError("paired point requires complete candidate and control")
    lost = _lost_successes(candidate, control)
    baseline_tokens = int(control.get("cumulative_full_tokens") or 0)
    selected_tokens = int(candidate.get("cumulative_materialized_tokens") or 0)
    if baseline_tokens <= 0:
        raise ValueError("persistent-FULL control has no positive token total")
    raw = 1.0 - selected_tokens / baseline_tokens
    return {
        "raw_saving_vs_persistent_full": raw,
        "failure_aware_saving_vs_persistent_full": 0.0 if lost else raw,
        "lost_persistent_full_successes": lost,
        "resolution_delta_vs_persistent_full": (
            int(candidate.get("official_resolved_count") or 0)
            - int(control.get("official_resolved_count") or 0)
        ),
    }


def persistent_full_controls_complete(
    state: Mapping[str, Any], sequence_id: str, *, required_repeats: int = 2,
) -> bool:
    cells = state.get("cells") or {}
    return all(
        (cells.get(f"{sequence_id}__S01_persistent_full__r{repeat:02d}") or {}).get(
            "status"
        ) == "complete"
        for repeat in range(1, required_repeats + 1)
    )


def recommend(state: Mapping[str, Any], sequence_id: str) -> dict[str, Any]:
    """Choose one next registered point, or return a terminal decision."""

    cells = state.get("cells") or {}
    control_id = f"{sequence_id}__S01_persistent_full__r01"
    control = cells.get(control_id)
    if not control or control.get("status") != "complete":
        return {"decision": "wait", "reason": "persistent-FULL control incomplete"}

    observed: dict[str, dict[str, Any]] = {}
    for config_id in CONFIGS:
        cell = cells.get(_cell_id(sequence_id, config_id))
        if cell and cell.get("status") == "complete":
            observed[config_id] = paired_point(cell, control)

    if "r4m2v2_tasks1" not in observed:
        return {"decision": "run", "strategy_config_id": "r4m2v2_tasks1"}

    current_id = "r4m2v2_tasks1"
    while True:
        point = observed[current_id]
        saving = point["failure_aware_saving_vs_persistent_full"]
        lost = point["lost_persistent_full_successes"]
        if not lost and 0.30 <= saving <= 0.50:
            return {
                "decision": "target_found",
                "strategy_config_id": current_id,
                "point": point,
                "reason": "30-50% paired saving with zero lost FULL successes",
            }

        rank = CONFIGS.index(current_id)
        if lost or saving > 0.50:
            next_rank = rank + 1
            direction = "retain_more"
        else:
            next_rank = rank - 1
            direction = "retain_less"
        if next_rank < 0 or next_rank >= len(CONFIGS):
            return {
                "decision": "bracket_exhausted",
                "strategy_config_id": current_id,
                "point": point,
                "reason": f"no registered configuration remains to {direction}",
            }
        next_id = CONFIGS[next_rank]
        if next_id not in observed:
            return {
                "decision": "run",
                "strategy_config_id": next_id,
                "reason": direction,
                "parent_config_id": current_id,
                "parent_point": point,
            }
        current_id = next_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--controls-complete", action="store_true")
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    if args.controls_complete:
        complete = persistent_full_controls_complete(state, args.sequence_id)
        print(json.dumps({"persistent_full_controls_complete": complete}))
        raise SystemExit(0 if complete else 2)
    decision = recommend(state, args.sequence_id)
    if args.config_only:
        if decision["decision"] != "run":
            raise SystemExit(2)
        print(decision["strategy_config_id"])
    else:
        print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
