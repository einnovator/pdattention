"""Structural screen for Paper 8.5 negative-selection heuristics.

This command never queries a model and therefore reports opportunity only,
not behavioral quality or task success.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .model import AgentMemoryBudget
from .negative_selection import (
    NEGATIVE_POLICY_RULES,
    NegativeHeuristicSelector,
    NegativeRule,
    NegativeSelectionConfig,
)
from .recordizer import recordize_minisweagent_messages
from .run_structural_screen import _decision_prefixes, _query, _token_counter


DEFAULT_POLICIES = tuple(NEGATIVE_POLICY_RULES)
LONG_HORIZON_SEARCH_DELAYS = (0, 2, 4, 8, 12, 16)
LONG_HORIZON_WRITE_DELAYS = (1, 2, 4, 8, 12)
LONG_HORIZON_READ_KEEPS = (1, 2, 3, 4, 6, 8)
LONG_HORIZON_WORKING_SETS = (2, 3, 4, 6, 8, 12)
LONG_HORIZON_SUFFIXES = (1, 10, 15, 20)


def negative_structural_screen(
    trajectory_paths: Sequence[Path],
    *,
    count_tokens: Callable[[str], int],
    tokenizer_identity: str,
    policies: Sequence[str] = DEFAULT_POLICIES,
    search_delays: Sequence[int] = (0, 1, 2, 4),
    write_delays: Sequence[int] = (1, 2, 4),
    read_keeps: Sequence[int] = (1, 2, 3),
    working_sets: Sequence[int] = (2, 3, 4, 6),
    protected_head_turns: int = 1,
    protected_tail_turns: int = 2,
    decision_suffixes: Sequence[int] = (1,),
    include_decision_rows: bool = False,
) -> dict[str, Any]:
    unknown = set(policies).difference(NEGATIVE_POLICY_RULES)
    if unknown:
        raise ValueError(f"unknown policies: {', '.join(sorted(unknown))}")
    if not decision_suffixes or any(value < 1 for value in decision_suffixes):
        raise ValueError("decision suffixes must contain positive one-based ordinals")
    decision_suffixes = tuple(sorted(set(decision_suffixes)))
    configurations: list[tuple[str, NegativeSelectionConfig]] = []
    for policy in policies:
        rules = NEGATIVE_POLICY_RULES[policy]
        kf_values = (
            search_delays
            if any(rule in {
                NegativeRule.H1_SEARCH_CONSUMED,
                NegativeRule.H1_ALL_BRANCHES_CONSUMED,
            } for rule in rules)
            else (0,)
        )
        kw_values = (
            write_delays if NegativeRule.H2_BARE_AGGRESSIVE in rules else (1,)
        )
        kr_values = (
            read_keeps if NegativeRule.H3_READ_SUPERSEDED in rules else (1,)
        )
        kx_values = (
            working_sets if NegativeRule.H4_WORKING_SET in rules else (4,)
        )
        for kf in kf_values:
            for kw in kw_values:
                for kr in kr_values:
                    for kx in kx_values:
                        treatment = f"{policy}:Kf={kf}:Kw={kw}:Kr={kr}:Kx={kx}"
                        configurations.append((treatment, NegativeSelectionConfig(
                            rules=rules,
                            search_delay_turns=kf,
                            write_delay_turns=kw,
                            same_span_reads_to_keep=kr,
                            working_set_resources=kx,
                            protected_head_turns=protected_head_turns,
                            protected_tail_turns=protected_tail_turns,
                        )))

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in trajectory_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload["messages"]
        instance_id = payload.get("instance_id", path.stem)
        inputs.append({
            "instance_id": instance_id,
            "path": str(path),
            "message_count": len(messages),
        })
        for decision_ordinal, (decision_index, prefix) in enumerate(
            _decision_prefixes(messages), start=1
        ):
            history = recordize_minisweagent_messages(prefix)
            full_tokens = sum(count_tokens(row.content) for row in history.records)
            for treatment, config in configurations:
                plan = NegativeHeuristicSelector(config).select(
                    history=history,
                    query=_query(prefix),
                    budget=AgentMemoryBudget(max_tokens=max(1, full_tokens)),
                    count_tokens=count_tokens,
                )
                by_rule: dict[str, dict[str, int]] = defaultdict(
                    lambda: {"groups": 0, "tokens": 0}
                )
                for exclusion in plan.exclusions:
                    by_rule[exclusion.rule_id]["groups"] += 1
                    by_rule[exclusion.rule_id]["tokens"] += exclusion.excluded_tokens
                rows.append({
                    "instance_id": instance_id,
                    "decision_ordinal": decision_ordinal,
                    "decision_message_index": decision_index,
                    "treatment": treatment,
                    "policy": plan.policy,
                    "full_history_tokens": plan.full_history_tokens,
                    "selected_tokens": plan.selected_tokens,
                    "excluded_tokens": sum(
                        row.excluded_tokens for row in plan.exclusions
                    ),
                    "realized_retention_fraction": plan.realized_retention_fraction,
                    "excluded_group_count": len(plan.exclusions),
                    "excluded_record_count": sum(
                        len(row.record_ids) for row in plan.exclusions
                    ),
                    "excluded_by_rule": dict(by_rule),
                    "inactive_tombstones": [row.tombstone for row in plan.exclusions],
                    "plan_digest": plan.digest,
                })

    aggregates: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "decision_count": 0,
        "full_history_tokens": 0,
        "selected_tokens": 0,
        "excluded_tokens": 0,
        "excluded_group_decisions": 0,
        "decisions_with_exclusion": 0,
    })
    for row in rows:
        aggregate = aggregates[(row["instance_id"], row["treatment"])]
        aggregate["decision_count"] += 1
        for field in ("full_history_tokens", "selected_tokens", "excluded_tokens"):
            aggregate[field] += row[field]
        aggregate["excluded_group_decisions"] += row["excluded_group_count"]
        aggregate["decisions_with_exclusion"] += int(row["excluded_group_count"] > 0)
    summary_rows = []
    for (instance_id, treatment), aggregate in sorted(aggregates.items()):
        full = aggregate["full_history_tokens"]
        summary_rows.append({
            "instance_id": instance_id,
            "treatment": treatment,
            **aggregate,
            "aggregate_retention_fraction": (
                aggregate["selected_tokens"] / full if full else 1.0
            ),
        })
    suffix_aggregates: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "decision_count": 0,
            "full_history_tokens": 0,
            "selected_tokens": 0,
            "excluded_tokens": 0,
            "decisions_with_exclusion": 0,
        }
    )
    for row in rows:
        for suffix_start in decision_suffixes:
            if row["decision_ordinal"] < suffix_start:
                continue
            aggregate = suffix_aggregates[
                (row["instance_id"], row["treatment"], suffix_start)
            ]
            aggregate["decision_count"] += 1
            for field in ("full_history_tokens", "selected_tokens", "excluded_tokens"):
                aggregate[field] += row[field]
            aggregate["decisions_with_exclusion"] += int(
                row["excluded_group_count"] > 0
            )
    suffix_summary_rows = []
    for (instance_id, treatment, suffix_start), aggregate in sorted(
        suffix_aggregates.items()
    ):
        full = aggregate["full_history_tokens"]
        suffix_summary_rows.append({
            "instance_id": instance_id,
            "treatment": treatment,
            "decision_suffix_start": suffix_start,
            **aggregate,
            "aggregate_retention_fraction": (
                aggregate["selected_tokens"] / full if full else 1.0
            ),
        })
    result = {
        "schema_version": 2,
        "study": "paper8_5_negative_heuristic_structural_screen",
        "evidence_class": "structural_opportunity_only_not_behavior_or_task_quality",
        "tokenizer": tokenizer_identity,
        "inputs": inputs,
        "policies": list(policies),
        "parameter_grid": {
            "Kf": list(search_delays),
            "Kw": list(write_delays),
            "Kr": list(read_keeps),
            "Kx": list(working_sets),
            "protected_head_turns": protected_head_turns,
            "protected_tail_turns": protected_tail_turns,
            "decision_suffixes": list(decision_suffixes),
        },
        "reacquisition_metrics_status": (
            "not_measured_requires_candidate_generation_or_autonomous_rollout"
        ),
        "summary_rows": summary_rows,
        "suffix_summary_rows": suffix_summary_rows,
    }
    if include_decision_rows:
        result["decision_rows"] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", nargs="+", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", nargs="+", choices=DEFAULT_POLICIES, default=DEFAULT_POLICIES)
    parser.add_argument("--search-delays", nargs="+", type=int, default=(0, 1, 2, 4))
    parser.add_argument("--write-delays", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--read-keeps", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--working-sets", nargs="+", type=int, default=(2, 3, 4, 6))
    parser.add_argument(
        "--long-horizon",
        action="store_true",
        help=(
            "use the predeclared larger K* grid and report call-10/15/20 "
            "suffixes for long successful trajectories"
        ),
    )
    parser.add_argument("--decision-suffixes", nargs="+", type=int)
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--tail", type=int, default=2)
    parser.add_argument("--include-decision-rows", action="store_true")
    args = parser.parse_args()
    if args.long_horizon:
        args.search_delays = LONG_HORIZON_SEARCH_DELAYS
        args.write_delays = LONG_HORIZON_WRITE_DELAYS
        args.read_keeps = LONG_HORIZON_READ_KEEPS
        args.working_sets = LONG_HORIZON_WORKING_SETS
    decision_suffixes = (
        args.decision_suffixes
        if args.decision_suffixes is not None
        else LONG_HORIZON_SUFFIXES if args.long_horizon else (1,)
    )
    counter, tokenizer_identity = _token_counter(args.tokenizer)
    result = negative_structural_screen(
        args.trajectory,
        count_tokens=counter,
        tokenizer_identity=tokenizer_identity,
        policies=args.policy,
        search_delays=args.search_delays,
        write_delays=args.write_delays,
        read_keeps=args.read_keeps,
        working_sets=args.working_sets,
        protected_head_turns=args.head,
        protected_tail_turns=args.tail,
        decision_suffixes=decision_suffixes,
        include_decision_rows=args.include_decision_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "evidence_class": result["evidence_class"],
        "summary_rows": len(result["summary_rows"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
