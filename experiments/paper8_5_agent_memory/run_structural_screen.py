"""Audit head/middle/tail policy geometry on frozen successful trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .model import AgentMemoryBudget
from .dag import (
    DagCertifiedExclusionSelector,
    ExclusionClass,
    build_resource_effect_dag,
)
from .recordizer import recordize_minisweagent_messages
from .selectors import (
    HeadMiddleTailConfig,
    HeadMiddleTailSelector,
    MiddleSelectionStrategy,
    whitespace_tokens,
)


DEFAULT_HEADS = (0, 1, 2, 4)
DEFAULT_TAILS = (1, 2, 3, 5, 10)
DEFAULT_BUDGETS = (1.0, 0.9, 0.75, 0.5, 0.25)


def _token_counter(tokenizer_name: str | None) -> tuple[Callable[[str], int], str]:
    if tokenizer_name is None:
        return whitespace_tokens, "whitespace_v1_structural_only"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False)), tokenizer_name


def _query(messages: Sequence[dict[str, Any]]) -> str:
    task = next(
        (str(row.get("content", "")) for row in messages if row.get("role") == "user"),
        "",
    )
    active = str(messages[-1].get("content", "")) if messages else ""
    return f"{task}\n{active}"


def _decision_prefixes(messages: Sequence[dict[str, Any]]):
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            yield index, messages[:index]


def _policy_selectors(head: int, tail: int):
    yield "middle_none", HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=head, tail_turns=tail,
        middle_strategy=MiddleSelectionStrategy.NONE,
    ))
    yield "middle_recency", HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=head, tail_turns=tail,
        middle_strategy=MiddleSelectionStrategy.RECENCY,
    ))
    yield "progress_spine", HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=head, tail_turns=tail,
        middle_strategy=MiddleSelectionStrategy.NONE,
        source_turns=1,
        mutation_turns=1,
        verification_turns=1,
        progress_turns=1,
        error_turns=1,
    ))
    yield "middle_lexical", HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=head, tail_turns=tail,
        middle_strategy=MiddleSelectionStrategy.LEXICAL,
    ))
    hybrid = HeadMiddleTailSelector(HeadMiddleTailConfig(
        head_turns=head, tail_turns=tail,
        middle_strategy=MiddleSelectionStrategy.HYBRID,
        source_turns=1,
        mutation_turns=1,
        verification_turns=1,
        progress_turns=1,
        error_turns=1,
    ))
    yield "spine_plus_lexical", hybrid
    yield "dag_certified_exclusion", DagCertifiedExclusionSelector(
        protected_head_turns=head, protected_tail_turns=tail,
    )
    yield "dag_certified_plus_lexical", DagCertifiedExclusionSelector(
        hybrid, protected_head_turns=head, protected_tail_turns=tail,
    )


def structural_screen(
    trajectory_paths: Sequence[Path],
    *,
    count_tokens: Callable[[str], int],
    tokenizer_identity: str,
    heads: Sequence[int] = DEFAULT_HEADS,
    tails: Sequence[int] = DEFAULT_TAILS,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    include_decision_rows: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in trajectory_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload["messages"]
        instance_id = payload.get("instance_id", path.stem)
        info = payload.get("info", {})
        if info.get("exit_status") != "Submitted" or not info.get("submission"):
            raise ValueError(f"trajectory is not a successful submitted reference: {path}")
        inputs.append({
            "instance_id": instance_id,
            "path": str(path),
            "message_count": len(messages),
        })
        for decision_turn, prefix in _decision_prefixes(messages):
            history = recordize_minisweagent_messages(prefix)
            dag = build_resource_effect_dag(history)
            certified_groups = {
                row.causal_group_id for row in dag.exclusion_candidates
                if row.classification == ExclusionClass.CERTIFIED_OPERATIONAL_DUPLICATE
            }
            heuristic_groups = {
                row.causal_group_id for row in dag.exclusion_candidates
                if row.classification == ExclusionClass.HEURISTIC_EXACT_DUPLICATE
            }
            superseded_groups = {
                row.causal_group_id for row in dag.exclusion_candidates
                if row.classification == ExclusionClass.SUPERSEDED_CURRENT_STATE
            }
            invalidated_groups = {
                row.causal_group_id for row in dag.exclusion_candidates
                if row.classification == ExclusionClass.INVALIDATED_BY_WRITE
            }
            full_tokens = sum(count_tokens(record.content) for record in history.records)
            if full_tokens == 0:
                continue
            query = _query(prefix)
            for fraction in budgets:
                max_tokens = max(1, math.ceil(full_tokens * fraction))
                for head in heads:
                    for tail in tails:
                        for label, selector in _policy_selectors(head, tail):
                            plan = selector.select(
                                history=history,
                                query=query,
                                budget=AgentMemoryBudget(max_tokens=max_tokens),
                                count_tokens=count_tokens,
                            )
                            rows.append({
                                "instance_id": instance_id,
                                "decision_message_index": decision_turn,
                                "policy": label,
                                "head_turns_requested": head,
                                "tail_turns_requested": tail,
                                "budget_fraction_requested": fraction,
                                "full_history_tokens": plan.full_history_tokens,
                                "selected_tokens": plan.selected_tokens,
                                "realized_retention_fraction": plan.realized_retention_fraction,
                                "mandatory_tokens": plan.mandatory_tokens,
                                "mandatory_overflow_tokens": plan.mandatory_overflow_tokens,
                                "middle_candidate_turns": plan.middle_candidate_turns,
                                "middle_selected_turns": plan.middle_selected_turns,
                                "selected_record_count": len(plan.selected_record_ids),
                                "dag_certified_candidate_groups": len(certified_groups),
                                "dag_heuristic_duplicate_groups": len(heuristic_groups),
                                "dag_superseded_current_state_groups": len(
                                    superseded_groups
                                ),
                                "dag_invalidated_by_write_groups": len(invalidated_groups),
                                "plan_digest": plan.digest,
                            })
    aggregates: dict[tuple[Any, ...], dict[str, Any]] = defaultdict(lambda: {
        "decision_count": 0,
        "full_history_tokens_sum": 0,
        "selected_tokens_sum": 0,
        "mandatory_tokens_sum": 0,
        "mandatory_overflow_decisions": 0,
        "mandatory_overflow_tokens_sum": 0,
        "mandatory_overflow_tokens_max": 0,
        "middle_candidate_turns_sum": 0,
        "middle_selected_turns_sum": 0,
        "dag_certified_candidate_groups_sum": 0,
        "dag_heuristic_duplicate_groups_sum": 0,
        "dag_superseded_current_state_groups_sum": 0,
        "dag_invalidated_by_write_groups_sum": 0,
    })
    for row in rows:
        key = (
            row["instance_id"], row["policy"], row["head_turns_requested"],
            row["tail_turns_requested"], row["budget_fraction_requested"],
        )
        aggregate = aggregates[key]
        aggregate["decision_count"] += 1
        for field in (
            "full_history_tokens", "selected_tokens", "mandatory_tokens",
            "mandatory_overflow_tokens", "middle_candidate_turns",
            "middle_selected_turns", "dag_certified_candidate_groups",
            "dag_heuristic_duplicate_groups",
            "dag_superseded_current_state_groups",
            "dag_invalidated_by_write_groups",
        ):
            aggregate[f"{field}_sum"] += row[field]
        if row["mandatory_overflow_tokens"]:
            aggregate["mandatory_overflow_decisions"] += 1
        aggregate["mandatory_overflow_tokens_max"] = max(
            aggregate["mandatory_overflow_tokens_max"],
            row["mandatory_overflow_tokens"],
        )

    summary_rows = []
    for key, aggregate in sorted(aggregates.items()):
        instance_id, policy, head, tail, fraction = key
        full_sum = aggregate["full_history_tokens_sum"]
        selected_sum = aggregate["selected_tokens_sum"]
        summary_rows.append({
            "instance_id": instance_id,
            "policy": policy,
            "head_turns_requested": head,
            "tail_turns_requested": tail,
            "budget_fraction_requested": fraction,
            **aggregate,
            "aggregate_realized_retention_fraction": (
                selected_sum / full_sum if full_sum else 0.0
            ),
        })

    result = {
        "schema_version": 1,
        "implementation": "paper8_5_agent_memory_v1",
        "study": "paper8_5_agent_memory_structural_screen",
        "evidence_class": "structural_only_not_task_quality",
        "tokenizer": tokenizer_identity,
        "inputs": inputs,
        "head_values": list(heads),
        "tail_values": list(tails),
        "budget_fractions": list(budgets),
        "decision_row_count": len(rows),
        "summary_rows": summary_rows,
    }
    if include_decision_rows:
        result["decision_rows"] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", nargs="+", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-decision-rows", action="store_true")
    args = parser.parse_args()
    counter, tokenizer_identity = _token_counter(args.tokenizer)
    result = structural_screen(
        args.trajectory,
        count_tokens=counter,
        tokenizer_identity=tokenizer_identity,
        include_decision_rows=args.include_decision_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
