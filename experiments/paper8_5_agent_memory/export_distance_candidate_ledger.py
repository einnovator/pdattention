"""Export auditable task/tool distance candidates for matched ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from pra_hf.agent_history import OpenAIRecordizer

from .dag import build_frontier_information_flow_dag
from .multi_issue_session import BoundaryMode, compose_multi_issue_session
from .negative_selection import classify_bash_operation
from .recordizer import annotate_minisweagent_messages
from .run_autonomous_swebench import _exact_token_counter, load_persistent_prefix


def run(args: argparse.Namespace) -> dict[str, Any]:
    prior_episodes, prefix_identity = load_persistent_prefix(args.persistent_prefix)
    source = json.loads(args.trajectory.read_text(encoding="utf-8"))
    trajectory = source.get("trajectory", source)
    if not isinstance(trajectory, Mapping):
        raise ValueError("trajectory is not an object")
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or len(messages) < args.message_prefix_length:
        raise ValueError("trajectory does not contain the requested prefix")
    current_messages = [
        dict(row) for row in messages[:args.message_prefix_length]
        if isinstance(row, Mapping)
    ]
    current_episode = {
        "instance_id": str(trajectory.get("instance_id") or "frozen-task"),
        "messages": annotate_minisweagent_messages(current_messages),
        "info": {},
    }
    composed = compose_multi_issue_session(
        (*prior_episodes, current_episode),
        session_id=str(prefix_identity.get("session_id") or "distance-ledger"),
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
    )
    recordization = OpenAIRecordizer().recordize(
        composed["messages"],
        request_metadata={"session_id": prefix_identity.get("session_id")},
    )
    if not recordization.exact:
        raise ValueError("recordization is ambiguous")
    history = recordization.history
    count_tokens, tokenizer_identity = _exact_token_counter(
        str(args.tokenizer), args.tokenizer_revision, allow_whitespace=False
    )
    dag = build_frontier_information_flow_dag(history, recent_user_prompts=1)
    eligible = {row.causal_group_id: row for row in dag.retirement_candidates}
    epoch_for_record = {
        record_id: epoch.epoch_index
        for epoch in dag.epochs for record_id in epoch.record_ids
    }
    latest_epoch = len(dag.epochs) - 1
    turns = sorted(
        (turn for turn in history.turns if turn.complete),
        key=lambda turn: turn.first_message_index,
        reverse=True,
    )
    tool_rows = []
    for distance, turn in enumerate(turns, start=1):
        records = tuple(history.record_by_id[value] for value in turn.record_ids)
        action = next((row for row in records if row.command), None)
        command = action.command if action is not None else None
        candidate = eligible.get(turn.causal_group_id)
        tool_rows.append({
            "tool_call_distance": distance,
            "causal_group_id": turn.causal_group_id,
            "record_ids": list(turn.record_ids),
            "instruction_distance": (
                latest_epoch - epoch_for_record[turn.record_ids[0]]
                if turn.record_ids[0] in epoch_for_record else None
            ),
            "tokens": sum(count_tokens(row.content) for row in records),
            "command": command,
            "operation_class": (
                classify_bash_operation(command).value if command else None
            ),
            "no_path_candidate": candidate is not None,
            "confidence": candidate.confidence.value if candidate else None,
            "resource_ids": list(candidate.resource_ids) if candidate else [],
        })
    task_rows = []
    for epoch in dag.epochs[:-1]:
        groups = {
            group for group in epoch.causal_group_ids
            if group in {turn.causal_group_id for turn in history.turns if turn.complete}
        }
        closed = any(
            bool(history.record_by_id[record_id].metadata.get(
                "protocol_completion_valid"
            ))
            for record_id in epoch.record_ids
        )
        task_rows.append({
            "instruction_distance": latest_epoch - epoch.epoch_index,
            "instruction_record_id": epoch.instruction_record_id,
            "record_ids": list(epoch.record_ids),
            "tokens": sum(
                count_tokens(history.record_by_id[record_id].content)
                for record_id in epoch.record_ids
            ),
            "closed": closed,
            "all_groups_no_path": bool(groups) and groups.issubset(eligible),
            "causal_group_ids": sorted(groups),
        })
    result = {
        "schema_version": 1,
        "study": "paper8_5_distance_candidate_ledger",
        "instance_id": trajectory.get("instance_id"),
        "prefix_identity": prefix_identity,
        "tokenizer": tokenizer_identity,
        "distance_semantics": {
            "instruction": "1=immediately preceding genuine-user prompt interval",
            "tool_call": "1=most recent complete action-observation group",
        },
        "task_candidates": task_rows,
        "tool_call_candidates": tool_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-prefix", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--message-prefix-length", type=int, default=2)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "tasks": len(result["task_candidates"]),
        "tool_calls": len(result["tool_call_candidates"]),
    }, indent=2))


if __name__ == "__main__":
    main()
