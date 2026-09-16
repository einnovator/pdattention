"""Measure fixed-trajectory headroom from instruction-epoch retirement.

This is an engine-independent logical oracle for the independent-issue
stratum.  It replays every model request from frozen trajectories, preserves
all genuine user instructions, keeps the current instruction epoch whole, and
retires assistant/tool detail from older epochs.  It measures opportunity; it
does not by itself establish autonomous task quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .model import AgentMemoryBudget
from .multi_issue_session import BoundaryMode, compose_multi_issue_session
from .recordizer import recordize_replay_messages
from .selectors import (
    PersistentInstructionEpochRetirementConfig,
    PersistentInstructionEpochRetirementSelector,
    immutable_instruction_record_ids,
    whitespace_tokens,
)


TokenCounter = Callable[[str], int]


def instruction_epoch_curve(
    trajectories: Sequence[Mapping[str, Any]],
    *,
    count_tokens: TokenCounter = whitespace_tokens,
    tokenizer_identity: str = "whitespace_v1_diagnostic",
    config: PersistentInstructionEpochRetirementConfig | None = None,
) -> dict[str, Any]:
    """Return cumulative N=1..K fixed-trajectory opportunity points."""

    composed = compose_multi_issue_session(
        trajectories,
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        session_id="paper8.5:instruction-epoch-oracle",
    )
    selector = PersistentInstructionEpochRetirementSelector(config)
    episode_for_decision: dict[int, int] = {}
    for episode in composed["episodes"]:
        for decision in range(
            int(episode["first_assistant_decision"]),
            int(episode["last_assistant_decision"]) + 1,
        ):
            episode_for_decision[decision] = int(episode["episode_index"])

    episode_rows: dict[int, dict[str, int]] = {
        index: {
            "request_count": 0,
            "full_tokens": 0,
            "selected_tokens": 0,
            "instruction_tokens": 0,
        }
        for index in range(1, len(trajectories) + 1)
    }
    decision = 0
    for message_index, message in enumerate(composed["messages"]):
        if message.get("role") != "assistant":
            continue
        decision += 1
        request_messages = composed["messages"][:message_index]
        history = recordize_replay_messages(request_messages)
        full_tokens = sum(count_tokens(record.content) for record in history.records)
        plan = selector.select(
            history=history,
            query="fixed-trajectory logical oracle",
            budget=AgentMemoryBudget(max_tokens=max(1, full_tokens)),
            count_tokens=count_tokens,
        )
        instruction_tokens = sum(
            count_tokens(history.record_by_id[record_id].content)
            for record_id in immutable_instruction_record_ids(history)
        )
        episode_index = episode_for_decision[decision]
        row = episode_rows[episode_index]
        row["request_count"] += 1
        row["full_tokens"] += full_tokens
        row["selected_tokens"] += plan.selected_tokens
        row["instruction_tokens"] += instruction_tokens

    points = []
    cumulative = {
        "request_count": 0,
        "full_tokens": 0,
        "selected_tokens": 0,
        "instruction_tokens": 0,
    }
    for issue_count in range(1, len(trajectories) + 1):
        row = episode_rows[issue_count]
        for key in cumulative:
            cumulative[key] += row[key]
        full = cumulative["full_tokens"]
        selected = cumulative["selected_tokens"]
        instructions = cumulative["instruction_tokens"]
        non_instruction_full = full - instructions
        non_instruction_selected = selected - instructions
        points.append({
            "issue_count": issue_count,
            "request_count": cumulative["request_count"],
            "full_cumulative_tokens": full,
            "selected_cumulative_tokens": selected,
            "gross_saving_fraction": 0.0 if not full else 1.0 - selected / full,
            "instruction_floor_fraction": 0.0 if not full else instructions / full,
            "assistant_tool_full_tokens": non_instruction_full,
            "assistant_tool_selected_tokens": non_instruction_selected,
            "assistant_tool_saving_fraction": (
                0.0
                if not non_instruction_full
                else 1.0 - non_instruction_selected / non_instruction_full
            ),
            "equal_size_active_only_asymptote": 1.0 - 1.0 / issue_count,
        })

    return {
        "schema_version": 1,
        "study": "paper8_5_instruction_epoch_fixed_trajectory_oracle",
        "claim_scope": "logical opportunity only; not autonomous quality evidence",
        "stratum": "independent_cross_repository",
        "instruction_floor": "system plus every TASK and USER_INPUT record",
        "active_floor": "complete latest genuine-user-instruction epoch",
        "tokenizer_identity": tokenizer_identity,
        "instance_ids": [str(row.get("instance_id") or "") for row in trajectories],
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        help="Optional local Hugging Face tokenizer path for exact content counts.",
    )
    parser.add_argument("--tokenizer-revision", default="unversioned")
    args = parser.parse_args()
    trajectories = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.trajectory
    ]
    count_tokens = whitespace_tokens
    tokenizer_identity = "whitespace_v1_diagnostic"
    if args.tokenizer:
        # Reuse the frozen campaign tokenizer contract without making the
        # logical analysis module depend on transformers at import time.
        from .run_autonomous_swebench import _exact_token_counter

        count_tokens, tokenizer_identity = _exact_token_counter(
            args.tokenizer,
            args.tokenizer_revision,
            allow_whitespace=False,
        )
    result = instruction_epoch_curve(
        trajectories,
        count_tokens=count_tokens,
        tokenizer_identity=tokenizer_identity,
    )
    result["source_trajectories"] = [
        {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in args.trajectory
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "points": result["points"],
    }))


if __name__ == "__main__":
    main()
