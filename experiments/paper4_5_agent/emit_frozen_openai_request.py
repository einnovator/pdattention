"""Emit one validated frozen Paper 8.5 plan as a direct OpenAI PRA request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context_treatment import transform_wire_agent_memory_plan_payload
from .frozen_agent_plan import load_frozen_agent_decisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-replay", type=Path, required=True)
    parser.add_argument("--selection-fixture", type=Path, required=True)
    parser.add_argument("--request-index", type=int, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--engine-request-index",
        type=int,
        default=1,
        help="Use one when importing the frozen transcript into a fresh engine session.",
    )
    args = parser.parse_args()

    decisions = load_frozen_agent_decisions(
        args.request_replay.resolve(), args.selection_fixture.resolve()
    )
    if not 1 <= args.request_index <= len(decisions):
        raise ValueError("--request-index is outside the frozen replay")
    decision = decisions[args.request_index - 1]
    messages = list(decision.logical_payload["messages"])
    identities = {
        f"record-{index:06d}": index for index in range(len(messages))
    }
    replacements = {
        row.record_id: row.content
        for row in decision.materialized_message_replacements
    }
    wire_plan = {
        "policy": decision.source_policy,
        "selected_record_ids": [
            f"record-{index:06d}"
            for index in decision.selected_message_indices
        ],
        "record_replacements": replacements,
    }
    payload = dict(decision.logical_payload)
    payload.update({
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "stream": False,
    })
    transformed, trace = transform_wire_agent_memory_plan_payload(
        payload,
        wire_plan=wire_plan,
        record_message_indices=identities,
        mandatory_message_indices=decision.mandatory_message_indices,
        session_id=args.session_id,
        request_index=args.engine_request_index,
        agent_history_selection_policy=decision.source_policy,
    )
    transformed.setdefault("metadata", {})
    transformed["metadata"] = {
        **dict(transformed.get("metadata") or {}),
        "frozen_request_input_sha256": decision.request_input_sha256,
        "frozen_source_request_index": decision.request_index,
        "frozen_source_plan_digest": decision.source_plan_digest,
        "emission_logical_retention_fraction": trace.budget_fraction,
    }
    print(json.dumps(transformed, sort_keys=True))


if __name__ == "__main__":
    main()
