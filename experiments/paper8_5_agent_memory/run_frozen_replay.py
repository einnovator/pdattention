"""Query ordinary chat endpoints with selected frozen agent histories.

This runner has no PRA gateway or cache API. The future reference action is
consulted only after generation, so it cannot leak into the selected request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.request

from .materialization import (
    MaterializationMode,
    ToolObservationMaterializer,
    materialize_plan,
)
from .model import AgentMemoryBudget
from .recordizer import recordize_minisweagent_messages
from .run_structural_screen import _policy_selectors, _query, _token_counter
from .selectors import FullHistorySelector
from .serialization import serialize_materialized_messages


_COMMAND = re.compile(r"```(?:mswea_bash_command|bash)?\s*\n(.*?)\n```", re.DOTALL)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _command(text: str) -> str | None:
    match = _COMMAND.search(text)
    return match.group(1).strip() if match else None


def _post(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str | None,
    timeout: int,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _selector(label: str, *, head: int, tail: int):
    if label == "full":
        return FullHistorySelector()
    selectors = dict(_policy_selectors(head, tail))
    try:
        return selectors[label]
    except KeyError as error:
        choices = ", ".join(("full", *selectors))
        raise ValueError(f"unknown policy {label!r}; choose one of {choices}") from error


def replay(
    *,
    trajectory: Mapping[str, Any],
    model: str,
    base_url: str,
    policy: str,
    head: int,
    tail: int,
    budget_fraction: float,
    count_tokens,
    tokenizer_identity: str,
    materialization_mode: MaterializationMode,
    materialization_threshold_tokens: int,
    max_decisions: int | None,
    seed: int,
    max_output_tokens: int,
    api_key: str | None,
    timeout: int,
) -> dict[str, Any]:
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    messages: Sequence[Mapping[str, Any]] = trajectory["messages"]
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if max_decisions is not None:
        assistant_indexes = assistant_indexes[:max_decisions]
    selector = _selector(policy, head=head, tail=tail)
    materializer = ToolObservationMaterializer(
        mode=materialization_mode,
        threshold_tokens=materialization_threshold_tokens,
    )
    rows = []
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    for decision, assistant_index in enumerate(assistant_indexes, start=1):
        prefix = messages[:assistant_index]
        reference = str(messages[assistant_index].get("content", ""))
        history = recordize_minisweagent_messages(prefix)
        full_tokens = sum(count_tokens(record.content) for record in history.records)
        plan = selector.select(
            history=history,
            query=_query(prefix),
            budget=AgentMemoryBudget(
                max_tokens=max(1, math.ceil(full_tokens * budget_fraction))
            ),
            count_tokens=count_tokens,
        )
        materialized = materialize_plan(
            history,
            plan,
            materializer,
            query=_query(prefix),
            count_tokens=count_tokens,
        )
        selected_messages = serialize_materialized_messages(history, materialized)
        raw = _post(endpoint, {
            "model": model,
            "messages": selected_messages,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": seed,
            "max_tokens": max_output_tokens,
        }, api_key=api_key, timeout=timeout)
        generated = str(raw["choices"][0].get("message", {}).get("content") or "")
        reference_command = _command(reference)
        generated_command = _command(generated)
        rows.append({
            "decision": decision,
            "trajectory_message_index": assistant_index,
            "reference_content_sha256": _digest(reference),
            "generated_content_sha256": _digest(generated),
            "exact_content": generated == reference,
            "reference_command": reference_command,
            "generated_command": generated_command,
            "exact_command": (
                reference_command is not None and reference_command == generated_command
            ),
            "full_history_tokens": plan.full_history_tokens,
            "selected_whole_record_tokens": plan.selected_tokens,
            "materialized_tokens": materialized.materialized_tokens,
            "requested_budget_tokens": plan.requested_budget_tokens,
            "realized_record_retention_fraction": plan.realized_retention_fraction,
            "mandatory_overflow_tokens": plan.mandatory_overflow_tokens,
            "selected_record_ids": plan.selected_record_ids,
            "selection_reasons": plan.selection_reasons,
            "plan_digest": plan.digest,
            "usage": raw.get("usage"),
        })
    exact_commands = sum(bool(row["exact_command"]) for row in rows)
    return {
        "schema_version": 1,
        "study": "paper8_5_agent_memory_frozen_replay",
        "evidence_class": "next_action_not_autonomous_task_quality",
        "instance_id": trajectory.get("instance_id"),
        "model": model,
        "tokenizer": tokenizer_identity,
        "generation": {"temperature": 0, "seed": seed},
        "policy": policy,
        "head_turns": head,
        "tail_turns": tail,
        "budget_fraction": budget_fraction,
        "materialization_mode": materialization_mode.value,
        "completed_decisions": len(rows),
        "exact_command_decisions": exact_commands,
        "exact_command_rate": exact_commands / len(rows) if rows else None,
        "first_command_divergence": next(
            (row["decision"] for row in rows if not row["exact_command"]), None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--policy", default="full")
    parser.add_argument("--head", type=int, default=1)
    parser.add_argument("--tail", type=int, default=2)
    parser.add_argument("--budget-fraction", type=float, default=1.0)
    parser.add_argument(
        "--materialization-mode",
        type=MaterializationMode,
        choices=tuple(MaterializationMode),
        default=MaterializationMode.WHOLE_RECORD,
    )
    parser.add_argument("--materialization-threshold-tokens", type=int, default=512)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    counter, tokenizer_identity = _token_counter(args.tokenizer)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    result = replay(
        trajectory=trajectory,
        model=args.model,
        base_url=args.base_url,
        policy=args.policy,
        head=args.head,
        tail=args.tail,
        budget_fraction=args.budget_fraction,
        count_tokens=counter,
        tokenizer_identity=tokenizer_identity,
        materialization_mode=args.materialization_mode,
        materialization_threshold_tokens=args.materialization_threshold_tokens,
        max_decisions=args.max_decisions,
        seed=args.seed,
        max_output_tokens=args.max_output_tokens,
        api_key=os.environ.get(args.api_key_env),
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "instance_id", "policy", "completed_decisions", "exact_command_rate",
        "first_command_divergence",
    )}, indent=2))


if __name__ == "__main__":
    main()
