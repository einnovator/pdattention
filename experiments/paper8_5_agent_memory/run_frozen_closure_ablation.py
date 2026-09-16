"""Compare closure-retirement variants on one frozen next-action request.

The experiment isolates a structural question discovered by the oracle audit:
does retaining one terminal finalization bundle per pinned old user instruction
repair the orphaned-prompt defect of E0?  It is intentionally a small,
interleaved next-action diagnostic, not an autonomous task-quality estimate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
import urllib.error
import urllib.request

from .autonomous_proxy import AutonomousSelectionConfig, transform_autonomous_payload
from .materialization import MaterializationMode
from .multi_issue_session import BoundaryMode
from .negative_receipts import NegativeRealizationMode
from .run_autonomous_swebench import _exact_token_counter, load_persistent_prefix


_COMMAND = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _post(endpoint: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("endpoint returned a non-object response")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    messages = trajectory.get("messages")
    if not isinstance(messages, list) or len(messages) < args.message_prefix_length:
        raise ValueError("trajectory does not contain the requested frozen prefix")
    frozen_messages = [dict(row) for row in messages[:args.message_prefix_length]]
    if str(frozen_messages[-1].get("role")) != "user":
        raise ValueError("frozen next-action request must end in a user observation/task")
    prior_episodes, prefix_identity = load_persistent_prefix(args.persistent_prefix)
    count_tokens, tokenizer_identity = _exact_token_counter(
        str(args.tokenizer), args.tokenizer_revision, allow_whitespace=False
    )
    common = dict(
        budget_fraction=1.0,
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        expected_model=args.model,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        max_calls=40,
        max_completion_tokens=args.max_completion_tokens,
        tokenizer_identity=tokenizer_identity,
        task_id=str(trajectory.get("instance_id") or "frozen-task"),
        session_id=str(prefix_identity.get("session_id") or "frozen-closure-ablation"),
        episode_index=len(prior_episodes) + 1,
        completed_recent_turns=0,
        completed_mutation_turns=0,
        completed_verification_turns=0,
        completed_protocol_turns=0,
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        negative_realization=NegativeRealizationMode.DROP,
        negative_fallback="none",
    )
    configs = {
        "FULL": AutonomousSelectionConfig(policy="full", **common),
        "E0": AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            completed_instruction_epochs=0,
            completed_finalization_turns=0,
            **common,
        ),
        "E0_F1": AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            completed_instruction_epochs=0,
            completed_finalization_turns=1,
            **common,
        ),
        "E0_F1C": AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            completed_instruction_epochs=0,
            completed_finalization_turns=1,
            compact_completed_finalizations=True,
            **common,
        ),
        "E0_ATOMIC": AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            completed_instruction_epochs=0,
            completed_finalization_turns=0,
            retire_closed_instructions=True,
            **common,
        ),
        "E1_ATOMIC": AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            completed_instruction_epochs=1,
            completed_finalization_turns=0,
            retire_closed_instructions=True,
            **common,
        ),
        "E2_ATOMIC": AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            completed_instruction_epochs=2,
            completed_finalization_turns=0,
            retire_closed_instructions=True,
            **common,
        ),
        "E2": AutonomousSelectionConfig(
            policy="persistent_instruction_epoch_retirement",
            completed_instruction_epochs=2,
            completed_finalization_turns=0,
            **common,
        ),
    }
    payload = {
        "model": args.model,
        "messages": frozen_messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "max_tokens": args.max_completion_tokens,
    }
    transformed = {
        name: transform_autonomous_payload(
            payload,
            config,
            count_tokens=count_tokens,
            prior_episodes=prior_episodes,
        )
        for name, config in configs.items()
    }
    input_digests = {
        row.trace["request_input_sha256"] for row in transformed.values()
    }
    if len(input_digests) != 1:
        raise AssertionError("arms do not share one canonical full request")

    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    names = tuple(args.arms or configs)
    unknown_arms = set(names).difference(configs)
    if unknown_arms:
        raise ValueError(f"unknown arms: {sorted(unknown_arms)}")
    rows = []
    for repeat in range(1, args.repeats + 1):
        offset = (repeat - 1) % len(names)
        for arm in (*names[offset:], *names[:offset]):
            treatment = transformed[arm]
            try:
                response = _post(endpoint, treatment.payload, args.timeout_seconds)
            except (OSError, TimeoutError, urllib.error.URLError) as error:
                rows.append({
                    "repeat": repeat,
                    "arm": arm,
                    "request_sha256": _digest(treatment.payload),
                    "selected_messages_sha256": treatment.trace["selected_messages_sha256"],
                    "selected_tokens": treatment.plan.selected_tokens,
                    "materialized_tokens": treatment.materialized_tokens,
                    "saving_fraction": 1.0 - (
                        treatment.materialized_tokens / treatment.plan.full_history_tokens
                    ),
                    "selected_message_count": treatment.trace["selected_message_count"],
                    "excluded_group_count": treatment.trace["excluded_causal_group_count"],
                    "action_valid": None,
                    "command": None,
                    "command_sha256": None,
                    "response_content": None,
                    "response_content_sha256": None,
                    "reported_usage": None,
                    "transport_error": type(error).__name__,
                })
                continue
            content = str(response["choices"][0]["message"].get("content") or "")
            commands = _COMMAND.findall(content)
            command = commands[0].strip() if len(commands) == 1 else None
            rows.append({
                "repeat": repeat,
                "arm": arm,
                "request_sha256": _digest(treatment.payload),
                "selected_messages_sha256": treatment.trace["selected_messages_sha256"],
                "selected_tokens": treatment.plan.selected_tokens,
                "materialized_tokens": treatment.materialized_tokens,
                "saving_fraction": 1.0 - (
                    treatment.materialized_tokens / treatment.plan.full_history_tokens
                ),
                "selected_message_count": treatment.trace["selected_message_count"],
                "excluded_group_count": treatment.trace["excluded_causal_group_count"],
                "action_valid": len(commands) == 1,
                "command": command,
                "command_sha256": (
                    hashlib.sha256(command.encode()).hexdigest() if command else None
                ),
                "response_content": content,
                "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                "reported_usage": response.get("usage"),
            })
            partial = {
                "schema_version": 1,
                "study": "paper8_5_frozen_closure_ablation",
                "evidence_class": "next_action_diagnostic_not_task_quality",
                "rows": rows,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "study": "paper8_5_frozen_closure_ablation",
        "evidence_class": "next_action_diagnostic_not_task_quality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": trajectory.get("instance_id"),
        "message_prefix_length": args.message_prefix_length,
        "prefix_identity": prefix_identity,
        "tokenizer": tokenizer_identity,
        "model": args.model,
        "generation": {"temperature": 0.0, "top_p": 1.0, "seed": 0},
        "canonical_request_sha256": input_digests.pop(),
        "arms": {
            name: {
                "selected_tokens": row.plan.selected_tokens,
                "materialized_tokens": row.materialized_tokens,
                "full_tokens": row.plan.full_history_tokens,
                "saving_fraction": 1.0 - (
                    row.materialized_tokens / row.plan.full_history_tokens
                ),
                "selected_message_count": row.trace["selected_message_count"],
                "selected_messages_sha256": row.trace["selected_messages_sha256"],
            }
            for name, row in transformed.items()
        },
        "repeats": args.repeats,
        "rows": rows,
        "guardrail": (
            "This experiment tests next-action validity/stability only. Autonomous "
            "official resolution is required before policy promotion."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-prefix", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--message-prefix-length", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--arm",
        dest="arms",
        action="append",
        choices=(
            "FULL", "E0", "E0_F1", "E0_F1C", "E0_ATOMIC",
            "E1_ATOMIC", "E2_ATOMIC", "E2",
        ),
        help="Run only the named arm; repeat to select multiple arms.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "rows": len(result["rows"]),
        "arms": result["arms"],
    }, indent=2))


if __name__ == "__main__":
    main()
