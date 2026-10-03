"""Measure mini-swe-agent sensitivity to the age of removed history.

The first stage is a frozen next-action experiment.  It interleaves an exact
FULL request with arms that remove either one closed disconnected instruction
component at task distance D, all such components at distance >= D, or one
complete no-path action--observation group at tool-call distance D.  No arm
receives evaluator task identities.  This is the prerequisite diagnostic for
later autonomous calls-to-solution experiments.
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
from .model_identity import fetch_ollama_model_identity
from .negative_receipts import NegativeRealizationMode
from .negative_selection import classify_bash_operation
from .miniswe_semantics import bash_action_contract
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


def _arm_configs(args: argparse.Namespace, common: Mapping[str, Any]):
    configs: dict[str, AutonomousSelectionConfig] = {
        "FULL": AutonomousSelectionConfig(policy="full", **common),
    }
    for distance in args.task_distance:
        configs[f"TASK_D{distance}"] = AutonomousSelectionConfig(
            policy="distance_conditioning_ablation",
            distance_task_distances=(distance,),
            frontier_allow_heuristic=args.allow_heuristic,
            distance_allow_superseded_unfinished_after=(
                args.allow_superseded_unfinished_after
            ),
            **common,
        )
    for distance in args.task_at_least:
        configs[f"TASK_GE{distance}"] = AutonomousSelectionConfig(
            policy="distance_conditioning_ablation",
            distance_task_at_least=distance,
            frontier_allow_heuristic=args.allow_heuristic,
            distance_allow_superseded_unfinished_after=(
                args.allow_superseded_unfinished_after
            ),
            **common,
        )
    for distance in args.tool_call_distance:
        suffix = (
            f"_W{args.tool_call_window_radius}"
            if args.tool_call_window_radius else ""
        )
        configs[f"TOOL_D{distance}{suffix}"] = AutonomousSelectionConfig(
            policy="distance_conditioning_ablation",
            distance_tool_call_distances=(distance,),
            distance_tool_call_window_radius=args.tool_call_window_radius,
            frontier_allow_heuristic=args.allow_heuristic,
            **common,
        )
    # Mitigation ladder: M1 is active-only atomic retirement; M2/M3 retain
    # one or two preceding natural instruction components.  P1 and W1 test
    # whether a natural protocol or workflow exemplar prevents divergence.
    for prompts in args.frontier_m:
        configs[f"DAG_M{prompts}"] = AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            frontier_recent_user_prompts=prompts,
            frontier_allow_heuristic=args.allow_heuristic,
            **common,
        )
    if args.include_exemplar_arms:
        configs["DAG_M1_P1"] = AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            frontier_recent_user_prompts=1,
            frontier_protocol_exemplars=1,
            frontier_allow_heuristic=args.allow_heuristic,
            **common,
        )
        configs["DAG_M1_W1"] = AutonomousSelectionConfig(
            policy="frontier_dag_retirement",
            frontier_recent_user_prompts=1,
            frontier_workflow_exemplars=1,
            frontier_allow_heuristic=args.allow_heuristic,
            **common,
        )
    return configs


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dry_run and (not args.ollama_tags_url or not args.model_revision):
        raise ValueError(
            "non-dry diagnostics require --ollama-tags-url and --model-revision"
        )
    observed_model_identity = (
        fetch_ollama_model_identity(
            args.ollama_tags_url,
            expected_model=args.model,
            expected_revision=args.model_revision,
        )
        if args.ollama_tags_url and args.model_revision else None
    )
    source = json.loads(args.trajectory.read_text(encoding="utf-8"))
    trajectory = source.get("trajectory", source)
    if not isinstance(trajectory, Mapping):
        raise ValueError("trajectory input is not an object")
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
        session_id=str(prefix_identity.get("session_id") or "distance-ablation"),
        episode_index=len(prior_episodes) + 1,
        completed_recent_turns=0,
        completed_mutation_turns=0,
        completed_verification_turns=0,
        completed_protocol_turns=0,
        boundary_mode=BoundaryMode.BOUNDARY_FREE,
        negative_realization=NegativeRealizationMode.DROP,
        negative_fallback="none",
    )
    configs = _arm_configs(args, common)
    payload = {
        "model": args.model,
        "observed_model_identity": observed_model_identity,
        "messages": frozen_messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": False,
        "max_tokens": args.max_completion_tokens,
    }
    if args.logprobs:
        payload.update({"logprobs": True, "top_logprobs": args.top_logprobs})
    transformed = {
        name: transform_autonomous_payload(
            payload,
            config,
            count_tokens=count_tokens,
            prior_episodes=prior_episodes,
        )
        for name, config in configs.items()
    }
    canonical_digests = {
        row.trace["request_input_sha256"] for row in transformed.values()
    }
    if len(canonical_digests) != 1:
        raise AssertionError("arms do not share one canonical full request")
    arm_summary = {
        name: {
            "selected_tokens": row.plan.selected_tokens,
            "materialized_tokens": row.materialized_tokens,
            "full_tokens": row.plan.full_history_tokens,
            "saving_fraction": 1.0 - (
                row.materialized_tokens / row.plan.full_history_tokens
            ),
            "selected_messages_sha256": row.trace["selected_messages_sha256"],
            "exclusions": [
                {
                    "unit_id": exclusion.causal_group_id,
                    "rule_id": exclusion.rule_id,
                    "classification": exclusion.classification,
                    "record_ids": list(exclusion.record_ids),
                    "excluded_tokens": exclusion.excluded_tokens,
                }
                for exclusion in row.plan.exclusions
            ],
            "exact_noop": not row.plan.exclusions,
        }
        for name, row in transformed.items()
    }
    rows: list[dict[str, Any]] = []
    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    if not args.dry_run:
        names = tuple(configs)
        for repeat in range(1, args.repeats + 1):
            offset = (repeat - 1) % len(names)
            for arm in (*names[offset:], *names[:offset]):
                treatment = transformed[arm]
                base_row = {
                    "repeat": repeat,
                    "arm": arm,
                    "request_sha256": _digest(treatment.payload),
                    "selected_messages_sha256": treatment.trace[
                        "selected_messages_sha256"
                    ],
                    "materialized_tokens": treatment.materialized_tokens,
                    "saving_fraction": arm_summary[arm]["saving_fraction"],
                    "excluded_units": [
                        row["unit_id"] for row in arm_summary[arm]["exclusions"]
                    ],
                }
                try:
                    response = _post(endpoint, treatment.payload, args.timeout_seconds)
                except (OSError, TimeoutError, urllib.error.URLError) as error:
                    rows.append({
                        **base_row,
                        "action_valid": None,
                        "command": None,
                        "command_sha256": None,
                        "operation_class": None,
                        "action_contract": None,
                        "response_content": None,
                        "response_content_sha256": None,
                        "choice_logprobs": None,
                        "transport_error": type(error).__name__,
                    })
                    continue
                choice = response["choices"][0]
                content = str(choice["message"].get("content") or "")
                commands = _COMMAND.findall(content)
                command = commands[0].strip() if len(commands) == 1 else None
                rows.append({
                    **base_row,
                    "action_valid": len(commands) == 1,
                    "command": command,
                    "command_sha256": (
                        hashlib.sha256(command.encode()).hexdigest()
                        if command else None
                    ),
                    "operation_class": (
                        classify_bash_operation(command).value if command else None
                    ),
                    "action_contract": (
                        bash_action_contract(command) if command else None
                    ),
                    "response_content": content,
                    "response_content_sha256": hashlib.sha256(
                        content.encode()
                    ).hexdigest(),
                    "choice_logprobs": choice.get("logprobs"),
                    "reported_usage": response.get("usage"),
                })
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({
                    "schema_version": 1,
                    "study": "paper8_5_distance_conditioning_ablation",
                    "evidence_class": "next_action_diagnostic_not_task_quality",
                    "arms": arm_summary,
                    "rows": rows,
                }, indent=2) + "\n", encoding="utf-8")

    full_by_repeat = {
        row["repeat"]: row for row in rows if row["arm"] == "FULL"
    }
    for row in rows:
        full = full_by_repeat.get(row["repeat"])
        row["diverges_from_repeat_full"] = (
            None if full is None or row["command_sha256"] is None
            or full["command_sha256"] is None
            else row["command_sha256"] != full["command_sha256"]
        )
        row["operation_class_diverges_from_repeat_full"] = (
            None if full is None or row.get("operation_class") is None
            or full.get("operation_class") is None
            else row["operation_class"] != full["operation_class"]
        )
        row["action_contract_diverges_from_repeat_full"] = (
            None if full is None or row.get("action_contract") is None
            or full.get("action_contract") is None
            else row["action_contract"] != full["action_contract"]
        )
    result = {
        "schema_version": 1,
        "study": "paper8_5_distance_conditioning_ablation",
        "evidence_class": "next_action_diagnostic_not_task_quality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": trajectory.get("instance_id"),
        "message_prefix_length": args.message_prefix_length,
        "prefix_identity": prefix_identity,
        "model": args.model,
        "tokenizer": tokenizer_identity,
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "logprobs_requested": args.logprobs,
        },
        "canonical_request_sha256": canonical_digests.pop(),
        "distance_semantics": {
            "task": "1=immediately preceding genuine-user instruction interval",
            "tool_call": "1=most recent complete action-observation turn",
        },
        "arms": arm_summary,
        "repeats": 0 if args.dry_run else args.repeats,
        "rows": rows,
        "guardrail": (
            "Next-action stability localizes conditioning effects. Autonomous "
            "official resolution and calls-to-solution are required before promotion."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-prefix", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="qwen3-coder:30b")
    parser.add_argument("--model-revision")
    parser.add_argument("--ollama-tags-url")
    parser.add_argument("--message-prefix-length", type=int, default=2)
    parser.add_argument("--task-distance", type=int, action="append", default=[])
    parser.add_argument("--task-at-least", type=int, action="append", default=[])
    parser.add_argument("--tool-call-distance", type=int, action="append", default=[])
    parser.add_argument("--tool-call-window-radius", type=int, default=0)
    parser.add_argument("--frontier-m", type=int, action="append", default=[])
    parser.add_argument(
        "--include-exemplar-arms",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-heuristic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--allow-superseded-unfinished-after", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--logprobs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "arms": result["arms"],
        "rows": len(result["rows"]),
    }, indent=2))


if __name__ == "__main__":
    main()
