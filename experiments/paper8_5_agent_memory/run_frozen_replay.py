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
import tempfile
from typing import Any, Mapping, Sequence
import urllib.request

from .materialization import (
    MaterializationMode,
    ToolObservationMaterializer,
    materialize_plan,
)
from .matched_token_tail import (
    MatchedTokenTailConfig,
    materialize_matched_token_tail,
)
from .model import AgentMemoryBudget
from .negative_selection import (
    NEGATIVE_POLICY_RULES,
    NegativeHeuristicSelector,
    NegativeSelectionConfig,
    reacquired_excluded_resources,
)
from .recordizer import recordize_minisweagent_messages
from .run_structural_screen import _policy_selectors, _query, _token_counter
from .selectors import FullHistorySelector
from .serialization import serialize_materialized_messages


_COMMAND = re.compile(r"```(?:mswea_bash_command|bash)?\s*\n(.*?)\n```", re.DOTALL)


class ReplayGenerationError(RuntimeError):
    """The endpoint returned no valid ordinary-text action for a decision."""


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_digest(value: Any) -> str:
    return _digest(json.dumps(value, sort_keys=True))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace ``path`` without exposing a partial JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(payload), handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _command(text: str) -> str | None:
    matches = _COMMAND.findall(text)
    return matches[0].strip() if len(matches) == 1 else None


def _response_diagnostics(raw: Any) -> tuple[str, str | None, dict[str, Any]]:
    response = raw if isinstance(raw, Mapping) else {}
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    choice = choice if isinstance(choice, Mapping) else {}
    message = choice.get("message")
    message = message if isinstance(message, Mapping) else {}
    content = message.get("content")
    generated = content if isinstance(content, str) else ""
    commands = _COMMAND.findall(generated)
    if not generated.strip():
        failure_reason = "empty_generated_content"
    elif not commands:
        failure_reason = "missing_generated_command"
    elif len(commands) != 1:
        failure_reason = "multiple_generated_commands"
    else:
        failure_reason = None
    diagnostics = {
        "response_id": response.get("id"),
        "response_model": response.get("model"),
        "response_keys": sorted(str(key) for key in response),
        "choice_keys": sorted(str(key) for key in choice),
        "finish_reason": choice.get("finish_reason"),
        "message_keys": sorted(str(key) for key in message),
        "message_role": message.get("role"),
        "message_content_type": type(content).__name__,
        "message_content_chars": len(generated),
        "message_content_sha256": _digest(generated),
        "tool_call_count": (
            len(message.get("tool_calls"))
            if isinstance(message.get("tool_calls"), list) else 0
        ),
        "function_call_present": message.get("function_call") is not None,
        "usage": response.get("usage"),
    }
    return generated, failure_reason, diagnostics


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


def _chat_endpoint(base_url: str) -> str:
    """Accept either a server root or an already-complete OpenAI chat URL."""

    normalized = base_url.rstrip("/")
    suffix = "/v1/chat/completions"
    return normalized if normalized.endswith(suffix) else f"{normalized}{suffix}"


def _selector(
    label: str,
    *,
    head: int,
    tail: int,
    round_up: bool = False,
    search_delay_turns: int = 0,
    write_delay_turns: int = 1,
    same_span_reads_to_keep: int = 1,
    working_set_resources: int = 4,
    negative_fallback: str = "none",
    h2b_allow_workspace_verification: bool = False,
):
    if label == "full":
        return FullHistorySelector()
    selectors = dict(_policy_selectors(head, tail, round_up=round_up))
    if label in NEGATIVE_POLICY_RULES:
        fallback = None
        if negative_fallback != "none":
            fallback_labels = {
                "recency": "middle_recency",
                "lexical": "middle_lexical",
                "spine_lexical": "spine_plus_lexical",
            }
            try:
                fallback = selectors[fallback_labels[negative_fallback]]
            except KeyError as error:
                raise ValueError(
                    "negative_fallback must be none, recency, lexical, or spine_lexical"
                ) from error
        return NegativeHeuristicSelector(
            NegativeSelectionConfig(
                rules=NEGATIVE_POLICY_RULES[label],
                search_delay_turns=search_delay_turns,
                write_delay_turns=write_delay_turns,
                same_span_reads_to_keep=same_span_reads_to_keep,
                working_set_resources=working_set_resources,
                protected_head_turns=head,
                protected_tail_turns=tail,
                h2b_allow_workspace_verification=h2b_allow_workspace_verification,
            ),
            fallback,
        )
    try:
        return selectors[label]
    except KeyError as error:
        choices = ", ".join((
            "full", "matched_token_tail", *selectors, *NEGATIVE_POLICY_RULES,
        ))
        raise ValueError(f"unknown policy {label!r}; choose one of {choices}") from error


def _result(
    *,
    trajectory: Mapping[str, Any],
    model: str,
    tokenizer_identity: str,
    policy: str,
    head: int,
    tail: int,
    budget_fraction: float,
    materialization_mode: MaterializationMode,
    seed: int | None,
    max_output_tokens: int | None,
    reference_replay_digest: str | None,
    matched_budget_source_digest: str | None,
    run_configuration: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    completed_rows = [row for row in rows if row.get("decision_status") == "completed"]
    reference_command_rows = [
        row for row in completed_rows if row.get("reference_command") is not None
    ]
    exact_commands = sum(bool(row["exact_command"]) for row in reference_command_rows)
    failed_row = next(
        (row for row in rows if row.get("decision_status") != "completed"), None
    )
    return {
        "schema_version": 2,
        "study": "paper8_5_agent_memory_frozen_replay",
        "evidence_class": "next_action_not_autonomous_task_quality",
        "instance_id": trajectory.get("instance_id"),
        "model": model,
        "tokenizer": tokenizer_identity,
        "generation": {
            "temperature": 0,
            "seed": seed,
            "max_output_tokens": max_output_tokens,
            "unspecified_parameters_use_endpoint_defaults": True,
        },
        "comparison_reference": (
            "contemporaneous_full_replay"
            if reference_replay_digest is not None else "historical_trajectory"
        ),
        "reference_replay_digest": reference_replay_digest,
        "policy": policy,
        "head_turns": head,
        "tail_turns": tail,
        "budget_fraction": budget_fraction,
        "matched_budget_source_digest": matched_budget_source_digest,
        "materialization_mode": materialization_mode.value,
        "run_configuration": dict(run_configuration),
        "run_configuration_digest": _json_digest(run_configuration),
        "attempted_decisions": len(rows),
        "completed_decisions": len(completed_rows),
        "exact_command_decisions": exact_commands,
        "reference_command_decisions": len(reference_command_rows),
        "exact_command_rate": (
            exact_commands / len(reference_command_rows)
            if reference_command_rows else None
        ),
        "first_command_divergence": next(
            (
                row["decision"] for row in reference_command_rows
                if not row["exact_command"]
            ),
            None,
        ),
        "terminal_failure": (
            {
                "decision": failed_row["decision"],
                "reason": failed_row["failure_reason"],
            }
            if failed_row is not None else None
        ),
        "rows": [dict(row) for row in rows],
    }


def _resume_rows(
    path: Path,
    *,
    run_configuration: Mapping[str, Any],
    assistant_indexes: Sequence[int],
) -> list[dict[str, Any]]:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot safely resume invalid replay artifact {path}") from error
    if saved.get("schema_version") != 2:
        raise ValueError("cannot safely resume a replay artifact without the v2 resume contract")
    saved_configuration = saved.get("run_configuration")
    saved_digest = saved.get("run_configuration_digest")
    if not isinstance(saved_configuration, dict) or saved_digest != _json_digest(saved_configuration):
        raise ValueError("resume artifact run_configuration_digest is invalid")
    expected_digest = _json_digest(run_configuration)
    if saved_digest != expected_digest:
        if saved_configuration.get("reference_replay_digest") != run_configuration.get(
            "reference_replay_digest"
        ):
            raise ValueError("resume artifact reference_replay_digest does not match")
        if saved_configuration.get("matched_budget_source_digest") != run_configuration.get(
            "matched_budget_source_digest"
        ):
            raise ValueError("resume artifact matched_budget_source_digest does not match")
        raise ValueError("resume artifact run configuration does not match")
    rows = saved.get("rows")
    if not isinstance(rows, list) or saved.get("attempted_decisions") != len(rows):
        raise ValueError("resume artifact has inconsistent completed decisions")
    if len(rows) > len(assistant_indexes):
        raise ValueError("resume artifact contains more decisions than this run")
    for offset, row in enumerate(rows):
        decision = offset + 1
        if (
            not isinstance(row, dict)
            or row.get("decision") != decision
            or row.get("trajectory_message_index") != assistant_indexes[offset]
        ):
            raise ValueError("resume artifact decisions are not a contiguous trajectory prefix")
        if row.get("decision_status") != "completed":
            raise ValueError(
                f"resume artifact has a terminal failure at decision {decision}; "
                "use --restart to begin a new attempt"
            )
    if saved.get("completed_decisions") != len(rows):
        raise ValueError("resume artifact has inconsistent completed decisions")
    return [dict(row) for row in rows]


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
    seed: int | None,
    max_output_tokens: int | None,
    api_key: str | None,
    timeout: int,
    reference_replay: Mapping[str, Any] | None = None,
    matched_budget_replay: Mapping[str, Any] | None = None,
    progress_path: Path | None = None,
    restart: bool = False,
    round_up_whole_turns: bool = False,
    search_delay_turns: int = 0,
    write_delay_turns: int = 1,
    same_span_reads_to_keep: int = 1,
    working_set_resources: int = 4,
    negative_fallback: str = "none",
    h2b_allow_workspace_verification: bool = False,
) -> dict[str, Any]:
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if policy == "matched_token_tail" and matched_budget_replay is None:
        raise ValueError("matched_token_tail requires a --matched-budget-replay artifact")
    if policy == "matched_token_tail" and round_up_whole_turns:
        raise ValueError("matched_token_tail is a strict ceiling and cannot round up")
    if (
        policy != "matched_token_tail"
        and materialization_mode == MaterializationMode.MATCHED_TOKEN_TAIL
    ):
        raise ValueError(
            "matched_token_tail materialization is available only through the "
            "matched_token_tail policy"
        )
    messages: Sequence[Mapping[str, Any]] = trajectory["messages"]
    assistant_indexes = [
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
    ]
    if max_decisions is not None:
        assistant_indexes = assistant_indexes[:max_decisions]
    selector = (
        None
        if policy == "matched_token_tail"
        else _selector(
            policy,
            head=head,
            tail=tail,
            round_up=round_up_whole_turns,
            search_delay_turns=search_delay_turns,
            write_delay_turns=write_delay_turns,
            same_span_reads_to_keep=same_span_reads_to_keep,
            working_set_resources=working_set_resources,
            negative_fallback=negative_fallback,
            h2b_allow_workspace_verification=h2b_allow_workspace_verification,
        )
    )
    effective_materialization_mode = (
        MaterializationMode.MATCHED_TOKEN_TAIL
        if policy == "matched_token_tail" else materialization_mode
    )
    materializer = ToolObservationMaterializer(
        mode=materialization_mode,
        threshold_tokens=materialization_threshold_tokens,
    )
    reference_replay_digest = (
        _json_digest(reference_replay) if reference_replay is not None else None
    )
    matched_budget_source_digest = (
        _json_digest(matched_budget_replay)
        if matched_budget_replay is not None else None
    )
    endpoint = _chat_endpoint(base_url)
    run_configuration = {
        "contract": "paper8_5_frozen_replay_resume_v1",
        "trajectory_digest": _json_digest(trajectory),
        "model": model,
        "endpoint": endpoint,
        "policy": policy,
        "head_turns": head,
        "tail_turns": tail,
        "budget_fraction": budget_fraction,
        "tokenizer": tokenizer_identity,
        "materialization_mode": effective_materialization_mode.value,
        "materialization_threshold_tokens": materialization_threshold_tokens,
        "max_decisions": max_decisions,
        "temperature": 0,
        "seed": seed,
        "max_output_tokens": max_output_tokens,
        "timeout": timeout,
        "invalid_generation_policy": "record_and_stop",
        "whole_turn_budget_interpretation": (
            "strict_materialized_token_ceiling"
            if policy == "matched_token_tail"
            else "retention_floor_round_up"
            if round_up_whole_turns else "hard_ceiling_round_down"
        ),
        "matched_budget_field": (
            "materialized_tokens" if policy == "matched_token_tail" else None
        ),
        "reference_replay_digest": reference_replay_digest,
        "matched_budget_source_digest": matched_budget_source_digest,
        "negative_selection": {
            "search_delay_turns_kf": search_delay_turns,
            "write_delay_turns_kw": write_delay_turns,
            "same_span_reads_to_keep_kr": same_span_reads_to_keep,
            "working_set_resources_kx": working_set_resources,
            "positive_fallback": negative_fallback,
            "h2b_allow_workspace_verification": h2b_allow_workspace_verification,
        } if policy in NEGATIVE_POLICY_RULES else None,
    }
    rows: list[dict[str, Any]] = []
    if progress_path is not None and progress_path.exists() and not restart:
        rows = _resume_rows(
            progress_path,
            run_configuration=run_configuration,
            assistant_indexes=assistant_indexes,
        )
    replay_references = {
        int(row["decision"]): row
        for row in (reference_replay or {}).get("rows", ())
    }
    matched_budgets: dict[int, int] = {}
    for row in (matched_budget_replay or {}).get("rows", ()):
        decision = int(row["decision"])
        source_field = (
            "materialized_tokens"
            if policy == "matched_token_tail" else "selected_whole_record_tokens"
        )
        if source_field not in row:
            raise ValueError(
                f"matched budget artifact decision {decision} lacks {source_field}"
            )
        ceiling = int(row[source_field])
        if ceiling <= 0:
            raise ValueError(
                f"matched budget artifact decision {decision} has a non-positive ceiling"
            )
        if decision in matched_budgets:
            raise ValueError(f"matched budget artifact repeats decision {decision}")
        matched_budgets[decision] = ceiling

    def current_result() -> dict[str, Any]:
        return _result(
            trajectory=trajectory,
            model=model,
            tokenizer_identity=tokenizer_identity,
            policy=policy,
            head=head,
            tail=tail,
            budget_fraction=budget_fraction,
            materialization_mode=effective_materialization_mode,
            seed=seed,
            max_output_tokens=max_output_tokens,
            reference_replay_digest=reference_replay_digest,
            matched_budget_source_digest=matched_budget_source_digest,
            run_configuration=run_configuration,
            rows=rows,
        )

    if progress_path is not None and (restart or not progress_path.exists()):
        _atomic_write_json(progress_path, current_result())
    for decision, assistant_index in enumerate(assistant_indexes, start=1):
        if decision <= len(rows):
            continue
        prefix = messages[:assistant_index]
        historical_reference = str(messages[assistant_index].get("content", ""))
        replay_reference = replay_references.get(decision)
        reference = (
            str(replay_reference["generated_content"])
            if replay_reference is not None
            else historical_reference
        )
        history = recordize_minisweagent_messages(prefix)
        full_tokens = sum(count_tokens(record.content) for record in history.records)
        requested_budget = matched_budgets.get(
            decision,
            max(1, math.ceil(full_tokens * budget_fraction)),
        )
        if policy == "matched_token_tail":
            if decision not in matched_budgets:
                raise ValueError(
                    f"matched budget artifact has no ceiling for decision {decision}"
                )
            materialized = materialize_matched_token_tail(
                history,
                max_materialized_tokens=requested_budget,
                config=MatchedTokenTailConfig(
                    tool_observation_threshold_tokens=materialization_threshold_tokens
                ),
                count_tokens=count_tokens,
            )
            plan = materialized.logical_plan
        else:
            assert selector is not None
            plan = selector.select(
                history=history,
                query=_query(prefix),
                budget=AgentMemoryBudget(
                    max_tokens=requested_budget
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
        request_payload: dict[str, Any] = {
            "model": model,
            "messages": selected_messages,
            "temperature": 0,
        }
        if seed is not None:
            request_payload["seed"] = seed
        if max_output_tokens is not None:
            request_payload["max_tokens"] = max_output_tokens
        raw = _post(
            endpoint,
            request_payload,
            api_key=api_key,
            timeout=timeout,
        )
        generated, failure_reason, response_diagnostics = _response_diagnostics(raw)
        reference_command = _command(reference)
        generated_command = _command(generated)
        generated_reacquisition = reacquired_excluded_resources(
            generated_command, plan.exclusions
        )
        reference_reacquisition = reacquired_excluded_resources(
            reference_command, plan.exclusions
        )
        policy_excess_reacquisition = tuple(sorted(
            set(generated_reacquisition).difference(reference_reacquisition)
        ))
        transport_failure = failure_reason == "empty_generated_content"
        rows.append({
            "decision": decision,
            "decision_status": (
                "failed_transport_generation" if transport_failure else "completed"
            ),
            "failure_reason": (
                failure_reason if transport_failure else None
            ),
            "action_valid": failure_reason is None,
            "action_validation_reason": failure_reason,
            "trajectory_message_index": assistant_index,
            "comparison_reference": (
                "contemporaneous_full_replay"
                if replay_reference is not None else "historical_trajectory"
            ),
            "historical_reference_content_sha256": _digest(historical_reference),
            "reference_content_sha256": _digest(reference),
            "generated_content_sha256": _digest(generated),
            "generated_content": generated,
            "exact_content": generated == reference,
            "reference_command": reference_command,
            "generated_command": generated_command,
            "exact_command": (
                reference_command is not None and reference_command == generated_command
            ),
            "full_history_tokens": plan.full_history_tokens,
            "selected_logical_tokens": plan.selected_tokens,
            "selected_whole_record_tokens": plan.selected_tokens,
            "materialized_tokens": materialized.materialized_tokens,
            "requested_budget_tokens": plan.requested_budget_tokens,
            "materialized_token_ceiling": (
                plan.requested_budget_tokens
                if policy == "matched_token_tail" else None
            ),
            "realized_record_retention_fraction": plan.realized_retention_fraction,
            "realized_logical_retention_fraction": plan.realized_retention_fraction,
            "realized_materialized_retention_fraction": (
                materialized.materialized_retention_fraction
            ),
            "mandatory_overflow_tokens": plan.mandatory_overflow_tokens,
            "whole_turn_budget_overshoot_tokens": max(
                0, plan.selected_tokens - plan.requested_budget_tokens
            ),
            "whole_turn_budget_undershoot_tokens": max(
                0, plan.requested_budget_tokens - plan.selected_tokens
            ),
            "materialized_budget_overshoot_tokens": max(
                0, materialized.materialized_tokens - plan.requested_budget_tokens
            ),
            "materialized_budget_unused_tokens": max(
                0, plan.requested_budget_tokens - materialized.materialized_tokens
            ),
            "selected_record_ids": plan.selected_record_ids,
            "selection_reasons": plan.selection_reasons,
            "excluded_group_count": len(plan.exclusions),
            "excluded_record_count": sum(len(row.record_ids) for row in plan.exclusions),
            "excluded_tokens": sum(row.excluded_tokens for row in plan.exclusions),
            "exclusions": [
                {
                    "causal_group_id": row.causal_group_id,
                    "record_ids": row.record_ids,
                    "rule_id": row.rule_id,
                    "classification": row.classification,
                    "reason": row.reason,
                    "resource_ids": row.resource_ids,
                    "witness_record_ids": row.witness_record_ids,
                    "inactive_tombstone": row.tombstone,
                    "excluded_tokens": row.excluded_tokens,
                    "tombstone_in_model_request": False,
                }
                for row in plan.exclusions
            ],
            "generated_reacquired_excluded_resource_ids": generated_reacquisition,
            "full_reference_reacquired_excluded_resource_ids": reference_reacquisition,
            "policy_excess_reacquired_resource_ids": policy_excess_reacquisition,
            "false_exclusion_immediate_reacquisition_proxy": bool(
                policy_excess_reacquisition
            ),
            "materialized_records": [
                {
                    "record_id": row.record_id,
                    "mode": row.mode.value,
                    "original_tokens": row.original_tokens,
                    "materialized_tokens": row.materialized_tokens,
                    "selected_line_spans": row.selected_line_spans,
                    "token_fallback_used": row.token_fallback_used,
                    "omitted_prefix_lines": row.omitted_prefix_lines,
                }
                for row in materialized.records
            ],
            "plan_digest": plan.digest,
            "usage": response_diagnostics["usage"],
            "response_diagnostics": response_diagnostics,
        })
        if progress_path is not None:
            _atomic_write_json(progress_path, current_result())
        if transport_failure:
            raise ReplayGenerationError(
                f"decision {decision} stopped: {failure_reason}; "
                "the failure is recorded and resume requires --restart"
            )
    return current_result()


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
    parser.add_argument("--search-delay-turns", type=int, default=0, metavar="Kf")
    parser.add_argument("--write-delay-turns", type=int, default=1, metavar="Kw")
    parser.add_argument("--same-span-reads-to-keep", type=int, default=1, metavar="Kr")
    parser.add_argument("--working-set-resources", type=int, default=4, metavar="Kx")
    parser.add_argument(
        "--negative-fallback",
        choices=("none", "recency", "lexical", "spine_lexical"),
        default="none",
        help="positive selector applied only after negative exclusion",
    )
    parser.add_argument(
        "--h2b-allow-workspace-verification",
        action="store_true",
        help="relaxed H2b arm; a passing workspace check may retire a write",
    )
    parser.add_argument(
        "--materialization-mode",
        type=MaterializationMode,
        choices=tuple(MaterializationMode),
        default=MaterializationMode.WHOLE_RECORD,
    )
    parser.add_argument("--materialization-threshold-tokens", type=int, default=512)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--round-up-whole-turns",
        action="store_true",
        help=(
            "treat the token target as a minimum retention floor and include "
            "the ranked causal turn that crosses it"
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard an existing progress artifact and begin a new replay attempt",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument(
        "--reference-replay",
        type=Path,
        help="FULL replay artifact used as the contemporaneous comparison reference",
    )
    parser.add_argument(
        "--matched-budget-replay",
        type=Path,
        help=(
            "artifact supplying per-decision ceilings; matched_token_tail uses "
            "its materialized_tokens field"
        ),
    )
    args = parser.parse_args()
    counter, tokenizer_identity = _token_counter(args.tokenizer)
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    reference_replay = (
        json.loads(args.reference_replay.read_text(encoding="utf-8"))
        if args.reference_replay else None
    )
    matched_budget_replay = (
        json.loads(args.matched_budget_replay.read_text(encoding="utf-8"))
        if args.matched_budget_replay else None
    )
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
        reference_replay=reference_replay,
        matched_budget_replay=matched_budget_replay,
        progress_path=args.output,
        restart=args.restart,
        round_up_whole_turns=args.round_up_whole_turns,
        search_delay_turns=args.search_delay_turns,
        write_delay_turns=args.write_delay_turns,
        same_span_reads_to_keep=args.same_span_reads_to_keep,
        working_set_resources=args.working_set_resources,
        negative_fallback=args.negative_fallback,
        h2b_allow_workspace_verification=args.h2b_allow_workspace_verification,
    )
    print(json.dumps({key: result[key] for key in (
        "instance_id", "policy", "completed_decisions", "exact_command_rate",
        "first_command_divergence",
    )}, indent=2))


if __name__ == "__main__":
    main()
