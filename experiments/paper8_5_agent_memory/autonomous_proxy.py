"""OpenAI-compatible text proxy for autonomous Paper 8.5 treatments.

The proxy owns no model state and knows nothing about K/V caches.  mini-swe-agent
keeps the canonical full trajectory.  For each ordinary chat request this
module derives a logical record plan, sends only the materialized text selected
for that decision, and records auditable content-token and exclusion metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse
import urllib.error
import urllib.request

from .materialization import (
    MaterializationMode,
    ToolObservationMaterializer,
    materialize_prior_finalization_receipts,
    materialize_plan,
)
from .multi_issue_session import BoundaryMode, compose_multi_issue_session
from .matched_token_tail import (
    MatchedTokenTailConfig,
    matched_token_tail_full_floor_record_ids,
    materialize_matched_token_tail,
)
from .dag import DagCertifiedExclusionSelector, FrontierDagRetirementSelector
from .model import AgentMemoryBudget, AgentMemoryPlan, AgentRecordRole
from .negative_selection import (
    BashOperation,
    NEGATIVE_POLICY_RULES,
    NegativeHeuristicSelector,
    NegativeSelectionConfig,
    classify_bash_operation,
    reacquired_excluded_resources,
)
from .negative_receipts import (
    NegativeRealizationMode,
    realize_negative_receipts,
)
from .oracle import restore_causal_groups
from .recordizer import active_task_content, annotate_minisweagent_messages, extract_resource_ids
from pra_hf.agent_history import OpenAIRecordizer
from pra_hf.deployment import PRAEngineCapabilities, PRAWireRequest
from pra_hf.mediation import RequestMediator, WireAgentMemoryPlan
from pra_hf.tool_semantics import ToolExecutionReceipt
from .selectors import (
    FullHistorySelector,
    HeadMiddleTailConfig,
    HeadMiddleTailSelector,
    MiddleSelectionStrategy,
    PersistentEpisodeRetirementConfig,
    PersistentEpisodeRetirementSelector,
    PersistentGlobalRetirementConfig,
    PersistentGlobalRetirementSelector,
    PersistentInstructionEpochRetirementConfig,
    PersistentInstructionEpochRetirementSelector,
    TokenCounter,
    immutable_instruction_record_ids,
    whitespace_tokens,
)
from .serialization import serialize_materialized_messages


_COMMAND = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
AUTONOMOUS_POSITIVE_POLICIES = (
    "head_tail_recency",
    "matched_token_tail",
    "full_structured_observation",
)
AUTONOMOUS_EPISODE_POLICIES = (
    "persistent_episode_retirement",
    "persistent_active_episode",
    "persistent_global_retirement",
    "persistent_instruction_epoch_retirement",
    "frontier_dag_retirement",
)
AUTONOMOUS_DAG_POLICIES = (
    "dag_certified_exclusion",
    "dag_certified_progress_spine",
)
AUTONOMOUS_POLICIES = (
    "full", *AUTONOMOUS_POSITIVE_POLICIES, *AUTONOMOUS_EPISODE_POLICIES,
    *AUTONOMOUS_DAG_POLICIES,
    *NEGATIVE_POLICY_RULES,
)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _query(messages: list[Mapping[str, Any]]) -> str:
    task = active_task_content(messages)
    active = str(messages[-1].get("content", "")) if messages else ""
    return f"{task}\n{active}"


def _native_mandatory_message_indices(
    messages: Sequence[Mapping[str, Any]], *, current_episode_start: int,
) -> tuple[int, ...]:
    """Keep system state and only the incomplete current causal tail inline."""

    system = {index for index, row in enumerate(messages) if row.get("role") == "system"}
    latest_observation = next(
        (
            index for index in range(len(messages) - 1, -1, -1)
            if messages[index].get("role") != "system"
        ),
        len(messages) - 1,
    )
    latest_action = next(
        (
            index
            for index in range(latest_observation - 1, current_episode_start - 1, -1)
            if messages[index].get("role") == "assistant"
        ),
        None,
    )
    tail_start = latest_action if latest_action is not None else latest_observation
    return tuple(sorted({
        *system,
        *(
            index for index in range(tail_start, latest_observation + 1)
            if messages[index].get("role") != "system"
        ),
    }))


def _assistant_command(response_body: bytes) -> str | None:
    content = _assistant_content(response_body)
    if content is None:
        return None
    matches = _COMMAND.findall(content)
    return matches[0].strip() if len(matches) == 1 else None


def _assistant_content(response_body: bytes) -> str | None:
    try:
        payload = json.loads(response_body.decode("utf-8"))
        choices = payload.get("choices") or ()
        message = choices[0].get("message") if choices else {}
        content = message.get("content", "") if isinstance(message, Mapping) else ""
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, IndexError):
        return None
    return content if isinstance(content, str) else None


def _response_usage(response_body: bytes) -> dict[str, int | None]:
    try:
        payload = json.loads(response_body.decode("utf-8"))
        usage = payload.get("usage") or {}
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        usage = {}

    def value(name: str) -> int | None:
        observed = usage.get(name)
        return int(observed) if isinstance(observed, (int, float)) else None

    return {
        "reported_prompt_tokens": value("prompt_tokens"),
        "reported_completion_tokens": value("completion_tokens"),
        "reported_total_tokens": value("total_tokens"),
    }


def join_instrumentation_sidecars(
    messages: list[dict[str, Any]],
    instrumentation_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach stripped observation metadata to a selector-only message copy.

    mini-swe-agent 2.4.6 removes message ``extra`` fields before calling the
    OpenAI endpoint.  The instrumented Docker environment writes an ordered
    receipt for every completed Bash action.  We join only when one receipt
    session has an exact, complete command-digest sequence.  Any missing,
    malformed, or ambiguous state returns the untouched copy, causing guarded
    negative rules to abstain.
    """

    copied = [dict(row) for row in messages]
    assistant_rows: list[tuple[int, str]] = []
    for index, message in enumerate(copied):
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        matches = _COMMAND.findall(content) if isinstance(content, str) else []
        if len(matches) != 1:
            return copied, {
                "status": "assistant_command_unparseable",
                "commands": len(assistant_rows),
                "receipts": 0,
                "joined": 0,
            }
        assistant_rows.append((index, matches[0].strip()))
    if not assistant_rows:
        return copied, {"status": "empty_exact", "commands": 0, "receipts": 0, "joined": 0}
    if instrumentation_root is None:
        return copied, {
            "status": "instrumentation_root_unset",
            "commands": len(assistant_rows),
            "receipts": 0,
            "joined": 0,
        }
    root = Path(instrumentation_root)
    paths = sorted(root.rglob("execution_*.json")) if root.is_dir() else []
    parents = {path.parent.resolve() for path in paths}
    if len(parents) > 1:
        return copied, {
            "status": "ambiguous_receipt_sessions",
            "commands": len(assistant_rows),
            "receipts": len(paths),
            "joined": 0,
        }
    receipts: list[dict[str, Any]] = []
    try:
        for path in paths:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(receipt, Mapping):
                raise ValueError("receipt is not an object")
            receipts.append(dict(receipt))
    except (OSError, json.JSONDecodeError, ValueError):
        return copied, {
            "status": "invalid_receipt",
            "commands": len(assistant_rows),
            "receipts": len(paths),
            "joined": 0,
        }
    receipts.sort(key=lambda row: int(row.get("step", -1)))
    steps = [row.get("step") for row in receipts]
    terminal_submission_without_receipt = False
    if len(assistant_rows) == len(receipts) + 1:
        terminal_index, terminal_command = assistant_rows[-1]
        terminal_submission_without_receipt = (
            terminal_index + 1 < len(copied)
            and copied[terminal_index + 1].get("role") == "exit"
            and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in terminal_command
        )
    if (
        (
            len(receipts) != len(assistant_rows)
            and not terminal_submission_without_receipt
        )
        or steps != list(range(len(receipts)))
        or any(
            row.get("command_sha256") != hashlib.sha256(command.encode()).hexdigest()
            for (_, command), row in zip(assistant_rows, receipts)
        )
    ):
        return copied, {
            "status": "missing_or_mismatched_receipts",
            "commands": len(assistant_rows),
            "receipts": len(receipts),
            "joined": 0,
        }

    pending: list[tuple[int, dict[str, Any]]] = []
    for (assistant_index, _), receipt in zip(assistant_rows, receipts):
        metadata = receipt.get("observation_metadata")
        if not isinstance(metadata, Mapping):
            return copied, {
                "status": "invalid_observation_metadata",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        observation_index = assistant_index + 1
        if (
            observation_index >= len(copied)
            or copied[observation_index].get("role") not in {"user", "tool"}
        ):
            return copied, {
                "status": "missing_paired_observation",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        pending.append((observation_index, dict(metadata)))
    for observation_index, metadata in pending:
        existing = copied[observation_index].get("extra")
        if existing is not None and not isinstance(existing, Mapping):
            return [dict(row) for row in messages], {
                "status": "conflicting_inline_metadata",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        if isinstance(existing, Mapping) and any(
            key in existing and existing[key] != value
            for key, value in metadata.items()
        ):
            return [dict(row) for row in messages], {
                "status": "conflicting_inline_metadata",
                "commands": len(assistant_rows),
                "receipts": len(receipts),
                "joined": 0,
            }
        copied[observation_index]["extra"] = {**dict(existing or {}), **metadata}
    return copied, {
        "status": (
            "exact_terminal_submission"
            if terminal_submission_without_receipt
            else "exact"
        ),
        "commands": len(assistant_rows),
        "receipts": len(receipts),
        "joined": len(pending),
    }


def summarize_openai_tool_receipts(
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate native tool-result coverage by execution receipts.

    OpenAI-tool agents receive receipts through message metadata rather than
    the mini-swe filesystem sidecar. DAG-dependent policies may proceed only
    when every visible tool result has one valid, matching receipt.
    """
    tool_results = 0
    valid_receipts = 0
    missing_call_ids: list[str] = []
    invalid_call_ids: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_results += 1
        call_id = str(message.get("tool_call_id") or "")
        metadata = message.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        receipt = metadata.get("pra_execution_receipt")
        if not isinstance(receipt, Mapping):
            missing_call_ids.append(call_id)
            continue
        if str(receipt.get("tool_call_id") or "") != call_id:
            invalid_call_ids.append(call_id)
            continue
        try:
            ToolExecutionReceipt.from_mapping(
                receipt,
                action_record_id=str(receipt.get("action_record_id") or call_id),
                observation_record_ids=tuple(
                    str(value)
                    for value in receipt.get("observation_record_ids", ())
                ),
            )
        except (TypeError, ValueError):
            invalid_call_ids.append(call_id)
            continue
        valid_receipts += 1
    if tool_results == 0:
        status = "empty_exact"
    elif valid_receipts == tool_results:
        status = "exact"
    else:
        status = "missing_or_invalid_execution_receipts"
    return {
        "status": status,
        "tool_results": tool_results,
        "receipts": valid_receipts,
        "joined": valid_receipts,
        "missing_tool_call_ids": missing_call_ids,
        "invalid_tool_call_ids": invalid_call_ids,
    }


@dataclass(frozen=True)
class AutonomousSelectionConfig:
    """Frozen policy and generation contract for one autonomous task arm."""

    policy: str = "full"
    budget_fraction: float = 1.0
    protected_head_turns: int = 1
    protected_tail_turns: int = 1
    search_delay_turns: int = 0
    write_delay_turns: int = 1
    same_span_reads_to_keep: int = 1
    working_set_resources: int = 4
    h2b_allow_workspace_verification: bool = False
    materialization_mode: MaterializationMode = MaterializationMode.WHOLE_RECORD
    materialization_threshold_tokens: int = 512
    materialization_head_lines: int = 20
    materialization_tail_lines: int = 30
    materialization_match_context_lines: int = 4
    materialization_max_matched_lines: int = 32
    matched_tail_boundary_compaction: str = "legacy"
    expected_model: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    max_calls: int = 40
    max_completion_tokens: int | None = None
    tokenizer_identity: str = "whitespace_v1_diagnostic"
    task_id: str = "unassigned"
    session_id: str | None = None
    episode_index: int = 1
    completed_recent_turns: int = 1
    completed_mutation_turns: int = 1
    completed_verification_turns: int = 1
    completed_protocol_turns: int = 0
    completed_finalization_turns: int = 0
    compact_completed_finalizations: bool = False
    retire_closed_instructions: bool = False
    completed_instruction_epochs: int = 0
    frontier_recent_user_prompts: int = 2
    frontier_protocol_exemplars: int = 0
    frontier_workflow_exemplars: int = 0
    frontier_allow_heuristic: bool = False
    keep_completed_task_statements: bool = True
    boundary_mode: BoundaryMode = BoundaryMode.EXPLICIT
    require_exact_sidecars: bool = True
    negative_realization: NegativeRealizationMode = NegativeRealizationMode.DROP
    negative_fallback: str = "none"
    input_protocol: str = "mini_swe_bash"
    tool_semantics_by_name: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    fill_missing_generation_parameters: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_mode", BoundaryMode(self.boundary_mode))
        object.__setattr__(
            self,
            "tool_semantics_by_name",
            {
                str(name): dict(value)
                for name, value in self.tool_semantics_by_name.items()
            },
        )
        if self.input_protocol not in {"mini_swe_bash", "openai_tools"}:
            raise ValueError(
                "input_protocol must be mini_swe_bash or openai_tools"
            )
        if self.matched_tail_boundary_compaction not in {
            "legacy", "declared_safe", "disabled",
        }:
            raise ValueError(
                "matched_tail_boundary_compaction must be legacy, "
                "declared_safe, or disabled"
            )
        if self.policy not in AUTONOMOUS_POLICIES:
            raise ValueError(
                f"policy must be one of {AUTONOMOUS_POLICIES}"
            )
        if not 0 < self.budget_fraction <= 1:
            raise ValueError("budget_fraction must be in (0, 1]")
        if self.protected_head_turns < 0:
            raise ValueError("protected_head_turns cannot be negative")
        if self.protected_tail_turns < 1:
            raise ValueError("at least one tail turn is required to preserve current state")
        if self.max_calls < 1:
            raise ValueError("max_calls must be positive")
        if self.episode_index < 1:
            raise ValueError("episode_index must be positive")
        if (
            self.policy in {
                "persistent_global_retirement",
                "persistent_instruction_epoch_retirement",
                "frontier_dag_retirement",
            }
            and self.boundary_mode is not BoundaryMode.BOUNDARY_FREE
        ):
            raise ValueError(
                f"{self.policy} requires boundary_free composition"
            )
        if any(value < 0 for value in (
            self.completed_recent_turns,
            self.completed_mutation_turns,
            self.completed_verification_turns,
            self.completed_protocol_turns,
            self.completed_finalization_turns,
            self.completed_instruction_epochs,
        )):
            raise ValueError("completed-episode turn floors cannot be negative")
        if self.frontier_recent_user_prompts < 1:
            raise ValueError("frontier_recent_user_prompts must be positive")
        if self.frontier_protocol_exemplars < 0:
            raise ValueError("frontier_protocol_exemplars cannot be negative")
        if self.frontier_workflow_exemplars < 0:
            raise ValueError("frontier_workflow_exemplars cannot be negative")
        if self.policy == "frontier_dag_retirement" and self.budget_fraction != 1.0:
            raise ValueError(
                "frontier_dag_retirement is a reachability policy and requires "
                "a 100% pre-retirement budget"
            )
        if self.max_completion_tokens is not None and self.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive")
        if self.policy == "full" and self.materialization_mode != MaterializationMode.WHOLE_RECORD:
            raise ValueError("FULL is an exact ordinary-text control and cannot compact records")
        if self.policy == "full" and self.negative_realization != NegativeRealizationMode.DROP:
            raise ValueError("FULL cannot use a negative-selection realization")
        if self.compact_completed_finalizations and (
            self.policy != "persistent_instruction_epoch_retirement"
            or self.completed_finalization_turns < 1
        ):
            raise ValueError(
                "compact_completed_finalizations requires instruction-epoch "
                "retirement with a positive finalization floor"
            )
        if self.retire_closed_instructions and self.policy != "persistent_instruction_epoch_retirement":
            raise ValueError(
                "retire_closed_instructions requires instruction-epoch retirement"
            )
        if self.policy == "full_structured_observation":
            if self.budget_fraction != 1.0:
                raise ValueError(
                    "full_structured_observation requires a 100% logical-history budget"
                )
            if self.materialization_mode != MaterializationMode.TOOL_STRUCTURED_EVIDENCE:
                raise ValueError(
                    "full_structured_observation requires tool_structured_evidence"
                )
        if self.policy in AUTONOMOUS_POSITIVE_POLICIES and self.negative_realization != NegativeRealizationMode.DROP:
            raise ValueError("positive-only policies cannot use a negative-selection realization")
        if self.negative_fallback not in {"none", "recency"}:
            raise ValueError("negative_fallback must be none or recency")
        if self.policy not in NEGATIVE_POLICY_RULES and self.negative_fallback != "none":
            raise ValueError("negative_fallback requires a negative-selection policy")
        if (
            self.policy in {"task_aware_progress_spine_v4", *AUTONOMOUS_DAG_POLICIES}
            and self.negative_realization not in {
                NegativeRealizationMode.OBSERVATION_RECEIPT,
                NegativeRealizationMode.PROTOCOL_STUB,
            }
        ):
            raise ValueError(
                f"{self.policy} requires observation_receipt or "
                "protocol_stub realization"
            )

    def selector(self):
        if self.policy == "full":
            return FullHistorySelector()
        if self.policy == "full_structured_observation":
            return FullHistorySelector()
        if self.policy == "persistent_episode_retirement":
            return PersistentEpisodeRetirementSelector(
                PersistentEpisodeRetirementConfig(
                    recent_turns=self.completed_recent_turns,
                    mutation_turns=self.completed_mutation_turns,
                    verification_turns=self.completed_verification_turns,
                    protocol_turns=self.completed_protocol_turns,
                    keep_completed_task_statements=(
                        self.keep_completed_task_statements
                    ),
                )
            )
        if self.policy == "persistent_active_episode":
            return PersistentEpisodeRetirementSelector(
                PersistentEpisodeRetirementConfig(
                    recent_turns=0,
                    mutation_turns=0,
                    verification_turns=0,
                    protocol_turns=0,
                    keep_completed_task_statements=False,
                )
            )
        if self.policy == "persistent_global_retirement":
            return PersistentGlobalRetirementSelector(
                PersistentGlobalRetirementConfig(
                    recent_turns=self.completed_recent_turns,
                    mutation_turns=self.completed_mutation_turns,
                    verification_turns=self.completed_verification_turns,
                    protocol_turns=self.completed_protocol_turns,
                )
            )
        if self.policy == "persistent_instruction_epoch_retirement":
            return PersistentInstructionEpochRetirementSelector(
                PersistentInstructionEpochRetirementConfig(
                    prior_recent_turns=self.completed_recent_turns,
                    prior_mutation_turns=self.completed_mutation_turns,
                    prior_verification_turns=self.completed_verification_turns,
                    prior_protocol_turns=self.completed_protocol_turns,
                    prior_finalization_turns=self.completed_finalization_turns,
                    prior_full_epochs=self.completed_instruction_epochs,
                    retire_closed_instructions=self.retire_closed_instructions,
                    keep_completed_task_statements=(
                        self.keep_completed_task_statements
                    ),
                )
            )
        if self.policy == "frontier_dag_retirement":
            return FrontierDagRetirementSelector(
                recent_user_prompts=self.frontier_recent_user_prompts,
                allow_heuristic=self.frontier_allow_heuristic,
                valid_protocol_exemplars=self.frontier_protocol_exemplars,
                valid_workflow_exemplars=self.frontier_workflow_exemplars,
            )
        if self.policy == "head_tail_recency":
            return HeadMiddleTailSelector(HeadMiddleTailConfig(
                head_turns=self.protected_head_turns,
                tail_turns=self.protected_tail_turns,
                middle_strategy=MiddleSelectionStrategy.RECENCY,
                round_up_to_budget=True,
            ))
        if self.policy == "matched_token_tail":
            raise RuntimeError(
                "matched_token_tail is realized directly under a materialized-token ceiling"
            )
        if self.policy == "dag_certified_exclusion":
            return DagCertifiedExclusionSelector(
                protected_head_turns=self.protected_head_turns,
                protected_tail_turns=self.protected_tail_turns,
            )
        if self.policy == "dag_certified_progress_spine":
            fallback = HeadMiddleTailSelector(HeadMiddleTailConfig(
                head_turns=self.protected_head_turns,
                tail_turns=self.protected_tail_turns,
                middle_strategy=MiddleSelectionStrategy.RECENCY,
                source_turns=1,
                mutation_turns=1,
                verification_turns=1,
                progress_turns=1,
                error_turns=1,
                round_up_to_budget=True,
            ))
            return DagCertifiedExclusionSelector(
                fallback,
                protected_head_turns=self.protected_head_turns,
                protected_tail_turns=self.protected_tail_turns,
            )
        fallback = None
        if self.negative_fallback == "recency":
            fallback = HeadMiddleTailSelector(HeadMiddleTailConfig(
                head_turns=self.protected_head_turns,
                tail_turns=self.protected_tail_turns,
                middle_strategy=MiddleSelectionStrategy.RECENCY,
                mutation_turns=1,
                verification_turns=1,
                progress_turns=1,
                error_turns=1,
                round_up_to_budget=True,
            ))
        return NegativeHeuristicSelector(NegativeSelectionConfig(
            rules=NEGATIVE_POLICY_RULES[self.policy],
            search_delay_turns=self.search_delay_turns,
            write_delay_turns=self.write_delay_turns,
            same_span_reads_to_keep=self.same_span_reads_to_keep,
            working_set_resources=self.working_set_resources,
            protected_head_turns=self.protected_head_turns,
            protected_tail_turns=self.protected_tail_turns,
            h2b_allow_workspace_verification=self.h2b_allow_workspace_verification,
        ), fallback)


@dataclass(frozen=True)
class AutonomousTransformation:
    payload: dict[str, Any]
    plan: AgentMemoryPlan
    materialized_tokens: int
    trace: dict[str, Any]
    logical_payload: dict[str, Any]
    wire_plan: dict[str, Any]
    record_message_indices: dict[str, int]
    mandatory_message_indices: tuple[int, ...]


def transform_autonomous_payload(
    payload: Mapping[str, Any],
    config: AutonomousSelectionConfig,
    *,
    count_tokens: TokenCounter = whitespace_tokens,
    instrumentation_root: Path | None = None,
    prior_episodes: Sequence[Mapping[str, Any]] = (),
    oracle_addback_causal_group_ids: Sequence[str] = (),
) -> AutonomousTransformation:
    """Apply one logical policy without mutating the caller's full history."""

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("chat request must contain a messages list")
    if bool(payload.get("stream")) and config.input_protocol == "mini_swe_bash":
        raise ValueError("streaming mini-swe chat is not supported by this audit proxy")
    incoming_messages = [dict(row) for row in raw_messages if isinstance(row, Mapping)]
    if len(incoming_messages) != len(raw_messages):
        raise ValueError("every chat message must be a JSON object")
    if config.input_protocol == "mini_swe_bash":
        current_selector_messages, sidecar_join = join_instrumentation_sidecars(
            incoming_messages, instrumentation_root
        )
    else:
        current_selector_messages = incoming_messages
        sidecar_join = summarize_openai_tool_receipts(current_selector_messages)
    current_episode_start = 0
    if prior_episodes:
        if config.input_protocol != "mini_swe_bash":
            raise ValueError(
                "persistent prefix composition for openai_tools must use the "
                "agent-native session rather than mini-swe episode exports"
            )
        current_episode = {
            "instance_id": config.task_id,
            "messages": current_selector_messages,
            "info": {},
        }
        composed = compose_multi_issue_session(
            (*prior_episodes, current_episode),
            session_id=config.session_id,
            boundary_mode=config.boundary_mode,
        )
        if composed["issue_count"] != config.episode_index:
            raise ValueError(
                "persistent prefix episode count does not match episode_index"
            )
        typed_selector_messages = composed["messages"]
        messages = [
            {"role": str(row.get("role", "")), "content": str(row.get("content", ""))}
            for row in typed_selector_messages
        ]
        current_visible_count = len(current_episode["messages"])
        if (
            current_episode["messages"]
            and current_episode["messages"][0].get("role") == "system"
        ):
            current_visible_count -= 1
        current_episode_start = len(messages) - current_visible_count
    else:
        messages = incoming_messages
        typed_selector_messages = (
            annotate_minisweagent_messages(current_selector_messages)
            if config.input_protocol == "mini_swe_bash"
            else current_selector_messages
        )
    recordization = OpenAIRecordizer().recordize(
        typed_selector_messages,
        request_metadata={
            "session_id": config.session_id or config.task_id,
            "tool_semantics_by_name": config.tool_semantics_by_name,
        },
    )
    if not recordization.exact:
        raise AssertionError(
            f"{config.input_protocol} recordization is ambiguous: "
            + ", ".join(recordization.ambiguity_reasons)
        )
    history = recordization.history
    full_tokens = sum(count_tokens(row.content) for row in history.records)
    budget_tokens = max(1, math.ceil(full_tokens * config.budget_fraction))
    sidecar_exact = sidecar_join["status"] in {
        "exact",
        "empty_exact",
        "exact_terminal_submission",
    }
    sidecar_dependent = bool(
        config.policy in {
            *AUTONOMOUS_DAG_POLICIES,
            *NEGATIVE_POLICY_RULES,
            "frontier_dag_retirement",
        }
    )
    selection_abstained = bool(
        sidecar_dependent and config.require_exact_sidecars and not sidecar_exact
    )
    matched_tail = config.policy == "matched_token_tail" and not selection_abstained
    matched_tail_config = MatchedTokenTailConfig(
        protected_head_turns=config.protected_head_turns,
        protected_tail_turns=config.protected_tail_turns,
        boundary_compaction=config.matched_tail_boundary_compaction,
    )
    mandatory_ids = (
        matched_token_tail_full_floor_record_ids(history, matched_tail_config)
        if matched_tail
            else immutable_instruction_record_ids(history)
        )
    mandatory_tokens = sum(
        count_tokens(history.record_by_id[record_id].content)
        for record_id in mandatory_ids
    )
    mandatory_overflow_tokens = max(0, mandatory_tokens - budget_tokens)
    oracle_addback = None
    if matched_tail:
        materialized = materialize_matched_token_tail(
            history,
            max_materialized_tokens=max(budget_tokens, mandatory_tokens),
            config=matched_tail_config,
            count_tokens=count_tokens,
        )
        mandatory_tokens = materialized.logical_plan.mandatory_tokens
        mandatory_overflow_tokens = max(0, mandatory_tokens - budget_tokens)
        plan = replace(
            materialized.logical_plan,
            requested_budget_tokens=budget_tokens,
            mandatory_tokens=mandatory_tokens,
            mandatory_overflow_tokens=mandatory_overflow_tokens,
        )
        materialized = replace(materialized, logical_plan=plan)
    else:
        selector = FullHistorySelector() if selection_abstained else config.selector()
        plan = selector.select(
            history=history,
            query=_query(typed_selector_messages),
            budget=AgentMemoryBudget(max_tokens=budget_tokens),
            count_tokens=count_tokens,
        )
        if oracle_addback_causal_group_ids:
            if config.policy not in {
                "persistent_episode_retirement",
                "persistent_instruction_epoch_retirement",
            }:
                raise ValueError(
                    "autonomous oracle add-back is restricted to "
                    "persistent retirement diagnostics"
                )
            plan, oracle_addback = restore_causal_groups(
                history=history,
                plan=plan,
                causal_group_ids=oracle_addback_causal_group_ids,
                count_tokens=count_tokens,
            )
        materializer = ToolObservationMaterializer(
            mode=config.materialization_mode,
            threshold_tokens=config.materialization_threshold_tokens,
            head_lines=config.materialization_head_lines,
            tail_lines=config.materialization_tail_lines,
            match_context_lines=config.materialization_match_context_lines,
            max_matched_lines=config.materialization_max_matched_lines,
        )
        materialized = materialize_plan(
            history,
            plan,
            materializer,
            query=_query(typed_selector_messages),
            count_tokens=count_tokens,
        )
        if config.compact_completed_finalizations:
            materialized = materialize_prior_finalization_receipts(
                history,
                materialized,
                count_tokens=count_tokens,
            )
    retention_floor = bool(
        config.policy in {"head_tail_recency", "dag_certified_progress_spine"}
        or config.negative_fallback == "recency"
    )
    certified_exclusion_tokens = (
        sum(row.excluded_tokens for row in plan.exclusions)
        if config.policy == "dag_certified_progress_spine" else 0
    )
    certified_exclusion_underfill_tokens = (
        min(
            max(0, budget_tokens - plan.selected_tokens),
            certified_exclusion_tokens,
        )
        if retention_floor else 0
    )
    unexplained_floor_underfill_tokens = (
        max(
            0,
            budget_tokens
            - plan.selected_tokens
            - certified_exclusion_underfill_tokens,
        )
        if retention_floor else 0
    )
    matched_tail_unexplained_overflow_tokens = (
        max(
            0,
            materialized.materialized_tokens
            - budget_tokens
            - mandatory_overflow_tokens,
        )
        if matched_tail else 0
    )
    receipt_realization = None
    if config.negative_realization in {
        NegativeRealizationMode.OBSERVATION_RECEIPT,
        NegativeRealizationMode.PROTOCOL_STUB,
    }:
        receipt_realization = realize_negative_receipts(
            history,
            plan,
            materialized,
            count_tokens=count_tokens,
            include_semantic_evidence=(
                config.negative_realization
                == NegativeRealizationMode.OBSERVATION_RECEIPT
            ),
        )
        materialized = receipt_realization.materialized

    all_record_ids = tuple(row.record_id for row in history.records)
    exact_logical_noop = (
        tuple(plan.selected_record_ids) == all_record_ids
        and all(
            row.mode == MaterializationMode.WHOLE_RECORD
            and row.content == history.record_by_id[row.record_id].content
            for row in materialized.records
        )
    )
    wire_plan = WireAgentMemoryPlan(
        schema_version=1,
        policy=config.policy,
        selected_record_ids=tuple(row.record_id for row in materialized.records),
        record_replacements={
            row.record_id: row.content
            for row in materialized.records
            if row.content != history.record_by_id[row.record_id].content
        },
        source_history_digest=history.digest,
        decision_metadata=(
            {
                "instruction_floor": "newest_user_instruction",
                "prior_instruction_retirement": "terminal_epoch_atomic",
            }
            if (
                config.retire_closed_instructions
                and not config.keep_completed_task_statements
            ) else
            {"instruction_floor": "all_user_instructions"}
        ),
    )
    # FULL is the behavioral control.  A negative policy that currently has
    # nothing to remove must be the same control too: retain every incoming
    # message dictionary instead of silently changing the request envelope by
    # round-tripping through the logical serializer.  Once a real exclusion or
    # detail reduction occurs, emit only ordinary role/content records so
    # selector metadata never leaks into the model prompt.
    if config.policy == "full" or exact_logical_noop:
        selected_messages = messages
    else:
        # Realize the frozen logical plan through the same common mediator used
        # by external gateways and embedded runtimes.  The mini-swe adapter is
        # used only to create typed input; plan consumption is agent-neutral.
        mediated_request = PRAWireRequest(
            model=str(payload.get("model") or config.expected_model or "model"),
            messages=tuple(typed_selector_messages),
            session_id=config.session_id or config.task_id,
            metadata={
                "agent_memory_plan": wire_plan.to_dict(),
                "tool_semantics_by_name": config.tool_semantics_by_name,
            },
        )
        mediated = RequestMediator({
            "location": "embedded",
            "mode": "auto",
            "history_selection": {
                "mode": "active",
                "policy": config.policy,
            },
            "result_compaction": (
                "active" if wire_plan.record_replacements else "off"
            ),
            "tool_disclosure": "off",
        }).prepare(
            mediated_request,
            PRAEngineCapabilities(
                adapter="paper8.5-logical-consumer",
                integration_level="E1",
                logical_refs=True,
                typed_records=True,
            ),
        )
        if config.input_protocol == "openai_tools":
            # Preserve tool_calls, tool_call_id, names, and provider-specific
            # protocol fields.  The shared mediator already copied the native
            # message dictionary while applying the frozen record plan.
            selected_messages = [dict(row) for row in mediated.request.messages]
        else:
            selected_messages = [
                {
                    "role": str(row.get("role", "")),
                    "content": str(row.get("content", "")),
                }
                for row in mediated.request.messages
            ]
            legacy_projection = serialize_materialized_messages(history, materialized)
            if selected_messages != legacy_projection:
                raise AssertionError("shared mediator and Paper 8.5 serializer disagree")
    selected_ids = set(plan.selected_record_ids)
    immutable_ids = set(immutable_instruction_record_ids(history))
    if (
        config.retire_closed_instructions
        and not config.keep_completed_task_statements
    ):
        genuine_instructions = [
            row.record_id for row in history.records
            if row.has_role(AgentRecordRole.TASK)
            or row.has_role(AgentRecordRole.USER_INPUT)
        ]
        immutable_ids = {
            row.record_id for row in history.records
            if row.has_role(AgentRecordRole.SYSTEM)
        }
        immutable_ids.add(genuine_instructions[-1])
    if not immutable_ids.issubset(selected_ids):
        raise AssertionError("selector removed an immutable user instruction")
    if history.records and history.records[-1].record_id not in selected_ids:
        raise AssertionError("selector removed the current trajectory record")

    transformed = dict(payload)
    transformed["messages"] = selected_messages
    observations = [
        row for row in history.records if row.primary_role.value in {
            "tool_observation", "source_view", "verification", "error_or_rejection"
        }
    ]
    version_rows = sum(bool(row.metadata.get("resource_version_fingerprints")) for row in observations)
    completeness_rows = sum(row.metadata.get("output_complete") is not None for row in observations)
    excluded_tokens = sum(row.excluded_tokens for row in plan.exclusions)
    compact_finalization_rows = [
        row for row in materialized.records
        if row.mode == MaterializationMode.PRIOR_FINALIZATION_RECEIPT
    ]
    compact_finalization_group_ids = {
        history.record_by_id[row.record_id].causal_group_id
        for row in compact_finalization_rows
    }
    trace = {
        "schema_version": 1,
        "study": "paper8_5_autonomous_agent_memory",
        "task_id": config.task_id,
        "session_id": config.session_id or config.task_id,
        "episode_index": config.episode_index,
        "prior_episode_count": len(prior_episodes),
        "persistent_prefix_applied": bool(prior_episodes),
        "policy": config.policy,
        "plan_policy": plan.policy,
        "plan_digest": plan.digest,
        "wire_plan_digest": wire_plan.digest,
        "wire_plan": wire_plan.to_dict(),
        "request_input_sha256": _digest(messages),
        "selected_messages_sha256": _digest(selected_messages),
        "request_message_roles": [str(row.get("role", "")) for row in messages],
        "request_message_content_sha256": [
            hashlib.sha256(str(row.get("content", "")).encode("utf-8")).hexdigest()
            for row in messages
        ],
        "selected_message_content_sha256": [
            hashlib.sha256(str(row.get("content", "")).encode("utf-8")).hexdigest()
            for row in selected_messages
        ],
        "exact_request_passthrough": bool(
            config.policy == "full" or exact_logical_noop
        ),
        "tokenizer": config.tokenizer_identity,
        "token_accounting_scope": "message_content_only_excludes_chat_template",
        "requested_budget_fraction": config.budget_fraction,
        "requested_budget_tokens": budget_tokens,
        "full_tokens": full_tokens,
        "selected_tokens": plan.selected_tokens,
        "materialized_tokens": materialized.materialized_tokens,
        "logical_retention_fraction": plan.realized_retention_fraction,
        "materialized_retention_fraction": materialized.materialized_retention_fraction,
        "protected_head_turns": config.protected_head_turns,
        "protected_tail_turns": config.protected_tail_turns,
        "matched_tail_boundary_compaction": (
            config.matched_tail_boundary_compaction
        ),
        "selection_reasons": dict(plan.selection_reasons),
        "compact_completed_finalizations": config.compact_completed_finalizations,
        "retire_closed_instructions": config.retire_closed_instructions,
        "keep_completed_task_statements": config.keep_completed_task_statements,
        "frontier_protocol_exemplars": config.frontier_protocol_exemplars,
        "frontier_workflow_exemplars": config.frontier_workflow_exemplars,
        "prior_finalization_receipt_count": len(compact_finalization_group_ids),
        "prior_finalization_receipt_tokens": sum(
            row.materialized_tokens for row in compact_finalization_rows
        ),
        "prior_finalization_receipt_token_saving": sum(
            row.original_tokens - row.materialized_tokens
            for row in compact_finalization_rows
        ),
        "budget_interpretation": (
            "strict_materialized_token_ceiling_with_mandatory_overflow"
            if matched_tail
            else
            "certified_exclusion_then_retention_floor"
            if config.policy == "dag_certified_progress_spine"
            else "retention_floor_round_up" if retention_floor
            else "hard_ceiling"
        ),
        "budget_satisfied": (
            matched_tail_unexplained_overflow_tokens == 0
            if matched_tail else unexplained_floor_underfill_tokens == 0
            if retention_floor else plan.selected_tokens <= budget_tokens
        ),
        "mandatory_budget_overflow_tokens": mandatory_overflow_tokens,
        "unexplained_materialized_budget_overflow_tokens": (
            matched_tail_unexplained_overflow_tokens
        ),
        "certified_exclusion_underfill_tokens": (
            certified_exclusion_underfill_tokens
        ),
        "unexplained_floor_underfill_tokens": unexplained_floor_underfill_tokens,
        "whole_turn_budget_overshoot_tokens": (
            max(0, plan.selected_tokens - budget_tokens) if retention_floor else 0
        ),
        "whole_turn_budget_undershoot_tokens": (
            max(0, budget_tokens - materialized.materialized_tokens)
            if matched_tail
            else max(0, budget_tokens - plan.selected_tokens)
            if retention_floor
            else 0
        ),
        "materialized_budget_unused_tokens": max(
            0, budget_tokens - materialized.materialized_tokens
        ),
        "materialized_budget_overshoot_tokens": max(
            0, materialized.materialized_tokens - budget_tokens
        ),
        "full_message_count": len(messages),
        "selected_message_count": len(selected_messages),
        "selected_record_ids": list(plan.selected_record_ids),
        "selected_causal_group_ids": list(plan.selected_causal_group_ids),
        "excluded_tokens": excluded_tokens,
        "excluded_causal_group_count": len(plan.exclusions),
        "exclusions": [asdict(row) for row in plan.exclusions],
        "negative_realization": config.negative_realization.value,
        "receipt_count": (
            len(receipt_realization.receipts) if receipt_realization is not None else 0
        ),
        "receipt_tokens": (
            receipt_realization.receipt_tokens if receipt_realization is not None else 0
        ),
        "receipt_token_saving": (
            receipt_realization.receipt_token_saving
            if receipt_realization is not None else 0
        ),
        "receipt_fail_closed_group_ids": (
            list(receipt_realization.fail_closed_group_ids)
            if receipt_realization is not None else []
        ),
        "observation_metadata_coverage": {
            "observation_records": len(observations),
            "complete_status_records": completeness_rows,
            "resource_version_records": version_rows,
        },
        "instrumentation_sidecar_join": sidecar_join,
        "recordization": {
            "source": recordization.source,
            "explicit_records": recordization.explicit_records,
            "inferred_records": recordization.inferred_records,
            "ambiguity_reasons": list(recordization.ambiguity_reasons),
        },
        "selection_abstained_for_sidecar": selection_abstained,
        "oracle_addback": oracle_addback,
    }
    logical_payload = dict(payload)
    logical_payload["messages"] = messages
    mandatory_message_indices = _native_mandatory_message_indices(
        messages, current_episode_start=current_episode_start,
    )
    record_message_indices = {
        row.record_id: row.message_index for row in history.records
    }
    selected_indices = {
        record_message_indices[record_id]
        for record_id in wire_plan.selected_record_ids
    }
    if not set(mandatory_message_indices).issubset(selected_indices):
        raise AssertionError("wire plan omits native mandatory current state")
    return AutonomousTransformation(
        transformed,
        plan,
        materialized.materialized_tokens,
        trace,
        logical_payload,
        wire_plan.to_dict(),
        record_message_indices,
        mandatory_message_indices,
    )


class AutonomousSelectionProxy:
    """Intercept non-streaming chat calls and forward selected ordinary text."""

    def __init__(
        self,
        upstream_base_url: str,
        *,
        config: AutonomousSelectionConfig,
        trace_path: Path,
        count_tokens: Callable[[str], int] = whitespace_tokens,
        upstream_api_key: str | None = None,
        timeout_seconds: int = 3600,
        instrumentation_root: Path | None = None,
        prior_episodes: Sequence[Mapping[str, Any]] = (),
        upstream_qualification_path: str | None = None,
        upstream_connect_attempts: int = 1,
        upstream_connect_retry_seconds: float = 1.0,
        upstream_curl_executable: str | None = None,
        curl_runner: Callable[..., Any] = subprocess.run,
        native_request_builder: Callable[..., Mapping[str, Any]] | None = None,
        first_request_capture_path: Path | None = None,
        request_content_normalizations: Sequence[tuple[str, str]] = (),
    ) -> None:
        if upstream_connect_attempts < 1:
            raise ValueError("upstream connect attempts must be positive")
        if upstream_connect_retry_seconds < 0:
            raise ValueError("upstream connect retry seconds cannot be negative")
        if upstream_qualification_path is not None and not upstream_qualification_path.startswith("/"):
            raise ValueError("upstream qualification path must be absolute")
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.config = config
        self.trace_path = Path(trace_path)
        self.count_tokens = count_tokens
        self.upstream_api_key = upstream_api_key
        self.timeout_seconds = timeout_seconds
        self.instrumentation_root = (
            Path(instrumentation_root) if instrumentation_root is not None else None
        )
        self.prior_episodes = tuple(dict(row) for row in prior_episodes)
        self.upstream_qualification_path = upstream_qualification_path
        self.upstream_connect_attempts = upstream_connect_attempts
        self.upstream_connect_retry_seconds = upstream_connect_retry_seconds
        self.upstream_curl_executable = upstream_curl_executable
        self._curl_runner = curl_runner
        self.native_request_builder = native_request_builder
        self.first_request_capture_path = (
            Path(first_request_capture_path)
            if first_request_capture_path is not None else None
        )
        self.request_content_normalizations = tuple(
            (re.compile(pattern), replacement)
            for pattern, replacement in request_content_normalizations
        )
        self._lock = threading.Lock()
        self._upstream_io_lock = threading.Lock()
        self._persistent_upstream: http.client.HTTPConnection | None = None
        self._request_count = 0
        self._successful_request_count = 0
        self._pending_reacquisition_count = 0
        self._pending_reacquisition_resources: tuple[str, ...] = ()
        self._execution_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self._upstream_failure = threading.Event()
        self._upstream_failure_detail: dict[str, str] | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                try:
                    proxy._forward(self)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self._error(400, "invalid_selection_request", error)
                except PermissionError as error:
                    self._error(429, "max_model_calls_exceeded", error)
                except (urllib.error.URLError, TimeoutError) as error:
                    self._error(502, "selection_upstream_unavailable", error)
                except Exception as error:  # pragma: no cover - defensive HTTP boundary
                    self._error(500, "selection_proxy_internal_error", error)

            def _error(self, status: int, code: str, error: Exception) -> None:
                body = json.dumps({"error": code, "message": str(error)}).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if urlparse(self.path).path == "/health":
                    body = json.dumps(proxy.health()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                if urlparse(self.path).path == "/v1/pra/execution-receipts":
                    try:
                        receipt = proxy._receive_execution_receipt(self)
                    except (ValueError, TypeError, json.JSONDecodeError) as error:
                        self._error(400, "invalid_execution_receipt", error)
                        return
                    body = json.dumps({
                        "status": "accepted",
                        "receipt_digest": receipt.get("receipt_digest"),
                    }).encode("utf-8")
                    self.send_response(202)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self._handle()

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{host}:{self._server.server_port}/v1"

    def close(self) -> None:
        with self._upstream_io_lock:
            if self._persistent_upstream is not None:
                self._persistent_upstream.close()
                self._persistent_upstream = None
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "protocol": "paper8.5-autonomous-text-selection-v1",
            "policy": self.config.policy,
            "engine_state_owned": False,
            "kv_metrics_available": self.native_request_builder is not None,
            "request_count": self._request_count,
            "successful_request_count": self._successful_request_count,
            "max_calls": self.config.max_calls,
            "upstream_failed": self._upstream_failure.is_set(),
            "execution_receipt_count": len(self._execution_receipts),
        }

    def _receive_execution_receipt(
        self, handler: BaseHTTPRequestHandler,
    ) -> dict[str, Any]:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("execution receipt must be a JSON object")
        receipt = dict(value)
        expected_session = self.config.session_id or self.config.task_id
        session_id = str(receipt.get("session_id") or "")
        tool_call_id = str(receipt.get("tool_call_id") or "")
        if not session_id or session_id != expected_session:
            raise ValueError("execution receipt session does not match the proxy session")
        if not tool_call_id:
            raise ValueError("execution receipt is missing tool_call_id")
        action_id = str(receipt.get("action_record_id") or tool_call_id)
        observation_ids = tuple(
            str(row) for row in receipt.get("observation_record_ids", ())
        )
        ToolExecutionReceipt.from_mapping(
            receipt,
            action_record_id=action_id,
            observation_record_ids=observation_ids,
        )
        key = (session_id, tool_call_id)
        with self._lock:
            previous = self._execution_receipts.get(key)
            if previous is not None and previous != receipt:
                raise ValueError("conflicting execution receipt for the same tool call")
            self._execution_receipts[key] = receipt
        return receipt

    def _attach_execution_receipts(
        self, payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], int]:
        copied = dict(payload)
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            return copied, 0
        session_id = self.config.session_id or self.config.task_id
        joined = 0
        messages: list[Any] = []
        with self._lock:
            receipts = dict(self._execution_receipts)
        for raw in raw_messages:
            if not isinstance(raw, Mapping):
                messages.append(raw)
                continue
            message = dict(raw)
            call_id = str(message.get("tool_call_id") or "")
            receipt = receipts.get((session_id, call_id)) if call_id else None
            if message.get("role") == "tool" and receipt is not None:
                metadata = dict(message.get("metadata") or {})
                existing = metadata.get("pra_execution_receipt")
                if existing is not None and existing != receipt:
                    raise ValueError("tool result contains a conflicting execution receipt")
                metadata["pra_execution_receipt"] = dict(receipt)
                message["metadata"] = metadata
                joined += 1
            messages.append(message)
        copied["messages"] = messages
        return copied, joined

    @property
    def upstream_failed(self) -> bool:
        return self._upstream_failure.is_set()

    @property
    def upstream_failure_detail(self) -> Mapping[str, str] | None:
        return self._upstream_failure_detail

    def _validate_generation(self, payload: Mapping[str, Any]) -> None:
        if self.config.expected_model is not None and payload.get("model") != self.config.expected_model:
            raise ValueError(
                f"model mismatch: expected {self.config.expected_model!r}, "
                f"observed {payload.get('model')!r}"
            )
        expected = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
        }
        for key, value in expected.items():
            if key not in payload or payload[key] != value:
                raise ValueError(
                    f"generation mismatch for {key}: expected {value!r}, "
                    f"observed {payload.get(key)!r}"
                )
        if self.config.max_completion_tokens is not None:
            observed = payload.get("max_completion_tokens", payload.get("max_tokens"))
            if observed != self.config.max_completion_tokens:
                raise ValueError(
                    "completion limit mismatch: expected "
                    f"{self.config.max_completion_tokens}, observed {observed!r}"
                )

    def _target(self, incoming_path: str) -> str:
        root = self.upstream_base_url.removesuffix("/v1")
        return root + incoming_path

    def _open_qualified_upstream(self) -> http.client.HTTPConnection:
        parsed = urlparse(self.upstream_base_url)
        connection_type = (
            http.client.HTTPSConnection
            if parsed.scheme == "https" else http.client.HTTPConnection
        )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        last_error: Exception | None = None
        for attempt in range(1, self.upstream_connect_attempts + 1):
            connection = connection_type(
                parsed.hostname, port, timeout=self.timeout_seconds
            )
            try:
                headers = {}
                if self.upstream_api_key is not None:
                    headers["Authorization"] = f"Bearer {self.upstream_api_key}"
                connection.request(
                    "GET", str(self.upstream_qualification_path), headers=headers
                )
                response = connection.getresponse()
                response.read()
                if response.status >= 500:
                    raise OSError(
                        "upstream connection qualification returned "
                        f"HTTP {response.status}"
                    )
                return connection
            except (OSError, http.client.HTTPException) as error:
                connection.close()
                last_error = error
                if attempt < self.upstream_connect_attempts:
                    time.sleep(self.upstream_connect_retry_seconds)
        raise urllib.error.URLError(last_error or "upstream qualification failed")

    def _qualified_upstream_request(
        self, request: urllib.request.Request
    ) -> tuple[bytes, int, Mapping[str, str]]:
        """Use one qualified connection; never replay a model POST."""

        with self._upstream_io_lock:
            if self._persistent_upstream is None:
                self._persistent_upstream = self._open_qualified_upstream()
            parsed = urlparse(request.full_url)
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            try:
                self._persistent_upstream.request(
                    request.get_method(), target,
                    body=request.data,
                    headers=dict(request.header_items()),
                )
                response = self._persistent_upstream.getresponse()
                body = response.read()
                return body, response.status, response.headers
            except (OSError, http.client.HTTPException) as error:
                self._persistent_upstream.close()
                self._persistent_upstream = None
                raise urllib.error.URLError(error) from error

    def _curl_upstream_request(
        self, request: urllib.request.Request
    ) -> tuple[bytes, int, Mapping[str, str]]:
        """Forward exactly once through curl; a transport error is fail-closed."""

        command = [
            str(self.upstream_curl_executable),
            "--silent", "--show-error",
            "--max-time", str(self.timeout_seconds),
            # A refused TCP connection proves that the HTTP request was not
            # delivered, so retrying it cannot duplicate a model generation.
            # Do not use ``--retry-all-errors`` here: a timeout after upload is
            # ambiguous and must remain fail-closed rather than replay a POST.
            "--retry", str(max(0, self.upstream_connect_attempts - 1)),
            "--retry-connrefused",
            "--retry-delay", str(self.upstream_connect_retry_seconds),
            "--request", request.get_method(),
        ]
        for key, value in request.header_items():
            command.extend(("--header", f"{key}: {value}"))
        if request.data is not None:
            command.extend(("--data-binary", "@-"))
        command.extend((
            "--output", "-", "--write-out", "\n%{http_code}",
            request.full_url,
        ))
        try:
            completed = self._curl_runner(
                command,
                input=request.data,
                capture_output=True,
                timeout=self.timeout_seconds + 10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise urllib.error.URLError(error) from error
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise urllib.error.URLError(
                f"curl exited {completed.returncode}: {detail}"
            )
        try:
            response_body, status_text = completed.stdout.rsplit(b"\n", 1)
            status = int(status_text)
        except (ValueError, TypeError) as error:
            raise urllib.error.URLError("curl response lacks HTTP status") from error
        return response_body, status, {"Content-Type": "application/json"}

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        transformation: AutonomousTransformation | None = None
        request_index: int | None = None
        native_request_index: int | None = None
        if handler.command == "POST" and urlparse(handler.path).path == "/v1/chat/completions":
            payload = json.loads(body.decode("utf-8"))
            if self.first_request_capture_path is not None:
                with self._lock:
                    if not self.first_request_capture_path.exists():
                        self.first_request_capture_path.parent.mkdir(
                            parents=True, exist_ok=True
                        )
                        self.first_request_capture_path.write_text(
                            json.dumps(payload, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
            if self.request_content_normalizations:
                normalized_messages = []
                for message in payload.get("messages") or ():
                    normalized = dict(message)
                    content = normalized.get("content")
                    if isinstance(content, str):
                        for pattern, replacement in self.request_content_normalizations:
                            content = pattern.sub(replacement, content)
                        normalized["content"] = content
                    elif isinstance(content, list):
                        normalized_content = []
                        for item in content:
                            if not isinstance(item, Mapping):
                                normalized_content.append(item)
                                continue
                            normalized_item = dict(item)
                            text = normalized_item.get("text")
                            if isinstance(text, str):
                                for pattern, replacement in self.request_content_normalizations:
                                    text = pattern.sub(replacement, text)
                                normalized_item["text"] = text
                            normalized_content.append(normalized_item)
                        normalized["content"] = normalized_content
                    normalized_messages.append(normalized)
                payload["messages"] = normalized_messages
            payload, joined_execution_receipts = self._attach_execution_receipts(payload)
            if self.config.fill_missing_generation_parameters:
                payload.setdefault("temperature", self.config.temperature)
                payload.setdefault("top_p", self.config.top_p)
                payload.setdefault("seed", self.config.seed)
                if (
                    self.config.max_completion_tokens is not None
                    and "max_completion_tokens" not in payload
                    and "max_tokens" not in payload
                ):
                    payload["max_tokens"] = self.config.max_completion_tokens
            self._validate_generation(payload)
            # ``max_completion_tokens`` is accepted by recent OpenAI clients,
            # but several OpenAI-compatible local runtimes silently ignore it.
            # Normalize the already-validated logical limit to the older,
            # widely enforced wire key before forwarding.  Without this step
            # an OpenHands request declared as 1,024 tokens was observed to
            # generate more than 12,000 tokens on Ollama.
            if self.config.max_completion_tokens is not None:
                payload["max_tokens"] = self.config.max_completion_tokens
                payload.pop("max_completion_tokens", None)
            with self._lock:
                if self._request_count >= self.config.max_calls:
                    raise PermissionError(
                        f"maximum {self.config.max_calls} model calls reached"
                    )
                self._request_count += 1
                request_index = self._request_count
                # Native bootstrap and turn sequencing belong to successful
                # logical generations, not raw HTTP attempts.  LiteLLM may
                # retry a request after selection, transport, or engine
                # failure; every such attempt must retain the same native
                # sequence index until an upstream completion succeeds.
                native_request_index = self._successful_request_count + 1
            transformation = transform_autonomous_payload(
                payload,
                self.config,
                count_tokens=self.count_tokens,
                instrumentation_root=self.instrumentation_root,
                prior_episodes=self.prior_episodes,
            )
            transformation.trace["execution_receipt_sidecar_join"] = {
                "stored": len(self._execution_receipts),
                "joined": joined_execution_receipts,
            }
            if self.native_request_builder is not None:
                native_payload = self.native_request_builder(
                    transformation.logical_payload,
                    wire_plan=transformation.wire_plan,
                    record_message_indices=(
                        transformation.record_message_indices
                    ),
                    mandatory_message_indices=(
                        transformation.mandatory_message_indices
                    ),
                    session_id=(self.config.session_id or self.config.task_id),
                    request_index=native_request_index,
                    count_tokens=self.count_tokens,
                    tokenizer_identity=self.config.tokenizer_identity,
                )
                if not isinstance(native_payload, Mapping):
                    raise TypeError("native request builder did not return a mapping")
                transformation = replace(
                    transformation, payload=dict(native_payload),
                )
            body = json.dumps(transformation.payload).encode("utf-8")

        headers = {
            key: value for key, value in handler.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        if self.upstream_api_key is not None:
            headers["Authorization"] = f"Bearer {self.upstream_api_key}"
        request = urllib.request.Request(
            self._target(handler.path),
            data=body if handler.command == "POST" else None,
            headers=headers,
            method=handler.command,
        )
        status = 200
        response_headers: Mapping[str, str]
        try:
            if self.upstream_curl_executable is not None:
                response_body, status, response_headers = (
                    self._curl_upstream_request(request)
                )
            elif self.upstream_qualification_path is not None:
                response_body, status, response_headers = (
                    self._qualified_upstream_request(request)
                )
            else:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_body = response.read()
                    status = response.status
                    response_headers = response.headers
        except urllib.error.HTTPError as error:
            response_body = error.read()
            status = error.code
            response_headers = error.headers
        except (urllib.error.URLError, TimeoutError) as error:
            self._upstream_failure_detail = {
                "error_type": type(error).__name__,
                "error_detail": str(error),
            }
            self._upstream_failure.set()
            if transformation is not None:
                self._append_trace({
                    **transformation.trace,
                    "request_index": request_index,
                    "generation": {
                        "model": transformation.payload.get("model"),
                        "temperature": transformation.payload.get("temperature"),
                        "top_p": transformation.payload.get("top_p"),
                        "seed": transformation.payload.get("seed"),
                        "max_completion_tokens": transformation.payload.get(
                            "max_completion_tokens",
                            transformation.payload.get("max_tokens"),
                        ),
                    },
                    "upstream_status": 0,
                    "upstream_transport_error": type(error).__name__,
                    "upstream_transport_error_detail": str(error),
                    "response_sha256": None,
                    "response_sha256_scope": None,
                    "assistant_content_sha256": None,
                    "reported_prompt_tokens": None,
                    "reported_completion_tokens": None,
                    "reported_total_tokens": None,
                    "assistant_command_sha256": None,
                    "assistant_operation": None,
                    "assistant_resource_ids": [],
                    "assistant_is_search": False,
                    "assistant_is_read": False,
                    "assistant_is_test": False,
                    "reacquired_excluded_resources": [],
                    "reacquisition_count": 0,
                    "reacquisition_proxy_for_false_exclusion": False,
                    "previous_reacquisition_count": 0,
                    "previous_reacquisition_resources": [],
                    "reacquired_observation_tokens_from_previous_action": 0,
                })
            raise

        if transformation is not None and 200 <= int(status) < 300:
            with self._lock:
                expected = self._successful_request_count + 1
                if native_request_index != expected:
                    raise RuntimeError(
                        "Native agent request sequencing observed concurrent "
                        "out-of-order completions."
                    )
                self._successful_request_count = expected

        if transformation is not None:
            raw_messages = payload.get("messages") or ()
            with self._lock:
                pending_reacquisition_count = self._pending_reacquisition_count
                pending_reacquisition_resources = self._pending_reacquisition_resources
            previous_observation_tokens = 0
            if (
                pending_reacquisition_count
                and raw_messages
                and isinstance(raw_messages[-1], Mapping)
                and raw_messages[-1].get("role") in {"user", "tool"}
            ):
                previous_observation_tokens = self.count_tokens(
                    str(raw_messages[-1].get("content", ""))
                )
            assistant_content = _assistant_content(response_body)
            response_usage = _response_usage(response_body)
            try:
                response_payload = json.loads(response_body.decode("utf-8"))
                engine_pra = response_payload.get("pra") or {}
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                engine_pra = {}
            command = _assistant_command(response_body)
            operation = classify_bash_operation(command)
            resources = extract_resource_ids(command, "") if command else ()
            reacquired = reacquired_excluded_resources(command, transformation.plan.exclusions)
            trace = {
                **transformation.trace,
                "request_index": request_index,
                "native_request_index": native_request_index,
                "generation": {
                    "model": transformation.payload.get("model"),
                    "temperature": transformation.payload.get("temperature"),
                    "top_p": transformation.payload.get("top_p"),
                    "seed": transformation.payload.get("seed"),
                    "max_completion_tokens": transformation.payload.get(
                        "max_completion_tokens", transformation.payload.get("max_tokens")
                    ),
                },
                "upstream_status": status,
                "response_sha256": hashlib.sha256(response_body).hexdigest(),
                "response_sha256_scope": (
                    "raw_http_body_includes_volatile_response_metadata"
                ),
                "assistant_content_sha256": (
                    hashlib.sha256(assistant_content.encode("utf-8")).hexdigest()
                    if assistant_content is not None else None
                ),
                **response_usage,
                "native_pra_delivery": self.native_request_builder is not None,
                "engine_pra_metrics": dict(engine_pra) if isinstance(engine_pra, Mapping) else {},
                "assistant_command_sha256": (
                    hashlib.sha256(command.encode("utf-8")).hexdigest() if command else None
                ),
                "assistant_operation": operation.value if command else None,
                "assistant_resource_ids": list(resources),
                "assistant_is_search": operation is BashOperation.SEARCH_DISCOVERY,
                "assistant_is_read": operation in {BashOperation.READ, BashOperation.DIFF},
                "assistant_is_test": operation is BashOperation.VERIFY,
                "reacquired_excluded_resources": list(reacquired),
                "reacquisition_count": len(reacquired),
                "reacquisition_proxy_for_false_exclusion": bool(reacquired),
                "previous_reacquisition_count": pending_reacquisition_count,
                "previous_reacquisition_resources": list(
                    pending_reacquisition_resources
                ),
                "reacquired_observation_tokens_from_previous_action": (
                    previous_observation_tokens
                ),
            }
            self._append_trace(trace)
            with self._lock:
                self._pending_reacquisition_count = len(reacquired)
                self._pending_reacquisition_resources = tuple(reacquired)

        handler.send_response(status)
        for key in ("Content-Type", "Retry-After"):
            if value := response_headers.get(key):
                handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(response_body)))
        handler.end_headers()
        handler.wfile.write(response_body)

    def _append_trace(self, trace: Mapping[str, Any]) -> None:
        row = json.dumps(dict(trace), sort_keys=True, default=str) + "\n"
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(row)
                handle.flush()
