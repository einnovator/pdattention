"""Fair, agent-visible context treatments for the SWE-bench PRA frontier."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urlparse

from pra_hf.large_record_index import LargeRecordIndex, LargeRecordSearchPolicy
from pra_hf.deployment import PRAAgentRetentionPolicy


_TOKEN = re.compile(r"\S+")
_NATURAL_OBSERVATION_BOUNDARY = re.compile(
    r"^(?:diff --git |@@ |Traceback \(most recent call last\):|"
    r"\s*File \"|(?:FAILED|ERROR|PASSED)\s+|={3,}\s*(?:FAILURES|ERRORS)|"
    r"\*{3,}\s*(?:FAILURES|ERRORS)|---\s+a/|\+\+\+\s+b/)",
)


class ContextTreatment(str, Enum):
    """Request transformations compared after baseline reproduction."""

    PASSTHROUGH = "gateway-passthrough"
    TRUNCATION = "truncation"
    PRA_SELECTED_CONTEXT = "gateway-pra"
    DIRECT_NATIVE_PRA = "direct-native-pra"
    GATEWAY_NATIVE_PRA = "gateway-native-pra"
    HEADROOM = "headroom"


class FrozenReplayDivergence(RuntimeError):
    """The paired trajectory no longer has a recorded exact request."""


CONSUMPTION_POLICIES = (
    "standard",
    "verification-guard-v1",
    "verification-enforced-v1",
)

AGENT_HISTORY_SELECTION_POLICIES = (
    "task-aware-v1",
    "paper8.5-matched-causal-token-tail-v1",
)
MATCHED_CAUSAL_TOKEN_TAIL_POLICY = AGENT_HISTORY_SELECTION_POLICIES[1]
FROZEN_AGENT_MEMORY_PLAN_CONTRACT = "frozen-agent-memory-plan-v1"


@dataclass(frozen=True)
class TreatmentTrace:
    """One request's disjoint logical, selected, and visible context accounting."""

    request_index: int
    session_id: str
    mode: str
    budget_fraction: float
    logical_input_tokens_estimate: int
    mandatory_tokens_estimate: int
    selected_tokens_estimate: int
    physical_input_tokens_estimate: int
    tokens_avoided_estimate: int
    token_saving_fraction_estimate: float
    candidate_segments: int
    selected_segments: int
    selected_resource_digest: str | None
    route_time_s: float
    token_estimator: str = "whitespace_v1"
    agent_history_selection_policy: str = "task-aware-v1"
    requested_budget_tokens: int | None = None
    logical_budget_unused_tokens: int | None = None
    mandatory_overflow_tokens: int = 0


def apply_consumption_policy(
    payload: Mapping[str, Any], policy: str,
    *, logical_messages: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    """Change model presentation only after routing, preserving selected records."""

    if policy not in CONSUMPTION_POLICIES:
        raise ValueError(
            f"unknown consumption policy {policy!r}; expected one of {CONSUMPTION_POLICIES}"
        )
    transformed = dict(payload)
    if policy == "standard":
        return transformed, 0
    guidance = _verification_guidance(logical_messages or payload.get("messages", ()))
    messages = [dict(row) for row in transformed.get("messages", ())]
    if guidance:
        current_index = next(
            (
                index for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") != "system"
            ),
            None,
        )
        tagged = (
            f'<pra_consumption_policy name="{policy}">\n'
            f"{guidance}\n</pra_consumption_policy>"
        )
        if current_index is None:
            messages.append({"role": "user", "content": tagged})
        else:
            content = str(messages[current_index].get("content") or "")
            messages[current_index]["content"] = f"{content}\n\n{tagged}"
        transformed["messages"] = messages
    envelope = dict(transformed.get("pra") or {})
    metadata = dict(envelope.get("metadata") or {})
    metadata["consumption_policy"] = policy
    envelope["metadata"] = metadata
    transformed["pra"] = envelope
    return transformed, _count_tokens(guidance)


def transform_chat_payload(
    payload: Mapping[str, Any],
    *,
    mode: ContextTreatment | str,
    budget_fraction: float,
    request_index: int = 0,
    segment_tokens: int | None = None,
    recent_completed_turns: int | None = None,
    recent_records_per_turn: int | None = None,
    recent_source_turns: int | None = None,
    recent_progress_turns: int | None = None,
    recent_mutation_turns: int | None = None,
    recent_verification_turns: int | None = None,
    max_records_per_turn_before_chunking: int | None = None,
    preserve_action_observation_pairs: bool | None = None,
    causal_bundle_round_up: bool | None = None,
    frozen_selection: Sequence[tuple[str, str]] | None = None,
    agent_history_selection_policy: str = "task-aware-v1",
    count_tokens: Callable[[str], int] | None = None,
    tokenizer_identity: str = "whitespace_v1",
) -> tuple[dict[str, Any], TreatmentTrace]:
    """Apply a matched budget, optionally replaying an exact recorded selection."""

    mode = ContextTreatment(mode)
    if agent_history_selection_policy not in AGENT_HISTORY_SELECTION_POLICIES:
        raise ValueError(
            "unknown agent-history selection policy "
            f"{agent_history_selection_policy!r}; expected one of "
            f"{AGENT_HISTORY_SELECTION_POLICIES}"
        )
    exact_count = count_tokens or _count_tokens
    request_envelope = payload.get("pra") or {}
    request_metadata = (
        request_envelope.get("metadata") or {}
        if isinstance(request_envelope, Mapping) else {}
    )
    request_retention = request_metadata.get("retention_policy") or {}
    policy = PRAAgentRetentionPolicy.from_mapping(request_retention)
    policy = PRAAgentRetentionPolicy(
        recent_completed_turns=(
            policy.recent_completed_turns
            if recent_completed_turns is None else recent_completed_turns
        ),
        recent_records_per_turn=(
            policy.recent_records_per_turn
            if recent_records_per_turn is None else recent_records_per_turn
        ),
        recent_source_turns=(
            policy.recent_source_turns
            if recent_source_turns is None else recent_source_turns
        ),
        recent_progress_turns=(
            policy.recent_progress_turns
            if recent_progress_turns is None else recent_progress_turns
        ),
        recent_mutation_turns=(
            policy.recent_mutation_turns
            if recent_mutation_turns is None else recent_mutation_turns
        ),
        recent_verification_turns=(
            policy.recent_verification_turns
            if recent_verification_turns is None else recent_verification_turns
        ),
        large_record_chunk_tokens=(
            policy.large_record_chunk_tokens
            if segment_tokens is None else segment_tokens
        ),
        max_records_per_turn_before_chunking=(
            policy.max_records_per_turn_before_chunking
            if max_records_per_turn_before_chunking is None
            else max_records_per_turn_before_chunking
        ),
        preserve_action_observation_pairs=(
            policy.preserve_action_observation_pairs
            if preserve_action_observation_pairs is None
            else preserve_action_observation_pairs
        ),
        causal_bundle_round_up=(
            policy.causal_bundle_round_up
            if causal_bundle_round_up is None else causal_bundle_round_up
        ),
    )
    segment_tokens = policy.large_record_chunk_tokens
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    transformed = dict(payload)
    messages = [dict(row) for row in payload.get("messages", ())]
    if not messages:
        raise ValueError("chat payload requires messages")
    session_id = session_id_for_messages(messages)
    logical_tokens = sum(exact_count(str(row.get("content") or "")) for row in messages)
    if mode in {ContextTreatment.PASSTHROUGH, ContextTreatment.HEADROOM}:
        if frozen_selection is not None:
            raise ValueError(f"{mode.value} mode cannot replay a context selection")
        return transformed, _trace(
            request_index, session_id, mode, budget_fraction, logical_tokens, logical_tokens,
            0, logical_tokens, 0, 0, None, 0.0,
        )

    mandatory_indices = _mandatory_indices(messages)
    task_indices = _pinned_task_indices(messages, mandatory_indices)
    if agent_history_selection_policy == MATCHED_CAUSAL_TOKEN_TAIL_POLICY:
        progress_indices = set()
        progress_classes = {
            "recent": set(), "source": set(), "progress_state": set(),
            "mutation": set(), "verification": set(),
        }
    else:
        progress_indices, progress_classes = _progress_pinned_indices(
            messages, mandatory_indices | task_indices,
            recent_turns=policy.recent_completed_turns,
            recent_records_per_turn=policy.recent_records_per_turn,
            source_turns=policy.recent_source_turns,
            progress_turns=policy.recent_progress_turns,
            mutation_turns=policy.recent_mutation_turns,
            verification_turns=policy.recent_verification_turns,
            max_records_per_turn_before_chunking=(
                policy.max_records_per_turn_before_chunking
            ),
            preserve_action_observation_pairs=policy.preserve_action_observation_pairs,
        )
    pinned_indices = task_indices | progress_indices
    mandatory_tokens = sum(
        exact_count(str(messages[index].get("content") or ""))
        for index in mandatory_indices
    )
    pinned_tokens = sum(
        exact_count(str(messages[index].get("content") or ""))
        for index in pinned_indices
    )
    requested_budget_tokens = math.ceil(logical_tokens * budget_fraction)
    target_tokens = max(
        mandatory_tokens + pinned_tokens,
        requested_budget_tokens,
    )
    resource_budget = max(0, target_tokens - mandatory_tokens)
    available_tokens = max(0, resource_budget - pinned_tokens)
    candidate_indices = [
        index for index in range(len(messages))
        if index not in mandatory_indices and index not in pinned_indices
    ]
    started = time.perf_counter()
    if mode is ContextTreatment.TRUNCATION:
        if frozen_selection is not None:
            raise ValueError("truncation mode cannot replay a context selection")
        selected = [
            (index, messages[index]) for index in sorted(pinned_indices)
        ] + _truncate_recent(messages, candidate_indices, available_tokens)
        selected_tokens = sum(
            exact_count(str(row[1].get("content") or "")) for row in selected
        )
        keep = [(index, messages[index]) for index in mandatory_indices] + selected
        transformed["messages"] = [row for _, row in sorted(keep, key=lambda item: item[0])]
        candidate_segments = len(candidate_indices) + len(pinned_indices)
        selected_segments = len(selected)
        selected_digest = _selection_digest(
            (f"m{index}", str(row.get("content", ""))) for index, row in selected
        )
    else:
        query = _task_aware_query(messages, task_indices, mandatory_indices)
        pinned_segments = _segments(messages, sorted(pinned_indices), segment_tokens)
        bundles = _turn_bundles(
            messages, candidate_indices, segment_tokens,
            preserve_action_observation_pairs=policy.preserve_action_observation_pairs,
        )
        segments = [segment for bundle in bundles for segment in bundle]
        matched_logical_tokens: int | None = None
        if frozen_selection is not None:
            selected_texts = list(frozen_selection)
        elif agent_history_selection_policy == MATCHED_CAUSAL_TOKEN_TAIL_POLICY:
            matched_indices = _select_matched_causal_token_tail_indices(
                messages,
                candidate_indices,
                available_tokens,
                count_tokens=exact_count,
            )
            materialized_indices = sorted(pinned_indices | set(matched_indices))
            selected_texts = _segments(
                messages, materialized_indices, segment_tokens
            )
            matched_logical_tokens = sum(
                exact_count(str(messages[index].get("content") or ""))
                for index in materialized_indices
            )
        else:
            selected_texts = _sort_segments([
                *pinned_segments,
                *_select_turn_bundles(
                    bundles, query, available_tokens,
                    round_up=policy.causal_bundle_round_up,
                    count_tokens=exact_count,
                ),
            ])
        if frozen_selection is not None:
            valid_segments = _sort_segments([*pinned_segments, *segments])
            valid_by_id = dict(valid_segments)
            selected_ids = [segment_id for segment_id, _ in selected_texts]
            if len(set(selected_ids)) != len(selected_ids):
                raise ValueError("frozen selection contains duplicate resource IDs")
            invalid = [
                segment_id for segment_id, text in selected_texts
                if valid_by_id.get(segment_id) != text
            ]
            if invalid:
                raise ValueError(
                    "frozen selection is not an exact subset of the current request: "
                    + ", ".join(invalid)
                )
            positions = {
                segment_id: position
                for position, (segment_id, _) in enumerate(valid_segments)
            }
            selected_positions = [positions[segment_id] for segment_id in selected_ids]
            if selected_positions != sorted(selected_positions):
                raise ValueError(
                    "frozen selection changes causal resource order"
                )
            frozen = dict(selected_texts)
            missing_pinned = [
                segment_id for segment_id, text in pinned_segments
                if frozen.get(segment_id) != text
            ]
            if missing_pinned:
                raise ValueError(
                    "frozen selection omits or changes pinned task segments: "
                    + ", ".join(missing_pinned)
                )
        selected_tokens = (
            matched_logical_tokens
            if matched_logical_tokens is not None
            else sum(exact_count(text) for _, text in selected_texts)
        )
        selected_digest = _selection_digest(selected_texts)
        selection_materialization = (
            "whole-causal-records-v1"
            if agent_history_selection_policy
            == MATCHED_CAUSAL_TOKEN_TAIL_POLICY
            else "record-aligned-segments-v1"
        )
        agent_memory_plan_digest = hashlib.sha256(json.dumps(
            {
                "contract": FROZEN_AGENT_MEMORY_PLAN_CONTRACT,
                "policy": agent_history_selection_policy,
                "tokenizer_identity": tokenizer_identity,
                "budget_fraction": float(budget_fraction),
                "materialization": selection_materialization,
                "mandatory_message_indices": sorted(mandatory_indices),
                "selected_resource_digest": selected_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        transformed["messages"] = [messages[index] for index in sorted(mandatory_indices)]
        resources = [
            {
                "resource_id": segment_id,
                "uri": f"pra://agent-trajectory/{segment_id}",
                "record_type": "agent_trajectory_segment",
                "text": text,
                "version": "v4-progress-spine",
                "source_fingerprint": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "authorization_scope": "swebench-agent-visible",
                "metadata": _segment_metadata(
                    segment_id,
                    messages,
                    selection_policy=(
                        MATCHED_CAUSAL_TOKEN_TAIL_POLICY
                        if agent_history_selection_policy
                        == MATCHED_CAUSAL_TOKEN_TAIL_POLICY
                        else "typed_bm25_embedding_rrf"
                    ),
                ),
            }
            for segment_id, text in selected_texts
        ]
        candidate_segments = len(pinned_segments) + len(segments)
        selected_segments = len(selected_texts)
        envelope = dict(transformed.get("pra") or {})
        native_requested = mode in {
            ContextTreatment.DIRECT_NATIVE_PRA,
            ContextTreatment.GATEWAY_NATIVE_PRA,
        }
        envelope.update({
            "tenant_id": "paper4-5-swebench",
            "session_id": session_id,
            "resources": resources,
            "budget": {
                "max_resources": max(1, len(resources)),
                # Retention is a floor at causal-bundle granularity.  Expose
                # the realized rounded allowance so the engine does not reject
                # the final indivisible bundle as an over-budget request.
                "max_selected_tokens": max(1, selected_tokens),
            },
            "allow_text_fallback": not native_requested,
            "required_capabilities": ["logical_refs", "native_kv"] if native_requested else [],
            "pra_policy": {
                "profile": (
                    MATCHED_CAUSAL_TOKEN_TAIL_POLICY
                    if agent_history_selection_policy
                    == MATCHED_CAUSAL_TOKEN_TAIL_POLICY
                    else "swebench-balanced-v1"
                )
            },
            "metadata": {
                **dict(envelope.get("metadata") or {}),
                "requested_mode": "native-memory" if native_requested else "selected-context",
                "connection": (
                    "direct" if mode is ContextTreatment.DIRECT_NATIVE_PRA else "gateway"
                ),
                "benchmark_fairness": "agent-visible-messages-only",
                "budget_fraction": float(budget_fraction),
                "target_retention_fraction": float(budget_fraction),
                "selection_contract": (
                    FROZEN_AGENT_MEMORY_PLAN_CONTRACT
                    if agent_history_selection_policy
                    == MATCHED_CAUSAL_TOKEN_TAIL_POLICY
                    else "minimum-retention-floor"
                ),
                "agent_history_selection_policy": agent_history_selection_policy,
                "selection_tokenizer_identity": tokenizer_identity,
                "selection_materialization": selection_materialization,
                "agent_memory_plan_digest": agent_memory_plan_digest,
                "realized_retention_fraction": (
                    (mandatory_tokens + selected_tokens) / logical_tokens
                    if logical_tokens else 1.0
                ),
                "retention_rounded_up": (
                    mandatory_tokens + selected_tokens > requested_budget_tokens
                ),
                "selection_budget_policy": (
                    "matched_causal_token_tail_strict_ceiling_v1"
                    if agent_history_selection_policy
                    == MATCHED_CAUSAL_TOKEN_TAIL_POLICY
                    else (
                        "causal_bundle_round_up_v1"
                        if policy.causal_bundle_round_up
                        else "causal_bundle_hard_cap_v1"
                    )
                ),
                "target_physical_tokens_estimate": requested_budget_tokens,
                "causal_round_up_tokens_estimate": max(
                    0, mandatory_tokens + selected_tokens - requested_budget_tokens,
                ),
                "retention_policy": policy.to_dict(),
                # Moving completed chat turns from the inline message list to
                # typed resources is an intentional representation change,
                # not a destructive rewrite of the logical agent history.
                # G11 uses this marker to retain the engine session and lets
                # the native adapter validate/backtrack the exact token prefix.
                "history_projection": "live-agent-kv-v1",
                "selection_complete": selected_segments == candidate_segments,
                # The engine needs stable logical record coordinates to bind
                # selected records to K/V cells captured when those records
                # were first evaluated.  Content remains in the ordinary
                # mandatory messages or the selected resource bodies; this
                # manifest carries only identities and hashes, so omitted
                # history cannot be silently re-materialized from it.
                "logical_message_manifest": [
                    {
                        "message_index": index,
                        "role": str(message.get("role", "")),
                        "content_sha256": hashlib.sha256(
                            str(message.get("content", "")).encode("utf-8")
                        ).hexdigest(),
                    }
                    for index, message in enumerate(messages)
                ],
                "mandatory_message_indices": sorted(mandatory_indices),
                "pinned_task_segments": [
                    segment_id for segment_id, _ in _segments(
                        messages, sorted(task_indices), segment_tokens,
                    )
                ],
                "pinned_progress_segments": [
                    segment_id for segment_id, _ in _segments(
                        messages, sorted(progress_indices), segment_tokens,
                    )
                ],
                "pinned_progress_state_segments": [
                    segment_id for segment_id, _ in _segments(
                        messages,
                        sorted(progress_classes["progress_state"]),
                        segment_tokens,
                    )
                ],
            },
        })
        transformed["pra"] = envelope
    physical_tokens = mandatory_tokens + selected_tokens
    return transformed, _trace(
        request_index, session_id, mode, budget_fraction, logical_tokens, mandatory_tokens,
        selected_tokens, physical_tokens, candidate_segments, selected_segments,
        selected_digest, time.perf_counter() - started,
        token_estimator=tokenizer_identity,
        agent_history_selection_policy=agent_history_selection_policy,
        requested_budget_tokens=requested_budget_tokens,
        logical_budget_unused_tokens=max(0, requested_budget_tokens - physical_tokens),
        mandatory_overflow_tokens=max(
            0, mandatory_tokens + pinned_tokens - requested_budget_tokens
        ),
    )


def session_id_for_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    """Identify an agent task from its first non-system message without labels."""

    first_task_message = next(
        (row for row in messages if row.get("role") != "system"), messages[-1]
    )
    material = json.dumps(first_task_message, sort_keys=True, default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _mandatory_indices(messages: Sequence[Mapping[str, Any]]) -> set[int]:
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
            index for index in range(latest_observation - 1, -1, -1)
            if messages[index].get("role") == "assistant"
        ),
        None,
    )
    # The current observation is not causally meaningful without the action
    # that produced it.  Keep the complete active tail visible, including any
    # user/tool format-recovery turns appended after a rejected action.  On the
    # first request there is no assistant yet, so the task itself is mandatory.
    tail_start = latest_action if latest_action is not None else latest_observation
    return {
        *system,
        *(
            index for index in range(tail_start, latest_observation + 1)
            if messages[index].get("role") != "system"
        ),
    }


def _pinned_task_indices(
    messages: Sequence[Mapping[str, Any]], mandatory_indices: set[int],
) -> set[int]:
    """Pin the initial task/instruction turn while selecting dynamic history."""

    first_task = next(
        (index for index, row in enumerate(messages) if row.get("role") == "user"),
        None,
    )
    if first_task is None or first_task in mandatory_indices:
        return set()
    return {first_task}


_MUTATION = re.compile(
    r"(?:\bsed\s+-i\b|\bapply_patch\b|\bperl\s+-[pi]\b|\bpatch\s+-p\d|"
    r"\bgit\s+apply\b|(?:^|&&|;)\s*(?:cp|mv)\s+)",
    re.IGNORECASE,
)
_VERIFICATION = re.compile(
    r"(?:\bpytest\b|\btox\b|\bmake\s+test\b|\bpython(?:3)?\s+-m\s+test|"
    r"\bgit\s+diff\b|\bpatch\.txt\b|COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT)",
    re.IGNORECASE,
)
_SOURCE_EVIDENCE = re.compile(
    r"(?:^|&&|;|\|\|)\s*(?:(?:cd|pushd)\s+[^;&|]+\s*&&\s*)?"
    # Pin the latest command that exposed source contents. Broad discovery
    # commands (find/grep/rg) remain retrieval candidates: otherwise a later
    # path-only search can evict the actual source view it depends on.
    r"(?:cat|head|tail|sed\s+-n)\b",
    re.IGNORECASE | re.MULTILINE,
)
_PROGRESS_STATE = re.compile(
    r"(?:\b(?:root cause|working hypothesis|current hypothesis|proposed fix|"
    r"intended fix)\b|\b(?:the|this)\s+(?:issue|problem|bug)\s+"
    r"(?:is|occurs|comes from|stems from)\b|\b(?:the|this)\s+fix\s+"
    r"(?:should|requires?|is to)\b)",
    re.IGNORECASE,
)
_COMMAND_BLOCK = re.compile(r"```(?:mswea_bash_command)?\s*\n(.*?)\n```", re.DOTALL)
_FOCUSED_TEST = re.compile(
    r"(?:\bpytest\b|\btox\b|\bmake\s+test\b|"
    r"\bpython(?:3)?\s+(?:-m\s+(?:test|pytest)|-c\s+))",
    re.IGNORECASE,
)
_PATCH_CREATE = re.compile(r"\bgit\s+diff\b[^\n]*>\s*patch\.txt\b", re.IGNORECASE)
_PATCH_INSPECT = re.compile(
    r"(?:\bcat\b|\bsed\b|\bhead\b|\btail\b)[^\n]*\bpatch\.txt\b", re.IGNORECASE,
)


def _verification_guidance(messages: Sequence[Mapping[str, Any]]) -> str:
    """Return persistent phase guidance derived from the complete causal workflow."""

    assistant_indices = [
        index for index, row in enumerate(messages) if row.get("role") == "assistant"
    ]
    actions: list[tuple[int, str, str]] = []
    for position, index in enumerate(assistant_indices):
        content = str(messages[index].get("content") or "")
        blocks = _COMMAND_BLOCK.findall(content)
        command = blocks[-1] if blocks else content
        end = (
            assistant_indices[position + 1]
            if position + 1 < len(assistant_indices) else len(messages)
        )
        observation = "\n".join(
            str(row.get("content") or "") for row in messages[index + 1:end]
            if row.get("role") != "assistant"
        )
        actions.append((index, command, observation))
    mutations = [row for row in actions if _MUTATION.search(row[1])]
    if not mutations:
        return ""
    mutation = mutations[-1]
    after_mutation = [row for row in actions if row[0] > mutation[0]]
    diffs = [
        row for row in after_mutation
        if re.search(r"\bgit\s+diff\b", row[1], re.IGNORECASE)
        and not _PATCH_CREATE.search(row[1])
    ]
    if not diffs:
        return (
            "A source mutation is still unverified. Before any other exploration, inspect "
            "only the changed source diff with `git diff -- <changed source files>`."
        )
    diff = diffs[-1]
    output = re.search(r"<output>\s*(.*?)\s*</output>", diff[2], re.DOTALL)
    if output is not None and not output.group(1).strip():
        return (
            "The source diff is empty, so the prior mutation was a silent no-op. "
            "Reapply the intended edit with a content-matched operation rather than "
            "a line-number-only command, then inspect the diff again."
        )
    tests = [
        row for row in after_mutation
        if row[0] > diff[0] and _FOCUSED_TEST.search(row[1])
    ]
    if not tests:
        return (
            "The nonempty source diff is still unverified. Run the narrowest relevant "
            "reproduction or test now; do not resume source exploration."
        )
    verification = tests[-1]
    if "<returncode>0</returncode>" not in verification[2]:
        return (
            "Focused verification failed. Diagnose the reported failure and repair the "
            "source before creating patch.txt or submitting."
        )
    patch_creations = [
        row for row in after_mutation
        if row[0] > verification[0] and _PATCH_CREATE.search(row[1])
    ]
    if not patch_creations:
        return (
            "Focused verification passed. Create patch.txt now from only the modified "
            "source files; do not resume source discovery."
        )
    patch_creation = patch_creations[-1]
    inspections = [
        row for row in after_mutation
        if row[0] > patch_creation[0] and _PATCH_INSPECT.search(row[1])
    ]
    if not inspections:
        return "Inspect patch.txt now as the required separate verification command."
    inspection = inspections[-1]
    submitted = any(
        row[0] > inspection[0] and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in row[1]
        for row in after_mutation
    )
    if submitted:
        return ""
    return (
        "The submission patch has been inspected. Submit it now with the exact completion "
        "command required by the task; do not resume exploration."
    )


def enforce_consumption_action(
    response_payload: Mapping[str, Any], policy: str,
    *, logical_messages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enforce mechanical boundaries without choosing task-specific edits or tests.

    This is deliberately a separate PRA-agent treatment. The engine still produces the
    candidate response, but deterministic boundary actions replace candidates that skip
    a required diff, test, patch inspection, or submission step.
    """

    transformed = json.loads(json.dumps(response_payload, default=str))
    metadata = {
        "policy": policy,
        "action_enforced": False,
        "enforcement_reason": None,
        "original_content_sha256": None,
        "enforced_command": None,
    }
    if policy != "verification-enforced-v1":
        return transformed, metadata
    choices = transformed.get("choices") or ()
    if not choices or not isinstance(choices[0], Mapping):
        return transformed, metadata
    message = choices[0].get("message") or {}
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        return transformed, metadata

    content = str(message["content"])
    blocks = _COMMAND_BLOCK.findall(content)
    proposed = blocks[-1] if blocks else content
    guidance = _verification_guidance(logical_messages)
    replacement: str | None = None
    reason: str | None = None
    if "source mutation is still unverified" in guidance:
        if not re.search(r"\bgit\s+diff\b", proposed, re.IGNORECASE):
            replacement = "git diff"
            reason = "diff_required_after_mutation"
    elif "silent no-op" in guidance:
        if not _MUTATION.search(proposed):
            replacement = (
                "echo 'PRA_AGENT_BLOCKED: reapply the intended source edit with a "
                "content-matched mutation before continuing'"
            )
            reason = "repair_required_after_empty_diff"
    elif "nonempty source diff is still unverified" in guidance:
        if not _FOCUSED_TEST.search(proposed):
            replacement = (
                "echo 'PRA_AGENT_BLOCKED: run the narrowest relevant test before "
                "continuing'"
            )
            reason = "focused_test_required"
    elif "Focused verification passed" in guidance:
        if not _PATCH_CREATE.search(proposed):
            replacement = "git diff > patch.txt"
            reason = "patch_creation_required"
    elif "Inspect patch.txt" in guidance:
        if not _PATCH_INSPECT.search(proposed):
            replacement = "cat patch.txt"
            reason = "patch_inspection_required"
    elif "submission patch has been inspected" in guidance:
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" not in proposed:
            replacement = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
            reason = "submission_required"

    if replacement is None:
        return transformed, metadata
    original_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    enforced_content = (
        f"THOUGHT: PRA-agent enforced workflow boundary ({reason}).\n"
        f"```mswea_bash_command\n{replacement}\n```"
    )
    transformed["choices"][0]["message"]["content"] = enforced_content
    pra = dict(transformed.get("pra") or {})
    pra_agent = dict(pra.get("agent") or {})
    pra_agent.update({
        "policy": policy,
        "action_enforced": True,
        "enforcement_reason": reason,
        "original_content_sha256": original_hash,
        "enforced_command": replacement,
    })
    pra["agent"] = pra_agent
    transformed["pra"] = pra
    metadata.update(pra_agent)
    # Keep the rejected candidate in the private interaction trace, not in the
    # response delivered to the agent. This makes causal divergence auditable
    # without giving the next turn access to an action that was not executed.
    metadata["original_content"] = content
    return transformed, metadata


def _progress_pinned_indices(
    messages: Sequence[Mapping[str, Any]], excluded_indices: set[int],
    *, recent_turns: int = 2,
    recent_records_per_turn: int = 2,
    source_turns: int = 1,
    progress_turns: int = 1,
    mutation_turns: int = 1,
    verification_turns: int = 1,
    max_records_per_turn_before_chunking: int = 8,
    preserve_action_observation_pairs: bool = True,
) -> tuple[set[int], dict[str, set[int]]]:
    """Keep a small causal progress spine independently of lexical retrieval.

    Agent history is state, not a document collection. The latest two completed
    action/observation turns preserve local plan continuity. The most recent
    source-evidence, explicit hypothesis/progress-state, mutation, and
    verification turns preserve durable task progress even after they leave
    that recency window. Progress-state matching only inspects the narrative
    before a command block and requires explicit diagnostic or fix language;
    an ordinary verbose THOUGHT block is not sufficient.
    """

    candidates = [
        index for index in range(len(messages)) if index not in excluded_indices
    ]
    bundles = _turn_index_bundles(messages, candidates)
    pinned: set[int] = set()
    classes: dict[str, set[int]] = {
        "recent": set(),
        "source": set(),
        "progress_state": set(),
        "mutation": set(),
        "verification": set(),
    }

    def pin_bundle(bundle: Sequence[int], category: str) -> None:
        retained = list(bundle)
        if len(retained) > max_records_per_turn_before_chunking:
            retained = retained[-recent_records_per_turn:] if recent_records_per_turn else []
            if preserve_action_observation_pairs and bundle:
                action_index = next(
                    (
                        index for index in bundle
                        if str(messages[index].get("role")) == "assistant"
                    ),
                    None,
                )
                if action_index is not None and retained:
                    retained.insert(0, action_index)
        pinned.update(retained)
        classes[category].update(retained)

    for bundle in (bundles[-recent_turns:] if recent_turns else ()):
        pin_bundle(bundle, "recent")
    for category, pattern, keep in (
        ("source", _SOURCE_EVIDENCE, source_turns),
        ("progress_state", _PROGRESS_STATE, progress_turns),
        ("mutation", _MUTATION, mutation_turns),
        ("verification", _VERIFICATION, verification_turns),
    ):
        if keep == 0:
            continue
        matched = [
            bundle for bundle in bundles
            if any(
                messages[index].get("role") == "assistant"
                and pattern.search(
                    str(messages[index].get("content", "")).split("```", 1)[0]
                    if category == "progress_state"
                    else str(messages[index].get("content", ""))
                )
                for index in bundle
            )
        ]
        for bundle in matched[-keep:]:
            pin_bundle(bundle, category)
    return pinned, classes


def _task_aware_query(
    messages: Sequence[Mapping[str, Any]], task_indices: set[int],
    mandatory_indices: set[int],
) -> str:
    """Query with goal, current action, and observation instead of output alone."""

    task = "\n".join(
        _take_words(str(messages[index].get("content", "")), 256)
        for index in sorted(task_indices)
    )
    active = "\n".join(
        _take_words(str(messages[index].get("content", "")), 192, from_end=True)
        for index in sorted(mandatory_indices)
        if messages[index].get("role") != "system"
    )
    return "\n".join(part for part in (task, active) if part)


def _sort_segments(segments: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Restore exact causal order after merging pinned and retrieved records."""

    def key(row: tuple[str, str]) -> tuple[int, int, str]:
        match = re.fullmatch(r"m(\d+)-(\d+)-(.+)", row[0])
        if match is None:
            return (sys.maxsize, sys.maxsize, row[0])
        return (int(match.group(1)), int(match.group(2)), match.group(3))

    return sorted(segments, key=key)


def _truncate_recent(
    messages: Sequence[Mapping[str, Any]], candidate_indices: Sequence[int], budget: int,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    remaining = budget
    for index in reversed(candidate_indices):
        if remaining <= 0:
            break
        row = dict(messages[index])
        words = _words(row.get("content"))
        if len(words) > remaining:
            row["content"] = " ".join(words[-remaining:])
            words = words[-remaining:]
        selected.append((index, row))
        remaining -= len(words)
    return selected


def _segments(
    messages: Sequence[Mapping[str, Any]], candidate_indices: Sequence[int], segment_tokens: int,
) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    for index in candidate_indices:
        role = str(messages[index].get("role", "unknown"))
        content = str(messages[index].get("content", ""))
        for segment_index, text in enumerate(
            _split_record_text(content, segment_tokens, natural=(role in {"tool", "user"}))
        ):
            if text:
                segments.append((f"m{index}-{segment_index}-{role}", text))
    return segments


def _split_record_text(
    content: str, segment_tokens: int, *, natural: bool,
) -> list[str]:
    """Split one record without allowing a child or overlap to cross records.

    Short records stay intact. Oversized tool/user observations first break at
    file, diff-hunk, stack-frame, or test-case boundaries; only an oversized
    natural region falls back to token windows. The slices are disjoint and
    rejoin byte-for-byte, so budget accounting never charges overlap twice.
    """

    words = list(_TOKEN.finditer(content))
    if not words:
        return [content] if content else []
    if len(words) <= segment_tokens:
        return [content]

    region_starts = [0]
    if natural:
        cursor = 0
        for line in content.splitlines(keepends=True):
            if cursor and _NATURAL_OBSERVATION_BOUNDARY.match(line):
                region_starts.append(cursor)
            cursor += len(line)
    region_starts.append(len(content))

    result: list[str] = []
    for region_start, region_end in zip(region_starts, region_starts[1:]):
        region = content[region_start:region_end]
        region_words = list(_TOKEN.finditer(region))
        for offset in range(0, len(region_words), segment_tokens):
            end = (
                region_words[offset + segment_tokens].start()
                if offset + segment_tokens < len(region_words)
                else len(region)
            )
            start = 0 if offset == 0 else region_words[offset].start()
            text = region[start:end]
            if text:
                result.append(text)
    if "".join(result) != content:
        raise AssertionError("record-aligned segmentation changed observation text")
    return result


def _turn_bundles(
    messages: Sequence[Mapping[str, Any]],
    candidate_indices: Sequence[int],
    segment_tokens: int,
    *,
    preserve_action_observation_pairs: bool = True,
) -> list[list[tuple[str, str]]]:
    """Return causal selection units for an agent transcript.

    An assistant action and every following user/tool observation before the
    next assistant action are one indivisible unit.  Selecting independent
    messages can drop the observation between two retained assistant actions,
    which produces a chat that Qwen's template correctly rejects.  User-only
    runs (for example parser-generated format errors after a rejected action)
    remain valid standalone units; the rejected assistant text is deliberately
    absent from mini-swe-agent's logical transcript.
    """

    grouped = (
        _turn_index_bundles(messages, candidate_indices)
        if preserve_action_observation_pairs
        else [[index] for index in sorted(dict.fromkeys(candidate_indices))]
    )
    bundles: list[list[tuple[str, str]]] = []
    for indices_in_bundle in grouped:
        bundle = _segments(messages, indices_in_bundle, segment_tokens)
        if bundle:
            bundles.append(bundle)
    return bundles


def _turn_index_bundles(
    messages: Sequence[Mapping[str, Any]], candidate_indices: Sequence[int],
) -> list[list[int]]:
    """Group historical message indices into complete causal agent turns."""

    indices = sorted(dict.fromkeys(int(index) for index in candidate_indices))
    grouped: list[list[int]] = []
    current: list[int] = []
    current_starts_with_assistant = False
    for index in indices:
        role = str(messages[index].get("role", "unknown"))
        if role == "assistant":
            if current:
                grouped.append(current)
            current = [index]
            current_starts_with_assistant = True
            continue
        if not current:
            current = [index]
            current_starts_with_assistant = False
        elif current_starts_with_assistant:
            current.append(index)
        else:
            current.append(index)
    if current:
        grouped.append(current)

    complete: list[list[int]] = []
    for indices_in_bundle in grouped:
        # A dangling assistant action has no causal result and may not be
        # surfaced as selected history.  This is defensive: normal agent
        # requests always make the latest observation mandatory.
        if (
            str(messages[indices_in_bundle[0]].get("role")) == "assistant"
            and len(indices_in_bundle) == 1
            and not any(
                later > indices_in_bundle[0]
                and str(messages[later].get("role")) not in {"assistant", "system"}
                for later in range(len(messages))
            )
        ):
            continue
        complete.append(indices_in_bundle)
    return complete


def _segment_metadata(
    segment_id: str, messages: Sequence[Mapping[str, Any]] | None = None,
    *, selection_policy: str = "task-aware-v1",
) -> dict[str, Any]:
    """Expose causal position without relying on resource retrieval order."""

    metadata: dict[str, Any] = {"selection_policy": selection_policy}
    match = re.fullmatch(r"m(\d+)-(\d+)-(.+)", segment_id)
    if match is not None:
        metadata.update(
            message_index=int(match.group(1)),
            segment_index=int(match.group(2)),
            role=match.group(3),
            parent_record_id=f"m{match.group(1)}",
        )
        if messages is not None:
            message_index = int(match.group(1))
            metadata["causal_group_id"] = _causal_group_id(messages, message_index)
            record = messages[message_index]
            for name in (
                "tool_call_id", "name", "return_code", "exit_code",
                "mutation_status", "result_metadata", "status",
            ):
                if name in record:
                    metadata[name] = record[name]
    return metadata


def _causal_group_id(
    messages: Sequence[Mapping[str, Any]], message_index: int,
) -> str:
    """Give an assistant action and its observations one stable causal ID."""

    if str(messages[message_index].get("role")) == "assistant":
        return f"turn:m{message_index}"
    for index in range(message_index - 1, -1, -1):
        role = str(messages[index].get("role"))
        if role == "assistant":
            return f"turn:m{index}"
        if role == "system":
            break
    return f"record:m{message_index}"


def _select_turn_bundles(
    bundles: Sequence[Sequence[tuple[str, str]]], query: str, budget: int,
    *, round_up: bool = True,
    count_tokens: Callable[[str], int] | None = None,
) -> list[tuple[str, str]]:
    """Rank whole turns and round the retention floor up to the next bundle."""

    if not bundles or budget <= 0:
        return []
    count = count_tokens or _count_tokens
    texts = ["".join(text for _, text in bundle) for bundle in bundles]
    costs = [sum(count(text) for _, text in bundle) for bundle in bundles]
    index = LargeRecordIndex(texts)
    result = index.search(
        query, policy=LargeRecordSearchPolicy.HYBRID,
        top_k=len(bundles), candidate_limit=len(bundles),
    )
    ranked = [int(hit.unit_id.split(":", 1)[1]) for hit in result.hits]
    ranked.extend(range(len(bundles) - 1, -1, -1))
    selected: set[int] = set()
    remaining = budget
    for bundle_index in ranked:
        if bundle_index in selected:
            continue
        if not round_up and costs[bundle_index] > remaining:
            continue
        selected.add(bundle_index)
        remaining -= costs[bundle_index]
        if remaining <= 0:
            break

    return [
        segment
        for bundle_index, bundle in enumerate(bundles)
        if bundle_index in selected
        for segment in bundle
    ]


def _select_matched_causal_token_tail_indices(
    messages: Sequence[Mapping[str, Any]],
    candidate_indices: Sequence[int],
    budget: int,
    *,
    count_tokens: Callable[[str], int],
) -> list[int]:
    """Select a contiguous newest-first tail of complete causal turns.

    This is the whole-record native transfer of Paper 8.5's matched token-tail
    control.  The ordinary-text study may compact an oversized boundary tool
    observation; the native vLLM gate deliberately does not claim that
    capability until record-internal K/V spans are supported.  Consequently a
    boundary turn that does not fit ends selection and leaves the remainder of
    the strict ceiling unused.
    """

    if budget <= 0:
        return []
    bundles = _turn_index_bundles(messages, candidate_indices)
    selected: list[Sequence[int]] = []
    remaining = int(budget)
    for bundle in reversed(bundles):
        cost = sum(
            count_tokens(str(messages[index].get("content") or ""))
            for index in bundle
        )
        if cost > remaining:
            break
        selected.append(bundle)
        remaining -= cost
    selected_ids = {index for bundle in selected for index in bundle}
    return [
        index
        for bundle in bundles
        for index in bundle
        if index in selected_ids
    ]


def _trace(
    request_index: int, session_id: str, mode: ContextTreatment, budget_fraction: float,
    logical: int, mandatory: int, selected: int, physical: int,
    candidates: int, selected_segments: int, selected_resource_digest: str | None,
    route_time_s: float,
    *,
    token_estimator: str = "whitespace_v1",
    agent_history_selection_policy: str = "task-aware-v1",
    requested_budget_tokens: int | None = None,
    logical_budget_unused_tokens: int | None = None,
    mandatory_overflow_tokens: int = 0,
) -> TreatmentTrace:
    avoided = max(0, logical - physical)
    return TreatmentTrace(
        request_index, session_id, mode.value, budget_fraction, logical, mandatory, selected,
        physical, avoided, avoided / logical if logical else 0.0,
        candidates, selected_segments, selected_resource_digest, route_time_s,
        token_estimator, agent_history_selection_policy, requested_budget_tokens,
        logical_budget_unused_tokens, mandatory_overflow_tokens,
    )


def _selection_digest(rows: Sequence[tuple[str, str]]) -> str:
    """Fingerprint ordered selected records without retaining their content in metadata."""

    material = [
        {
            "resource_id": resource_id,
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        for resource_id, text in rows
    ]
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _words(value: Any) -> list[str]:
    if isinstance(value, str):
        return _TOKEN.findall(value)
    if value is None:
        return []
    return _TOKEN.findall(json.dumps(value, sort_keys=True, default=str))


def _take_words(value: str, limit: int, *, from_end: bool = False) -> str:
    """Take a word-bounded slice while retaining original internal formatting."""

    matches = list(_TOKEN.finditer(value))
    if limit <= 0 or not matches:
        return ""
    if len(matches) <= limit:
        return value
    if from_end:
        return value[matches[-limit].start():]
    return value[:matches[limit - 1].end()]


def _count_tokens(value: Any) -> int:
    return len(_words(value))


class TreatmentProxy:
    """Small OpenAI-compatible proxy that persists a trace for every request."""

    def __init__(
        self, target_base_url: str, *, mode: ContextTreatment | str,
        budget_fraction: float, trace_path: Path,
        selection_record_path: Path | None = None,
        selection_replay_path: Path | None = None,
        interaction_trace_path: Path | None = None,
        request_overrides: Mapping[str, Any] | None = None,
        recent_completed_turns: int = 2,
        recent_records_per_turn: int = 2,
        recent_source_turns: int = 1,
        recent_progress_turns: int = 1,
        recent_mutation_turns: int = 1,
        recent_verification_turns: int = 1,
        large_record_chunk_tokens: int = 256,
        max_records_per_turn_before_chunking: int = 8,
        preserve_action_observation_pairs: bool = True,
        causal_bundle_round_up: bool = True,
        consumption_policy: str = "standard",
        session_namespace: str | None = None,
        agent_history_selection_policy: str = "task-aware-v1",
        count_tokens: Callable[[str], int] | None = None,
        tokenizer_identity: str = "whitespace_v1",
    ) -> None:
        if selection_record_path is not None and selection_replay_path is not None:
            raise ValueError("selection recording and replay are mutually exclusive")
        self.target_base_url = target_base_url.rstrip("/")
        self.mode = ContextTreatment(mode)
        self.budget_fraction = budget_fraction
        self.trace_path = trace_path
        self.selection_record_path = selection_record_path
        self.selection_replay_path = selection_replay_path
        self.interaction_trace_path = interaction_trace_path
        self.request_overrides = dict(request_overrides or {})
        self.retention_policy = PRAAgentRetentionPolicy(
            recent_completed_turns=recent_completed_turns,
            recent_records_per_turn=recent_records_per_turn,
            recent_source_turns=recent_source_turns,
            recent_progress_turns=recent_progress_turns,
            recent_mutation_turns=recent_mutation_turns,
            recent_verification_turns=recent_verification_turns,
            large_record_chunk_tokens=large_record_chunk_tokens,
            max_records_per_turn_before_chunking=max_records_per_turn_before_chunking,
            preserve_action_observation_pairs=preserve_action_observation_pairs,
            causal_bundle_round_up=causal_bundle_round_up,
        )
        if consumption_policy not in CONSUMPTION_POLICIES:
            raise ValueError(
                f"unknown consumption policy {consumption_policy!r}; "
                f"expected one of {CONSUMPTION_POLICIES}"
            )
        self.consumption_policy = consumption_policy
        if agent_history_selection_policy not in AGENT_HISTORY_SELECTION_POLICIES:
            raise ValueError(
                "unknown agent-history selection policy "
                f"{agent_history_selection_policy!r}"
            )
        self.agent_history_selection_policy = agent_history_selection_policy
        self.count_tokens = count_tokens or _count_tokens
        self.tokenizer_identity = str(tokenizer_identity)
        self.session_namespace = (
            None if session_namespace is None else str(session_namespace)
        )
        self._frozen_selections = _load_selection_fixture(selection_replay_path)
        self._lock = threading.Lock()
        self._request_index = 0
        self._native_session_ids: set[str] = set()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def _json_error(self, status: int, code: str, error: Exception) -> None:
                body = json.dumps({
                    "error": code,
                    "message": str(error),
                }).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _handle_forward(self) -> None:
                try:
                    proxy._forward(self)
                except FrozenReplayDivergence as error:
                    self._json_error(409, "frozen_replay_diverged", error)
                except (ValueError, TypeError, json.JSONDecodeError) as error:
                    self._json_error(400, "invalid_treatment_request", error)
                except (urllib.error.URLError, TimeoutError) as error:
                    self._json_error(502, "treatment_upstream_unavailable", error)
                except Exception as error:
                    self._json_error(500, "treatment_proxy_internal_error", error)

            def do_GET(self) -> None:  # noqa: N802
                if urlparse(self.path).path == "/health":
                    payload = json.dumps(proxy.health()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self._handle_forward()

            def do_POST(self) -> None:  # noqa: N802
                self._handle_forward()

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{host}:{self._server.server_port}/v1"

    def health(self) -> dict[str, Any]:
        """Advertise the local proxy's effective, fail-closed treatment contract."""

        gateway_mode = {
            ContextTreatment.PASSTHROUGH: "G00",
            ContextTreatment.HEADROOM: "HEADROOM",
            ContextTreatment.PRA_SELECTED_CONTEXT: "G10",
            ContextTreatment.TRUNCATION: "TRUNCATION",
        }.get(self.mode)
        return {
            "status": "ok",
            "gateway_mode": gateway_mode,
            "protocol_version": "paper4.5-agent-treatment-v1",
            "effective_capabilities": {
                "selected_context": self.mode is ContextTreatment.PRA_SELECTED_CONTEXT,
                "native_kv": False,
            },
        }

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # A treatment proxy owns the engine sessions it creates.  Reusing the
        # task-derived ID in the next arm would otherwise attach its first
        # request to the preceding arm's completed ledger and fail the live-K/V
        # append-stability invariant before selection even begins.
        with self._lock:
            native_session_ids = tuple(sorted(self._native_session_ids))
            self._native_session_ids.clear()
        endpoint = self.target_base_url.removesuffix("/v1")
        for session_id in native_session_ids:
            request = urllib.request.Request(
                f"{endpoint}/v1/pra/sessions/{quote(session_id, safe='')}",
                method="DELETE",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()
            except Exception:  # noqa: BLE001 - best-effort lifecycle cleanup
                # Cleanup must not replace a completed benchmark result with a
                # transport error.  The run-scoped ID still prevents a stale
                # engine session from contaminating a later arm.
                pass

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        trace = None
        request_index = None
        input_digest = None
        if handler.command == "POST" and urlparse(handler.path).path == "/v1/chat/completions":
            payload = json.loads(body.decode("utf-8"))
            payload.update(self.request_overrides)
            envelope = dict(payload.get("pra") or {})
            metadata = dict(envelope.get("metadata") or {})
            requested_retention = dict(metadata.get("retention_policy") or {})
            effective_retention = PRAAgentRetentionPolicy.from_mapping({
                **self.retention_policy.to_dict(),
                **requested_retention,
            })
            logical_payload = json.loads(json.dumps(payload, default=str))
            with self._lock:
                self._request_index += 1
                request_index = self._request_index
            input_digest = _selection_input_digest(payload.get("messages", ()))
            frozen = self._frozen_selections.get(input_digest)
            if self.selection_replay_path is not None and frozen is None:
                self._record_interaction_event({
                    "event": "frozen_replay_divergence",
                    "request_index": request_index,
                    "request_input_sha256": input_digest,
                    "error": "frozen_replay_diverged",
                    "message": (
                        "frozen selection replay has no exact request match; "
                        "the paired trajectories have diverged"
                    ),
                    "logical_payload": logical_payload,
                })
                raise FrozenReplayDivergence(
                    "frozen selection replay has no exact request match for "
                    f"{input_digest}; the paired trajectories have diverged"
                )
            payload, trace = transform_chat_payload(
                payload, mode=self.mode, budget_fraction=self.budget_fraction,
                request_index=request_index, frozen_selection=frozen,
                segment_tokens=effective_retention.large_record_chunk_tokens,
                recent_completed_turns=effective_retention.recent_completed_turns,
                recent_records_per_turn=effective_retention.recent_records_per_turn,
                recent_source_turns=effective_retention.recent_source_turns,
                recent_progress_turns=effective_retention.recent_progress_turns,
                recent_mutation_turns=effective_retention.recent_mutation_turns,
                recent_verification_turns=effective_retention.recent_verification_turns,
                max_records_per_turn_before_chunking=(
                    effective_retention.max_records_per_turn_before_chunking
                ),
                preserve_action_observation_pairs=(
                    effective_retention.preserve_action_observation_pairs
                ),
                causal_bundle_round_up=effective_retention.causal_bundle_round_up,
                agent_history_selection_policy=(
                    self.agent_history_selection_policy
                ),
                count_tokens=self.count_tokens,
                tokenizer_identity=self.tokenizer_identity,
            )
            native_session_id = trace.session_id
            if self.session_namespace and isinstance(payload.get("pra"), Mapping):
                scoped_session_id = hashlib.sha256(
                    f"{self.session_namespace}\0{trace.session_id}".encode("utf-8")
                ).hexdigest()[:24]
                envelope = dict(payload["pra"])
                envelope["session_id"] = scoped_session_id
                payload["pra"] = envelope
                # Keep telemetry keyed by the stable logical task session so
                # normalization can join it to the trajectory.  The scoped
                # identifier belongs only to the engine lifecycle.
                native_session_id = scoped_session_id
            if self.mode in {
                ContextTreatment.DIRECT_NATIVE_PRA,
                ContextTreatment.GATEWAY_NATIVE_PRA,
            }:
                with self._lock:
                    self._native_session_ids.add(native_session_id)
            payload, policy_tokens = apply_consumption_policy(
                payload, self.consumption_policy,
                logical_messages=logical_payload.get("messages", ()),
            )
            if policy_tokens:
                physical_tokens = trace.physical_input_tokens_estimate + policy_tokens
                avoided_tokens = max(
                    0, trace.logical_input_tokens_estimate - physical_tokens,
                )
                trace = replace(
                    trace,
                    physical_input_tokens_estimate=physical_tokens,
                    tokens_avoided_estimate=avoided_tokens,
                    token_saving_fraction_estimate=(
                        avoided_tokens / trace.logical_input_tokens_estimate
                        if trace.logical_input_tokens_estimate else 0.0
                    ),
                )
            if self.selection_record_path is not None:
                resources = (payload.get("pra") or {}).get("resources") or ()
                self._record_selection(input_digest, trace, resources)
            self._record_interaction_event({
                "event": "request",
                "request_index": request_index,
                "request_input_sha256": input_digest,
                "selected_resource_digest": trace.selected_resource_digest,
                "logical_payload": logical_payload,
                "physical_payload": payload,
            })
            body = json.dumps(payload).encode("utf-8")
        target = self.target_base_url.removesuffix("/v1") + handler.path
        headers = {
            key: value for key, value in handler.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        request = urllib.request.Request(target, data=body if handler.command == "POST" else None,
                                         headers=headers, method=handler.command)
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                response_body = response.read()
                handler.send_response(response.status)
                handler.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as error:
            response_body = error.read()
            handler.send_response(error.code)
            handler.send_header("Content-Type", error.headers.get("Content-Type", "application/json"))
        if request_index is not None:
            try:
                response_payload: Any = json.loads(response_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                response_payload = {"raw_body_sha256": hashlib.sha256(response_body).hexdigest()}
            enforcement = None
            if isinstance(response_payload, Mapping):
                response_payload, enforcement = enforce_consumption_action(
                    response_payload,
                    self.consumption_policy,
                    logical_messages=logical_payload.get("messages", ()),
                )
                response_body = json.dumps(response_payload).encode("utf-8")
            self._record_interaction_event({
                "event": "response",
                "request_index": request_index,
                "request_input_sha256": input_digest,
                "payload": response_payload,
                "pra_agent_enforcement": enforcement,
            })
        handler.send_header("Content-Length", str(len(response_body)))
        handler.end_headers()
        handler.wfile.write(response_body)
        if trace is not None:
            trace_row = asdict(trace)
            trace_row.update(_response_execution_metrics(response_body))
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(trace_row, sort_keys=True) + "\n")

    def _record_interaction_event(self, row: Mapping[str, Any]) -> None:
        """Persist the exact logical, physical, and model-visible exchange."""

        if self.interaction_trace_path is None:
            return
        self.interaction_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.interaction_trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def _record_selection(
        self, input_digest: str, trace: TreatmentTrace, resources: Sequence[Mapping[str, Any]],
    ) -> None:
        """Persist replayable selected content while telemetry retains hashes only."""

        row = {
            "request_input_sha256": input_digest,
            "session_id": trace.session_id,
            "selected_resource_digest": trace.selected_resource_digest,
            "resources": [
                {"resource_id": str(item["resource_id"]), "text": str(item["text"])}
                for item in resources
            ],
        }
        assert self.selection_record_path is not None
        self.selection_record_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.selection_record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")


def _selection_input_digest(messages: Sequence[Mapping[str, Any]]) -> str:
    """Bind replay to the complete agent-visible request, including trajectory order."""

    encoded = json.dumps(messages, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _response_execution_metrics(body: bytes) -> dict[str, Any]:
    """Extract only cache/native counters physically reported by the endpoint."""

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            "prefix_cache_observed": False,
            "prefix_cached_tokens": None,
            "engine_cached_tokens_total": None,
            "native_tokens": None,
            "wire_tokens": None,
            "physical_kv_copy": None,
            "physical_kv_copy_bytes": None,
            "total_kv_copy_bytes": None,
            "canonical_suffix_graft_d2d_bytes": None,
            "host_to_device_bytes": None,
            "selected_kv_tokens": None,
            "selected_text_reencoded_tokens": None,
            "selected_history_reencoded_tokens": None,
            "realized_retention_fraction": None,
            "engine_reported_history_kv_retention_fraction": None,
            "consumer_temporary_bytes": None,
            "consumer_temporary_peak_bytes": None,
            "fused_attention_calls": None,
            "full_retention": None,
            "resource_update_mode": None,
            "resource_prefix_cached_tokens": None,
            "resource_evaluated_tokens": None,
            "resource_total_tokens": None,
        }
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    pra = payload.get("pra") or {}
    timings = payload.get("timings") or {}
    cached = details.get("cached_tokens")
    if cached is None:
        cached = pra.get("prefix_cached_tokens", timings.get("cache_n"))
    native_tokens = pra.get("native_tokens")
    wire_tokens = pra.get("wire_tokens")
    physical_copy = pra.get("physical_kv_copy")
    physical_copy_bytes = pra.get("physical_kv_copy_bytes")
    total_kv_copy_bytes = pra.get("total_kv_copy_bytes")
    canonical_suffix_graft_d2d_bytes = pra.get(
        "canonical_suffix_graft_d2d_bytes"
    )
    host_to_device_bytes = pra.get("host_to_device_bytes")
    selected_kv = pra.get("selected_kv_tokens")
    selected_reencoded = pra.get("selected_text_reencoded_tokens")
    selected_history_reencoded = pra.get("selected_history_reencoded_tokens")
    realized_retention = pra.get("realized_retention_fraction")
    consumer_temporary = pra.get(
        "consumer_temporary_bytes",
        pra.get("transient_attention_bytes", pra.get("temporary_allocation_bytes")),
    )
    consumer_temporary_peak = pra.get(
        "consumer_temporary_peak_bytes", pra.get("consumer_peak_delta_bytes")
    )
    fused_attention_calls = pra.get("fused_attention_calls")
    full_retention = pra.get("full_retention")
    engine_cached_tokens = pra.get("engine_cached_tokens_total")
    resource_update_mode = pra.get("resource_update_mode")
    resource_cached = pra.get("resource_prefix_cached_tokens")
    resource_evaluated = pra.get("resource_evaluated_tokens")
    resource_total = pra.get("resource_total_tokens")
    for row in payload.get("pra_trace") or ():
        if not isinstance(row, Mapping):
            continue
        if cached is None:
            cached = row.get("prefix_cached_tokens", row.get("cache_n"))
        if native_tokens is None:
            native_tokens = row.get("native_tokens")
        if wire_tokens is None:
            wire_tokens = row.get("wire_tokens")
        if physical_copy is None:
            physical_copy = row.get("physical_kv_copy")
        if physical_copy_bytes is None:
            physical_copy_bytes = row.get("physical_kv_copy_bytes")
        if total_kv_copy_bytes is None:
            total_kv_copy_bytes = row.get("total_kv_copy_bytes")
        if canonical_suffix_graft_d2d_bytes is None:
            canonical_suffix_graft_d2d_bytes = row.get(
                "canonical_suffix_graft_d2d_bytes"
            )
        if host_to_device_bytes is None:
            host_to_device_bytes = row.get("host_to_device_bytes")
        if selected_kv is None:
            selected_kv = row.get("selected_kv_tokens")
        if selected_reencoded is None:
            selected_reencoded = row.get("selected_text_reencoded_tokens")
        if selected_history_reencoded is None:
            selected_history_reencoded = row.get("selected_history_reencoded_tokens")
        if realized_retention is None:
            realized_retention = row.get("realized_retention_fraction")
        if consumer_temporary is None:
            consumer_temporary = row.get(
                "consumer_temporary_bytes",
                row.get("transient_attention_bytes", row.get("temporary_allocation_bytes")),
            )
        if consumer_temporary_peak is None:
            consumer_temporary_peak = row.get(
                "consumer_temporary_peak_bytes", row.get("consumer_peak_delta_bytes")
            )
        if fused_attention_calls is None:
            fused_attention_calls = row.get("fused_attention_calls")
        if full_retention is None:
            full_retention = row.get("full_retention")
        if engine_cached_tokens is None:
            engine_cached_tokens = row.get("engine_cached_tokens_total")
        if resource_update_mode is None:
            resource_update_mode = row.get("resource_update_mode")
        if resource_cached is None:
            resource_cached = row.get("resource_prefix_cached_tokens")
        if resource_evaluated is None:
            resource_evaluated = row.get("resource_evaluated_tokens")
        if resource_total is None:
            resource_total = row.get("resource_total_tokens")
    return {
        "prefix_cache_observed": cached is not None,
        "prefix_cached_tokens": int(cached) if cached is not None else None,
        "engine_cached_tokens_total": (
            int(engine_cached_tokens) if engine_cached_tokens is not None else None
        ),
        "native_tokens": int(native_tokens) if native_tokens is not None else None,
        "wire_tokens": int(wire_tokens) if wire_tokens is not None else None,
        "physical_kv_copy": bool(physical_copy) if physical_copy is not None else None,
        "physical_kv_copy_bytes": (
            int(physical_copy_bytes) if physical_copy_bytes is not None else None
        ),
        "total_kv_copy_bytes": (
            int(total_kv_copy_bytes) if total_kv_copy_bytes is not None else None
        ),
        "canonical_suffix_graft_d2d_bytes": (
            int(canonical_suffix_graft_d2d_bytes)
            if canonical_suffix_graft_d2d_bytes is not None else None
        ),
        "host_to_device_bytes": (
            int(host_to_device_bytes) if host_to_device_bytes is not None else None
        ),
        "selected_kv_tokens": int(selected_kv) if selected_kv is not None else None,
        "selected_text_reencoded_tokens": (
            int(selected_reencoded) if selected_reencoded is not None else None
        ),
        "selected_history_reencoded_tokens": (
            int(selected_history_reencoded)
            if selected_history_reencoded is not None
            else int(selected_reencoded) if selected_reencoded is not None else None
        ),
        "realized_retention_fraction": (
            float(realized_retention) if realized_retention is not None else None
        ),
        "engine_reported_history_kv_retention_fraction": (
            float(realized_retention) if realized_retention is not None else None
        ),
        "consumer_temporary_bytes": (
            int(consumer_temporary) if consumer_temporary is not None else None
        ),
        "consumer_temporary_peak_bytes": (
            int(consumer_temporary_peak) if consumer_temporary_peak is not None else None
        ),
        "fused_attention_calls": (
            int(fused_attention_calls) if fused_attention_calls is not None else None
        ),
        "full_retention": (
            bool(full_retention) if full_retention is not None else None
        ),
        "resource_update_mode": resource_update_mode,
        "resource_prefix_cached_tokens": (
            int(resource_cached) if resource_cached is not None else None
        ),
        "resource_evaluated_tokens": (
            int(resource_evaluated) if resource_evaluated is not None else None
        ),
        "resource_total_tokens": (
            int(resource_total) if resource_total is not None else None
        ),
    }


def _load_selection_fixture(path: Path | None) -> dict[str, list[tuple[str, str]]]:
    """Load a direct-run fixture and reject ambiguous duplicate request identities."""

    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"selection replay fixture does not exist: {path}")
    selections: dict[str, list[tuple[str, str]]] = {}
    digests: dict[str, str | None] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        request_digest = str(row["request_input_sha256"])
        resources = [
            (str(item["resource_id"]), str(item["text"]))
            for item in row.get("resources", ())
        ]
        selection_digest = _selection_digest(resources)
        expected_digest = row.get("selected_resource_digest")
        if expected_digest != selection_digest:
            raise ValueError(
                f"selection fixture line {line_number} failed its content digest"
            )
        if request_digest in selections and digests[request_digest] != selection_digest:
            raise ValueError(
                f"selection fixture has conflicting rows for request {request_digest}"
            )
        selections[request_digest] = resources
        digests[request_digest] = selection_digest
    return selections
