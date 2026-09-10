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
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from pra_hf.large_record_index import LargeRecordIndex, LargeRecordSearchPolicy


_TOKEN = re.compile(r"\S+")


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
    segment_tokens: int = 256,
    recent_completed_turns: int = 2,
    recent_mutation_turns: int = 1,
    recent_verification_turns: int = 1,
    frozen_selection: Sequence[tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], TreatmentTrace]:
    """Apply a matched budget, optionally replaying an exact recorded selection."""

    mode = ContextTreatment(mode)
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if segment_tokens <= 0:
        raise ValueError("segment_tokens must be positive")
    retention_counts = (
        recent_completed_turns, recent_mutation_turns,
        recent_verification_turns,
    )
    if any(count < 0 for count in retention_counts):
        raise ValueError("retained turn counts must be non-negative")
    transformed = dict(payload)
    messages = [dict(row) for row in payload.get("messages", ())]
    if not messages:
        raise ValueError("chat payload requires messages")
    session_id = session_id_for_messages(messages)
    logical_tokens = sum(_count_tokens(row.get("content")) for row in messages)
    if mode in {ContextTreatment.PASSTHROUGH, ContextTreatment.HEADROOM}:
        if frozen_selection is not None:
            raise ValueError(f"{mode.value} mode cannot replay a context selection")
        return transformed, _trace(
            request_index, session_id, mode, budget_fraction, logical_tokens, logical_tokens,
            0, logical_tokens, 0, 0, None, 0.0,
        )

    mandatory_indices = _mandatory_indices(messages)
    task_indices = _pinned_task_indices(messages, mandatory_indices)
    progress_indices = _progress_pinned_indices(
        messages, mandatory_indices | task_indices,
        recent_turns=recent_completed_turns,
        mutation_turns=recent_mutation_turns,
        verification_turns=recent_verification_turns,
    )
    pinned_indices = task_indices | progress_indices
    mandatory_tokens = sum(_count_tokens(messages[index].get("content")) for index in mandatory_indices)
    pinned_tokens = sum(_count_tokens(messages[index].get("content")) for index in pinned_indices)
    target_tokens = max(
        mandatory_tokens + pinned_tokens,
        math.ceil(logical_tokens * budget_fraction),
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
        selected_tokens = sum(_count_tokens(row[1].get("content")) for row in selected)
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
        bundles = _turn_bundles(messages, candidate_indices, segment_tokens)
        segments = [segment for bundle in bundles for segment in bundle]
        selected_texts = (
            list(frozen_selection)
            if frozen_selection is not None
            else _sort_segments([
                *pinned_segments,
                *_select_turn_bundles(bundles, query, available_tokens),
            ])
        )
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
        selected_tokens = sum(_count_tokens(text) for _, text in selected_texts)
        if selected_tokens > resource_budget:
            raise ValueError(
                "frozen selection exceeds the matched request context budget: "
                f"{selected_tokens} > {resource_budget}"
            )
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
                "metadata": _segment_metadata(segment_id),
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
                "max_selected_tokens": max(1, resource_budget),
            },
            "allow_text_fallback": not native_requested,
            "required_capabilities": ["logical_refs", "native_kv"] if native_requested else [],
            "pra_policy": {"profile": "swebench-balanced-v1"},
            "metadata": {
                "requested_mode": "native-memory" if native_requested else "selected-context",
                "connection": (
                    "direct" if mode is ContextTreatment.DIRECT_NATIVE_PRA else "gateway"
                ),
                "benchmark_fairness": "agent-visible-messages-only",
                "budget_fraction": float(budget_fraction),
                "retention_policy": {
                    "recent_completed_turns": recent_completed_turns,
                    "recent_mutation_turns": recent_mutation_turns,
                    "recent_verification_turns": recent_verification_turns,
                },
                # Moving completed chat turns from the inline message list to
                # typed resources is an intentional representation change,
                # not a destructive rewrite of the logical agent history.
                # G11 uses this marker to retain the engine session and lets
                # the native adapter validate/backtrack the exact token prefix.
                "history_projection": "detached-agent-trajectory-v1",
                "selection_complete": selected_segments == candidate_segments,
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
            },
        })
        transformed["pra"] = envelope
        selected_digest = _selection_digest(selected_texts)
    physical_tokens = mandatory_tokens + selected_tokens
    return transformed, _trace(
        request_index, session_id, mode, budget_fraction, logical_tokens, mandatory_tokens,
        selected_tokens, physical_tokens, candidate_segments, selected_segments,
        selected_digest, time.perf_counter() - started,
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
    mutation_turns: int = 1,
    verification_turns: int = 1,
) -> set[int]:
    """Keep a small causal progress spine independently of lexical retrieval.

    Agent history is state, not a document collection. The latest two completed
    action/observation turns preserve local plan continuity. The most recent
    mutation and verification turns preserve durable task progress even after
    they leave that recency window.
    """

    candidates = [
        index for index in range(len(messages)) if index not in excluded_indices
    ]
    bundles = _turn_index_bundles(messages, candidates)
    pinned = {
        index
        for bundle in (bundles[-recent_turns:] if recent_turns else ())
        for index in bundle
    }
    for pattern, keep in (
        (_MUTATION, mutation_turns),
        (_VERIFICATION, verification_turns),
    ):
        if keep == 0:
            continue
        matched = [
            bundle for bundle in bundles
            if any(
                messages[index].get("role") == "assistant"
                and pattern.search(str(messages[index].get("content", "")))
                for index in bundle
            )
        ]
        for bundle in matched[-keep:]:
            pinned.update(bundle)
    return pinned


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
        words = list(_TOKEN.finditer(content))
        for offset in range(0, len(words), segment_tokens):
            final = min(offset + segment_tokens, len(words)) - 1
            # Include the separator following a complete segment. Rejoining
            # every segment from one message therefore reconstructs its exact
            # original text, including newlines at segment boundaries.
            end = (
                words[offset + segment_tokens].start()
                if offset + segment_tokens < len(words)
                else len(content)
            )
            start = 0 if offset == 0 else words[offset].start()
            text = content[start:end]
            if text:
                segments.append((f"m{index}-{offset // segment_tokens}-{role}", text))
    return segments


def _turn_bundles(
    messages: Sequence[Mapping[str, Any]],
    candidate_indices: Sequence[int],
    segment_tokens: int,
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

    grouped = _turn_index_bundles(messages, candidate_indices)
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


def _segment_metadata(segment_id: str) -> dict[str, Any]:
    """Expose causal position without relying on resource retrieval order."""

    metadata: dict[str, Any] = {"selection_policy": "typed_bm25_embedding_rrf"}
    match = re.fullmatch(r"m(\d+)-(\d+)-(.+)", segment_id)
    if match is not None:
        metadata.update(
            message_index=int(match.group(1)),
            segment_index=int(match.group(2)),
            role=match.group(3),
        )
    return metadata


def _select_turn_bundles(
    bundles: Sequence[Sequence[tuple[str, str]]], query: str, budget: int,
) -> list[tuple[str, str]]:
    """Rank trajectory turns while admitting every segment of a chosen turn."""

    if not bundles or budget <= 0:
        return []
    texts = ["".join(text for _, text in bundle) for bundle in bundles]
    costs = [sum(_count_tokens(text) for _, text in bundle) for bundle in bundles]
    index = LargeRecordIndex(texts)
    result = index.search(
        query, policy=LargeRecordSearchPolicy.HYBRID,
        top_k=len(bundles), candidate_limit=len(bundles),
    )
    selected: set[int] = set()
    remaining = budget
    for hit in result.hits:
        bundle_index = int(hit.unit_id.split(":", 1)[1])
        cost = costs[bundle_index]
        if cost <= remaining:
            selected.add(bundle_index)
            remaining -= cost

    # Preserve the conservative recency fallback, but at turn granularity.
    for bundle_index in range(len(bundles) - 1, -1, -1):
        if bundle_index in selected:
            continue
        cost = costs[bundle_index]
        if cost <= remaining:
            selected.add(bundle_index)
            remaining -= cost

    return [
        segment
        for bundle_index, bundle in enumerate(bundles)
        if bundle_index in selected
        for segment in bundle
    ]


def _trace(
    request_index: int, session_id: str, mode: ContextTreatment, budget_fraction: float,
    logical: int, mandatory: int, selected: int, physical: int,
    candidates: int, selected_segments: int, selected_resource_digest: str | None,
    route_time_s: float,
) -> TreatmentTrace:
    avoided = max(0, logical - physical)
    return TreatmentTrace(
        request_index, session_id, mode.value, budget_fraction, logical, mandatory, selected,
        physical, avoided, avoided / logical if logical else 0.0,
        candidates, selected_segments, selected_resource_digest, route_time_s,
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
        recent_mutation_turns: int = 1,
        recent_verification_turns: int = 1,
        consumption_policy: str = "standard",
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
        self.recent_completed_turns = recent_completed_turns
        self.recent_mutation_turns = recent_mutation_turns
        self.recent_verification_turns = recent_verification_turns
        if consumption_policy not in CONSUMPTION_POLICIES:
            raise ValueError(
                f"unknown consumption policy {consumption_policy!r}; "
                f"expected one of {CONSUMPTION_POLICIES}"
            )
        self.consumption_policy = consumption_policy
        self._frozen_selections = _load_selection_fixture(selection_replay_path)
        self._lock = threading.Lock()
        self._request_index = 0
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

    def _forward(self, handler: BaseHTTPRequestHandler) -> None:
        body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
        trace = None
        request_index = None
        input_digest = None
        if handler.command == "POST" and urlparse(handler.path).path == "/v1/chat/completions":
            payload = json.loads(body.decode("utf-8"))
            payload.update(self.request_overrides)
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
                recent_completed_turns=self.recent_completed_turns,
                recent_mutation_turns=self.recent_mutation_turns,
                recent_verification_turns=self.recent_verification_turns,
            )
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
