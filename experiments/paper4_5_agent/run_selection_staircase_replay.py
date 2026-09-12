"""Replay a frozen agent trajectory while omitting old causal bundles gradually.

This is a correctness/policy diagnostic, not an autonomous SWE-bench score.  Every
request and observation comes from one frozen successful run.  A condition remains
admitted only while the engine reproduces the recorded assistant action exactly.

The current cross-engine live-history contract selects whole logical records.  The
staircase therefore removes complete old assistant/observation bundles.  It never
pretends that dropping one child segment of a large observation saves K/V: engines
that expose only message boundaries would still attach the whole message span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context_treatment import (
    _count_tokens,
    _mandatory_indices,
    _pinned_task_indices,
    _progress_pinned_indices,
    _turn_index_bundles,
)


_RESOURCE_ID = re.compile(r"m(?P<message>\d+)-(?P<segment>\d+)-(?P<role>.+)")
_COMMAND = re.compile(r"```mswea_bash_command\s*\n(?P<command>[\s\S]*?)\n```")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assistant_text(payload: Mapping[str, Any]) -> str:
    return str(payload["choices"][0].get("message", {}).get("content") or "")


def _command(text: str) -> str | None:
    match = _COMMAND.search(text.replace("\r\n", "\n"))
    return match.group("command").strip() if match else None


def load_interactions(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Load adjacent request/response events and reject incomplete histories."""

    events = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    requests = {int(row["request_index"]): row for row in events if row.get("event") == "request"}
    responses = {int(row["request_index"]): row for row in events if row.get("event") == "response"}
    if not requests or set(requests) != set(responses):
        raise ValueError("interaction history must contain one response for every request")
    expected = set(range(1, len(requests) + 1))
    if set(requests) != expected:
        raise ValueError("interaction request indexes must be contiguous and one-based")
    return [(requests[index], responses[index]) for index in sorted(requests)]


def protected_message_indices(
    messages: Sequence[Mapping[str, Any]],
    *,
    recent_completed_turns: int = 2,
    recent_source_turns: int = 1,
    recent_progress_turns: int = 1,
    recent_mutation_turns: int = 1,
    recent_verification_turns: int = 1,
) -> set[int]:
    """Return the non-negotiable task, active tail, and progress spine."""

    mandatory = _mandatory_indices(messages)
    task = _pinned_task_indices(messages, mandatory)
    progress, _ = _progress_pinned_indices(
        messages,
        mandatory | task,
        recent_turns=recent_completed_turns,
        source_turns=recent_source_turns,
        progress_turns=recent_progress_turns,
        mutation_turns=recent_mutation_turns,
        verification_turns=recent_verification_turns,
    )
    return mandatory | task | progress


def oldest_eligible_causal_bundles(
    messages: Sequence[Mapping[str, Any]], max_bundles: int,
) -> list[list[int]]:
    """Choose old complete turns while preserving all protected state."""

    if max_bundles < 0:
        raise ValueError("max_bundles must be non-negative")
    protected = protected_message_indices(messages)
    candidates = [index for index in range(len(messages)) if index not in protected]
    eligible = [
        bundle for bundle in _turn_index_bundles(messages, candidates)
        if bundle
        and str(messages[bundle[0]].get("role")) == "assistant"
        and any(str(messages[index].get("role")) != "assistant" for index in bundle[1:])
    ]
    return eligible[:max_bundles]


def forced_old_causal_bundle(
    messages: Sequence[Mapping[str, Any]], start_index: int,
) -> list[list[int]]:
    """Select one named completed bundle after it leaves the active/recent tail.

    This diagnostic deliberately ignores semantic progress pinning. It tests
    whether an early analysis/search turn is actually dispensable instead of
    assuming that a regex match proves causal importance.
    """

    mandatory = _mandatory_indices(messages)
    task = _pinned_task_indices(messages, mandatory)
    recent, _ = _progress_pinned_indices(
        messages,
        mandatory | task,
        recent_turns=2,
        source_turns=0,
        progress_turns=0,
        mutation_turns=0,
        verification_turns=0,
    )
    candidates = [
        index for index in range(len(messages))
        if index not in mandatory | task | recent
    ]
    return [
        bundle for bundle in _turn_index_bundles(messages, candidates)
        if bundle and bundle[0] == start_index
    ][:1]


def ablate_physical_payload(
    request_row: Mapping[str, Any], *, max_omitted_bundles: int,
    forced_bundle_start: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drop selected whole records and make the realized selection auditable."""

    logical = request_row.get("logical_payload") or {}
    messages = [dict(row) for row in logical.get("messages", ())]
    physical = json.loads(json.dumps(request_row.get("physical_payload") or logical))
    bundles = (
        forced_old_causal_bundle(messages, forced_bundle_start)
        if forced_bundle_start is not None
        else oldest_eligible_causal_bundles(messages, max_omitted_bundles)
    )
    omitted_indices = {index for bundle in bundles for index in bundle}
    envelope = dict(physical.get("pra") or {})
    resources = list(envelope.get("resources") or ())

    def message_index(resource: Mapping[str, Any]) -> int | None:
        metadata = resource.get("metadata") or {}
        if metadata.get("message_index") is not None:
            return int(metadata["message_index"])
        match = _RESOURCE_ID.fullmatch(str(resource.get("resource_id", "")))
        return int(match.group("message")) if match else None

    retained = [row for row in resources if message_index(row) not in omitted_indices]
    removed = [row for row in resources if message_index(row) in omitted_indices]
    envelope["resources"] = retained
    selected_tokens = sum(_count_tokens(row.get("text")) for row in retained)
    mandatory_tokens = sum(
        _count_tokens(row.get("content")) for row in physical.get("messages", ())
    )
    logical_tokens = sum(_count_tokens(row.get("content")) for row in messages)
    budget = dict(envelope.get("budget") or {})
    budget["max_resources"] = max(1, len(retained))
    budget["max_selected_tokens"] = max(1, selected_tokens)
    envelope["budget"] = budget
    metadata = dict(envelope.get("metadata") or {})
    realized = (
        (mandatory_tokens + selected_tokens) / logical_tokens if logical_tokens else 1.0
    )
    metadata.update({
        "selection_complete": not removed,
        "selection_ablation": (
            "forced_old_complete_causal_bundle_v1"
            if forced_bundle_start is not None
            else "oldest_complete_causal_bundles_v1"
        ),
        "selection_ablation_max_bundles": int(max_omitted_bundles),
        "selection_ablation_forced_bundle_start": forced_bundle_start,
        "selection_ablation_omitted_message_indices": sorted(omitted_indices),
        "selection_ablation_omitted_resource_ids": [
            str(row.get("resource_id")) for row in removed
        ],
        "target_retention_fraction": realized,
        "realized_retention_fraction": realized,
    })
    envelope["metadata"] = metadata
    physical["pra"] = envelope
    audit = {
        "logical_tokens_estimate": logical_tokens,
        "mandatory_tokens_estimate": mandatory_tokens,
        "selected_tokens_estimate": selected_tokens,
        "realized_retention_fraction_estimate": realized,
        "omitted_causal_bundles": bundles,
        "omitted_message_indices": sorted(omitted_indices),
        "omitted_resource_ids": [str(row.get("resource_id")) for row in removed],
        "retained_resource_count": len(retained),
        "removed_resource_count": len(removed),
        "selection_granularity": "whole_logical_records",
        "subrecord_kv_selection_qualified": False,
        "forced_bundle_start": forced_bundle_start,
    }
    return physical, audit


def _post(url: str, payload: Mapping[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response), time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    interactions = load_interactions(args.interaction_history)
    artifact: dict[str, Any] = {
        "schema": "paper4.5-selection-staircase-replay-v1",
        "status": "running",
        "engine": args.engine,
        "model": args.model,
        "source_interaction_history": str(args.interaction_history),
        "source_interaction_history_sha256": _sha256(
            args.interaction_history.read_text(encoding="utf-8")
        ),
        "temperature": 0,
        "max_omitted_causal_bundles_per_request": args.max_omitted_bundles,
        "forced_bundle_start": args.forced_bundle_start,
        "policy": {
            "task_statement": "always retained in full",
            "current_action_observation": "always retained in full",
            "recent_completed_turns": 2,
            "progress_spine": "latest source, hypothesis, mutation, verification",
            "omission_order": "oldest eligible complete causal bundle first",
            "subrecord_tool_output_ablation": "not yet engine-qualified",
        },
        "turns": [],
    }
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for request_row, response_row in interactions[: args.turns or None]:
        physical, selection = ablate_physical_payload(
            request_row, max_omitted_bundles=args.max_omitted_bundles,
            forced_bundle_start=args.forced_bundle_start,
        )
        physical["model"] = args.model
        physical.update({
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        })
        raw, elapsed = _post(endpoint, physical, args.timeout)
        expected = _assistant_text(response_row["payload"])
        observed = _assistant_text(raw)
        row = {
            "request_index": int(request_row["request_index"]),
            "elapsed_seconds": elapsed,
            "selection": selection,
            "expected_action_sha256": _sha256(expected),
            "observed_action_sha256": _sha256(observed),
            "exact_action": observed == expected,
            "expected_command": _command(expected),
            "observed_command": _command(observed),
            "exact_command": _command(observed) == _command(expected),
            "engine_pra": raw.get("pra"),
            "usage": raw.get("usage"),
        }
        artifact["turns"].append(row)
        args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        if not row["exact_action"]:
            artifact["status"] = "diverged"
            artifact["first_divergent_request"] = row["request_index"]
            break
    else:
        artifact["status"] = "exact"
        artifact["first_divergent_request"] = None
    artifact["completed_turns"] = len(artifact["turns"])
    artifact["exact_action_turns"] = sum(row["exact_action"] for row in artifact["turns"])
    artifact["all_actions_exact"] = artifact["status"] == "exact"
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18101/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--max-omitted-bundles", type=int, default=0)
    parser.add_argument("--forced-bundle-start", type=int)
    parser.add_argument("--turns", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({key: result[key] for key in (
        "status", "engine", "model", "completed_turns", "exact_action_turns",
        "first_divergent_request",
    )}, indent=2))
    if result["status"] != "exact":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
