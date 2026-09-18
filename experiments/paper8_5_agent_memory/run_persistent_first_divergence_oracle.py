"""Run same-prefix first-divergence controls and causal-group add-backs.

This is diagnostic oracle evidence, not autonomous task-quality evidence.  It
freezes the completed-episode prefix and active-task request, then compares
FULL, the candidate retirement policy, and one excluded causal group restored
at a time.  Repeated FULL and candidate requests expose backend variance under
byte-identical model input.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import urllib.request

from pra_hf.agent_history import OpenAIRecordizer

from .autonomous_proxy import (
    AutonomousSelectionConfig,
    transform_autonomous_payload,
)
from .materialization import MaterializationMode
from .multi_issue_session import compose_multi_issue_session
from .negative_receipts import NegativeRealizationMode
from .run_autonomous_swebench import _exact_token_counter, load_persistent_prefix


_COMMAND = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _selection_config(manifest: Mapping[str, Any]) -> AutonomousSelectionConfig:
    selection = dict(manifest["selection"])
    selection["materialization_mode"] = MaterializationMode(
        selection["materialization_mode"]
    )
    selection["negative_realization"] = NegativeRealizationMode(
        selection["negative_realization"]
    )
    return AutonomousSelectionConfig(**selection)


def _full_control_config(
    candidate_config: AutonomousSelectionConfig,
) -> AutonomousSelectionConfig:
    """Return a semantically neutral FULL control for any candidate policy.

    Candidate-only lifecycle options must be disabled while changing the
    policy.  Leaving either option enabled can make the cloned configuration
    invalid, or worse, make a purported FULL control retire history.
    """

    return replace(
        candidate_config,
        policy="full",
        budget_fraction=1.0,
        materialization_mode=MaterializationMode.WHOLE_RECORD,
        negative_realization=NegativeRealizationMode.DROP,
        negative_fallback="none",
        compact_completed_finalizations=False,
        retire_closed_instructions=False,
    )


def _completed_epoch_addback_batches(
    *,
    composed: Mapping[str, Any],
    history,
    exclusions: Sequence[Any],
) -> list[dict[str, Any]]:
    """Group retired causal records by their evaluator-hidden source epoch.

    Boundary-free policies deliberately strip episode IDs from model-visible
    metadata.  The diagnostic may nevertheless use the frozen prefix ledger
    to ask which retired *source epoch* changes the next action.  This mapping
    remains oracle-only and is never exposed to the selector or model.
    """

    episodes = list(composed.get("episodes") or ())
    records = list(history.records)
    record_epoch: dict[str, int] = {}
    offset = 0
    for episode in episodes:
        count = int(episode.get("model_visible_messages") or 0)
        if count < 1 or offset + count > len(records):
            raise ValueError("composed episode ledger does not align with records")
        epoch_index = int(episode["episode_index"])
        for record in records[offset:offset + count]:
            record_epoch[record.record_id] = epoch_index
        offset += count
    if offset != len(records):
        raise ValueError("composed episode ledger does not cover every record")

    groups_by_epoch: dict[int, list[str]] = {}
    tokens_by_epoch: dict[int, int] = {}
    for exclusion in exclusions:
        epochs = {
            record_epoch[record_id]
            for record_id in exclusion.record_ids
        }
        if len(epochs) != 1:
            raise ValueError(
                f"causal group {exclusion.causal_group_id} crosses source epochs"
            )
        epoch = epochs.pop()
        groups_by_epoch.setdefault(epoch, []).append(exclusion.causal_group_id)
        tokens_by_epoch[epoch] = (
            tokens_by_epoch.get(epoch, 0) + int(exclusion.excluded_tokens)
        )

    return [
        {
            "epoch_index": epoch,
            "causal_group_ids": tuple(groups_by_epoch[epoch]),
            "excluded_tokens": tokens_by_epoch[epoch],
        }
        for epoch in sorted(groups_by_epoch)
    ]


def _causal_group_addback_batches(
    *,
    composed: Mapping[str, Any],
    history,
    exclusions: Sequence[Any],
    addback_epoch: int | None = None,
) -> list[dict[str, Any]]:
    """Build one-group add-backs, optionally within one source epoch only."""

    eligible_groups: set[str] | None = None
    if addback_epoch is not None:
        matching = [
            batch
            for batch in _completed_epoch_addback_batches(
                composed=composed,
                history=history,
                exclusions=exclusions,
            )
            if batch["epoch_index"] == addback_epoch
        ]
        if len(matching) != 1:
            raise ValueError(
                f"add-back epoch {addback_epoch} does not identify exactly one "
                "retired source epoch"
            )
        eligible_groups = set(matching[0]["causal_group_ids"])

    return [
        {
            "epoch_index": addback_epoch,
            "causal_group_ids": (exclusion.causal_group_id,),
            "excluded_tokens": int(exclusion.excluded_tokens),
        }
        for exclusion in exclusions
        if eligible_groups is None or exclusion.causal_group_id in eligible_groups
    ]


def _payload(manifest: Mapping[str, Any], messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": manifest["served_model"],
        "messages": [dict(row) for row in messages],
        "temperature": manifest["temperature"],
        "top_p": manifest["top_p"],
        "seed": manifest["seed"],
        "stream": False,
    }
    maximum = manifest.get("max_completion_tokens")
    if maximum is not None:
        payload["max_tokens"] = maximum
    return payload


def _find_frozen_request(
    *,
    trajectory: Mapping[str, Any],
    target_selected_digest: str,
    manifest: Mapping[str, Any],
    config: AutonomousSelectionConfig,
    prior_episodes: Sequence[Mapping[str, Any]],
    count_tokens,
):
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError("trajectory has no messages list")
    matches = []
    for end in range(1, len(messages) + 1):
        candidate = _payload(manifest, messages[:end])
        # A chat request always ends in user/task/observation state.
        if str(messages[end - 1].get("role")) not in {"user", "system"}:
            continue
        transformed = transform_autonomous_payload(
            candidate,
            config,
            count_tokens=count_tokens,
            prior_episodes=prior_episodes,
        )
        if transformed.trace["selected_messages_sha256"] == target_selected_digest:
            matches.append((end, candidate, transformed))
    if len(matches) != 1:
        raise ValueError(
            "candidate trace did not identify exactly one active-message prefix: "
            f"matches={len(matches)}"
        )
    return matches[0]


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


def _generation_row(
    *,
    arm: str,
    repeat: int,
    transformation,
    endpoint: str,
    timeout: float,
) -> dict[str, Any]:
    response = _post(endpoint, transformation.payload, timeout)
    content = str(response["choices"][0]["message"].get("content") or "")
    commands = _COMMAND.findall(content)
    return {
        "arm": arm,
        "repeat": repeat,
        "request_sha256": _digest(transformation.payload),
        "selected_messages_sha256": transformation.trace["selected_messages_sha256"],
        "selected_tokens": transformation.plan.selected_tokens,
        "materialized_tokens": transformation.materialized_tokens,
        "oracle_addback": transformation.trace.get("oracle_addback"),
        "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "response_content": content,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "action_valid": len(commands) == 1,
        "command": commands[0].strip() if len(commands) == 1 else None,
        "command_sha256": (
            hashlib.sha256(commands[0].strip().encode()).hexdigest()
            if len(commands) == 1 else None
        ),
        "reported_usage": response.get("usage"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_json(args.manifest)
    trajectory = _load_json(args.trajectory)
    trace_rows = [
        json.loads(line)
        for line in args.candidate_trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target = next(
        row for row in trace_rows if int(row["request_index"]) == args.request_index
    )
    prior_episodes, prefix_identity = load_persistent_prefix(args.persistent_prefix)
    count_tokens, tokenizer_identity = _exact_token_counter(
        args.tokenizer, manifest.get("tokenizer_revision"), allow_whitespace=False
    )
    candidate_config = _selection_config(manifest)
    end, frozen_payload, candidate = _find_frozen_request(
        trajectory=trajectory,
        target_selected_digest=target["selected_messages_sha256"],
        manifest=manifest,
        config=candidate_config,
        prior_episodes=prior_episodes,
        count_tokens=count_tokens,
    )
    if candidate.trace["request_input_sha256"] != target["request_input_sha256"]:
        raise AssertionError("reconstructed full request does not match candidate trace")
    full_config = _full_control_config(candidate_config)
    full = transform_autonomous_payload(
        frozen_payload,
        full_config,
        count_tokens=count_tokens,
        prior_episodes=prior_episodes,
    )
    composed = compose_multi_issue_session(
        (*prior_episodes, {
            "instance_id": candidate_config.task_id,
            "messages": frozen_payload["messages"],
            "info": {},
        }),
        session_id=candidate_config.session_id,
        boundary_mode=candidate_config.boundary_mode,
    )
    history = OpenAIRecordizer().recordize(composed["messages"]).history
    selected = set(candidate.plan.selected_record_ids)
    preserved_roles = {}
    for role in ("mutation", "verification", "finalization"):
        matching = [
            row.record_id for row in history.records
            if role in {item.value for item in row.semantic_roles}
            and row.metadata.get("episode_status") == "completed"
        ]
        preserved_roles[role] = {
            "available": matching,
            "selected": [record_id for record_id in matching if record_id in selected],
        }
    exclusions = sorted(
        candidate.plan.exclusions,
        key=lambda row: (row.excluded_tokens, row.causal_group_id),
    )
    endpoint = args.base_url.rstrip("/")
    if not endpoint.endswith("/v1/chat/completions"):
        endpoint += "/v1/chat/completions"
    rows = []
    for repeat in range(1, args.control_repeats + 1):
        rows.append(_generation_row(
            arm="FULL_same_prefix", repeat=repeat, transformation=full,
            endpoint=endpoint, timeout=args.timeout_seconds,
        ))
        rows.append(_generation_row(
            arm="candidate_same_prefix", repeat=repeat, transformation=candidate,
            endpoint=endpoint, timeout=args.timeout_seconds,
        ))
    if args.addback_mode == "completed_epoch":
        if args.addback_epoch is not None:
            raise ValueError("--addback-epoch requires --addback-mode causal_group")
        addback_batches = _completed_epoch_addback_batches(
            composed=composed,
            history=history,
            exclusions=exclusions,
        )
    else:
        addback_batches = _causal_group_addback_batches(
            composed=composed,
            history=history,
            exclusions=exclusions,
            addback_epoch=args.addback_epoch,
        )
    for index, batch in enumerate(addback_batches, 1):
        transformed = transform_autonomous_payload(
            frozen_payload,
            candidate_config,
            count_tokens=count_tokens,
            prior_episodes=prior_episodes,
            oracle_addback_causal_group_ids=batch["causal_group_ids"],
        )
        row = _generation_row(
            arm=(
                f"addback_epoch_{batch['epoch_index']:02d}"
                if batch["epoch_index"] is not None
                else f"addback_{index:03d}"
            ),
            repeat=1,
            transformation=transformed,
            endpoint=endpoint,
            timeout=args.timeout_seconds,
        )
        row["addback_epoch_index"] = batch["epoch_index"]
        row["addback_causal_group_ids"] = list(batch["causal_group_ids"])
        row["addback_excluded_tokens"] = batch["excluded_tokens"]
        rows.append(row)
        partial = {
            "schema_version": 1,
            "study": "paper8_5_persistent_first_divergence_oracle",
            "evidence_class": "diagnostic_oracle_not_autonomous_task_quality",
            "rows": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "study": "paper8_5_persistent_first_divergence_oracle",
        "evidence_class": "diagnostic_oracle_not_autonomous_task_quality",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instance_id": trajectory.get("instance_id"),
        "request_index": args.request_index,
        "active_message_prefix_length": end,
        "prefix_identity": prefix_identity,
        "tokenizer": tokenizer_identity,
        "candidate_trace_selected_messages_sha256": target["selected_messages_sha256"],
        "candidate_reconstruction_exact": True,
        "full_same_prefix_selected_messages_sha256": full.trace["selected_messages_sha256"],
        "preserved_completed_state": preserved_roles,
        "excluded_group_count": len(exclusions),
        "addback_mode": args.addback_mode,
        "addback_epoch_filter": args.addback_epoch,
        "addback_batch_count": len(addback_batches),
        "addback_batches": addback_batches,
        "control_repeats": args.control_repeats,
        "rows": rows,
        "interpretation_guardrail": (
            "Add-backs diagnose first-request sensitivity only. They do not establish "
            "autonomous resolution or a deployable selection policy."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument("--persistent-prefix", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=1)
    parser.add_argument("--control-repeats", type=int, default=3)
    parser.add_argument(
        "--addback-mode",
        choices=("causal_group", "completed_epoch"),
        default="causal_group",
        help=(
            "restore one excluded causal group at a time, or restore every "
            "excluded group from one evaluator-hidden completed source epoch"
        ),
    )
    parser.add_argument(
        "--addback-epoch",
        type=int,
        help=(
            "with causal_group mode, test only retired groups from this "
            "evaluator-hidden source epoch"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(args.output),
        "rows": len(result["rows"]),
        "excluded_group_count": result["excluded_group_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
