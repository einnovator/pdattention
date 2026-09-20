"""Translate OpenHands SDK events into the portable PRA receipt contract.

This module is intentionally an edge adapter.  It may understand OpenHands
event names, but it emits only generic operation, outcome, and resource fields.
PRA policy code and engine backends must not import it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_EDITOR_OPERATIONS = {
    "view": "read",
    "create": "write",
    "str_replace": "write",
    "insert": "write",
    "undo_edit": "write",
}


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_events(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping):
            rows.append(row)
    return rows


def _result_complete(observation: Mapping[str, Any]) -> bool:
    if bool(observation.get("timeout")):
        return False
    content = observation.get("content")
    if not isinstance(content, list):
        return True
    text = "\n".join(
        str(row.get("text") or "") for row in content if isinstance(row, Mapping)
    )
    return "<response clipped>" not in text


def _semantic_status(observation: Mapping[str, Any]) -> tuple[str, str | None]:
    if bool(observation.get("timeout")):
        return "failed", "timeout"
    exit_code = observation.get("exit_code")
    if bool(observation.get("is_error")):
        return "failed", "tool_error"
    if isinstance(exit_code, int) and exit_code != 0:
        return "failed", "nonzero_exit"
    return "succeeded", None


def _portable_operation(action: Mapping[str, Any]) -> tuple[str, str, list[dict[str, Any]], bool]:
    kind = str(action.get("kind") or "")
    if kind == "FileEditorAction":
        operation = _EDITOR_OPERATIONS.get(str(action.get("command") or ""), "unknown")
        path = action.get("path")
        resources = []
        if isinstance(path, str) and path:
            resources.append({
                "resource_id": f"file://{path}",
                "kind": operation if operation in {"read", "write"} else "unknown",
            })
        # The SDK event identifies the intended path but carries no before/after
        # version.  Do not certify the effect trace until middleware adds it.
        return "filesystem", operation, resources, False
    if kind == "TaskTrackerAction":
        return "agent_progress", "progress", [{
            "resource_id": "agent://task-tracker",
            "kind": "write",
            "version_after": _digest(action),
        }], True
    if kind == "FinishAction":
        return "agent_protocol", "finalization", [], True
    if kind == "ThinkAction":
        return "agent_progress", "progress", [], True
    if kind == "TerminalAction":
        # Arbitrary shell remains an unknown-effect barrier.  Its outcome is
        # still useful even though its workspace effects are not certified.
        return "shell", "unknown", [], False
    return "unknown", "unknown", [], False


def build_openhands_receipts(path: Path) -> list[dict[str, Any]]:
    """Return one generic execution receipt per OpenHands action event."""

    actions: dict[str, Mapping[str, Any]] = {}
    observations: dict[str, Mapping[str, Any]] = {}
    observation_ids: dict[str, str] = {}
    for row in _load_events(path):
        event = row.get("event")
        if not isinstance(event, Mapping):
            continue
        if row.get("type") == "ActionEvent" and event.get("id"):
            actions[str(event["id"])] = event
        elif row.get("type") == "ObservationEvent" and event.get("action_id"):
            action_id = str(event["action_id"])
            observation = event.get("observation")
            if isinstance(observation, Mapping):
                observations[action_id] = observation
                observation_ids[action_id] = str(event.get("id") or "")

    receipts: list[dict[str, Any]] = []
    for action_id, event in actions.items():
        action = event.get("action")
        action = action if isinstance(action, Mapping) else {}
        category, operation, resources, effect_complete = _portable_operation(action)
        observation = observations.get(action_id)
        if observation is None:
            semantic_status, error_kind = "unknown", None
            transport_status = "incomplete"
            result_complete = False
            return_code = None
        else:
            semantic_status, error_kind = _semantic_status(observation)
            transport_status = "completed"
            result_complete = _result_complete(observation)
            return_code = observation.get("exit_code")
        receipt = {
            "schema_version": 1,
            "action_record_id": action_id,
            "observation_record_ids": (
                [observation_ids[action_id]] if action_id in observation_ids else []
            ),
            "tool_category": category,
            "operation_kind": operation,
            "transport_status": transport_status,
            "semantic_status": semantic_status,
            "return_code": return_code,
            "error_kind": error_kind,
            "result_complete": result_complete,
            "effect_trace_complete": bool(effect_complete and observation is not None),
            "provenance": "runtime_traced",
            "cwd": (
                str(observation.get("metadata", {}).get("working_dir"))
                if isinstance(observation, Mapping)
                and isinstance(observation.get("metadata"), Mapping)
                and observation["metadata"].get("working_dir") is not None
                else None
            ),
            "resources": resources,
            "source": {
                "agent": "openhands-sdk",
                "action_kind": str(action.get("kind") or "unknown"),
                "tool_call_id": str(event.get("tool_call_id") or ""),
            },
        }
        receipt["receipt_digest"] = _digest(receipt)
        receipts.append(receipt)
    return receipts


def summarize_receipts(receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes: dict[str, int] = {}
    categories: dict[str, int] = {}
    for receipt in receipts:
        status = str(receipt.get("semantic_status") or "unknown")
        category = str(receipt.get("tool_category") or "unknown")
        outcomes[status] = outcomes.get(status, 0) + 1
        categories[category] = categories.get(category, 0) + 1
    return {
        "schema_version": 1,
        "adapter": "openhands-sdk-to-portable-pra-receipt-v1",
        "receipt_count": len(receipts),
        "paired_receipt_count": sum(
            1 for row in receipts if row.get("transport_status") == "completed"
        ),
        "result_complete_count": sum(
            1 for row in receipts if bool(row.get("result_complete"))
        ),
        "effect_trace_complete_count": sum(
            1 for row in receipts if bool(row.get("effect_trace_complete"))
        ),
        "semantic_outcomes": dict(sorted(outcomes.items())),
        "tool_categories": dict(sorted(categories.items())),
        "unknown_effect_barrier_count": sum(
            1 for row in receipts if not bool(row.get("effect_trace_complete"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = build_openhands_receipts(args.events)
    payload = {
        "summary": summarize_receipts(receipts),
        "receipts": receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
