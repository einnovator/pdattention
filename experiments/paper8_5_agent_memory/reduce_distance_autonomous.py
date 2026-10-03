"""Reduce repeated autonomous distance-policy runs with task-level outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .trajectory_diagnostics import (
    compare_miniswe_trajectories,
    extract_miniswe_steps,
)


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not an object")
    return value


def _episode(directory: Path) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]], str]:
    payload = _load(directory / "persistent_episode_export.json")
    trajectory = payload.get("trajectory", payload)
    messages = trajectory.get("messages") if isinstance(trajectory, Mapping) else None
    if not isinstance(messages, list):
        raise ValueError(f"{directory} has no episode messages")
    normalized = [
        {"role": str(row.get("role")), "content": str(row.get("content") or "")}
        for row in messages if isinstance(row, Mapping)
    ]
    digest = hashlib.sha256(json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return messages, extract_miniswe_steps(messages), digest


def reduce(reference: Path, candidates: list[Path]) -> dict[str, Any]:
    reference_messages, reference_steps, reference_digest = _episode(reference)
    reference_result = _load(reference / "official_result.json")
    reference_metrics = _load(reference / "autonomous_metrics.json")
    rows = []
    for directory in candidates:
        messages, steps, digest = _episode(directory)
        official = _load(directory / "official_result.json")
        metrics = _load(directory / "autonomous_metrics.json")
        rows.append({
            "directory": str(directory),
            "official_resolved": official.get("resolved"),
            "calls": metrics.get("calls"),
            "call_delta_vs_reference": (
                int(metrics["calls"]) - int(reference_metrics["calls"])
            ),
            "cumulative_full_tokens": metrics.get("cumulative_full_tokens"),
            "cumulative_materialized_tokens": metrics.get(
                "cumulative_materialized_tokens"
            ),
            "cumulative_saving_fraction": 1.0 - float(
                metrics["cumulative_materialized_retention_fraction"]
            ),
            "reacquisition_events": metrics.get("reacquisition_events"),
            "message_count": len(messages),
            "message_content_sha256": digest,
            "trajectory_comparison": compare_miniswe_trajectories(
                reference_steps, steps
            ),
        })
    return {
        "schema_version": 1,
        "study": "paper8_5_distance_autonomous_reduction",
        "reference": {
            "directory": str(reference),
            "official_resolved": reference_result.get("resolved"),
            "calls": reference_metrics.get("calls"),
            "message_count": len(reference_messages),
            "message_content_sha256": reference_digest,
        },
        "candidates": rows,
        "repeat_exact_message_content": len({
            row["message_content_sha256"] for row in rows
        }) == 1,
        "all_candidates_resolved": all(
            row["official_resolved"] is True for row in rows
        ),
        "guardrail": (
            "Repeated executions of one task/model/backend do not estimate "
            "cross-task or cross-agent reliability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
