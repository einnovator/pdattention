"""Reduce per-turn sparse-engine gates without weakening their claim boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    allocation = dict(payload.get("disjoint_attention_allocation") or {})
    checks = dict(payload.get("checks") or {})
    return {
        "turn": int(payload["turn"]),
        "source_tokens": int(payload["source_tokens"]),
        "selected_kv_tokens": int(payload["selected_kv_tokens"]),
        "realized_retention_fraction": float(
            payload["realized_retention_fraction"]
        ),
        "selected_kv_segments": int(payload["selected_kv_segments"]),
        "has_holes": bool(payload["has_holes"]),
        "max_abs_logit_delta_same_subset": float(
            payload["max_abs_logit_delta_same_subset"]
        ),
        "selection_pack_bytes": int(payload["selection_pack_bytes"]),
        "selection_active_memory_delta_bytes": int(
            payload["selection_active_memory_delta_bytes"]
        ),
        "selected_layer_kv_bytes": int(allocation["selected_layer_kv_bytes"]),
        "consumer_peak_delta_bytes": int(allocation["peak_delta_bytes"]),
        "consumer_peak_delta_to_selected_kv_ratio": float(
            allocation["peak_delta_to_selected_layer_kv_ratio"]
        ),
        "full_selected_kv_sized_allocation_observed": bool(
            allocation["full_selected_kv_sized_allocation_observed"]
        ),
        "selected_history_reencoded_tokens": int(
            payload["selected_text_reencoded_tokens"]
        ),
        "physical_kv_copy": bool(payload["physical_kv_copy"]),
        "same_subset_token_exact": bool(checks["same_subset_token_exact"]),
        "same_subset_logit_within_tolerance": bool(
            checks["same_subset_logit_within_tolerance"]
        ),
        "engine_lifecycle_qualified": bool(
            payload["engine_lifecycle_qualified"]
        ),
        "qualification_blockers": list(payload.get("qualification_blockers") or ()),
        "elapsed_seconds": (
            None
            if payload.get("elapsed_seconds") is None
            else float(payload["elapsed_seconds"])
        ),
        "artifact": str(path),
        "artifact_sha256": _sha256(path),
    }


def summarize(paths: Iterable[Path], expected_turns: tuple[int, ...]) -> dict[str, Any]:
    files = tuple(sorted(paths, key=lambda item: item.name))
    if not files:
        raise ValueError("At least one per-turn gate artifact is required.")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    engines = {str(payload["engine"]) for payload in payloads}
    models = {str(payload["model"]) for payload in payloads}
    policies = {str(payload["materialization_policy"]) for payload in payloads}
    consumers = {str(payload["consumer_implementation"]) for payload in payloads}
    if len(engines) != 1 or len(models) != 1 or len(policies) != 1 or len(consumers) != 1:
        raise ValueError("Per-turn artifacts do not describe one frozen engine cell.")
    rows = sorted(
        (_row(path, payload) for path, payload in zip(files, payloads)),
        key=lambda row: row["turn"],
    )
    observed_turns = tuple(row["turn"] for row in rows)
    if observed_turns != expected_turns:
        raise ValueError(
            f"Expected turns {expected_turns}, observed {observed_turns}."
        )
    full_coverage_qualified = all(
        row["engine_lifecycle_qualified"]
        and not row["qualification_blockers"]
        and row["has_holes"]
        and row["same_subset_token_exact"]
        and row["same_subset_logit_within_tolerance"]
        and row["selection_pack_bytes"] == 0
        and row["selection_active_memory_delta_bytes"] == 0
        and row["selected_history_reencoded_tokens"] == 0
        and not row["physical_kv_copy"]
        and not row["full_selected_kv_sized_allocation_observed"]
        for row in rows
    )
    return {
        "schema_version": "paper4.5.full-turn-sparse-gate.v1",
        "probe": "frozen_task02_full_seven_turn_sparse_coverage",
        "engine": next(iter(engines)),
        "model": next(iter(models)),
        "materialization_policy": next(iter(policies)),
        "consumer_implementation": next(iter(consumers)),
        "expected_turns": list(expected_turns),
        "completed_turns": len(rows),
        "qualified_turns": sum(
            int(row["engine_lifecycle_qualified"]) for row in rows
        ),
        "full_coverage_qualified": full_coverage_qualified,
        "realized_retention_fraction": {
            "minimum": min(row["realized_retention_fraction"] for row in rows),
            "maximum": max(row["realized_retention_fraction"] for row in rows),
        },
        "max_abs_logit_delta_same_subset": max(
            row["max_abs_logit_delta_same_subset"] for row in rows
        ),
        "selection_pack_bytes": sum(row["selection_pack_bytes"] for row in rows),
        "selection_active_memory_delta_bytes": sum(
            row["selection_active_memory_delta_bytes"] for row in rows
        ),
        "selected_history_reencoded_tokens": sum(
            row["selected_history_reencoded_tokens"] for row in rows
        ),
        "physical_kv_copy_turns": sum(int(row["physical_kv_copy"]) for row in rows),
        "max_consumer_peak_delta_bytes": max(
            row["consumer_peak_delta_bytes"] for row in rows
        ),
        "max_consumer_peak_delta_to_selected_kv_ratio": max(
            row["consumer_peak_delta_to_selected_kv_ratio"] for row in rows
        ),
        "full_selected_kv_sized_allocation_turns": sum(
            int(row["full_selected_kv_sized_allocation_observed"])
            for row in rows
        ),
        "elapsed_seconds": (
            sum(
                float(row["elapsed_seconds"])
                for row in rows
                if row["elapsed_seconds"] is not None
            )
            if any(row["elapsed_seconds"] is not None for row in rows)
            else None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-turns", default="4,5,6,7,8,9,10")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected_turns = tuple(int(value) for value in args.expected_turns.split(","))
    result = summarize(args.inputs, expected_turns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "engine": result["engine"],
        "completed_turns": result["completed_turns"],
        "qualified_turns": result["qualified_turns"],
        "full_coverage_qualified": result["full_coverage_qualified"],
    }, indent=2))
    raise SystemExit(0 if result["full_coverage_qualified"] else 1)


if __name__ == "__main__":
    main()
