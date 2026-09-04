"""Summarize an official no-PRA agent cohort without changing its gate status."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable, Mapping


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(rows: list[Mapping[str, object]], field: str) -> dict[str, float | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(rows: list[Mapping[str, object]], *, minimum_tasks: int) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one task result is required")
    resolved = sum(bool(row["resolved"]) for row in rows)
    cumulative = sum(int(row["cumulative_prompt_tokens"]) for row in rows)
    unique = sum(int(row["unique_context_tokens_estimate"]) for row in rows)
    repeated = sum(int(row["repeated_context_tokens_estimate"]) for row in rows)
    return {
        "schema_version": "paper4.5-agent-baseline-summary-v1",
        "cohort_status": "ADEQUATE_FOR_TRAJECTORY_ESTIMATION"
        if len(rows) >= minimum_tasks else "INSUFFICIENT_COHORT",
        "minimum_tasks": minimum_tasks,
        "tasks": len(rows),
        "resolved": resolved,
        "success_rate": resolved / len(rows),
        "success_wilson_95_ci": _wilson(resolved, len(rows)),
        "pra_treatment_unlocked": False,
        "token_totals": {
            "cumulative_prompt_tokens": cumulative,
            "unique_context_tokens_estimate": unique,
            "repeated_context_tokens_estimate": repeated,
            "repeated_context_fraction_estimate": repeated / cumulative if cumulative else 0.0,
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        },
        "behavior_totals": {
            "model_calls": sum(int(row["model_call_count"]) for row in rows),
            "tool_calls": sum(int(row["tool_call_count"]) for row in rows),
            "patch_bytes": sum(int(row["patch_bytes"]) for row in rows),
        },
        "task_distributions": {
            field: _distribution(rows, field)
            for field in (
                "cumulative_prompt_tokens",
                "repeated_context_fraction_estimate",
                "model_call_count",
                "tool_call_count",
                "trajectory_length",
                "patch_bytes",
                "wall_time_s",
                "decode_time_s",
                "tool_time_s",
            )
        },
        "missing_engine_metrics": sorted(
            field
            for field in ("ttft_ms", "prefill_time_s", "peak_memory_bytes", "kv_bytes")
            if all(row.get(field) is None for row in rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tasks", type=int, default=20)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = summarize(rows, minimum_tasks=args.minimum_tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
