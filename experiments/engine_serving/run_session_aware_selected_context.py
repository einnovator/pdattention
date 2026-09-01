"""Measure repeated-session Selected Context against an OpenAI-compatible engine."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from pra_hf.serving_benchmark import percentile, stream_chat_completion


SEEDS = (11, 23, 37, 53, 71)
EXPECTED = ("BETA_7319", "DELTA_8842")


def _resources(seed: int) -> dict[str, str]:
    padding = f" session-marker-{seed}" * 20
    return {
        "A": "Resource A: unrelated code ALPHA_1042." + padding,
        "B": "Resource B: requested first code BETA_7319." + padding,
        "C": "Resource C: unrelated code GAMMA_5520." + padding,
        "D": "Resource D: requested second code DELTA_8842." + padding,
    }


def _context(resources: dict[str, str], names: tuple[str, ...]) -> str:
    return "\n".join(f"[{name}] {resources[name]}" for name in names)


def _messages(seed: int, condition: str, assistant: str) -> list[dict[str, str]]:
    resources = _resources(seed)
    first = (
        "Selected PRA context:\n"
        + _context(resources, ("A", "B", "C"))
        + "\n\nAcknowledge that the records are active."
    )
    second_names = {
        "full_context": ("A", "B", "C", "D"),
        "selected_context_without_logical_reuse": ("B", "D"),
        "session_aware_selected_context": ("D",),
    }[condition]
    second = (
        "Newly supplied PRA context:\n"
        + _context(resources, second_names)
        + "\n\nUsing all active conversation context, return exactly the two requested "
        "codes from resources B and D."
    )
    return [
        {"role": "system", "content": "Return requested evidence codes without explanation."},
        {"role": "user", "content": first},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": second},
    ]


def _first_turn(seed: int) -> list[dict[str, str]]:
    resources = _resources(seed)
    return [
        {"role": "system", "content": "Return requested evidence codes without explanation."},
        {
            "role": "user",
            "content": "Selected PRA context:\n"
            + _context(resources, ("A", "B", "C"))
            + "\n\nAcknowledge that the records are active.",
        },
    ]


def _resource_accounting(seed: int, condition: str) -> dict[str, int]:
    resources = _resources(seed)
    counts = {name: len(text.split()) for name, text in resources.items()}
    first = sum(counts[name] for name in ("A", "B", "C"))
    second_names = {
        "full_context": ("A", "B", "C", "D"),
        "selected_context_without_logical_reuse": ("B", "D"),
        "session_aware_selected_context": ("D",),
    }[condition]
    new = sum(counts[name] for name in second_names)
    reuse = counts["B"] if condition == "session_aware_selected_context" else 0
    return {
        "visible_resource_tokens": first + new,
        "new_materialized_resource_tokens": first + new,
        "turn_2_new_materialized_resource_tokens": new,
        "logical_reuse_resource_tokens": reuse,
    }


def _aggregate(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in selected if row.get(name) is not None]

    quality = statistics.fmean(float(row["quality_success"]) for row in selected)
    baseline = {
        (row["seed"], row["repeat"]): row["output_text"]
        for row in rows
        if row["condition"] == "selected_context_without_logical_reuse"
    }
    paired_equivalence = statistics.fmean(
        float(row["output_text"] == baseline[(row["seed"], row["repeat"])])
        for row in selected
    )
    elapsed = sum(values("completion_latency_ms")) / 1000.0
    return {
        "condition": condition,
        "requests": len(selected),
        "quality_success_rate": quality,
        "paired_output_equivalence_to_no_logical_reuse": paired_equivalence,
        "mean_visible_resource_tokens": statistics.fmean(values("visible_resource_tokens")),
        "mean_new_materialized_resource_tokens": statistics.fmean(
            values("new_materialized_resource_tokens")
        ),
        "mean_turn_2_new_materialized_resource_tokens": statistics.fmean(
            values("turn_2_new_materialized_resource_tokens")
        ),
        "mean_logical_reuse_resource_tokens": statistics.fmean(
            values("logical_reuse_resource_tokens")
        ),
        "mean_prompt_tokens": statistics.fmean(values("prompt_tokens")) if values("prompt_tokens") else None,
        "mean_cached_tokens": statistics.fmean(values("cached_tokens")) if values("cached_tokens") else None,
        "prefix_cache_metric_status": "MEASURED" if values("cached_tokens") else "NOT_MEASURED",
        "ttft_ms": {
            "p50": percentile(values("ttft_ms"), 0.50),
            "p95": percentile(values("ttft_ms"), 0.95),
            "p99": percentile(values("ttft_ms"), 0.99),
        },
        "itl_ms": {
            "p50": percentile(values("mean_itl_ms"), 0.50),
            "p95": percentile(values("mean_itl_ms"), 0.95),
            "p99": percentile(values("mean_itl_ms"), 0.99),
        },
        "completion_latency_ms": {
            "p50": percentile(values("completion_latency_ms"), 0.50),
            "p95": percentile(values("completion_latency_ms"), 0.95),
            "p99": percentile(values("completion_latency_ms"), 0.99),
        },
        "successful_requests_per_second": (
            len(selected) * quality / elapsed if elapsed else None
        ),
        "tail_latency_status": "MEASURED" if len(selected) >= 20 else "CONTROLLED_SAMPLE_TOO_SMALL",
        "logical_visibility_status": "MEASURED",
        "native_memory_status": "NOT_MEASURED",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    conditions = (
        "full_context",
        "selected_context_without_logical_reuse",
        "session_aware_selected_context",
    )
    for seed in args.seeds:
        for repeat in range(args.repeats_per_seed):
            for condition in conditions:
                first = stream_chat_completion(
                    args.base_url,
                    model=args.model,
                    messages=_first_turn(seed),
                    timeout_seconds=args.timeout_seconds,
                    cache_salt=None,
                    max_tokens=8,
                )
                second = stream_chat_completion(
                    args.base_url,
                    model=args.model,
                    messages=_messages(seed, condition, first["output_text"]),
                    timeout_seconds=args.timeout_seconds,
                    cache_salt=None,
                    max_tokens=16,
                )
                rows.append({
                    "seed": seed,
                    "repeat": repeat,
                    "condition": condition,
                    "quality_success": all(code in second["output_text"] for code in EXPECTED),
                    "output_text": second["output_text"],
                    **_resource_accounting(seed, condition),
                    **{key: second[key] for key in (
                        "ttft_ms", "completion_latency_ms", "mean_itl_ms",
                        "prompt_tokens", "completion_tokens", "cached_tokens",
                    )},
                })
    return {
        "schema_version": "1.0",
        "benchmark": "session-aware-selected-context-live-v1",
        "engine": args.engine,
        "model": args.model,
        "seeds": list(args.seeds),
        "repeats_per_seed": args.repeats_per_seed,
        "selector_frozen": True,
        "selection": {"turn_1": ["A", "B", "C"], "turn_2": ["B", "D"]},
        "claim_boundary": (
            "E0 Selected Context only. Logical visibility, engine prefix cache, and "
            "native memory are reported independently; native memory is NOT_MEASURED."
        ),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
        "aggregates": [_aggregate(rows, condition) for condition in conditions],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default="tensorrt-llm")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--repeats-per-seed", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "requests": len(payload["rows"])}))


if __name__ == "__main__":
    main()
