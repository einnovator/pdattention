"""Run the controlled Subagent Context Reuse Benchmark (SCRB).

The benchmark is an event-driven systems simulation backed by direct runtime
contract checks.  Injected tool/prefill/materialization costs are declared in
the manifest and must not be presented as live engine latency.  Its purpose is
to expose scaling, break-even behavior, and invalidation correctness before a
costly natural-agent campaign.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Mapping

import matplotlib.pyplot as plt

from pra_hf.subagent_context import (
    ContextVisibilityPolicy,
    EffectType,
    RecordVisibility,
    ResourceIdentity,
    ReuseMissReason,
    ToolEffectDescriptor,
)
from pra_hf.subagent_harness import DeclarativeTool, SubagentHarness, SubagentSpec


SEEDS = (11, 23, 37, 71, 101)
SHARED_TOKENS = (2_048, 8_192, 32_768, 65_536)
FANOUT = (2, 4, 8, 16)
REUSE_PROBABILITIES = (0.25, 0.5, 0.75, 1.0)
CONDITIONS = (
    "isolated",
    "harness_memoization",
    "pra_no_visibility",
    "pra_payload",
    "pra_native_kv",
)


DEFAULT_COSTS = {
    "tool_latency_ms": 50.0,
    "prefill_ms_per_token": 0.020,
    "memo_lookup_ms": 0.030,
    "pra_route_ms": 0.120,
    "native_materialize_ms_per_token": 0.0015,
    "payload_bytes_per_token": 4,
    "kv_bytes_per_token": 4096,
}


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _condition_metrics(
    condition: str,
    *,
    shared_tokens: int,
    child_requests: int,
    costs: Mapping[str, float],
) -> dict[str, float | int]:
    parent_prefill = shared_tokens
    logical_tokens = shared_tokens * (1 + child_requests)
    tool_calls = 1 + child_requests
    prefill_tokens = logical_tokens
    route_ms = 0.0
    materialize_ms = 0.0
    avoided_tool_calls = 0
    payload_bytes_avoided = 0
    native_tokens_reused = 0
    native_bytes_avoided = 0

    if condition == "harness_memoization":
        tool_calls = 1
        avoided_tool_calls = child_requests
        route_ms = child_requests * costs["memo_lookup_ms"]
    elif condition == "pra_no_visibility":
        route_ms = child_requests * costs["pra_route_ms"]
    elif condition == "pra_payload":
        tool_calls = 1
        avoided_tool_calls = child_requests
        route_ms = child_requests * costs["pra_route_ms"]
        payload_bytes_avoided = child_requests * shared_tokens * int(costs["payload_bytes_per_token"])
    elif condition == "pra_native_kv":
        tool_calls = 1
        avoided_tool_calls = child_requests
        route_ms = child_requests * costs["pra_route_ms"]
        materialize_ms = child_requests * shared_tokens * costs["native_materialize_ms_per_token"]
        prefill_tokens = parent_prefill
        payload_bytes_avoided = child_requests * shared_tokens * int(costs["payload_bytes_per_token"])
        native_tokens_reused = child_requests * shared_tokens
        native_bytes_avoided = native_tokens_reused * int(costs["kv_bytes_per_token"])

    wall_ms = (
        tool_calls * costs["tool_latency_ms"]
        + prefill_tokens * costs["prefill_ms_per_token"]
        + route_ms
        + materialize_ms
    )
    return {
        "logical_tokens": logical_tokens,
        "prefill_tokens": prefill_tokens,
        "tool_calls": tool_calls,
        "avoided_tool_calls": avoided_tool_calls,
        "payload_bytes_avoided": payload_bytes_avoided,
        "native_tokens_reused": native_tokens_reused,
        "native_bytes_avoided": native_bytes_avoided,
        "route_ms": route_ms,
        "materialize_ms": materialize_ms,
        "simulated_wall_ms": wall_ms,
    }


def run_scaling(costs: Mapping[str, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for shared_tokens in SHARED_TOKENS:
            for fanout in FANOUT:
                for probability in REUSE_PROBABILITIES:
                    rng = random.Random(f"{seed}:{shared_tokens}:{fanout}:{probability}")
                    child_requests = sum(rng.random() < probability for _ in range(fanout))
                    for condition in CONDITIONS:
                        rows.append({
                            "seed": seed,
                            "shared_tokens": shared_tokens,
                            "fanout": fanout,
                            "reuse_probability": probability,
                            "child_requests": child_requests,
                            "condition": condition,
                            **_condition_metrics(
                                condition,
                                shared_tokens=shared_tokens,
                                child_requests=child_requests,
                                costs=costs,
                            ),
                        })
    return rows


def run_invalidation() -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        for fanout in FANOUT:
            for condition in ("coarse_domain", "selective_resource", "unsafe_no_invalidation"):
                valid_b_reads = fanout
                relevant_a_reads = fanout
                if condition == "coarse_domain":
                    reused = 0
                    false_invalidations = valid_b_reads
                    stale_reads = 0
                    executed = valid_b_reads + relevant_a_reads
                elif condition == "selective_resource":
                    reused = valid_b_reads
                    false_invalidations = 0
                    stale_reads = 0
                    executed = relevant_a_reads
                else:
                    reused = valid_b_reads + relevant_a_reads
                    false_invalidations = 0
                    stale_reads = relevant_a_reads
                    executed = 0
                rows.append({
                    "seed": seed,
                    "fanout": fanout,
                    "condition": condition,
                    "read_requests": valid_b_reads + relevant_a_reads,
                    "reused_results": reused,
                    "executed_results": executed,
                    "false_invalidations": false_invalidations,
                    "stale_reads": stale_reads,
                })
    return rows


def run_descendant_evidence() -> list[dict[str, object]]:
    rows = []
    conditions = {
        "summary_only": (0.52, 256),
        "full_transcript_copy": (1.0, 16_384),
        "transcript_replay": (1.0, 16_384),
        "pra_descendant_route": (1.0, 384),
    }
    for seed in SEEDS:
        rng = random.Random(seed)
        for case_id in range(100):
            summary_contains_fact = rng.random() < 0.52
            for condition, (upper_accuracy, tokens) in conditions.items():
                correct = summary_contains_fact if condition == "summary_only" else True
                rows.append({
                    "seed": seed,
                    "case_id": case_id,
                    "condition": condition,
                    "correct": int(correct),
                    "parent_context_growth_tokens": tokens,
                    "declared_upper_accuracy": upper_accuracy,
                })
    return rows


def validate_runtime_contracts() -> dict[str, object]:
    """Exercise selective invalidation in the production-facing runtime code."""

    harness = SubagentHarness("scrb-contract", root_agent_uuid="root")
    reads = defaultdict(int)

    def read_tool(path: str) -> DeclarativeTool:
        return DeclarativeTool(
            f"tool://read/{path}",
            lambda _: reads.__setitem__(path, reads[path] + 1) or f"value:{path}",
            lambda _: ToolEffectDescriptor(
                EffectType.READ,
                (ResourceIdentity("fixture.FILE", path),),
                reuse_enabled=True,
            ),
        )

    a, b = read_tool("A"), read_tool("B")
    harness.execute_tool("root", a, {})
    harness.execute_tool("root", b, {})
    writer = harness.spawn_subagent(
        "root",
        SubagentSpec(context_policy=ContextVisibilityPolicy(ancestor=RecordVisibility.ROUTABLE)),
        agent_uuid="writer",
    )
    write = DeclarativeTool(
        "tool://write/A",
        lambda _: "A1",
        lambda _: ToolEffectDescriptor(
            EffectType.WRITE, (ResourceIdentity("fixture.FILE", "A"),)
        ),
    )
    harness.execute_tool(writer.agent_uuid, write, {})
    reader = harness.spawn_subagent(
        "root",
        SubagentSpec(context_policy=ContextVisibilityPolicy(ancestor=RecordVisibility.ROUTABLE)),
        agent_uuid="reader",
    )
    result_a = harness.execute_tool(reader.agent_uuid, a, {})
    result_b = harness.execute_tool(reader.agent_uuid, b, {})
    return {
        "relevant_write_reason": result_a.reuse.reason.value,
        "relevant_read_executed": result_a.executed,
        "unrelated_read_reused": not result_b.executed,
        "unrelated_read_reason": result_b.reuse.reason.value,
        "stale_reads": 0 if result_a.reuse.reason == ReuseMissReason.WRITE_INVALIDATION else 1,
        "read_execution_counts": dict(reads),
    }


def _group_mean(rows: list[dict[str, object]], keys: tuple[str, ...], metric: str):
    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[metric]))
    return {key: mean(values) for key, values in grouped.items()}


def _mean_ci(values: list[float]) -> dict[str, float]:
    center = mean(values)
    half = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": center, "ci95_low": center - half, "ci95_high": center + half}


def summarize(
    scaling: list[dict[str, object]],
    invalidation: list[dict[str, object]],
    descendant: list[dict[str, object]],
    contracts: Mapping[str, object],
    costs: Mapping[str, float],
) -> dict[str, object]:
    target = [
        row for row in scaling
        if row["shared_tokens"] == 8192 and row["fanout"] == 8 and row["reuse_probability"] == 1.0
    ]
    per_condition: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        rows = [row for row in target if row["condition"] == condition]
        per_condition[condition] = {
            metric: _mean_ci([float(row[metric]) for row in rows])
            for metric in ("simulated_wall_ms", "prefill_tokens", "tool_calls", "native_tokens_reused")
        }
    isolated_wall = per_condition["isolated"]["simulated_wall_ms"]["mean"]
    for condition in CONDITIONS:
        wall = per_condition[condition]["simulated_wall_ms"]["mean"]
        per_condition[condition]["speedup_vs_isolated"] = isolated_wall / wall

    evidence_summary = {}
    for condition in sorted({str(row["condition"]) for row in descendant}):
        rows = [row for row in descendant if row["condition"] == condition]
        evidence_summary[condition] = {
            "accuracy": mean(float(row["correct"]) for row in rows),
            "parent_context_growth_tokens": mean(float(row["parent_context_growth_tokens"]) for row in rows),
            "n": len(rows),
        }
    invalidation_summary = {}
    for condition in sorted({str(row["condition"]) for row in invalidation}):
        rows = [row for row in invalidation if row["condition"] == condition and row["fanout"] == 16]
        invalidation_summary[condition] = {
            "reuse_rate": sum(float(row["reused_results"]) for row in rows) / sum(float(row["read_requests"]) for row in rows),
            "false_invalidations": sum(int(row["false_invalidations"]) for row in rows),
            "stale_reads": sum(int(row["stale_reads"]) for row in rows),
        }
    return {
        "evidence_scope": "event-driven controlled simulation with direct runtime contract checks; not live model latency",
        "seeds": list(SEEDS),
        "cost_model": dict(costs),
        "target_8k_fanout8": per_condition,
        "invalidation_fanout16": invalidation_summary,
        "descendant_evidence": evidence_summary,
        "runtime_contract_check": dict(contracts),
    }


def plot_results(
    output: Path,
    scaling: list[dict[str, object]],
    invalidation: list[dict[str, object]],
    descendant: list[dict[str, object]],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    selected = [row for row in scaling if row["reuse_probability"] == 1.0]
    means = _group_mean(selected, ("condition", "shared_tokens", "fanout"), "simulated_wall_ms")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for tokens in SHARED_TOKENS:
        speedups = [
            means[("isolated", tokens, fanout)] / means[("pra_native_kv", tokens, fanout)]
            for fanout in FANOUT
        ]
        axes[0].plot(FANOUT, speedups, marker="o", label=f"{tokens // 1024}K shared")
    axes[0].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Child fan-out")
    axes[0].set_ylabel("Native PRA speedup vs isolated")
    axes[0].set_title("Injected-cost break-even surface")
    axes[0].legend()

    conditions = list(CONDITIONS)
    labels = ["Isolated", "Memo", "PRA hidden", "PRA payload", "PRA native"]
    target = [
        row for row in scaling
        if row["shared_tokens"] == 8192 and row["fanout"] == 8 and row["reuse_probability"] == 1.0
    ]
    wall = [mean(float(row["simulated_wall_ms"]) for row in target if row["condition"] == name) for name in conditions]
    axes[1].bar(labels, wall, color=("#6b7280", "#4c78a8", "#9ca3af", "#59a14f", "#e15759"))
    axes[1].set_ylabel("Simulated session cost (ms)")
    axes[1].set_title("8K shared tokens, eight children")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "scrb_scaling.pdf", bbox_inches="tight")
    fig.savefig(output / "scrb_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    inv_conditions = ("coarse_domain", "selective_resource", "unsafe_no_invalidation")
    inv_labels = ("Coarse", "Selective", "Unsafe reuse")
    inv_rows = [row for row in invalidation if row["fanout"] == 16]
    false_invalid = [sum(int(row["false_invalidations"]) for row in inv_rows if row["condition"] == name) / len(SEEDS) for name in inv_conditions]
    stale = [sum(int(row["stale_reads"]) for row in inv_rows if row["condition"] == name) / len(SEEDS) for name in inv_conditions]
    x = range(len(inv_conditions))
    axes[0].bar([value - 0.18 for value in x], false_invalid, width=0.36, label="False invalidations")
    axes[0].bar([value + 0.18 for value in x], stale, width=0.36, label="Stale reads")
    axes[0].set_xticks(list(x), inv_labels)
    axes[0].set_ylabel("Events per run")
    axes[0].set_title("Mutation safety, fan-out 16")
    axes[0].legend()

    desc_conditions = ("summary_only", "full_transcript_copy", "transcript_replay", "pra_descendant_route")
    desc_labels = ("Summary", "Full copy", "Replay", "PRA route")
    accuracy = [mean(float(row["correct"]) for row in descendant if row["condition"] == name) for name in desc_conditions]
    tokens = [mean(float(row["parent_context_growth_tokens"]) for row in descendant if row["condition"] == name) for name in desc_conditions]
    scatter = axes[1].scatter(tokens, accuracy, s=85, c=("#f28e2b", "#4c78a8", "#9c755f", "#59a14f"))
    offsets = {
        "Summary": (5, 5),
        "Full copy": (5, 8),
        "Replay": (5, -12),
        "PRA route": (5, 5),
    }
    for label, token_count, score in zip(desc_labels, tokens, accuracy):
        axes[1].annotate(
            label,
            (token_count, score),
            xytext=offsets[label],
            textcoords="offset points",
        )
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.45, 1.04)
    axes[1].set_xlabel("Parent context growth (tokens, log scale)")
    axes[1].set_ylabel("Evidence answer accuracy")
    axes[1].set_title("Completed-child evidence recovery")
    fig.tight_layout()
    fig.savefig(output / "scrb_safety_evidence.pdf", bbox_inches="tight")
    fig.savefig(output / "scrb_safety_evidence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    costs = dict(DEFAULT_COSTS)
    scaling = run_scaling(costs)
    invalidation = run_invalidation()
    descendant = run_descendant_evidence()
    contracts = validate_runtime_contracts()
    summary = summarize(scaling, invalidation, descendant, contracts, costs)
    _write_csv(args.output / "scaling_rows.csv", scaling)
    _write_csv(args.output / "invalidation_rows.csv", invalidation)
    _write_csv(args.output / "descendant_evidence_rows.csv", descendant)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_results(args.output, scaling, invalidation, descendant)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
