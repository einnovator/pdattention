"""Compare PRA storage eviction policies on one realistic mixed record trace."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pra_hf.storage_lifecycle import (
    InMemoryHotBridge,
    MemoryKVStore,
    PRARetentionClass,
    PRAStorageEntry,
    PRAStorageManager,
    PRAStoragePolicy,
    PRAStorageTier,
    PRAStorageTierConfig,
)


DEFAULT_OUTPUT = Path(
    "docs/papers/shared/results/paper4_5_runtime/storage_policy_trace"
)
POLICIES = ("lru", "size_aware_lru", "weighted_lru")


@dataclass(frozen=True)
class TraceResource:
    key: str
    record_type: str
    size: int
    reconstruction_ms: float
    task_status: str | None = None
    shared_references: int = 0
    dependents: int = 0
    utility: float = 1.0


def resources() -> tuple[TraceResource, ...]:
    values = [
        TraceResource("shared-design", "generic_document", 24_000, 18.0, shared_references=12, utility=4.0),
        TraceResource("shared-api", "generic_document", 20_000, 14.0, shared_references=8, utility=3.5),
        TraceResource("waiting-task", "task_state", 7_000, 4.0, "waiting", dependents=3, utility=5.0),
        TraceResource("active-task", "task_state", 6_000, 3.0, "active", dependents=2, utility=4.0),
        TraceResource("completed-result", "task_result", 14_000, 9.0, "completed", dependents=1, utility=2.5),
        TraceResource("rag-db", "rag_result", 32_000, 35.0, utility=3.0),
    ]
    values.extend(
        TraceResource(f"message-{index}", "user_message", 4_000, 1.0, utility=1.5)
        for index in range(8)
    )
    values.extend(
        TraceResource(f"tool-{index}", "tool_response", 10_000 + index * 500, 7.0, "active", utility=2.0)
        for index in range(10)
    )
    values.extend(
        TraceResource(f"unused-tool-{index}", "tool_response", 9_000, 7.0, "completed", utility=0.25)
        for index in range(4)
    )
    return tuple(values)


def access_trace(turns: int = 120) -> tuple[str, ...]:
    """Build shared, task, message, tool, and reconstructable-RAG accesses."""

    trace: list[str] = []
    for turn in range(turns):
        trace.append("shared-design")
        if turn % 2 == 0:
            trace.append("shared-api")
        trace.append("waiting-task" if turn < (3 * turns) // 4 else "completed-result")
        if turn % 3 == 0:
            trace.append("active-task")
        trace.append(f"message-{turn % 8}")
        if turn % 5 == 0:
            trace.append(f"tool-{(turn // 5) % 10}")
        if turn % 11 == 0:
            trace.append("rag-db")
    return tuple(trace)


def _entry(spec: TraceResource) -> PRAStorageEntry:
    return PRAStorageEntry(
        logical_key=spec.key,
        record_type=spec.record_type,
        retention_class=PRARetentionClass.RECONSTRUCTABLE,
        tenant_id="benchmark",
        session_id="session-long",
        task_id="task-long" if spec.task_status else None,
        task_status=spec.task_status,
        resource_version="v1",
        detail_bytes=spec.size,
        reconstruction_cost_ms=spec.reconstruction_ms,
        shared_reference_count=spec.shared_references,
        dependent_record_count=spec.dependents,
        created_ns=0,
        last_access_ns=0,
    )


def _manager(policy_name: str, quota: int) -> PRAStorageManager:
    policy = PRAStoragePolicy(
        profile=f"storage-trace-{policy_name}",
        hot=PRAStorageTierConfig(max_bytes=64_000),
        warm=PRAStorageTierConfig(max_bytes=quota, cold_grace_seconds=0),
        cold=PRAStorageTierConfig(enabled=False),
        eviction_policy=policy_name,
        immediate_persistence=False,
        task_aware=policy_name == "weighted_lru",
    )
    return PRAStorageManager(
        policy,
        hot=InMemoryHotBridge(),
        warm=MemoryKVStore(),
        cold=None,
    )


def run_policy(policy_name: str, *, quota: int = 80_000, turns: int = 120) -> dict[str, object]:
    specs = {spec.key: spec for spec in resources()}
    manager = _manager(policy_name, quota)
    persisted: set[str] = set()
    peak_hot = 0
    peak_warm = 0
    for index, spec in enumerate(specs.values()):
        payload = bytes([index % 251]) * spec.size
        manager.register(
            _entry(spec),
            payload,
            source_loader=lambda value=payload: value,
            now_ns=index,
        )
        manager.demote_hot(spec.key, now_ns=index + 1)
        manager.run_maintenance(now_ns=index + 2)
        persisted.add(spec.key)
        usage = manager.usage()
        peak_hot = max(peak_hot, usage["hot_bytes"])
        peak_warm = max(peak_warm, usage["warm_bytes"])

    hits = 0
    misses = 0
    weighted_hits = 0.0
    weighted_total = 0.0
    source_cost_ms = 0.0
    task_hits = 0
    task_accesses = 0
    shared_hits = 0
    shared_accesses = 0
    tool_hits = 0
    tool_accesses = 0
    access_counts = {key: 0 for key in specs}
    started = time.perf_counter_ns()
    for event_index, key in enumerate(access_trace(turns), 1):
        spec = specs[key]
        entry = manager.entries[key]
        hit = entry.current_tier != PRAStorageTier.SOURCE
        hits += int(hit)
        misses += int(not hit)
        weighted_hits += spec.utility * int(hit)
        weighted_total += spec.utility
        source_cost_ms += 0.0 if hit else spec.reconstruction_ms
        access_counts[key] += 1
        if key == "waiting-task":
            task_accesses += 1
            task_hits += int(hit)
        if key.startswith("shared-"):
            shared_accesses += 1
            shared_hits += int(hit)
        if key.startswith("tool-"):
            tool_accesses += 1
            tool_hits += int(hit)
        now_ns = event_index * 1_000_000_000
        manager.promote(key, now_ns=now_ns)
        usage = manager.usage()
        peak_hot = max(peak_hot, usage["hot_bytes"])
        peak_warm = max(peak_warm, usage["warm_bytes"])
        manager.record_access(key, selected=True, consumed=True, now_ns=now_ns + 1)
        manager.demote_hot(key, now_ns=now_ns + 2)
        manager.run_maintenance(now_ns=now_ns + 3)
        usage = manager.usage()
        peak_hot = max(peak_hot, usage["hot_bytes"])
        peak_warm = max(peak_warm, usage["warm_bytes"])
    wall_ms = (time.perf_counter_ns() - started) / 1e6
    wasted_write_bytes = sum(
        specs[key].size for key in persisted if access_counts[key] <= 1
    )
    metrics = manager.metrics.to_dict()
    return {
        "policy": policy_name,
        "turns": turns,
        "accesses": hits + misses,
        "warm_hits": hits,
        "source_reconstructions": misses,
        "hit_rate": hits / (hits + misses),
        "utility_weighted_hit_rate": weighted_hits / weighted_total,
        "task_protected_hit_rate": task_hits / task_accesses,
        "shared_document_hit_rate": shared_hits / shared_accesses,
        "tool_result_hit_rate": tool_hits / tool_accesses,
        "peak_hot_bytes": peak_hot,
        "peak_warm_bytes": peak_warm,
        "persistence_writes": metrics["persistence_writes"],
        "persistence_bytes": metrics["bytes_written"],
        "wasted_write_bytes": wasted_write_bytes,
        "evictions": metrics["evictions"],
        "reloads": metrics["reloads"],
        "task_aware_retention_events": metrics["task_aware_retention_hits"],
        "estimated_context_resolution_ms": source_cost_ms,
        "measured_policy_wall_ms": wall_ms,
        "quality_proxy": weighted_hits / weighted_total,
        "quality_proxy_definition": "utility-weighted required-record availability before promotion",
    }


def write_results(output: Path, *, turns: int, quota: int) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    rows = [run_policy(policy, quota=quota, turns=turns) for policy in POLICIES]
    with (output / "storage_policy_trace_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": "1.0",
        "experiment": "paper4_5_storage_policy_mixed_trace_v1",
        "evidence_tier": "CONTROLLED_POLICY_TRACE",
        "warm_quota_bytes": quota,
        "turns": turns,
        "rows": rows,
        "scope": "engine-neutral byte lifecycle; quality is a declared availability proxy",
    }
    (output / "storage_policy_trace_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    table = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Policy & Hit & Utility hit & Task hit & Reloads & Resolution ms \\\\",
        "\\midrule",
    ]
    for row in rows:
        table.append(
            f"{str(row['policy']).replace('_', ' ')} & "
            f"{100 * float(row['hit_rate']):.1f}\\% & "
            f"{100 * float(row['utility_weighted_hit_rate']):.1f}\\% & "
            f"{100 * float(row['task_protected_hit_rate']):.1f}\\% & "
            f"{row['reloads']} & {float(row['estimated_context_resolution_ms']):.0f} \\\\"
        )
    table.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_storage_policy_trace_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )
    try:
        import matplotlib.pyplot as plt

        labels = [str(row["policy"]).replace("_", " ") for row in rows]
        figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
        axes[0].bar(labels, [row["hit_rate"] for row in rows], label="Unweighted")
        axes[0].scatter(labels, [row["utility_weighted_hit_rate"] for row in rows], color="#b45309", label="Utility weighted", zorder=3)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Required-record hit rate")
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].bar(labels, [row["estimated_context_resolution_ms"] for row in rows], color="#2878b5")
        axes[1].set_ylabel("Estimated reconstruction cost (ms)")
        for axis in axes:
            axis.tick_params(axis="x", rotation=15)
            axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output / "storage_policy_trace.pdf")
        figure.savefig(output / "storage_policy_trace.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--turns", type=int, default=120)
    parser.add_argument("--warm-quota-bytes", type=int, default=80_000)
    args = parser.parse_args()
    print(json.dumps(write_results(
        args.output, turns=args.turns, quota=args.warm_quota_bytes
    ), indent=2))


if __name__ == "__main__":
    main()
