"""Run the common engine-neutral PRA storage lifecycle workload (E1--E9)."""

from __future__ import annotations

import csv
import json
import statistics
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pra_hf.storage_lifecycle import (
    InMemoryHotBridge,
    MemoryKVStore,
    PRARetentionClass,
    PRAStorageEntry,
    PRAStorageEvictionPolicy,
    PRAStorageManager,
    PRAStoragePolicy,
    PRAStorageTier,
    PRAStorageTierConfig,
    dequantize_int8_array,
    quantize_int8_array,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "papers" / "shared" / "results" / "paper4_5_runtime" / "storage_lifecycle"
ENGINES = ("hf", "vllm", "sglang", "mlx")
SEEDS = (11, 23, 37, 53, 71)


def _entry(key: str, record_type: str, *, task_status: str | None = None, dependent: int = 0, shared: int = 0) -> PRAStorageEntry:
    return PRAStorageEntry(
        logical_key=key,
        record_type=record_type,
        retention_class=PRARetentionClass.RECONSTRUCTABLE,
        tenant_id="benchmark",
        security_scope=None,
        session_id="session-a",
        task_id="task-a",
        task_status=task_status,
        resource_version="v1",
        detail_bytes=0,
        created_ns=0,
        last_access_ns=0,
        dependent_record_count=dependent,
        shared_reference_count=shared,
    )


def _policy(*, quota: int = 1 << 20, eviction: str = "weighted_lru", immediate: bool = False, task_aware: bool = True) -> PRAStoragePolicy:
    return PRAStoragePolicy(
        profile="benchmark",
        hot=PRAStorageTierConfig(max_bytes=quota),
        warm=PRAStorageTierConfig(max_bytes=quota, cold_grace_seconds=10),
        cold=PRAStorageTierConfig(max_bytes=quota, compression="none"),
        eviction_policy=eviction,
        immediate_persistence=immediate,
        task_aware=task_aware,
    )


def _manager(**kwargs) -> PRAStorageManager:
    return PRAStorageManager(
        _policy(**kwargs),
        hot=InMemoryHotBridge(),
        warm=MemoryKVStore(),
        cold=MemoryKVStore(),
    )


def _run(engine: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    native = rng.normal(size=(8, 2, 64, 64)).astype(np.float32)
    payload = native.tobytes()

    # E1/E8: exact HOT -> WARM -> COLD -> HOT and SOURCE reconstruction.
    lifecycle = _manager()
    lifecycle.register(_entry("document", "generic_document"), payload, now_ns=0)
    lifecycle.demote_hot("document", now_ns=1)
    lifecycle.run_maintenance(now_ns=8 * 86400 * 1_000_000_000)
    cold_tier = lifecycle.entries["document"].current_tier.value
    promoted = lifecycle.promote("document")
    lifecycle._drop_to_source("document", remove_warm=True, remove_cold=True)
    reconstructed = lifecycle.promote("document")

    # E2: quantization is measured independently from compression.
    quant_started = time.perf_counter_ns()
    int8_payload, quant_meta = quantize_int8_array(native)
    quant_ns = time.perf_counter_ns() - quant_started
    dequant_started = time.perf_counter_ns()
    restored = dequantize_int8_array(int8_payload, quant_meta)
    dequant_ns = time.perf_counter_ns() - dequant_started

    # E3: fixed pressure compares plain LRU with semantic weighted retention.
    retention = {}
    for policy_name in ("lru", "weighted_lru"):
        cache = _manager(quota=24, eviction=policy_name)
        cache.register(_entry("document", "generic_document", shared=8), b"D" * 16, now_ns=0)
        cache.demote_hot("document", now_ns=1)
        cache.register(_entry("terminal", "terminal_output"), b"L" * 16, now_ns=0)
        cache.demote_hot("terminal", now_ns=2)
        cache.run_maintenance(now_ns=3)
        retention[policy_name] = cache.entries["document"].current_tier == PRAStorageTier.WARM

    # E4--E6: task status, closure, and downstream dependency.
    open_task = _manager()
    open_task.register(_entry("open", "tool_response", task_status="active"), b"O" * 64, now_ns=0)
    open_task.run_maintenance(now_ns=3600 * 1_000_000_000)
    open_retained = open_task.entries["open"].current_tier == PRAStorageTier.HOT
    open_task.entries["open"] = PRAStorageEntry(**{
        **open_task.entries["open"].__dict__, "dependent_record_count": 1
    })
    open_task.update_task_status("task-a", "completed", now_ns=0)
    open_task.run_maintenance(now_ns=400 * 1_000_000_000)
    dependency_retained = open_task.entries["open"].current_tier != PRAStorageTier.SOURCE
    open_task.entries["open"] = PRAStorageEntry(**{
        **open_task.entries["open"].__dict__, "dependent_record_count": 0
    })
    before_close = open_task.usage()["total_native_bytes"]
    open_task.run_maintenance(now_ns=401 * 1_000_000_000)
    task_close_freed = before_close - open_task.usage()["total_native_bytes"]

    # E7: session compaction keeps shared documents and releases transient logs.
    session = _manager()
    session.register(_entry("shared", "generic_document", shared=4), b"S" * 64)
    session.register(_entry("log", "terminal_output"), b"T" * 64)
    session_close_freed = session.close_session("session-a")

    # E9: delayed persistence avoids writes for short-lived tool output.
    immediate = _manager(immediate=True)
    immediate.register(_entry("tool", "tool_response"), b"X" * 64, now_ns=0)
    delayed = _manager()
    delayed.register(_entry("tool", "tool_response"), b"X" * 64, now_ns=0)
    delayed.update_task_status("task-a", "completed", now_ns=0)
    delayed.run_maintenance(now_ns=300 * 1_000_000_000)

    return {
        "engine": engine,
        "seed": seed,
        "e1_cold_tier_reached": cold_tier == "cold",
        "e1_lossless_round_trip": promoted == payload,
        "e2_int8_byte_ratio": len(int8_payload) / len(payload),
        "e2_int8_rmse": float(np.sqrt(np.mean((restored - native) ** 2))),
        "e2_quantize_ms": quant_ns / 1e6,
        "e2_dequantize_ms": dequant_ns / 1e6,
        "e3_lru_retained_document": retention["lru"],
        "e3_weighted_retained_document": retention["weighted_lru"],
        "e4_open_task_retained": open_retained,
        "e5_task_close_bytes_freed": task_close_freed,
        "e6_dependency_retained": dependency_retained,
        "e7_session_close_bytes_freed": session_close_freed,
        "e8_source_reconstruction_exact": reconstructed == payload,
        "e9_immediate_writes": immediate.metrics.persistence_writes,
        "e9_grace_writes": delayed.metrics.persistence_writes,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [_run(engine, seed) for engine in ENGINES for seed in SEEDS]
    with (OUTPUT / "storage_lifecycle_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "engines": list(ENGINES),
        "seeds": list(SEEDS),
        "runs": len(rows),
        "lossless_round_trip_rate": statistics.mean(row["e1_lossless_round_trip"] for row in rows),
        "source_reconstruction_rate": statistics.mean(row["e8_source_reconstruction_exact"] for row in rows),
        "int8_byte_ratio": statistics.mean(row["e2_int8_byte_ratio"] for row in rows),
        "int8_rmse": statistics.mean(row["e2_int8_rmse"] for row in rows),
        "int8_quantize_ms_median": statistics.median(row["e2_quantize_ms"] for row in rows),
        "int8_dequantize_ms_median": statistics.median(row["e2_dequantize_ms"] for row in rows),
        "lru_document_retention_rate": statistics.mean(row["e3_lru_retained_document"] for row in rows),
        "weighted_document_retention_rate": statistics.mean(row["e3_weighted_retained_document"] for row in rows),
        "open_task_retention_rate": statistics.mean(row["e4_open_task_retained"] for row in rows),
        "dependency_retention_rate": statistics.mean(row["e6_dependency_retained"] for row in rows),
        "mean_task_close_bytes_freed": statistics.mean(row["e5_task_close_bytes_freed"] for row in rows),
        "mean_session_close_bytes_freed": statistics.mean(row["e7_session_close_bytes_freed"] for row in rows),
        "immediate_persistence_writes": sum(row["e9_immediate_writes"] for row in rows),
        "grace_persistence_writes": sum(row["e9_grace_writes"] for row in rows),
        "scope": "engine-neutral policy and byte-transport benchmark; not end-task model quality",
    }
    (OUTPUT / "storage_lifecycle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table_rows = (
        ("Shared document retained under pressure", summary["lru_document_retention_rate"], summary["weighted_document_retention_rate"]),
        ("Open-task record retained", "---", summary["open_task_retention_rate"]),
        ("Dependent completed-task record retained", "---", summary["dependency_retention_rate"]),
        ("Transient persistence writes", summary["immediate_persistence_writes"], summary["grace_persistence_writes"]),
        ("COLD native-byte ratio", "1.00 FP32", f"{summary['int8_byte_ratio']:.2f} int8"),
    )
    table = ["\\begin{tabular}{lrr}\\toprule", "Control & Baseline & Semantic lifecycle \\\\", "\\midrule"]
    for label, baseline, semantic in table_rows:
        baseline_text = f"{baseline:.2f}" if isinstance(baseline, float) else str(baseline)
        semantic_text = f"{semantic:.2f}" if isinstance(semantic, float) else str(semantic)
        table.append(f"{label} & {baseline_text} & {semantic_text} \\\\")
    table.append("\\bottomrule\\end{tabular}")
    (OUTPUT / "generated_storage_lifecycle_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].bar(("Plain LRU", "Weighted", "Open task", "Dependency"), (
        summary["lru_document_retention_rate"], summary["weighted_document_retention_rate"],
        summary["open_task_retention_rate"], summary["dependency_retention_rate"],
    ), color=("#6b7280", "#167d78", "#2878b5", "#8a5a9b"))
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Required record retained")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(("FP32", "int8"), (1.0, summary["int8_byte_ratio"]), color=("#555555", "#d97732"))
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Relative native bytes")
    axes[1].set_title(f"int8 RMSE {summary['int8_rmse']:.4f}")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "storage_lifecycle_tradeoffs.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / "storage_lifecycle_tradeoffs.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
