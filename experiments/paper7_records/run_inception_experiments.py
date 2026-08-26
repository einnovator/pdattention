"""Run the Paper 7 deterministic inception study and render paper figures."""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorPolicy,
    DeploymentTopology,
    MaterializationEvent,
    StoragePolicy,
    TypeContextPolicy,
)
from pra_hf.context_records import RecordType, RecordViewName
from pra_hf.context_store import LocalBackingStore, RecordScope


OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/inception"
SEEDS = (11, 23, 37, 53, 71)


def payload_fixtures(seed: int) -> tuple[tuple[RecordType, object, str, str], ...]:
    """Return typed payloads with one low-salience required action each."""

    offset = seed % 7
    db_rows = [
        {"id": index, "latency_ms": 10 + index, "state": "ok", "detail": f"row-{index}"}
        for index in range(40)
    ]
    db_rows[17 + offset]["detail"] = "RARE_DB_Z91 requires ESCALATE_DATABASE"
    graph_nodes = [{"id": f"node-{index}", "label": f"ordinary {index}"} for index in range(32)]
    graph_nodes[9 + offset]["label"] = "RARE_GRAPH_G44 requires INSPECT_NEIGHBOR"
    rag_lines = [f"Background passage {index}." for index in range(40)]
    rag_lines[11 + offset] = "RARE_RAG_R82 means VERIFY_CITATION before answering."
    file_lines = [f"configuration line {index}=normal" for index in range(35)]
    file_lines[8 + offset] = "RARE_FILE_F13=ROTATE_SECRET"
    generic_lines = [f"Observation {index} is routine." for index in range(30)]
    generic_lines[12 + offset] = "RARE_TEXT_T61 requires RECHECK_ASSUMPTION."
    return (
        (
            RecordType.TOOL_RESPONSE,
            {
                "status": "ok", "request_id": f"req-{seed}", "duration_ms": 43,
                "items": 19, "trace": "ordinary", "diagnostic": "RARE_TOOL_A17 requires ROLLBACK_PAYMENT",
            },
            "RARE_TOOL_A17",
            "ROLLBACK_PAYMENT",
        ),
        (
            RecordType.LOG_BLOCK,
            {
                "source": "worker",
                "events": [f"INFO batch {index} complete" for index in range(45)],
            },
            "RARE_LOG_L72",
            "RESTART_WORKER",
        ),
        (
            RecordType.TERMINAL_OUTPUT,
            {
                "command": "build", "exit_status": 1, "working_directory": "/workspace",
                "stdout": "\n".join(f"compile unit {index}" for index in range(30)),
                "stderr": "FATAL RARE_TERM_C31 requires CLEAN_CACHE",
            },
            "RARE_TERM_C31",
            "CLEAN_CACHE",
        ),
        (
            RecordType.DB_RESULT,
            {"columns": ["id", "latency_ms", "state", "detail"], "rows": db_rows},
            "RARE_DB_Z91",
            "ESCALATE_DATABASE",
        ),
        (
            RecordType.GRAPH_RESULT,
            {
                "nodes": graph_nodes,
                "edges": [{"source": f"node-{i}", "target": f"node-{i + 1}"} for i in range(31)],
            },
            "RARE_GRAPH_G44",
            "INSPECT_NEIGHBOR",
        ),
        (RecordType.RAG_RESULT, "\n".join(rag_lines), "RARE_RAG_R82", "VERIFY_CITATION"),
        (RecordType.FILE_READ, "\n".join(file_lines), "RARE_FILE_F13", "ROTATE_SECRET"),
        (
            RecordType.API_RESULT,
            {
                "status": 200, "request": f"api-{seed}", "count": 8, "next": None,
                "metadata": "normal", "warning": "RARE_API_P20 requires REFRESH_TOKEN",
            },
            "RARE_API_P20",
            "REFRESH_TOKEN",
        ),
        (RecordType.GENERIC_TEXT, "\n".join(generic_lines), "RARE_TEXT_T61", "RECHECK_ASSUMPTION"),
    )


def _inject_log_trigger(payloads, seed: int):
    rows = list(payloads)
    record_type, payload, trigger, action = rows[1]
    payload = dict(payload)
    events = list(payload["events"])
    events[21 + seed % 7] = f"FATAL {trigger} requires {action}"
    payload["events"] = events
    rows[1] = (record_type, payload, trigger, action)
    return tuple(rows)


def run_compression_and_trigger_study() -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        with tempfile.TemporaryDirectory(prefix="paper7-compression-") as directory:
            policy = ContextPolicy(
                local_store=directory,
                persistent_store=True,
                record_policies={record_type: TypeContextPolicy(unit_limit=4) for record_type in RecordType},
            )
            runtime = AdaptiveContextRuntime(RecordScope("paper7", f"seed-{seed}"), policy)
            for record_type, payload, trigger, action in _inject_log_trigger(payload_fixtures(seed), seed):
                record = runtime.ingest(payload, record_type=record_type, provenance={"seed": seed})
                compact_text = json.dumps(record.compact_view(), sort_keys=True, default=str).casefold()
                explicit = runtime.search_records(trigger, top_k=1)
                latent = runtime.search_records("continue safely", top_k=1)
                proactive = runtime.search_records(action, top_k=1)
                exact = runtime.retrieve_record(record.record_id).payload == payload
                rows.append({
                    "seed": seed,
                    "record_type": record_type.value,
                    "record_id": record.record_id,
                    "original_bytes": record.backing.size_bytes,
                    "compact_bytes": record.compact_bytes,
                    "byte_savings_fraction": record.metadata["byte_savings_fraction"],
                    "trigger_visible_compact": trigger.casefold() in compact_text,
                    "explicit_address_recovery": bool(explicit and explicit[0].record_id == record.record_id),
                    "latent_query_recovery": bool(latent and latent[0].record_id == record.record_id),
                    "proactive_action_probe_recovery": bool(proactive and proactive[0].record_id == record.record_id),
                    "exact_recovery": exact,
                })
    return rows


def _mechanism_runtime(directory: str, mode: str) -> AdaptiveContextRuntime:
    return AdaptiveContextRuntime(
        RecordScope("paper7", f"mechanism-{mode}"),
        ContextPolicy(
            storage=StoragePolicy.ON_DEMAND,
            retrieval_mode=mode,
            topology=DeploymentTopology.REMOTE_MODEL,
            local_store=directory,
            persistent_store=True,
            cursor_policy=CursorPolicy(page_size=5, max_page_size=20),
            record_policies={RecordType.DB_RESULT: TypeContextPolicy(unit_limit=4)},
        ),
    )


def run_retrieval_modes() -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        payload = {
            "columns": ["id", "group", "value"],
            "rows": [{"id": index, "group": index % 4, "value": index + seed} for index in range(100)],
        }
        for mode in ("native_event", "tool", "mixed", "proactive"):
            with tempfile.TemporaryDirectory(prefix="paper7-mode-") as directory:
                runtime = _mechanism_runtime(directory, mode)
                record = runtime.ingest(payload, record_type=RecordType.DB_RESULT)
                started = __import__("time").perf_counter()
                if mode == "native_event":
                    result = runtime.materialize(MaterializationEvent(record.record_id))
                    returned = result.payload["rows"]
                    result_bytes = result.payload_bytes
                    network_bytes = result.network_bytes
                    round_trips = result.round_trips
                elif mode == "tool":
                    result = runtime.retrieve_record(record.record_id)
                    returned = result.payload["rows"]
                    result_bytes = result.payload_bytes
                    network_bytes = result.network_bytes
                    round_trips = result.round_trips
                elif mode == "mixed":
                    cursor = runtime.open_cursor(record.record_id)
                    page = runtime.fetch_cursor(cursor.cursor_id)
                    returned = page.items
                    result_bytes = len(json.dumps(page.items, default=str).encode("utf-8"))
                    network_bytes = result_bytes
                    round_trips = 1
                else:
                    result = runtime.retrieve_record(
                        record.record_id,
                        level=RecordViewName.SELECTED,
                        selector={"rows": [0, 5]},
                    )
                    returned = result.payload["rows"]
                    result_bytes = result.payload_bytes
                    network_bytes = result.network_bytes
                    round_trips = result.round_trips
                elapsed = __import__("time").perf_counter() - started
                rows.append({
                    "seed": seed,
                    "mode": mode,
                    "returned_rows": len(returned),
                    "row_coverage": len(returned) / 100,
                    "visible_bytes": result_bytes,
                    "network_bytes": network_bytes,
                    "round_trips": round_trips,
                    "runtime_seconds": elapsed,
                })
    return rows


def run_cursor_analytics() -> list[dict[str, object]]:
    rows = []
    for seed in SEEDS:
        with tempfile.TemporaryDirectory(prefix="paper7-cursor-") as directory:
            runtime = _mechanism_runtime(directory, "mixed")
            payload = {
                "columns": ["group", "value", "anomaly"],
                "rows": [
                    {"group": group, "value": seed + index, "anomaly": index == 17}
                    for index, group in enumerate(["a", "b", "c", "b"] * 20)
                ],
            }
            record = runtime.ingest(payload, record_type=RecordType.DB_RESULT)
            cursor = runtime.open_cursor(record.record_id, filters={"group": "b"})
            aggregate = runtime.cursors.aggregate(cursor.cursor_id, "value", scope=runtime.scope)
            anomalies = runtime.cursors.search(cursor.cursor_id, "true", scope=runtime.scope, limit=20)
            expected = [row for row in payload["rows"] if row["group"] == "b"]
            rows.append({
                "seed": seed,
                "filtered_count_correct": aggregate["count"] == len(expected),
                "mean_correct": abs(aggregate["mean"] - statistics.mean(row["value"] for row in expected)) < 1e-12,
                "anomaly_recovered": any(row["anomaly"] for row in anomalies),
                "cursor_page_size": cursor.page_size,
                "full_rows": len(payload["rows"]),
                "filtered_rows": len(expected),
            })
    return rows


def run_transport_sweep() -> list[dict[str, object]]:
    rows = []
    for topology in DeploymentTopology:
        for size in (10_000, 100_000, 1_000_000):
            for expected_reuse in (0.1, 0.8):
                with tempfile.TemporaryDirectory(prefix="paper7-transport-") as directory:
                    runtime = AdaptiveContextRuntime(
                        RecordScope("paper7", f"{topology.value}-{size}-{expected_reuse}"),
                        ContextPolicy(
                            storage="adaptive",
                            topology=topology,
                            local_store=directory,
                            persistent_store=True,
                            upfront_max_bytes=100_000,
                            adaptive_reuse_max_bytes=1_000_000,
                        ),
                    )
                    record = runtime.ingest(
                        "x" * size,
                        record_type=RecordType.GENERIC_TEXT,
                        expected_reuse=expected_reuse,
                    )
                    decision = runtime.decisions[record.record_id]
                    rows.append({
                        "topology": topology.value,
                        "requested_bytes": size,
                        "stored_bytes": record.backing.size_bytes,
                        "expected_reuse": expected_reuse,
                        "decision": decision.storage.value,
                        "ingest_network_bytes": decision.bytes_transferred,
                        "expected_round_trips": decision.expected_round_trips,
                        "reason": decision.reason,
                    })
    return rows


def summarize(compression, modes, cursors, transport):
    by_type = defaultdict(list)
    for row in compression:
        by_type[row["record_type"]].append(row)
    compression_summary = {
        record_type: {
            "examples": len(values),
            "mean_byte_savings_fraction": statistics.mean(row["byte_savings_fraction"] for row in values),
            "compact_trigger_recall": statistics.mean(row["trigger_visible_compact"] for row in values),
            "explicit_address_recall": statistics.mean(row["explicit_address_recovery"] for row in values),
            "latent_query_recall": statistics.mean(row["latent_query_recovery"] for row in values),
            "proactive_action_probe_recall": statistics.mean(row["proactive_action_probe_recovery"] for row in values),
            "exact_recovery": statistics.mean(row["exact_recovery"] for row in values),
        }
        for record_type, values in sorted(by_type.items())
    }
    by_mode = defaultdict(list)
    for row in modes:
        by_mode[row["mode"]].append(row)
    mode_summary = {
        mode: {
            "examples": len(values),
            "mean_visible_bytes": statistics.mean(row["visible_bytes"] for row in values),
            "mean_network_bytes": statistics.mean(row["network_bytes"] for row in values),
            "mean_row_coverage": statistics.mean(row["row_coverage"] for row in values),
            "mean_runtime_ms": statistics.mean(row["runtime_seconds"] for row in values) * 1000,
            "median_runtime_ms": statistics.median(row["runtime_seconds"] for row in values) * 1000,
        }
        for mode, values in sorted(by_mode.items())
    }
    return {
        "protocol": {"seeds": list(SEEDS), "scope": "deterministic mechanism study"},
        "compression_by_type": compression_summary,
        "overall_trigger_recovery": {
            "compact": statistics.mean(row["trigger_visible_compact"] for row in compression),
            "explicit_address": statistics.mean(row["explicit_address_recovery"] for row in compression),
            "latent_query": statistics.mean(row["latent_query_recovery"] for row in compression),
            "proactive_action_probe": statistics.mean(row["proactive_action_probe_recovery"] for row in compression),
            "exact_recovery": statistics.mean(row["exact_recovery"] for row in compression),
        },
        "retrieval_modes": mode_summary,
        "cursor_analytics": {
            "examples": len(cursors),
            "filtered_count_accuracy": statistics.mean(row["filtered_count_correct"] for row in cursors),
            "mean_accuracy": statistics.mean(row["mean_correct"] for row in cursors),
            "anomaly_recall": statistics.mean(row["anomaly_recovered"] for row in cursors),
        },
        "transport_rows": len(transport),
    }


def render_figures(summary):
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    types = list(summary["compression_by_type"])
    savings = [summary["compression_by_type"][name]["mean_byte_savings_fraction"] for name in types]
    fig, ax = plt.subplots(figsize=(8.4, 4.1))
    ax.barh(types, savings, color="#277da1")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean serialized-byte savings fraction")
    ax.set_title("Typed compact views, five seeds")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "typed_compression.pdf")
    fig.savefig(figure_dir / "typed_compression.png", dpi=180)
    plt.close(fig)

    recovery = summary["overall_trigger_recovery"]
    labels = ["Compact", "Explicit\naddress", "Latent\nquery", "Proactive\naction probe", "Exact\nbacking"]
    values = [recovery[key] for key in ("compact", "explicit_address", "latent_query", "proactive_action_probe", "exact_recovery")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(labels, values, color=["#f8961e", "#43aa8b", "#d1495b", "#577590", "#277da1"])
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Trigger/evidence reachability")
    ax.set_title("Addressability mechanism check, 45 records")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "trigger_recovery.pdf")
    fig.savefig(figure_dir / "trigger_recovery.png", dpi=180)
    plt.close(fig)


def render_tex(summary):
    compression = summary["compression_by_type"]
    mean_savings = statistics.mean(
        value["mean_byte_savings_fraction"] for value in compression.values()
    )
    recovery = summary["overall_trigger_recovery"]
    modes = summary["retrieval_modes"]
    cursor = summary["cursor_analytics"]
    lines = [
        "% Generated by experiments/paper7_records/run_inception_experiments.py.",
        f"\\newcommand{{\\PaperSevenSeeds}}{{{len(SEEDS)}}}",
        f"\\newcommand{{\\PaperSevenTypedRecords}}{{{len(SEEDS) * len(compression)}}}",
        f"\\newcommand{{\\PaperSevenMeanSavings}}{{{mean_savings:.3f}}}",
        f"\\newcommand{{\\PaperSevenCompactTrigger}}{{{recovery['compact']:.3f}}}",
        f"\\newcommand{{\\PaperSevenExplicitAddress}}{{{recovery['explicit_address']:.3f}}}",
        f"\\newcommand{{\\PaperSevenLatentQuery}}{{{recovery['latent_query']:.3f}}}",
        f"\\newcommand{{\\PaperSevenProactiveProbe}}{{{recovery['proactive_action_probe']:.3f}}}",
        f"\\newcommand{{\\PaperSevenExactRecovery}}{{{recovery['exact_recovery']:.3f}}}",
        f"\\newcommand{{\\PaperSevenFullBytes}}{{{modes['native_event']['mean_network_bytes']:.0f}}}",
        f"\\newcommand{{\\PaperSevenMixedBytes}}{{{modes['mixed']['mean_network_bytes']:.0f}}}",
        f"\\newcommand{{\\PaperSevenSelectedBytes}}{{{modes['proactive']['mean_network_bytes']:.0f}}}",
        f"\\newcommand{{\\PaperSevenMixedCoverage}}{{{modes['mixed']['mean_row_coverage']:.3f}}}",
        f"\\newcommand{{\\PaperSevenNativeMedianMs}}{{{modes['native_event']['median_runtime_ms']:.2f}}}",
        f"\\newcommand{{\\PaperSevenToolMedianMs}}{{{modes['tool']['median_runtime_ms']:.2f}}}",
        f"\\newcommand{{\\PaperSevenMixedMedianMs}}{{{modes['mixed']['median_runtime_ms']:.2f}}}",
        f"\\newcommand{{\\PaperSevenProactiveMedianMs}}{{{modes['proactive']['median_runtime_ms']:.2f}}}",
        f"\\newcommand{{\\PaperSevenCursorAggregate}}{{{cursor['mean_accuracy']:.3f}}}",
        f"\\newcommand{{\\PaperSevenCursorAnomaly}}{{{cursor['anomaly_recall']:.3f}}}",
    ]
    (OUTPUT / "generated_inception_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    compression = run_compression_and_trigger_study()
    modes = run_retrieval_modes()
    cursors = run_cursor_analytics()
    transport = run_transport_sweep()
    summary = summarize(compression, modes, cursors, transport)
    artifacts = {
        "compression_trigger_rows.json": compression,
        "retrieval_mode_rows.json": modes,
        "cursor_rows.json": cursors,
        "transport_rows.json": transport,
        "summary.json": summary,
    }
    for name, value in artifacts.items():
        (OUTPUT / name).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    render_figures(summary)
    render_tex(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
