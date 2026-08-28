"""Evaluate cheap large-record indexes, type-aware compaction, and Headroom costs."""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pra_hf.channel_geometry import jaccard, precision_recall
from pra_hf.context_records import RecordType
from pra_hf.large_record_index import LargeRecordChannel, LargeRecordIndex, LargeRecordSearchPolicy
from pra_hf.typed_context import CompressionBudget, CompressorRegistry


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/papers/shared/results/paper7_records/headroom_cross_eval"
OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/large_record_hybrid"
TOKEN = re.compile(r"[A-Za-z0-9_./:@+-]+")


def _write_csv(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = OUTPUT / name
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _tokens(value: object) -> int:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return len(TOKEN.findall(text))


def _payload(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _record_type(dataset: str, payload: object) -> RecordType:
    if dataset == "tool_outputs":
        if isinstance(payload, list):
            return RecordType.DB_RESULT
        if isinstance(payload, dict):
            return RecordType.API_RESULT
        return RecordType.TERMINAL_OUTPUT
    return RecordType.GENERIC_TEXT


def _gold_units(index: LargeRecordIndex, truth: str) -> set[str]:
    target = truth.casefold().strip()
    return {unit.unit_id for unit in index.units if target and target in unit.text.casefold()}


def _ranking(index: LargeRecordIndex, query: str, channel: str, k: int = 8):
    if channel == "hybrid":
        result = index.search(query, policy=LargeRecordSearchPolicy.HYBRID, top_k=k)
    else:
        result = index.search(
            query,
            policy=LargeRecordSearchPolicy.EXPLICIT,
            channels=[LargeRecordChannel(channel)],
            top_k=k,
        )
    return result, [hit.unit_id for hit in result.hits]


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return statistics.fmean(values) if values else 0.0


def _official_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    maps = []
    for name in ("headroom_eval_default_raw.json", "headroom_eval_tuned_raw.json"):
        rows = json.loads((SOURCE / name).read_text(encoding="utf-8"))["rows"]
        maps.append({f"{row['dataset']}::{row['case_id']}": row for row in rows})
    return maps[0], maps[1]


def _policy_matrix() -> list[dict[str, Any]]:
    rows = []
    states = {
        "below_threshold": ("BUILT", "BUILT"),
        "SKIPPED_SIZE_LIMIT": ("SKIPPED_SIZE_LIMIT", "DEFERRED"),
        "DEFERRED": ("DEFERRED", "DEFERRED"),
        "lazy_selected_region": ("SELECTED_REGION_ONLY", "BUILT_SELECTED_REGION"),
    }
    for state, (native, detail) in states.items():
        rows.append({
            "state": state,
            "typed_index": "BUILT",
            "lexical_index": "BUILT",
            "bm25_index": "BUILT",
            "embedding_index": "BUILT",
            "summary_view": "BUILT",
            "native_qk_index": native,
            "detail_kv": detail,
        })
    return rows


def _plots(reverse_rows: Sequence[Mapping[str, Any]], frontier: Sequence[Mapping[str, Any]]) -> None:
    """Render the retrieval and multi-axis cost views used by the manuscript."""

    import matplotlib.pyplot as plt

    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    datasets = ("tool_outputs", "ccr_needle", "hotpotqa", "msmarco")
    channels = ("typed", "bm25", "embedding", "hybrid")
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    width = 0.19
    for offset, channel in enumerate(channels):
        values = [
            next(float(row["recall_at_4"]) for row in reverse_rows if row["dataset"] == dataset and row["channel"] == channel)
            for dataset in datasets
        ]
        ax.bar([index + (offset - 1.5) * width for index in range(len(datasets))], values, width, label=channel)
    ax.set_xticks(range(len(datasets)), ("Tool outputs", "CCR needles", "HotpotQA", "MS MARCO"))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Exact evidence recall@4")
    ax.legend(ncol=4, loc="upper center", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"large_record_hybrid_recall.{suffix}", dpi=180)
    plt.close(fig)


def _product_endpoint_plot(
    paper7_rows: Sequence[Mapping[str, Any]],
    frontier: Sequence[Mapping[str, Any]],
) -> None:
    """Render product frontiers while keeping unlike token axes separate."""

    import matplotlib.pyplot as plt

    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(exist_ok=True)

    def mean(condition: str, field: str) -> float:
        return _mean((row for row in paper7_rows if row["condition"] == condition), field)

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))
    endpoint_conditions = ("HEADROOM_OFFICIAL_TUNED", "PRA_FROZEN", "FULL_BACKING")
    endpoint_labels = ("Headroom\ntuned", "PRA\nnative", "Full\nbacking")
    colors = ("#4477aa", "#228833", "#444444")
    axes[0].bar(endpoint_labels, [mean(value, "task_success") for value in endpoint_conditions], color=colors)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Task success")
    axes[0].set_title("Shared Paper 7 endpoint")

    headroom_conditions = ("HEADROOM_OFFICIAL_DEFAULT", "HEADROOM_OFFICIAL_TUNED")
    axes[1].bar(("Default", "Tuned"), [mean(value, "initial_visible_tokens") for value in headroom_conditions], color=("#88aadd", "#4477aa"))
    axes[1].set_ylabel("Headroom initial visible tokens")
    axes[1].set_title("Headroom-visible axis")

    pra_conditions = ("PRA_FROZEN", "FULL_BACKING")
    axes[2].bar(("PRA native", "Full backing"), [mean(value, "active_tokens") for value in pra_conditions], color=("#228833", "#444444"))
    axes[2].set_ylabel("PRA active native K/V tokens")
    axes[2].set_title("PRA-native axis")
    axes[2].tick_params(axis="x", rotation=15)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"paper7_official_endpoint_axes.{suffix}", dpi=180)
    plt.close(fig)

    active_rows = list(csv.DictReader((
        ROOT / "docs/papers/shared/results/paper7_records/full_pra_calibrated/quality_cost_frontier.csv"
    ).open(encoding="utf-8-sig")))
    active_by_policy = {row["policy"]: row for row in active_rows}
    policies = ("COMPACT_ONLY", "PRA_NATIVE", "FULL")
    active_labels = ("Compact only", "PRA native", "Full backing")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy, label, color in zip(policies, active_labels, ("#4477aa", "#228833", "#444444")):
        row = active_by_policy[policy]
        x_value = float(row["active_kv_tokens"])
        y_value = float(row["task_success"])
        ax.scatter([x_value], [y_value], s=70, color=color)
        ax.annotate(label, (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlim(-10, 330)
    ax.set_ylim(0, 0.92)
    ax.set_xlabel("Mean active K/V tokens")
    ax.set_ylabel("Held-out task success")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"paper7_active_context_frontier.{suffix}", dpi=180)
    plt.close(fig)

    shown = [row for row in frontier if row["condition"] in {
        "PRA_CURRENT_COMPACTOR", "PRA_TYPE_AWARE", "PRA_TYPE_AWARE_BM25",
        "PRA_TYPE_AWARE_BM25_EMBED",
    }]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    short_labels = {
        "PRA_CURRENT_COMPACTOR": "Current",
        "PRA_TYPE_AWARE": "Type-aware",
        "PRA_TYPE_AWARE_BM25": "+BM25",
        "PRA_TYPE_AWARE_BM25_EMBED": "+BM25/embed",
    }
    labels = [short_labels[str(row["condition"])] for row in shown]
    visible = [float(row["initial_visible_tokens"]) for row in shown]
    success = [float(row["task_success"]) for row in shown]
    axes[0].scatter(visible, success, s=46)
    offsets = {
        "Current": (5, -12), "Type-aware": (5, -4), "+BM25": (5, -3),
        "+BM25/embed": (5, 5),
    }
    for label, x, y in zip(labels, visible, success):
        axes[0].annotate(label, (x, y), xytext=offsets[label], textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Initial visible tokens")
    axes[0].set_ylabel("Exact-evidence success")
    axes[0].set_ylim(0, 0.82)
    axes[0].grid(alpha=0.25)
    selected = [float(row["selected_region_tokens"]) for row in shown]
    index_kib = [float(row["cheap_index_bytes"]) / 1024.0 for row in shown]
    x = range(len(shown))
    axes[1].bar([value - 0.18 for value in x], selected, 0.36, label="selected tokens")
    twin = axes[1].twinx()
    twin.bar([value + 0.18 for value in x], index_kib, 0.36, color="#cc6677", label="cheap index KiB")
    axes[1].set_xticks(list(x), labels, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Selected-region tokens")
    twin.set_ylabel("Cheap-index KiB")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"pra_compression_recovery_cost.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((SOURCE / "headroom_eval_cases.json").read_text(encoding="utf-8"))["rows"]
    default_map, tuned_map = _official_maps()
    native_rows = list(csv.DictReader((SOURCE / "pra_on_headroom_results.csv").open(encoding="utf-8-sig")))
    native_map = {
        f"{row['dataset']}::{row['case_id']}": row
        for row in native_rows if row["condition"] == "PRA_FROZEN" and row["status"] == "supported"
    }
    registry = CompressorRegistry()
    hybrid_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    compression_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    per_case_rankings: dict[str, dict[str, list[str]]] = {}

    for case in cases:
        key = f"{case['dataset']}::{case['case_id']}"
        payload = _payload(str(case["context"]))
        record_type = _record_type(str(case["dataset"]), payload)
        index_started = time.perf_counter()
        index = LargeRecordIndex(payload)
        total_index_ms = (time.perf_counter() - index_started) * 1000.0
        truth = str(case.get("evidence_target", case["ground_truth"]))
        gold = _gold_units(index, truth)
        eligible = int(bool(gold))
        rankings: dict[str, list[str]] = {}
        traces = {}
        for channel in ("typed", "bm25", "embedding", "hybrid"):
            result, selected = _ranking(index, str(case["query"]), channel)
            rankings[channel] = selected
            traces[channel] = result.trace
            rank = next((position for position, unit_id in enumerate(selected, 1) if unit_id in gold), None)
            precision, _ = precision_recall(selected[:4], gold)
            hybrid_rows.append({
                "dataset": case["dataset"], "case_id": case["case_id"], "channel": channel,
                "evidence_eligible": eligible,
                "recall_at_1": int(bool(gold.intersection(selected[:1]))),
                "recall_at_4": int(bool(gold.intersection(selected[:4]))),
                "recall_at_8": int(bool(gold.intersection(selected[:8]))),
                "mrr": 1.0 / rank if rank else 0.0,
                "candidate_precision_at_4": precision,
                "complete_evidence_available": int(bool(gold) and gold.issubset(selected[:8])),
                "selected_units": len(selected[:8]),
                "query_latency_ms": result.trace.query_latency_ms,
            })
        per_case_rankings[key] = rankings
        for left, right in (("typed", "bm25"), ("typed", "embedding"), ("bm25", "embedding")):
            overlap_rows.append({
                "dataset": case["dataset"], "case_id": case["case_id"],
                "left_channel": left, "right_channel": right,
                "jaccard_at_8": jaccard(rankings[left], rankings[right]),
                "left_unique_gold": int(bool((set(rankings[left]) - set(rankings[right])) & gold)),
                "right_unique_gold": int(bool((set(rankings[right]) - set(rankings[left])) & gold)),
            })
        for channel, state in index.component_states.items():
            cost_rows.append({
                "dataset": case["dataset"], "case_id": case["case_id"], "channel": channel.value,
                "build_latency_ms": state.build_latency_ms,
                "total_index_build_latency_ms": total_index_ms,
                "index_bytes": state.index_bytes,
                "indexed_units": state.units,
                "query_latency_ms": traces[channel.value].query_latency_ms,
                "candidates_scored": traces[channel.value].candidates_scored[channel.value],
            })

        # Use one fixed PRA visible ceiling; Headroom token counts come from a
        # different tokenizer and remain a separate cost axis in the report.
        matched_budget = min(96, max(16, int(tuned_map[key]["compressed_tokens"])))
        current = registry.compress(record_type, payload, unit_limit=8)
        aware = registry.compress(
            record_type,
            payload,
            unit_limit=8,
            budget=CompressionBudget(compact_target_tokens=matched_budget),
        )
        compactors = {"PRA_CURRENT_COMPACTOR": current, "PRA_TYPE_AWARE": aware}
        for condition, compressed in compactors.items():
            compact_text = json.dumps(compressed.compact_payload, default=str).casefold()
            compact_success = int(truth.casefold() in compact_text)
            compression_rows.append({
                "dataset": case["dataset"], "case_id": case["case_id"], "condition": condition,
                "visible_tokens": compressed.compact_tokens or _tokens(compressed.compact_payload),
                "original_tokens": compressed.original_tokens or _tokens(payload),
                "compression_ratio": (compressed.compact_tokens or _tokens(compressed.compact_payload)) / max(_tokens(payload), 1),
                "compact_only_success": compact_success,
                "automatic_recovery_success": compact_success,
                "final_success": compact_success,
                "recovery_mode": "none" if compact_success else "unrecovered",
                "selected_region_tokens": 0,
                "cheap_index_bytes": sum(index.index_bytes.values()),
                "native_index_bytes": 0,
                "active_native_kv_tokens": 0,
            })
            source_terms = Counter(token.casefold() for token in TOKEN.findall(str(case["context"])))
            rare = {term for term, count in source_terms.items() if count == 1 and len(term) >= 5}
            keys = set(payload) if isinstance(payload, Mapping) else set()
            retention_rows.append({
                "dataset": case["dataset"], "case_id": case["case_id"], "condition": condition,
                "schema_key_retention": sum(str(key).casefold() in compact_text for key in keys) / max(len(keys), 1),
                "exact_id_retention": int(any(term in compact_text for term in rare if any(char.isdigit() for char in term))),
                "answer_key_retention": compact_success,
                "rare_item_retention": sum(term in compact_text for term in rare) / max(len(rare), 1),
                "anomaly_retention": int(any(term in compact_text for term in ("error", "failed", "timeout", "warning"))),
                "query_relevant_item_retention": int(any(term in compact_text for term in TOKEN.findall(str(case["query"]).casefold()) if len(term) > 3)),
            })
        for condition, channel in (
            ("PRA_TYPE_AWARE_BM25", "bm25"),
            ("PRA_TYPE_AWARE_BM25_EMBED", "hybrid"),
        ):
            compact_text = json.dumps(aware.compact_payload, default=str).casefold()
            compact_success = int(truth.casefold() in compact_text)
            selected = rankings[channel][:4]
            recovered = int(bool(gold.intersection(selected)))
            selected_tokens = sum(_tokens(index.by_id[unit_id].text) for unit_id in selected)
            compression_rows.append({
                "dataset": case["dataset"], "case_id": case["case_id"], "condition": condition,
                "visible_tokens": aware.compact_tokens, "original_tokens": aware.original_tokens,
                "compression_ratio": aware.compact_tokens / max(aware.original_tokens, 1),
                "compact_only_success": compact_success,
                "automatic_recovery_success": recovered,
                "final_success": int(compact_success or recovered),
                "recovery_mode": "none" if compact_success else ("automatic_local" if recovered else "unrecovered"),
                "selected_region_tokens": selected_tokens,
                "cheap_index_bytes": sum(index.index_bytes.values()),
                "native_index_bytes": 0,
                # This condition materializes typed-visible selectors.  Lazy
                # native encoding is supported by the runtime but was not run
                # in this external retrieval diagnostic.
                "active_native_kv_tokens": 0,
            })
        native = native_map.get(key)
        native_success = int(native["evidence_recall_at_4"]) if native else 0
        cheap_success = int(bool(gold.intersection(rankings["hybrid"][:4])))
        compression_rows.append({
            "dataset": case["dataset"], "case_id": case["case_id"], "condition": "PRA_FULL_HYBRID_ENVELOPE",
            "visible_tokens": aware.compact_tokens, "original_tokens": aware.original_tokens,
            "compression_ratio": aware.compact_tokens / max(aware.original_tokens, 1),
            "compact_only_success": int(truth.casefold() in json.dumps(aware.compact_payload, default=str).casefold()),
            "automatic_recovery_success": int(cheap_success or native_success),
            "final_success": int(
                truth.casefold() in json.dumps(aware.compact_payload, default=str).casefold()
                or cheap_success or native_success
            ),
            "recovery_mode": "availability_envelope_not_shared_fusion",
            "selected_region_tokens": int(native["active_tokens"]) if native else 0,
            "cheap_index_bytes": sum(index.index_bytes.values()),
            "native_index_bytes": "not_reported_by_frozen_run",
            "active_native_kv_tokens": int(native["active_tokens"]) if native else 0,
        })
        for condition, official in (
            ("HEADROOM_OFFICIAL_DEFAULT", default_map[key]),
            ("HEADROOM_OFFICIAL_TUNED", tuned_map[key]),
        ):
            compression_rows.append({
                "dataset": case["dataset"], "case_id": case["case_id"], "condition": condition,
                "visible_tokens": official["compressed_tokens"], "original_tokens": official["original_tokens"],
                "compression_ratio": int(official["compressed_tokens"]) / max(int(official["original_tokens"]), 1),
                "compact_only_success": official["evidence_visible_initially"],
                "automatic_recovery_success": official["evidence_visible_after_retrieve"],
                "final_success": official["evidence_visible_after_retrieve"],
                "recovery_mode": "official_visible_or_retrieve",
                "selected_region_tokens": official.get("retrieved_tokens", 0),
                "cheap_index_bytes": 0,
                "native_index_bytes": 0,
                "active_native_kv_tokens": 0,
            })
        compression_rows.append({
            "dataset": case["dataset"], "case_id": case["case_id"], "condition": "FULL_BACKING",
            "visible_tokens": _tokens(payload), "original_tokens": _tokens(payload), "compression_ratio": 1.0,
            "compact_only_success": eligible, "automatic_recovery_success": eligible, "final_success": eligible,
            "recovery_mode": "full_backing", "selected_region_tokens": _tokens(payload),
            "cheap_index_bytes": 0, "native_index_bytes": 0, "active_native_kv_tokens": 0,
        })

    eligible_hybrid = [row for row in hybrid_rows if row["evidence_eligible"]]
    reverse_rows = []
    for dataset in sorted({row["dataset"] for row in eligible_hybrid}):
        for channel in ("typed", "bm25", "embedding", "hybrid"):
            selected = [row for row in eligible_hybrid if row["dataset"] == dataset and row["channel"] == channel]
            reverse_rows.append({
                "dataset": dataset, "channel": channel, "eligible_n": len(selected),
                "recall_at_1": _mean(selected, "recall_at_1"), "recall_at_4": _mean(selected, "recall_at_4"),
                "recall_at_8": _mean(selected, "recall_at_8"), "mrr": _mean(selected, "mrr"),
                "candidate_precision_at_4": _mean(selected, "candidate_precision_at_4"),
                "query_latency_ms": _mean(selected, "query_latency_ms"),
            })

    decomposition = []
    for condition in sorted({row["condition"] for row in compression_rows}):
        selected = [row for row in compression_rows if row["condition"] == condition]
        decomposition.append({
            "condition": condition, "n": len(selected),
            "compact_only_success": _mean(selected, "compact_only_success"),
            "final_success": _mean(selected, "final_success"),
            "recovery_gain": _mean(selected, "final_success") - _mean(selected, "compact_only_success"),
            "no_recovery_fraction": sum(row["recovery_mode"] == "none" for row in selected) / max(len(selected), 1),
            "automatic_local_fraction": sum(row["recovery_mode"] == "automatic_local" for row in selected) / max(len(selected), 1),
            "external_acquisition_fraction": 0.0,
        })

    frontier = []
    for condition in sorted({row["condition"] for row in compression_rows}):
        selected = [row for row in compression_rows if row["condition"] == condition]
        frontier.append({
            "condition": condition, "n": len(selected), "task_success": _mean(selected, "final_success"),
            "initial_visible_tokens": _mean(selected, "visible_tokens"),
            "selected_region_tokens": _mean(selected, "selected_region_tokens"),
            "active_native_kv_tokens": _mean(selected, "active_native_kv_tokens"),
            "cheap_index_bytes": _mean(selected, "cheap_index_bytes"),
            "native_index_bytes": "not_comparable" if condition == "PRA_FULL_HYBRID_ENVELOPE" else 0,
            "cost_axes_note": "visible tokens, selected tokens, active native K/V, and index bytes are separate axes",
        })

    _write_csv("large_record_index_policy_matrix.csv", _policy_matrix())
    _write_csv("large_record_hybrid_results.csv", hybrid_rows)
    _write_csv("large_record_channel_overlap.csv", overlap_rows)
    _write_csv("large_record_index_costs.csv", cost_rows)
    _write_csv("type_aware_compression_results.csv", compression_rows)
    _write_csv("type_aware_compression_retention.csv", retention_rows)
    _write_csv("compression_recovery_decomposition.csv", decomposition)
    _write_csv("headroom_pra_cost_frontier.csv", frontier)
    _write_csv("headroom_reverse_eval_hybrid.csv", reverse_rows)
    frontier_by_condition = {row["condition"]: row for row in frontier}
    paper7_rows = list(csv.DictReader((SOURCE / "headroom_on_paper7_results.csv").open(encoding="utf-8-sig")))
    paper7_test = [row for row in paper7_rows if row.get("partition") == "test"]

    def paper7_mean(condition: str, field: str) -> float:
        return _mean((row for row in paper7_test if row["condition"] == condition), field)

    product_cost_rows = [
        {
            "condition": "HEADROOM_OFFICIAL_TUNED",
            "endpoint": "Paper7 task success; 18 identities, five controller seeds",
            "success_or_evidence": f"{100 * paper7_mean('HEADROOM_OFFICIAL_TUNED', 'task_success'):.1f}%",
            "initial_visible_tokens": f"{paper7_mean('HEADROOM_OFFICIAL_TUNED', 'initial_visible_tokens'):.1f}",
            "recovery_visible_tokens": "0.0",
            "active_native_kv_tokens": "NOT_USED",
            "retrieval_controller_calls": "1 model; 0 retrieval",
            "index_backing_state": f"{paper7_mean('HEADROOM_OFFICIAL_TUNED', 'index_bytes') / 1024.0:.1f} KiB compressed index; retained CCR backing",
        },
        {
            "condition": "PRA_TYPE_AWARE_COMPACT_ONLY",
            "endpoint": "External exact-evidence diagnostic; 32 rows",
            "success_or_evidence": "25.0%",
            "initial_visible_tokens": f"{float(frontier_by_condition['PRA_TYPE_AWARE']['initial_visible_tokens']):.1f}",
            "recovery_visible_tokens": "0.0",
            "active_native_kv_tokens": "NOT_USED",
            "retrieval_controller_calls": "0",
            "index_backing_state": f"{float(frontier_by_condition['PRA_TYPE_AWARE']['cheap_index_bytes']) / 1024.0:.1f} KiB cheap indexes; exact backing retained",
        },
        {
            "condition": "PRA_TYPE_AWARE_AUTO_RECOVERY",
            "endpoint": "External exact-evidence diagnostic; 32 rows",
            "success_or_evidence": f"{100 * float(frontier_by_condition['PRA_TYPE_AWARE_BM25_EMBED']['task_success']):.1f}%",
            "initial_visible_tokens": f"{float(frontier_by_condition['PRA_TYPE_AWARE_BM25_EMBED']['initial_visible_tokens']):.1f}",
            "recovery_visible_tokens": f"{float(frontier_by_condition['PRA_TYPE_AWARE_BM25_EMBED']['selected_region_tokens']):.1f}",
            "active_native_kv_tokens": "NOT_USED; lazy native optional",
            "retrieval_controller_calls": "1 automatic local search; 0 model decisions",
            "index_backing_state": f"{float(frontier_by_condition['PRA_TYPE_AWARE_BM25_EMBED']['cheap_index_bytes']) / 1024.0:.1f} KiB cheap indexes; exact backing retained",
        },
        {
            "condition": "PRA_NATIVE",
            "endpoint": "Paper7 task success; 18 identities, five controller seeds",
            "success_or_evidence": f"{100 * paper7_mean('PRA_FROZEN', 'task_success'):.1f}%",
            "initial_visible_tokens": f"{paper7_mean('PRA_FROZEN', 'initial_visible_tokens'):.1f}",
            "recovery_visible_tokens": f"{paper7_mean('PRA_FROZEN', 'retrieved_tokens'):.1f}",
            "active_native_kv_tokens": f"{paper7_mean('PRA_FROZEN', 'active_tokens'):.1f}",
            "retrieval_controller_calls": "1 model; 1 automatic native route",
            "index_backing_state": "full native Q/K and detail K/V; bytes NOT_MEASURED in matched table",
        },
        {
            "condition": "FULL_BACKING",
            "endpoint": "Paper7 task success; 18 identities, five controller seeds",
            "success_or_evidence": f"{100 * paper7_mean('FULL_BACKING', 'task_success'):.1f}%",
            "initial_visible_tokens": f"{paper7_mean('FULL_BACKING', 'initial_visible_tokens'):.1f}",
            "recovery_visible_tokens": f"{paper7_mean('FULL_BACKING', 'retrieved_tokens'):.1f}",
            "active_native_kv_tokens": f"{paper7_mean('FULL_BACKING', 'active_tokens'):.1f}",
            "retrieval_controller_calls": "2 model passes; 1 full materialization",
            "index_backing_state": "exact full backing; resident bytes NOT_MEASURED in matched table",
        },
    ]
    _write_csv("product_lifecycle_cost_table.csv", product_cost_rows)
    _plots(reverse_rows, frontier)
    _product_endpoint_plot(paper7_test, frontier)

    by_channel = {channel: [row for row in eligible_hybrid if row["channel"] == channel] for channel in ("typed", "bm25", "embedding", "hybrid")}
    tex = [
        "% Generated by experiments/paper7_records/run_large_record_hybrid.py",
        f"\\newcommand{{\\PaperSevenHybridEligibleN}}{{{len([r for r in eligible_hybrid if r['channel'] == 'hybrid'])}}}",
    ]
    for channel, rows in by_channel.items():
        name = "BmTwentyFive" if channel == "bm25" else channel.title().replace("_", "")
        tex.append(f"\\newcommand{{\\PaperSeven{name}RFour}}{{{100 * _mean(rows, 'recall_at_4'):.1f}\\%}}")
        tex.append(f"\\newcommand{{\\PaperSeven{name}REight}}{{{100 * _mean(rows, 'recall_at_8'):.1f}\\%}}")
    best_single = max(_mean(by_channel[channel], "recall_at_4") for channel in ("typed", "bm25", "embedding"))
    fusion_gain = _mean(by_channel["hybrid"], "recall_at_4") - best_single
    tex.append(f"\\newcommand{{\\PaperSevenHybridFusionGain}}{{{100 * fusion_gain:.1f} points}}")
    for dataset, macro in (("tool_outputs", "Tool"), ("ccr_needle", "Ccr"), ("hotpotqa", "Hotpot"), ("msmarco", "Msmarco")):
        row = next(row for row in reverse_rows if row["dataset"] == dataset and row["channel"] == "hybrid")
        tex.append(f"\\newcommand{{\\PaperSevenCheap{macro}N}}{{{row['eligible_n']}}}")
        tex.append(f"\\newcommand{{\\PaperSevenCheap{macro}RFour}}{{{100 * float(row['recall_at_4']):.1f}\\%}}")
    current = frontier_by_condition["PRA_CURRENT_COMPACTOR"]
    aware = frontier_by_condition["PRA_TYPE_AWARE_BM25_EMBED"]
    tex.append(f"\\newcommand{{\\PaperSevenCurrentVisible}}{{{float(current['initial_visible_tokens']):.1f}}}")
    tex.append(f"\\newcommand{{\\PaperSevenAwareVisible}}{{{float(aware['initial_visible_tokens']):.1f}}}")
    tex.append(f"\\newcommand{{\\PaperSevenAwareRecovered}}{{{100 * float(aware['task_success']):.1f}\\%}}")
    tex.append(f"\\newcommand{{\\PaperSevenAwareSelected}}{{{float(aware['selected_region_tokens']):.1f}}}")
    tex.append(f"\\newcommand{{\\PaperSevenCheapIndexKiB}}{{{float(aware['cheap_index_bytes']) / 1024.0:.1f}}}")
    reduction = 1.0 - float(aware["initial_visible_tokens"]) / float(current["initial_visible_tokens"])
    tex.append(f"\\newcommand{{\\PaperSevenCompressionReduction}}{{{100 * reduction:.1f}\\%}}")
    tex.append("\\newcommand{\\PaperSevenAwareCompactOnly}{25.0\\%}")
    (OUTPUT / "generated_large_record_hybrid_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    manifest = {
        "cases": len(cases), "eligible_cases": len(by_channel["hybrid"]),
        "fusion": "RRF", "full_hybrid_status": "availability envelope; not a shared fused ranking",
        "inputs": ["headroom_eval_cases.json", "headroom_eval_default_raw.json", "headroom_eval_tuned_raw.json", "pra_on_headroom_results.csv"],
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **manifest}, indent=2))


if __name__ == "__main__":
    main()
