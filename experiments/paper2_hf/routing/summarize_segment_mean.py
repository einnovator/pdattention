"""Summarize the matched Qwen contiguous segment-mean routing study."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "routing"
STAGE_FILE = "qwen_routing_segment_mean.json"
BASELINE_FILES = (
    "qwen_routing_confirmation.json",
    "qwen_routing_confirmation_seed20260812.json",
)
MULTI_FILES = (
    "qwen_routing_segment_mean_confirmation.json",
    "qwen_routing_segment_mean_confirmation_seed20260812.json",
)
GATE = {
    "combined_recall_at_3_min": 0.70,
    "per_dataset_recall_at_3_min": 0.60,
    "combined_mrr_min": 0.50,
    "selected_fraction_max": 0.10,
    "absolute_score_position_correlation_max": 0.15,
}


def _read(filename: str) -> dict:
    return json.loads((RESULTS / filename).read_text(encoding="utf-8"))


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _wilson(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / count + z * z / (4 * count * count)
    ) / denominator
    return [center - radius, center + radius]


def _group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _aggregate(rows: list[dict], *keys: str) -> list[dict]:
    metrics = (
        "any_evidence_recall",
        "all_evidence_recall",
        "winning_gist_evidence_recall",
        "mrr",
        "selected_fraction",
        "materialized_fraction",
        "score_position_correlation",
        "routing_gist_bytes",
        "detail_kv_bytes",
        "extra_routing_cache_fraction",
        "packed_index_build_seconds",
        "warm_routing_topk_seconds",
        "topk_primitive_seconds",
        "materialized_tokens",
        "active_kv_bytes",
    )
    output = []
    for group_key, values in sorted(_group(rows, *keys).items()):
        record = dict(zip(keys, group_key))
        record["evaluations"] = len(values)
        for metric in metrics:
            record[metric] = _mean(values, metric)
        successes = sum(int(row["any_evidence_recall"]) for row in values)
        record["any_evidence_recall_ci95"] = _wilson(successes, len(values))
        output.append(record)
    return output


def _normalize_baseline(rows: list[dict]) -> list[dict]:
    normalized = []
    for source in rows:
        row = dict(source)
        row["gist_mode"] = "segment_mean"
        row["gist_count"] = 1
        # One segment covers the complete parent chunk, so these events coincide.
        row["winning_gist_evidence_recall"] = row["any_evidence_recall"]
        normalized.append(row)
    return normalized


def _paired_effects(rows: list[dict], top_k: int = 3) -> list[dict]:
    selected = [row for row in rows if int(row["top_k"]) == top_k]
    key_fields = ("seed", "dataset", "example_id")
    by_gist = {
        gist_count: {
            tuple(row[field] for field in key_fields): row
            for row in selected
            if int(row["gist_count"]) == gist_count
        }
        for gist_count in (1, 2, 4, 8)
    }
    baseline = by_gist[1]
    effects = []
    for gist_count in (2, 4, 8):
        pairs = [(baseline[key], by_gist[gist_count][key]) for key in sorted(baseline)]
        recall_deltas = [
            float(candidate["any_evidence_recall"])
            - float(reference["any_evidence_recall"])
            for reference, candidate in pairs
        ]
        mrr_deltas = [
            float(candidate["mrr"]) - float(reference["mrr"])
            for reference, candidate in pairs
        ]
        gains = sum(delta > 0 for delta in recall_deltas)
        losses = sum(delta < 0 for delta in recall_deltas)
        discordant = gains + losses
        tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
        exact_p = min(1.0, 2 * tail / (2**discordant)) if discordant else 1.0
        effects.append(
            {
                "gist_count": gist_count,
                "pairs": len(pairs),
                "recall_at_3_mean_delta": statistics.fmean(recall_deltas),
                "recall_at_3_gains": gains,
                "recall_at_3_losses": losses,
                "recall_at_3_ties": len(pairs) - discordant,
                "recall_at_3_exact_mcnemar_p": exact_p,
                "mrr_mean_delta": statistics.fmean(mrr_deltas),
                "mrr_improved": sum(delta > 0 for delta in mrr_deltas),
                "mrr_worsened": sum(delta < 0 for delta in mrr_deltas),
                "mrr_tied": sum(delta == 0 for delta in mrr_deltas),
            }
        )
    return effects


def _gate_checks(combined: list[dict], by_dataset: list[dict]) -> list[dict]:
    checks = []
    for gist_count in (1, 2, 4, 8):
        row = next(
            value
            for value in combined
            if int(value["gist_count"]) == gist_count and int(value["top_k"]) == 3
        )
        dataset_rows = [
            value
            for value in by_dataset
            if int(value["gist_count"]) == gist_count and int(value["top_k"]) == 3
        ]
        conditions = {
            "combined_recall_at_3": row["any_evidence_recall"]
            >= GATE["combined_recall_at_3_min"],
            "per_dataset_recall_at_3": all(
                value["any_evidence_recall"] >= GATE["per_dataset_recall_at_3_min"]
                for value in dataset_rows
            ),
            "combined_mrr": row["mrr"] >= GATE["combined_mrr_min"],
            "selected_fraction": row["selected_fraction"] <= GATE["selected_fraction_max"],
            "score_position_correlation": abs(row["score_position_correlation"])
            <= GATE["absolute_score_position_correlation_max"],
        }
        checks.append(
            {
                "gist_count": gist_count,
                "conditions": conditions,
                "passed": all(conditions.values()),
            }
        )
    return checks


def _write_csv(path: Path, rows: list[dict]) -> None:
    scalar_keys = sorted(
        {key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))}
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_keys} for row in rows)


def _plot_sparse_quality(combined: list[dict]) -> None:
    rows = sorted(
        (row for row in combined if int(row["top_k"]) == 3),
        key=lambda row: int(row["gist_count"]),
    )
    counts = [int(row["gist_count"]) for row in rows]
    recall = [row["any_evidence_recall"] for row in rows]
    mrr = [row["mrr"] for row in rows]
    lower = [value - row["any_evidence_recall_ci95"][0] for value, row in zip(recall, rows)]
    upper = [row["any_evidence_recall_ci95"][1] - value for value, row in zip(recall, rows)]

    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.errorbar(
        counts,
        recall,
        yerr=[lower, upper],
        marker="o",
        linewidth=2,
        capsize=4,
        label="Recall@3 (95% Wilson interval)",
    )
    axis.plot(counts, mrr, marker="s", linewidth=2, label="MRR")
    axis.set_xticks(counts)
    axis.set_xlabel("Contiguous mean gists per 32-token parent chunk")
    axis.set_ylabel("Evidence-ranking quality")
    axis.set_ylim(0, 0.6)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(RESULTS / f"qwen_routing_segment_mean_sparse_quality.{suffix}", dpi=180)
    plt.close(figure)


def _plot_dataset_recall(by_dataset: list[dict]) -> None:
    counts = (1, 2, 4, 8)
    datasets = ("hotpotqa", "qasper")
    lookup = {
        (row["dataset"], int(row["gist_count"])): row
        for row in by_dataset
        if int(row["top_k"]) == 3
    }
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    width = 0.36
    centers = list(range(len(counts)))
    for offset, dataset, color in zip((-width / 2, width / 2), datasets, ("#4c72b0", "#c44e52")):
        axis.bar(
            [center + offset for center in centers],
            [lookup[(dataset, count)]["any_evidence_recall"] for count in counts],
            width=width,
            label=dataset,
            color=color,
        )
    axis.set_xticks(centers, counts)
    axis.set_xlabel("Contiguous mean gists per 32-token parent chunk")
    axis.set_ylabel("Any-evidence recall@3")
    axis.set_ylim(0, 0.72)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(RESULTS / f"qwen_routing_segment_mean_dataset_recall_at3.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    stage = _read(STAGE_FILE)
    baseline_artifacts = [_read(filename) for filename in BASELINE_FILES]
    multi_artifacts = [_read(filename) for filename in MULTI_FILES]
    baseline_rows = _normalize_baseline(
        [row for artifact in baseline_artifacts for row in artifact["rows"]]
    )
    multi_rows = [row for artifact in multi_artifacts for row in artifact["rows"]]
    confirmation_rows = [*baseline_rows, *multi_rows]

    combined = _aggregate(confirmation_rows, "gist_count", "top_k")
    by_dataset = _aggregate(confirmation_rows, "dataset", "gist_count", "top_k")
    stage_combined = _aggregate(stage["rows"], "gist_count", "top_k")
    paired = _paired_effects(confirmation_rows)
    gates = _gate_checks(combined, by_dataset)
    unique_examples = {
        (row["dataset"], row["example_id"]) for row in confirmation_rows
    }
    summary = {
        "artifact_git_sha": multi_artifacts[-1]["runtime"]["git_sha"],
        "model_id": stage["model_id"],
        "model_revision": stage["model_revision"],
        "protocol": (
            "frozen Qwen hidden-state routing; 32-token parents; contiguous means; "
            "max gist aggregation; unchanged post-RoPE native K/V"
        ),
        "stage_evaluations_per_gist_count": len(stage["rows"]) // (4 * 3),
        "confirmation_evaluations_per_gist_count": len(baseline_rows) // 3,
        "confirmation_unique_examples": len(unique_examples),
        "baseline_equivalence": (
            "segment_mean G=1 is exact to legacy mean by unit test; existing two-seed "
            "hidden-state confirmation supplies its matched baseline"
        ),
        "stage": stage_combined,
        "confirmation": combined,
        "confirmation_by_dataset": by_dataset,
        "paired_effects_at_3": paired,
        "promotion_gate": GATE,
        "promotion_gate_checks": gates,
        "expected_vs_observed": {
            "H5": {
                "expected": "multiple segment means improve recall@3 and MRR",
                "observed": "rejected; every multi-gist count reduced both sparse metrics",
                "matches": "no",
            },
            "H6": {
                "expected": "quality gains saturate before eight gists",
                "observed": "not applicable because no sparse gain occurred; G=8 helped only at k=16",
                "matches": "no",
            },
            "H7": {
                "expected": "index cost grows while native-K/V materialization remains fixed",
                "observed": "supported for bytes and materialization; warm routing latency was nearly flat at this scale",
                "matches": "partial",
            },
        },
        "decision": (
            "Retain one hidden-state mean as the zero-parameter baseline and proceed to a small "
            "evidence-supervised router with frozen Qwen."
        ),
    }
    (RESULTS / "qwen_routing_segment_mean_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(RESULTS / "qwen_routing_segment_mean_confirmation_combined.csv", combined)
    _write_csv(RESULTS / "qwen_routing_segment_mean_confirmation_by_dataset.csv", by_dataset)
    _write_csv(RESULTS / "qwen_routing_segment_mean_paired_effects.csv", paired)
    _plot_sparse_quality(combined)
    _plot_dataset_recall(by_dataset)


if __name__ == "__main__":
    main()
