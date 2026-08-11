"""Merge staged Paper 2 routing artifacts into paper-ready tables and plots."""

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
STAGE_FILES = {
    "representation": "qwen_routing_representation.json",
    "chunk_size": "qwen_routing_chunk_size.json",
    "confirmation_seed_20260811": "qwen_routing_confirmation.json",
    "confirmation_seed_20260812": "qwen_routing_confirmation_seed20260812.json",
}
GATE = {
    "combined_recall_at_3_min": 0.70,
    "per_dataset_recall_at_3_min": 0.60,
    "combined_mrr_min": 0.50,
    "selected_fraction_max": 0.10,
    "absolute_score_position_correlation_max": 0.15,
}


def _load() -> dict[str, dict]:
    return {
        name: json.loads((RESULTS / filename).read_text(encoding="utf-8"))
        for name, filename in STAGE_FILES.items()
    }


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
        "mrr",
        "target_coverage",
        "selected_fraction",
        "materialized_fraction",
        "score_position_correlation",
        "mean_selected_normalized_position",
        "materialized_tokens",
        "active_kv_bytes",
        "extra_routing_cache_fraction",
        "packed_index_build_seconds",
        "warm_routing_topk_seconds",
        "topk_primitive_seconds",
    )
    output = []
    for group_key, values in sorted(_group(rows, *keys).items()):
        record = dict(zip(keys, group_key))
        record["examples"] = len(values)
        for metric in metrics:
            record[metric] = _mean(values, metric)
        successes = sum(int(row["any_evidence_recall"]) for row in values)
        record["any_evidence_recall_ci95"] = _wilson(successes, len(values))
        output.append(record)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row if not isinstance(row[key], list)})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)


def _json_csv(stem: str, rows: list[dict]) -> None:
    (RESULTS / f"{stem}.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(RESULTS / f"{stem}.csv", rows)


def _plot_recall(rows: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    colors = {
        "post_rope_key": "#c44e52",
        "pre_rope_key": "#4c72b0",
        "hidden_state": "#55a868",
    }
    for representation, values in sorted(_group(rows, "routing_representation").items()):
        values = sorted(values, key=lambda row: row["selected_fraction"])
        axis.plot(
            [row["selected_fraction"] for row in values],
            [row["any_evidence_recall"] for row in values],
            marker="o",
            linewidth=2,
            color=colors[representation[0]],
            label=representation[0].replace("_", " "),
        )
    axis.set_xlabel("Selected chunk fraction")
    axis.set_ylabel("Any-evidence recall")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(RESULTS / f"qwen_routing_recall_vs_selected_fraction.{suffix}", dpi=180)
    plt.close(figure)


def _plot_position(rows: list[dict]) -> None:
    values = sorted(rows, key=lambda row: row["routing_representation"])
    labels = [row["routing_representation"].replace("_", "\n") for row in values]
    correlations = [row["score_position_correlation"] for row in values]
    colors = ["#55a868", "#c44e52", "#4c72b0"]
    figure, axis = plt.subplots(figsize=(6.6, 4.0))
    axis.bar(labels, correlations, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Mean score-position correlation")
    axis.set_ylim(-0.2, 0.75)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(RESULTS / f"qwen_routing_position_bias.{suffix}", dpi=180)
    plt.close(figure)


def main() -> None:
    artifacts = _load()
    representation_rows = artifacts["representation"]["rows"]
    chunk_rows = artifacts["chunk_size"]["rows"]
    confirmation_rows = [
        *artifacts["confirmation_seed_20260811"]["rows"],
        *artifacts["confirmation_seed_20260812"]["rows"],
    ]

    representation = _aggregate(
        representation_rows, "routing_representation", "routing_chunk_size", "top_k"
    )
    chunk_size = _aggregate(
        chunk_rows, "routing_representation", "routing_chunk_size", "top_k"
    )
    confirmation = _aggregate(confirmation_rows, "routing_representation", "top_k")
    confirmation_by_dataset = _aggregate(
        confirmation_rows, "dataset", "routing_representation", "top_k"
    )
    position_bias = _aggregate(
        [row for row in representation_rows if row["top_k"] == 3],
        "routing_representation",
    )
    _json_csv("qwen_routing_topk", representation)
    _json_csv("qwen_routing_position_bias", position_bias)

    k3 = next(row for row in confirmation if row["top_k"] == 3)
    dataset_k3 = [row for row in confirmation_by_dataset if row["top_k"] == 3]
    unique_examples = {
        (row["dataset"], row["example_id"]) for row in confirmation_rows
    }
    gate_checks = {
        "combined_recall_at_3": k3["any_evidence_recall"] >= GATE["combined_recall_at_3_min"],
        "per_dataset_recall_at_3": all(
            row["any_evidence_recall"] >= GATE["per_dataset_recall_at_3_min"]
            for row in dataset_k3
        ),
        "combined_mrr": k3["mrr"] >= GATE["combined_mrr_min"],
        "selected_fraction": k3["selected_fraction"] <= GATE["selected_fraction_max"],
        "score_position_correlation": abs(k3["score_position_correlation"])
        <= GATE["absolute_score_position_correlation_max"],
    }
    summary = {
        "artifact_git_sha": artifacts["confirmation_seed_20260812"]["runtime"]["git_sha"],
        "model_id": artifacts["representation"]["model_id"],
        "model_revision": artifacts["representation"]["model_revision"],
        "stage1_examples": len({(r["dataset"], r["example_id"]) for r in representation_rows}),
        "confirmation_evaluations": len(confirmation_rows) // 3,
        "confirmation_unique_examples": len(unique_examples),
        "representation_comparison": representation,
        "chunk_size_comparison": chunk_size,
        "confirmation": confirmation,
        "confirmation_by_dataset": confirmation_by_dataset,
        "promotion_gate": GATE,
        "promotion_gate_checks": gate_checks,
        "promotion_gate_passed": all(gate_checks.values()),
        "expected_vs_observed": {
            "H1": {
                "expected": "pre-RoPE routing beats post-RoPE and reduces position bias",
                "observed": "supported; pre-RoPE improved sparse recall and removed the strong late-position correlation",
                "matches": "yes",
            },
            "H2": {
                "expected": "post-RoPE degrades more than position-neutral routing as chunks grow",
                "observed": "not directly tested after post-RoPE failed Stage 1; finalist recall rose with chunk size but selected fraction rose sharply",
                "matches": "not evaluated",
            },
            "H3": {
                "expected": "larger k raises recall at greater selection cost",
                "observed": "supported across every representation and chunk size",
                "matches": "yes",
            },
            "H4": {
                "expected": "hidden-state routing may be competitive",
                "observed": "supported; hidden-state routing won the low-k tradeoff and the independent confirmation",
                "matches": "yes",
            },
        },
        "decision": "Do not proceed to Llama; test a small evidence-supervised router with frozen Qwen.",
    }
    (RESULTS / "qwen_routing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot_recall(representation)
    _plot_position(position_bias)


if __name__ == "__main__":
    main()
