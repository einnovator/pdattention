"""Build publication artifacts for corrected PRA layer-placement experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt


CONTIGUOUS = (
    "all",
    "last_24",
    "last_20",
    "last_16",
    "last_14",
    "last_12",
    "last_8",
    "last_4",
    "last_1",
)
SPARSE = ("early_4", "middle_4", "even_4", "last_4", "even_8", "last_8")
METRICS = (
    "gold_mean_logprob_delta_vs_none",
    "gold_first_token_rank",
    "evidence_attention_mass",
    "memory_attention_mass",
    "residual_divergence_mean",
    "residual_divergence_final",
    "native_kv_token_states",
    "native_kv_bytes",
    "materialized_unique_tokens",
    "teacher_forced_seconds",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _bootstrap_ci(values: list[float], seed: int, samples: int = 5_000) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def _aggregate(rows: list[dict[str, str]], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)

    output: list[dict] = []
    for values, members in sorted(groups.items()):
        item = dict(zip(keys, values))
        item["n"] = len(members)
        for metric in METRICS:
            samples = [value for row in members if (value := _number(row, metric)) is not None]
            if samples:
                item[f"{metric}_mean"] = mean(samples)
                item[f"{metric}_sd"] = pstdev(samples)
                if metric == "gold_mean_logprob_delta_vs_none":
                    seed = sum(ord(character) for value in values for character in value)
                    low, high = _bootstrap_ci(samples, seed)
                    item[f"{metric}_ci_low"] = low
                    item[f"{metric}_ci_high"] = high
        item["consumer_layer_count"] = int(float(members[0]["consumer_layer_count"]))
        item["consumer_layer_fraction"] = float(members[0]["consumer_layer_fraction"])
        output.append(item)
    return output


def _transport_audit(root: Path) -> None:
    rows: list[dict] = []
    raw_by_mode: dict[str, list[dict[str, str]]] = {}
    for mode in ("original", "corrected"):
        raw = _read_csv(root / f"{mode}_transport_audit.csv")
        raw_by_mode[mode] = raw
        grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in raw:
            grouped[(row["dataset"], row["condition"])].append(row)
        for (dataset, condition), members in sorted(grouped.items()):
            rows.append(
                {
                    "position_mode": mode,
                    "dataset": dataset,
                    "condition": condition,
                    "n": len(members),
                    "gold_mean_logprob_delta_vs_none_mean": mean(
                        float(row["gold_mean_logprob_delta_vs_none"]) for row in members
                    ),
                    "gold_first_token_rank_mean": mean(
                        float(row["gold_first_token_rank"]) for row in members
                    ),
                    "query_position_offset_mean": mean(
                        float(row["query_position_offset"]) for row in members
                    ),
                }
            )
    _write_csv(root / "layer_reconciliation_original_vs_fixed.csv", rows)

    audit = {
        "scope": "diagnostic rerun; two identities per dataset",
        "historical_bug": (
            "Direct-query positions restarted at zero while external native K/V retained "
            "source-relative RoPE positions."
        ),
        "corrected_transport": (
            "Query and decode positions begin after the source encoding interval; source "
            "K/V retain their native source-relative positions."
        ),
        "invariants": {
            "source_kv": "copied from the host model without projection",
            "physical_heads": "host GQA/MQA K/V heads are retained",
            "attention": "direct and memory logits share one softmax",
            "deduplication": "physical source positions are unique after interval union",
            "boundaries": "materialized positions are clipped to source bounds",
            "lifetime": "reference detail K/V persists for request prefill and decode",
        },
        "raw_rows": {mode: len(raw) for mode, raw in raw_by_mode.items()},
        "publication_table": "layer_reconciliation_original_vs_fixed.csv",
    }
    (root / "layer_replay_transport_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


def _pareto(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        candidates = [row for row in rows if row["dataset"] == dataset]
        for row in candidates:
            quality = row["gold_mean_logprob_delta_vs_none_mean"]
            cost = row["native_kv_token_states_mean"]
            dominated = any(
                other["gold_mean_logprob_delta_vs_none_mean"] >= quality
                and other["native_kv_token_states_mean"] <= cost
                and (
                    other["gold_mean_logprob_delta_vs_none_mean"] > quality
                    or other["native_kv_token_states_mean"] < cost
                )
                for other in candidates
            )
            output.append({**row, "pareto": not dominated})
    return output


def _recommendations(contiguous: list[dict]) -> dict:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in contiguous:
        by_dataset[row["dataset"]].append(row)

    per_dataset = {}
    for dataset, rows in sorted(by_dataset.items()):
        best = max(rows, key=lambda row: row["gold_mean_logprob_delta_vs_none_mean"])
        positive = [row for row in rows if row["gold_mean_logprob_delta_vs_none_mean"] > 0]
        economy = min(positive, key=lambda row: row["native_kv_token_states_mean"])
        balanced_pool = [
            row
            for row in positive
            if row["gold_mean_logprob_delta_vs_none_mean"]
            >= 0.70 * best["gold_mean_logprob_delta_vs_none_mean"]
        ]
        balanced = min(balanced_pool, key=lambda row: row["native_kv_token_states_mean"])
        per_dataset[dataset] = {
            "reference_correctness": "all (conservative fallback, not quality-optimal)",
            "quality_max": best["profile"],
            "balanced": balanced["profile"],
            "economy": economy["profile"],
            "quality_max_delta_nats": best["gold_mean_logprob_delta_vs_none_mean"],
            "balanced_delta_nats": balanced["gold_mean_logprob_delta_vs_none_mean"],
            "economy_delta_nats": economy["gold_mean_logprob_delta_vs_none_mean"],
        }

    return {
        "status": "held-out calibration, not a universal model default",
        "transport": "corrected source-offset native-K/V replay",
        "selection_rule": {
            "quality_max": "highest held-out mean gold-answer log-probability delta",
            "balanced": "lowest native-K/V token-state cost retaining at least 70% of the best positive delta",
            "economy": "lowest-cost profile with a positive held-out mean delta",
            "reference_correctness": "all eligible consumer layers until sparse correctness is calibrated",
        },
        "per_dataset": per_dataset,
        "shared_recommendation": {
            "quality_max": "last_20",
            "balanced": "last_8",
            "economy": "last_1",
            "warning": (
                "last_1 has near-zero average gain and is an economy probe, not a quality profile; "
                "early and evenly scattered consumers are contraindicated on both workloads."
            ),
        },
    }


def _plots(root: Path, contiguous: list[dict], sparse: list[dict], materialization: list[dict]) -> None:
    colors = {"2wikimultihopqa": "#167D8D", "musique": "#C6533D"}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for dataset in sorted(colors):
        rows = [row for row in contiguous if row["dataset"] == dataset]
        axes[0].plot(
            [row["consumer_layer_count"] for row in rows],
            [row["gold_mean_logprob_delta_vs_none_mean"] for row in rows],
            marker="o",
            label=dataset,
            color=colors[dataset],
        )
        axes[1].plot(
            [row["native_kv_token_states_mean"] for row in rows],
            [row["gold_mean_logprob_delta_vs_none_mean"] for row in rows],
            marker="o",
            label=dataset,
            color=colors[dataset],
        )
    for axis in axes:
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.set_ylabel("Gold-answer log-probability delta (nats)")
    axes[0].set_xlabel("Late contiguous consumer layers")
    axes[1].set_xlabel("Materialized native K/V token-states")
    axes[0].legend(frameon=False)
    fig.savefig(root / "layer_contiguous_quality_cost.pdf")
    fig.savefig(root / "layer_contiguous_quality_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    names = list(SPARSE)
    width = 0.36
    for index, dataset in enumerate(sorted(colors)):
        values = {
            row["profile"]: row["gold_mean_logprob_delta_vs_none_mean"]
            for row in sparse
            if row["dataset"] == dataset
        }
        positions = [i + (index - 0.5) * width for i in range(len(names))]
        ax.bar(positions, [values[name] for name in names], width, label=dataset, color=colors[dataset])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(range(len(names)), [name.replace("_", " ") for name in names])
    ax.set_ylabel("Gold-answer log-probability delta (nats)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.savefig(root / "layer_sparse_placement.pdf")
    fig.savefig(root / "layer_sparse_placement.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True, sharey=True)
    geometries = ["exact_core", "expanded_window", "full_selected_record", "whole_parent"]
    for axis, dataset in zip(axes, sorted(colors)):
        for profile in ("all", "last_14", "last_8", "even_8"):
            values = {
                row["geometry"]: row["gold_mean_logprob_delta_vs_none_mean"]
                for row in materialization
                if row["dataset"] == dataset and row["profile"] == profile
            }
            axis.plot(range(len(geometries)), [values[g] for g in geometries], marker="o", label=profile)
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_title(dataset)
        axis.set_xticks(range(len(geometries)), [g.replace("_", "\n") for g in geometries])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Gold-answer log-probability delta (nats)")
    axes[1].legend(frameon=False, ncol=2)
    fig.savefig(root / "layer_materialization_cross.pdf")
    fig.savefig(root / "layer_materialization_cross.png", dpi=180)
    plt.close(fig)


def summarize(root: Path) -> None:
    rows = _read_csv(root / "layer_reconciliation_rows.csv")
    aggregate = _aggregate(rows, ("dataset", "profile", "geometry"))
    contiguous = [
        row for row in aggregate if row["geometry"] == "whole_parent" and row["profile"] in CONTIGUOUS
    ]
    contiguous.sort(key=lambda row: (row["dataset"], -row["consumer_layer_count"]))
    sparse = [
        row for row in aggregate if row["geometry"] == "whole_parent" and row["profile"] in SPARSE
    ]
    materialization = [
        row for row in aggregate if row["profile"] in ("all", "last_14", "last_8", "even_8")
    ]

    _transport_audit(root)
    _write_csv(root / "layer_contiguous_sweep.csv", contiguous)
    _write_csv(root / "layer_sparse_vs_contiguous.csv", sparse)
    _write_csv(root / "layer_materialization_cross.csv", materialization)
    _write_csv(root / "layer_workload_cross.csv", aggregate)

    residual_rows: list[dict] = []
    for row in rows:
        divergence = json.loads(row["residual_divergence_by_layer"])
        layer_values = divergence.items() if isinstance(divergence, dict) else enumerate(divergence)
        for layer, value in layer_values:
            residual_rows.append(
                {
                    "dataset": row["dataset"],
                    "profile": row["profile"],
                    "geometry": row["geometry"],
                    "example_id": row["example_id"],
                    "layer": int(layer),
                    "residual_divergence": float(value),
                }
            )
    residual_aggregate: list[dict] = []
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in residual_rows:
        grouped[(row["dataset"], row["profile"], row["geometry"], row["layer"])].append(
            row["residual_divergence"]
        )
    for (dataset, profile, geometry, layer), values in sorted(grouped.items()):
        residual_aggregate.append(
            {
                "dataset": dataset,
                "profile": profile,
                "geometry": geometry,
                "layer": layer,
                "n": len(values),
                "residual_divergence_mean": mean(values),
                "residual_divergence_sd": pstdev(values),
            }
        )
    _write_csv(root / "layer_residual_divergence.csv", residual_aggregate)

    pareto = _pareto(contiguous)
    _write_csv(root / "layer_pareto_profiles.csv", pareto)
    recommendations = _recommendations(contiguous)
    (root / "layer_profile_recommendations.json").write_text(
        json.dumps(recommendations, indent=2) + "\n", encoding="utf-8"
    )
    _plots(root, contiguous, sparse, materialization)

    manifest = {
        "source_rows": len(rows),
        "identities": len({(row["dataset"], row["example_id"]) for row in rows}),
        "datasets": sorted({row["dataset"] for row in rows}),
        "position_mode": sorted({row["position_mode"] for row in rows}),
        "profiles": sorted({row["profile"] for row in rows}),
        "geometries": sorted({row["geometry"] for row in rows}),
        "artifacts": sorted(path.name for path in root.glob("layer_*")),
    }
    (root / "layer_reconciliation_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root)


if __name__ == "__main__":
    main()
