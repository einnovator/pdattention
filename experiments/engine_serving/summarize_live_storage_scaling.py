"""Summarize larger live-engine storage cohorts without pooling model identities."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def _percentile(values: Iterable[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a finite sample."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty latency sample.")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _artifact_row(payload: dict[str, object]) -> dict[str, object]:
    """Reduce one engine/model/dataset artifact to disjoint quality and cost fields."""

    rows = list(payload["rows"])
    if not rows:
        raise ValueError(f"Lifecycle artifact for {payload['engine']} has no rows.")
    datasets = sorted({str(row["dataset"]) for row in rows})
    result: dict[str, object] = {
        "engine": payload["engine"],
        "model_id": payload["model_id"],
        "dataset": "+".join(datasets),
        "examples": len(rows),
        "warm_exact_rate": sum(bool(row["hot_warm_exact"]) for row in rows)
        / len(rows),
        "int8_exact_rate": sum(bool(row["hot_cold_int8_exact"]) for row in rows)
        / len(rows),
        "int8_first_token_rate": sum(
            bool(row["hot_cold_first_token_equal"]) for row in rows
        )
        / len(rows),
        "int8_common_prefix_mean": sum(
            int(row["hot_cold_common_prefix_tokens"]) for row in rows
        )
        / len(rows),
        "int8_f1_delta_mean": sum(float(row["cold_int8_f1_delta"]) for row in rows)
        / len(rows),
        "native_bytes_mean": sum(int(row["native_bytes"]) for row in rows)
        / len(rows),
        "restart_recovered": bool(payload["summary"]["restart_recovered"]),
    }
    for family, field in (
        ("request", "lifecycle_request_latency_ms"),
        ("transition", "background_transition_latency_ms"),
    ):
        for tier in ("hot", "warm", "cold"):
            values = [float(row[field][tier]) for row in rows]
            for label, quantile in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
                result[f"{family}_{tier}_{label}_ms"] = _percentile(values, quantile)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        _artifact_row(json.loads(path.read_text(encoding="utf-8")))
        for path in args.inputs
    ]
    rows.sort(key=lambda row: (str(row["engine"]), str(row["model_id"]), str(row["dataset"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "experiment": "cross_engine_live_storage_scaling_v1",
        "scope": (
            "Natural-QA live generation with lossless WARM and diagnostic int8 "
            "COLD. Percentiles describe the checked-in finite cohorts, not a "
            "production request distribution."
        ),
        "rows": rows,
    }
    (args.output_dir / "live_storage_scaling_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "live_storage_scaling_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Engine & Model & Data & $n$ & WARM exact & int8 exact & HOT p50 & WARM p50 & WARM p95 \\",
        r"\midrule",
    ]
    for row in rows:
        model = str(row["model_id"]).split("/")[-1].replace("_", r"\_")
        dataset = str(row["dataset"]).replace("_", r"\_")
        table.append(
            f"{row['engine']} & {model} & {dataset} & {row['examples']} & "
            f"{100 * float(row['warm_exact_rate']):.1f}\\% & "
            f"{100 * float(row['int8_exact_rate']):.1f}\\% & "
            f"{float(row['request_hot_p50_ms']):.0f} & "
            f"{float(row['request_warm_p50_ms']):.0f} & "
            f"{float(row['request_warm_p95_ms']):.0f} \\\\"
        )
    table.extend((r"\bottomrule", r"\end{tabular}"))
    (args.output_dir / "generated_live_storage_scaling_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    labels = [
        f"{row['engine']}\n{str(row['model_id']).split('/')[-1]}\n{row['dataset']}"
        for row in rows
    ]
    positions = list(range(len(rows)))
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(8.4, 1.45 * len(rows)), 4.2))
    for offset, tier, color in (
        (-width, "hot", "#167d78"),
        (0.0, "warm", "#2878b5"),
        (width, "cold", "#d97732"),
    ):
        axis.bar(
            [position + offset for position in positions],
            [float(row[f"request_{tier}_p50_ms"]) for row in rows],
            width,
            color=color,
            label=f"{tier.upper()} p50",
        )
        axis.errorbar(
            [position + offset for position in positions],
            [float(row[f"request_{tier}_p50_ms"]) for row in rows],
            yerr=[
                max(
                    0.0,
                    float(row[f"request_{tier}_p95_ms"])
                    - float(row[f"request_{tier}_p50_ms"]),
                )
                for row in rows
            ],
            fmt="none",
            ecolor="#333333",
            capsize=2,
            linewidth=0.8,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Lifecycle request latency (ms)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "live_storage_scaling.png", dpi=180)
    figure.savefig(args.output_dir / "live_storage_scaling.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
