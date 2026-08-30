"""Summarize matched live-engine HOT/WARM/COLD lifecycle probes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def _engine_row(payload: dict[str, object]) -> dict[str, object]:
    rows = list(payload["rows"])
    summary = dict(payload["summary"])
    hot = [float(row["lifecycle_request_latency_ms"]["hot"]) for row in rows]
    warm = [float(row["lifecycle_request_latency_ms"]["warm"]) for row in rows]
    cold = [float(row["lifecycle_request_latency_ms"]["cold"]) for row in rows]
    warm_transition = [
        float(row["background_transition_latency_ms"]["warm"]) for row in rows
    ]
    cold_transition = [
        float(row["background_transition_latency_ms"]["cold"]) for row in rows
    ]
    return {
        "engine": payload["engine"],
        "model_id": payload["model_id"],
        "examples": len(rows),
        "hot_warm_exact_rate": summary["hot_warm_exact"] / max(len(rows), 1),
        "cold_int8_exact_rate": summary["hot_cold_int8_exact"] / max(len(rows), 1),
        "cold_int8_first_token_rate": summary["hot_cold_first_token_equal"]
        / max(len(rows), 1),
        "cold_int8_common_prefix_tokens": summary[
            "mean_hot_cold_common_prefix_tokens"
        ],
        "cold_int8_f1_delta": summary["mean_cold_int8_f1_delta"],
        "hot_lifecycle_median_ms": statistics.median(hot),
        "warm_lifecycle_median_ms": statistics.median(warm),
        "cold_lifecycle_median_ms": statistics.median(cold),
        "warm_transition_median_ms": statistics.median(warm_transition),
        "cold_transition_median_ms": statistics.median(cold_transition),
        "warm_over_hot_latency": statistics.median(warm)
        / max(statistics.median(hot), 1e-9),
        "cold_over_hot_latency": statistics.median(cold)
        / max(statistics.median(hot), 1e-9),
        "restart_recovered": summary["restart_recovered"],
        "promotions": summary["metrics"]["promotions"],
        "demotions": summary["metrics"]["demotions"],
        "bytes_read": summary["metrics"]["bytes_read"],
        "bytes_written": summary["metrics"]["bytes_written"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = [_engine_row(json.loads(path.read_text(encoding="utf-8"))) for path in args.inputs]
    rows.sort(key=lambda row: str(row["engine"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "live_storage_lifecycle_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "cross_engine_live_storage_lifecycle_v1",
                "rows": rows,
                "scope": (
                    "Three-example natural-QA mechanism probe; quantized COLD "
                    "is diagnostic and not a production quality certification."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "live_storage_lifecycle_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Engine & WARM exact & int8 exact & int8 first & Prefix & WARM/HOT & COLD/HOT \\\\",
        "\\midrule",
    ]
    for row in rows:
        table.append(
            f"{row['engine']} & {100 * row['hot_warm_exact_rate']:.0f}\\% & "
            f"{100 * row['cold_int8_exact_rate']:.0f}\\% & "
            f"{100 * row['cold_int8_first_token_rate']:.0f}\\% & "
            f"{row['cold_int8_common_prefix_tokens']:.1f} & "
            f"{row['warm_over_hot_latency']:.2f} & "
            f"{row['cold_over_hot_latency']:.2f} \\\\"
        )
    table.extend(("\\bottomrule", "\\end{tabular}"))
    (args.output_dir / "generated_live_storage_lifecycle_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    labels = [str(row["engine"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].bar(
        labels,
        [float(row["hot_warm_exact_rate"]) for row in rows],
        color="#167d78",
        label="WARM lossless",
    )
    axes[0].bar(
        labels,
        [float(row["cold_int8_exact_rate"]) for row in rows],
        color="#d97732",
        width=0.45,
        label="COLD int8",
    )
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Exact output rate")
    axes[0].legend(frameon=False)
    width = 0.34
    positions = list(range(len(rows)))
    axes[1].bar(
        [value - width / 2 for value in positions],
        [float(row["warm_over_hot_latency"]) for row in rows],
        width,
        color="#2878b5",
        label="WARM/HOT",
    )
    axes[1].bar(
        [value + width / 2 for value in positions],
        [float(row["cold_over_hot_latency"]) for row in rows],
        width,
        color="#8a5a9b",
        label="COLD/HOT",
    )
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Median completion ratio")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "live_storage_lifecycle.png", dpi=180)
    figure.savefig(args.output_dir / "live_storage_lifecycle.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
