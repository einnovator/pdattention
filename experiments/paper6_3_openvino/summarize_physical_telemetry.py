"""Render publication artifacts for OpenVINO physical telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


LABELS = {"selected_text_e0": "selected E0", "full_context_e0": "FULL E0"}


def summarize(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for source in payload["aggregates"]:
        rows.append(
            {
                "condition": source["condition"],
                "label": LABELS[source["condition"]],
                "samples": source["samples"],
                "source_tokens": source["mean_source_tokens"],
                "token_f1": source["mean_token_f1"],
                "ttft_p50_ms": source["ttft_p50_ms"],
                "ttft_p95_ms": source["ttft_p95_ms"],
                "completion_p50_ms": source["completion_p50_ms"],
                "completion_p95_ms": source["completion_p95_ms"],
                "rss_peak_mib": source["rss_peak_bytes"] / 2**20,
                "gpu_memory_peak_mib": source["plugin_numeric_peaks"][
                    "GPU_MEMORY_STATISTICS"
                ]
                / 2**20,
                "process_cpu_seconds": source["mean_process_cpu_seconds"],
                "process_read_kib": source["mean_process_read_bytes"] / 2**10,
                "process_write_kib": source["mean_process_write_bytes"] / 2**10,
            }
        )
    selected = next(row for row in rows if row["condition"] == "selected_text_e0")
    full = next(row for row in rows if row["condition"] == "full_context_e0")
    return {
        "schema_version": "paper6.3-openvino-physical-summary-v1",
        "source_schema_version": payload["schema_version"],
        "evidence_tier": payload["evidence_tier"],
        "model_id": payload["model_id"],
        "device": payload["device"],
        "plugin_physical_properties": payload["plugin_physical_properties"],
        "energy_status": payload["energy_status"],
        "gpu_utilization_status": payload["gpu_utilization_status"],
        "rows": rows,
        "full_over_selected": {
            key: full[key] / selected[key]
            for key in (
                "source_tokens",
                "ttft_p50_ms",
                "ttft_p95_ms",
                "completion_p50_ms",
                "completion_p95_ms",
                "rss_peak_mib",
                "gpu_memory_peak_mib",
                "process_cpu_seconds",
                "process_read_kib",
                "process_write_kib",
            )
        },
        "selected_minus_full_f1": selected["token_f1"] - full["token_f1"],
    }


def _table(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Condition & source & F1 & \multicolumn{2}{c}{TTFT (ms)} & GPU MiB & RSS MiB & CPU s & read KiB \\",
        r" & tokens & & p50 & p95 & peak & peak & mean & mean \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        lines.append(
            "{} & {:.1f} & {:.3f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.2f} & {:.1f} \\\\".format(
                row["label"],
                row["source_tokens"],
                row["token_f1"],
                row["ttft_p50_ms"],
                row["ttft_p95_ms"],
                row["gpu_memory_peak_mib"],
                row["rss_peak_mib"],
                row["process_cpu_seconds"],
                row["process_read_kib"],
            )
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(summary: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    ratios = summary["full_over_selected"]
    labels = ("source tokens", "TTFT p50", "TTFT p95", "GPU memory", "CPU time")
    values = (
        ratios["source_tokens"],
        ratios["ttft_p50_ms"],
        ratios["ttft_p95_ms"],
        ratios["gpu_memory_peak_mib"],
        ratios["process_cpu_seconds"],
    )
    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    bars = axis.bar(labels, values, color=("#4c78a8", "#f28e2b", "#e15759", "#59a14f", "#b07aa1"))
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_ylabel("FULL / selected E0")
    axis.set_title("Intel physical and latency cost")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.2f}x", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = summarize(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "physical_telemetry_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _table(summary, args.output_dir / "generated_physical_telemetry_table.tex")
    _plot(summary, args.output_dir / "physical_telemetry.png")


if __name__ == "__main__":
    main()
