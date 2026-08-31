"""Reduce vLLM CUDA connector mechanism and natural-QA artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def summarize(
    controlled: Mapping[str, Any], natural: Mapping[str, Any]
) -> dict[str, object]:
    """Combine controlled causality and natural matched-pair evidence."""

    rows = []
    for aggregate in natural["aggregates"]:
        dataset_rows = [
            row for row in natural["rows"] if row["dataset"] == aggregate["dataset"]
        ]
        e0_completion = [float(row["full"]["completion_ms"]) for row in dataset_rows]
        e2_completion = [float(row["native"]["completion_ms"]) for row in dataset_rows]
        e0_ingestion = [
            float(row["e0_ingestion"]["completion_ms"]) for row in dataset_rows
        ]
        e2_ingestion = [
            float(row["ingestion"]["completion_ms"]) for row in dataset_rows
        ]
        cached = [float(row["full"]["cached_prompt_tokens"]) for row in dataset_rows]
        native_bytes = [float(row["stored_native_bytes"]) for row in dataset_rows]
        rows.append(
            {
                **aggregate,
                "e0_completion_p95_ms": _percentile(e0_completion, 0.95),
                "native_completion_p95_ms": _percentile(e2_completion, 0.95),
                "native_over_e0_completion": statistics.median(e2_completion)
                / statistics.median(e0_completion),
                "e0_ingestion_p50_ms": statistics.median(e0_ingestion),
                "native_ingestion_p50_ms": statistics.median(e2_ingestion),
                "mean_apc_cached_tokens": statistics.fmean(cached),
                "mean_native_mib": statistics.fmean(native_bytes) / 1024**2,
            }
        )
    return {
        "schema_version": "paper6-vllm-cuda-connector-summary-v1",
        "engine": natural["engine"],
        "engine_version": natural["engine_version"],
        "model_id": natural["model_id"],
        "device": natural["device"],
        "integration_status": natural["integration_status"],
        "controlled": controlled["summary"],
        "natural_samples": len(natural["rows"]),
        "natural_exact_pairs": sum(
            bool(row["exact_output_parity"]) for row in natural["rows"]
        ),
        "rows": rows,
        "qualification": {
            "native_kv_consumed": True,
            "source_content_visible": False,
            "apc_coexists": bool(natural["apc_enabled"]),
            "source_slots_scheduler_visible": bool(
                natural["source_slots_scheduler_visible"]
            ),
            "full_detached_e2_validated": False,
        },
    }


def write_table(summary: Mapping[str, Any], path: Path) -> None:
    labels = {
        "qasper": "QASPER",
        "hotpotqa": "HotpotQA",
        "2wikimultihopqa": "2Wiki",
    }
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & $n$ & Exact pair & F1 E0/cand. & Completion E0/cand. (ms) & Ratio & Native MiB \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        lines.append(
            f"{labels[str(row['dataset'])]} & {row['samples']} & "
            f"{100 * row['exact_output_parity']:.0f}\\% & "
            f"{row['full_f1']:.3f}/{row['native_f1']:.3f} & "
            f"{row['full_completion_ms']:.1f}/{row['native_completion_ms']:.1f} & "
            f"{row['native_over_e0_completion']:.3f} & "
            f"{row['mean_native_mib']:.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(summary: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(summary["rows"])
    labels = [
        {"qasper": "QASPER", "hotpotqa": "HotpotQA", "2wikimultihopqa": "2Wiki"}[
            str(row["dataset"])
        ]
        for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    axes[0].bar(x - width / 2, [row["full_f1"] for row in rows], width, label="E0 APC")
    axes[0].bar(
        x + width / 2,
        [row["native_f1"] for row in rows],
        width,
        label="native candidate",
    )
    axes[0].set(ylabel="token F1", xticks=x, xticklabels=labels, title="Matched quality")
    axes[1].bar(
        x - width / 2,
        [row["full_completion_ms"] for row in rows],
        width,
        label="E0 APC",
    )
    axes[1].bar(
        x + width / 2,
        [row["native_completion_ms"] for row in rows],
        width,
        label="native candidate",
    )
    axes[1].set(
        ylabel="median completion (ms)",
        xticks=x,
        xticklabels=labels,
        title="Request completion",
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controlled", type=Path)
    parser.add_argument("natural", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    controlled = json.loads(args.controlled.read_text(encoding="utf-8"))
    natural = json.loads(args.natural.read_text(encoding="utf-8"))
    summary = summarize(controlled, natural)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_table(summary, args.table)
    write_plot(summary, args.plot)


if __name__ == "__main__":
    main()
