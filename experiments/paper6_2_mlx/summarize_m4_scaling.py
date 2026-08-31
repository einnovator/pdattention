"""Summarize the M4 Pro cross-model MLX oracle-evidence experiment."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from typing import Iterable


CONDITIONS = (
    "ordinary_split",
    "native_fp",
    "native_int8_resident",
    "native_shuffled",
    "no_memory",
)
DATASET_LABELS = {
    "qasper": "QASPER",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2Wiki",
}


def summarize(payloads: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate each model, dataset, and execution condition across seeds."""

    result: list[dict[str, object]] = []
    for payload in payloads:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in payload["rows"]:  # type: ignore[index]
            grouped[str(row["condition"])].append(row)
        for condition in CONDITIONS:
            rows = grouped[condition]
            if not rows:
                continue
            result.append(
                {
                    "model_id": payload["model_id"],
                    "dataset": payload["dataset"],
                    "condition": condition,
                    "sample_count": len(rows),
                    "seed_count": len({int(row["seed"]) for row in rows}),
                    "token_f1": fmean(float(row["token_f1"]) for row in rows),
                    "gold_answer_logprob": fmean(
                        float(row["gold_answer_logprob"]) for row in rows
                    ),
                    "completion_latency_ms": fmean(
                        float(row["completion_latency_ms"]) for row in rows
                    ),
                    "resident_selected_kv_bytes": fmean(
                        float(row["resident_selected_kv_bytes"]) for row in rows
                    ),
                }
            )
    return result


def comparison_rows(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    """Collapse full condition summaries into one publication row per cohort."""

    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in summary:
        key = (str(row["model_id"]), str(row["dataset"]))
        grouped[key][str(row["condition"])] = row

    comparisons = []
    for (model_id, dataset), conditions in sorted(grouped.items()):
        ordinary = conditions["ordinary_split"]
        native = conditions["native_fp"]
        int8 = conditions["native_int8_resident"]
        shuffled = conditions["native_shuffled"]
        no_memory = conditions["no_memory"]
        comparisons.append(
            {
                "model_id": model_id,
                "dataset": dataset,
                "sample_count": ordinary["sample_count"],
                "ordinary_f1": ordinary["token_f1"],
                "native_f1": native["token_f1"],
                "int8_f1": int8["token_f1"],
                "shuffled_f1": shuffled["token_f1"],
                "no_memory_f1": no_memory["token_f1"],
                "ordinary_latency_ms": ordinary["completion_latency_ms"],
                "native_latency_ms": native["completion_latency_ms"],
                "native_over_ordinary": float(native["completion_latency_ms"])
                / float(ordinary["completion_latency_ms"]),
                "native_resident_mib": float(native["resident_selected_kv_bytes"])
                / 1048576,
                "int8_resident_mib": float(int8["resident_selected_kv_bytes"])
                / 1048576,
                "gold_logprob_delta_shuffled": float(
                    shuffled["gold_answer_logprob"]
                )
                - float(native["gold_answer_logprob"]),
            }
        )
    return comparisons


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the compact cross-model table consumed by Paper 6.2."""

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Dataset & $n$ & E0 F1 & E2 F1 & Shuf. F1 & E2/E0 & FP/int8 MiB \\",
        r"\midrule",
    ]
    for row in rows:
        model = "Qwen3-4B" if "4B" in str(row["model_id"]) else "Qwen3-1.7B"
        lines.append(
            f"{model} & {DATASET_LABELS[str(row['dataset'])]} & "
            f"{int(row['sample_count'])} & {float(row['ordinary_f1']):.3f} & "
            f"{float(row['native_f1']):.3f} & {float(row['shuffled_f1']):.3f} & "
            f"{float(row['native_over_ordinary']):.3f} & "
            f"{float(row['native_resident_mib']):.1f}/{float(row['int8_resident_mib']):.1f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot transport quality controls and native completion-time ratios."""

    import matplotlib.pyplot as plt
    import numpy as np

    labels = [
        ("4B" if "4B" in str(row["model_id"]) else "1.7B")
        + "\n"
        + DATASET_LABELS[str(row["dataset"])]
        for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.23
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    axes[0].bar(x - width, [row["ordinary_f1"] for row in rows], width, label="selected E0")
    axes[0].bar(x, [row["native_f1"] for row in rows], width, label="native E2")
    axes[0].bar(x + width, [row["shuffled_f1"] for row in rows], width, label="shuffled")
    axes[0].set_ylabel("Token F1")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    ratios = [float(row["native_over_ordinary"]) for row in rows]
    axes[1].bar(x, ratios, color="#2a9d8f")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Native E2 / selected E0 time")
    axes[1].set_xticks(x, labels)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    full_summary = summarize(payloads)
    rows = comparison_rows(full_summary)
    result = {
        "schema_version": "paper6-2-mlx-m4-cross-model-v1",
        "evidence_tier": "NATURAL_QA_ORACLE_EVIDENCE_MATERIALIZATION",
        "hardware": "Apple M4 Pro, 20-core GPU, 48 GiB unified memory",
        "conditions": full_summary,
        "comparisons": rows,
    }
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_table(args.table, rows)
    write_plot(args.plot, rows)


if __name__ == "__main__":
    main()
