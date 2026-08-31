"""Summarize Qwen3-4B bounded-residency sessions on the M4 Pro."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean
from typing import Iterable


DATASET_LABELS = {
    "qasper": "QASPER",
    "hotpotqa": "HotpotQA",
    "2wikimultihopqa": "2Wiki",
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def summarize(payloads: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate request and lifecycle metrics by dataset and resident budget."""

    result: list[dict[str, object]] = []
    for payload in payloads:
        by_budget: dict[int, list[dict[str, object]]] = defaultdict(list)
        summaries: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in payload["rows"]:  # type: ignore[index]
            by_budget[int(row["resident_resource_budget"])].append(row)
        for row in payload["seed_summaries"]:  # type: ignore[index]
            summaries[int(row["resident_resource_budget"])].append(row)
        for budget, rows in sorted(by_budget.items()):
            lifecycle = summaries[budget]
            resolve = [float(row["resolve_ms"]) for row in rows]
            completion = [float(row["completion_latency_ms"]) for row in rows]
            result.append(
                {
                    "dataset": payload["dataset"],
                    "model_id": payload["model_id"],
                    "resident_resource_budget": budget,
                    "logical_resources": int(payload["resources_per_seed"]),
                    "sample_count": len(rows),
                    "seed_count": len({int(row["seed"]) for row in rows}),
                    "token_f1": fmean(float(row["token_f1"]) for row in rows),
                    "gold_answer_logprob": fmean(
                        float(row["gold_answer_logprob"]) for row in rows
                    ),
                    "reload_fraction": fmean(
                        float(row["reload_on_request"]) for row in rows
                    ),
                    "resolve_ms_mean": fmean(resolve),
                    "resolve_ms_p95": _percentile(resolve, 0.95),
                    "completion_ms_mean": fmean(completion),
                    "completion_ms_p95": _percentile(completion, 0.95),
                    "resident_mib_mean": fmean(
                        float(row["resident_bytes_after_request"]) for row in rows
                    )
                    / 1048576,
                    "loads_mean": fmean(float(row["loads"]) for row in lifecycle),
                    "evictions_mean": fmean(
                        float(row["evictions"]) for row in lifecycle
                    ),
                    "reloads_mean": fmean(float(row["reloads"]) for row in lifecycle),
                }
            )
    return result


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & Budget & $n$ & F1 & Reload & Resolve ms & p95 ms & Resident MiB \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{DATASET_LABELS[str(row['dataset'])]} & "
            f"{int(row['resident_resource_budget'])}/"
            f"{int(row['logical_resources'])} & {int(row['sample_count'])} & "
            f"{float(row['token_f1']):.3f} & {float(row['reload_fraction']):.3f} & "
            f"{float(row['resolve_ms_mean']):.1f} & {float(row['resolve_ms_p95']):.1f} & "
            f"{float(row['resident_mib_mean']):.1f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for dataset in DATASET_LABELS:
        selected = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: int(row["resident_resource_budget"]),
        )
        if not selected:
            continue
        budgets = [int(row["resident_resource_budget"]) for row in selected]
        axes[0].plot(
            budgets,
            [float(row["reload_fraction"]) for row in selected],
            marker="o",
            label=DATASET_LABELS[dataset],
        )
        axes[1].plot(
            budgets,
            [float(row["resolve_ms_mean"]) for row in selected],
            marker="o",
            label=DATASET_LABELS[dataset],
        )
    axes[0].set_xlabel("Resident resources")
    axes[0].set_ylabel("Reload fraction")
    axes[1].set_xlabel("Resident resources")
    axes[1].set_ylabel("Mean resolve time (ms)")
    for axis in axes:
        axis.set_xticks((1, 2, 4, 8))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
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
    rows = summarize(payloads)
    result = {
        "schema_version": "paper6-2-mlx-m4-pressure-v1",
        "evidence_tier": "CONTROLLED_NATURAL_QA_PRESSURE",
        "hardware": "Apple M4 Pro, 20-core GPU, 48 GiB unified memory",
        "rows": rows,
    }
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_table(args.table, rows)
    write_plot(args.plot, rows)


if __name__ == "__main__":
    main()
