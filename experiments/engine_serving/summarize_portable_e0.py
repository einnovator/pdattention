"""Summarize portable natural-QA and warm-load E0 engine measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


CONDITION_LABELS = {
    "no_context": "None",
    "selected_context": "Selected",
    "full_context": "Full",
}
REPRESENTATION_LABELS = {"pra_only": "Selected", "full_context": "Full"}
WORKLOAD_LABELS = {
    "shared_resource": "Shared",
    "independent_resources": "Independent",
}


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _natural_table(payload: Mapping[str, object]) -> str:
    rows = [
        r"Dataset & Context & F1 & Contain. & Prompt tok. & TTFT p50 & TTFT p95 \\",
        r"\midrule",
    ]
    for row in payload["aggregates"]:
        rows.append(
            "{} & {} & {:.3f} & {:.3f} & {:.0f} & {:.1f} & {:.1f} \\\\".format(
                str(row["dataset"]).replace("2wikimultihopqa", "2Wiki"),
                CONDITION_LABELS[str(row["condition"])],
                float(row["token_f1"]),
                float(row["answer_containment"]),
                float(row["mean_prompt_tokens"]),
                float(row["ttft_ms"]["p50"]),
                float(row["ttft_ms"]["p95"]),
            )
        )
    return "\n".join(rows) + "\n"


def _load_table(payload: Mapping[str, object]) -> str:
    rows = [
        r"Workload & Context & $C$ & Req/s & TTFT p50 & TTFT p99 & ITL p99 \\",
        r"\midrule",
    ]
    wanted = {1, 8, 16}
    for row in payload["aggregates"]:
        if int(row["concurrency"]) not in wanted:
            continue
        rows.append(
            "{} & {} & {} & {:.2f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(
                WORKLOAD_LABELS[str(row["workload"])],
                REPRESENTATION_LABELS[str(row["representation"])],
                int(row["concurrency"]),
                float(row["request_throughput_s"]),
                float(row["ttft_ms"]["p50"]),
                float(row["ttft_ms"]["p99"]),
                float(row["itl_ms"]["p99"]),
            )
        )
    return "\n".join(rows) + "\n"


def _plot_load(payload: Mapping[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    colors = {"pra_only": "#0072B2", "full_context": "#D55E00"}
    styles = {"shared_resource": "-", "independent_resources": "--"}
    for representation in REPRESENTATION_LABELS:
        for workload in WORKLOAD_LABELS:
            rows = [
                row
                for row in payload["aggregates"]
                if row["representation"] == representation
                and row["workload"] == workload
            ]
            rows.sort(key=lambda row: int(row["concurrency"]))
            label = (
                f"{REPRESENTATION_LABELS[representation]}, "
                f"{WORKLOAD_LABELS[workload].lower()}"
            )
            x = [int(row["concurrency"]) for row in rows]
            axes[0].plot(
                x,
                [float(row["request_throughput_s"]) for row in rows],
                color=colors[representation],
                linestyle=styles[workload],
                marker="o",
                label=label,
            )
            axes[1].plot(
                x,
                [float(row["ttft_ms"]["p99"]) for row in rows],
                color=colors[representation],
                linestyle=styles[workload],
                marker="o",
                label=label,
            )
    axes[0].set_ylabel("Requests/s")
    axes[1].set_ylabel("TTFT p99 (ms)")
    for axis in axes:
        axis.set_xlabel("Concurrent requests")
        axis.set_xticks(sorted({int(r["concurrency"]) for r in payload["aggregates"]}))
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    natural = _load(args.natural)
    (args.output_dir / "generated_portable_natural_table.tex").write_text(
        _natural_table(natural), encoding="utf-8"
    )
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "engine": natural["engine"],
        "model_id": natural["model_id"],
        "natural_source": str(args.natural),
        "natural_aggregates": natural["aggregates"],
    }
    if args.load:
        load = _load(args.load)
        (args.output_dir / "generated_portable_load_table.tex").write_text(
            _load_table(load), encoding="utf-8"
        )
        _plot_load(load, args.output_dir / "portable_e0_load.png")
        summary["load_source"] = str(args.load)
        summary["load_aggregates"] = load["aggregates"]
    (args.output_dir / "portable_e0_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
