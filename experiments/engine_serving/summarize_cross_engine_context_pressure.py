"""Compare context-pressure curves using within-engine normalized ratios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


WORKLOAD_LABELS = {
    "shared_resource": "Shared",
    "independent_resources": "Independent",
}


def _paired_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    rows = payload["rows"]
    pairs = []
    keys = list(dict.fromkeys((row["size"], row["workload"]) for row in rows))
    for size, workload in keys:
        selected = next(
            row
            for row in rows
            if row["size"] == size
            and row["workload"] == workload
            and row["representation"] == "pra_only"
        )
        full = next(
            row
            for row in rows
            if row["size"] == size
            and row["workload"] == workload
            and row["representation"] == "full_context"
        )
        pairs.append(
            {
                "size": size,
                "workload": workload,
                "selected_tokens": selected["mean_prompt_tokens"],
                "full_tokens": full["mean_prompt_tokens"],
                "selected_success": selected["quality_success_rate"],
                "full_success": full["quality_success_rate"],
                "selected_request_throughput_s": selected["request_throughput_s"],
                "full_request_throughput_s": full["request_throughput_s"],
                "throughput_ratio": (
                    selected["request_throughput_s"] / full["request_throughput_s"]
                ),
                "selected_ttft_ms_p99": selected["ttft_ms_p99"],
                "full_ttft_ms_p99": full["ttft_ms_p99"],
                "ttft_ratio": full["ttft_ms_p99"] / selected["ttft_ms_p99"],
            }
        )
    return pairs


def summarize(inputs: list[tuple[str, Mapping[str, object]]]) -> Mapping[str, object]:
    rows = []
    for engine, payload in inputs:
        rows.extend({"engine": engine, **row} for row in _paired_rows(payload))
    return {
        "schema_version": "1.0",
        "benchmark": "cross_engine_context_pressure_v1",
        "normalization": "within-engine selected/full at concurrency 16",
        "rows": rows,
    }


def _table(rows: list[Mapping[str, object]]) -> str:
    large = [row for row in rows if row["size"] == "Large"]
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Engine & Workload & Sel./full tok. & Succ. S/F & Req/s S/F & Req. ratio & p99 S/F & Tail ratio \\",
        r"\midrule",
    ]
    for row in large:
        lines.append(
            "{} & {} & {:.0f}/{:.0f} & {:.2f}/{:.2f} & {:.1f}/{:.1f} & "
            "{:.2f}$\\times$ & {:.0f}/{:.0f} & {:.2f}$\\times$ \\\\".format(
                row["engine"],
                WORKLOAD_LABELS[str(row["workload"])],
                row["selected_tokens"],
                row["full_tokens"],
                row["selected_success"],
                row["full_success"],
                row["selected_request_throughput_s"],
                row["full_request_throughput_s"],
                row["throughput_ratio"],
                row["selected_ttft_ms_p99"],
                row["full_ttft_ms_p99"],
                row["ttft_ratio"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _plot(rows: list[Mapping[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    sizes = ["Small", "Medium", "Large"]
    x = list(range(len(sizes)))
    colors = {"vLLM CUDA": "#0072B2", "TensorRT-LLM": "#009E73", "OpenVINO": "#D55E00"}
    styles = {"shared_resource": "-", "independent_resources": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for engine in dict.fromkeys(str(row["engine"]) for row in rows):
        for workload in WORKLOAD_LABELS:
            selected = [
                next(
                    row
                    for row in rows
                    if row["engine"] == engine
                    and row["workload"] == workload
                    and row["size"] == size
                )
                for size in sizes
            ]
            label = f"{engine}, {WORKLOAD_LABELS[workload].lower()}"
            axes[0].plot(
                x,
                [row["throughput_ratio"] for row in selected],
                marker="o",
                color=colors[engine],
                linestyle=styles[workload],
                label=label,
            )
            axes[1].plot(
                x,
                [row["ttft_ratio"] for row in selected],
                marker="o",
                color=colors[engine],
                linestyle=styles[workload],
                label=label,
            )
    axes[0].set_ylabel("Selected/full throughput")
    axes[1].set_ylabel("Full/selected TTFT p99")
    for axis in axes:
        axis.axhline(1.0, color="#555555", linewidth=0.8)
        axis.set_xticks(x, sizes)
        axis.set_xlabel("Full-context pressure point")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", action="append", required=True, help="LABEL=JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = []
    for value in args.engine:
        label, separator, path = value.partition("=")
        if not separator:
            parser.error(f"Invalid engine {value!r}; expected LABEL=JSON")
        inputs.append((label, json.loads(Path(path).read_text(encoding="utf-8"))))
    result = summarize(inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "generated_table.tex").write_text(
        _table(result["rows"]), encoding="utf-8"
    )
    _plot(result["rows"], args.output_dir / "normalized_pressure.png")


if __name__ == "__main__":
    main()
