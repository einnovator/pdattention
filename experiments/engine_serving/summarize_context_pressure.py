"""Summarize matched context-size pressure sweeps at peak concurrency."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping


REPRESENTATIONS = {"pra_only": "Selected", "full_context": "Full"}
WORKLOADS = {"shared_resource": "Shared", "independent_resources": "Independent"}


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tail(row: Mapping[str, object], percentile: str) -> float:
    nested = row.get("ttft_ms")
    if isinstance(nested, Mapping):
        return float(nested[percentile])
    return float(row[f"ttft_ms_{percentile}"])


def summarize(points: list[tuple[str, Mapping[str, object]]]) -> Mapping[str, object]:
    rows = []
    for size, payload in points:
        max_concurrency = max(int(value) for value in payload["concurrency"])
        for aggregate in payload["aggregates"]:
            if int(aggregate["concurrency"]) != max_concurrency:
                continue
            matching = [
                sample
                for sample in payload["samples"]
                if sample["representation"] == aggregate["representation"]
                and sample["workload"] == aggregate["workload"]
                and int(sample["concurrency"]) == max_concurrency
            ]
            prompt_values = [
                float(sample.get("prompt_tokens", sample.get("input_tokens")))
                for sample in matching
                if sample.get("prompt_tokens", sample.get("input_tokens")) is not None
            ]
            rows.append(
                {
                    "size": size,
                    "representation": aggregate["representation"],
                    "workload": aggregate["workload"],
                    "concurrency": max_concurrency,
                    "mean_prompt_tokens": (
                        statistics.fmean(prompt_values) if prompt_values else None
                    ),
                    "quality_success_rate": float(aggregate["quality_success_rate"]),
                    "request_throughput_s": float(aggregate["request_throughput_s"]),
                    "output_throughput_tokens_s": float(
                        aggregate["output_throughput_tokens_s"]
                    ),
                    "ttft_ms_p99": _tail(aggregate, "p99"),
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_context_pressure_summary_v1",
        "engine": points[0][1]["engine"] if "engine" in points[0][1] else "openvino-genai",
        "model_id": points[0][1].get("model_id", points[0][1].get("model")),
        "rows": rows,
    }


def _table(rows: list[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Size & Workload & Context & Prompt tok. & Succ. & Req/s & Out tok/s & TTFT p99 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {:.0f} & {:.2f} & {:.2f} & {:.1f} & {:.1f} \\\\".format(
                row["size"],
                WORKLOADS[str(row["workload"])],
                REPRESENTATIONS[str(row["representation"])],
                float(row["mean_prompt_tokens"]),
                float(row["quality_success_rate"]),
                float(row["request_throughput_s"]),
                float(row["output_throughput_tokens_s"]),
                float(row["ttft_ms_p99"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _plot(rows: list[Mapping[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt

    sizes = list(dict.fromkeys(str(row["size"]) for row in rows))
    x = list(range(len(sizes)))
    colors = {"pra_only": "#0072B2", "full_context": "#D55E00"}
    styles = {"shared_resource": "-", "independent_resources": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for representation in REPRESENTATIONS:
        for workload in WORKLOADS:
            selected = [
                next(
                    row
                    for row in rows
                    if row["size"] == size
                    and row["representation"] == representation
                    and row["workload"] == workload
                )
                for size in sizes
            ]
            label = f"{REPRESENTATIONS[representation]}, {WORKLOADS[workload].lower()}"
            axes[0].plot(
                x,
                [float(row["request_throughput_s"]) for row in selected],
                marker="o",
                color=colors[representation],
                linestyle=styles[workload],
                label=label,
            )
            axes[1].plot(
                x,
                [float(row["ttft_ms_p99"]) for row in selected],
                marker="o",
                color=colors[representation],
                linestyle=styles[workload],
                label=label,
            )
    axes[0].set_ylabel("Requests/s")
    axes[1].set_ylabel("TTFT p99 (ms)")
    for axis in axes:
        axis.set_xticks(x, sizes)
        axis.set_xlabel("Full-context pressure point")
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point",
        action="append",
        required=True,
        help="Ordered LABEL=JSON benchmark point.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    points = []
    for value in args.point:
        label, separator, path = value.partition("=")
        if not separator:
            parser.error(f"Invalid point {value!r}; expected LABEL=JSON")
        points.append((label, _load(Path(path))))
    result = summarize(points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "context_pressure_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "generated_context_pressure_table.tex").write_text(
        _table(result["rows"]), encoding="utf-8"
    )
    _plot(result["rows"], args.output_dir / "context_pressure.png")


if __name__ == "__main__":
    main()
