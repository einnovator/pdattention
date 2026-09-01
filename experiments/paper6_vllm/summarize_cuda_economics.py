"""Summarize the selector-frozen vLLM CUDA economics matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LABELS = {
    "full": "FULL",
    "e0_selected_text": "E0",
    "e2_hot": "E2-HOT",
    "e2_warm": "E2-WARM",
}


def _number(value: float | int | None, digits: int = 1) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the publication-facing fields and preserve measurement status."""

    rows = []
    for source in payload["rows"]:
        rows.append(
            {
                "condition": source["condition"],
                "label": LABELS[source["condition"]],
                "requests": source["requests"],
                "success_rate": source["success_rate"],
                "successful_requests_per_second": source[
                    "successful_requests_per_second"
                ],
                "ttft_p50_ms": source["ttft_ms"]["p50"],
                "ttft_p95_ms": source["ttft_ms"]["p95"],
                "ttft_p99_ms": source["ttft_ms"]["p99"],
                "itl_p50_ms": source["mean_itl_ms"]["p50"],
                "itl_p95_ms": source["mean_itl_ms"]["p95"],
                "itl_p99_ms": source["mean_itl_ms"]["p99"],
                "peak_allocated_mib": source["peak_allocated_bytes"] / 2**20,
                "apc_blocks_mean": source["apc_blocks_mean"],
                "pra_logical_blocks": source["pra_logical_blocks"],
                "pra_hot_source_mib": source["pra_hot_source_bytes"] / 2**20,
                "pra_warm_persisted_mib": source["pra_warm_persisted_bytes"] / 2**20,
                "h2d_mib_per_request": source["h2d_bytes_per_request"] / 2**20,
                "d2d_mib_per_request": source["d2d_bytes_per_request"] / 2**20,
                "tail_status": source["tail_status"],
            }
        )
    return {
        "schema_version": "paper6-vllm-cuda-economics-summary-v1",
        "source_schema_version": payload["schema_version"],
        "evidence_tier": payload["evidence_tier"],
        "integration_status": payload["integration_status"],
        "engine_version": payload["engine_version"],
        "model_id": payload["model_id"],
        "device": payload["device"],
        "selector_frozen": payload["selector_frozen"],
        "source_slots_scheduler_visible_in_e2": payload[
            "source_slots_scheduler_visible_in_e2"
        ],
        "concurrency": payload["concurrency"],
        "requests_per_condition": payload["requests_per_condition"],
        "hbm_decomposition": payload["hbm_decomposition"],
        "rows": rows,
        "limitations": payload["limitations"],
    }


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = list(summary["rows"][0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["rows"])


def _write_tables(summary: dict[str, Any], output_dir: Path) -> None:
    latency = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Condition & Success & succ. req/s & \multicolumn{3}{c}{TTFT (ms)} & \multicolumn{3}{c}{mean ITL (ms)} \\",
        r" &  &  & p50 & p95 & p99 & p50 & p95 & p99 \\",
        r"\midrule",
    ]
    memory = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Condition & peak HBM & APC blocks & PRA blocks & HOT MiB & WARM MiB & H2D/req & D2D/req \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        latency.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["label"],
                _number(row["success_rate"], 3),
                _number(row["successful_requests_per_second"], 1),
                _number(row["ttft_p50_ms"]),
                _number(row["ttft_p95_ms"]),
                _number(row["ttft_p99_ms"]),
                _number(row["itl_p50_ms"]),
                _number(row["itl_p95_ms"]),
                _number(row["itl_p99_ms"]),
            )
        )
        memory.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["label"],
                _number(row["peak_allocated_mib"]),
                _number(row["apc_blocks_mean"], 2),
                row["pra_logical_blocks"],
                _number(row["pra_hot_source_mib"], 2),
                _number(row["pra_warm_persisted_mib"], 2),
                _number(row["h2d_mib_per_request"], 2),
                _number(row["d2d_mib_per_request"], 2),
            )
        )
    latency.extend((r"\bottomrule", r"\end{tabular}"))
    memory.extend((r"\bottomrule", r"\end{tabular}"))
    (output_dir / "generated_cuda_economics_latency_table.tex").write_text(
        "\n".join(latency) + "\n", encoding="utf-8"
    )
    (output_dir / "generated_cuda_economics_memory_table.tex").write_text(
        "\n".join(memory) + "\n", encoding="utf-8"
    )


def _write_plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["label"] for row in summary["rows"]]
    rates = [row["successful_requests_per_second"] for row in summary["rows"]]
    ttft50 = [row["ttft_p50_ms"] for row in summary["rows"]]
    ttft99 = [row["ttft_p99_ms"] for row in summary["rows"]]
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    axes[0].bar(labels, rates, color=("#4c78a8", "#59a14f", "#e15759", "#f28e2b"))
    axes[0].set_ylabel("successful requests/s")
    axes[0].set_title("Useful throughput")
    x = range(len(labels))
    axes[1].plot(x, ttft50, marker="o", label="p50")
    axes[1].plot(x, ttft99, marker="s", label="p99")
    axes[1].set_xticks(list(x), labels)
    axes[1].set_ylabel("TTFT (ms)")
    axes[1].set_title("First-token latency")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
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
    (args.output_dir / "cuda_economics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(summary, args.output_dir / "cuda_economics_summary.csv")
    _write_tables(summary, args.output_dir)
    _write_plot(summary, args.output_dir / "cuda_economics_summary.png")


if __name__ == "__main__":
    main()
