"""Generate Paper 6.4 tables and figures from committed raw artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def generate(results: Path, output: Path) -> None:
    smoke = _load(results / "e0_serving_smoke.json")
    concurrency = _load(results / "e0_concurrency.json")
    audit = _load(results / "environment_audit.json")
    output.mkdir(parents=True, exist_ok=True)

    smoke_rows = []
    for row in smoke["aggregates"]:
        smoke_rows.append(
            "{} & {:.0f} & {:.2f} & {:.2f} & {:.1f} & {:.1f} \\\\".format(
                _tex_escape(str(row["condition"])),
                float(row["mean_prompt_tokens"]),
                float(row["quality_success_rate"]),
                float(row["cold_ttft_ms"]),
                float(row["warm_ttft_ms_mean"]),
                float(row["completion_latency_ms_p50"]),
            )
        )
    (output / "generated_e0_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{lrrrrr}\toprule",
                r"Condition & Visible & Success & Cold TTFT & Warm TTFT & Latency p50\\\midrule",
                *smoke_rows,
                r"\bottomrule\end{tabular}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selected_rows = [
        row
        for row in concurrency["aggregates"]
        if int(row["concurrency"]) in {1, 8}
    ]
    table_rows = []
    for row in selected_rows:
        label = "selected" if row["representation"] == "pra_only" else "full"
        shared = "shared" if row["workload"] == "shared_resource" else "independent"
        table_rows.append(
            "{} / {} & {} & {:.1f} & {:.1f} & {:.1f} & {:.0f} \\\\".format(
                label,
                shared,
                row["concurrency"],
                float(row["request_throughput_s"]),
                float(row["ttft_ms_p99"]),
                float(row["completion_latency_ms_p99"]),
                float(row["mean_cached_tokens"]),
            )
        )
    (output / "generated_concurrency_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{llrrrr}\toprule",
                r"Transport / resources & $C$ & req/s & TTFT p99 & completion p99 & cached\\\midrule",
                *table_rows,
                r"\bottomrule\end{tabular}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": "1.0",
        "environment": {
            "packages": audit["packages"],
            "gpu": audit["gpu"],
            "gates": audit["gates"],
        },
        "e0_quality": {
            row["condition"]: row["quality_success_rate"]
            for row in smoke["aggregates"]
        },
        "concurrency": concurrency["aggregates"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    styles = {
        ("pra_only", "shared_resource"): ("#0072B2", "o", "Selected, shared"),
        ("pra_only", "independent_resources"): ("#009E73", "s", "Selected, independent"),
        ("full_context", "shared_resource"): ("#D55E00", "^", "Full, shared"),
        ("full_context", "independent_resources"): ("#CC79A7", "D", "Full, independent"),
    }
    for key, (color, marker, label) in styles.items():
        rows = [
            row
            for row in concurrency["aggregates"]
            if (row["representation"], row["workload"]) == key
        ]
        rows.sort(key=lambda row: int(row["concurrency"]))
        x = [row["concurrency"] for row in rows]
        axes[0].plot(x, [row["request_throughput_s"] for row in rows], color=color, marker=marker, label=label)
        axes[1].plot(x, [row["ttft_ms_p99"] for row in rows], color=color, marker=marker, label=label)
    axes[0].set_ylabel("Requests/s")
    axes[1].set_ylabel("TTFT p99 (ms)")
    for axis in axes:
        axis.set_xlabel("Concurrent requests")
        axis.set_xticks([1, 2, 4, 8])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "e0_concurrency.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.results, args.output)


if __name__ == "__main__":
    main()
