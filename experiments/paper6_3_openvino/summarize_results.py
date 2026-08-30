"""Generate Paper 6.3 tables and figures from committed raw artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping


def _load(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape(value: str) -> str:
    return value.replace("_", r"\_")


def generate(results: Path) -> None:
    cpu = _load(results / "e0_cpu.json")
    gpu = _load(results / "e0_gpu.json")
    batch = _load(results / "continuous_batching_gpu.json")
    audit = _load(results / "environment_audit.json")

    e0_rows = []
    for device, payload in (("CPU", cpu), ("GPU", gpu)):
        for row in payload["aggregates"]:
            if row["condition"] not in {"pra_only", "prefix_plus_pra", "full_context"}:
                continue
            e0_rows.append(
                "{} & {} & {:.0f} & {:.2f} & {:.1f} & {:.1f} & {:.1f} \\\\".format(
                    device,
                    _escape(str(row["condition"])),
                    float(row["mean_input_tokens"]),
                    float(row["quality_success_rate"]),
                    float(row["cold_ttft_ms"]),
                    float(row["warm_ttft_ms_mean"]),
                    float(row["completion_latency_ms_p50"]),
                )
            )
    (results / "generated_e0_device_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{llrrrrr}\toprule",
                r"Device & condition & visible & success & cold TTFT & warm TTFT & latency p50\\\midrule",
                *e0_rows,
                r"\bottomrule\end{tabular}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    table_rows = []
    for row in batch["aggregates"]:
        if int(row["concurrency"]) not in {1, 8}:
            continue
        transport = "selected" if row["representation"] == "pra_only" else "full"
        resources = "shared" if row["workload"] == "shared_resource" else "independent"
        table_rows.append(
            "{} / {} & {} & {:.2f} & {:.1f} & {:.1f} & {:.0f} \\\\".format(
                transport,
                resources,
                row["concurrency"],
                float(row["request_throughput_s"]),
                float(row["ttft_ms_p99"]),
                float(row["generation_ms_p99"]),
                float(row["rss_after_bytes"]) / (1024**2),
            )
        )
    (results / "generated_batching_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{llrrrr}\toprule",
                r"Transport / resources & $C$ & req/s & TTFT p99 & generation p99 & RSS MiB\\\midrule",
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
            "devices": audit["devices"],
            "gates": audit["gates"],
        },
        "e0": {"CPU": cpu["aggregates"], "GPU": gpu["aggregates"]},
        "continuous_batching_gpu": batch["aggregates"],
    }
    (results / "summary.json").write_text(
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
            row for row in batch["aggregates"]
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
    fig.savefig(results / "continuous_batching_gpu.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    generate(args.results)


if __name__ == "__main__":
    main()
