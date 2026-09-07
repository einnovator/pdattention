"""Compare Paper 9 live MLX reuse across memory-capacity regimes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def host_points(directory: Path) -> list[dict]:
    rows = read_rows(directory / "rows.csv")
    sessions = read_rows(directory / "session_rows.csv")
    points = []
    for size in sorted({int(row["target_shared_tokens"]) for row in sessions}):
        fanout = max(
            int(row["fanout"])
            for row in sessions
            if int(row["target_shared_tokens"]) == size
        )
        selected_sessions = [
            row for row in sessions
            if int(row["target_shared_tokens"]) == size and int(row["fanout"]) == fanout
        ]
        text = [
            float(row["request_ms"]) for row in rows
            if int(row["target_shared_tokens"]) == size
            and row["condition"] == "host_split_text"
        ]
        native = [
            float(row["request_ms"]) for row in rows
            if int(row["target_shared_tokens"]) == size
            and row["condition"] == "pra_native_kv"
        ]
        points.append({
            "shared_tokens": size,
            "max_measured_fanout": fanout,
            "amortized_speedup": statistics.mean(
                float(row["amortized_speedup"]) for row in selected_sessions
            ),
            "physical_token_reduction": statistics.mean(
                float(row["physical_token_reduction"]) for row in selected_sessions
            ),
            "native_to_text_request_ratio": statistics.mean(native) / statistics.mean(text),
            "argmax_parity": statistics.mean(
                float(row["native_argmax_parity"]) for row in selected_sessions
            ),
        })
    return points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m4", type=Path, required=True)
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    hosts = {
        "M4 Pro, 48 GB": host_points(arguments.m4),
        "M5, 16 GB": host_points(arguments.m5),
    }
    artifact = {
        "protocol": "paper9-mlx-cross-host-v1",
        "interpretation": "Observed host-model timings; maximum measured fan-out is 16 through 8K and 4 at 32K.",
        "hosts": hosts,
    }
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / "summary.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    colors = {"M4 Pro, 48 GB": "#255b96", "M5, 16 GB": "#b5533c"}
    for host, points in hosts.items():
        sizes = [row["shared_tokens"] for row in points]
        axes[0].plot(
            sizes,
            [row["amortized_speedup"] for row in points],
            marker="o",
            color=colors[host],
            label=host,
        )
        axes[1].plot(
            sizes,
            [row["native_to_text_request_ratio"] for row in points],
            marker="o",
            color=colors[host],
            label=host,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.axhline(1.0, color="#333333", linewidth=0.8)
        axis.set_xlabel("Shared source tokens")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Amortized session speedup")
    axes[1].set_ylabel("Native / text child latency")
    figure.tight_layout()
    figure.savefig(arguments.output / "mlx_cross_host.pdf", bbox_inches="tight")
    figure.savefig(arguments.output / "mlx_cross_host.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
