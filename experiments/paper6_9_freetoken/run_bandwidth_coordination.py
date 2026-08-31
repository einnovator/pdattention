"""Controlled expert/PRA transfer sweep for Paper 6.9."""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

from pra_freetoken import PrefetchDemand, PrefetchStrategy, coordinate_prefetch


def hardware() -> dict[str, str]:
    result = {"platform": platform.platform(), "processor": platform.processor()}
    try:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).splitlines()[0]
        result["gpu"] = line.strip()
    except (OSError, subprocess.SubprocessError):
        result["gpu"] = "not exposed to benchmark process"
    return result


def make_demands(seed: int) -> tuple[PrefetchDemand, ...]:
    rng = random.Random(seed)
    expert_mib = rng.uniform(96, 192)
    pra_mib = rng.uniform(12, 48)
    return (
        PrefetchDemand(
            "expert",
            int(expert_mib * 2**20),
            need_time_s=rng.uniform(0.012, 0.025),
        ),
        PrefetchDemand(
            "pra",
            int(pra_mib * 2**20),
            need_time_s=rng.uniform(0.025, 0.060),
        ),
    )


def run() -> dict[str, object]:
    rows = []
    bandwidths = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    seeds = (11, 23, 37, 71, 101)
    for bandwidth_gib_s in bandwidths:
        for seed in seeds:
            demands = make_demands(seed)
            for strategy in PrefetchStrategy:
                outcome = coordinate_prefetch(
                    demands,
                    bandwidth_bytes_per_s=bandwidth_gib_s * 2**30,
                    strategy=strategy,
                )
                rows.append(
                    {
                        "bandwidth_gib_s": bandwidth_gib_s,
                        "seed": seed,
                        **asdict(outcome),
                        "strategy": outcome.strategy.value,
                        "ready_fraction": outcome.ready_fraction,
                    }
                )
    aggregates = []
    for bandwidth in bandwidths:
        for strategy in PrefetchStrategy:
            selected = [
                row
                for row in rows
                if row["bandwidth_gib_s"] == bandwidth
                and row["strategy"] == strategy.value
            ]
            latencies = sorted(float(row["exposed_latency_s"]) for row in selected)
            aggregates.append(
                {
                    "bandwidth_gib_s": bandwidth,
                    "strategy": strategy.value,
                    "mean_exposed_latency_ms": 1000 * statistics.fmean(latencies),
                    "p95_exposed_latency_ms": 1000 * latencies[-1],
                    "mean_ready_fraction": statistics.fmean(
                        float(row["ready_fraction"]) for row in selected
                    ),
                    "mean_expert_mib": statistics.fmean(
                        int(row["expert_bytes"]) / 2**20 for row in selected
                    ),
                    "mean_pra_mib": statistics.fmean(
                        int(row["pra_bytes"]) / 2**20 for row in selected
                    ),
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_9_controlled_bandwidth_coordination_v1",
        "evidence_tier": "CONTROLLED_SCHEDULER_MODEL",
        "measurement_status": "MEASURED",
        "freetoken_commit": "3a20a79038338c33bd051c52152e6d1faa4d9791",
        "host": hardware(),
        "seeds": list(seeds),
        "bandwidth_gib_s": list(bandwidths),
        "interpretation_boundary": (
            "The deterministic single-link model tests coordination policy only; "
            "it is not live FreeToken E2 or E3 evidence."
        ),
        "aggregates": aggregates,
        "rows": rows,
    }


def plot(payload: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = payload["aggregates"]
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    for strategy in PrefetchStrategy:
        selected = [row for row in rows if row["strategy"] == strategy.value]
        axis.plot(
            [row["bandwidth_gib_s"] for row in selected],
            [row["mean_exposed_latency_ms"] for row in selected],
            marker="o",
            label=strategy.value.replace("_", " "),
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Available host-device bandwidth (GiB/s)")
    axis.set_ylabel("Mean exposed transfer latency (ms)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/papers/shared/results/paper6_9_freetokens/bandwidth_coordination.json"
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path(
            "docs/papers/paper6_9_freetokens/figures/bandwidth_coordination.png"
        ),
    )
    args = parser.parse_args()
    payload = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    plot(payload, args.plot)
    print(json.dumps({"output": str(args.output), "plot": str(args.plot)}))


if __name__ == "__main__":
    main()
