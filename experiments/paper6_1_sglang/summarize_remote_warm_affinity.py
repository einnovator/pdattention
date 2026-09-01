"""Build Paper 6.1 tables and plots for off-node WARM placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    leads = payload["lead_curve"]
    placements = payload["placement_curve"]

    selected = [
        row
        for row in placements
        if row["concurrency"] in {8, 16}
    ]
    lines = [
        "\\begin{tabular}{rrlrrrr}",
        "\\toprule",
        "Concurrency & Workload & Policy & HOT/requests & p95 stall & Remote MiB & Requests/s \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['concurrency']} & {row['workload']} & {row['policy']} & "
            f"{row['hot_at_schedule']}/{row['requests']} & "
            f"{row['stall_p95_ms']:.1f} & "
            f"{row['remote_read_bytes'] / 2**20:.1f} & "
            f"{row['requests_per_second']:.2f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(1, 3, figsize=(9.2, 2.75))
    lead_ms = [row["lead_ms"] for row in leads]
    axes[0].plot(lead_ms, [row["stall_p50_ms"] for row in leads], "o-", label="p50")
    axes[0].plot(lead_ms, [row["stall_p95_ms"] for row in leads], "s--", label="p95")
    axes[0].set_xlabel("Prefetch lead (ms)")
    axes[0].set_ylabel("Demand stall (ms)")
    axes[0].set_title("Off-node lead-time curve")
    axes[0].legend(frameon=False)

    for policy, color in (("affinity", "#4C78A8"), ("random", "#F58518")):
        rows = [
            row for row in placements
            if row["workload"] == "mixed" and row["policy"] == policy
        ]
        axes[1].plot(
            [row["concurrency"] for row in rows],
            [row["remote_read_bytes"] / 2**20 for row in rows],
            "o-",
            color=color,
            label=policy,
        )
        axes[2].plot(
            [row["concurrency"] for row in rows],
            [row["stall_p95_ms"] / 1000 for row in rows],
            "o-",
            color=color,
            label=policy,
        )
    axes[1].set_xlabel("Concurrent requests")
    axes[1].set_ylabel("Remote read (MiB)")
    axes[1].set_title("Mixed-resource traffic")
    axes[2].set_xlabel("Concurrent requests")
    axes[2].set_ylabel("Demand stall p95 (s)")
    axes[2].set_title("Mixed-resource tail")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
