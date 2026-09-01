"""Generate the compact Paper 6.7 qualification figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_7_llamacpp"
OUTPUT = ROOT / "docs/papers/paper6_7_llamacpp/figures/e0_qualification.png"
NATIVE_OUTPUT = ROOT / "docs/papers/paper6_7_llamacpp/figures/native_server_attach.png"
DATASETS = ("hotpotqa", "qasper", "2wikimultihopqa")


def indexed(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (row["dataset"], row["condition"]): row for row in payload["aggregates"]
    }


def main() -> None:
    metal = indexed(RESULTS / "metal_natural_e0.json")
    cpu = indexed(RESULTS / "cpu_natural_e0.json")
    x = np.arange(len(DATASETS))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for offset, (label, rows, condition) in enumerate(
        (
            ("Metal selected", metal, "selected_context"),
            ("Metal full", metal, "full_context"),
            ("CPU selected", cpu, "selected_context"),
            ("CPU full", cpu, "full_context"),
        )
    ):
        axes[0].bar(
            x + (offset - 1.5) * width,
            [rows[(dataset, condition)]["ttft_ms"]["p50"] for dataset in DATASETS],
            width,
            label=label,
        )
    ratios = [
        metal[(dataset, "selected_context")]["mean_prompt_tokens"]
        / metal[(dataset, "full_context")]["mean_prompt_tokens"]
        for dataset in DATASETS
    ]
    axes[1].bar(x, ratios, color="#287271")
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Median time to first token (ms)")
    axes[1].set_ylabel("Selected/full visible prompt ratio")
    for axis in axes:
        axis.set_xticks(x, ("HotpotQA", "QASPER", "2Wiki"), rotation=18)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)

    native = json.loads(
        (RESULTS / "native_server_attach.json").read_text(encoding="utf-8")
    )
    summary = native["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35))
    labels = ("E0 selected", "E2 cold attach", "E2 warm attach")
    means = (
        summary["mean_e0_ms"], summary["mean_e2_ms"], summary["mean_e2_warm_ms"]
    )
    p95 = (
        summary["e0_p95_ms"], summary["e2_p95_ms"], summary["e2_warm_p95_ms"]
    )
    x = np.arange(3)
    axes[0].bar(x - 0.18, means, 0.36, label="Mean", color="#287271")
    axes[0].bar(x + 0.18, p95, 0.36, label="p95", color="#d17a22")
    axes[0].set_xticks(x, labels, rotation=14)
    axes[0].set_ylabel("Request latency (ms)")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    checks = (
        summary["e0_e2_exact"] / summary["runs"],
        summary["e2_warm_exact"] / summary["runs"],
        summary["concurrent_exact"] / summary["concurrent_requests"],
        summary["absent_differs"] / summary["runs"],
    )
    axes[1].bar(
        np.arange(4), checks, color=("#2f6690", "#287271", "#4f772d", "#7f5539")
    )
    axes[1].set_xticks(
        np.arange(4), ("E0/E2", "Warm", "Concurrent", "Absent differs"), rotation=14
    )
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Fraction of checks")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(NATIVE_OUTPUT, dpi=180)
    print(NATIVE_OUTPUT)


if __name__ == "__main__":
    main()
