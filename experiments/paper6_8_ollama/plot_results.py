"""Generate context-pressure and lifecycle figures for Paper 6.8."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_8_ollama"
OUTPUT = ROOT / "docs/papers/paper6_8_ollama/figures/ollama_qualification.png"


def main() -> None:
    natural = json.loads((RESULTS / "natural_e0.json").read_text(encoding="utf-8"))
    lifecycle = json.loads((RESULTS / "lifecycle.json").read_text(encoding="utf-8"))
    indexed = {
        (row["dataset"], row["condition"]): row for row in natural["aggregates"]
    }
    datasets = ("hotpotqa", "qasper", "2wikimultihopqa")
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].bar(
        x - 0.18,
        [indexed[(dataset, "selected_context")]["ttft_ms"]["p50"] for dataset in datasets],
        0.36,
        label="Selected",
    )
    axes[0].bar(
        x + 0.18,
        [indexed[(dataset, "full_context")]["ttft_ms"]["p50"] for dataset in datasets],
        0.36,
        label="Full",
    )
    lifecycle_rows = lifecycle["rows"]
    labels = [row["label"] for row in lifecycle_rows]
    axes[1].bar(np.arange(len(labels)), [row["elapsed_ms"] for row in lifecycle_rows], color="#8c564b")
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    axes[0].set_xticks(x, ("HotpotQA", "QASPER", "2Wiki"), rotation=18)
    axes[0].set_ylabel("Median time to first token (ms)")
    axes[1].set_ylabel("Request latency (ms)")
    axes[0].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
