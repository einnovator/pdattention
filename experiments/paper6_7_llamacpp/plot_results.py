"""Generate the compact Paper 6.7 qualification figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_7_llamacpp"
OUTPUT = ROOT / "docs/papers/paper6_7_llamacpp/figures/e0_qualification.png"
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


if __name__ == "__main__":
    main()
