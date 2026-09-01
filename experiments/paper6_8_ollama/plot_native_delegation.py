"""Plot the live Ollama-to-llama.cpp native delegation result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    summary = payload["summary"]
    colors = ("#4C78A8", "#F58518", "#54A24B")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    latency = (
        summary["mean_e0_ms"],
        summary["mean_e2_ms"],
        summary["mean_e2_warm_ms"],
    )
    axes[0].bar(("E0 selected", "E2 encode", "E2 warm"), latency, color=colors)
    axes[0].set_ylabel("Mean request latency (ms)")
    axes[0].set_title("Same evidence and model artifact")
    axes[0].tick_params(axis="x", rotation=18)
    for index, value in enumerate(latency):
        axes[0].text(index, value + 3, f"{value:.1f}", ha="center", fontsize=8)

    tokens = (
        summary["mean_e0_prompt_tokens"],
        summary["mean_e2_wire_tokens"],
        summary["mean_e2_native_tokens"],
    )
    axes[1].bar(("E0 visible", "E2 wire", "E2 native"), tokens, color=colors)
    axes[1].set_ylabel("Mean input tokens")
    axes[1].set_title("Visible transport vs native attachment")
    axes[1].tick_params(axis="x", rotation=18)
    for index, value in enumerate(tokens):
        axes[1].text(index, value + 1, f"{value:.1f}", ha="center", fontsize=8)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
