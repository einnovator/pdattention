"""Render Paper 6.6 architecture and controlled-result figures."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


MIB = 1024 * 1024


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.3, 2.35, 1.65, 0.75, "Typed records\n+ query"),
        (2.25, 2.35, 1.65, 0.75, "PRA routing\n+ selection"),
        (4.2, 2.35, 1.65, 0.75, "Layer-local\nselected K/V"),
        (6.15, 2.35, 1.65, 0.75, "HF attention\n+ cache"),
        (8.1, 2.35, 1.95, 0.75, "Generated\ntokens"),
        (2.25, 0.55, 1.65, 0.75, "AirLLM layer\nshards"),
        (4.2, 0.55, 1.65, 0.75, "Weight\nprefetch"),
        (6.15, 0.55, 1.65, 0.75, "One/few layers\nresident"),
    ]
    colors = ["#d9edf7"] * 5 + ["#f7e6c4"] * 3
    for (x, y, w, h, label), color in zip(boxes, colors):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#253746", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    for start, end in [((1.95, 2.72), (2.25, 2.72)), ((3.9, 2.72), (4.2, 2.72)),
                       ((5.85, 2.72), (6.15, 2.72)), ((7.8, 2.72), (8.1, 2.72)),
                       ((3.9, 0.92), (4.2, 0.92)), ((5.85, 0.92), (6.15, 0.92)),
                       ((6.98, 1.3), (6.98, 2.35))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#253746"})
    ax.text(3.05, 3.45, "semantic-context plane", ha="center", fontsize=10, weight="bold")
    ax.text(4.95, 0.12, "model-weight plane", ha="center", fontsize=10, weight="bold")
    ax.text(6.02, 1.7, "compose at\nconsumer layer", ha="right", va="center", fontsize=8)
    _save(fig, output_dir, "airllm_pra_architecture")


def controlled_plots(report: dict, output_dir: Path) -> dict:
    rows = report["rows"]
    contexts = [2048, 8192, 32768, 65536]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    styles = {
        "full_context": ("Full visible context", "#b23a48", "o"),
        "selected_text": ("Selected text E0", "#4472c4", "s"),
    }
    for mode, (label, color, marker) in styles.items():
        selected = [next(row for row in rows if row["execution_mode"] == mode and row["source_tokens"] == n) for n in contexts]
        ax.plot([n / 1024 for n in contexts], [row["memory_bytes"]["peak"] / MIB for row in selected],
                label=label, color=color, marker=marker, linewidth=2)
    native = [next(row for row in rows if row["execution_mode"] == "native_pra"
                   and row["source_tokens"] == n and row["profile"] == "balanced"
                   and row["residency"] == "layer_streamed" and row["prefetch"] == "coordinated"
                   and row["tier"] == "warm") for n in contexts]
    ax.plot([n / 1024 for n in contexts], [row["memory_bytes"]["peak"] / MIB for row in native],
            label="Native PRA, layer-streamed", color="#2f7d32", marker="^", linewidth=2)
    ax.set_xlabel("Backing context (K tokens)")
    ax.set_ylabel("Controlled peak working set (MiB)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save(fig, output_dir, "context_memory_frontier")

    target = [row for row in rows if row["execution_mode"] == "native_pra"
              and row["source_tokens"] == 32768 and row["profile"] == "balanced"
              and row["residency"] == "hybrid" and row["tier"] == "warm"]
    modes = ["none", "independent_parallel", "coordinated"]
    latency = [statistics.mean(row["latency_mean_ms"] for row in target if row["prefetch"] == mode) for mode in modes]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    bars = ax.bar(["None", "Independent", "Shared executor"], latency,
                  color=["#888888", "#2f7d32", "#d2872c"])
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylabel("Mean controlled pass latency (ms)")
    ax.set_ylim(0, max(latency) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir, "prefetch_overlap")

    balanced = [row for row in rows if row["execution_mode"] == "native_pra"
                and row["source_tokens"] == 32768 and row["profile"] == "balanced"
                and row["prefetch"] == "independent_parallel" and row["tier"] == "warm"]
    residency = {}
    for mode in ("hot", "hybrid", "layer_streamed"):
        values = [row for row in balanced if row["residency"] == mode]
        residency[mode] = {
            "latency_mean_ms": statistics.mean(row["latency_mean_ms"] for row in values),
            "pra_peak_mib": statistics.mean(row["memory_bytes"]["pra_hot"] for row in values) / MIB,
        }
    summary = {
        "evidence_tier": report["evidence_tier"],
        "balanced_32k_residency": residency,
        "balanced_32k_hybrid_prefetch_latency_ms": dict(zip(modes, latency)),
        "full_64k_peak_mib": next(row for row in rows if row["execution_mode"] == "full_context" and row["source_tokens"] == 65536)["memory_bytes"]["peak"] / MIB,
        "native_64k_peak_mib": native[-1]["memory_bytes"]["peak"] / MIB,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.input.read_text(encoding="utf-8"))
    architecture(args.output_dir)
    summary = controlled_plots(report, args.output_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
