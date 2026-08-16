"""Run the complete Paper 3.5 adaptive controller and systems study."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.paper3_5_adaptive_pra.adaptive_experiment import run_adaptive_experiment
from experiments.paper3_5_adaptive_pra.systems_benchmarks import run_systems_benchmarks


DEFAULT_OUTPUT = ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save(figure: plt.Figure, output: Path, name: str) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(figure)


def plot_adaptive_frontier(output: Path) -> None:
    rows = _rows(output / "adaptive_compute_frontier.csv")
    methods = (
        "fixed_E0_low",
        "fixed_E1_medium",
        "fixed_E2_high",
        "learned_direct",
        "cheap_default_retry",
    )
    labels = {
        "fixed_E0_low": "fixed E0",
        "fixed_E1_medium": "fixed E1",
        "fixed_E2_high": "fixed E2",
        "learned_direct": "learned",
        "cheap_default_retry": "cheap+retry",
    }
    colors = {
        "fixed_E0_low": "#7f8c8d",
        "fixed_E1_medium": "#3498db",
        "fixed_E2_high": "#2c3e50",
        "learned_direct": "#16a085",
        "cheap_default_retry": "#c0392b",
    }
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for axis, dataset in zip(axes, ("hotpotqa", "qasper")):
        for method in methods:
            row = next(value for value in rows if value["dataset"] == dataset and value["method"] == method)
            axis.scatter(
                float(row["mean_effort_cost"]),
                float(row["quality"]),
                s=65,
                color=colors[method],
                label=labels[method],
            )
            offsets = {
                ("hotpotqa", "fixed_E0_low"): (4, -12),
                ("hotpotqa", "fixed_E1_medium"): (4, 5),
                ("hotpotqa", "fixed_E2_high"): (-52, 14),
                ("hotpotqa", "learned_direct"): (-48, -1),
                ("hotpotqa", "cheap_default_retry"): (5, -14),
                ("qasper", "fixed_E0_low"): (4, 4),
                ("qasper", "fixed_E1_medium"): (4, -12),
                ("qasper", "fixed_E2_high"): (-52, 10),
                ("qasper", "learned_direct"): (5, 5),
                ("qasper", "cheap_default_retry"): (5, -13),
            }
            axis.annotate(
                labels[method],
                (float(row["mean_effort_cost"]), float(row["quality"])),
                xytext=offsets[(dataset, method)],
                textcoords="offset points",
                fontsize=7.5,
            )
        axis.set_title(dataset)
        axis.set_xlabel("mean abstract effort")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("complete evidence-chain rate")
    axes[0].set_ylim(-0.03, 1.03)
    _save(figure, output, "adaptive_quality_cost_frontier")


def plot_calibration(output: Path) -> None:
    calibration = json.loads((output / "controller_calibration.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for axis, key, title in (
        (axes[0], "routing_failure_calibration", "Routing failure"),
        (axes[1], "output_entropy_calibration", "Output error"),
    ):
        curve = calibration[key]["risk_coverage"]
        axis.plot([row["coverage"] for row in curve], [row["risk"] for row in curve], color="#c0392b")
        axis.set_title(title)
        axis.set_xlabel("coverage")
        axis.set_ylabel("selective risk")
        axis.set_xlim(0, 1.02)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    _save(figure, output, "controller_risk_coverage")


def plot_indexed_search(output: Path) -> None:
    rows = _rows(output / "indexed_search_benchmarks.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    styles = {
        ("brute_force", "0"): ("loop exact", "#7f8c8d", "o"),
        ("gemm", "0"): ("GEMM exact", "#2c3e50", "s"),
        ("coarse_to_fine", "2"): ("coarse p2", "#e67e22", "^"),
        ("coarse_to_fine", "4"): ("coarse p4", "#16a085", "^"),
        ("coarse_to_fine", "8"): ("coarse p8", "#2980b9", "^"),
    }
    for identity, (label, color, marker) in styles.items():
        values = sorted(
            [row for row in rows if (row["backend"], row["probes"]) == identity],
            key=lambda row: int(row["memory_vectors"]),
        )
        axes[0].plot(
            [int(row["memory_vectors"]) for row in values],
            [float(row["median_search_seconds"]) * 1000 for row in values],
            label=label,
            color=color,
            marker=marker,
        )
        axes[1].plot(
            [int(row["memory_vectors"]) for row in values],
            [float(row["recall_at_8_vs_exact"]) for row in values],
            label=label,
            color=color,
            marker=marker,
        )
    axes[0].set_ylabel("median search latency (ms)")
    axes[1].set_ylabel("Top-8 recall vs exact")
    for axis in axes:
        axis.set_xlabel("indexed native keys")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    axes[1].set_ylim(0, 1.03)
    axes[0].legend(fontsize=7)
    _save(figure, output, "indexed_search_recall_latency")


def plot_concurrency(output: Path) -> None:
    rows = _rows(output / "batching_concurrency.csv")
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    colors = {"pad_to_max": "#7f8c8d", "bucketed": "#3498db", "packed_ragged": "#16a085", "page_based": "#8e44ad"}
    for strategy, color in colors.items():
        values = sorted([row for row in rows if row["strategy"] == strategy], key=lambda row: int(row["concurrent_requests"]))
        axes[0].plot(
            [int(row["concurrent_requests"]) for row in values],
            [float(row["throughput_requests_per_second"]) for row in values],
            marker="o",
            color=color,
            label=strategy,
        )
        axes[1].plot(
            [int(row["concurrent_requests"]) for row in values],
            [float(row["padding_or_fragmentation_fraction"]) for row in values],
            marker="o",
            color=color,
            label=strategy,
        )
    axes[0].set_ylabel("prototype throughput (requests/s)")
    axes[1].set_ylabel("padding / fragmentation fraction")
    for axis in axes:
        axis.set_xlabel("concurrent requests")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    _save(figure, output, "batching_concurrency_frontier")


def plot_baselines(output: Path) -> None:
    rag = _rows(output / "rag_baselines.csv")
    long_context = _rows(output / "long_context_baselines.csv")
    points = []
    for row in rag:
        points.append((row["method"], float(row["cost_units"]), float(row["answer_quality_proxy"]), "retrieval/PRA"))
    for row in long_context:
        points.append((row["method"], float(row["generation_cost_units"]), float(row["answer_quality_proxy"]), "long context"))
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    colors = {"retrieval/PRA": "#16a085", "long context": "#2c3e50"}
    offsets = {
        "single_shot_rag_top4": (-38, 9),
        "multi_query_rag_top8": (5, 7),
        "iterative_rag_top4_plus_links": (5, -13),
        "fixed_pra_low": (4, -13),
        "adaptive_pra_medium": (-8, 7),
        "native_full_context": (-80, 5),
        "truncate_first_128": (5, 5),
        "matched_budget_8_documents": (5, -22),
    }
    for name, cost, quality, family in points:
        axis.scatter(cost, quality, color=colors[family], s=55)
        axis.annotate(
            name.replace("_", " "),
            (cost, quality),
            xytext=offsets[name],
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xscale("log")
    axis.set_xlabel("controlled total inference-cost units (log scale)")
    axis.set_ylabel("answer-quality proxy")
    axis.set_ylim(-0.11, 1.03)
    axis.grid(alpha=0.25)
    _save(figure, output, "matched_baseline_quality_cost")


def plot_architecture(output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 5.2))
    axis.set_xlim(0, 11.2)
    axis.set_ylim(0, 5.2)
    axis.axis("off")
    boxes = {
        "query / current state": (0.3, 3.7, 1.7, 0.8, "#ecf0f1"),
        "observable features": (2.35, 3.7, 1.7, 0.8, "#d6eaf8"),
        "E0 / E1 / E2\ncontroller": (4.4, 3.7, 1.7, 0.8, "#d5f5e3"),
        "indexed native-Q/K\nsearch": (6.45, 3.7, 1.7, 0.8, "#fdebd0"),
        "logical interval\nmaterializer": (8.5, 3.7, 1.7, 0.8, "#f5eef8"),
        "paged K/V cache": (8.5, 1.8, 1.7, 0.8, "#f5eef8"),
        "native attention +\ngeneration": (6.45, 1.8, 1.7, 0.8, "#fdebd0"),
        "confidence /\nconsistency": (4.4, 1.8, 1.7, 0.8, "#d5f5e3"),
        "stop, fallback,\nor retry": (2.35, 1.8, 1.7, 0.8, "#fadbd8"),
        "structured trace": (0.3, 1.8, 1.7, 0.8, "#ecf0f1"),
    }
    centers = {}
    for label, (x, y, width, height, color) in boxes.items():
        rectangle = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="#2c3e50", linewidth=1.1)
        axis.add_patch(rectangle)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
        centers[label] = (x + width / 2, y + height / 2)
    top = ["query / current state", "observable features", "E0 / E1 / E2\ncontroller", "indexed native-Q/K\nsearch", "logical interval\nmaterializer"]
    bottom = ["paged K/V cache", "native attention +\ngeneration", "confidence /\nconsistency", "stop, fallback,\nor retry", "structured trace"]
    for left, right in zip(top, top[1:]):
        left_box, right_box = boxes[left], boxes[right]
        axis.annotate(
            "",
            xy=(right_box[0], centers[right][1]),
            xytext=(left_box[0] + left_box[2], centers[left][1]),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#34495e"},
        )
    top_box, page_box = boxes[top[-1]], boxes["paged K/V cache"]
    axis.annotate(
        "",
        xy=(centers["paged K/V cache"][0], page_box[1] + page_box[3]),
        xytext=(centers[top[-1]][0], top_box[1]),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#34495e"},
    )
    for left, right in zip(bottom, bottom[1:]):
        left_box, right_box = boxes[left], boxes[right]
        axis.annotate(
            "",
            xy=(right_box[0] + right_box[2], centers[right][1]),
            xytext=(left_box[0], centers[left][1]),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#34495e"},
        )
    stop_box, controller_box = boxes["stop, fallback,\nor retry"], boxes["E0 / E1 / E2\ncontroller"]
    axis.annotate(
        "",
        xy=(centers["E0 / E1 / E2\ncontroller"][0], controller_box[1]),
        xytext=(centers["stop, fallback,\nor retry"][0], stop_box[1] + stop_box[3]),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#c0392b", "connectionstyle": "arc3,rad=-0.28"},
    )
    axis.text(3.85, 3.0, "incremental escalation + reuse", color="#c0392b", fontsize=8.5, ha="center")
    axis.text(0.3, 4.85, "Control plane", fontsize=10, weight="bold", color="#2c3e50")
    axis.text(0.3, 3.0, "Memory / generation plane", fontsize=10, weight="bold", color="#2c3e50")
    _save(figure, output, "adaptive_pra_runtime_architecture")


def build_findings(adaptive: dict, systems: dict, output: Path) -> dict:
    frontier = adaptive["frontier"]

    def result(dataset: str, method: str) -> dict:
        return next(row for row in frontier if row["dataset"] == dataset and row["method"] == method)

    qasper_high = result("qasper", "fixed_E2_high")
    qasper_learned = result("qasper", "learned_direct")
    hotpot_high = result("hotpotqa", "fixed_E2_high")
    hotpot_learned = result("hotpotqa", "learned_direct")
    findings = {
        "schema_version": "1.0",
        "study_scope": "frozen_trace_adaptive_control_plus_controlled_systems_prototypes",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "systems_benchmark_device": "cpu",
        },
        "adaptive": adaptive,
        "systems": systems,
        "primary_findings": {
            "qasper_quality_high": qasper_high["quality"],
            "qasper_quality_learned": qasper_learned["quality"],
            "qasper_effort_high": qasper_high["mean_effort_cost"],
            "qasper_effort_learned": qasper_learned["mean_effort_cost"],
            "qasper_effort_reduction": 1.0 - qasper_learned["mean_effort_cost"] / qasper_high["mean_effort_cost"],
            "hotpot_quality_high": hotpot_high["quality"],
            "hotpot_quality_learned": hotpot_learned["quality"],
            "hotpot_effort_high": hotpot_high["mean_effort_cost"],
            "hotpot_effort_learned": hotpot_learned["mean_effort_cost"],
            "interpretation": "The controller saves effort on heterogeneous QASPER but correctly saturates at E2 on uniformly hard HotpotQA; retry is a recovery mechanism, not a guaranteed cost reduction.",
        },
        "claim_boundaries": [
            "Adaptive quality is evaluated on frozen Paper-2.5 routing traces, not a live production scheduler.",
            "Output entropy is calibrated separately on the Paper-3 controlled model and is not sufficient by itself.",
            "Search, gather, page, cache, and batching timings are standalone CPU prototype measurements.",
            "RAG, long-context, and KV-cache rows are controlled matched-budget proxies, not upstream production implementations.",
            "Inherited Qwen serving rows are measured single-request observations; concurrency HBM is extrapolated.",
        ],
    }
    (output / "paper3_5_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    return findings


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    adaptive = run_adaptive_experiment(
        ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/monotonic_adaptive_competition/transition_policy_rows.csv",
        ROOT / "docs/papers/shared/results/paper3_kv_materialization/toy_materialization/toy_materialization_rows.csv",
        output,
    )
    systems = run_systems_benchmarks(
        output,
        ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation/gate3_generation_rows.csv",
    )
    plot_adaptive_frontier(output)
    plot_calibration(output)
    plot_indexed_search(output)
    plot_concurrency(output)
    plot_baselines(output)
    plot_architecture(output)
    return build_findings(adaptive, systems, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.output_dir)
    print(json.dumps(result["primary_findings"], indent=2))
