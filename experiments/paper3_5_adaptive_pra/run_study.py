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
from experiments.paper3_5_adaptive_pra.addon_study import run_addon_studies
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
    figure, axis = plt.subplots(figsize=(11.2, 5.0))
    axis.set_xlim(0, 11.2)
    axis.set_ylim(0, 5.0)
    axis.axis("off")
    boxes = {
        "query / session\nstate": (0.25, 3.45, 1.55, 0.78, "#ecf0f1"),
        "competing query\ninterpretations": (2.05, 3.45, 1.65, 0.78, "#d6eaf8"),
        "direct effort\ncontroller": (3.95, 3.45, 1.55, 0.78, "#d5f5e3"),
        "PRA attempt": (5.75, 3.45, 1.45, 0.78, "#fdebd0"),
        "confidence /\nfailure evidence": (7.45, 3.45, 1.65, 0.78, "#f5eef8"),
        "stop + answer": (9.35, 3.45, 1.5, 0.78, "#d5f5e3"),
        "interpret\nQ regions, F": (2.05, 1.35, 1.65, 0.78, "#d6eaf8"),
        "search\nR, K, H, Bs": (3.95, 1.35, 1.55, 0.78, "#fdebd0"),
        "admit\nBkv, threshold": (5.75, 1.35, 1.45, 0.78, "#f5eef8"),
        "one targeted\ncorrection": (7.45, 1.35, 1.65, 0.78, "#fadbd8"),
    }
    centers = {}
    for label, (x, y, width, height, color) in boxes.items():
        rectangle = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="#2c3e50", linewidth=1.1)
        axis.add_patch(rectangle)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8.8)
        centers[label] = (x + width / 2, y + height / 2)

    top = ["query / session\nstate", "competing query\ninterpretations", "direct effort\ncontroller", "PRA attempt", "confidence /\nfailure evidence", "stop + answer"]
    for left, right in zip(top, top[1:]):
        left_box, right_box = boxes[left], boxes[right]
        axis.annotate("", xy=(right_box[0], centers[right][1]), xytext=(left_box[0] + left_box[2], centers[left][1]), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#34495e"})

    lower = ["interpret\nQ regions, F", "search\nR, K, H, Bs", "admit\nBkv, threshold", "one targeted\ncorrection"]
    for left, right in zip(lower, lower[1:]):
        left_box, right_box = boxes[left], boxes[right]
        axis.annotate("", xy=(right_box[0], centers[right][1]), xytext=(left_box[0] + left_box[2], centers[left][1]), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#34495e"})
    axis.annotate("", xy=(centers["interpret\nQ regions, F"][0], boxes["interpret\nQ regions, F"][1] + boxes["interpret\nQ regions, F"][3]), xytext=(centers["direct effort\ncontroller"][0], boxes["direct effort\ncontroller"][1]), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#34495e"})
    axis.annotate("", xy=(centers["one targeted\ncorrection"][0], boxes["one targeted\ncorrection"][1] + boxes["one targeted\ncorrection"][3]), xytext=(centers["confidence /\nfailure evidence"][0], boxes["confidence /\nfailure evidence"][1]), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#c0392b"})
    axis.annotate("", xy=(boxes["PRA attempt"][0], centers["PRA attempt"][1]), xytext=(centers["one targeted\ncorrection"][0], boxes["one targeted\ncorrection"][1] + boxes["one targeted\ncorrection"][3]), arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#c0392b", "connectionstyle": "arc3,rad=-0.22"})
    axis.text(5.55, 0.62, "bounded retry preserves compatible search and K/V state", color="#c0392b", fontsize=8.5, ha="center")
    axis.text(0.25, 4.65, "Adaptive control endpoint", fontsize=10, weight="bold", color="#2c3e50")
    axis.text(0.25, 2.55, "Factorized action", fontsize=10, weight="bold", color="#2c3e50")
    _save(figure, output, "adaptive_pra_runtime_architecture")


def plot_query_region_results(output: Path) -> None:
    rows = _rows(output / "query_region_layout_rows.csv")
    layouts = ["L0_context_query", "L1_query_context", "L2_context_query_context", "L3_instruction_context_query", "L4_query_long_payload"]
    methods = ["head", "explicit", "structural", "auto_retry"]
    labels = {"head": "head", "explicit": "explicit span", "structural": "structural", "auto_retry": "head + reinterpret"}
    colors = {"head": "#7f8c8d", "explicit": "#2c3e50", "structural": "#16a085", "auto_retry": "#c0392b"}
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    for method in methods:
        recall, effort = [], []
        for layout in layouts:
            selected = [row for row in rows if row["method"] == method and row["layout"] == layout]
            recall.append(sum(float(row["root_recall_at_1"]) for row in selected) / len(selected))
            effort.append(sum(float(row["search_effort"]) for row in selected) / len(selected))
        axes[0].plot(range(len(layouts)), recall, marker="o", color=colors[method], label=labels[method])
        axes[1].plot(range(len(layouts)), effort, marker="o", color=colors[method], label=labels[method])
    short = ["C+Q", "Q+C", "C1+Q+C2", "I+C+Q", "Q+long C"]
    for axis in axes:
        axis.set_xticks(range(len(layouts)), short, rotation=18)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("root recall at 1")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("mean lexical comparisons")
    axes[0].legend(fontsize=7)
    _save(figure, output, "query_region_layout_frontier")

    displacement = _rows(output / "query_region_head_displacement.csv")
    figure, axis = plt.subplots(figsize=(6.8, 4.0))
    for method in ("head", "explicit", "structural"):
        selected = sorted([row for row in displacement if row["method"] == method], key=lambda row: float(row["actual_tokens_after_query"]))
        axis.plot([float(row["actual_tokens_after_query"]) for row in selected], [float(row["root_recall_at_1"]) for row in selected], marker="o", label=labels.get(method, method), color=colors[method])
    axis.set_xscale("symlog", linthresh=32)
    axis.set_xlabel("tokens serialized after query")
    axis.set_ylabel("root recall at 1")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save(figure, output, "query_region_displacement")


def plot_router_variants(output: Path) -> None:
    findings = json.loads((output / "router_findings.json").read_text(encoding="utf-8"))
    rows = findings["summary"]
    labels = {"R0_profile": "R0 profiles", "R1_feature_mlp": "R1 feature MLP", "R2_encoder_mlp": "R2 semantic MLP", "R3A_autoregressive": "R3A autoregressive"}
    colors = {"R0_profile": "#2c3e50", "R1_feature_mlp": "#3498db", "R2_encoder_mlp": "#16a085", "R3A_autoregressive": "#8e44ad"}
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for row in rows:
        variant = row["variant"]
        axis.scatter(float(row["cost"]), float(row["quality"]), s=65, color=colors[variant])
        axis.annotate(labels[variant], (float(row["cost"]), float(row["quality"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("mean abstract effort")
    axis.set_ylabel("complete evidence-chain rate")
    axis.set_ylim(0.55, 0.64)
    axis.grid(alpha=0.25)
    _save(figure, output, "router_variant_quality_cost")


def build_findings(adaptive: dict, systems: dict, addons: dict, output: Path) -> dict:
    frontier = adaptive["frontier"]

    def result(dataset: str, method: str) -> dict:
        return next(row for row in frontier if row["dataset"] == dataset and row["method"] == method)

    qasper_high = result("qasper", "fixed_E2_high")
    qasper_learned = result("qasper", "learned_direct")
    hotpot_high = result("hotpotqa", "fixed_E2_high")
    hotpot_learned = result("hotpotqa", "learned_direct")
    factorized_path = output / "paper3_5_next_findings.json"
    factorized = json.loads(factorized_path.read_text(encoding="utf-8")) if factorized_path.exists() else None
    findings = {
        "schema_version": "1.0",
        "study_scope": "adaptive_initial_control_query_interpretation_factorized_actions_and_bounded_retry",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "systems_benchmark_device": "cpu",
        },
        "adaptive": adaptive,
        "systems": systems,
        "query_regions": addons["query_regions"],
        "router_variants": addons["router_variants"],
        "factorized_control": factorized,
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
            "Query-region results are deterministic matched lexical controls, not natural-prompt semantic understanding.",
            "R2 uses a dependency-free hashing encoder interface rather than a benchmarked pretrained NLP encoder.",
            "Factorized control replays frozen native scores and does not impute generated-answer quality.",
            "The targeted corrective-action audit is an evaluator-side upper bound, not a learned online retry policy.",
            "Kernel, cache, batching, and serving-engine work is inherited handoff evidence for Papers 5.5 and 6.",
        ],
    }
    if factorized is not None:
        findings["primary_findings"]["factorized"] = factorized["datasets"]
        findings["primary_findings"]["interpretation"] = (
            "Profiles remain the safest learned controller, but the factorized oracle shows that "
            "profile coupling hides substantial lower-cost actions. Bounded retry has measured "
            "correction headroom; learning an oracle-free targeted correction remains open."
        )
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
    addons = run_addon_studies(
        ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/monotonic_adaptive_competition/transition_policy_rows.csv",
        output,
    )
    plot_adaptive_frontier(output)
    plot_calibration(output)
    plot_indexed_search(output)
    plot_concurrency(output)
    plot_baselines(output)
    plot_architecture(output)
    plot_query_region_results(output)
    plot_router_variants(output)
    return build_findings(adaptive, systems, addons, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.output_dir)
    print(json.dumps(result["primary_findings"], indent=2))
