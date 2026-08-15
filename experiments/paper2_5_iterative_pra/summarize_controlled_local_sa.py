"""Summarize controlled LocalSA/PRA results and render Paper 2.5 figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt

from experiments.paper2_5_iterative_pra.run_controlled_local_sa import (
    DEFAULT_OUTPUT,
    _read_csv,
    _write_csv,
)


WINDOW_ORDER = {"w16": 0, "w32": 1, "w64": 2, "w128": 3, "global": 4}


def _number(value) -> float:
    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return float(text == "true")
    return float(value)


def _mean_ci(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _seed_mean(rows: list[dict], group_fields: tuple[str, ...], metric: str) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in group_fields), []).append(_number(row[metric]))
    return [
        {
            **dict(zip(group_fields, key)),
            f"{metric}_mean": _mean_ci(values)[0],
            f"{metric}_ci95": _mean_ci(values)[1],
            "n": len(values),
        }
        for key, values in groups.items()
    ]


def _plot_window_metric(rows: list[dict], metric: str, ylabel: str, path: Path) -> None:
    final_layer = max(int(row["layer_id"]) for row in rows)
    selected = [row for row in rows if int(row["layer_id"]) == final_layer]
    summary = _seed_mean(selected, ("window",), metric)
    summary.sort(key=lambda row: WINDOW_ORDER[row["window"]])
    fig, axis = plt.subplots(figsize=(5.2, 3.3))
    axis.errorbar(
        [row["window"].replace("w", "W=") for row in summary],
        [row[f"{metric}_mean"] for row in summary],
        yerr=[row[f"{metric}_ci95"] for row in summary],
        marker="o",
        capsize=3,
        color="#155e75",
    )
    axis.set_xlabel("training attention window")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _recovery_depth_rows(raw_edges: list[dict]) -> list[dict]:
    """Find the first layer whose complete evidence path survives at R@4."""
    examples: dict[tuple[str, str, str], list[dict]] = {}
    for row in raw_edges:
        examples.setdefault((row["window"], row["seed"], row["example_id"]), []).append(row)
    output = []
    for (window, seed, example_id), rows in examples.items():
        layers = sorted({int(row["layer_id"]) for row in rows})
        recovered = [
            layer
            for layer in layers
            if all(
                _number(row["recall_at_4"]) == 1.0
                for row in rows
                if int(row["layer_id"]) == layer
            )
        ]
        first = min(recovered) + 1 if recovered else len(layers) + 1
        output.append(
            {
                "window": window,
                "seed": seed,
                "example_id": example_id,
                "depth": rows[0]["depth"],
                "minimum_native_recovery_depth": first,
                "unreachable_within_model": int(not recovered),
                "recovery_depth_cap": len(layers) + 1,
            }
        )
    return output


def _plot_recovery_depth(rows: list[dict], path: Path) -> None:
    per_seed = _seed_mean(
        rows,
        ("window", "seed"),
        "minimum_native_recovery_depth",
    )
    normalized = [
        {
            "window": row["window"],
            "minimum_native_recovery_depth": row[
                "minimum_native_recovery_depth_mean"
            ],
        }
        for row in per_seed
    ]
    summary = _seed_mean(
        normalized,
        ("window",),
        "minimum_native_recovery_depth",
    )
    summary.sort(key=lambda row: WINDOW_ORDER[row["window"]])
    fig, axis = plt.subplots(figsize=(5.2, 3.3))
    axis.errorbar(
        [row["window"].replace("w", "W=") for row in summary],
        [row["minimum_native_recovery_depth_mean"] for row in summary],
        yerr=[row["minimum_native_recovery_depth_ci95"] for row in summary],
        marker="o",
        capsize=3,
        color="#6d28d9",
    )
    axis.set_xlabel("training attention window")
    axis.set_ylabel("minimum complete-path recovery layer")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_edge_recall_curves(rows: list[dict], path: Path) -> None:
    final_layer = max(int(row["layer_id"]) for row in rows)
    selected = [row for row in rows if int(row["layer_id"]) == final_layer]
    fig, axis = plt.subplots(figsize=(5.5, 3.4))
    ordered_windows = sorted({row["window"] for row in selected}, key=WINDOW_ORDER.get)
    for cutoff in (1, 2, 4, 6, 8):
        values = []
        for window in ordered_windows:
            group = [
                _number(row[f"edge_recall_at_{cutoff}"])
                for row in selected
                if row["window"] == window
            ]
            values.append(statistics.fmean(group))
        axis.plot(
            [window.replace("w", "W=") for window in ordered_windows],
            values,
            marker="o",
            label=f"R@{cutoff}",
        )
    axis.set_xlabel("training attention window")
    axis.set_ylabel("native edge recall")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_layerwise(
    topology: list[dict],
    context: list[dict],
    path: Path,
) -> None:
    available = {row["window"] for row in topology}
    representatives = [window for window in ("w16", "global") if window in available]
    if not representatives:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for window in representatives:
        topo = [row for row in topology if row["window"] == window]
        context_rows = [row for row in context if row["window"] == window]
        layers = sorted({int(row["layer_id"]) for row in topo})
        axes[0].plot(
            layers,
            [
                statistics.fmean(
                    _number(row["edge_recall_at_4"])
                    for row in topo
                    if int(row["layer_id"]) == layer
                )
                for layer in layers
            ],
            marker="o",
            label=window,
        )
        axes[1].plot(
            layers,
            [
                statistics.fmean(
                    _number(row["restricted_context_dependence"])
                    for row in context_rows
                    if int(row["layer_id"]) == layer
                )
                for layer in layers
            ],
            marker="o",
            label=window,
        )
    axes[0].set_ylabel("edge R@4")
    axes[1].set_ylabel("restricted-context dependence")
    for axis in axes:
        axis.set_xlabel("decoder layer")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_spacing(rows: list[dict], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["condition"] in {"spacing_1", "spacing_2", "spacing_4", "spacing_8"}
        and int(float(row["depth"])) <= 4
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for window in sorted({row["window"] for row in selected}, key=WINDOW_ORDER.get):
        window_rows = [row for row in selected if row["window"] == window]
        points = []
        for condition in ("spacing_1", "spacing_2", "spacing_4", "spacing_8"):
            values = [_number(row["correct"]) for row in window_rows if row["condition"] == condition]
            displacement = [
                _number(row["mean_intervention_state_displacement"])
                for row in window_rows
                if row["condition"] == condition
            ]
            if values:
                points.append((int(condition.split("_")[1]), statistics.fmean(values), statistics.fmean(displacement)))
        if points:
            axes[0].plot([p[0] for p in points], [p[1] for p in points], marker="o", label=window)
            axes[1].plot([p[0] for p in points], [p[2] for p in points], marker="o", label=window)
    axes[0].set_ylabel("answer accuracy")
    axes[1].set_ylabel("query-state displacement")
    for axis in axes:
        axis.set_xlabel("PRA spacing")
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pra_gain(rows: list[dict], path: Path) -> None:
    compact = [row for row in rows if int(float(row["depth"])) <= 4]
    per_seed: dict[tuple[str, str, str], list[float]] = {}
    for row in compact:
        per_seed.setdefault((row["window"], row["seed"], row["condition"]), []).append(_number(row["correct"]))
    gains: dict[str, list[float]] = {}
    for window in {key[0] for key in per_seed}:
        for seed in {key[1] for key in per_seed if key[0] == window}:
            one = per_seed.get((window, seed, "one_shot"))
            iterative = per_seed.get((window, seed, "iterative_matched"))
            if one and iterative:
                gains.setdefault(window, []).append(statistics.fmean(iterative) - statistics.fmean(one))
    ordered = sorted(gains, key=WINDOW_ORDER.get)
    means_ci = [_mean_ci(gains[window]) for window in ordered]
    fig, axis = plt.subplots(figsize=(5.2, 3.3))
    axis.axhline(0.0, color="#444444", linewidth=0.8)
    axis.errorbar(
        [window.replace("w", "W=") for window in ordered],
        [value[0] for value in means_ci],
        yerr=[value[1] for value in means_ci],
        marker="o",
        capsize=3,
        color="#9f1239",
    )
    axis.set_xlabel("training attention window")
    axis.set_ylabel("matched iterative minus one-shot accuracy")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _export_inherited_pretrained(output_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    shared = output_dir.parent.parent / "paper2_hf"
    gemma_json = shared / "gemma3_1b/gemma3_1b_summary.json"
    gemma_rows = []
    if gemma_json.exists():
        gemma = json.loads(gemma_json.read_text(encoding="utf-8"))
        for dataset, metrics in gemma["five_seed_identity_recall"].items():
            gemma_rows.append(
                {
                    "architecture": "Gemma-3-1B-IT",
                    "dataset": dataset,
                    "native_attention": "five local layers then one global layer",
                    "routing_mrr": metrics["MRR"]["mean"],
                    "routing_recall_at_16": metrics["R@16"]["mean"],
                    "baseline_f1": gemma["no_memory_f1"],
                    "pra_f1": gemma["learned_f1"],
                    "pra_f1_gain": gemma["learned_f1"] - gemma["no_memory_f1"],
                    "best_pra_depth_pattern": "global slots 17,23",
                    "source": "inherited frozen-backbone Paper 2 artifact",
                }
            )
    _write_csv(output_dir / "gemma_bridge_results.csv", gemma_rows)

    llama_source = shared / "productization/cross_family_productization.csv"
    llama_rows = []
    qwen_rows = []
    for row in _read_csv(llama_source):
        if "Llama" in row.get("model", ""):
            llama_rows.append(
                {
                    "architecture": row["model"],
                    "dataset": "qasper",
                    "native_attention": "global",
                    "routing_recall_at_5_percent": row["qasper_r5_mean"],
                    "routing_recall_at_20_percent": row["qasper_r20_mean"],
                    "routing_auc_0_30": row["qasper_auc_0_30_mean"],
                    "output_f1": row["demo_routed_f1"],
                    "best_pra_depth_pattern": "inherited sparse late",
                    "source": "SmolLM2 Llama-family control; not Llama 3.x",
                }
            )
        if row.get("model") == "Qwen3-0.6B":
            qwen_rows.append(
                {
                    "architecture": row["model"],
                    "dataset": "qasper",
                    "native_attention": "global",
                    "routing_recall_at_5_percent": row["qasper_r5_mean"],
                    "routing_recall_at_20_percent": row["qasper_r20_mean"],
                    "routing_auc_0_30": row["qasper_auc_0_30_mean"],
                    "output_f1": row["demo_routed_f1"],
                    "best_pra_depth_pattern": "inherited sparse late",
                    "source": "inherited frozen-backbone Paper 2 artifact",
                }
            )
    _write_csv(output_dir / "llama_replication_results.csv", llama_rows)
    return gemma_rows, llama_rows, qwen_rows


def _cross_architecture(
    output_dir: Path,
    topology: list[dict],
    pra: list[dict],
    gemma: list[dict],
    llama: list[dict],
    qwen: list[dict],
) -> None:
    rows = []
    if topology:
        final_layer = max(int(row["layer_id"]) for row in topology)
        for window in ("w16", "w32", "w64", "w128", "global"):
            selected = [
                row for row in topology
                if row["window"] == window and int(row["layer_id"]) == final_layer
            ]
            if not selected:
                continue
            one = [row for row in pra if row["window"] == window and row["condition"] == "one_shot" and int(float(row["depth"])) <= 4]
            iterative = [row for row in pra if row["window"] == window and row["condition"] == "iterative_matched" and int(float(row["depth"])) <= 4]
            gain = (
                statistics.fmean(_number(row["correct"]) for row in iterative)
                - statistics.fmean(_number(row["correct"]) for row in one)
                if one and iterative else None
            )
            rows.append(
                {
                    "architecture": f"Controlled {window}",
                    "native_attention": "global" if window == "global" else "local",
                    "native_edge_recall_at_4": statistics.fmean(_number(row["edge_recall_at_4"]) for row in selected),
                    "shortcut_rate": statistics.fmean(_number(row["shortcut_rate"]) for row in selected),
                    "iterative_pra_accuracy_gain": gain,
                    "best_pra_depth_pattern": "measured spacing sweep",
                }
            )
    for row in gemma:
        if row["dataset"] == "combined":
            rows.append(
                {
                    "architecture": row["architecture"],
                    "native_attention": row["native_attention"],
                    "native_edge_recall_at_4": "not matched",
                    "shortcut_rate": "not matched",
                    "iterative_pra_accuracy_gain": row["pra_f1_gain"],
                    "best_pra_depth_pattern": row["best_pra_depth_pattern"],
                }
            )
    for row in [*llama, *qwen]:
        rows.append(
            {
                "architecture": row["architecture"],
                "native_attention": row["native_attention"],
                "native_edge_recall_at_4": "not matched",
                "shortcut_rate": "not matched",
                "iterative_pra_accuracy_gain": "not measured",
                "best_pra_depth_pattern": row["best_pra_depth_pattern"],
            }
        )
    _write_csv(output_dir / "cross_architecture_summary.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    topology = _read_csv(args.output_dir / "receptive_field_topology.csv")
    raw_topology = _read_csv(args.output_dir / "receptive_field_topology_rows.csv")
    context = _read_csv(args.output_dir / "layer_contextualization_by_window.csv")
    pra = _read_csv(args.output_dir / "local_pra_one_shot_iterative.csv")
    if topology:
        _plot_window_metric(topology, "edge_recall_at_4", "native edge R@4", figures / "edge_recall_by_window.png")
        _plot_window_metric(topology, "shortcut_rate", "shortcut rate", figures / "shortcut_rate_by_window.png")
        _plot_window_metric(topology, "complete_path_survival_at_4", "complete path survival @4", figures / "path_survival_by_window.png")
        _plot_edge_recall_curves(topology, figures / "edge_recall_curves_by_window.png")
        _plot_layerwise(topology, context, figures / "layerwise_topology_context.png")
    if raw_topology:
        recovery = _recovery_depth_rows(raw_topology)
        _write_csv(args.output_dir / "native_recovery_depth.csv", recovery)
        _plot_recovery_depth(recovery, figures / "native_recovery_depth_by_window.png")
    if pra:
        _plot_pra_gain(pra, figures / "iterative_gain_by_window.png")
        _plot_spacing(pra, figures / "quality_and_state_by_spacing.png")
    gemma, llama, qwen = _export_inherited_pretrained(args.output_dir)
    _cross_architecture(args.output_dir, topology, pra, gemma, llama, qwen)
    print(f"wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
