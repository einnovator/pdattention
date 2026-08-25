"""Summarize Paper 6.5 M6.5/M7 semantic-hard discovery experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
M6_LABELS = {
    "P0_token": "Token",
    "P1_bm25": "BM25",
    "P2_dictionary": "Dictionary",
    "P3_tags": "Tags",
    "P5_english_embedding": "English embedding",
    "P6_multilingual_embedding": "Multilingual embedding",
    "P8_lexical_dictionary_embedding": "Hybrid",
    "P10_staged_external": "Staged",
}
M7_LABELS = {
    "external_lexical_bm25": "BM25",
    "external_dictionary": "Tags",
    "external_compact_embedding": "English embedding",
    "external_hybrid_p8": "Hybrid",
    "native_mean_k": "Native mean K",
    "native_token_qk": "Native token QK",
    "paper2_8_rank16_zero_shot": "P2.8 rank-16",
    "paper2_8_rank8_centroids_zero_shot": "P2.8 rank-8/8c",
}
COLORS = {
    "Token": "#6b7280",
    "BM25": "#2563eb",
    "Dictionary": "#0f766e",
    "Tags": "#15803d",
    "English embedding": "#c2410c",
    "Multilingual embedding": "#9333ea",
    "Hybrid": "#be123c",
    "Staged": "#111827",
    "Native mean K": "#7c3aed",
    "Native token QK": "#a855f7",
    "P2.8 rank-16": "#d97706",
    "P2.8 rank-8/8c": "#eab308",
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: str) -> bool:
    return value.lower() == "true"


def _summary(rows: Sequence[dict[str, str]], mode: str) -> dict[str, object]:
    selected = [row for row in rows if row["mode"] == mode]
    latencies = np.array([float(row["routing_seconds"]) for row in selected]) * 1_000
    return {
        "mode": mode,
        "label": M6_LABELS.get(mode, M7_LABELS.get(mode, mode)),
        "queries": len(selected),
        "top1": np.mean([_truth(row["top1_correct"]) for row in selected]),
        "mrr": np.mean([float(row["mrr"]) for row in selected]),
        "recall_at_3": np.mean([float(row["recall_at_3"]) for row in selected]),
        "recall_at_5": np.mean([float(row["recall_at_5"]) for row in selected]),
        "median_latency_ms": np.median(latencies),
        "p95_latency_ms": np.quantile(latencies, 0.95),
        "index_bytes": max(int(row["index_bytes"]) for row in selected),
        "model_bytes": max(int(row["model_bytes"]) for row in selected),
    }


def _by(rows: Sequence[dict[str, str]], modes: Iterable[str], field: str) -> list[dict[str, object]]:
    output = []
    for mode in modes:
        values = sorted({row[field] for row in rows if row["mode"] == mode})
        for value in values:
            selected = [row for row in rows if row["mode"] == mode and row[field] == value]
            output.append({
                "mode": mode,
                "label": M6_LABELS.get(mode, M7_LABELS.get(mode, mode)),
                field: value,
                "queries": len(selected),
                "top1": np.mean([_truth(row["top1_correct"]) for row in selected]),
                "mrr": np.mean([float(row["mrr"]) for row in selected]),
                "recall_at_3": np.mean([float(row["recall_at_3"]) for row in selected]),
            })
    return output


def _paired_effect(
    rows: Sequence[dict[str, str]], left: str, right: str, *, samples: int = 10_000
) -> dict[str, object]:
    by_mode = {
        mode: {row["query_id"]: float(_truth(row["top1_correct"])) for row in rows if row["mode"] == mode}
        for mode in (left, right)
    }
    identities = sorted(set(by_mode[left]) & set(by_mode[right]))
    effects = np.array([by_mode[left][identity] - by_mode[right][identity] for identity in identities])
    rng = np.random.default_rng(65007)
    draws = effects[rng.integers(0, len(effects), size=(samples, len(effects)))].mean(axis=1)
    wins = int(np.sum(effects > 0))
    losses = int(np.sum(effects < 0))
    discordant = wins + losses
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(wins, losses) + 1)) / 2**discordant
        exact_p = min(1.0, 2 * tail)
    else:
        exact_p = 1.0
    return {
        "left": left,
        "right": right,
        "identities": len(identities),
        "top1_effect": effects.mean(),
        "bootstrap_ci_low": np.quantile(draws, 0.025),
        "bootstrap_ci_high": np.quantile(draws, 0.975),
        "discordant_left_wins": wins,
        "discordant_right_wins": losses,
        "exact_two_sided_p": exact_p,
    }


def _calibration(rows: Sequence[dict[str, str]]) -> dict[str, object]:
    selected = [row for row in rows if row["mode"] == "P10_staged_external" and row["split"] == "test"]
    actioned = [row for row in selected if row["decision"] == "select"]
    probabilities = np.array([float(row["calibrated_confidence"]) for row in selected])
    outcomes = np.array([float(_truth(row["top1_correct"])) for row in selected])
    ece = 0.0
    for low in np.linspace(0, 0.9, 10):
        mask = (probabilities >= low) & (probabilities < low + 0.1 + (1e-9 if low == 0.9 else 0))
        if mask.any():
            ece += mask.mean() * abs(probabilities[mask].mean() - outcomes[mask].mean())
    unsafe_actions = 0
    for row in actioned:
        unsafe = set(filter(None, row["unsafe_tools"].split("|")))
        unsafe_actions += row["top1_tool"] in unsafe
    correct_actions = sum(_truth(row["top1_correct"]) for row in actioned)
    return {
        "queries": len(selected),
        "select": len(actioned),
        "ask": sum(row["decision"] == "ask" for row in selected),
        "abstain": sum(row["decision"] == "abstain" for row in selected),
        "coverage": len(actioned) / len(selected),
        "selective_accuracy": correct_actions / max(len(actioned), 1),
        "actionable_accuracy": correct_actions / len(selected),
        "false_action_rate": (len(actioned) - correct_actions) / len(selected),
        "unsafe_action_rate": unsafe_actions / len(selected),
        "brier": np.mean((probabilities - outcomes) ** 2),
        "ece_10": ece,
    }


def _overlap_rows(rows: Sequence[dict[str, str]], modes: Iterable[str]) -> list[dict[str, object]]:
    bins = ((-1.0, 0.0, "zero"), (0.0, 0.1, "low"), (0.1, 0.25, "medium"), (0.25, 1.0, "high"))
    output = []
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        for low, high, label in bins:
            selected = [row for row in mode_rows if low < float(row["token_overlap"]) <= high]
            output.append({
                "mode": mode,
                "label": M6_LABELS[mode],
                "overlap_bin": label,
                "mean_token_overlap": np.mean([float(row["token_overlap"]) for row in selected]),
                "mean_lexical_distance": np.mean([1 - float(row["token_overlap"]) for row in selected]),
                "queries": len(selected),
                "top1": np.mean([_truth(row["top1_correct"]) for row in selected]),
            })
    return output


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    })


def _save(fig, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.pdf")
    fig.savefig(output_dir / f"{name}.png", dpi=180)
    plt.close(fig)


def _plots(m6_rows, m7_rows, by_level, by_language, overlap_rows, output_dir: Path) -> None:
    _style()
    plot_modes = ("P1_bm25", "P2_dictionary", "P3_tags", "P5_english_embedding", "P8_lexical_dictionary_embedding")
    levels = [f"H{index}" for index in range(6)]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for mode in plot_modes:
        data = {row["hardness_level"]: row for row in by_level if row["mode"] == mode}
        label = M6_LABELS[mode]
        ax.plot(levels, [data[level]["top1"] for level in levels], marker="o", label=label, color=COLORS[label])
    ax.set(xlabel="Semantic-hardness stratum", ylabel="Top-1 accuracy", ylim=(-0.03, 1.03))
    ax.legend(ncol=3, frameon=False)
    _save(fig, output_dir, "semantic_quality_by_hardness")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    bin_order = ["high", "medium", "low", "zero"]
    for mode in ("P1_bm25", "P2_dictionary", "P5_english_embedding", "P8_lexical_dictionary_embedding"):
        data = {row["overlap_bin"]: row for row in overlap_rows if row["mode"] == mode}
        label = M6_LABELS[mode]
        ax.plot(
            [data[key]["mean_lexical_distance"] for key in bin_order],
            [data[key]["top1"] for key in bin_order],
            marker="o",
            label=label,
            color=COLORS[label],
        )
    ax.set(xlabel="Lexical distance (1 - token overlap)", ylabel="Top-1 accuracy", ylim=(-0.03, 1.03))
    ax.legend(ncol=2, frameon=False)
    _save(fig, output_dir, "semantic_quality_vs_lexical_distance")

    languages = ("es", "fr", "pt")
    language_modes = ("P1_bm25", "P2_dictionary", "P3_tags", "P6_multilingual_embedding", "P8_lexical_dictionary_embedding")
    x = np.arange(len(languages))
    width = 0.15
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    for index, mode in enumerate(language_modes):
        data = {row["language"]: row for row in by_language if row["mode"] == mode}
        label = M6_LABELS[mode]
        ax.bar(x + (index - 2) * width, [data[lang]["top1"] for lang in languages], width, label=label, color=COLORS[label])
    ax.set(xticks=x, xticklabels=["Spanish", "French", "Portuguese"], ylabel="Top-1 accuracy", ylim=(0, 1.05))
    ax.legend(ncol=3, frameon=False, fontsize=8)
    _save(fig, output_dir, "semantic_multilingual_quality")

    m7_modes = tuple(M7_LABELS)
    summaries = [_summary(m7_rows, mode) for mode in m7_modes]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    offsets = {
        "Native mean K": (4, 13),
        "Native token QK": (4, -13),
        "P2.8 rank-16": (4, 2),
        "P2.8 rank-8/8c": (4, 2),
    }
    for row in summaries:
        ax.scatter(row["median_latency_ms"], row["top1"], s=42, color=COLORS[row["label"]])
        ax.annotate(
            row["label"],
            (row["median_latency_ms"], row["top1"]),
            xytext=offsets.get(row["label"], (4, 3)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set(xlabel="Median routing latency (ms, log scale)", ylabel="Top-1 accuracy", ylim=(-0.03, 1.03))
    _save(fig, output_dir, "semantic_quality_latency_frontier")

    staged = [row for row in m6_rows if row["mode"] == "P10_staged_external"]
    stages = sorted({row["selected_stage"] for row in staged})
    counts = defaultdict(Counter)
    totals = Counter()
    for row in staged:
        counts[row["hardness_level"]][row["selected_stage"]] += 1
        totals[row["hardness_level"]] += 1
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    bottom = np.zeros(len(levels))
    stage_colors = ("#2563eb", "#0f766e", "#c2410c", "#9ca3af")
    stage_labels = {
        "P1_bm25": "Lexical/BM25",
        "P4_lexical_dictionary_tags": "Dictionary/tags",
        "P8_lexical_dictionary_embedding": "Embedding hybrid",
        "ask": "ASK",
    }
    for stage, color in zip(stages, stage_colors):
        values = np.array([counts[level][stage] / totals[level] for level in levels])
        ax.bar(levels, values, bottom=bottom, label=stage_labels[stage], color=color)
        bottom += values
    ax.set(xlabel="Semantic-hardness stratum", ylabel="Fraction of queries", ylim=(0, 1))
    ax.legend(ncol=2, frameon=False, fontsize=8)
    _save(fig, output_dir, "semantic_staged_escalation")


def _macro(name: str, value: object) -> str:
    if isinstance(value, float):
        rendered = f"{value:.3f}"
        if rendered.startswith("0."):
            rendered = rendered[1:]
    else:
        rendered = str(value)
    return f"\\newcommand{{\\{name}}}{{{rendered}}}"


def run(args) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    m6 = [row for row in _read(args.m6_5_dir / "semantic_hardness_rows.csv") if row["split"] == "test"]
    m7 = _read(args.m7_dir / "m7_semantic_hard_rows.csv")
    m6_modes = [mode for mode in M6_LABELS if any(row["mode"] == mode for row in m6)]
    m7_modes = [mode for mode in M7_LABELS if any(row["mode"] == mode for row in m7)]
    policy_summary = [_summary(m6, mode) for mode in m6_modes]
    native_summary = [_summary(m7, mode) for mode in m7_modes]
    by_level = _by(m6, m6_modes, "hardness_level")
    by_language = _by(m6, m6_modes, "language")
    overlap = _overlap_rows(m6, ("P1_bm25", "P2_dictionary", "P5_english_embedding", "P8_lexical_dictionary_embedding"))
    calibration = _calibration(m6)
    effects = [
        _paired_effect(m6, "P3_tags", "P1_bm25"),
        _paired_effect(m6, "P8_lexical_dictionary_embedding", "P1_bm25"),
        _paired_effect(m7, "external_dictionary", "native_mean_k"),
        _paired_effect(m7, "external_dictionary", "native_token_qk"),
        _paired_effect(m7, "external_dictionary", "paper2_8_rank16_zero_shot"),
    ]
    _write(args.output_dir / "semantic_policy_summary.csv", policy_summary)
    _write(args.output_dir / "semantic_native_summary.csv", native_summary)
    _write(args.output_dir / "semantic_quality_by_hardness.csv", by_level)
    _write(args.output_dir / "semantic_quality_by_language.csv", by_language)
    _write(args.output_dir / "semantic_quality_by_overlap.csv", overlap)
    _write(args.output_dir / "semantic_paired_effects.csv", effects)
    _write(args.output_dir / "semantic_selective_calibration.csv", [calibration])
    _plots(m6, m7, by_level, by_language, overlap, args.output_dir)

    policy = {row["mode"]: row for row in policy_summary}
    native = {row["mode"]: row for row in native_summary}
    lexical_effect, hybrid_effect = effects[:2]
    findings = {
        "benchmark_test_queries": len({row["query_id"] for row in m6}),
        "best_fixed_external_mode": max(policy_summary, key=lambda row: row["top1"])["mode"],
        "best_fixed_external_top1": max(row["top1"] for row in policy_summary),
        "bm25_top1": policy["P1_bm25"]["top1"],
        "hybrid_top1": policy["P8_lexical_dictionary_embedding"]["top1"],
        "staged": calibration,
        "native_results": native_summary,
        "paired_effects": effects,
        "native_training_gate_open": False,
        "native_training_gate_reason": "All frozen native modes remain at catalog chance and provide no positive diagnostic for tool-specific training.",
    }
    (args.output_dir / "semantic_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    macros = [
        _macro("PaperSixFiveSemanticTestQueries", findings["benchmark_test_queries"]),
        _macro("PaperSixFiveSemanticBMTopOne", policy["P1_bm25"]["top1"]),
        _macro("PaperSixFiveSemanticTagsTopOne", policy["P3_tags"]["top1"]),
        _macro("PaperSixFiveSemanticHybridTopOne", policy["P8_lexical_dictionary_embedding"]["top1"]),
        _macro("PaperSixFiveSemanticHybridRecallThree", policy["P8_lexical_dictionary_embedding"]["recall_at_3"]),
        _macro("PaperSixFiveSemanticCoverage", calibration["coverage"]),
        _macro("PaperSixFiveSemanticSelectiveAccuracy", calibration["selective_accuracy"]),
        _macro("PaperSixFiveSemanticFalseAction", calibration["false_action_rate"]),
        _macro("PaperSixFiveSemanticBrier", calibration["brier"]),
        _macro("PaperSixFiveSemanticECE", calibration["ece_10"]),
        _macro("PaperSixFiveSemanticTagGain", lexical_effect["top1_effect"]),
        _macro("PaperSixFiveSemanticTagGainLow", lexical_effect["bootstrap_ci_low"]),
        _macro("PaperSixFiveSemanticTagGainHigh", lexical_effect["bootstrap_ci_high"]),
        _macro("PaperSixFiveSemanticTagGainP", lexical_effect["exact_two_sided_p"]),
        _macro("PaperSixFiveSemanticHybridGain", hybrid_effect["top1_effect"]),
        _macro("PaperSixFiveMSevenMeanTopOne", native["native_mean_k"]["top1"]),
        _macro("PaperSixFiveMSevenTokenTopOne", native["native_token_qk"]["top1"]),
        _macro("PaperSixFiveMSevenRankSixteenTopOne", native["paper2_8_rank16_zero_shot"]["top1"]),
        _macro("PaperSixFiveMSevenRankEightTopOne", native["paper2_8_rank8_centroids_zero_shot"]["top1"]),
    ]
    (args.output_dir / "generated_semantic_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--m6-5-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/m6_5_semantic_hard",
    )
    parser.add_argument(
        "--m7-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/m7_semantic_hard_native",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/semantic_summary",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
