"""Summarize the corrected native-consumption ladder for Paper 8."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper8_tasks/production_pra/native_bug_ablation"
FIGURES = ROOT / "docs/papers/shared/figures/paper8_tasks"
GENERATED = ROOT / "docs/papers/shared/results/paper8_tasks/production_pra/generated_native_bug_ablation.tex"


def _rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows, field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def main() -> None:
    rows = _rows("native_full_scope_parity.csv") + _rows("native_progressive_removal.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    accuracy = {name: 100.0 * _mean(values, "answer_correct") for name, values in grouped.items()}
    tokens = {
        name: _mean(values, "unique_native_tokens")
        for name, values in grouped.items()
        if values[0].get("unique_native_tokens") not in {None, ""}
    }
    prefix = _rows("native_prefix_equivalence.csv")
    e0 = next(row for row in prefix if row["condition"] == "E0_ONE_FULL_REFERENCE")
    kv = [row for row in prefix if row["condition"] == "KV_IDENTITY"]
    lifetime = _rows("native_decode_lifetime.csv")
    dedup = _rows("native_interval_dedup.csv")

    order = [
        "VISIBLE_FULL_SESSION",
        "NATIVE_FULL_SESSION",
        "VISIBLE_FULL_TASK_SCOPE",
        "NATIVE_FULL_TASK_SCOPE",
        "A2_SELECTED_RECORDS_FULL",
        "LAYER_ABLATION_FULL_SELECTED_SPARSE4",
        "LAYER_ABLATION_FULL_SELECTED_LATE1",
        "A6_PAPER3_EVIDENCE_RADIUS0_ALL28",
    ]
    labels = [
        "Visible session",
        "Native session",
        "Visible task scope",
        "Native task scope, all 28",
        "Selected full, all 28",
        "Selected full, sparse 4",
        "Selected full, late 1",
        "Paper 3 radius 0",
    ]
    colors = ["#687386", "#25364a", "#40916c", "#1b7f79", "#2a9d8f", "#d97706", "#b45309", "#9b2226"]
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    axes[0].barh(range(len(order)), [accuracy[name] for name in order], color=colors)
    axes[0].set_yticks(range(len(order)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Answer accuracy (%)")
    axes[0].set_xlim(0, 82)
    axes[0].grid(axis="x", alpha=0.25)

    window_order = [512, 256, 128, 96, 64, 32]
    names = [f"A4_A5_SELECTED_WINDOW_{width}" for width in window_order]
    coordinates = defaultdict(list)
    for width, name in zip(window_order, names):
        coordinates[(round(tokens[name], 3), round(accuracy[name], 3))].append(width)
    for (x_value, y_value), widths in coordinates.items():
        axes[1].scatter([x_value], [y_value], color="#1b7f79", s=55)
        label = str(widths[0]) if len(widths) == 1 else f"{min(widths)}-{max(widths)}"
        axes[1].annotate(
            f"requested {label}",
            (x_value, y_value),
            xytext=(-8, 8 if len(widths) > 1 else -16),
            textcoords="offset points",
            ha="right",
        )
    axes[1].scatter(
        [tokens["A6_PAPER3_EVIDENCE_RADIUS0_ALL28"]],
        [accuracy["A6_PAPER3_EVIDENCE_RADIUS0_ALL28"]],
        color="#9b2226",
        marker="x",
        s=80,
    )
    axes[1].annotate(
        "Paper 3 radius 0",
        (
            tokens["A6_PAPER3_EVIDENCE_RADIUS0_ALL28"],
            accuracy["A6_PAPER3_EVIDENCE_RADIUS0_ALL28"],
        ),
        xytext=(8, 8),
        textcoords="offset points",
    )
    axes[1].set_xlabel("Unique native tokens per case")
    axes[1].set_ylabel("Answer accuracy (%)")
    axes[1].set_ylim(-3, 68)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"native_bug_ablation.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    values = {
        "NativeBugVisibleSessionAccuracy": accuracy["VISIBLE_FULL_SESSION"],
        "NativeBugNativeSessionAccuracy": accuracy["NATIVE_FULL_SESSION"],
        "NativeBugVisibleTaskAccuracy": accuracy["VISIBLE_FULL_TASK_SCOPE"],
        "NativeBugNativeTaskAccuracy": accuracy["NATIVE_FULL_TASK_SCOPE"],
        "NativeBugRequiredFullAccuracy": accuracy["A1_REQUIRED_RECORDS_FULL"],
        "NativeBugSelectedFullAccuracy": accuracy["A2_SELECTED_RECORDS_FULL"],
        "NativeBugSparseFourAccuracy": accuracy["LAYER_ABLATION_FULL_SELECTED_SPARSE4"],
        "NativeBugLateOneAccuracy": accuracy["LAYER_ABLATION_FULL_SELECTED_LATE1"],
        "NativeBugPaperThreeAccuracy": accuracy["A6_PAPER3_EVIDENCE_RADIUS0_ALL28"],
        "NativeBugNativeVisibleGap": accuracy["VISIBLE_FULL_TASK_SCOPE"] - accuracy["NATIVE_FULL_TASK_SCOPE"],
        "NativeBugSelectedTokens": tokens["A2_SELECTED_RECORDS_FULL"],
        "NativeBugPaperThreeTokens": tokens["A6_PAPER3_EVIDENCE_RADIUS0_ALL28"],
        "NativeBugPrefixMaxError": float(e0["max_logit_error"]),
        "NativeBugPrefixMeanError": float(e0["mean_logit_error"]),
        "NativeBugKVMaxError": max(float(row["max_k_error"]) for row in kv),
        "NativeBugDecodeLifetimePass": 100.0 * _mean(lifetime, "decode_lifetime_pass"),
        "NativeBugMeanOverlapRemoved": _mean(dedup, "overlap_removed_tokens"),
        "NativeBugMaxDuplicationRatio": max(float(row["duplication_ratio"]) for row in dedup),
    }
    lines = ["% Generated by analyze_native_bug_ablation.py"]
    for name, value in values.items():
        precision = 4 if "Error" in name else 1
        lines.append(f"\\newcommand{{\\{name}}}{{{value:.{precision}f}}}")
    GENERATED.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "conditions": {name: {"accuracy_percent": accuracy[name], "mean_unique_tokens": tokens.get(name)} for name in order},
        "invariants": values,
        "diagnosis": (
            "The position-zero collision was fixed. Full selected records consumed at all 28 layers "
            "recover 60% task answer accuracy; late sparse injection and radius-zero evidence remain "
            "insufficient for this frozen checkpoint."
        ),
    }
    (RESULTS / "native_bug_diagnosis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
