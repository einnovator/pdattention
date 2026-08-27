"""Create Paper 7 figures and TeX macros from calibrated PRA artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "docs/papers/shared/results/paper7_records/full_pra_calibrated"

READER_POLICY_LABELS = {
    "COMPACT_ONLY": "COMPACT_ONLY",
    "PRA_NATIVE": "PRA_NATIVE",
    "PRA_ADAPTIVE": "PRA_ADAPTIVE",
    "PRA_ADAPTIVE_ORACLE": "PRA_ADAPTIVE_ORACLE",
    "CCR_TOOL": "CCR_STYLE",
    "FULL": "FULL_BACKING",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows) if rows else 0.0


def _weighted_recall(rows: list[dict[str, str]]) -> float:
    total = sum(int(row["n"]) for row in rows)
    return sum(float(row["recall"]) * int(row["n"]) for row in rows) / max(total, 1)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _addressability(input_dir: Path, figures: Path) -> None:
    rows = [row for row in _rows(input_dir / "compact_vs_backing_addressability.csv") if row["partition"] == "test"]
    labels = [
        "COMPACT_ONLY\nliteral",
        "PRA_COMPACT\nR@4",
        "PRA_NATIVE\nR@4",
        "FULL_BACKING\nliteral",
    ]
    values = [
        _mean(rows, "compact_trigger_literal"),
        _mean(rows, "pra_compact_recall_at_4"),
        _mean(rows, "pra_native_recall_at_4"),
        _mean(rows, "backing_trigger_literal"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(labels, values, color=["#808080", "#4C78A8", "#2A9D8F", "#E9C46A"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Evidence addressability")
    ax.set_title("Visible compression versus full-backing addresses")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, figures / "visible_backing_addressability")


def _routing(input_dir: Path, figures: Path) -> None:
    rows = [row for row in _rows(input_dir / "pra_native_routing_variants.csv") if row["partition"] == "test"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], int(row["k"]))].append(row)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for policy, color in (("PRA_FALLBACK", "#808080"), ("PRA_NATIVE_SEMANTIC", "#4C78A8"), ("PRA_NATIVE_HYBRID", "#2A9D8F")):
        ks = (1, 2, 4, 8)
        ys = [_weighted_recall(grouped[(policy, k)]) for k in ks]
        ax.plot(ks, ys, marker="o", label=policy.replace("PRA_NATIVE_", "native ").replace("PRA_FALLBACK", "lexical fallback"), color=color)
    ax.set_xticks((1, 2, 4, 8))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Selected chunks K")
    ax.set_ylabel("Backing evidence Recall@K")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    _save(fig, figures / "native_routing_recall")


def _controller(input_dir: Path, figures: Path) -> None:
    rows = _rows(input_dir / "controller_description_calibration.csv")
    labels = [f"{row['model']}\n{row['description_level']}/{row['controller_protocol']}" for row in rows]
    x = list(range(len(rows)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.bar([i - width for i in x], [float(row["need_more_recall"]) for row in rows], width, label="need-more recall", color="#4C78A8")
    ax.bar(x, [float(row["operation_accuracy_given_need_more"]) for row in rows], width, label="operation accuracy", color="#2A9D8F")
    ax.bar([i + width for i in x], [float(row["false_escalation"]) for row in rows], width, label="false escalation", color="#E76F51")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        frameon=False, ncol=3, fontsize=8, loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    _save(fig, figures / "controller_validation")


def _frontier(input_dir: Path, figures: Path) -> None:
    rows_by_policy = {
        row["policy"]: row for row in _rows(input_dir / "quality_cost_frontier.csv")
    }
    policies = (
        "COMPACT_ONLY",
        "PRA_NATIVE",
        "PRA_ADAPTIVE",
        "PRA_ADAPTIVE_ORACLE",
        "CCR_TOOL",
        "FULL",
    )
    rows = [rows_by_policy[policy] for policy in policies]
    offsets = {
        "FULL": (-8, -15),
        "COMPACT_ONLY": (5, 7),
        "PRA_NATIVE": (5, -13),
        "PRA_ADAPTIVE": (-5, 18),
        "PRA_ADAPTIVE_ORACLE": (5, -12),
        "CCR_TOOL": (5, 5),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for row in rows:
        x = float(row["active_kv_tokens"])
        y = float(row["task_success"])
        ax.scatter([x], [y], s=45)
        offset = offsets[row["policy"]]
        ax.annotate(
            READER_POLICY_LABELS[row["policy"]],
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=7, ha="right" if offset[0] < 0 else "left",
        )
    ax.set_xlabel("Mean active K/V tokens")
    ax.set_ylabel("Held-out task success")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    _save(fig, figures / "adaptive_quality_cost_frontier")


def _failures(input_dir: Path, figures: Path) -> None:
    wanted = {"MODEL_ONLY", "PRA_ADAPTIVE", "PRA_ADAPTIVE_ORACLE", "CCR_TOOL"}
    rows = [row for row in _rows(input_dir / "adaptive_failure_decomposition.csv") if row["policy"] in wanted]
    labels = [row["policy"] for row in rows]
    stages = (
        ("need_more_recall", "insufficiency", "#4C78A8"),
        ("operation_accuracy_given_escalation", "operation", "#2A9D8F"),
        ("runtime_recovery", "recovery", "#E9C46A"),
        ("final_use_given_visible", "final use", "#E76F51"),
    )
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    width = 0.18
    for offset, (field, label, color) in enumerate(stages):
        positions = [i + (offset - 1.5) * width for i in x]
        ax.bar(positions, [float(row[field]) for row in rows], width, label=label, color=color)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Conditional stage rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        frameon=False, ncol=4, fontsize=8, loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    _save(fig, figures / "adaptive_failure_decomposition")


def _macros(input_dir: Path) -> None:
    selected = json.loads((input_dir / "selected_controller.json").read_text(encoding="utf-8"))
    frontier = {row["policy"]: row for row in _rows(input_dir / "quality_cost_frontier.csv")}
    address = [row for row in _rows(input_dir / "compact_vs_backing_addressability.csv") if row["partition"] == "test"]
    routing = [row for row in _rows(input_dir / "pra_native_routing_variants.csv") if row["partition"] == "test" and row["k"] == "4"]
    by_policy = defaultdict(list)
    for row in routing:
        by_policy[row["policy"]].append(row)
    config = selected["config"]
    metrics = selected["validation_metrics"]
    lines = [
        "% Generated by summarize_full_pra_calibrated.py; do not edit by hand.",
        rf"\newcommand{{\PaperSevenCalController}}{{\texttt{{{config['model']}}} {config['description_level']}/{config['protocol']}}}",
        rf"\newcommand{{\PaperSevenCalControllerAccuracy}}{{{float(metrics['decision_accuracy']):.3f}}}",
        rf"\newcommand{{\PaperSevenCalControllerNeedRecall}}{{{float(metrics['need_more_recall']):.3f}}}",
        rf"\newcommand{{\PaperSevenCalControllerFalseEsc}}{{{float(metrics['false_escalation']):.3f}}}",
        rf"\newcommand{{\PaperSevenCalControllerOperation}}{{{float(metrics['operation_accuracy_given_need_more']):.3f}}}",
        rf"\newcommand{{\PaperSevenCalControllerPromptTokens}}{{{float(metrics['prompt_tokens']):.1f}}}",
        rf"\newcommand{{\PaperSevenCalControllerDescTokens}}{{{float(metrics['fixed_description_token_estimate']):.0f}}}",
        rf"\newcommand{{\PaperSevenCalControllerLatency}}{{{float(metrics['latency_seconds']):.2f}}}",
        rf"\newcommand{{\PaperSevenCalNativePolicy}}{{\texttt{{\detokenize{{{json.loads((input_dir / 'pra_native_selection.json').read_text())['selected_policy']}}}}}}}",
        rf"\newcommand{{\PaperSevenCalCompactRFour}}{{{_mean(address, 'pra_compact_recall_at_4'):.3f}}}",
        rf"\newcommand{{\PaperSevenCalNativeRFour}}{{{_mean(address, 'pra_native_recall_at_4'):.3f}}}",
        rf"\newcommand{{\PaperSevenCalCompactBytes}}{{{_mean(address, 'compact_bytes'):.1f}}}",
        rf"\newcommand{{\PaperSevenCalBackingBytes}}{{{_mean(address, 'full_backing_bytes'):.1f}}}",
        rf"\newcommand{{\PaperSevenCalCompressionMs}}{{{1000 * _mean(address, 'agent_compression_seconds'):.2f}}}",
        rf"\newcommand{{\PaperSevenCalIndexSeconds}}{{{_mean(address, 'indexing_seconds'):.2f}}}",
        rf"\newcommand{{\PaperSevenCalIndexBytes}}{{{_mean(address, 'routing_index_bytes'):.1f}}}",
        rf"\newcommand{{\PaperSevenCalResidentKVBytes}}{{{_mean(address, 'resident_detail_kv_bytes'):.1f}}}",
        rf"\newcommand{{\PaperSevenCalRoutingMs}}{{{1000 * _mean(address, 'routing_seconds'):.2f}}}",
    ]
    for policy, command in (("PRA_NATIVE_SEMANTIC", "Semantic"), ("PRA_NATIVE_HYBRID", "Hybrid"), ("PRA_FALLBACK", "Fallback")):
        lines.append(rf"\newcommand{{\PaperSevenCal{command}RFour}}{{{_weighted_recall(by_policy[policy]):.3f}}}")
    for policy, command in (("COMPACT_ONLY", "Compact"), ("PRA_NATIVE", "Native"), ("PRA_ADAPTIVE", "Adaptive"), ("PRA_ADAPTIVE_ORACLE", "Oracle"), ("CCR_TOOL", "CCR"), ("FULL", "FullBacking")):
        lines.append(rf"\newcommand{{\PaperSevenCal{command}Success}}{{{float(frontier[policy]['task_success']):.3f}}}")
        lines.append(rf"\newcommand{{\PaperSevenCal{command}SuccessPct}}{{{100 * float(frontier[policy]['task_success']):.1f}}}")
        lines.append(rf"\newcommand{{\PaperSevenCal{command}Tokens}}{{{float(frontier[policy]['active_kv_tokens']):.1f}}}")
        lines.append(rf"\newcommand{{\PaperSevenCal{command}TypedTokens}}{{{float(frontier[policy]['materialized_tokens']):.1f}}}")
        lines.append(rf"\newcommand{{\PaperSevenCal{command}ControllerTokens}}{{{float(frontier[policy]['controller_prompt_tokens']):.1f}}}")
    native_tokens = float(frontier["PRA_NATIVE"]["active_kv_tokens"])
    full_tokens = float(frontier["FULL"]["active_kv_tokens"])
    lines.append(
        rf"\newcommand{{\PaperSevenCalActiveKVReductionPct}}{{{100 * (full_tokens - native_tokens) / full_tokens:.1f}}}"
    )
    (input_dir / "generated_full_pra_calibrated_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (input_dir / "experiment_manifest.json").write_text(
        json.dumps({
            "protocol": "paper7-full-pra-calibrated-v1",
            "routing": json.loads((input_dir / "pra_native_selection.json").read_text()),
            "controller": selected,
            "heldout_cases": len({row["case_id"] for row in _rows(input_dir / "adaptive_oracle_results.csv")}),
            "heldout_seeds": sorted({int(row["seed"]) for row in _rows(input_dir / "adaptive_oracle_results.csv")}),
            "answer_gate": "exact expected answer marker visible after authorized materialization",
            "native_consumption": "live one-token smoke verified requested equals consumed; repeated selections use requested-token accounting",
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    figures = args.input_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _addressability(args.input_dir, figures)
    _routing(args.input_dir, figures)
    _controller(args.input_dir, figures)
    _frontier(args.input_dir, figures)
    _failures(args.input_dir, figures)
    _macros(args.input_dir)


if __name__ == "__main__":
    main()
