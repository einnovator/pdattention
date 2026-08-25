"""Summarize Paper 6.5 M2--M6 into reviewable tables, figures, and TeX macros."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools"
OUT = RESULTS / "m2_m6_summary"


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _bool(value: str) -> bool:
    return value.casefold() == "true"


def _mean(rows, field) -> float:
    return fmean(float(row[field]) for row in rows) if rows else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _m2_m4_summary():
    m2 = _read(RESULTS / "pretrained_bridge/m2_rows.csv")
    m3 = _read(RESULTS / "pretrained_bridge/m3_rows.csv")
    m4 = [row for row in _read(RESULTS / "pretrained_bridge/m4_rows.csv") if row["milestone"] == "M4-summary"]
    rows = []
    for condition in sorted({row["condition"] for row in m2}):
        group = [row for row in m2 if row["condition"] == condition]
        rows.append({"milestone": "M2", "condition": condition, "success": fmean(_bool(row["end_to_end_success"]) for row in group), "n": len(group)})
    rows.extend((
        {"milestone": "M3", "condition": "execution_accepted", "success": fmean(_bool(row["execution_accepted"]) for row in m3), "n": len(m3)},
        {"milestone": "M3", "condition": "observation_grounded", "success": fmean(_bool(row["observation_grounded"]) for row in m3), "n": len(m3)},
    ))
    for condition in sorted({row["condition"] for row in m4}):
        group = [row for row in m4 if row["condition"] == condition]
        rows.append({"milestone": "M4", "condition": condition, "success": fmean(_bool(row["executed"]) for row in group), "n": len(group)})
    return rows


def _m5_summary():
    policy = _read(RESULTS / "m5_disclosure/m5_policy_rows.csv")
    model = _read(RESULTS / "m5_disclosure/m5_model_rows.csv")
    rows = []
    for name in sorted({row["policy"] for row in policy}):
        group = [row for row in policy if row["policy"] == name]
        model_group = [row for row in model if row["policy"] == name]
        rows.append({
            "policy": name,
            "required_recall": _mean(group, "required_recall_at_k"),
            "initial_tools": _mean(group, "initial_disclosed_tools"),
            "definition_tokens": _mean(group, "initial_definition_tokens"),
            "unsafe_exposure": _mean(group, "unsafe_exposure_count"),
            "model_task_success": fmean(_bool(row["task_success"]) for row in model_group) if model_group else "",
            "model_n": len(model_group),
        })
    return rows


def _m6_summary():
    raw = _read(RESULTS / "m6_native/m6_rows.csv")
    rows = []
    for mode in sorted({row["mode"] for row in raw}):
        group = [row for row in raw if row["mode"] == mode]
        single = [row for row in group if int(row["plan_horizon"]) == 1]
        multi = [row for row in group if int(row["plan_horizon"]) > 1]
        rows.append({
            "mode": mode,
            "single_top1": fmean(_bool(row["top1_correct"]) for row in single),
            "multi_required_recall": _mean(multi, "required_recall_at_budget"),
            "multi_successor_recall": _mean(multi, "successor_recall_at_budget"),
            "all_required_recovered": fmean(_bool(row["all_required_recovered"]) for row in multi),
            "mrr": _mean(group, "mrr"),
            "index_bytes": int(group[0]["index_bytes"]),
        })
    return rows


def _plot_gates(rows):
    selected = [
        row for row in rows
        if (row["milestone"], row["condition"]) in {
            ("M2", "selected"), ("M2", "eager"), ("M2", "shuffled"),
            ("M3", "observation_grounded"), ("M4", "reactive_jit"),
            ("M4", "eager_required"), ("M4", "no_refresh"),
        }
    ]
    short = {
        ("M2", "eager"): "M2\neager",
        ("M2", "selected"): "M2\nselected",
        ("M2", "shuffled"): "M2\nshuffled",
        ("M3", "observation_grounded"): "M3\nobservation",
        ("M4", "eager_required"): "M4\neager",
        ("M4", "no_refresh"): "M4\nno refresh",
        ("M4", "reactive_jit"): "M4\nreactive",
    }
    labels = [short[(row["milestone"], row["condition"])] for row in selected]
    values = [row["success"] for row in selected]
    colors = ["#2b6f77" if value >= 0.5 else "#b14a3b" for value in values]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Strict success fraction")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "m2_m4_gate_summary.pdf")
    fig.savefig(OUT / "m2_m4_gate_summary.png", dpi=180)
    plt.close(fig)


def _plot_m5(rows):
    names = ["P1 direct", "P6 combined", "P7 reactive", "P8 speculative", "P9 oracle"]
    keys = ["p1_direct_top1", "p6_combined_graph", "p7_reactive_jit", "p8_speculative_planning", "p9_oracle_capabilities"]
    by_name = {row["policy"]: row for row in rows}
    recall = [float(by_name[key]["required_recall"]) for key in keys]
    success = [float(by_name[key]["model_task_success"]) for key in keys]
    x = list(range(len(keys)))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar([value - .19 for value in x], recall, width=.38, label="Initial required-tool recall", color="#46789b")
    ax.bar([value + .19 for value in x], success, width=.38, label="Model task success", color="#d07a3e")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction")
    ax.legend(frameon=False, ncols=2, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "m5_recall_vs_execution.pdf")
    fig.savefig(OUT / "m5_recall_vs_execution.png", dpi=180)
    plt.close(fig)


def _plot_m6(rows):
    labels = {
        "token": "Token",
        "index": "Index",
        "external_signed_hash": "Signed hash",
        "input_embedding_mean": "Input mean",
        "native_mean_k": "Native mean K",
        "native_token_qk": "Native token QK",
        "paper2_8_rank16_ensemble": "P2.8 rank-16",
        "paper2_8_rank8_centroids": "P2.8 r8/8c",
        "lexical_native_hybrid": "Lexical+QK",
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for row in rows:
        ax.scatter(row["index_bytes"], row["multi_required_recall"], s=58)
        ax.annotate(labels[row["mode"]], (row["index_bytes"], row["multi_required_recall"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Persistent routing-index bytes (declared payload)")
    ax.set_ylabel("Multi-step required-tool recall at horizon budget")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "m6_quality_cost_frontier.pdf")
    fig.savefig(OUT / "m6_quality_cost_frontier.png", dpi=180)
    plt.close(fig)


def _macro(name: str, value: float) -> str:
    return f"\\newcommand{{\\{name}}}{{{value:.3f}}}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gates = _m2_m4_summary()
    m5 = _m5_summary()
    m6 = _m6_summary()
    _write_csv(OUT / "m2_m4_summary.csv", gates)
    _write_csv(OUT / "m5_summary.csv", m5)
    _write_csv(OUT / "m6_summary.csv", m6)
    _plot_gates(gates)
    _plot_m5(m5)
    _plot_m6(m6)
    gate = {(row["milestone"], row["condition"]): row["success"] for row in gates}
    m5_by = {row["policy"]: row for row in m5}
    m6_by = {row["mode"]: row for row in m6}
    macros = [
        "% Generated by experiments/paper6_5_tools/summarize_m2_m6.py",
        _macro("PaperSixFiveMTwoSelected", gate[("M2", "selected")]),
        _macro("PaperSixFiveMTwoEager", gate[("M2", "eager")]),
        _macro("PaperSixFiveMThreeGrounded", gate[("M3", "observation_grounded")]),
        _macro("PaperSixFiveMFourReactive", gate[("M4", "reactive_jit")]),
        _macro("PaperSixFiveMFourEager", gate[("M4", "eager_required")]),
        _macro("PaperSixFiveMFiveCombinedRecall", float(m5_by["p6_combined_graph"]["required_recall"])),
        _macro("PaperSixFiveMFiveCombinedSuccess", float(m5_by["p6_combined_graph"]["model_task_success"])),
        _macro("PaperSixFiveMSixIndexTopOne", float(m6_by["index"]["single_top1"])),
        _macro("PaperSixFiveMSixTokenRecall", float(m6_by["token"]["multi_required_recall"])),
        _macro("PaperSixFiveMSixNativeMeanTopOne", float(m6_by["native_mean_k"]["single_top1"])),
        _macro("PaperSixFiveMSixRankSixteenRecall", float(m6_by["paper2_8_rank16_ensemble"]["multi_required_recall"])),
    ]
    (OUT / "generated_m2_m6_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    findings = {
        "m2_selected_success": gate[("M2", "selected")],
        "m2_eager_success": gate[("M2", "eager")],
        "m3_observation_grounded": gate[("M3", "observation_grounded")],
        "m4_reactive_success": gate[("M4", "reactive_jit")],
        "m4_eager_success": gate[("M4", "eager_required")],
        "m5_combined_required_recall": float(m5_by["p6_combined_graph"]["required_recall"]),
        "m5_combined_model_success": float(m5_by["p6_combined_graph"]["model_task_success"]),
        "m5_stop_gate": "negative: static graph disclosure did not improve task success",
        "m6_best_single_top1_mode": max(m6, key=lambda row: row["single_top1"])["mode"],
        "m6_best_multi_recall_mode": max(m6, key=lambda row: row["multi_required_recall"])["mode"],
        "m6_native_stop_gate": "negative: native and transferred low-rank QK did not beat indexed lexical discovery",
    }
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
