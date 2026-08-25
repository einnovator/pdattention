"""Summarize Paper 6.5 automatic records, union discovery, and E4--E6."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools/auto_union_records"


def _read(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(RESULTS / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(RESULTS / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def _macro(name: str, value: float) -> str:
    return f"\\newcommand{{\\{name}}}{{{value:.3f}}}"


def main() -> None:
    semantic = _read("auto_vs_manual_summary.csv")
    frontier = _read("union_recall_frontier.csv")
    jit = _read("union_jit_summary.csv")
    atomic = _read("tool_atomicity_summary.csv")
    overall = {row["condition"]: row for row in semantic if row["stratum"] == "all"}

    conditions = ("manual_hybrid", "auto_python_only", "auto_dictionary", "auto_embedding", "auto_hybrid")
    labels = ("Manual\nhybrid", "Auto\nPython", "Auto +\ndictionary", "Auto +\nembedding", "Auto\nhybrid")
    x = range(len(conditions))
    plt.figure(figsize=(7.4, 3.4))
    plt.bar([value - 0.18 for value in x], [float(overall[row]["top1"]) for row in conditions], width=.36, label="Top-1", color="#247BA0")
    plt.bar([value + 0.18 for value in x], [float(overall[row]["recall_at_3"]) for row in conditions], width=.36, label="Recall@3", color="#F2A541")
    plt.xticks(list(x), labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Test quality")
    plt.legend(frameon=False, ncol=2)
    plt.grid(axis="y", alpha=.25)
    _save("auto_vs_manual_quality")

    colors = {"single_channel": "#247BA0", "fused_score": "#D1495B", "raw_union": "#3B8C6E", "diversity_union": "#6B5CA5"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for strategy, color in colors.items():
        rows = sorted((row for row in frontier if row["strategy"] == strategy), key=lambda row: int(row["max_candidates"]))
        axes[0].plot([int(row["max_candidates"]) for row in rows], [float(row["required_recall"]) for row in rows], marker="o", label=strategy.replace("_", " "), color=color)
        axes[1].plot([float(row["mean_schema_tokens"]) for row in rows], [float(row["required_recall"]) for row in rows], marker="o", color=color)
    axes[0].set_xlabel("Maximum candidates")
    axes[0].set_ylabel("Required-tool recall")
    axes[1].set_xlabel("Mean schema tokens")
    axes[1].set_ylabel("Required-tool recall")
    for axis in axes:
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    _save("union_recall_frontier")

    plt.figure(figsize=(7.5, 3.6))
    display = [row for row in jit if row["condition"] in {"top1_jit", "union_jit_k2", "union_jit_k4", "union_jit_k6", "union_jit_k8", "static_oracle", "static_graph", "all_tools"}]
    for row in display:
        plt.scatter(float(row["mean_context_tokens"]), float(row["task_success"]), s=52)
        plt.annotate(row["condition"].replace("union_jit_", "K=").replace("_jit", "").replace("static_", ""), (float(row["mean_context_tokens"]), float(row["task_success"])), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("Mean cumulative tool-definition tokens")
    plt.ylabel("Strict workflow success")
    plt.ylim(-.03, 1.03)
    plt.grid(alpha=.25)
    _save("union_jit_frontier")

    atomic_by = {row["condition"]: row for row in atomic}
    atomic_conditions = (
        "record_full", "parameters_only", "description_only", "generic_chunk_reroute",
        "drop_required_field", "flat_serialization",
    )
    atomic_labels = (
        "Full\nrecord", "No\ndescription", "Description\nonly", "Generic chunk\nreroute",
        "Drop required\nfield", "Flat\ntext",
    )
    x = range(len(atomic_conditions))
    plt.figure(figsize=(7.6, 3.5))
    plt.bar([value - .18 for value in x], [float(atomic_by[row]["argument_accuracy"]) for row in atomic_conditions], width=.36, color="#247BA0", label="Exact call")
    plt.bar([value + .18 for value in x], [float(atomic_by[row]["required_argument_omission"]) for row in atomic_conditions], width=.36, color="#D1495B", label="Required omission")
    plt.xticks(list(x), atomic_labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Rate")
    plt.legend(frameon=False, ncol=2)
    plt.grid(axis="y", alpha=.25)
    _save("tool_atomicity_controls")

    frontier_by = {(row["strategy"], int(row["max_candidates"])): row for row in frontier}
    jit_by = {row["condition"]: row for row in jit}
    full = atomic_by["record_full"]
    dropped = atomic_by["drop_required_field"]
    macros = [
        _macro("PaperSixFiveAutoTopOne", float(overall["auto_hybrid"]["top1"])),
        _macro("PaperSixFiveManualTopOne", float(overall["manual_hybrid"]["top1"])),
        _macro("PaperSixFiveAutoRecallThree", float(overall["auto_hybrid"]["recall_at_3"])),
        _macro("PaperSixFiveFusedKFourRecall", float(frontier_by[("fused_score", 4)]["required_recall"])),
        _macro("PaperSixFiveUnionKFourRecall", float(frontier_by[("diversity_union", 4)]["required_recall"])),
        _macro("PaperSixFiveTopOneJITSuccess", float(jit_by["top1_jit"]["task_success"])),
        _macro("PaperSixFiveUnionKTwoSuccess", float(jit_by["union_jit_k2"]["task_success"])),
        _macro("PaperSixFiveUnionKFourSuccess", float(jit_by["union_jit_k4"]["task_success"])),
        _macro("PaperSixFiveUnionKSixSuccess", float(jit_by["union_jit_k6"]["task_success"])),
        _macro("PaperSixFiveUnionKEightSuccess", float(jit_by["union_jit_k8"]["task_success"])),
        _macro("PaperSixFiveAllToolsSuccess", float(jit_by["all_tools"]["task_success"])),
        _macro("PaperSixFiveAtomicCallValidity", float(full["execution_acceptance"])),
        _macro("PaperSixFiveDescriptionOnlyValidity", float(atomic_by["description_only"]["execution_acceptance"])),
        _macro("PaperSixFiveGenericChunkValidity", float(atomic_by["generic_chunk_reroute"]["execution_acceptance"])),
        _macro("PaperSixFiveDroppedFieldValidity", float(dropped["execution_acceptance"])),
        _macro("PaperSixFiveDroppedFieldOmission", float(dropped["required_argument_omission"])),
    ]
    (RESULTS / "generated_auto_union_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    table_labels = {
        "top1_jit": "Top-1 JIT",
        "union_jit_k2": "Union JIT, $K=2$",
        "union_jit_k4": "Union JIT, $K=4$",
        "union_jit_k6": "Union JIT, $K=6$",
        "union_jit_k8": "Union JIT, $K=8$",
        "static_oracle": "Static required-set oracle",
        "static_graph": "Static graph set",
        "all_tools": "All tools",
    }
    table_lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Policy & Success & Required recall & Schema tokens & Disclosed tools \\\\",
        "\\midrule",
    ]
    for condition in table_labels:
        row = jit_by[condition]
        table_lines.append(
            f"{table_labels[condition]} & {float(row['task_success']):.3f} & "
            f"{float(row['candidate_required_recall']):.3f} & "
            f"{float(row['mean_context_tokens']):.0f} & {float(row['mean_disclosed_tools']):.1f} \\\\"
        )
    table_lines.extend(("\\bottomrule", "\\end{tabular}"))
    (RESULTS / "generated_union_jit_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    findings = {
        "auto_top1": float(overall["auto_hybrid"]["top1"]),
        "manual_top1": float(overall["manual_hybrid"]["top1"]),
        "fused_k4_recall": float(frontier_by[("fused_score", 4)]["required_recall"]),
        "diversity_k4_recall": float(frontier_by[("diversity_union", 4)]["required_recall"]),
        "best_jit": max(jit, key=lambda row: (float(row["task_success"]), -float(row["mean_context_tokens"]))),
        "full_atomic": full,
        "drop_required_field": dropped,
    }
    (RESULTS / "summary_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
