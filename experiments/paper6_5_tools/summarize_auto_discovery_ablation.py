"""Summarize the Paper 6.5 zero-configuration discovery ablations."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools/auto_discovery_ablation"


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


def _metric(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    return next(row for row in rows if row[key] == value)


def _plot_ladder(primary: list[dict[str, str]]) -> None:
    order = [
        "A0_raw_bm25",
        "A1_auto_keywords",
        "A2_keywords_synonyms",
        "A3_inferred_concepts",
        "A4_auto_tags",
        "A5_embedding_selected",
        "A6_auto_hybrid",
        "manual_rich_ceiling",
    ]
    labels = ("A0\nBM25", "A1\nkeywords", "A2\n+ synonyms", "A3\n+ concepts", "A4\n+ tags", "A5\nembedding", "A6\nhybrid", "Manual\nceiling")
    by_policy = {row["policy"]: row for row in primary}
    x = list(range(len(order)))
    plt.figure(figsize=(9.1, 3.65))
    plt.bar([value - .19 for value in x], [float(by_policy[row]["top1"]) for row in order], width=.38, color="#247BA0", label="Top-1")
    plt.bar([value + .19 for value in x], [float(by_policy[row]["recall_at_3"]) for row in order], width=.38, color="#F2A541", label="Recall@3")
    plt.xticks(x, labels)
    plt.ylim(0, 1.03)
    plt.ylabel("Frozen-test quality")
    plt.grid(axis="y", alpha=.25)
    plt.legend(frameon=False, ncol=2, loc="upper left")
    _save("auto_discovery_ladder")


def _plot_hardness(rows: list[dict[str, str]]) -> None:
    conditions = {
        "A0_raw_bm25": ("Raw BM25", "#247BA0"),
        "A2_keywords_synonyms": ("Keywords + synonyms", "#3B8C6E"),
        "A5_embedding_selected": ("Embedding", "#6B5CA5"),
        "A6_auto_hybrid": ("Auto hybrid", "#D1495B"),
        "manual_rich_ceiling": ("Manual ceiling", "#444444"),
    }
    strata = [f"H{index}" for index in range(6)]
    plt.figure(figsize=(8.0, 3.8))
    for condition, (label, color) in conditions.items():
        selected = {row["stratum"]: row for row in rows if row["condition"] == condition}
        plt.plot(strata, [float(selected[stratum]["top1"]) for stratum in strata], marker="o", linewidth=1.8, label=label, color=color)
    plt.ylim(-.02, 1.03)
    plt.xlabel("Semantic-hardness stratum")
    plt.ylabel("Top-1")
    plt.grid(alpha=.25)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    _save("auto_discovery_by_hardness")


def _plot_quality(docstrings: list[dict[str, str]], names: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), sharey=True)
    specifications = (
        (axes[0], docstrings, ("good", "minimal_one_line", "none"), ("Good", "Minimal", "None"), "Docstring quality"),
        (axes[1], names, ("descriptive", "abbreviated", "opaque"), ("Descriptive", "Abbreviated", "Opaque"), "Function-name quality"),
    )
    for axis, rows, order, labels, title in specifications:
        by = {(row["quality"], row["policy"]): row for row in rows}
        x = list(range(len(order)))
        axis.bar([value - .18 for value in x], [float(by[(quality, "A1_auto_keywords")]["top1"]) for quality in order], width=.36, color="#247BA0", label="Auto keywords")
        axis.bar([value + .18 for value in x], [float(by[(quality, "A5_embedding")]["top1"]) for quality in order], width=.36, color="#6B5CA5", label="Embedding")
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.25)
        axis.set_ylim(0, .65)
    axes[0].set_ylabel("Frozen-test Top-1")
    axes[1].legend(frameon=False, fontsize=8)
    _save("auto_callable_quality_sensitivity")


def _plot_union(frontier: list[dict[str, str]]) -> None:
    colors = {"fused_score": "#D1495B", "raw_union": "#3B8C6E", "diversity_union": "#6B5CA5"}
    labels = {"fused_score": "Fused Top-K", "raw_union": "Raw union", "diversity_union": "Diversity union"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for strategy, color in colors.items():
        rows = sorted(
            (row for row in frontier if row["split"] == "test" and row["strategy"] == strategy),
            key=lambda row: int(row["max_candidates"]),
        )
        budgets = [int(row["max_candidates"]) for row in rows]
        axes[0].plot(budgets, [float(row["required_recall"]) for row in rows], marker="o", label=labels[strategy], color=color)
        axes[1].plot(budgets, [float(row["unsafe_exposure"]) for row in rows], marker="o", label=labels[strategy], color=color)
    axes[0].set_ylabel("Required-tool recall")
    axes[1].set_ylabel("Unsafe-candidate exposure")
    for axis in axes:
        axis.set_xlabel("Maximum candidates")
        axis.set_ylim(-.02, 1.03)
        axis.grid(alpha=.25)
    axes[0].legend(frameon=False, fontsize=8)
    _save("auto_union_vs_fusion")


def _write_table(primary: list[dict[str, str]]) -> None:
    labels = {
        "A0_raw_bm25": "A0 Raw BM25",
        "A1_auto_keywords": "A1 Auto keywords",
        "A2_keywords_synonyms": "A2 Keywords + synonyms",
        "A3_inferred_concepts": "A3 Inferred concepts",
        "A4_auto_tags": "A4 Automatic tags",
        "A5_embedding_selected": "A5 Compact embedding",
        "A6_auto_hybrid": "A6 Automatic hybrid",
        "manual_rich_ceiling": "Manual-rich ceiling",
    }
    lines = [
        "\\begin{tabular}{lccccrrr}",
        "\\toprule",
        "Policy & Manual & Syn. & Concepts & Embed. & Top-1 & R@3 & R@5 \\\\",
        "\\midrule",
    ]
    for row in primary:
        if row["policy"].startswith("A7_"):
            label = f"A7 {row['policy'].split('_k')[0][3:].replace('_', ' ')} ($K={row['union_selected_k']}$)"
        else:
            label = labels[row["policy"]]
        top1 = "--" if row["top1"] == "n/a" else f"{float(row['top1']):.3f}"
        lines.append(
            f"{label} & {'yes' if row['manual_metadata'] == '1' else 'no'} & "
            f"{'yes' if row['synonyms'] == '1' else 'no'} & "
            f"{'yes' if row['inferred_concepts'] == '1' else 'no'} & "
            f"{'yes' if row['embedding'] == '1' else 'no'} & {top1} & "
            f"{float(row['recall_at_3']):.3f} & {float(row['recall_at_5']):.3f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (RESULTS / "generated_auto_discovery_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    primary = _read("auto_discovery_ablation.csv")
    hardness = _read("auto_discovery_by_hardness.csv")
    source = _read("auto_keyword_source_ablation.csv")
    types = _read("auto_type_schema_ablation.csv")
    docstrings = _read("docstring_quality_ablation.csv")
    names = _read("function_name_quality_ablation.csv")
    frontier = _read("union_vs_fusion_frontier.csv")
    complementarity = _read("union_channel_complementarity.csv")
    jit = _read("union_jit_ablation.csv")[0]

    _plot_ladder(primary)
    _plot_hardness(hardness)
    _plot_quality(docstrings, names)
    _plot_union(frontier)
    _write_table(primary)

    by_policy = {row["policy"]: row for row in primary}
    by_source = {row["condition"]: row for row in source}
    by_type = {row["condition"]: row for row in types}
    by_doc = {(row["quality"], row["policy"]): row for row in docstrings}
    by_name = {(row["quality"], row["policy"]): row for row in names}
    test_frontier = {(row["strategy"], int(row["max_candidates"])): row for row in frontier if row["split"] == "test"}
    selected_union = next(row for row in primary if row["policy"].startswith("A7_"))
    selected_budget = int(selected_union["union_selected_k"])
    selected_strategy = selected_union["policy"][3:].rsplit("_k", 1)[0]
    counts = Counter(row["category"] for row in complementarity)

    macros = [
        _macro("PaperSixFiveAblationRawTopOne", float(by_policy["A0_raw_bm25"]["top1"])),
        _macro("PaperSixFiveAblationKeywordTopOne", float(by_policy["A1_auto_keywords"]["top1"])),
        _macro("PaperSixFiveAblationSynonymTopOne", float(by_policy["A2_keywords_synonyms"]["top1"])),
        _macro("PaperSixFiveAblationAutoTopOne", float(by_policy["A6_auto_hybrid"]["top1"])),
        _macro("PaperSixFiveAblationAutoRecallThree", float(by_policy["A6_auto_hybrid"]["recall_at_3"])),
        _macro("PaperSixFiveAblationManualTopOne", float(by_policy["manual_rich_ceiling"]["top1"])),
        _macro("PaperSixFiveAblationManualRecallThree", float(by_policy["manual_rich_ceiling"]["recall_at_3"])),
        _macro("PaperSixFiveDocstringRemovalRecallThree", float(by_source["minus_docstring"]["recall_at_3"])),
        _macro("PaperSixFiveTypeRecallThreeGain", float(by_type["A1_with_type_schema"]["recall_at_3"]) - float(by_type["A1_without_type_schema"]["recall_at_3"])),
        _macro("PaperSixFiveNoDocEmbeddingTopOne", float(by_doc[("none", "A5_embedding")]["top1"])),
        _macro("PaperSixFiveOpaqueNameEmbeddingTopOne", float(by_name[("opaque", "A5_embedding")]["top1"])),
        _macro("PaperSixFiveAutoFusedKFourRecall", float(test_frontier[("fused_score", 4)]["required_recall"])),
        _macro("PaperSixFiveAutoRawUnionKFourRecall", float(test_frontier[("raw_union", 4)]["required_recall"])),
        _macro("PaperSixFiveAutoDiversityKFourRecall", float(test_frontier[("diversity_union", 4)]["required_recall"])),
        _macro("PaperSixFiveAutoSelectedUnionRecall", float(selected_union["union_selected_recall"])),
        _macro("PaperSixFiveAutoSelectedUnionUnsafe", float(test_frontier[(selected_strategy, selected_budget)]["unsafe_exposure"])),
    ]
    (RESULTS / "generated_auto_discovery_results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    findings = {
        "automatic_ladder": {row["policy"]: {key: row[key] for key in ("top1", "recall_at_3", "recall_at_5")} for row in primary},
        "callable_quality": {
            "no_docstring_embedding_top1": float(by_doc[("none", "A5_embedding")]["top1"]),
            "opaque_name_embedding_top1": float(by_name[("opaque", "A5_embedding")]["top1"]),
            "type_schema_recall_at_3_gain": float(by_type["A1_with_type_schema"]["recall_at_3"]) - float(by_type["A1_without_type_schema"]["recall_at_3"]),
        },
        "union": {
            "selected_strategy": selected_strategy,
            "selected_budget": selected_budget,
            "selected_test_recall": float(selected_union["union_selected_recall"]),
            "selected_test_unsafe_exposure": float(test_frontier[(selected_strategy, selected_budget)]["unsafe_exposure"]),
            "complementarity_counts": dict(sorted(counts.items())),
            "jit_gate": jit,
        },
    }
    (RESULTS / "auto_discovery_summary.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
