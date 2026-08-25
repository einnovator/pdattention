"""Summarize M9 typed progressive disclosure and build paper artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools/progressive_disclosure"
AUTO_RESULTS = ROOT / "docs/papers/shared/results/paper6_5_tools/auto_discovery_ablation"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float(row: Mapping[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    return float(value) if value not in (None, "") else default


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else math.nan


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(name: str) -> None:
    plt.tight_layout()
    plt.savefig(RESULTS / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(RESULTS / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def _context_fit(row: Mapping[str, str]) -> bool:
    value = row.get("context_fit", "")
    if value == "":
        return not bool(row.get("generation_error"))
    return bool(int(value))


def _aggregate(
    rows: Sequence[dict[str, str]],
    *,
    resource_type: str,
    recall_field: str,
    quality_fields: Sequence[str],
    cost_by_key: Mapping[tuple[str, str, int], dict[str, str]],
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["max_candidates"]), row["condition"])].append(row)
    summary: list[dict[str, object]] = []
    for (budget, condition), group in sorted(groups.items()):
        fitting = [row for row in group if _context_fit(row)]
        full_tokens = [
            _float(cost_by_key[(resource_type, row["query_id"], budget)], "all_candidate_full_tokens")
            for row in group
        ]
        total_tokens = [_float(row, "total_capability_tokens") for row in fitting]
        item: dict[str, object] = {
            "resource_type": resource_type,
            "max_candidates": budget,
            "condition": condition,
            "cases": len(group),
            "quality_cases": len(fitting),
            "context_fit_rate": _mean(float(_context_fit(row)) for row in group),
            "required_recall_at_k": _mean(_float(row, recall_field) for row in group),
            "phase_a_tokens": _mean(_float(row, "phase_a_capability_tokens") for row in fitting),
            "phase_b_tokens": _mean(_float(row, "phase_b_capability_tokens") for row in fitting),
            "total_capability_tokens": _mean(total_tokens),
            "all_candidate_full_tokens": _mean(full_tokens),
            "candidate_full_tokens_avoided": _mean(
                full - total for full, total in zip(full_tokens, total_tokens)
            ) if len(fitting) == len(group) else math.nan,
            "total_disclosure_ratio": _mean(
                total / full for total, full in zip(total_tokens, full_tokens) if full
            ) if len(fitting) == len(group) else math.nan,
            "native_kv_mib": _mean(_float(row, "native_kv_bytes") for row in fitting) / (1024 * 1024),
            "model_invocations": _mean(_float(row, "model_invocations") for row in fitting),
            "ttft_seconds": _mean(_float(row, "total_ttft_seconds") for row in fitting),
            "transition_seconds": _mean(_float(row, "phase_transition_overhead_seconds") for row in fitting),
            "materialization_seconds": _mean(_float(row, "materialization_seconds") for row in fitting),
            "wall_clock_seconds": _mean(_float(row, "wall_clock_seconds") for row in fitting),
        }
        for field in quality_fields:
            item[field] = _mean(_float(row, field) for row in fitting)
        summary.append(item)
    return summary


def _skill_discovery_summary() -> list[dict[str, object]]:
    rows = [row for row in _read(RESULTS / "skill_discovery_results.csv") if row["split"] == "test"]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    return [
        {
            "resource_type": "skill",
            "policy": policy,
            "queries": len(group),
            "top1": _mean(_float(row, "top1") for row in group),
            "recall_at_2": _mean(_float(row, "recall_at_2") for row in group),
            "recall_at_4": _mean(_float(row, "recall_at_4") for row in group),
            "recall_at_8": _mean(_float(row, "recall_at_8") for row in group),
        }
        for policy, group in sorted(groups.items())
    ]


def _tool_discovery_summary() -> list[dict[str, object]]:
    source = {row["policy"]: row for row in _read(AUTO_RESULTS / "auto_discovery_ablation.csv")}
    mapping = {
        "bm25": "A0_raw_bm25",
        "auto_semantic": "A4_auto_tags",
        "embedding": "A5_embedding_selected",
        "fused": "A6_auto_hybrid",
    }
    return [
        {
            "resource_type": "tool",
            "policy": label,
            "queries": 144,
            "top1": float(source[name]["top1"]),
            "recall_at_2": "",
            "recall_at_4": "",
            "recall_at_8": "",
            "recall_at_3": float(source[name]["recall_at_3"]),
        }
        for label, name in mapping.items()
    ]


def _distribution(rows: Sequence[dict[str, str]], resource_type: str) -> dict[str, float | str]:
    full = [_float(row, "full_tokens") for row in rows]
    ratios = [_float(row, "selection_full_ratio") for row in rows]
    ordered = sorted(full)

    def percentile(fraction: float) -> float:
        return ordered[math.ceil(fraction * len(ordered)) - 1]

    return {
        "resource_type": resource_type,
        "records": len(rows),
        "median_full_tokens": statistics.median(full),
        "p90_full_tokens": percentile(0.90),
        "p95_full_tokens": percentile(0.95),
        "max_full_tokens": max(full),
        "median_selection_full_ratio": statistics.median(ratios),
    }


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{float(value):.{digits}f}"


def _tex_rate(value: object) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "--"
    return f"{float(value):.2f}"


def _write_tex_table(path: Path, rows: Sequence[dict[str, object]], kind: str) -> None:
    condition_labels = {
        "T0_full_all": "T0 full-all",
        "T1_selection_only": "T1 selection-only",
        "T2_selection_to_full": "T2 select$\\rightarrow$full",
        "T3_oracle_full": "T3 oracle-full",
        "S0_full_all": "S0 full-all",
        "S1_selection_only": "S1 selection-only",
        "S2_selection_to_full": "S2 select$\\rightarrow$full",
        "S3_oracle_full": "S3 oracle-full",
    }
    quality = "schema_valid_call" if kind == "tool" else "instruction_following_success"
    lines = [
        "\\begin{tabular}{rrl r@{\\hspace{1.1em}}r@{\\hspace{1.1em}}r@{\\hspace{1.1em}}r@{\\hspace{1.1em}}r@{\\hspace{1.1em}}r}",
        "\\toprule",
        "$K$ & $n$ & Condition & Recall & Choice & Valid/follow & Success & Tokens & Wall (s) \\\\",
        "\\midrule",
    ]
    previous = None
    for row in rows:
        budget = int(row["max_candidates"])
        if previous is not None and budget != previous:
            lines.append("\\addlinespace")
        lines.append(
            f"{budget} & {int(row['quality_cases'])} & {condition_labels[row['condition']]} & "
            f"{_tex_rate(row['required_recall_at_k'])} & {_tex_rate(row['capability_choice_correct'])} & "
            f"{_tex_rate(row[quality])} & {_tex_rate(row['task_success'])} & "
            f"{_tex_rate(row['total_capability_tokens'])} & {_tex_rate(row['wall_clock_seconds'])} \\\\"
        )
        previous = budget
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tool_rows = _read(RESULTS / "tool_progressive_disclosure_results.csv")
    skill_rows = _read(RESULTS / "skill_progressive_disclosure_results.csv")
    costs = _read(RESULTS / "capability_disclosure_costs.csv")
    cost_by_key = {
        (row["resource_type"], row["query_id"], int(row["max_candidates"])): row for row in costs
    }
    tool_summary = _aggregate(
        tool_rows,
        resource_type="tool",
        recall_field="required_tool_recall_at_k",
        quality_fields=(
            "capability_choice_correct", "wrong_tool_choice", "unsafe_tool_choice",
            "call_parse_valid", "required_argument_coverage", "argument_semantic_correct",
            "enum_type_valid", "schema_valid_call", "execution_acceptance", "task_success",
            "wrong_tool_execution_attempt", "host_rejection",
        ),
        cost_by_key=cost_by_key,
    )
    skill_summary = _aggregate(
        skill_rows,
        resource_type="skill",
        recall_field="required_skill_recall_at_k",
        quality_fields=(
            "capability_choice_correct", "wrong_skill_use", "instruction_following_success",
            "constraint_violation", "task_success",
        ),
        cost_by_key=cost_by_key,
    )
    for output, rows in (
        ("tool_progressive_disclosure_summary.csv", tool_summary),
        ("skill_progressive_disclosure_summary.csv", skill_summary),
    ):
        _write_csv(RESULTS / output, [{key: _fmt(value, 6) if isinstance(value, float) else value for key, value in row.items()} for row in rows])

    discovery = _tool_discovery_summary() + _skill_discovery_summary()
    _write_csv(RESULTS / "capability_discovery_comparison.csv", discovery)
    schema_rows = _read(RESULTS / "tool_schema_size_distribution.csv")
    skill_catalog = [json.loads(line) for line in (RESULTS / "skill_catalog.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((RESULTS / "record_view_manifest.json").read_text(encoding="utf-8"))
    tool_sizes = [
        {
            "full_tokens": row["full_schema_tokens"],
            "selection_full_ratio": row["selection_full_ratio"],
        }
        for row in schema_rows
    ]
    skill_token_stats = manifest["skill_full_tokens"]
    distributions = [
        _distribution(tool_sizes, "tool"),
        {
            "resource_type": "skill",
            "records": len(skill_catalog),
            "median_full_tokens": skill_token_stats["median"],
            "p90_full_tokens": skill_token_stats["p90"],
            "p95_full_tokens": skill_token_stats["p95"],
            "max_full_tokens": skill_token_stats["max"],
            "median_selection_full_ratio": manifest["skill_selection_full_ratio_median"],
        },
    ]
    _write_csv(RESULTS / "capability_size_summary.csv", distributions)

    colors = {
        "full": "#247BA0",
        "selection": "#D1495B",
        "progressive": "#3B8C6E",
        "oracle": "#6B5CA5",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for axis, rows, title in ((axes[0], tool_summary, "Tools, K=8"), (axes[1], skill_summary, "Skills, K=8")):
        for row in rows:
            if int(row["max_candidates"]) != 8:
                continue
            condition = str(row["condition"])
            key = "progressive" if "to_full" in condition else "selection" if "selection_only" in condition else "oracle" if "oracle" in condition else "full"
            axis.scatter(row["total_capability_tokens"], row["task_success"], s=60, color=colors[key])
            axis.annotate(condition.split("_")[0], (row["total_capability_tokens"], row["task_success"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("Mean capability tokens exposed")
        axis.set_ylim(-0.04, 1.04)
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Strict task success")
    _save_figure("progressive_disclosure_pareto")

    labels = ("BM25", "Auto semantic", "Embedding", "Fused")
    policies = ("bm25", "auto_semantic", "embedding", "fused")
    tool_by = {row["policy"]: row for row in discovery if row["resource_type"] == "tool"}
    skill_by = {row["policy"]: row for row in discovery if row["resource_type"] == "skill"}
    x = list(range(len(labels)))
    plt.figure(figsize=(7.4, 3.5))
    plt.bar([value - .18 for value in x], [float(tool_by[policy]["top1"]) for policy in policies], width=.36, color="#247BA0", label="Tools (144 queries)")
    plt.bar([value + .18 for value in x], [float(skill_by[policy]["top1"]) for policy in policies], width=.36, color="#F2A541", label="Skills (25 test queries)")
    plt.xticks(x, labels)
    plt.ylim(0, 1.02)
    plt.ylabel("Top-1 discovery accuracy")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=.25)
    _save_figure("capability_discovery_generalization")

    figure, token_axis = plt.subplots(figsize=(6.8, 3.5))
    x = [0, 1]
    ratio_axis = token_axis.twinx()
    token_bars = token_axis.bar([value - .18 for value in x], [row["median_full_tokens"] for row in distributions], width=.36, color="#247BA0", label="Median full tokens")
    ratio_bars = ratio_axis.bar([value + .18 for value in x], [100 * row["median_selection_full_ratio"] for row in distributions], width=.36, color="#D1495B", label="Selection/full ratio")
    token_axis.set_xticks(x, ("Tool schemas", "Skill instructions"))
    token_axis.set_ylabel("Median full-view tokens", color="#247BA0")
    ratio_axis.set_ylabel("Selection/full ratio (%)", color="#D1495B")
    ratio_axis.set_ylim(0, 100)
    token_axis.legend((token_bars, ratio_bars), ("Median full tokens", "Selection/full ratio"), frameon=False, loc="upper left")
    token_axis.grid(axis="y", alpha=.25)
    _save_figure("capability_view_size_distribution")

    _write_tex_table(RESULTS / "generated_tool_progressive_table.tex", tool_summary, "tool")
    _write_tex_table(RESULTS / "generated_skill_progressive_table.tex", skill_summary, "skill")
    tool_by_key = {(int(row["max_candidates"]), row["condition"]): row for row in tool_summary}
    skill_by_key = {(int(row["max_candidates"]), row["condition"]): row for row in skill_summary}
    skill_discovery = {row["policy"]: row for row in discovery if row["resource_type"] == "skill"}
    macros = {
        "PaperSixFiveMNineToolCases": len({row["query_id"] for row in tool_rows}),
        "PaperSixFiveMNineSkillCases": len({row["query_id"] for row in skill_rows}),
        "PaperSixFiveMNineSkillCatalog": len(skill_catalog),
        "PaperSixFiveMNineToolKFourFullSuccess": tool_by_key[(4, "T0_full_all")]["task_success"],
        "PaperSixFiveMNineToolKFourProgressiveSuccess": tool_by_key[(4, "T2_selection_to_full")]["task_success"],
        "PaperSixFiveMNineToolKFourFullChoice": tool_by_key[(4, "T0_full_all")]["capability_choice_correct"],
        "PaperSixFiveMNineToolKFourProgressiveChoice": tool_by_key[(4, "T2_selection_to_full")]["capability_choice_correct"],
        "PaperSixFiveMNineToolKFourDisclosure": tool_by_key[(4, "T2_selection_to_full")]["total_disclosure_ratio"],
        "PaperSixFiveMNineToolKEightFullChoice": tool_by_key[(8, "T0_full_all")]["capability_choice_correct"],
        "PaperSixFiveMNineToolKEightProgressiveChoice": tool_by_key[(8, "T2_selection_to_full")]["capability_choice_correct"],
        "PaperSixFiveMNineToolKEightDisclosure": tool_by_key[(8, "T2_selection_to_full")]["total_disclosure_ratio"],
        "PaperSixFiveMNineToolKEightWallOverhead": tool_by_key[(8, "T2_selection_to_full")]["wall_clock_seconds"] - tool_by_key[(8, "T0_full_all")]["wall_clock_seconds"],
        "PaperSixFiveMNineSkillKFourFullSuccess": skill_by_key[(4, "S0_full_all")]["task_success"],
        "PaperSixFiveMNineSkillKFourProgressiveSuccess": skill_by_key[(4, "S2_selection_to_full")]["task_success"],
        "PaperSixFiveMNineSkillKEightFullSuccess": skill_by_key[(8, "S0_full_all")]["task_success"],
        "PaperSixFiveMNineSkillKEightProgressiveSuccess": skill_by_key[(8, "S2_selection_to_full")]["task_success"],
        "PaperSixFiveMNineSkillKEightDisclosure": skill_by_key[(8, "S2_selection_to_full")]["total_disclosure_ratio"],
        "PaperSixFiveMNineSkillKEightFullWall": skill_by_key[(8, "S0_full_all")]["wall_clock_seconds"],
        "PaperSixFiveMNineSkillKEightProgressiveWall": skill_by_key[(8, "S2_selection_to_full")]["wall_clock_seconds"],
        "PaperSixFiveMNineSkillAllFullFit": skill_by_key[(25, "S0_full_all")]["context_fit_rate"],
        "PaperSixFiveMNineSkillFusedTopOne": skill_discovery["fused"]["top1"],
        "PaperSixFiveMNineSkillBmTwentyTopOne": skill_discovery["bm25"]["top1"],
    }
    tex = []
    for name, value in macros.items():
        rendered = str(value) if isinstance(value, int) else f"{float(value):.3f}"
        tex.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
    (RESULTS / "generated_progressive_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    findings = {
        "protocol": {
            "tool_cases": macros["PaperSixFiveMNineToolCases"],
            "skill_cases": macros["PaperSixFiveMNineSkillCases"],
            "skill_catalog": macros["PaperSixFiveMNineSkillCatalog"],
            "matched_budgets": [2, 4, 6, 8],
            "stress_budgets": {"tool": 18, "skill": 25},
        },
        "tool_k8": {key: value for key, value in tool_by_key[(8, "T2_selection_to_full")].items() if key not in {"resource_type", "max_candidates", "condition"}},
        "skill_k8": {key: value for key, value in skill_by_key[(8, "S2_selection_to_full")].items() if key not in {"resource_type", "max_candidates", "condition"}},
        "skill_discovery": skill_discovery,
        "size_distributions": distributions,
        "paper7_boundary": "No callback, mid-token expansion, arbitrary compression, or insufficiency-triggered disclosure is implemented.",
    }
    (RESULTS / "progressive_disclosure_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
