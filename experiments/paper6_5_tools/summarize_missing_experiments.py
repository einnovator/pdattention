"""Summarize large-catalog palette use and progressive disclosure for Paper 6.5."""

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper6_5_tools.prepare_missing_experiments import OUTPUT
from pra_hf.progressive_disclosure import capability_choice_accounting


SCALING = ROOT / "docs/papers/shared/results/paper6_5_tools/auto_discovery_scaling"
M9 = ROOT / "docs/papers/shared/results/paper6_5_tools/progressive_disclosure"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Iterable[Mapping[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.mean(values) if values else 0.0


def _palette_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["catalog_size"]), int(row["max_candidates"]), row["policy"], row["candidate_view"])].append(row)
    output = []
    for (size, budget, policy, view), values in sorted(groups.items()):
        accounting = capability_choice_accounting(tuple(
            (bool(int(row["target_in_palette"])), bool(int(row["choice_correct"])))
            for row in values
        ))
        unsafe_choices = sum(int(row["unsafe_choice"]) for row in values)
        output.append({
            "catalog_size": size, "max_candidates": budget, "policy": policy,
            "candidate_view": view, "examples": accounting.examples,
            "catalog_seeds": len({row["seed"] for row in values}),
            "target_in_palette_count": accounting.target_in_palette,
            "correct_choice_count": accounting.correct_choices,
            "retrieval_recall": accounting.retrieval_recall,
            "conditional_choice_accuracy": accounting.conditional_choice_accuracy,
            "end_to_end_choice_accuracy": accounting.end_to_end_choice_accuracy,
            "unsafe_exposure": _mean(values, "unsafe_exposure"),
            "unsafe_choice_rate": unsafe_choices / len(values),
            "selection_view_tokens": _mean(values, "selection_view_tokens"),
            "all_candidate_full_tokens": _mean(values, "all_candidate_full_tokens"),
            "model_choice_seconds": _mean(values, "amortized_wall_seconds"),
        })
    return output


def _progressive_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["catalog_size"]), int(row["max_candidates"]), row["policy"], row["condition"])].append(row)
    output = []
    metrics = (
        "target_in_palette", "capability_choice_correct", "schema_valid_call",
        "required_argument_coverage", "argument_semantic_correct", "enum_type_valid",
        "execution_acceptance", "host_rejection", "task_success", "unsafe_tool_choice",
        "phase_a_capability_tokens", "phase_b_capability_tokens", "total_capability_tokens",
        "native_kv_bytes", "full_schemas_materialized", "full_schema_tokens_avoided",
        "disclosure_ratio_a", "disclosure_ratio_total", "total_ttft_seconds",
        "wall_clock_seconds", "materialization_seconds",
    )
    for (size, budget, policy, condition), values in sorted(groups.items()):
        row: dict[str, object] = {
            "catalog_size": size, "max_candidates": budget, "policy": policy,
            "condition": condition, "examples": len(values),
            "catalog_seeds": len({value["seed"] for value in values}),
        }
        row.update({metric: _mean(values, metric) for metric in metrics})
        output.append(row)
    return output


def _jit_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["policy"], int(row["max_candidates"]))].append(row)
    output = []
    for (policy, budget), values in sorted(groups.items()):
        workflows = {(row["seed"], row["workflow_id"]): int(row["workflow_success"]) for row in values}
        output.append({
            "policy": policy, "max_candidates": budget, "steps": len(values),
            "workflows": len(workflows), "workflow_success": statistics.mean(workflows.values()),
            "step_target_recall": _mean(values, "target_in_palette"),
            "conditional_choice_accuracy": sum(int(row["conditional_choice_correct"]) for row in values if int(row["target_in_palette"])) / max(sum(int(row["target_in_palette"]) for row in values), 1),
            "tool_call_acceptance": _mean(values, "tool_call_acceptance"),
            "wrong_tool_proposal": _mean(values, "wrong_tool_proposal"),
            "unsafe_tool_proposal": _mean(values, "unsafe_tool_proposal"),
            "host_rejection": _mean(values, "host_rejection"),
            "capability_tokens": _mean(values, "capability_tokens"),
            "full_schemas_materialized": _mean(values, "full_schemas_materialized"),
            "resolver_latency_seconds": _mean(values, "resolver_latency_seconds"),
            "model_latency_seconds": _mean(values, "model_latency_seconds"),
        })
    return output


def _exact_paired_p(left: Sequence[int], right: Sequence[int]) -> tuple[int, int, float]:
    """Return discordant counts and the exact two-sided sign/McNemar p-value."""

    left_only = sum(a == 1 and b == 0 for a, b in zip(left, right))
    right_only = sum(a == 0 and b == 1 for a, b in zip(left, right))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)) / (2 ** discordant)
    return left_only, right_only, min(1.0, 2 * tail)


def _paired_effects(palette_rows, progressive_rows) -> list[dict[str, object]]:
    output = []
    palette_key = lambda row: (row["seed"], row["query_id"])
    groups = {
        policy: {palette_key(row): int(row["choice_correct"]) for row in palette_rows
                 if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 10
                 and row["candidate_view"] == "compact" and row["policy"] == policy}
        for policy in ("A1_fused", "A2_raw_union", "A3_diversity_union", "A4_agreement_union")
    }
    for policy in ("A2_raw_union", "A3_diversity_union", "A4_agreement_union"):
        keys = sorted(set(groups["A1_fused"]) & set(groups[policy]))
        left = [groups[policy][key] for key in keys]
        right = [groups["A1_fused"][key] for key in keys]
        left_only, right_only, p = _exact_paired_p(left, right)
        output.append({
            "experiment": "palette_choice_8192_k10", "left": policy, "right": "A1_fused",
            "pairs": len(keys), "left_success": sum(left), "right_success": sum(right),
            "left_only": left_only, "right_only": right_only, "exact_two_sided_p": p,
        })
    progressive_key = lambda row: (row["seed"], row["query_id"])
    groups = {
        policy: {progressive_key(row): int(row["task_success"]) for row in progressive_rows
                 if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 8
                 and row["condition"] == "C2_compact_to_full" and row["policy"] == policy}
        for policy in ("A1_fused", "A2_raw_union", "A3_diversity_union")
    }
    for policy in ("A2_raw_union", "A3_diversity_union"):
        keys = sorted(set(groups["A1_fused"]) & set(groups[policy]))
        left = [groups[policy][key] for key in keys]
        right = [groups["A1_fused"][key] for key in keys]
        left_only, right_only, p = _exact_paired_p(left, right)
        output.append({
            "experiment": "progressive_task_8192_k8", "left": policy, "right": "A1_fused",
            "pairs": len(keys), "left_success": sum(left), "right_success": sum(right),
            "left_only": left_only, "right_only": right_only, "exact_two_sided_p": p,
        })
    return output


def _matched_view_quality(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    by = {(row["catalog_size"], row["seed"], row["query_id"], row["policy"], row["max_candidates"], row["candidate_view"]): row for row in rows}
    output = []
    for key, full in by.items():
        if key[-1] != "full_all":
            continue
        compact_key = (*key[:-1], "compact")
        compact = by.get(compact_key)
        if compact is None:
            continue
        output.append({
            "catalog_size": full["catalog_size"], "seed": full["seed"],
            "query_id": full["query_id"], "policy": full["policy"],
            "max_candidates": full["max_candidates"],
            "full_choice_correct": full["choice_correct"],
            "compact_choice_correct": compact["choice_correct"],
            "full_unsafe_choice": full["unsafe_choice"],
            "compact_unsafe_choice": compact["unsafe_choice"],
            "full_tokens": full["all_candidate_full_tokens"],
            "compact_tokens": compact["selection_view_tokens"],
            "compact_full_ratio": float(compact["selection_view_tokens"]) / max(float(full["all_candidate_full_tokens"]), 1),
        })
    return output


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def _figures(palette, progressive) -> None:
    retrieval = _read(SCALING / "scaled_union_frontier.csv")
    plt.figure(figsize=(6.2, 3.6))
    for strategy, label, color in (
        ("A6_topk", "Fused", "#2C6E9B"),
        ("A7_raw_union", "Raw union", "#C45A35"),
    ):
        rows = [row for row in retrieval if row["strategy"] == strategy and int(row["max_candidates"]) == 10]
        plt.plot([int(row["catalog_size"]) for row in rows], [float(row["required_recall_mean"]) for row in rows], marker="o", label=label, color=color)
    plt.xscale("log", base=2)
    plt.xlabel("Logical catalog size")
    plt.ylabel("Recall@10")
    plt.ylim(0, 1.02)
    plt.grid(alpha=.25)
    plt.legend(frameon=False)
    _save("large_n_retrieval_frontier")

    plt.figure(figsize=(6.2, 3.6))
    for policy, label, color in (
        ("A1_fused", "Fused", "#2C6E9B"),
        ("A2_raw_union", "Raw union", "#C45A35"),
    ):
        rows = [row for row in palette if int(row["catalog_size"]) == 8192 and row["policy"] == policy and row["candidate_view"] == "compact"]
        plt.plot([int(row["max_candidates"]) for row in rows], [float(row["conditional_choice_accuracy"]) for row in rows], marker="o", label=label, color=color)
    plt.xlabel("Candidate budget K")
    plt.ylabel("Conditional LLM choice accuracy")
    plt.ylim(0, 1.02)
    plt.grid(alpha=.25)
    plt.legend(frameon=False)
    _save("large_catalog_palette_use")

    plt.figure(figsize=(6.2, 3.8))
    markers = {"C0_full_all_one_pass": "o", "C1_compact_only": "s", "C2_compact_to_full": "^", "C3_oracle_to_full": "D"}
    colors = {
        "A1_fused": "#2C6E9B", "A2_raw_union": "#C45A35",
        "A3_diversity_union": "#3B8C6E",
    }
    for row in progressive:
        if int(row["catalog_size"]) != 8192 or int(row["max_candidates"]) != 8:
            continue
        plt.scatter(float(row["total_capability_tokens"]), float(row["task_success"]), marker=markers[row["condition"]], color=colors[row["policy"]], s=60)
    plt.xlabel("Mean capability tokens exposed")
    plt.ylabel("Strict correct accepted tool call")
    plt.ylim(-.02, 1.02)
    plt.grid(alpha=.25)
    _save("large_catalog_progressive_pareto")

    skills = _read(M9 / "skill_progressive_disclosure_summary.csv")
    plt.figure(figsize=(6.2, 3.8))
    for condition, marker, color in (
        ("S0_full_all", "o", "#2C6E9B"),
        ("S1_selection_only", "s", "#8A8A8A"),
        ("S2_selection_to_full", "^", "#3B8C6E"),
    ):
        rows = [row for row in skills if row["condition"] == condition and int(row["max_candidates"]) <= 8]
        plt.plot([float(row["total_capability_tokens"]) for row in rows], [float(row["task_success"]) for row in rows], marker=marker, color=color, label=condition.split("_", 1)[1])
    plt.xlabel("Mean skill tokens exposed")
    plt.ylabel("Strict skill-task success")
    plt.ylim(-.02, 1.02)
    plt.grid(alpha=.25)
    plt.legend(frameon=False)
    _save("skill_progressive_disclosure_frontier")


def _write_tex_tables(palette, progressive, jit) -> None:
    palette_rows = [row for row in palette if int(row["catalog_size"]) == 8192
                    and int(row["max_candidates"]) == 10 and row["candidate_view"] == "compact"]
    names = {
        "A0_bm25": "BM25", "A1_fused": "Fused", "A2_raw_union": "Raw union",
        "A3_diversity_union": "Diversity union", "A4_agreement_union": "Agreement union",
    }
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Policy & $n$ & Recall & Cond. choice & End-to-end & Unsafe choice & Tokens \\\\", "\\midrule"]
    for row in palette_rows:
        lines.append(
            f"{names[row['policy']]} & {row['examples']} & {float(row['retrieval_recall']):.3f} & "
            f"{float(row['conditional_choice_accuracy']):.3f} & {float(row['end_to_end_choice_accuracy']):.3f} & "
            f"{float(row['unsafe_choice_rate']):.3f} & {float(row['selection_view_tokens']):.0f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (OUTPUT / "generated_large_palette_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = [row for row in progressive if int(row["catalog_size"]) == 8192
            and int(row["max_candidates"]) == 8
            and row["condition"] in {"C0_full_all_one_pass", "C2_compact_to_full", "C3_oracle_to_full"}]
    conditions = {"C0_full_all_one_pass": "Full-all", "C2_compact_to_full": "Select$\\to$full", "C3_oracle_to_full": "Oracle full"}
    lines = ["\\begin{tabular}{llrrrrr}", "\\toprule", "Policy & View & Recall & Choice & Accepted & Strict & Tokens \\\\", "\\midrule"]
    for row in rows:
        lines.append(
            f"{names[row['policy']]} & {conditions[row['condition']]} & {float(row['target_in_palette']):.3f} & "
            f"{float(row['capability_choice_correct']):.3f} & {float(row['execution_acceptance']):.3f} & "
            f"{float(row['task_success']):.3f} & {float(row['total_capability_tokens']):.0f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (OUTPUT / "generated_large_progressive_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = [row for row in jit if int(row["max_candidates"]) == 8]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule", "Policy & Steps & Recall & Cond. choice & Accepted & Workflow & Tokens \\\\", "\\midrule"]
    for row in rows:
        lines.append(
            f"{names[row['policy']]} & {row['steps']} & {float(row['step_target_recall']):.3f} & "
            f"{float(row['conditional_choice_accuracy']):.3f} & {float(row['tool_call_acceptance']):.3f} & "
            f"{float(row['workflow_success']):.3f} & {float(row['capability_tokens']):.0f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (OUTPUT / "generated_large_jit_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> None:
    palette_rows = _read(OUTPUT / "large_catalog_palette_choice.csv")
    progressive_rows = _read(OUTPUT / "progressive_tool_disclosure.csv")
    jit_rows = _read(OUTPUT / "large_catalog_jit.csv")
    palette = _palette_summary(palette_rows)
    progressive = _progressive_summary(progressive_rows)
    jit = _jit_summary(jit_rows)
    _write(OUTPUT / "large_catalog_palette_choice_summary.csv", palette)
    _write(OUTPUT / "progressive_tool_disclosure_summary.csv", progressive)
    _write(OUTPUT / "large_catalog_jit_summary.csv", jit)
    _write(OUTPUT / "selection_view_quality.csv", _matched_view_quality(palette_rows))
    _write(OUTPUT / "paired_effects.csv", _paired_effects(palette_rows, progressive_rows))

    # Preserve the requested artifact names while keeping M9's original files.
    for source, target in (
        ("skill_catalog.jsonl", "skill_catalog.jsonl"),
        ("skill_semantic_hard_queries.jsonl", "skill_semantic_hard_queries.jsonl"),
        ("skill_discovery_results.csv", "skill_discovery.csv"),
        ("skill_progressive_disclosure_results.csv", "skill_progressive_disclosure.csv"),
    ):
        shutil.copyfile(M9 / source, OUTPUT / target)

    skill_summary = _read(M9 / "skill_progressive_disclosure_summary.csv")
    frontier = [
        {"resource_type": "tool", "catalog_size": row["catalog_size"], "max_candidates": row["max_candidates"], "policy": row["policy"], "condition": row["condition"], "tokens": row["total_capability_tokens"], "quality": row["task_success"]}
        for row in progressive
    ] + [
        {"resource_type": "skill", "catalog_size": 25, "max_candidates": row["max_candidates"], "policy": "fused", "condition": row["condition"], "tokens": row["total_capability_tokens"], "quality": row["task_success"]}
        for row in skill_summary
    ]
    _write(OUTPUT / "capability_context_frontier.csv", frontier)
    _figures(palette, progressive)
    _write_tex_tables(palette, progressive, jit)

    compact = next(row for row in palette if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 10 and row["policy"] == "A2_raw_union" and row["candidate_view"] == "compact")
    fused = next(row for row in palette if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 10 and row["policy"] == "A1_fused" and row["candidate_view"] == "compact")
    diversity = next(row for row in palette if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 10 and row["policy"] == "A3_diversity_union" and row["candidate_view"] == "compact")
    prog = next(row for row in progressive if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 8 and row["policy"] == "A2_raw_union" and row["condition"] == "C2_compact_to_full")
    full = next(row for row in progressive if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 8 and row["policy"] == "A2_raw_union" and row["condition"] == "C0_full_all_one_pass")
    jit_union = next(row for row in jit if row["policy"] == "A2_raw_union" and int(row["max_candidates"]) == 8)
    jit_fused = next(row for row in jit if row["policy"] == "A1_fused" and int(row["max_candidates"]) == 8)
    jit_diversity = next(row for row in jit if row["policy"] == "A3_diversity_union" and int(row["max_candidates"]) == 8)
    prog_diversity = next(row for row in progressive if int(row["catalog_size"]) == 8192 and int(row["max_candidates"]) == 8 and row["policy"] == "A3_diversity_union" and row["condition"] == "C2_compact_to_full")
    findings = {
        "palette_8192_k10": {"fused": fused, "raw_union": compact, "diversity_union": diversity},
        "progressive_8192_k8": {"raw_union_full_all": full, "raw_union_compact_to_full": prog, "diversity_compact_to_full": prog_diversity},
        "jit_8192_k8": {"fused": jit_fused, "raw_union": jit_union, "diversity_union": jit_diversity},
        "default_union_gate": "open" if float(jit_diversity["workflow_success"]) > float(jit_fused["workflow_success"]) and float(diversity["unsafe_choice_rate"]) <= float(fused["unsafe_choice_rate"]) else "closed",
        "choice_protocol": "Frozen Ollama qwen3:0.6b Q4 raw generation of a single-token label bound to exact in-palette identities.",
    }
    (OUTPUT / "missing_experiments_findings.json").write_text(json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8")
    macros = {
        "PaperSixFiveMissingUnionConditional": float(compact["conditional_choice_accuracy"]),
        "PaperSixFiveMissingFusedConditional": float(fused["conditional_choice_accuracy"]),
        "PaperSixFiveMissingUnionEndToEnd": float(compact["end_to_end_choice_accuracy"]),
        "PaperSixFiveMissingFusedEndToEnd": float(fused["end_to_end_choice_accuracy"]),
        "PaperSixFiveMissingDiversityConditional": float(diversity["conditional_choice_accuracy"]),
        "PaperSixFiveMissingDiversityEndToEnd": float(diversity["end_to_end_choice_accuracy"]),
        "PaperSixFiveMissingProgressiveTokens": float(prog["total_capability_tokens"]),
        "PaperSixFiveMissingFullTokens": float(full["total_capability_tokens"]),
        "PaperSixFiveMissingProgressiveSuccess": float(prog["task_success"]),
        "PaperSixFiveMissingFullSuccess": float(full["task_success"]),
        "PaperSixFiveMissingUnionJIT": float(jit_union["workflow_success"]),
        "PaperSixFiveMissingFusedJIT": float(jit_fused["workflow_success"]),
        "PaperSixFiveMissingDiversityJIT": float(jit_diversity["workflow_success"]),
    }
    lines = [f"\\newcommand{{\\{name}}}{{{value:.3f}}}" for name, value in macros.items()]
    (OUTPUT / "generated_missing_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    run()
