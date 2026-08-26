"""Build the final Paper 6.5 quality/cost curves from frozen artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/papers/shared/results/paper6_5_tools"
SCALING = BASE / "auto_discovery_scaling"
MISSING = BASE / "missing_experiments"
PROGRESSIVE = BASE / "progressive_disclosure"
OUTPUT = BASE / "final_curves"
KV_BYTES_PER_TOKEN = 114_688
POLICY_NAMES = {
    "A2_topk": "lexical",
    "A6_topk": "fused",
    "A7_fused_score": "equal_fusion",
    "A7_raw_union": "raw_union",
    "A7_diversity_union": "diversity_union",
    "A0_bm25": "bm25",
    "A1_fused": "fused",
    "A2_raw_union": "raw_union",
    "A3_diversity_union": "diversity_union",
}


def _write(name: str, frame: pd.DataFrame) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / name, index=False)


def _save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.savefig(OUTPUT / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close()


def _candidate_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(SCALING / "scaled_union_frontier.csv")
    rows = rows[rows.catalog_size == 8192].copy()
    rows["policy"] = rows.strategy.map(POLICY_NAMES).fillna(rows.strategy)
    recall = rows[[
        "catalog_size", "policy", "strategy", "max_candidates",
        "required_recall_mean", "required_recall_std", "useful_precision_mean",
        "unsafe_exposure_mean",
    ]].rename(columns={"required_recall_mean": "candidate_recall", "required_recall_std": "candidate_recall_std"})
    tokens = rows[[
        "catalog_size", "policy", "strategy", "max_candidates",
        "context_tokens_mean", "context_tokens_std",
    ]].rename(columns={"context_tokens_mean": "selection_tokens", "context_tokens_std": "selection_tokens_std"})
    return recall, tokens


def _choice_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = [
        OUTPUT / "tool_choice_k_curve_raw.csv",
        MISSING / "large_catalog_palette_choice.csv",
    ]
    source = next((path for path in sources if path.exists()), None)
    if source is None:
        empty = pd.DataFrame(columns=["catalog_size", "policy", "max_candidates"])
        return empty, empty
    rows = pd.read_csv(source)
    rows = rows[(rows.catalog_size == 8192) & (rows.candidate_view == "compact")].copy()
    rows["policy"] = rows.policy.map(POLICY_NAMES).fillna(rows.policy)
    group = rows.groupby(["catalog_size", "policy", "max_candidates"], as_index=False)
    summary = group.agg(
        examples=("query_id", "size"),
        target_in_palette=("target_in_palette", "sum"),
        correct_choices=("choice_correct", "sum"),
        candidate_recall=("target_in_palette", "mean"),
        end_to_end_choice=("choice_correct", "mean"),
        wrong_or_unsafe_choice=("unsafe_choice", "mean"),
    )
    summary["conditional_choice"] = summary.correct_choices / summary.target_in_palette.clip(lower=1)
    return (
        summary[[
            "catalog_size", "policy", "max_candidates", "examples",
            "target_in_palette", "correct_choices", "conditional_choice",
        ]],
        summary[[
            "catalog_size", "policy", "max_candidates", "examples",
            "candidate_recall", "conditional_choice", "end_to_end_choice",
            "wrong_or_unsafe_choice",
        ]],
    )


def _disclosure_curves() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    skill_source = pd.read_csv(PROGRESSIVE / "capability_disclosure_costs.csv")
    skill_source = skill_source[skill_source.resource_type == "skill"].copy()
    skill_source["policy"] = "fused"
    skill_rows = skill_source.groupby(
        ["resource_type", "policy", "max_candidates"], as_index=False
    ).agg(
        examples=("query_id", "size"),
        full_all_tokens=("all_candidate_full_tokens", "mean"),
        selection_tokens=("phase_a_selection_tokens", "mean"),
        selected_full_tokens=("phase_b_selected_full_tokens", "mean"),
        progressive_tokens=("progressive_tokens", "mean"),
        tokens_saved=("tokens_saved", "mean"),
        token_savings_fraction=("token_savings_fraction", "mean"),
    )
    palettes = []
    needed = set()
    with (MISSING / "large_catalog_palettes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["catalog_size"]) != 8192 or row["policy"] not in {
                "A1_fused", "A2_raw_union", "A3_diversity_union"
            }:
                continue
            palettes.append(row)
            needed.update((int(row["seed"]), name) for name in row["candidate_names"])
    views = {}
    with (MISSING / "progressive_tool_views.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (int(row["seed"]), row["name"])
            if int(row["catalog_size"]) == 8192 and key in needed:
                views[key] = (int(row["selection_tokens"]), int(row["full_tokens"]))
    tool_costs = []
    for row in palettes:
        sizes = [views[(int(row["seed"]), name)] for name in row["candidate_names"]]
        selection = sum(value[0] for value in sizes)
        full = sum(value[1] for value in sizes)
        selected_name = row["target_name"] if row["target_in_palette"] else row["candidate_names"][0]
        selected_full = views[(int(row["seed"]), selected_name)][1]
        progressive = selection + selected_full
        tool_costs.append({
            "resource_type": "tool",
            "policy": POLICY_NAMES[row["policy"]],
            "max_candidates": row["max_candidates"],
            "query_id": row["query_id"],
            "full_all_tokens": full,
            "selection_tokens": selection,
            "selected_full_tokens": selected_full,
            "progressive_tokens": progressive,
            "tokens_saved": full - progressive,
            "token_savings_fraction": 1 - progressive / max(full, 1),
        })
    tool_rows = pd.DataFrame(tool_costs).groupby(
        ["resource_type", "policy", "max_candidates"], as_index=False
    ).agg(
        examples=("query_id", "size"),
        full_all_tokens=("full_all_tokens", "mean"),
        selection_tokens=("selection_tokens", "mean"),
        selected_full_tokens=("selected_full_tokens", "mean"),
        progressive_tokens=("progressive_tokens", "mean"),
        tokens_saved=("tokens_saved", "mean"),
        token_savings_fraction=("token_savings_fraction", "mean"),
    )
    rows = pd.concat((tool_rows, skill_rows), ignore_index=True)
    token = rows.copy()
    kv = rows[["resource_type", "policy", "max_candidates", "examples"]].copy()
    kv["native_kv_bytes_per_token"] = KV_BYTES_PER_TOKEN
    kv["full_all_active_kv_bytes"] = rows.full_all_tokens * KV_BYTES_PER_TOKEN
    kv["progressive_active_kv_bytes"] = rows.progressive_tokens * KV_BYTES_PER_TOKEN
    kv["active_kv_bytes_saved"] = kv.full_all_active_kv_bytes - kv.progressive_active_kv_bytes
    kv["active_kv_savings_fraction"] = kv.active_kv_bytes_saved / kv.full_all_active_kv_bytes.clip(lower=1)
    savings = token[["resource_type", "policy", "max_candidates", "token_savings_fraction"]].merge(
        kv[["resource_type", "policy", "max_candidates", "active_kv_savings_fraction"]],
        on=["resource_type", "policy", "max_candidates"],
    )
    return token, kv, savings


def _kmin() -> pd.DataFrame:
    compact = []
    with (MISSING / "large_catalog_palettes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            compact.append({
                "catalog_size": row["catalog_size"],
                "seed": row["seed"],
                "query_id": row["query_id"],
                "policy": row["policy"],
                "max_candidates": row["max_candidates"],
                "target_in_palette": row["target_in_palette"],
            })
    rows = pd.DataFrame(compact)
    rows = rows[rows.catalog_size == 8192].copy()
    rows["policy"] = rows.policy.map(POLICY_NAMES).fillna(rows.policy)
    found = rows[rows.target_in_palette == 1]
    minima = found.groupby(["seed", "query_id", "policy"], as_index=False).max_candidates.min()
    identities = rows[["seed", "query_id", "policy"]].drop_duplicates()
    merged = identities.merge(minima, how="left", on=["seed", "query_id", "policy"])
    merged = merged.rename(columns={"max_candidates": "k_min"})
    merged["found_by_k48"] = merged.k_min.notna().astype(int)
    merged["k_min"] = merged.k_min.fillna(math.inf)
    return merged


def _jit() -> pd.DataFrame:
    path = MISSING / "large_catalog_jit.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    rows["policy"] = rows.policy.map(POLICY_NAMES).fillna(rows.policy)
    return rows.groupby(["policy", "max_candidates"], as_index=False).agg(
        steps=("step", "size"),
        workflows=("workflow_id", "nunique"),
        per_step_recall=("target_in_palette", "mean"),
        conditional_choice=("conditional_choice_correct", "mean"),
        step_success=("step_success", "mean"),
        workflow_success=("workflow_success", "mean"),
        mean_capability_tokens=("capability_tokens", "mean"),
    )


def _progressive_execution() -> pd.DataFrame:
    path = PROGRESSIVE / "skill_progressive_disclosure_results.csv"
    frames = []
    for resource_type, filename, recall in (
        ("tool", "tool_progressive_disclosure_results.csv", "required_tool_recall_at_k"),
        ("skill", "skill_progressive_disclosure_results.csv", "required_skill_recall_at_k"),
    ):
        candidate = PROGRESSIVE / filename
        if not candidate.exists():
            continue
        rows = pd.read_csv(candidate)
        rows["resource_type"] = resource_type
        rows["required_recall"] = rows[recall]
        frames.append(rows)
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    return rows.groupby(["resource_type", "max_candidates", "condition"], as_index=False).agg(
        cases=("query_id", "size"),
        context_fit=("context_fit", "mean"),
        required_recall=("required_recall", "mean"),
        capability_choice=("capability_choice_correct", "mean"),
        task_success=("task_success", "mean"),
        capability_tokens=("total_capability_tokens", "mean"),
        native_kv_bytes=("native_kv_bytes", "mean"),
        transition_seconds=("phase_transition_overhead_seconds", "mean"),
        wall_seconds=("wall_clock_seconds", "mean"),
    )


def _skill_discovery() -> pd.DataFrame:
    rows = pd.read_csv(PROGRESSIVE / "skill_discovery_results.csv")
    rows = rows[rows.split == "test"].copy()
    budgets = (1, 2, 3, 4, 5, 8, 16, 24, 32)
    output = []
    for policy, group in rows.groupby("policy"):
        for budget in budgets:
            output.append({
                "policy": policy,
                "max_candidates": budget,
                "queries": len(group),
                "candidate_recall": float((group["rank"] <= budget).mean()),
                "mrr": float(group.reciprocal_rank.mean()),
                "top1": float(group.top1.mean()),
            })
    return pd.DataFrame(output)


def _skill_breakdowns(execution: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sizes = pd.read_csv(PROGRESSIVE / "skill_view_size_distribution.csv")
    queries = pd.read_json(PROGRESSIVE / "skill_semantic_hard_queries.jsonl", lines=True)
    discovery = pd.read_csv(PROGRESSIVE / "skill_discovery_results.csv")
    discovery = discovery[(discovery.split == "test") & (discovery.policy == "fused")]
    detailed = discovery.merge(
        queries[["query_id", "target_skill", "family", "hardness_level", "query_style", "language"]],
        on=["query_id", "target_skill", "family", "hardness_level", "query_style", "language"],
    ).merge(sizes, left_on="target_skill", right_on="skill_name")
    disclosure_costs = pd.read_csv(PROGRESSIVE / "capability_disclosure_costs.csv")
    disclosure_costs = disclosure_costs[
        (disclosure_costs.resource_type == "skill")
        & (disclosure_costs.max_candidates == 8)
    ]
    detailed = detailed.merge(
        disclosure_costs[["query_id", "token_savings_fraction"]], on="query_id", how="left"
    )
    groups = []
    for dimension in ("family", "hardness_level", "query_style", "language", "length_bucket"):
        summary = detailed.groupby(dimension, as_index=False).agg(
            queries=("query_id", "size"),
            top1=("top1", "mean"),
            recall_at_3=("recall_at_3", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            recall_at_8=("recall_at_8", "mean"),
            mrr=("reciprocal_rank", "mean"),
            selection_tokens=("selection_tokens", "mean"),
            instruction_tokens=("instruction_tokens", "mean"),
            full_tokens=("full_tokens", "mean"),
            selection_full_ratio=("selection_full_ratio", "mean"),
            token_saving=("token_savings_fraction", "mean"),
        )
        summary.insert(0, "dimension", dimension)
        summary = summary.rename(columns={dimension: "group"})
        groups.append(summary)
    execution_rows = pd.read_csv(PROGRESSIVE / "skill_progressive_disclosure_results.csv")
    execution_rows = execution_rows[
        (execution_rows.condition == "S2_selection_to_full")
        & (execution_rows.max_candidates == 8)
    ].merge(sizes, left_on="target_name", right_on="skill_name")
    execution_rows = execution_rows.merge(
        disclosure_costs[["query_id", "token_savings_fraction"]], on="query_id", how="left"
    )
    execution_by_length = execution_rows.groupby("length_bucket", as_index=False).agg(
        cases=("query_id", "size"),
        instruction_tokens=("instruction_tokens", "mean"),
        full_tokens=("full_tokens", "mean"),
        selection_tokens=("selection_tokens", "mean"),
        token_saving=("token_savings_fraction", "mean"),
        context_fit=("context_fit", "mean"),
        choice_accuracy=("capability_choice_correct", "mean"),
        instruction_following=("instruction_following_success", "mean"),
        task_success=("task_success", "mean"),
    )
    return pd.concat(groups, ignore_index=True), execution_by_length


def _mixed_summary() -> pd.DataFrame:
    path = OUTPUT / "mixed_capability_k_curve.csv"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    return rows.groupby(["target_type", "max_candidates"], as_index=False).agg(
        examples=("query_id", "size"),
        target_recall=("target_in_palette", "mean"),
        type_accuracy=("capability_type_correct", "mean"),
        resource_accuracy=("resource_choice_correct", "mean"),
        wrong_type_choice=("wrong_type_choice", "mean"),
        selection_tokens=("selection_tokens", "mean"),
        full_all_tokens=("full_all_tokens", "mean"),
        progressive_tokens=("progressive_tokens_oracle", "mean"),
    )


def _skill_length_table(breakdowns: pd.DataFrame, execution: pd.DataFrame) -> None:
    discovery = breakdowns[breakdowns.dimension == "length_bucket"].set_index("group")
    use = execution.set_index("length_bucket")
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Bucket & $n$ & Instr. & Select & Full & Saved & R@8 & Choice & Success \\",
        r"\midrule",
    ]
    for bucket in ("short", "medium", "long"):
        row = discovery.loc[bucket]
        result = use.loc[bucket] if bucket in use.index else None
        choice = "--" if result is None else f"{float(result.choice_accuracy):.2f}"
        success = "--" if result is None else f"{float(result.task_success):.2f}"
        lines.append(
            f"{bucket.title()} & {int(row.queries)} & {float(row.instruction_tokens):.0f} & "
            f"{float(row.selection_tokens):.0f} & {float(row.full_tokens):.0f} & "
            f"{float(row.token_saving):.2f} & {float(row.recall_at_8):.2f} & "
            f"{choice} & {success} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (OUTPUT / "generated_skill_length_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _tool_choice_high_k_table(end_to_end: pd.DataFrame) -> None:
    """Write the measured diversity-union palette curve without interpolation."""

    selected = end_to_end[
        (end_to_end.policy == "diversity_union")
        & (end_to_end.max_candidates.isin((10, 12, 16, 24, 32)))
    ].sort_values("max_candidates")
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$K$ & Rows & Target present & Recall & Conditional & End-to-end \\",
        r"\midrule",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"{int(row.max_candidates)} & {int(row.examples)} & "
            f"{int(round(row.candidate_recall * row.examples))} & "
            f"{row.candidate_recall:.3f} & {row.conditional_choice:.3f} & "
            f"{row.end_to_end_choice:.3f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (OUTPUT / "generated_tool_choice_high_k_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plots(recall: pd.DataFrame, choice: pd.DataFrame, end: pd.DataFrame, savings: pd.DataFrame, skills: pd.DataFrame) -> None:
    focus = ("fused", "raw_union", "diversity_union")
    colors = {"fused": "#247BA0", "raw_union": "#D1495B", "diversity_union": "#3B8C6E"}
    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.7))
    for policy in focus:
        selected = recall[recall.policy == policy].sort_values("max_candidates")
        axes[0, 0].plot(selected.max_candidates, selected.candidate_recall, marker="o", color=colors[policy], label=policy.replace("_", " "))
        selected = choice[choice.policy == policy].sort_values("max_candidates")
        axes[0, 1].plot(selected.max_candidates, selected.conditional_choice, marker="o", color=colors[policy], label=policy.replace("_", " "))
        selected = end[end.policy == policy].sort_values("max_candidates")
        axes[1, 0].plot(selected.max_candidates, selected.end_to_end_choice, marker="o", color=colors[policy], label=policy.replace("_", " "))
    for kind, policy, color in (("tool", "diversity_union", "#247BA0"), ("skill", "fused", "#F2A541")):
        selected = savings[
            (savings.resource_type == kind) & (savings.policy == policy)
        ].sort_values("max_candidates")
        axes[1, 1].plot(selected.max_candidates, 100 * selected.token_savings_fraction, marker="o", color=color, label=kind)
    labels = (("Required-capability recall", "Candidate budget K"), ("Conditional LLM choice", "Candidate budget K"), ("End-to-end capability choice", "Candidate budget K"), ("Progressive token saving (%)", "Candidate budget K"))
    for axis, (ylabel, xlabel) in zip(axes.flat, labels):
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].legend(frameon=False, fontsize=8)
    _save("final_quality_cost_frontier")

    plt.figure(figsize=(6.4, 3.7))
    for policy, color in (("fused", "#3B8C6E"), ("embedding", "#6B5CA5"), ("bm25", "#247BA0")):
        selected = skills[skills.policy == policy].sort_values("max_candidates")
        plt.plot(selected.max_candidates, selected.candidate_recall, marker="o", label=policy, color=color)
    plt.xlabel("Skill candidate budget K")
    plt.ylabel("Target-skill recall")
    plt.ylim(0, 1.02)
    plt.grid(alpha=.25)
    plt.legend(frameon=False)
    _save("skill_k_curve")


def _architecture_figure() -> None:
    fig, axis = plt.subplots(figsize=(9.3, 3.4))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = (
        (0.2, 1.35, 1.35, 1.3, "Typed registry\ntools + skills", "#E8F1F8"),
        (1.8, 1.35, 1.35, 1.3, "External\ndiscovery/index", "#F4E9D8"),
        (3.4, 1.35, 1.35, 1.3, "Lazy selection\nview palette", "#E8F1F8"),
        (5.0, 1.35, 1.35, 1.3, "Model chooses\nstable identity", "#EEF4E7"),
        (6.6, 1.35, 1.35, 1.3, "Runtime activates\nexact full view", "#EEF4E7"),
        (8.35, 2.25, 1.4, 1.0, "Tool schema\narguments + host", "#F5E3E3"),
        (8.35, 0.75, 1.4, 1.0, "Skill instructions\nguided generation", "#E9E4F4"),
    )
    for x, y, width, height, label, color in boxes:
        axis.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.05",
            facecolor=color, edgecolor="#333333", linewidth=1,
        ))
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)
    for start, end in ((1.55, 1.8), (3.15, 3.4), (4.75, 5.0), (6.35, 6.6)):
        axis.annotate("", xy=(end, 2.0), xytext=(start, 2.0), arrowprops={"arrowstyle": "->", "lw": 1.3})
    axis.annotate("", xy=(8.35, 2.75), xytext=(7.95, 2.25), arrowprops={"arrowstyle": "->", "lw": 1.3})
    axis.annotate("", xy=(8.35, 1.25), xytext=(7.95, 1.75), arrowprops={"arrowstyle": "->", "lw": 1.3})
    axis.text(4.1, 3.25, "Broad discovery stays outside model context", ha="center", fontsize=9, color="#444444")
    axis.text(6.7, 0.28, "No semantic rediscovery after identity selection", ha="center", fontsize=9, color="#444444")
    _save("typed_capability_architecture")


def _lazy_figure() -> None:
    rows = pd.read_csv(OUTPUT / "lazy_encoding_economics.csv")
    metrics = (
        ("lazy_registration_seconds", "register"),
        ("lazy_first_selection_seconds", "first selection"),
        ("lazy_first_full_mean_seconds", "first full"),
        ("lazy_warm_full_mean_seconds", "warm full"),
    )
    positions = range(len(metrics))
    width = 0.35
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    for offset, resource_type, color in ((-width / 2, "tool", "#247BA0"), (width / 2, "skill", "#F2A541")):
        row = rows[rows.resource_type == resource_type].iloc[0]
        axis.bar(
            [value + offset for value in positions],
            [float(row[field]) * 1000 for field, _label in metrics],
            width=width,
            label=resource_type,
            color=color,
        )
    axis.set_yscale("log")
    axis.set_xticks(list(positions), [label for _field, label in metrics])
    axis.set_ylabel("Local runtime latency (ms, log scale)")
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False)
    _save("lazy_encoding_lifecycle")


def _mixed_figure(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
    for axis, target_type, color in (
        (axes[0], "tool", "#247BA0"),
        (axes[1], "skill", "#F2A541"),
    ):
        selected = rows[rows.target_type == target_type].sort_values("max_candidates")
        axis.plot(selected.max_candidates, selected.target_recall, marker="o", color=color, label="retrieval")
        axis.plot(selected.max_candidates, selected.type_accuracy, marker="s", color="#3B8C6E", label="type")
        axis.plot(selected.max_candidates, selected.resource_accuracy, marker="^", color="#D1495B", label="resource")
        axis.set_title(f"{target_type} targets")
        axis.set_xlabel("Mixed palette K")
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Accuracy / recall")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=8)
    _save("mixed_capability_palette")


def main() -> None:
    recall, token_cost = _candidate_curves()
    choice, end = _choice_curves()
    progressive_tokens, kv, savings = _disclosure_curves()
    kmin = _kmin()
    jit = _jit()
    execution = _progressive_execution()
    skills = _skill_discovery()
    skill_breakdowns, skill_execution_by_length = _skill_breakdowns(execution)
    mixed = _mixed_summary()
    _write("k_curve_candidate_recall.csv", recall)
    _write("k_curve_conditional_choice.csv", choice)
    _write("k_curve_end_to_end.csv", end)
    _tool_choice_high_k_table(end)
    _write("k_curve_progressive_execution.csv", execution)
    _write("k_curve_token_cost.csv", token_cost)
    _write("k_curve_kv_cost.csv", kv)
    _write("k_curve_savings.csv", savings)
    _write("kmin_distribution.csv", kmin)
    _write("jit_k_sweep.csv", jit)
    _write("skill_k_curve.csv", skills)
    _write("skill_discovery_breakdowns.csv", skill_breakdowns)
    _write("skill_execution_by_length.csv", skill_execution_by_length)
    _skill_length_table(skill_breakdowns, skill_execution_by_length)
    if not mixed.empty:
        _write("mixed_capability_summary.csv", mixed)
    _write("capability_progressive_token_cost.csv", progressive_tokens)
    _plots(recall, choice, end, savings, skills)
    _architecture_figure()
    _lazy_figure()
    _mixed_figure(mixed)

    focus = recall[(recall.policy == "diversity_union")].set_index("max_candidates")
    choice_focus = end[(end.policy == "diversity_union")].set_index("max_candidates")
    skill_focus = skills[skills.policy == "fused"].set_index("max_candidates")
    lazy = pd.read_csv(OUTPUT / "lazy_encoding_economics.csv")
    findings = {
        "protocol": {
            "tool_catalog": 8192,
            "tool_seeds": 5,
            "tool_discovery_queries_per_seed": 144,
            "tool_model_choice_queries_per_seed": 8,
            "skill_catalog": 50,
            "skill_test_queries": int(skills.queries.max()),
            "native_kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        },
        "tool_diversity_recall": {
            str(k): float(focus.loc[k, "candidate_recall"])
            for k in (8, 16, 24, 32, 48)
        },
        "tool_diversity_model_choice": {
            str(k): {
                "candidate_recall": float(choice_focus.loc[k, "candidate_recall"]),
                "conditional_choice": float(choice_focus.loc[k, "conditional_choice"]),
                "end_to_end_choice": float(choice_focus.loc[k, "end_to_end_choice"]),
            }
            for k in (10, 12, 16, 24, 32)
        },
        "skill_fused_recall": {
            str(k): float(skill_focus.loc[k, "candidate_recall"])
            for k in (4, 8, 16, 24, 32)
        },
        "lazy_runtime": lazy.to_dict(orient="records"),
        "interpretation": (
            "Typed progressive disclosure buys candidate breadth economically, but "
            "candidate recall, conditional model choice, and host authorization remain distinct gates."
        ),
    }
    (OUTPUT / "final_capability_findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
