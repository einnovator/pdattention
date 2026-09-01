"""Summarize the matched 0.6B/14B capability-choice decomposition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/papers/shared/results/paper6_5_tools"
FINAL = BASE / "final_curves"
PROGRESSIVE = BASE / "progressive_disclosure"
OUTPUT = BASE / "chooser_decomposition"
K_VALUES = (10, 16, 32)
SMALL = "Qwen3-0.6B Q4"
LARGE = "Qwen3-14B 4-bit"


def _small_choice() -> pd.DataFrame:
    rows = pd.read_csv(FINAL / "tool_choice_k_curve_raw.csv")
    rows = rows[
        (rows.catalog_size == 8192)
        & (rows.policy == "A3_diversity_union")
        & rows.max_candidates.isin(K_VALUES)
    ].copy()
    rows["model"] = SMALL
    return rows


def _large_choice() -> pd.DataFrame:
    rows = pd.read_csv(OUTPUT / "qwen3_14b_mlx_choice.csv")
    rows["model"] = LARGE
    return rows


def _validate_frozen(small: pd.DataFrame, large: pd.DataFrame) -> None:
    key = ["seed", "query_id", "max_candidates"]
    fields = key + ["target_name", "target_in_palette", "candidate_names"]
    left = small[fields].sort_values(key).reset_index(drop=True).astype(str)
    right = large[fields].sort_values(key).reset_index(drop=True).astype(str)
    if not left.equals(right):
        raise RuntimeError("The stronger-model run did not use the frozen 0.6B palettes.")


def _small_execution() -> pd.DataFrame:
    rows = pd.read_csv(PROGRESSIVE / "tool_progressive_disclosure_results.csv")
    rows = rows[rows.condition == "T3_oracle_full"].copy()
    fields = [
        "query_id", "target_name", "call_parse_valid", "argument_semantic_correct",
        "task_success",
    ]
    # The oracle condition is repeated at each K but is deterministic here.
    for _, group in rows.groupby("query_id"):
        if group[fields[2:]].drop_duplicates().shape[0] != 1:
            raise RuntimeError("The 0.6B oracle execution result varies across repeated K rows.")
    rows = rows.groupby("query_id", as_index=False).first()[fields]
    rows = rows.rename(columns={
        "argument_semantic_correct": "arguments_correct",
        "task_success": "execution_correct",
    })
    rows["tool_correct"] = 1
    rows["model"] = SMALL
    return rows


def _large_execution() -> pd.DataFrame:
    rows = pd.read_csv(OUTPUT / "qwen3_14b_mlx_execution.csv")
    rows["model"] = LARGE
    return rows


def summarize() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    small_choice = _small_choice()
    large_choice = _large_choice()
    _validate_frozen(small_choice, large_choice)
    choices = pd.concat([small_choice, large_choice], ignore_index=True)

    execution = pd.concat([_small_execution(), _large_execution()], ignore_index=True)
    execution_map = execution.set_index(["model", "query_id"])["execution_correct"]
    choices["execution_given_oracle"] = [
        int(execution_map[(row.model, row.query_id)]) for row in choices.itertuples()
    ]
    choices["pipeline_execution"] = (
        choices.choice_correct.astype(int) * choices.execution_given_oracle.astype(int)
    )

    summary = choices.groupby(["model", "max_candidates"], as_index=False).agg(
        examples=("query_id", "size"),
        target_present=("target_in_palette", "sum"),
        correct_choices=("choice_correct", "sum"),
        pipeline_successes=("pipeline_execution", "sum"),
        discovery_recall=("target_in_palette", "mean"),
        end_to_end_choice=("choice_correct", "mean"),
        end_to_end_execution=("pipeline_execution", "mean"),
    )
    summary["conditional_choice"] = (
        summary.correct_choices / summary.target_present.clip(lower=1)
    )
    summary = summary[[
        "model", "max_candidates", "examples", "target_present", "correct_choices",
        "pipeline_successes", "discovery_recall", "conditional_choice",
        "end_to_end_choice", "end_to_end_execution",
    ]]
    summary.to_csv(OUTPUT / "chooser_decomposition_summary.csv", index=False)

    execution_summary = execution.groupby("model", as_index=False).agg(
        cases=("query_id", "size"),
        parse_valid=("call_parse_valid", "mean"),
        tool_correct=("tool_correct", "mean"),
        arguments_correct=("arguments_correct", "mean"),
        execution_correct=("execution_correct", "mean"),
    )
    execution_summary.to_csv(OUTPUT / "oracle_execution_summary.csv", index=False)

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Chooser & $K$ & Present & $A_K$ & $E_K$ & Exec. ceiling & Full pipeline \\",
        r"\midrule",
    ]
    execution_by_model = execution_summary.set_index("model")
    for model in (SMALL, LARGE):
        label = "Qwen3-0.6B" if model == SMALL else "Qwen3-14B"
        for row in summary[summary.model == model].sort_values("max_candidates").itertuples():
            lines.append(
                f"{label} & {int(row.max_candidates)} & {int(row.target_present)}/{int(row.examples)} & "
                f"{row.conditional_choice:.3f} & {row.end_to_end_choice:.3f} & "
                f"{execution_by_model.loc[model, 'execution_correct']:.3f} & "
                f"{row.end_to_end_execution:.3f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUTPUT / "generated_chooser_decomposition_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharex=True)
    metrics = (
        ("conditional_choice", "Choice | target present"),
        ("end_to_end_choice", "Discovery + choice"),
        ("end_to_end_execution", "Discovery + choice + execution"),
    )
    colors = {SMALL: "#3B6EA8", LARGE: "#C45A3C"}
    markers = {SMALL: "o", LARGE: "s"}
    for axis, (metric, title) in zip(axes, metrics):
        for model in (SMALL, LARGE):
            rows = summary[summary.model == model].sort_values("max_candidates")
            axis.plot(
                rows.max_candidates, rows[metric], marker=markers[model],
                color=colors[model], linewidth=1.8, label=model,
            )
        axis.set_title(title, fontsize=9)
        axis.set_ylim(0, 0.75)
        axis.set_xticks(K_VALUES)
        axis.set_xlabel("Palette size K")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(OUTPUT / "chooser_decomposition.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "chooser_decomposition.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    summarize()
