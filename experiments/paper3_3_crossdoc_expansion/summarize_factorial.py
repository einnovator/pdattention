"""Summarize expansion-by-interaction and context-budget generation controls."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "paper3.3-crossdoc-factorial-publication-v1"
BOOTSTRAP_SEEDS = (11, 23, 37, 71, 101)
METRICS = ("token_f1", "official_multihop_rag_score")
CONDITIONS = (
    "PACKED_RAG",
    "INDEPENDENT_PRA",
    "PAIR_SA_ONLY",
    "PAIR_SA_ONLY_BOUNDARY",
    "EXPANSION_ONLY",
    "CROSSDOC_EXPANSION_PAIR_SA",
    "CROSSDOC_EXPANSION_PAIR_SA_BOUNDARY",
)
DISPLAY = {
    "PACKED_RAG": "Packed RAG",
    "INDEPENDENT_PRA": "Independent PRA",
    "PAIR_SA_ONLY": "Pair SA only",
    "PAIR_SA_ONLY_BOUNDARY": "Boundary SA only",
    "EXPANSION_ONLY": "Expansion only",
    "CROSSDOC_EXPANSION_PAIR_SA": "Expansion + pair SA",
    "CROSSDOC_EXPANSION_PAIR_SA_BOUNDARY": "Expansion + boundary SA",
}


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(probability * len(ordered)), len(ordered) - 1)]


def _interval(values: Sequence[float], *, replicates: int) -> list[float]:
    intervals = []
    for seed in BOOTSTRAP_SEEDS:
        generator = random.Random(seed)
        draws = [
            statistics.fmean(generator.choice(values) for _ in values)
            for _ in range(replicates)
        ]
        intervals.append((_percentile(draws, 0.025), _percentile(draws, 0.975)))
    return [min(row[0] for row in intervals), max(row[1] for row in intervals)]


def _index(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, Mapping[str, object]]]:
    result: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        result.setdefault(str(row["condition"]), {})[str(row["example_id"])] = row
    return result


def _effect(
    left: Mapping[str, Mapping[str, object]],
    right: Mapping[str, Mapping[str, object]],
    metric: str,
    *,
    replicates: int,
) -> dict[str, object]:
    common = sorted(set(left).intersection(right))
    values = [float(left[key][metric]) - float(right[key][metric]) for key in common]
    return {
        "pairs": len(values),
        "mean": statistics.fmean(values),
        "ci95": _interval(values, replicates=replicates),
    }


def build_factorial_summary(
    runs: Mapping[int, tuple[Mapping[str, object], Sequence[Mapping[str, object]]]],
    *,
    replicates: int = 2_000,
) -> dict[str, object]:
    """Build paired budget rows and the expansion-by-interaction decomposition."""

    budget_rows = []
    indices = {}
    cohort_ids = None
    for budget, (run, rows) in sorted(runs.items()):
        index = _index(rows)
        indices[budget] = index
        current_ids = set(index["INDEPENDENT_PRA"])
        if cohort_ids is None:
            cohort_ids = current_ids
        elif current_ids != cohort_ids:
            raise ValueError("budget runs do not contain the same frozen identities")
        for condition in CONDITIONS:
            by_id = index.get(condition)
            if not by_id:
                continue
            item: dict[str, object] = {
                "budget": budget,
                "condition": condition,
                "examples": len(by_id),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in by_id.values()]
                item[metric] = statistics.fmean(values)
                item[metric + "_ci95"] = _interval(values, replicates=replicates)
                if condition != "INDEPENDENT_PRA":
                    item[metric + "_vs_independent"] = _effect(
                        by_id,
                        index["INDEPENDENT_PRA"],
                        metric,
                        replicates=replicates,
                    )
            budget_rows.append(item)

    factorial = {}
    index = indices[min(indices)]
    required = {
        "none_none": "INDEPENDENT_PRA",
        "none_interaction": "PAIR_SA_ONLY",
        "expansion_none": "EXPANSION_ONLY",
        "expansion_interaction": "CROSSDOC_EXPANSION_PAIR_SA",
    }
    if all(condition in index for condition in required.values()):
        for metric in METRICS:
            common = sorted(
                set.intersection(*(set(index[condition]) for condition in required.values()))
            )
            values = []
            for key in common:
                cell = {
                    name: float(index[condition][key][metric])
                    for name, condition in required.items()
                }
                values.append(
                    (cell["expansion_interaction"] - cell["expansion_none"])
                    - (cell["none_interaction"] - cell["none_none"])
                )
            factorial[metric] = {
                "pairs": len(values),
                "interaction_effect": statistics.fmean(values),
                "ci95": _interval(values, replicates=replicates),
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "budgets": sorted(runs),
        "examples": len(cohort_ids or ()),
        "model": next(iter(runs.values()))[0]["model"],
        "budget_rows": budget_rows,
        "factorial_at_smallest_budget": factorial,
        "bootstrap": {"seeds": list(BOOTSTRAP_SEEDS), "replicates_per_seed": replicates},
    }


def _write_tables(summary: Mapping[str, object], output: Path) -> None:
    rows = summary["budget_rows"]
    with (output / "budget_conditions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("budget", "condition", "examples", *METRICS),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    selected = [
        row
        for row in rows
        if row["condition"]
        in {"PACKED_RAG", "INDEPENDENT_PRA", "PAIR_SA_ONLY", "PAIR_SA_ONLY_BOUNDARY"}
    ]
    lines = [
        "\\begin{tabular}{rlrrr}",
        "\\toprule",
        "Budget & Condition & $n$ & F1 & Official \\\\",
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            f'{int(row["budget"]):,} & {DISPLAY[str(row["condition"])]} & '
            f'{int(row["examples"])} & {float(row["token_f1"]):.4f} & '
            f'{float(row["official_multihop_rag_score"]):.4f} \\\\'
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_budget_interaction_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot(summary: Mapping[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = summary["budget_rows"]
    conditions = ("PACKED_RAG", "INDEPENDENT_PRA", "PAIR_SA_ONLY", "PAIR_SA_ONLY_BOUNDARY")
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    for axis, metric, ylabel in zip(
        axes,
        METRICS,
        ("Answer token F1", "Official MultiHop-RAG score"),
    ):
        for condition in conditions:
            points = sorted(
                (row for row in rows if row["condition"] == condition),
                key=lambda row: int(row["budget"]),
            )
            axis.plot(
                [int(row["budget"]) for row in points],
                [float(row[metric]) for row in points],
                marker="o",
                label=DISPLAY[condition],
            )
        axis.set_xlabel("Selected source-token budget")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="best")
    figure.savefig(output / "budget_interaction_quality.pdf", bbox_inches="tight")
    figure.savefig(output / "budget_interaction_quality.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _load_run(path: Path) -> tuple[Mapping[str, object], Sequence[Mapping[str, object]]]:
    return (
        json.loads((path / "summary.json").read_text(encoding="utf-8")),
        [json.loads(line) for line in (path / "rows.jsonl").read_text(encoding="utf-8").splitlines()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-512", type=Path, required=True)
    parser.add_argument("--run-1024", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    args = parser.parse_args()
    summary = build_factorial_summary(
        {512: _load_run(args.run_512), 1024: _load_run(args.run_1024)},
        replicates=args.bootstrap_replicates,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "publication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_tables(summary, args.output)
    _plot(summary, args.output)
    print(json.dumps({"output": str(args.output), "examples": summary["examples"]}))


if __name__ == "__main__":
    main()
