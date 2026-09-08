"""Export paired quality intervals for a frozen expansion generation run."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "paper3.3-crossdoc-expansion-generation-publication-v1"
BOOTSTRAP_SEEDS = (11, 23, 37, 71, 101)
QUALITY_METRICS = (
    "token_f1",
    "exact_match",
    "official_multihop_rag_score",
    "gold_answer_mean_nll",
)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(probability * len(ordered)), len(ordered) - 1)]


def _bootstrap_interval(
    values: Sequence[float], *, seed: int, replicates: int
) -> tuple[float, float]:
    generator = random.Random(seed)
    draws = [
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(replicates)
    ]
    return _percentile(draws, 0.025), _percentile(draws, 0.975)


def summarize_generation(
    run: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    replicates: int = 2_000,
) -> dict[str, object]:
    """Summarize absolute quality and paired effects for every replay condition."""

    grouped: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), {})[str(row["example_id"])] = row
    if not grouped:
        raise ValueError("generation run contains no rows")

    absolute = []
    for condition, by_id in sorted(grouped.items()):
        metrics: dict[str, object] = {}
        for metric in QUALITY_METRICS:
            values = [float(row[metric]) for row in by_id.values() if row.get(metric) is not None]
            intervals = [
                _bootstrap_interval(values, seed=seed, replicates=replicates)
                for seed in BOOTSTRAP_SEEDS
            ]
            metrics[metric] = {
                "mean": statistics.fmean(values),
                "conservative_ci95": [
                    min(interval[0] for interval in intervals),
                    max(interval[1] for interval in intervals),
                ],
            }
        absolute.append(
            {"condition": condition, "examples": len(by_id), "metrics": metrics}
        )

    paired = []
    for reference_condition in ("PACKED_RAG", "INDEPENDENT_PRA"):
        reference = grouped.get(reference_condition)
        if reference is None:
            continue
        for condition, by_id in sorted(grouped.items()):
            if condition == reference_condition:
                continue
            common = sorted(set(reference).intersection(by_id))
            effects: dict[str, object] = {}
            for metric in QUALITY_METRICS:
                differences = [
                    float(by_id[example_id][metric])
                    - float(reference[example_id][metric])
                    for example_id in common
                    if by_id[example_id].get(metric) is not None
                    and reference[example_id].get(metric) is not None
                ]
                intervals = [
                    _bootstrap_interval(differences, seed=seed, replicates=replicates)
                    for seed in BOOTSTRAP_SEEDS
                ]
                effects[metric] = {
                    "mean_difference": statistics.fmean(differences),
                    "conservative_ci95": [
                        min(interval[0] for interval in intervals),
                        max(interval[1] for interval in intervals),
                    ],
                }
            paired.append(
                {
                    "condition": condition,
                    "reference_condition": reference_condition,
                    "paired_examples": len(common),
                    "effects": effects,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "git_commit": run["git_commit"],
            "model": run["model"],
            "model_revision": run["model_revision"],
            "split": run["split"],
            "selection_cache_sha256": run.get("selection_cache_sha256"),
            "policy": run["policy"],
            "examples": run["examples"],
        },
        "uncertainty": {
            "method": "paired_or_absolute_nonparametric_bootstrap",
            "seeds": list(BOOTSTRAP_SEEDS),
            "replicates_per_seed": replicates,
            "note": "seeds repeat resampling; they are not independent model runs",
        },
        "condition_summary": run["summary"],
        "absolute": absolute,
        "paired_effects": paired,
    }


def plot_generation(summary: Mapping[str, object], output: Path) -> None:
    """Plot absolute F1 and official score with conservative bootstrap intervals."""

    import matplotlib.pyplot as plt

    rows = summary["absolute"]
    display_names = {
        "CROSSDOC_EXPANSION_PAIR_SA": "Linked pair SA",
        "CROSSDOC_EXPANSION_PAIR_SA_BOUNDARY": "Linked boundary SA",
        "CROSSDOC_EXPANSION_TOP_ATTENTION": "Linked top-edge",
        "EXPANSION_ONLY": "Expansion only",
        "INDEPENDENT_PRA": "Independent PRA",
        "PAIR_SA_ONLY": "Pair SA only",
        "PAIR_SA_ONLY_BOUNDARY": "Boundary SA only",
        "PACKED_RAG": "Packed RAG",
    }
    labels = [display_names.get(str(row["condition"]), str(row["condition"])) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("token_f1", "official_multihop_rag_score"),
        ("Answer token F1", "Official MultiHop-RAG score"),
    ):
        values = [float(row["metrics"][metric]["mean"]) for row in rows]
        intervals = [row["metrics"][metric]["conservative_ci95"] for row in rows]
        errors = [
            [value - float(interval[0]) for value, interval in zip(values, intervals)],
            [float(interval[1]) - value for value, interval in zip(values, intervals)],
        ]
        axis.bar(range(len(rows)), values, color="#3078b8", alpha=0.85)
        axis.errorbar(range(len(rows)), values, yerr=errors, fmt="none", color="#202020")
        axis.set_xticks(range(len(rows)), labels, rotation=30, ha="right", fontsize=8)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "generation_quality.pdf", bbox_inches="tight")
    figure.savefig(output / "generation_quality.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines()]
    result = summarize_generation(run, rows, replicates=args.bootstrap_replicates)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "publication_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_generation(result, args.output)
    print(json.dumps({"output": str(args.output), "examples": run["examples"]}))


if __name__ == "__main__":
    main()
