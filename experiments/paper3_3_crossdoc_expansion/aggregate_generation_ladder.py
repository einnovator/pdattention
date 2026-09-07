"""Combine frozen cross-document generation summaries across model sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "paper3.3-crossdoc-expansion-model-ladder-v1"
CONDITIONS = (
    "EXPANSION_ONLY",
    "CROSSDOC_EXPANSION_PAIR_SA",
    "CROSSDOC_EXPANSION_PAIR_SA_BOUNDARY",
)


def build_ladder(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Extract matched effects against independent PRA from each model run."""

    models = []
    for summary in summaries:
        source = summary["source"]
        effects = {
            str(row["condition"]): row
            for row in summary["paired_effects"]
            if row["reference_condition"] == "INDEPENDENT_PRA"
            and row["condition"] in CONDITIONS
        }
        if set(effects) != set(CONDITIONS):
            missing = sorted(set(CONDITIONS).difference(effects))
            raise ValueError(f"model summary lacks ladder conditions: {missing}")
        models.append(
            {
                "model": source["model"],
                "model_revision": source["model_revision"],
                "examples": source["examples"],
                "selection_cache_sha256": source["selection_cache_sha256"],
                "effects_vs_independent_pra": [effects[name] for name in CONDITIONS],
            }
        )
    cache_digests = {str(row["selection_cache_sha256"]) for row in models}
    if len(cache_digests) != 1:
        raise ValueError("model ladder must replay one frozen selection cache")
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_cache_sha256": next(iter(cache_digests)),
        "models": models,
    }


def plot_ladder(ladder: Mapping[str, object], output: Path) -> None:
    """Plot paired answer effects relative to independent PRA."""

    import matplotlib.pyplot as plt
    import numpy as np

    models = ladder["models"]
    labels = [str(row["model"]).split("/")[-1].replace("-4bit", "") for row in models]
    colors = ("#3078b8", "#d47726", "#3a9360")
    display = ("Expansion only", "Linked pair SA", "Linked boundary SA")
    x = np.arange(len(models), dtype=float)
    width = 0.24
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for condition_index, (condition, label, color) in enumerate(
        zip(CONDITIONS, display, colors)
    ):
        offset = (condition_index - 1) * width
        for axis, metric, title in zip(
            axes,
            ("token_f1", "official_multihop_rag_score"),
            ("F1 delta vs independent PRA", "Official-score delta vs independent PRA"),
        ):
            entries = [
                next(
                    item
                    for item in model["effects_vs_independent_pra"]
                    if item["condition"] == condition
                )["effects"][metric]
                for model in models
            ]
            values = [float(entry["mean_difference"]) for entry in entries]
            intervals = [entry["conservative_ci95"] for entry in entries]
            errors = [
                [value - float(interval[0]) for value, interval in zip(values, intervals)],
                [float(interval[1]) - value for value, interval in zip(values, intervals)],
            ]
            axis.errorbar(
                x + offset,
                values,
                yerr=errors,
                marker="o",
                linestyle="none",
                capsize=3,
                color=color,
                label=label,
            )
            axis.axhline(0.0, color="#444444", linewidth=0.8)
            axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
            axis.set_ylabel(title)
            axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "model_ladder_effects.pdf", bbox_inches="tight")
    figure.savefig(output / "model_ladder_effects.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    result = build_ladder(summaries)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_ladder(result, args.output)
    print(json.dumps({"output": str(args.output), "models": len(result["models"])}))


if __name__ == "__main__":
    main()
