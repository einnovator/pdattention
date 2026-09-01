"""Summarize matched MLX model-size and consumer-depth scaling."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable


MODEL_LABELS = {
    "mlx-community/Qwen3-4B-4bit": "4B",
    "mlx-community/Qwen3-8B-4bit": "8B",
    "mlx-community/Qwen3-14B-4bit": "14B",
    "mlx-community/Qwen3-32B-4bit": "32B",
}


def _identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["dataset"]), int(row["seed"]), str(row["example_id"])


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def _bootstrap_interval(values: list[float], seed: int = 1729) -> tuple[float, float]:
    """Return a deterministic paired bootstrap interval for the mean."""

    if not values:
        return float("nan"), float("nan")
    generator = random.Random(seed)
    draws = sorted(
        _mean(values[generator.randrange(len(values))] for _ in values)
        for _ in range(5000)
    )
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def summarize_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute paired quality and matched-state cost ratios for one model."""

    rows = list(payload["rows"])
    baselines = {
        _identity(row): row for row in rows if row["condition"] == "E0_WARM"
    }
    conditions = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        paired = [(row, baselines[_identity(row)]) for row in selected]
        f1_deltas = [
            float(row["token_f1"]) - float(base["token_f1"])
            for row, base in paired
        ]
        logprob_deltas = [
            float(row["gold_answer_logprob"])
            - float(base["gold_answer_logprob"])
            for row, base in paired
        ]
        ci_low, ci_high = _bootstrap_interval(f1_deltas)
        warm_ratios = [
            float(row["warm_request_ms"]) / float(base["warm_request_ms"])
            for row, base in paired
        ]
        cold_ratios = [
            float(row["cold_usable_context_ms"])
            / float(base["cold_usable_context_ms"])
            for row, base in paired
        ]
        warm_ratio = _mean(warm_ratios)
        cold_ratio = _mean(cold_ratios)
        summary = {
            "condition": condition,
            "samples": len(selected),
            "seed_count": len({int(row["seed"]) for row in selected}),
            "consumer_layer_fraction": _mean(
                float(row["consumer_layer_fraction"]) for row in selected
            ),
            "token_f1": _mean(float(row["token_f1"]) for row in selected),
            "mean_f1_delta": _mean(f1_deltas),
            "f1_delta_ci95": [ci_low, ci_high],
            "mean_gold_logprob_delta": _mean(logprob_deltas),
            "mean_absolute_gold_logprob_delta": _mean(map(abs, logprob_deltas)),
            "sequence_agreement": _mean(
                float(row["sequence_agreement_vs_e0"]) for row in selected
            ),
            "warm_cost_ratio_vs_e0": warm_ratio,
            "warm_cost_ratio_ci95": list(_bootstrap_interval(warm_ratios)),
            "cold_cost_ratio_vs_e0": cold_ratio,
            "cold_cost_ratio_ci95": list(_bootstrap_interval(cold_ratios)),
            "active_detail_mib": _mean(
                float(row["active_detail_bytes"]) / 2**20 for row in selected
            ),
            "strict_transport_gate": False,
            "balanced_smoke_gate": False,
        }
        summary["strict_transport_gate"] = bool(
            summary["sequence_agreement"] >= 0.95
            and summary["mean_absolute_gold_logprob_delta"] <= 0.10
            and summary["mean_f1_delta"] >= -0.01
        )
        summary["balanced_smoke_gate"] = bool(
            ci_low >= -0.03
            and summary["mean_absolute_gold_logprob_delta"] <= 0.50
        )
        conditions.append(summary)

    segmented = [
        row for row in conditions if str(row["condition"]).startswith("E2_SEGMENTED")
    ]
    balanced = [row for row in segmented if row["balanced_smoke_gate"]]
    balanced_reduced = [
        row
        for row in balanced
        if float(row["consumer_layer_fraction"]) < 1.0
    ]
    strict = [row for row in segmented if row["strict_transport_gate"]]
    return {
        "model_id": payload["model_id"],
        "model_label": MODEL_LABELS.get(payload["model_id"], payload["model_id"]),
        "model_revision": payload["model_revision"],
        "layer_count": payload["layer_count"],
        "samples": len(baselines),
        "seed_count": len({identity[1] for identity in baselines}),
        "conditions": conditions,
        "minimum_balanced_smoke_fraction": (
            min(float(row["consumer_layer_fraction"]) for row in balanced)
            if balanced
            else None
        ),
        "minimum_balanced_reduced_fraction": (
            min(float(row["consumer_layer_fraction"]) for row in balanced_reduced)
            if balanced_reduced
            else None
        ),
        "minimum_strict_transport_fraction": (
            min(float(row["consumer_layer_fraction"]) for row in strict)
            if strict
            else None
        ),
    }


def _condition(model: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in model["conditions"] if row["condition"] == name)


def write_table(path: Path, models: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & $n$ & concat warm & concat cold & seg. warm & seg. agr. & balanced & PRA MiB \\",
        r"\midrule",
    ]
    for model in models:
        concat = _condition(model, "E2_CONCAT_WARM")
        segmented = _condition(model, "E2_SEGMENTED_ALL_LAYERS")
        balanced = model["minimum_balanced_reduced_fraction"]
        balanced_text = "none" if balanced is None else f"{float(balanced):.3f}"
        lines.append(
            f"Qwen3-{model['model_label']} & {model['samples']} & "
            f"{concat['warm_cost_ratio_vs_e0']:.3f} & "
            f"{concat['cold_cost_ratio_vs_e0']:.3f} & "
            f"{segmented['warm_cost_ratio_vs_e0']:.3f} & "
            f"{segmented['sequence_agreement']:.3f} & {balanced_text} & "
            f"{segmented['active_detail_mib']:.1f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_consumer_table(path: Path, models: list[dict[str, Any]]) -> None:
    """Write the quality/cost surface for each segmented consumer suffix."""

    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & profile & fraction & $\Delta$F1 & $|\Delta|$LP & agreement & warm ratio \\",
        r"\midrule",
    ]
    for model in models:
        profiles = sorted(
            (
                row
                for row in model["conditions"]
                if str(row["condition"]).startswith("E2_SEGMENTED")
            ),
            key=lambda row: -float(row["consumer_layer_fraction"]),
        )
        for row in profiles:
            label = str(row["condition"]).removeprefix("E2_SEGMENTED_").lower()
            label = {
                "all_layers": "all",
                "last_7_8": "last 7/8",
                "last_3_4": "last 3/4",
                "last_2_3": "last 2/3",
                "last_1_2": "last 1/2",
            }.get(label, label.replace("_", r"\_"))
            lines.append(
                f"Qwen3-{model['model_label']} & {label} & "
                f"{row['consumer_layer_fraction']:.3f} & "
                f"{row['mean_f1_delta']:+.3f} & "
                f"{row['mean_absolute_gold_logprob_delta']:.3f} & "
                f"{row['sequence_agreement']:.3f} & "
                f"{row['warm_cost_ratio_vs_e0']:.3f} \\\\"
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, models: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(model["model_label"]) for model in models]
    concat_warm = [
        _condition(model, "E2_CONCAT_WARM")["warm_cost_ratio_vs_e0"]
        for model in models
    ]
    concat_cold = [
        _condition(model, "E2_CONCAT_WARM")["cold_cost_ratio_vs_e0"]
        for model in models
    ]
    segmented = [
        _condition(model, "E2_SEGMENTED_ALL_LAYERS")["warm_cost_ratio_vs_e0"]
        for model in models
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    concat_warm_ci = [
        _condition(model, "E2_CONCAT_WARM")["warm_cost_ratio_ci95"]
        for model in models
    ]
    concat_cold_ci = [
        _condition(model, "E2_CONCAT_WARM")["cold_cost_ratio_ci95"]
        for model in models
    ]
    segmented_ci = [
        _condition(model, "E2_SEGMENTED_ALL_LAYERS")["warm_cost_ratio_ci95"]
        for model in models
    ]

    def errors(values: list[float], intervals: list[list[float]]) -> list[list[float]]:
        return [
            [value - interval[0] for value, interval in zip(values, intervals)],
            [interval[1] - value for value, interval in zip(values, intervals)],
        ]

    axes[0].errorbar(
        labels,
        concat_warm,
        yerr=errors(concat_warm, concat_warm_ci),
        marker="o",
        capsize=3,
        label="concat E2 warm",
    )
    axes[0].errorbar(
        labels,
        concat_cold,
        yerr=errors(concat_cold, concat_cold_ci),
        marker="s",
        capsize=3,
        label="concat E2 cold",
    )
    axes[0].errorbar(
        labels,
        segmented,
        yerr=errors(segmented, segmented_ci),
        marker="^",
        capsize=3,
        label="segmented E2 warm",
    )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("matched E2 / E0 cost")
    axes[0].set_xlabel("Qwen3 model size")
    axes[0].legend(frameon=False, fontsize=8)

    for model in models:
        profiles = sorted(
            (
                row
                for row in model["conditions"]
                if str(row["condition"]).startswith("E2_SEGMENTED")
            ),
            key=lambda row: row["consumer_layer_fraction"],
        )
        axes[1].plot(
            [row["consumer_layer_fraction"] for row in profiles],
            [row["mean_f1_delta"] for row in profiles],
            marker="o",
            label=str(model["model_label"]),
        )
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].axhline(-0.03, color="black", linestyle=":", linewidth=1)
    axes[1].set_xlabel("consumer-layer fraction")
    axes[1].set_ylabel("paired token-F1 delta vs E0")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    models = [
        summarize_model(json.loads(path.read_text(encoding="utf-8")))
        for path in args.inputs
    ]
    models.sort(key=lambda row: int(str(row["model_label"]).removesuffix("B")))
    report = {
        "schema_version": "paper6.2-mlx-model-consumer-scaling-summary-v2",
        "evidence_tier": "MODEL_BACKED_NATURAL_QA_SCALING",
        "timing_contract": "matched cold/cold and warm/warm; legacy mixed-state ratio excluded",
        "balanced_smoke_gate": (
            "paired F1 95% bootstrap lower bound >= -0.03 and mean absolute "
            "gold-logprobability delta <= 0.50 nats"
        ),
        "strict_transport_gate": (
            "sequence agreement >= 0.95, mean absolute gold-logprobability "
            "delta <= 0.10 nats, and mean F1 delta >= -0.01"
        ),
        "models": models,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_consumer_scaling_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_table(args.output_dir / "generated_model_consumer_scaling_table.tex", models)
    write_consumer_table(
        args.output_dir / "generated_model_consumer_quality_table.tex", models
    )
    write_plot(args.output_dir / "model_consumer_scaling.png", models)


if __name__ == "__main__":
    main()
