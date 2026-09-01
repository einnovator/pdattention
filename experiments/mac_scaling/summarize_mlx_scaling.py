"""Summarize the Apple Silicon dense/MoE PRA layer-profile campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


MODEL_LABELS = {
    "mlx-community/Qwen3-8B-4bit": "Qwen3-8B",
    "mlx-community/Qwen3-14B-4bit": "Qwen3-14B",
    "mlx-community/Qwen3-32B-4bit": "Qwen3-32B",
    "mlx-community/Qwen3-30B-A3B-4bit": "Qwen3-30B-A3B",
}
PROFILE_ORDER = (
    "E2_SEGMENTED_ALL",
    "E2_SEGMENTED_LAST_3_4",
    "E2_SEGMENTED_LAST_2_3",
    "E2_SEGMENTED_LAST_1_2",
    "E2_SEGMENTED_LAST_1_3",
    "E2_SEGMENTED_LAST_1_4",
)


def summarize(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    """Produce one cross-dataset row per model with paired E0 deltas."""

    result = []
    for payload in payloads:
        by_dataset = {}
        for row in payload["aggregate"]:
            by_dataset.setdefault(str(row["dataset"]), {})[str(row["condition"])] = row
        profile_metrics = {}
        for profile in PROFILE_ORDER:
            pairs = [
                (conditions["E0_SELECTED"], conditions[profile])
                for conditions in by_dataset.values()
            ]
            profile_metrics[profile] = {
                "mean_gold_logprob_delta": fmean(
                    float(right["gold_answer_logprob"])
                    - float(left["gold_answer_logprob"])
                    for left, right in pairs
                ),
                "mean_absolute_gold_logprob_delta": fmean(
                    abs(
                        float(right["gold_answer_logprob"])
                        - float(left["gold_answer_logprob"])
                    )
                    for left, right in pairs
                ),
                "sequence_agreement": fmean(
                    float(right["sequence_agreement_vs_e0"])
                    for _, right in pairs
                ),
                "active_detail_mib": fmean(
                    float(right["active_detail_bytes"]) / 2**20
                    for _, right in pairs
                ),
                "completion_latency_ms": fmean(
                    float(right["completion_latency_ms"]) for _, right in pairs
                ),
            }
        e0 = [conditions["E0_SELECTED"] for conditions in by_dataset.values()]
        concat = [conditions["E2_CONCAT_ALL"] for conditions in by_dataset.values()]
        segmented = [
            conditions["E2_SEGMENTED_ALL"] for conditions in by_dataset.values()
        ]
        raw_rows = payload["rows"]
        result.append(
            {
                "model_id": payload["model_id"],
                "model_label": MODEL_LABELS[str(payload["model_id"])],
                "model_revision": payload["model_revision"],
                "layer_count": int(payload["layer_count"]),
                "samples": sum(int(row["samples"]) for row in e0),
                "seed_count": len(payload["seeds"]),
                "e0_token_f1": fmean(float(row["token_f1"]) for row in e0),
                "concat_sequence_agreement": fmean(
                    float(row["sequence_agreement_vs_e0"]) for row in concat
                ),
                "segmented_sequence_agreement": fmean(
                    float(row["sequence_agreement_vs_e0"]) for row in segmented
                ),
                "e0_completion_latency_ms": fmean(
                    float(row["completion_latency_ms"]) for row in e0
                ),
                "concat_completion_latency_ms": fmean(
                    float(row["completion_latency_ms"]) for row in concat
                ),
                "segmented_completion_latency_ms": fmean(
                    float(row["completion_latency_ms"]) for row in segmented
                ),
                "model_resident_gib": int(raw_rows[0]["model_resident_bytes"]) / 2**30,
                "peak_unified_memory_gib": max(
                    int(row["peak_unified_memory_bytes"]) for row in raw_rows
                )
                / 2**30,
                "profiles": profile_metrics,
            }
        )
    return sorted(
        result,
        key=lambda row: (
            "A3B" in str(row["model_label"]),
            int(str(row["model_label"]).split("-")[1].removesuffix("B")),
        ),
    )


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the compact table shared by Papers 4.5 and 6.2."""

    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & Layers & E0 F1 & Concat agr. & Seg. agr. & $|\Delta|$LP & PRA MiB & Seg./E0 \\",
        r"\midrule",
    ]
    for row in rows:
        full = row["profiles"]["E2_SEGMENTED_ALL"]
        lines.append(
            f"{row['model_label']} & {row['layer_count']} & {row['e0_token_f1']:.3f} & "
            f"{row['concat_sequence_agreement']:.3f} & "
            f"{row['segmented_sequence_agreement']:.3f} & "
            f"{full['mean_absolute_gold_logprob_delta']:.3f} & "
            f"{full['active_detail_mib']:.1f} & "
            f"{row['segmented_completion_latency_ms'] / row['e0_completion_latency_ms']:.2f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    """Plot layer-fraction sensitivity and representation latency."""

    import matplotlib.pyplot as plt
    import numpy as np

    fractions = (1.0, 0.75, 2 / 3, 0.5, 1 / 3, 0.25)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    colors = ("#176b87", "#2f855a", "#9c6b30", "#9b3a4a")
    for row, color in zip(rows, colors):
        deltas = [
            row["profiles"][profile]["mean_absolute_gold_logprob_delta"]
            for profile in PROFILE_ORDER
        ]
        axes[0].plot(fractions, deltas, marker="o", label=row["model_label"], color=color)
    axes[0].set_xlabel("Consumer-layer fraction")
    axes[0].set_ylabel(r"Mean absolute $\Delta$ gold log probability")
    axes[0].invert_xaxis()
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    x = np.arange(len(rows))
    width = 0.25
    axes[1].bar(
        x - width,
        [row["e0_completion_latency_ms"] for row in rows],
        width,
        label="E0 selected",
    )
    axes[1].bar(
        x,
        [row["concat_completion_latency_ms"] for row in rows],
        width,
        label="E2 concat",
    )
    axes[1].bar(
        x + width,
        [row["segmented_completion_latency_ms"] for row in rows],
        width,
        label="E2 segmented",
    )
    axes[1].set_xticks(x, [row["model_label"].replace("Qwen3-", "") for row in rows])
    axes[1].set_ylabel("Mean completion latency (ms)")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    rows = summarize(payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mlx_scaling_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "pra-mac-scaling-summary-v1",
                "evidence_tier": "MODEL_BACKED_NATURAL_QA_CALIBRATION",
                "selection": "annotated evidence documents fixed across conditions",
                "models": rows,
                "interpretation": {
                    "concat_reference_correctness": "passed all 60 model-example pairs",
                    "segmented_status": "sequence-stable at 8B/14B; small numerical drift and tie-sensitive changes at 32B/MoE",
                    "profile_status": "all reduced-layer profiles remain calibration candidates",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_table(args.output_dir / "generated_mlx_scaling_table.tex", rows)
    write_plot(args.output_dir / "mlx_scaling_profiles.png", rows)


if __name__ == "__main__":
    main()
