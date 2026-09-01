"""Summarize graph-compiled segmented attention at kernel and model levels."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _identity(row: dict[str, Any]) -> tuple[str, int, str]:
    return str(row["dataset"]), int(row["seed"]), str(row["example_id"])


def _model_ratios(
    payload: dict[str, Any], identities: set[tuple[str, int, str]] | None = None
) -> dict[str, Any]:
    rows = [
        row
        for row in payload["rows"]
        if identities is None or _identity(row) in identities
    ]
    baseline = {
        _identity(row): row for row in rows if row["condition"] == "E0_WARM"
    }
    result = {}
    for condition in ("E2_CONCAT_WARM", "E2_SEGMENTED_ALL_LAYERS"):
        selected = [row for row in rows if row["condition"] == condition]
        result[condition] = {
            "samples": len(selected),
            "warm_ratio_vs_e0": statistics.fmean(
                float(row["warm_request_ms"])
                / float(baseline[_identity(row)]["warm_request_ms"])
                for row in selected
            ),
            "sequence_agreement_vs_e0": statistics.fmean(
                float(row["sequence_agreement_vs_e0"]) for row in selected
            ),
            "mean_f1_delta_vs_e0": statistics.fmean(
                float(row["token_f1"])
                - float(baseline[_identity(row)]["token_f1"])
                for row in selected
            ),
        }
    return result


def summarize(
    kernel: dict[str, Any],
    compiled_model: dict[str, Any],
    eager_model: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for row in kernel["rows"]:
        concat = float(row["concatenated_mean_ms"])
        rows.append(
            {
                "occupied_context_tokens": int(row["local_tokens"]),
                "selected_native_tokens": int(row["memory_tokens"]),
                "concat_mean_ms": concat,
                "eager_segmented_mean_ms": float(row["segmented_mean_ms"]),
                "compiled_segmented_mean_ms": float(
                    row["compiled_segmented_mean_ms"]
                ),
                "eager_over_concat": float(row["segmented_mean_ms"]) / concat,
                "compiled_over_concat": float(row["compiled_segmented_mean_ms"])
                / concat,
                "concat_temporary_bytes_avoided": int(
                    row["kv_concat_temporary_bytes_avoided"]
                ),
                "compiled_max_absolute_error": float(
                    row["compiled_max_absolute_error"]
                ),
            }
        )
    return {
        "schema_version": "paper6.2-segmented-compilation-summary-v1",
        "evidence_tier": "METAL_KERNEL_AND_MODEL_BACKED_NATURAL_QA",
        "claim_boundary": (
            "MLX graph compilation, not a custom fused Metal kernel; kernel "
            "profiles and full-model request ratios are reported separately."
        ),
        "kernel_rows": rows,
        "model_id": compiled_model["model_id"],
        "model_revision": compiled_model["model_revision"],
        "model_hardware": compiled_model["runtime"]["hardware_model"],
        "matched_model_examples": len(
            {
                _identity(row)
                for row in compiled_model["rows"]
                if row["condition"] == "E0_WARM"
            }
        ),
        "compiled_model_conditions": _model_ratios(compiled_model),
        "eager_model_conditions_on_compiled_subset": _model_ratios(
            eager_model,
            {
                _identity(row)
                for row in compiled_model["rows"]
                if row["condition"] == "E0_WARM"
            },
        ),
    }


def write_table(path: Path, report: dict[str, Any]) -> None:
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Local tokens & concat ms & eager/concat & compiled/concat & avoided MiB & max error \\",
        r"\midrule",
    ]
    for row in report["kernel_rows"]:
        lines.append(
            f"{row['occupied_context_tokens']:,} & {row['concat_mean_ms']:.3f} & "
            f"{row['eager_over_concat']:.3f} & {row['compiled_over_concat']:.3f} & "
            f"{row['concat_temporary_bytes_avoided'] / 2**20:.1f} & "
            f"{row['compiled_max_absolute_error']:.2e} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path: Path, report: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    rows = report["kernel_rows"]
    x = [row["occupied_context_tokens"] for row in rows]
    figure, axis = plt.subplots(figsize=(5.6, 3.4))
    axis.plot(x, [1.0] * len(x), marker="o", label="concatenated")
    axis.plot(x, [row["eager_over_concat"] for row in rows], marker="s", label="eager segmented")
    axis.plot(x, [row["compiled_over_concat"] for row in rows], marker="^", label="compiled segmented")
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [f"{value // 1024}K" for value in x])
    axis.set_xlabel("occupied local context")
    axis.set_ylabel("attention latency / concat")
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--compiled-model", type=Path, required=True)
    parser.add_argument("--eager-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        json.loads(args.kernel.read_text(encoding="utf-8")),
        json.loads(args.compiled_model.read_text(encoding="utf-8")),
        json.loads(args.eager_model.read_text(encoding="utf-8")),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "segmented_compilation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_table(args.output_dir / "generated_segmented_compilation_table.tex", report)
    write_plot(args.output_dir / "segmented_compilation.png", report)


if __name__ == "__main__":
    main()
