"""Aggregate Paper 3.2 B/C and cross-document composition precision runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence


PRECISION_ORDER = {"FP32": 0, "FP16": 1, "INT8": 2, "INT4": 3}
COMPOSITION_CONDITIONS = (
    "D_GIST_SA_APPEND",
    "E_GIST_SA_BOUNDARY_8",
    "F_GIST_SA_BOUNDARY_32",
)


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _load(path: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    with gzip.open(path.parent / "condition_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    with gzip.open(path.parent / "bc_layer_diagnostics.jsonl.gz", "rt", encoding="utf-8") as stream:
        diagnostics = [json.loads(line) for line in stream if line.strip()]
    return manifest, rows, diagnostics


def aggregate(paths: Sequence[Path]) -> dict[str, object]:
    """Group exact paired diagnostics by explicit precision mode."""

    if not paths:
        raise ValueError("precision aggregation requires at least one manifest")
    grouped: dict[str, list[tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]]] = {}
    for path in paths:
        run = _load(path)
        precision = str((run[0].get("precision") or {}).get("precision_mode", "UNKNOWN"))
        if precision == "UNKNOWN":
            raise ValueError(f"manifest lacks explicit precision metadata: {path}")
        grouped.setdefault(precision, []).append(run)

    precision_rows = []
    composition_rows = []
    for precision in sorted(grouped, key=lambda value: PRECISION_ORDER.get(value, 99)):
        runs = grouped[precision]
        all_rows = [row for _, rows, _ in runs for row in rows]
        by_example: dict[tuple[int, str], dict[str, Mapping[str, object]]] = {}
        for manifest, rows, _ in runs:
            seed = int(manifest["seed"])
            for row in rows:
                by_example.setdefault((seed, str(row["example_id"])), {})[
                    str(row["condition"])
                ] = row
        pairs = [
            pair for pair in by_example.values()
            if "B_NO_CROSS_DOC_RAG" in pair
            and "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS" in pair
        ]
        diagnostics = [row for _, _, values in runs for row in values]
        layer_count = max((len(row.get("layers", ())) for row in diagnostics), default=0)
        layerwise = []
        for layer_index in range(layer_count):
            values = [
                row["layers"][layer_index]
                for row in diagnostics
                if len(row.get("layers", ())) > layer_index
            ]
            layerwise.append(
                {
                    "layer": layer_index,
                    "key_rmse": _mean([float(row["key_rmse"]) for row in values]),
                    "value_rmse": _mean([float(row["value_rmse"]) for row in values]),
                    "key_max_abs_delta": max(
                        (float(row["key_max_abs_delta"]) for row in values), default=None
                    ),
                    "value_max_abs_delta": max(
                        (float(row["value_max_abs_delta"]) for row in values), default=None
                    ),
                }
            )
        precision_rows.append(
            {
                "precision_mode": precision,
                "runs": len(runs),
                "pairs": len(pairs),
                "output_match_rate": _mean([
                    float(
                        pair["B_NO_CROSS_DOC_RAG"]["prediction"]
                        == pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["prediction"]
                    )
                    for pair in pairs
                ]),
                "first_step_logit_hash_match_rate": _mean([
                    float(
                        pair["B_NO_CROSS_DOC_RAG"]["first_step_logits_sha256"]
                        == pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["first_step_logits_sha256"]
                    )
                    for pair in pairs
                ]),
                "mean_first_step_js": _mean([
                    float(pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["first_step_js_divergence"])
                    for pair in pairs
                ]),
                "mean_first_step_logit_max_abs_delta": _mean([
                    float(pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["first_step_logit_max_abs_delta"])
                    for pair in pairs
                ]),
                "mean_gold_nll_abs_delta": _mean([
                    abs(
                        float(pair["B_NO_CROSS_DOC_RAG"]["gold_answer_mean_nll"])
                        - float(pair["C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"]["gold_answer_mean_nll"])
                    )
                    for pair in pairs
                ]),
                "max_layer_key_rmse": max(
                    (float(row["key_rmse"]) for row in layerwise), default=None
                ),
                "max_layer_value_rmse": max(
                    (float(row["value_rmse"]) for row in layerwise), default=None
                ),
                "layerwise": layerwise,
                "metadata": runs[0][0]["precision"],
            }
        )
        for condition in COMPOSITION_CONDITIONS:
            selected = [row for row in all_rows if row["condition"] == condition]
            if not selected:
                continue
            composition_rows.append(
                {
                    "precision_mode": precision,
                    "condition": condition,
                    "examples": len(selected),
                    "token_f1": _mean([float(row["token_f1"]) for row in selected]),
                    "official_score": _mean([
                        float(row["official_multihop_rag_score"]) for row in selected
                    ]),
                    "gold_answer_nll": _mean([
                        float(row["gold_answer_mean_nll"]) for row in selected
                    ]),
                    "request_composition_ms": _mean([
                        float(row["request_composition_ms"]) for row in selected
                    ]),
                    "request_local_native_tokens": _mean([
                        float(row["request_local_native_tokens"]) for row in selected
                    ]),
                    "cross_document_interaction_edges": _mean([
                        float(row["cross_document_interaction_edges"]) for row in selected
                    ]),
                }
            )
    return {
        "schema_version": "paper3.2-crossdoc-precision-aggregate-v1",
        "experiment": "crossdoc_kv_precision_qualification",
        "precision_conditions": precision_rows,
        "composition_conditions": composition_rows,
        "source_manifests": [str(path) for path in paths],
    }


def _write_csv(result: Mapping[str, object], path: Path) -> None:
    fields = (
        "precision_mode", "runs", "pairs", "output_match_rate",
        "first_step_logit_hash_match_rate", "mean_first_step_js",
        "mean_first_step_logit_max_abs_delta", "mean_gold_nll_abs_delta",
        "max_layer_key_rmse", "max_layer_value_rmse",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["precision_conditions"]:  # type: ignore[index]
            writer.writerow({field: row.get(field) for field in fields})


def _write_latex(result: Mapping[str, object], output: Path) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Precision & Pairs & Output match & JS & $|\\Delta\\mathrm{NLL}|$ & Max K RMSE & Max V RMSE \\\\",
        "\\midrule",
    ]
    for row in result["precision_conditions"]:  # type: ignore[index]
        lines.append(
            f"{row['precision_mode']} & {row['pairs']} & "
            f"{float(row['output_match_rate']):.3f} & "
            f"{float(row['mean_first_step_js']):.6f} & "
            f"{float(row['mean_gold_nll_abs_delta']):.4f} & "
            f"{float(row['max_layer_key_rmse']):.4f} & "
            f"{float(row['max_layer_value_rmse']):.4f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    (output / "generated_precision_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _plot_layers(result: Mapping[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for precision in result["precision_conditions"]:  # type: ignore[index]
        rows = precision["layerwise"]
        layers = [int(row["layer"]) for row in rows]
        axes[0].plot(
            layers, [max(float(row["key_rmse"]), 1e-9) for row in rows],
            label=precision["precision_mode"],
        )
        axes[1].plot(
            layers, [max(float(row["value_rmse"]), 1e-9) for row in rows],
            label=precision["precision_mode"],
        )
    for axis, label in zip(axes, ("Key RMSE", "Value RMSE")):
        axis.set_yscale("log")
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(result, args.output / "precision_summary.csv")
    _write_latex(result, args.output)
    _plot_layers(result, args.output / "precision_layerwise_rmse.pdf")
    _plot_layers(result, args.output / "precision_layerwise_rmse.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
