"""Aggregate replicated Paper 3.2 pre-RoPE causal-decomposition runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _row_mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    """Average one optional numeric artifact field without inventing zeros."""

    return _mean(
        [float(row[field]) for row in rows if row.get(field) is not None]
    )


def _condition_order(condition: str) -> tuple[int, int, str]:
    fixed = {
        "A_FULL_CAUSAL_RAG": 0,
        "B_NO_CROSS_DOC_RAG": 1,
        "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS": 2,
        "M_PREVIOUS_DOC_ONLY": 3,
        "M_TOP_RANKED_TO_ALL": 4,
    }
    if condition in fixed:
        return fixed[condition], 0, condition
    if condition.startswith("M4_BOUNDARY_ONLY_"):
        return 5, int(condition.rsplit("_", 1)[1]), condition
    return 6, 0, condition


def _bootstrap(values: Sequence[float], *, seed: int = 3205) -> list[float] | None:
    if not values:
        return None
    generator = random.Random(seed)
    samples = sorted(
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(10_000)
    )
    return [samples[249], samples[9749]]


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _load(
    manifest_path: Path,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_path = manifest_path.parent / "condition_results.jsonl.gz"
    with gzip.open(rows_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    diagnostics_path = manifest_path.parent / "bc_layer_diagnostics.jsonl.gz"
    diagnostics = []
    if diagnostics_path.exists():
        with gzip.open(diagnostics_path, "rt", encoding="utf-8") as stream:
            diagnostics = [json.loads(line) for line in stream if line.strip()]
    return manifest, rows, diagnostics


def aggregate(manifest_paths: Sequence[Path]) -> dict[str, object]:
    if len(manifest_paths) < 2:
        raise ValueError("causal aggregation requires at least two seed manifests")
    runs = [_load(path) for path in manifest_paths]
    identities = {
        (
            manifest["dataset"],
            manifest["model"],
            manifest["model_revision"],
            manifest["reranker"],
            manifest["reranker_revision"],
            manifest["token_budget"],
            manifest["max_resources"],
        )
        for manifest, _, _ in runs
    }
    if len(identities) != 1:
        raise ValueError("seed manifests do not share one frozen protocol")

    condition_names = sorted(
        {str(row["condition"]) for _, rows, _ in runs for row in rows},
        key=_condition_order,
    )
    conditions = []
    for condition in condition_names:
        seed_rows = []
        for manifest, rows, _ in runs:
            selected = [row for row in rows if row["condition"] == condition]
            seed_rows.append(
                {
                    "seed": int(manifest["seed"]),
                    "examples": len(selected),
                    "token_f1": _mean([float(row["token_f1"]) for row in selected]),
                    "exact_match": _mean(
                        [float(row["exact_match"]) for row in selected]
                    ),
                    "gold_answer_mean_nll": _mean(
                        [float(row["gold_answer_mean_nll"]) for row in selected]
                    ),
                    "first_step_js": _mean(
                        [
                            float(row["first_step_js_divergence"])
                            for row in selected
                            if row["first_step_js_divergence"] is not None
                        ]
                    ),
                    "cross_document_attention_edges_allowed": _mean(
                        [
                            float(row["cross_document_attention_edges_allowed"])
                            for row in selected
                        ]
                    ),
                    "request_rope_transform_ms": _mean(
                        [float(row["request_rope_transform_ms"]) for row in selected]
                    ),
                    "official_multihop_rag_score": _row_mean(
                        selected, "official_multihop_rag_score"
                    ),
                    "supporting_document_coverage": _row_mean(
                        selected, "supporting_document_coverage"
                    ),
                    "physical_native_tokens": _row_mean(
                        selected, "physical_native_tokens"
                    ),
                    "native_bytes": _row_mean(selected, "native_bytes"),
                    "encode_ms": _row_mean(selected, "encode_ms"),
                    "ttft_ms": _row_mean(selected, "ttft_ms"),
                    "ttft_with_materialization_ms": _row_mean(
                        selected, "ttft_with_materialization_ms"
                    ),
                    "total_latency_ms": _row_mean(selected, "total_latency_ms"),
                    "total_with_materialization_ms": _row_mean(
                        selected, "total_with_materialization_ms"
                    ),
                }
            )
        f1 = [float(row["token_f1"]) for row in seed_rows]
        nll = [float(row["gold_answer_mean_nll"]) for row in seed_rows]
        conditions.append(
            {
                "condition": condition,
                "seeds": seed_rows,
                "seed_mean_token_f1": statistics.fmean(f1),
                "seed_bootstrap_token_f1_95_ci": _bootstrap(f1),
                "seed_mean_exact_match": statistics.fmean(
                    float(row["exact_match"]) for row in seed_rows
                ),
                "seed_mean_gold_answer_nll": statistics.fmean(nll),
                "seed_bootstrap_gold_answer_nll_95_ci": _bootstrap(nll),
                "seed_mean_first_step_js": _mean(
                    [
                        float(row["first_step_js"])
                        for row in seed_rows
                        if row["first_step_js"] is not None
                    ]
                ),
                "seed_mean_cross_document_edges": statistics.fmean(
                    float(row["cross_document_attention_edges_allowed"])
                    for row in seed_rows
                ),
                "seed_mean_request_rope_transform_ms": statistics.fmean(
                    float(row["request_rope_transform_ms"]) for row in seed_rows
                ),
                "seed_mean_official_multihop_rag_score": _row_mean(
                    seed_rows, "official_multihop_rag_score"
                ),
                "seed_mean_supporting_document_coverage": _row_mean(
                    seed_rows, "supporting_document_coverage"
                ),
                "seed_mean_physical_native_tokens": _row_mean(
                    seed_rows, "physical_native_tokens"
                ),
                "seed_mean_native_bytes": _row_mean(seed_rows, "native_bytes"),
                "seed_mean_encode_ms": _row_mean(seed_rows, "encode_ms"),
                "seed_mean_ttft_ms": _row_mean(seed_rows, "ttft_ms"),
                "seed_mean_ttft_with_materialization_ms": _row_mean(
                    seed_rows, "ttft_with_materialization_ms"
                ),
                "seed_mean_total_latency_ms": _row_mean(
                    seed_rows, "total_latency_ms"
                ),
                "seed_mean_total_with_materialization_ms": _row_mean(
                    seed_rows, "total_with_materialization_ms"
                ),
            }
        )

    seed_effects = []
    bc_output_matches = 0
    bc_pairs = 0
    bc_hash_matches = 0
    bc_js_values = []
    bc_nll_deltas = []
    layer_key_rmse = []
    layer_value_rmse = []
    all_layer_diagnostics: list[Mapping[str, object]] = []
    for manifest, rows, diagnostics in runs:
        all_layer_diagnostics.extend(diagnostics)
        by_example: dict[str, dict[str, Mapping[str, object]]] = {}
        for row in rows:
            by_example.setdefault(str(row["example_id"]), {})[
                str(row["condition"])
            ] = row
        ab_f1 = []
        ab_nll = []
        for pair in by_example.values():
            a = pair.get("A_FULL_CAUSAL_RAG")
            b = pair.get("B_NO_CROSS_DOC_RAG")
            c = pair.get("C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS")
            if a and b:
                ab_f1.append(float(a["token_f1"]) - float(b["token_f1"]))
                ab_nll.append(
                    float(a["gold_answer_mean_nll"])
                    - float(b["gold_answer_mean_nll"])
                )
            if b and c:
                bc_pairs += 1
                bc_output_matches += int(b["prediction"] == c["prediction"])
                bc_hash_matches += int(
                    b["first_step_logits_sha256"] == c["first_step_logits_sha256"]
                )
                bc_js_values.append(float(c["first_step_js_divergence"]))
                bc_nll_deltas.append(
                    abs(
                        float(b["gold_answer_mean_nll"])
                        - float(c["gold_answer_mean_nll"])
                    )
                )
        seed_effects.append(
            {
                "seed": int(manifest["seed"]),
                "a_minus_b_token_f1": _mean(ab_f1),
                "a_minus_b_gold_nll": _mean(ab_nll),
            }
        )
        bc_summary = manifest["summary"]["b_minus_c"]
        layer_key_rmse.append(float(bc_summary["max_layer_key_rmse"]))
        layer_value_rmse.append(float(bc_summary["max_layer_value_rmse"]))

    layer_count = max(
        (
            len(diagnostic.get("layers", []))
            for diagnostic in all_layer_diagnostics
        ),
        default=0,
    )
    layerwise = []
    for layer_index in range(layer_count):
        layer_rows = [
            diagnostic["layers"][layer_index]
            for diagnostic in all_layer_diagnostics
            if len(diagnostic.get("layers", [])) > layer_index
        ]
        key_values = [float(row["key_rmse"]) for row in layer_rows]
        value_values = [float(row["value_rmse"]) for row in layer_rows]
        layerwise.append(
            {
                "layer": layer_index,
                "pairs": len(layer_rows),
                "mean_key_rmse": _mean(key_values),
                "p95_key_rmse": _percentile(key_values, 0.95),
                "max_key_rmse": max(key_values, default=None),
                "mean_value_rmse": _mean(value_values),
                "p95_value_rmse": _percentile(value_values, 0.95),
                "max_value_rmse": max(value_values, default=None),
            }
        )

    ab_f1_seed = [float(row["a_minus_b_token_f1"]) for row in seed_effects]
    ab_nll_seed = [float(row["a_minus_b_gold_nll"]) for row in seed_effects]
    first_manifest = runs[0][0]
    return {
        "schema_version": "paper3.2-prerope-causal-aggregate-v1",
        "experiment": "prerope_causal_decomposition_five_seed",
        "dataset": first_manifest["dataset"],
        "model": first_manifest["model"],
        "model_revision": first_manifest["model_revision"],
        "reranker": first_manifest["reranker"],
        "reranker_revision": first_manifest["reranker_revision"],
        "seeds": [int(manifest["seed"]) for manifest, _, _ in runs],
        "replication_unit": "seed_cohort",
        "conditions": conditions,
        "a_minus_b": {
            "seed_effects": seed_effects,
            "seed_mean_token_f1_delta": statistics.fmean(ab_f1_seed),
            "seed_bootstrap_token_f1_delta_95_ci": _bootstrap(ab_f1_seed),
            "seed_mean_gold_nll_delta": statistics.fmean(ab_nll_seed),
            "seed_bootstrap_gold_nll_delta_95_ci": _bootstrap(ab_nll_seed),
        },
        "b_minus_c": {
            "pairs": bc_pairs,
            "output_matches": bc_output_matches,
            "output_match_rate": bc_output_matches / bc_pairs,
            "first_step_logit_hash_matches": bc_hash_matches,
            "first_step_logit_hash_match_rate": bc_hash_matches / bc_pairs,
            "mean_first_step_js_divergence": statistics.fmean(bc_js_values),
            "mean_gold_nll_abs_delta": statistics.fmean(bc_nll_deltas),
            "max_layer_key_rmse_across_runs": max(layer_key_rmse),
            "max_layer_value_rmse_across_runs": max(layer_value_rmse),
            "layerwise": layerwise,
        },
        "source_manifests": [str(path) for path in manifest_paths],
    }


def _write_table(result: Mapping[str, object], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "condition",
                "seed_mean_token_f1",
                "f1_ci_low",
                "f1_ci_high",
                "seed_mean_gold_answer_nll",
                "nll_ci_low",
                "nll_ci_high",
                "seed_mean_first_step_js",
                "seed_mean_cross_document_edges",
                "seed_mean_request_rope_transform_ms",
                "seed_mean_official_multihop_rag_score",
                "seed_mean_supporting_document_coverage",
                "seed_mean_physical_native_tokens",
                "seed_mean_native_bytes",
                "seed_mean_encode_ms",
                "seed_mean_ttft_ms",
                "seed_mean_ttft_with_materialization_ms",
                "seed_mean_total_latency_ms",
                "seed_mean_total_with_materialization_ms",
            ),
        )
        writer.writeheader()
        for row in result["conditions"]:  # type: ignore[index]
            f1_ci = row["seed_bootstrap_token_f1_95_ci"]
            nll_ci = row["seed_bootstrap_gold_answer_nll_95_ci"]
            writer.writerow(
                {
                    "condition": row["condition"],
                    "seed_mean_token_f1": row["seed_mean_token_f1"],
                    "f1_ci_low": f1_ci[0],
                    "f1_ci_high": f1_ci[1],
                    "seed_mean_gold_answer_nll": row["seed_mean_gold_answer_nll"],
                    "nll_ci_low": nll_ci[0],
                    "nll_ci_high": nll_ci[1],
                    "seed_mean_first_step_js": row["seed_mean_first_step_js"],
                    "seed_mean_cross_document_edges": row[
                        "seed_mean_cross_document_edges"
                    ],
                    "seed_mean_request_rope_transform_ms": row[
                        "seed_mean_request_rope_transform_ms"
                    ],
                    "seed_mean_official_multihop_rag_score": row[
                        "seed_mean_official_multihop_rag_score"
                    ],
                    "seed_mean_supporting_document_coverage": row[
                        "seed_mean_supporting_document_coverage"
                    ],
                    "seed_mean_physical_native_tokens": row[
                        "seed_mean_physical_native_tokens"
                    ],
                    "seed_mean_native_bytes": row["seed_mean_native_bytes"],
                    "seed_mean_encode_ms": row["seed_mean_encode_ms"],
                    "seed_mean_ttft_ms": row["seed_mean_ttft_ms"],
                    "seed_mean_ttft_with_materialization_ms": row[
                        "seed_mean_ttft_with_materialization_ms"
                    ],
                    "seed_mean_total_latency_ms": row[
                        "seed_mean_total_latency_ms"
                    ],
                    "seed_mean_total_with_materialization_ms": row[
                        "seed_mean_total_with_materialization_ms"
                    ],
                }
            )


def _condition_label(condition: object) -> str:
    return (
        str(condition)
        .replace("A_FULL_CAUSAL_RAG", "A full")
        .replace("B_NO_CROSS_DOC_RAG", "B isolated")
        .replace("C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS", "C pre-RoPE")
        .replace("M4_BOUNDARY_ONLY_", "boundary ")
        .replace("M_PREVIOUS_DOC_ONLY", "previous")
        .replace("M_TOP_RANKED_TO_ALL", "top-ranked")
    )


def _plot_quality(result: Mapping[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(result["conditions"])  # type: ignore[arg-type]
    names = [_condition_label(row["condition"]) for row in rows]
    values = [float(row["seed_mean_token_f1"]) for row in rows]
    lower = [
        value - float(row["seed_bootstrap_token_f1_95_ci"][0])
        for value, row in zip(values, rows)
    ]
    upper = [
        float(row["seed_bootstrap_token_f1_95_ci"][1]) - value
        for value, row in zip(values, rows)
    ]
    nll = [float(row["seed_mean_gold_answer_nll"]) for row in rows]
    nll_lower = [
        value - float(row["seed_bootstrap_gold_answer_nll_95_ci"][0])
        for value, row in zip(nll, rows)
    ]
    nll_upper = [
        float(row["seed_bootstrap_gold_answer_nll_95_ci"][1]) - value
        for value, row in zip(nll, rows)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    colors = ["#276FBF", "#C44536", "#2A9D8F"] + ["#6C757D"] * max(0, len(rows) - 3)
    axes[0].bar(range(len(rows)), values, color=colors[: len(rows)], width=0.72)
    axes[0].errorbar(
        range(len(rows)), values, yerr=(lower, upper), fmt="none", color="#202020", capsize=3
    )
    axes[0].set_ylabel("Token F1")
    axes[0].set_xticks(range(len(rows)), names, rotation=35, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(range(len(rows)), nll, color=colors[: len(rows)], width=0.72)
    axes[1].errorbar(
        range(len(rows)), nll, yerr=(nll_lower, nll_upper), fmt="none", color="#202020", capsize=3
    )
    axes[1].set_ylabel("Gold-answer NLL (lower is better)")
    axes[1].set_xticks(range(len(rows)), names, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_interaction_budget(result: Mapping[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row for row in result["conditions"]  # type: ignore[index]
        if row["condition"] != "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS"
    ]
    rows.sort(key=lambda row: float(row["seed_mean_cross_document_edges"]))
    edges = [float(row["seed_mean_cross_document_edges"]) for row in rows]
    values = [float(row["seed_mean_token_f1"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(edges, values, marker="o", color="#276FBF")
    for edge, value, row in zip(edges, values, rows):
        axis.annotate(_condition_label(row["condition"]), (edge, value), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set_xlabel("Allowed cross-document attention edges (seed mean)")
    axis.set_ylabel("Token F1")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _plot_bc_fidelity(result: Mapping[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    summary = result["b_minus_c"]  # type: ignore[index]
    rows = list(summary["layerwise"])
    if not rows:
        return
    layers = [int(row["layer"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for axis, prefix, title in (
        (axes[0], "key", "Key"),
        (axes[1], "value", "Value"),
    ):
        mean = [max(float(row[f"mean_{prefix}_rmse"]), 1e-8) for row in rows]
        p95 = [max(float(row[f"p95_{prefix}_rmse"]), 1e-8) for row in rows]
        axis.plot(layers, mean, label="mean", color="#276FBF")
        axis.plot(layers, p95, label="p95", color="#C44536", linestyle="--")
        axis.set_yscale("log")
        axis.set_xlabel("Decoder layer")
        axis.set_ylabel(f"{title} RMSE")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle(
        "B/C output match "
        f"{float(summary['output_match_rate']):.1%}; "
        f"first-step JS {float(summary['mean_first_step_js_divergence']):.2e}"
    )
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
    _write_table(result, args.output / "condition_summary.csv")
    _plot_quality(result, args.output / "causal_decomposition_quality.pdf")
    _plot_quality(result, args.output / "causal_decomposition_quality.png")
    _plot_interaction_budget(result, args.output / "cross_document_budget.pdf")
    _plot_interaction_budget(result, args.output / "cross_document_budget.png")
    _plot_bc_fidelity(result, args.output / "bc_layer_fidelity.pdf")
    _plot_bc_fidelity(result, args.output / "bc_layer_fidelity.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
