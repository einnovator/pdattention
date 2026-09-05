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


def _load(manifest_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows_path = manifest_path.parent / "condition_results.jsonl.gz"
    with gzip.open(rows_path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return manifest, rows


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
        for manifest, _ in runs
    }
    if len(identities) != 1:
        raise ValueError("seed manifests do not share one frozen protocol")

    condition_names = sorted(
        {str(row["condition"]) for _, rows in runs for row in rows},
        key=_condition_order,
    )
    conditions = []
    for condition in condition_names:
        seed_rows = []
        for manifest, rows in runs:
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
    for manifest, rows in runs:
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
        "seeds": [int(manifest["seed"]) for manifest, _ in runs],
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
                }
            )


def _plot(result: Mapping[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = list(result["conditions"])  # type: ignore[arg-type]
    names = [
        str(row["condition"])
        .replace("A_FULL_CAUSAL_RAG", "A full")
        .replace("B_NO_CROSS_DOC_RAG", "B isolated")
        .replace("C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS", "C pre-RoPE")
        .replace("M4_BOUNDARY_ONLY_", "boundary ")
        .replace("M_PREVIOUS_DOC_ONLY", "previous")
        .replace("M_TOP_RANKED_TO_ALL", "top-ranked")
        for row in rows
    ]
    values = [float(row["seed_mean_token_f1"]) for row in rows]
    lower = [
        value - float(row["seed_bootstrap_token_f1_95_ci"][0])
        for value, row in zip(values, rows)
    ]
    upper = [
        float(row["seed_bootstrap_token_f1_95_ci"][1]) - value
        for value, row in zip(values, rows)
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#276FBF", "#C44536", "#2A9D8F"] + ["#6C757D"] * max(0, len(rows) - 3)
    axis.bar(range(len(rows)), values, color=colors[: len(rows)], width=0.72)
    axis.errorbar(
        range(len(rows)), values, yerr=(lower, upper), fmt="none", color="#202020", capsize=3
    )
    axis.set_ylabel("Token F1")
    axis.set_xticks(range(len(rows)), names, rotation=35, ha="right")
    axis.grid(axis="y", alpha=0.25)
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
    _plot(result, args.output / "causal_decomposition_f1.pdf")
    _plot(result, args.output / "causal_decomposition_f1.png")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
