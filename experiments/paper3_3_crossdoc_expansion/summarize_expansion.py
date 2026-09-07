"""Select a frozen expansion policy and export paired publication statistics."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "paper3.3-crossdoc-expansion-publication-v1"
DEFAULT_BOOTSTRAP_SEEDS = (11, 23, 37, 71, 101)


def _config_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["mode"],
        bool(row["query_conditioned"]),
        row["direction"],
        row["granularity"],
        int(row["top_k_per_pair"]),
        int(row["max_extra_tokens"]),
    )


def select_validation_config(
    summary: Sequence[Mapping[str, object]],
    *,
    max_extra_fraction: float = 0.20,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Choose the non-oracle frontier point and matched-budget oracle on validation."""

    eligible = [
        row
        for row in summary
        if row["mode"] not in {"none", "oracle"}
        and float(row["extra_native_fraction_mean"]) <= max_extra_fraction
    ]
    oracles = [
        row
        for row in summary
        if row["mode"] == "oracle"
        and float(row["extra_native_fraction_mean"]) <= max_extra_fraction
    ]
    if not eligible or not oracles:
        raise ValueError("validation summary lacks an eligible policy or oracle")
    ordering = lambda row: (
        -float(row["supporting_span_coverage_delta"]),
        float(row["distractor_fraction_mean"]),
        float(row["extra_native_fraction_mean"]),
        _config_key(row),
    )
    return min(eligible, key=ordering), min(oracles, key=ordering)


def _matching_rows(
    rows: Sequence[Mapping[str, object]], config: Mapping[str, object]
) -> list[Mapping[str, object]]:
    key = _config_key(config)
    matched = [row for row in rows if _config_key(row) == key]
    return sorted(matched, key=lambda row: str(row["example_id"]))


def paired_bootstrap_summary(
    rows: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    *,
    metric: str,
    initial_metric: str,
    seeds: Sequence[int] = DEFAULT_BOOTSTRAP_SEEDS,
    replicates: int = 2_000,
) -> dict[str, object]:
    """Bootstrap paired per-example changes without relabeling seeds as trials."""

    matched = _matching_rows(rows, config)
    differences = [float(row[metric]) - float(row[initial_metric]) for row in matched]
    if not differences:
        raise ValueError(f"no rows match {_config_key(config)}")
    intervals = []
    for seed in seeds:
        generator = random.Random(seed)
        draws = sorted(
            statistics.fmean(generator.choice(differences) for _ in differences)
            for _ in range(replicates)
        )
        intervals.append(
            {
                "seed": seed,
                "ci95": [
                    draws[int(0.025 * replicates)],
                    draws[min(int(0.975 * replicates), replicates - 1)],
                ],
            }
        )
    return {
        "examples": len(differences),
        "observed_mean_delta": statistics.fmean(differences),
        "bootstrap_replicates_per_seed": replicates,
        "bootstrap_seed_intervals": intervals,
        "conservative_ci95": [
            min(row["ci95"][0] for row in intervals),
            max(row["ci95"][1] for row in intervals),
        ],
        "uncertainty_label": "five_seed_paired_bootstrap_not_independent_policy_runs",
    }


def build_publication_summary(
    run: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    max_extra_fraction: float = 0.20,
) -> dict[str, object]:
    """Build the compact, provenance-linked artifact consumed by the paper."""

    best, oracle = select_validation_config(
        run["summary"], max_extra_fraction=max_extra_fraction
    )
    best_gain = float(best["supporting_span_coverage_delta"])
    oracle_gain = float(oracle["supporting_span_coverage_delta"])
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "git_commit": run["git_commit"],
            "dataset": run["dataset"],
            "split_name": run["split_name"],
            "split_digest": run["split_digest"],
            "selector": run["selector"],
            "reranker_revision": run["reranker_revision"],
            "dense_encoder": run.get("dense_encoder"),
            "dense_revision": run.get("dense_revision"),
            "policy_parameters": run.get("policy_parameters"),
            "selection_cache_sha256": run.get("selection_cache_sha256"),
            "examples": run["question_count_evaluated"],
        },
        "selection_rule": {
            "split": "validation",
            "objective": "max_support_span_coverage_delta_then_min_distractors_then_min_extra_kv",
            "max_extra_native_fraction": max_extra_fraction,
            "test_locked": True,
        },
        "selected_policy": dict(best),
        "oracle": dict(oracle),
        "oracle_gap_recovery": best_gain / oracle_gain if oracle_gain else None,
        "requested_to_deduplicated_token_ratio": (
            float(best["requested_cross_tokens_mean"])
            / float(best["deduplicated_cross_tokens_mean"])
            if float(best["deduplicated_cross_tokens_mean"])
            else None
        ),
        "paired_uncertainty": {
            "supporting_span_coverage": paired_bootstrap_summary(
                rows,
                best,
                metric="supporting_span_coverage",
                initial_metric="initial_supporting_span_coverage",
            ),
            "gold_chunk_recall": paired_bootstrap_summary(
                rows,
                best,
                metric="gold_chunk_recall",
                initial_metric="initial_gold_chunk_recall",
            ),
            "answer_string_availability": paired_bootstrap_summary(
                rows,
                best,
                metric="answer_string_availability",
                initial_metric="initial_answer_string_availability",
            ),
        },
        "qualification": {
            "status": "RESEARCH_ONLY",
            "sdk_exposable": False,
            "reasons": [
                "generation gate pending",
                "model-size and cross-family gates pending",
                "parameter-free policy recovers less than half the oracle gap",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-extra-fraction", type=float, default=0.20)
    args = parser.parse_args()
    run = json.loads(args.run.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines()]
    result = build_publication_summary(
        run, rows, max_extra_fraction=args.max_extra_fraction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected": result["selected_policy"]}))


if __name__ == "__main__":
    main()
