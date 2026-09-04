"""Aggregate Paper 3.2 manifests while retaining seed-level summaries."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _seed_statistic(values: list[float]) -> dict[str, Any]:
    generator = random.Random(3205)
    bootstrapped = [
        mean(generator.choice(values) for _ in values) for _ in range(10_000)
    ]
    return {
        "seed_values": values,
        "mean": mean(values),
        "standard_deviation": stdev(values) if len(values) > 1 else 0.0,
        "bootstrap_95_ci": [
            _percentile(bootstrapped, 0.025),
            _percentile(bootstrapped, 0.975),
        ],
    }


def _weighted_rows(
    manifests: list[dict[str, Any]], *, weight_key: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        for row in manifest["summary"]["conditions"]:
            key = (str(row["condition"]),)
            if "resource_order_name" in row:
                key += (str(row["resource_order_name"]),)
            grouped[key].append(row)

    result = []
    for rows in grouped.values():
        total_weight = sum(int(row[weight_key]) for row in rows)
        aggregate: dict[str, Any] = {
            "condition": rows[0]["condition"],
            weight_key: total_weight,
        }
        if "resource_order_name" in rows[0]:
            aggregate["resource_order_name"] = rows[0]["resource_order_name"]
        for field in rows[0]:
            if field in aggregate or field in {"condition", "resource_order_name"}:
                continue
            values = [row.get(field) for row in rows]
            if all(value is None for value in values):
                aggregate[field] = None
            elif all(isinstance(value, (int, float)) for value in values if value is not None):
                weighted = [
                    (float(value), int(row[weight_key]))
                    for row, value in zip(rows, values, strict=True)
                    if value is not None
                ]
                aggregate[field] = sum(value * weight for value, weight in weighted) / sum(
                    weight for _, weight in weighted
                )
        result.append(aggregate)
    return sorted(
        result,
        key=lambda row: (str(row["condition"]), str(row.get("resource_order_name", ""))),
    )


def _aggregate_composition(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in manifests:
        for name, row in manifest["summary"]["fresh_packed_comparisons"].items():
            comparisons[name].append(row)

    aggregate_comparisons: dict[str, dict[str, Any]] = {}
    seed_statistics: dict[str, dict[str, Any]] = {}
    count_fields = {"pairs", "output_matches", "first_step_logit_hash_matches"}
    for name, rows in comparisons.items():
        pair_count = sum(int(row["pairs"]) for row in rows)
        aggregate: dict[str, Any] = {
            field: sum(int(row[field]) for row in rows) for field in count_fields
        }
        for field in rows[0]:
            if field in count_fields:
                continue
            aggregate[field] = sum(float(row[field]) * int(row["pairs"]) for row in rows) / pair_count
        aggregate_comparisons[name] = aggregate
        seed_statistics[name] = {
            "output_parity": _seed_statistic(
                [float(row["output_matches"]) / float(row["pairs"]) for row in rows]
            ),
            "first_step_js_divergence": _seed_statistic(
                [float(row["first_step_js_divergence_mean"]) for row in rows]
            ),
            "gold_nll_mean_abs_delta": _seed_statistic(
                [float(row["gold_nll_mean_abs_delta"]) for row in rows]
            ),
        }

    order_sensitivity: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        seed = manifest["seed"]
        for example_id, row in manifest["summary"]["fresh_packed_order_sensitivity"].items():
            order_sensitivity[f"seed{seed}:{example_id}"] = row

    return {
        "conditions": _weighted_rows(manifests, weight_key="examples"),
        "fresh_packed_comparisons": aggregate_comparisons,
        "fresh_packed_order_sensitivity": order_sensitivity,
        "seed_statistics": seed_statistics,
    }


def _aggregate_nonprefix(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    sequence_rows = []
    for manifest in manifests:
        for row in manifest["summary"]["sequence_cumulative"]:
            sequence_rows.append({"seed": manifest["seed"], **row})
    conditions = _weighted_rows(manifests, weight_key="turns")
    seed_statistics = {}
    for condition in {row["condition"] for row in conditions}:
        rows = [
            next(
                row
                for row in manifest["summary"]["conditions"]
                if row["condition"] == condition
            )
            for manifest in manifests
        ]
        seed_statistics[condition] = {
            metric: _seed_statistic([float(row[metric]) for row in rows])
            for metric in (
                "exact_output_parity_with_fresh",
                "newly_encoded_tokens",
                "reused_tokens",
                "token_f1",
                "total_with_materialization_ms",
            )
        }
    return {
        "conditions": conditions,
        "sequence_cumulative": sequence_rows,
        "seed_statistics": seed_statistics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("composition", "nonprefix"), required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in args.manifest]
    seeds = [int(manifest["seed"]) for manifest in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds are not allowed: {seeds}")
    models = {str(manifest["model"]) for manifest in manifests}
    if len(models) != 1:
        raise ValueError(f"all manifests must use the same model: {sorted(models)}")

    aggregate = {
        "schema_version": "paper3.2-multiseed-aggregate-v1",
        "kind": args.kind,
        "model": next(iter(models)),
        "model_revision": manifests[0]["model_revision"],
        "seeds": seeds,
        "source_manifests": [str(path.as_posix()) for path in args.manifest],
        "seed_summaries": [
            {"seed": manifest["seed"], "summary": manifest["summary"]}
            for manifest in manifests
        ],
        "summary": (
            _aggregate_composition(manifests)
            if args.kind == "composition"
            else _aggregate_nonprefix(manifests)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
