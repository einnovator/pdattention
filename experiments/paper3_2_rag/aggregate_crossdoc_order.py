"""Aggregate frozen-selection cross-document composition order controls."""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence


CONDITIONS = (
    "C_PRA_PRE_ROPE_EXACT_PACKED_OFFSETS",
    "D_GIST_SA_APPEND",
    "E_GIST_SA_BOUNDARY_8",
    "F_GIST_SA_BOUNDARY_32",
)


def _load(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    with gzip.open(path.parent / "condition_results.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return manifest, rows


def _js_topk_tail(
    left: Mapping[str, object], right: Mapping[str, object]
) -> float:
    """Approximate JS on the union of retained tokens plus one tail bucket."""

    left_map = dict(zip(left["token_ids"], left["probabilities"]))
    right_map = dict(zip(right["token_ids"], right["probabilities"]))
    support = sorted(set(left_map) | set(right_map))
    left_values = [float(left_map.get(token, 0.0)) for token in support]
    right_values = [float(right_map.get(token, 0.0)) for token in support]
    left_values.append(float(left["tail_probability"]))
    right_values.append(float(right["tail_probability"]))
    total_left = sum(left_values)
    total_right = sum(right_values)
    left_values = [value / total_left for value in left_values]
    right_values = [value / total_right for value in right_values]
    middle = [(a + b) / 2.0 for a, b in zip(left_values, right_values)]

    def kl(values: Sequence[float]) -> float:
        return sum(
            value * math.log(value / center)
            for value, center in zip(values, middle)
            if value > 0.0 and center > 0.0
        )

    return 0.5 * (kl(left_values) + kl(right_values))


def aggregate(paths: Sequence[Path]) -> dict[str, object]:
    if len(paths) < 2:
        raise ValueError("order aggregation requires at least two manifests")
    runs = [_load(path) for path in paths]
    orders = {str(manifest.get("record_order", "canonical")) for manifest, _ in runs}
    if len(orders) < 2:
        raise ValueError("order aggregation requires at least two record orders")
    grouped: dict[
        tuple[int, str, str], dict[str, Mapping[str, object]]
    ] = {}
    for manifest, rows in runs:
        order = str(manifest.get("record_order", "canonical"))
        seed = int(manifest["seed"])
        for row in rows:
            condition = str(row["condition"])
            if condition in CONDITIONS:
                grouped.setdefault(
                    (seed, str(row["example_id"]), condition), {}
                )[order] = row

    summaries = []
    for condition in CONDITIONS:
        pairs = [
            values
            for (_, _, name), values in grouped.items()
            if name == condition and len(values) == len(orders)
        ]
        pairwise_js = []
        for values in pairs:
            for left_order, right_order in itertools.combinations(sorted(orders), 2):
                left = values[left_order].get("first_step_distribution_topk")
                right = values[right_order].get("first_step_distribution_topk")
                if left is not None and right is not None:
                    pairwise_js.append(_js_topk_tail(left, right))
        summaries.append(
            {
                "condition": condition,
                "matched_examples": len(pairs),
                "mean_unique_outputs": statistics.fmean(
                    len({str(row["prediction"]) for row in values.values()})
                    for values in pairs
                ) if pairs else None,
                "output_flip_rate": statistics.fmean(
                    float(len({str(row["prediction"]) for row in values.values()}) > 1)
                    for values in pairs
                ) if pairs else None,
                "mean_f1_variance": statistics.fmean(
                    statistics.pvariance(float(row["token_f1"]) for row in values.values())
                    for values in pairs
                ) if pairs else None,
                "mean_nll_variance": statistics.fmean(
                    statistics.pvariance(
                        float(row["gold_answer_mean_nll"]) for row in values.values()
                    )
                    for values in pairs
                ) if pairs else None,
                "mean_pairwise_topk_tail_js": (
                    statistics.fmean(pairwise_js) if pairwise_js else None
                ),
                "orders": {
                    order: {
                        "token_f1": statistics.fmean(
                            float(values[order]["token_f1"]) for values in pairs
                        ) if pairs else None,
                        "gold_answer_nll": statistics.fmean(
                            float(values[order]["gold_answer_mean_nll"])
                            for values in pairs
                        ) if pairs else None,
                    }
                    for order in sorted(orders)
                },
            }
        )
    return {
        "schema_version": "paper3.2-crossdoc-order-aggregate-v1",
        "experiment": "crossdoc_composition_order_robustness",
        "orders": sorted(orders),
        "conditions": summaries,
        "source_manifests": [str(path) for path in paths],
        "js_estimator": "union_topk_plus_collapsed_tail_bucket",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
