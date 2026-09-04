"""Aggregate native-record/reranker manifests at the seed replication level."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _seed_statistic(values: Sequence[float]) -> dict[str, object]:
    rows = [float(value) for value in values]
    generator = random.Random(3217)
    bootstrapped = [
        statistics.fmean(generator.choice(rows) for _ in rows) for _ in range(10_000)
    ]
    return {
        "seed_values": rows,
        "mean": statistics.fmean(rows),
        "standard_deviation": statistics.stdev(rows) if len(rows) > 1 else 0.0,
        "bootstrap_95_ci": [
            _percentile(bootstrapped, 0.025),
            _percentile(bootstrapped, 0.975),
        ],
    }


def _numeric_mean(rows: Sequence[Mapping[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return statistics.fmean(values) if values else None


def _condition_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return str(row["selector"]), str(row["representation"]), str(row["order_name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in args.manifest]
    seeds = [int(row["seed"]) for row in manifests]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds are not allowed: {seeds}")
    models = {(row["model"]["id"], row["model"]["revision"]) for row in manifests}
    if len(models) != 1:
        raise ValueError("native-record aggregates require one immutable model")

    grouped: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for manifest in manifests:
        for row in manifest["condition_summary"]:
            grouped[_condition_key(row)].append(row)

    metrics = (
        "exact_match",
        "token_f1",
        "official_score",
        "gold_answer_mean_nll",
        "supporting_document_coverage",
        "gold_chunk_recall",
        "false_selected_document_fraction",
        "answer_string_availability",
        "selected_source_tokens",
        "visible_prompt_tokens",
        "newly_encoded_native_tokens",
        "reused_native_tokens",
        "ttft_ms",
        "total_latency_ms",
        "reranker_latency_ms",
        "exact_output_agreement_with_packed",
        "first_step_logit_agreement_with_packed",
        "first_step_js_vs_packed",
    )
    conditions = []
    seed_statistics = {}
    for key, rows in sorted(grouped.items()):
        selector, representation, order_name = key
        total_examples = sum(int(row["examples"]) for row in rows)
        condition = {
            "selector": selector,
            "representation": representation,
            "order_name": order_name,
            "examples": total_examples,
        }
        for metric in metrics:
            weighted = [
                (float(row[metric]), int(row["examples"]))
                for row in rows
                if row.get(metric) is not None
            ]
            condition[metric] = (
                sum(value * weight for value, weight in weighted)
                / sum(weight for _, weight in weighted)
                if weighted
                else None
            )
        conditions.append(condition)
        seed_statistics["|".join(key)] = {
            metric: _seed_statistic(
                [float(row[metric]) for row in rows if row.get(metric) is not None]
            )
            for metric in (
                "token_f1",
                "gold_answer_mean_nll",
                "exact_output_agreement_with_packed",
                "first_step_js_vs_packed",
            )
            if any(row.get(metric) is not None for row in rows)
        }

    representation_deltas = {}
    for selector in sorted({str(row["selector"]) for row in conditions}):
        seed_rows = []
        for manifest in manifests:
            canonical = {
                str(row["representation"]): row
                for row in manifest["condition_summary"]
                if row["selector"] == selector
                and row["order_name"] == "canonical"
                and row["representation"]
                in {
                    "PACKED_RAG_TEXT",
                    "PRA_EXPLICIT_RECORDS",
                    "PRA_ROUTED_ROOT",
                }
            }
            if "PACKED_RAG_TEXT" not in canonical:
                continue
            packed = canonical["PACKED_RAG_TEXT"]
            for representation in ("PRA_EXPLICIT_RECORDS", "PRA_ROUTED_ROOT"):
                if representation in canonical:
                    row = canonical[representation]
                    seed_rows.append(
                        {
                            "seed": manifest["seed"],
                            "representation": representation,
                            "token_f1_delta": float(row["token_f1"])
                            - float(packed["token_f1"]),
                            "nll_delta": float(row["gold_answer_mean_nll"])
                            - float(packed["gold_answer_mean_nll"]),
                        }
                    )
        for representation in ("PRA_EXPLICIT_RECORDS", "PRA_ROUTED_ROOT"):
            values = [row for row in seed_rows if row["representation"] == representation]
            if values:
                representation_deltas[f"{selector}|{representation}"] = {
                    "token_f1_delta": _seed_statistic(
                        [float(row["token_f1_delta"]) for row in values]
                    ),
                    "gold_nll_delta": _seed_statistic(
                        [float(row["nll_delta"]) for row in values]
                    ),
                }

    order = {}
    for field in (
        "packed_mean_pairwise_js",
        "record_mean_pairwise_js",
        "packed_unique_outputs",
        "record_unique_outputs",
        "packed_token_f1_variance",
        "record_token_f1_variance",
    ):
        seed_values = []
        for manifest in manifests:
            value = _numeric_mean(manifest["order_sensitivity"], field)
            if value is not None:
                seed_values.append(value)
        if seed_values:
            order[field] = _seed_statistic(seed_values)

    reuse = {}
    for field in (
        "mean_overlap_fraction",
        "mean_exact_prefix_reusable_tokens",
        "mean_newly_encoded_native_tokens",
        "mean_reused_native_tokens",
        "mean_token_f1",
        "mean_packed_token_f1",
        "mean_token_f1_delta_vs_packed",
    ):
        values = [
            float(manifest["reuse_summary"][field])
            for manifest in manifests
            if manifest["reuse_summary"].get(field) is not None
        ]
        if values:
            reuse[field] = _seed_statistic(values)

    model_id, model_revision = next(iter(models))
    aggregate = {
        "schema_version": "paper3.2-native-record-multiseed-v1",
        "model": {"id": model_id, "revision": model_revision},
        "seeds": seeds,
        "source_manifests": [str(path.as_posix()) for path in args.manifest],
        "conditions": conditions,
        "seed_statistics": seed_statistics,
        "representation_deltas": representation_deltas,
        "order_sensitivity": order,
        "reuse": reuse,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()

