"""Reduce the task-aware region/layer oracle without mixing selection and use."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper3_3_crossdoc_expansion.summarize_generation import (
    BOOTSTRAP_SEEDS,
    _bootstrap_interval,
    summarize_generation,
)


SCHEMA_VERSION = "paper3.3-region-layer-audit-publication-v1"
REGIONS = ("prefix", "middle", "suffix")


def _estimate(values: Sequence[float], *, replicates: int) -> dict[str, object]:
    if not values:
        raise ValueError("cannot summarize an empty paired measurement")
    intervals = [
        _bootstrap_interval(values, seed=seed, replicates=replicates)
        for seed in BOOTSTRAP_SEEDS
    ]
    return {
        "mean": statistics.fmean(values),
        "conservative_ci95": [
            min(interval[0] for interval in intervals),
            max(interval[1] for interval in intervals),
        ],
        "examples": len(values),
    }


def summarize_region_layer_audit(
    run: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
    *,
    replicates: int = 2_000,
) -> dict[str, object]:
    """Report selector utility and realized consumption as separate estimands."""

    if replicates <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    policy = run["policy"]
    band_count = int(policy["layer_band_count"])
    expected_cells = band_count * len(REGIONS) ** 2
    diagnostics_by_example: dict[str, dict[tuple[int, str, str], Mapping[str, object]]] = {}
    for row in diagnostics:
        example_id = str(row["example_id"])
        key = (
            int(row["band_index"]),
            str(row["source_region"]),
            str(row["target_region"]),
        )
        by_cell = diagnostics_by_example.setdefault(example_id, {})
        if key in by_cell:
            raise ValueError(f"duplicate diagnostic cell for {example_id}: {key}")
        by_cell[key] = row
    if not diagnostics_by_example:
        raise ValueError("region/layer audit contains no diagnostic rows")
    incomplete = {
        example_id: len(by_cell)
        for example_id, by_cell in diagnostics_by_example.items()
        if len(by_cell) != expected_cells
    }
    if incomplete:
        raise ValueError(f"incomplete region/layer matrices: {incomplete}")

    cell_summary = []
    for band_index in range(band_count):
        for source_region in REGIONS:
            for target_region in REGIONS:
                key = (band_index, source_region, target_region)
                cell_rows = [by_cell[key] for by_cell in diagnostics_by_example.values()]
                gains = [float(row["incremental_gold_nll_gain"]) for row in cell_rows]
                cell_summary.append(
                    {
                        "band_index": band_index,
                        "layers": list(cell_rows[0]["layers"]),
                        "source_region": source_region,
                        "target_region": target_region,
                        "is_boundary_hypothesis": (
                            source_region == "suffix" and target_region == "prefix"
                        ),
                        "incremental_gold_nll_gain": _estimate(
                            gains, replicates=replicates
                        ),
                        "positive_gain_fraction": sum(gain > 0.0 for gain in gains)
                        / len(gains),
                        "selected_physical_edge_fraction_mean": statistics.fmean(
                            float(row["selected_physical_edge_fraction"])
                            for row in cell_rows
                        ),
                    }
                )
    cell_summary.sort(
        key=lambda row: -float(row["incremental_gold_nll_gain"]["mean"])
    )

    boundary_specificity = []
    for band_index in range(band_count):
        contrasts = []
        for by_cell in diagnostics_by_example.values():
            boundary = float(
                by_cell[(band_index, "suffix", "prefix")][
                    "incremental_gold_nll_gain"
                ]
            )
            controls = [
                float(row["incremental_gold_nll_gain"])
                for key, row in by_cell.items()
                if key[0] == band_index and key[1:] != ("suffix", "prefix")
            ]
            contrasts.append(boundary - statistics.fmean(controls))
        boundary_specificity.append(
            {
                "band_index": band_index,
                "boundary_minus_equal_cost_controls": _estimate(
                    contrasts, replicates=replicates
                ),
            }
        )

    rows_by_example: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        example_id = str(row["example_id"])
        condition = str(row["condition"])
        by_condition = rows_by_example.setdefault(example_id, {})
        if condition in by_condition:
            raise ValueError(f"duplicate condition row for {example_id}: {condition}")
        by_condition[condition] = row
    required = {
        "NO_CROSS_DOC_PACKED",
        "TASK_ORACLE_REGION_LAYER_SINGLETON",
        "TASK_ORACLE_REGION_LAYER_UNION",
    }
    incomplete_conditions = {
        example_id: sorted(required - set(by_condition))
        for example_id, by_condition in rows_by_example.items()
        if not required.issubset(by_condition)
    }
    if incomplete_conditions:
        raise ValueError(f"incomplete consumption conditions: {incomplete_conditions}")
    if set(rows_by_example) != set(diagnostics_by_example):
        raise ValueError("selection and consumption example cohorts do not match")

    consumption = []
    for condition in (
        "TASK_ORACLE_REGION_LAYER_SINGLETON",
        "TASK_ORACLE_REGION_LAYER_UNION",
    ):
        condition_rows = [by_condition[condition] for by_condition in rows_by_example.values()]
        blocked_rows = [
            by_condition["NO_CROSS_DOC_PACKED"]
            for by_condition in rows_by_example.values()
        ]
        consumption.append(
            {
                "condition": condition,
                "predicted_additive_gold_nll_gain": _estimate(
                    [
                        float(row["additive_predicted_gold_nll_gain"])
                        for row in condition_rows
                    ],
                    replicates=replicates,
                ),
                "realized_gold_nll_gain": _estimate(
                    [
                        float(blocked["gold_answer_mean_nll"])
                        - float(row["gold_answer_mean_nll"])
                        for blocked, row in zip(blocked_rows, condition_rows)
                    ],
                    replicates=replicates,
                ),
                "realized_token_f1_gain": _estimate(
                    [
                        float(row["token_f1"]) - float(blocked["token_f1"])
                        for blocked, row in zip(blocked_rows, condition_rows)
                    ],
                    replicates=replicates,
                ),
                "realized_official_score_gain": _estimate(
                    [
                        float(row["official_multihop_rag_score"])
                        - float(blocked["official_multihop_rag_score"])
                        for blocked, row in zip(blocked_rows, condition_rows)
                    ],
                    replicates=replicates,
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "git_commit": run["git_commit"],
            "model": run["model"],
            "model_revision": run["model_revision"],
            "split": run["split"],
            "selection_cache_sha256": run.get("selection_cache_sha256"),
            "policy": policy,
            "examples": len(rows_by_example),
        },
        "interpretation": {
            "selection_quality": "gold-answer-NLL singleton oracle; not deployable",
            "consumption_quality": "realized decoding from selected singleton or ranked union",
            "runtime": "dense host replay; no sparse-kernel speed claim",
            "bootstrap_note": "resampling seeds are not independent model runs",
        },
        "selection_quality": {
            "cell_summary": cell_summary,
            "boundary_specificity": boundary_specificity,
        },
        "consumption_quality": consumption,
        "generation_quality": summarize_generation(
            run, rows, replicates=replicates
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    args = parser.parse_args()
    run = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (args.run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    diagnostics = [
        json.loads(line)
        for line in (args.run_dir / "region_layer_diagnostics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    result = summarize_region_layer_audit(
        run, rows, diagnostics, replicates=args.bootstrap_replicates
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "region_layer_publication_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "examples": run["examples"]}))


if __name__ == "__main__":
    main()
