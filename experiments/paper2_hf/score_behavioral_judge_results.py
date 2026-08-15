"""Validate, unblind, and summarize Paper 2 behavioral-judge responses.

The blind package presents every underlying comparison in both A/B orders. This
scorer restores the true condition labels from the private truth mapping and
collapses the two presentations before aggregation, so order reversals are
diagnostics rather than duplicate observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCORE_FIELDS = (
    "semantic_equivalence",
    "relative_quality",
    "validity_a",
    "validity_b",
    "confidence",
)
TARGET_CONDITION = {
    "native_no_context_vs_pra": "pra_routed_frozen",
    "native_direct_evidence_vs_pra": "pra_routed_frozen",
    "native_full_context_vs_pra": "pra_routed_frozen",
    "frozen_pra_vs_adapted_pra": "pra_routed_residual_16",
    "gate3_balanced_vs_one_shot": "graph_balanced",
    "gate3_high_vs_one_shot": "graph_high",
    "gate3_balanced_vs_native": "graph_balanced",
    "gate3_balanced_vs_oracle": "graph_balanced",
    "gate3_selected_band_vs_all_layers": "graph_balanced_selected_band",
    "calibration_identical": "control_identical_copy",
    "calibration_corrupted": "control_corrupted_answer",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def _sample_sd(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _validate_response(
    response: dict[str, Any], truth: dict[str, Any], *, allow_partial: bool = False
) -> None:
    if response.get("schema_version") != truth.get("schema_version"):
        raise ValueError("Judge response and truth schema versions differ.")
    if not str(response.get("judge_name", "")).strip():
        raise ValueError("Judge response has no judge_name.")

    rows = response.get("items")
    if not isinstance(rows, list):
        raise ValueError("Judge response items must be a list.")
    truth_ids = {row["item_id"] for row in truth["items"]}
    response_ids = [row.get("item_id") for row in rows]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Judge response contains duplicate item IDs.")
    missing = sorted(truth_ids - set(response_ids))
    extra = sorted(set(response_ids) - truth_ids)
    if extra or (missing and not allow_partial):
        raise ValueError(
            f"Judge response IDs do not match truth: missing={missing[:3]}, extra={extra[:3]}"
        )

    for row in rows:
        for field in SCORE_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{row.get('item_id')} has non-numeric {field}.")
            lower = -100 if field == "relative_quality" else 0
            if not lower <= float(value) <= 100:
                raise ValueError(f"{row.get('item_id')} has out-of-range {field}.")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{row.get('item_id')} has no reason.")
        if len(reason.split()) > 40:
            raise ValueError(f"{row.get('item_id')} reason exceeds 40 words.")


def _orient(row: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    group = truth["comparison_group"]
    target = TARGET_CONDITION.get(group)
    if target is None:
        raise ValueError(f"No target condition is defined for {group}.")
    if target == truth["condition_b"]:
        target_validity = float(row["validity_b"])
        comparator_validity = float(row["validity_a"])
        relative_target = float(row["relative_quality"])
        comparator = truth["condition_a"]
    elif target == truth["condition_a"]:
        target_validity = float(row["validity_a"])
        comparator_validity = float(row["validity_b"])
        relative_target = -float(row["relative_quality"])
        comparator = truth["condition_b"]
    else:
        raise ValueError(f"Target condition {target} is absent from {truth['item_id']}.")
    return {
        "item_id": row["item_id"],
        "pair_group_id": truth["pair_group_id"],
        "source_example_id": truth["source_example_id"],
        "dataset": truth["dataset"],
        "comparison_group": group,
        "target_condition": target,
        "comparator_condition": comparator,
        "semantic_equivalence": float(row["semantic_equivalence"]),
        "relative_quality_target": relative_target,
        "target_validity": target_validity,
        "comparator_validity": comparator_validity,
        "confidence": float(row["confidence"]),
    }


def _collapse_pair(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    if len(rows) != 2:
        raise ValueError(f"Expected two order presentations for {rows[0]['pair_group_id']}.")
    invariant = (
        "source_example_id",
        "dataset",
        "comparison_group",
        "target_condition",
        "comparator_condition",
    )
    if any(rows[0][field] != rows[1][field] for field in invariant):
        raise ValueError(f"Order-reversed pair metadata differs for {rows[0]['pair_group_id']}.")
    metrics = (
        "semantic_equivalence",
        "relative_quality_target",
        "target_validity",
        "comparator_validity",
        "confidence",
    )
    collapsed = {field: rows[0][field] for field in invariant}
    collapsed["pair_group_id"] = rows[0]["pair_group_id"]
    collapsed.update({field: _mean(row[field] for row in rows) for field in metrics})
    consistency = {f"{field}_absolute_difference": abs(rows[0][field] - rows[1][field]) for field in metrics}
    consistency["relative_quality_direction_agreement"] = float(
        math.copysign(1, rows[0]["relative_quality_target"])
        == math.copysign(1, rows[1]["relative_quality_target"])
        if rows[0]["relative_quality_target"] and rows[1]["relative_quality_target"]
        else rows[0]["relative_quality_target"] == rows[1]["relative_quality_target"]
    )
    return collapsed, consistency


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    semantic = [row["semantic_equivalence"] for row in rows]
    relative = [row["relative_quality_target"] for row in rows]
    target_validity = [row["target_validity"] for row in rows]
    comparator_validity = [row["comparator_validity"] for row in rows]
    return {
        "pair_count": len(rows),
        "semantic_equivalence_mean": _mean(semantic),
        "semantic_equivalence_sd": _sample_sd(semantic),
        "semantic_equivalence_ge_75_rate": _mean(value >= 75 for value in semantic),
        "semantic_equivalence_ge_90_rate": _mean(value >= 90 for value in semantic),
        "relative_quality_target_mean": _mean(relative),
        "relative_quality_target_sd": _sample_sd(relative),
        "target_preferred_rate": _mean(value > 0 for value in relative),
        "comparator_preferred_rate": _mean(value < 0 for value in relative),
        "target_validity_mean": _mean(target_validity),
        "comparator_validity_mean": _mean(comparator_validity),
        "target_minus_comparator_validity_mean": _mean(
            target - comparator for target, comparator in zip(target_validity, comparator_validity)
        ),
        "confidence_mean": _mean(row["confidence"] for row in rows),
    }


def score_response(
    response: dict[str, Any], truth: dict[str, Any], *, allow_partial: bool = False
) -> dict[str, Any]:
    """Return pair-collapsed metrics for one validated judge response."""

    _validate_response(response, truth, allow_partial=allow_partial)
    truth_by_id = {row["item_id"]: row for row in truth["items"]}
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in response["items"]:
        oriented = _orient(row, truth_by_id[row["item_id"]])
        by_pair[oriented["pair_group_id"]].append(oriented)

    pairs: list[dict[str, Any]] = []
    consistency: list[dict[str, float]] = []
    for pair_id in sorted(by_pair):
        pair, diagnostics = _collapse_pair(by_pair[pair_id])
        pairs.append(pair)
        consistency.append(diagnostics)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(row["dataset"], row["comparison_group"])].append(row)
    aggregates = []
    for (dataset, group), rows in sorted(grouped.items()):
        aggregates.append(
            {
                "dataset": dataset,
                "comparison_group": group,
                "target_condition": rows[0]["target_condition"],
                "comparator_condition": rows[0]["comparator_condition"],
                **_aggregate(rows),
            }
        )

    consistency_metrics = {
        key: _mean(row[key] for row in consistency)
        for key in consistency[0]
    }
    return {
        "judge_name": response["judge_name"],
        "presentation_count": len(response["items"]),
        "truth_presentation_count": len(truth["items"]),
        "presentation_coverage": len(response["items"]) / len(truth["items"]),
        "underlying_pair_count": len(pairs),
        "truth_pair_count": len(truth["items"]) // 2,
        "pair_coverage": len(pairs) / (len(truth["items"]) // 2),
        "order_reversal": consistency_metrics,
        "aggregates": aggregates,
        "pairs": pairs,
    }


def _compare_judges(
    left: dict[str, Any],
    right: dict[str, Any],
    pair_ids: list[str],
    *,
    dataset: str | None,
    comparison_group: str | None,
) -> dict[str, Any]:
    left_pairs = {row["pair_group_id"]: row for row in left["pairs"]}
    right_pairs = {row["pair_group_id"]: row for row in right["pairs"]}
    record: dict[str, Any] = {
        "judge_a": left["judge_name"],
        "judge_b": right["judge_name"],
        "dataset": dataset,
        "comparison_group": comparison_group,
        "pair_count": len(pair_ids),
    }
    for field in ("semantic_equivalence", "relative_quality_target"):
        a = [left_pairs[pair_id][field] for pair_id in pair_ids]
        b = [right_pairs[pair_id][field] for pair_id in pair_ids]
        record[f"{field}_pearson"] = _pearson(a, b)
        record[f"{field}_mean_absolute_difference"] = _mean(
            abs(x - y) for x, y in zip(a, b)
        )
    record["relative_quality_direction_agreement"] = _mean(
        _sign(left_pairs[pair_id]["relative_quality_target"])
        == _sign(right_pairs[pair_id]["relative_quality_target"])
        for pair_id in pair_ids
    )
    return record


def _cross_judge(
    scored: list[dict[str, Any]], *, allow_partial: bool = False
) -> list[dict[str, Any]]:
    comparisons = []
    for index, left in enumerate(scored):
        left_pairs = {row["pair_group_id"]: row for row in left["pairs"]}
        for right in scored[index + 1 :]:
            right_pairs = {row["pair_group_id"]: row for row in right["pairs"]}
            if set(left_pairs) != set(right_pairs) and not allow_partial:
                raise ValueError("Judges do not cover the same underlying pairs.")
            ids = sorted(set(left_pairs) & set(right_pairs))
            if not ids:
                raise ValueError("Judges do not share any underlying pairs.")
            comparisons.append(
                _compare_judges(
                    left, right, ids, dataset=None, comparison_group=None
                )
            )
            groups: dict[tuple[str, str], list[str]] = defaultdict(list)
            for pair_id in ids:
                row = left_pairs[pair_id]
                groups[(row["dataset"], row["comparison_group"])].append(pair_id)
            for (dataset, group), group_ids in sorted(groups.items()):
                comparisons.append(
                    _compare_judges(
                        left,
                        right,
                        group_ids,
                        dataset=dataset,
                        comparison_group=group,
                    )
                )
    return comparisons


def build_report(
    truth_path: Path,
    response_paths: list[Path],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    scored = []
    response_metadata = []
    for path in response_paths:
        response = json.loads(path.read_text(encoding="utf-8"))
        result = score_response(response, truth, allow_partial=allow_partial)
        scored.append(result)
        response_metadata.append(
            {"judge_name": result["judge_name"], "source_file": path.name, "sha256": _sha256(path)}
        )
    names = [row["judge_name"] for row in scored]
    if len(names) != len(set(names)):
        raise ValueError("Judge names must be unique across response files.")
    return {
        "schema_version": "1.0",
        "evaluation_name": truth["evaluation_name"],
        "partial_responses_allowed": allow_partial,
        "truth_sha256": _sha256(truth_path),
        "responses": response_metadata,
        "judges": [{key: value for key, value in row.items() if key != "pairs"} for row in scored],
        "cross_judge": _cross_judge(scored, allow_partial=allow_partial),
    }


def write_derived_artifacts(
    truth_path: Path,
    response_paths: list[Path],
    output_dir: Path,
    *,
    allow_partial: bool = False,
) -> None:
    """Write validation receipts and unblinded pair rows separately from aggregates."""

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in response_paths:
        response = json.loads(path.read_text(encoding="utf-8"))
        scored = score_response(response, truth, allow_partial=allow_partial)
        slug = "".join(
            character.lower() if character.isalnum() else "_"
            for character in scored["judge_name"]
        ).strip("_")
        receipt = {
            "schema_version": "1.0",
            "judge_name": scored["judge_name"],
            "source_file": path.name,
            "source_sha256": _sha256(path),
            "truth_sha256": _sha256(truth_path),
            "schema_and_ids_valid": True,
            "presentation_count": scored["presentation_count"],
            "truth_presentation_count": scored["truth_presentation_count"],
            "presentation_coverage": scored["presentation_coverage"],
            "underlying_pair_count": scored["underlying_pair_count"],
            "truth_pair_count": scored["truth_pair_count"],
            "pair_coverage": scored["pair_coverage"],
            "order_reversal": scored["order_reversal"],
        }
        pairs = {
            "schema_version": "1.0",
            "judge_name": scored["judge_name"],
            "source_file": path.name,
            "source_sha256": _sha256(path),
            "truth_sha256": _sha256(truth_path),
            "orientation": (
                "relative_quality_target is positive when the named target condition is "
                "preferred; exact A/B reversals are averaged into one row"
            ),
            "pairs": scored["pairs"],
        }
        (output_dir / f"behavioral_judge_validation_{slug}.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / f"behavioral_judge_unblinded_pairs_{slug}.json").write_text(
            json.dumps(pairs, indent=2) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--derived-output-dir",
        type=Path,
        help="Optional directory for separate validation receipts and unblinded pair rows.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Accept response files that contain only complete A/B-reversed pair subsets; "
            "coverage is reported and cross-judge metrics use pair intersections."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        args.truth, args.responses, allow_partial=args.allow_partial
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.derived_output_dir:
        write_derived_artifacts(
            args.truth,
            args.responses,
            args.derived_output_dir,
            allow_partial=args.allow_partial,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
