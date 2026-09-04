"""Fit and evaluate a small held-out contextual-repair policy.

The policy deliberately uses only request-visible geometry.  It never reads an
answer, a support label, or an evaluation outcome at decision time.  Outcomes
from calibration seeds select an action for each coarse feature cell; separate
seed manifests measure the frozen policy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def _rows(manifest_path: Path) -> list[dict[str, object]]:
    result_path = manifest_path.parent / "condition_results.jsonl.gz"
    with gzip.open(result_path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def _median(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return statistics.median(float(row[field]) for row in rows)


def _feature_key(
    row: Mapping[str, object], *, token_median: float, resource_median: float
) -> tuple[str, str, str]:
    return (
        str(row.get("question_type", "unknown")),
        "short" if float(row["physical_native_tokens"]) <= token_median else "long",
        "few" if len(row["resource_order"]) <= resource_median else "many",
    )


def _is_candidate(row: Mapping[str, object], max_repair_fraction: float) -> bool:
    condition = str(row["condition"])
    if condition == "NATIVE_GLOBAL_REBOUND":
        return True
    if not condition.startswith("REPAIR_"):
        return False
    fraction = row.get("repair_fraction")
    return fraction is not None and float(fraction) <= max_repair_fraction


def _loss(row: Mapping[str, object], cost_weight: float) -> float:
    fraction = float(row.get("repair_fraction") or 0.0)
    return float(row["first_step_js_divergence"]) + cost_weight * fraction


def _best_action(rows: Iterable[Mapping[str, object]], cost_weight: float) -> str:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(_loss(row, cost_weight))
    if not grouped:
        return "NATIVE_GLOBAL_REBOUND"
    return min(grouped, key=lambda name: (statistics.fmean(grouped[name]), name))


def fit_policy(
    rows: Sequence[Mapping[str, object]],
    *,
    cost_weight: float,
    max_repair_fraction: float,
) -> dict[str, object]:
    base = [row for row in rows if _is_candidate(row, max_repair_fraction)]
    token_median = _median(base, "physical_native_tokens")
    resource_median = statistics.median(len(row["resource_order"]) for row in base)
    cells: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in base:
        cells[
            _feature_key(
                row, token_median=token_median, resource_median=resource_median
            )
        ].append(row)
    return {
        "token_median": token_median,
        "resource_median": resource_median,
        "global_action": _best_action(base, cost_weight),
        "cell_actions": {
            "|".join(key): _best_action(values, cost_weight)
            for key, values in sorted(cells.items())
        },
        "cost_weight": cost_weight,
        "max_repair_fraction": max_repair_fraction,
    }


def _summarize(
    rows: Sequence[Mapping[str, object]], fresh: Mapping[tuple[str, str], Mapping[str, object]]
) -> dict[str, object]:
    return {
        "examples": len(rows),
        "mean_first_step_js": statistics.fmean(
            float(row.get("first_step_js_divergence") or 0.0) for row in rows
        ),
        "exact_output_recovery": statistics.fmean(
            str(row["prediction"])
            == str(fresh[(str(row["example_id"]), str(row["resource_order_name"]))]["prediction"])
            for row in rows
        ),
        "mean_token_f1": statistics.fmean(float(row["token_f1"]) for row in rows),
        "mean_gold_answer_nll": statistics.fmean(
            float(row["gold_answer_mean_nll"]) for row in rows
        ),
        "mean_repair_fraction": statistics.fmean(
            float(row.get("repair_fraction") or 0.0) for row in rows
        ),
        "mean_repaired_tokens": statistics.fmean(
            float(row.get("repaired_token_count") or 0.0) for row in rows
        ),
    }


def evaluate_policy(
    rows: Sequence[Mapping[str, object]], policy: Mapping[str, object]
) -> dict[str, object]:
    fresh = {
        (str(row["example_id"]), str(row["resource_order_name"])): row
        for row in rows
        if row["condition"] == "FRESH_PACKED"
    }
    candidates: dict[tuple[str, str], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        if _is_candidate(row, float(policy["max_repair_fraction"])):
            key = str(row["example_id"]), str(row["resource_order_name"])
            candidates[key][str(row["condition"])] = row

    selected = []
    global_rows = []
    rebound_rows = []
    cell_actions = policy["cell_actions"]
    for key, actions in sorted(candidates.items()):
        exemplar = next(iter(actions.values()))
        cell = "|".join(
            _feature_key(
                exemplar,
                token_median=float(policy["token_median"]),
                resource_median=float(policy["resource_median"]),
            )
        )
        action = str(cell_actions.get(cell, policy["global_action"]))
        selected.append(actions.get(action, actions["NATIVE_GLOBAL_REBOUND"]))
        global_rows.append(
            actions.get(str(policy["global_action"]), actions["NATIVE_GLOBAL_REBOUND"])
        )
        rebound_rows.append(actions["NATIVE_GLOBAL_REBOUND"])

    selected_repair_tokens = statistics.fmean(
        float(row.get("repaired_token_count") or 0.0) for row in selected
    )
    fixed_repair_tokens = statistics.fmean(
        float(row.get("repaired_token_count") or 0.0) for row in global_rows
    )

    return {
        "fresh_packed_reference": _summarize(list(fresh.values()), fresh),
        "query_conditioned_policy": _summarize(selected, fresh),
        "best_fixed_calibration_action": {
            "action": policy["global_action"],
            **_summarize(global_rows, fresh),
        },
        "rebound_without_repair": _summarize(rebound_rows, fresh),
        "repaired_token_savings_vs_best_fixed": (
            1.0 - selected_repair_tokens / fixed_repair_tokens
            if fixed_repair_tokens > 0.0
            else 0.0
        ),
        "selected_action_counts": dict(
            sorted(
                (action, sum(row["condition"] == action for row in selected))
                for action in {str(row["condition"]) for row in selected}
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", action="append", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", action="append", type=Path, required=True)
    parser.add_argument("--cost-weight", type=float, default=0.05)
    parser.add_argument("--max-repair-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.max_repair_fraction < 1.0:
        parser.error("max repair fraction must be in [0, 1)")
    calibration_rows = [
        row for path in args.calibration_manifest for row in _rows(path)
    ]
    evaluation_rows = [row for path in args.evaluation_manifest for row in _rows(path)]
    policy = fit_policy(
        calibration_rows,
        cost_weight=args.cost_weight,
        max_repair_fraction=args.max_repair_fraction,
    )
    result = {
        "schema_version": "paper3.2-heldout-repair-policy-v1",
        "calibration_manifests": [str(path.as_posix()) for path in args.calibration_manifest],
        "evaluation_manifests": [str(path.as_posix()) for path in args.evaluation_manifest],
        "policy": policy,
        "evaluation": evaluate_policy(evaluation_rows, policy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
