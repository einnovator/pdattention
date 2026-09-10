"""Report retention and acquisition separately for paired Easy-50 treatments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def stratified_outcomes(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any],
) -> dict[str, Any]:
    """Reduce two official aggregates without hiding success/failure conditioning."""

    baseline_ids = list(baseline["submitted_ids"])
    treatment_ids = list(treatment["submitted_ids"])
    if baseline_ids != treatment_ids:
        raise ValueError("baseline and treatment must use the same ordered parent cohort")
    baseline_solved = set(baseline["resolved_ids"])
    treatment_solved = set(treatment["resolved_ids"])
    retained = [item for item in baseline_ids if item in baseline_solved & treatment_solved]
    regressed = [item for item in baseline_ids if item in baseline_solved - treatment_solved]
    acquired = [item for item in baseline_ids if item in treatment_solved - baseline_solved]
    still_unsolved = [
        item for item in baseline_ids if item not in baseline_solved | treatment_solved
    ]
    positive_count = len(baseline_solved)
    negative_count = len(baseline_ids) - positive_count
    return {
        "parent_total": len(baseline_ids),
        "baseline_success_count": positive_count,
        "baseline_failure_count": negative_count,
        "retained_count": len(retained),
        "retention_rate": len(retained) / positive_count if positive_count else None,
        "regressed_count": len(regressed),
        "regression_rate": len(regressed) / positive_count if positive_count else None,
        "acquired_count": len(acquired),
        "acquisition_rate": len(acquired) / negative_count if negative_count else None,
        "still_unsolved_count": len(still_unsolved),
        "overall_solved": len(treatment_solved),
        "overall_score": len(treatment_solved) / len(baseline_ids),
        "net_solve_delta": len(acquired) - len(regressed),
        "retained_ids": retained,
        "regressed_ids": regressed,
        "acquired_ids": acquired,
        "still_unsolved_ids": still_unsolved,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = stratified_outcomes(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.treatment.read_text(encoding="utf-8")),
    )
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
