"""Aggregate the preregistered five-seed Paper 4 Tier-1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev

from experiments.paper4_training.run_pretrained_tier1 import SEEDS


def aggregate(paths: list[Path]) -> dict:
    """Validate independent seeds and summarize each adaptation regime."""

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    observed = tuple(sorted(int(payload["split_seed"]) for payload in payloads))
    if observed != tuple(sorted(SEEDS)):
        raise ValueError(f"Expected seeds {SEEDS}, received {observed}.")
    models = {payload["model_id"] for payload in payloads}
    if len(models) != 1:
        raise ValueError(f"Seed runs must use one model, received {sorted(models)}.")

    by_regime: dict[str, list[dict]] = {}
    for payload in payloads:
        for result in payload["results"]:
            by_regime.setdefault(result["regime"], []).append(result)

    regimes = []
    for regime, rows in sorted(by_regime.items()):
        evidence = [float(row["evidence_nll_delta"]) for row in rows]
        retention = [float(row["ordinary_retention_nll_delta"]) for row in rows]
        passes = [
            row["regime"] != "frozen_pra"
            and float(row["evidence_nll_delta"])
            <= -float(payloads[0]["configuration"]["minimum_evidence_nll_gain"])
            and float(row["ordinary_retention_nll_delta"])
            <= float(payloads[0]["configuration"]["maximum_retention_nll_loss"])
            for row in rows
        ]
        regimes.append(
            {
                "regime": regime,
                "seed_count": len(rows),
                "mean_evidence_nll_delta": fmean(evidence),
                "evidence_nll_delta_stdev": stdev(evidence),
                "mean_ordinary_retention_nll_delta": fmean(retention),
                "ordinary_retention_nll_delta_stdev": stdev(retention),
                "passing_seeds": sum(passes),
                "all_seeds_pass": all(passes),
            }
        )
    passing = [row["regime"] for row in regimes if row["all_seeds_pass"]]
    return {
        "schema_version": "paper4-pretrained-tier1-five-seed-v1",
        "experiment": "ordinary_language_pretrained_consumer_learning_five_seed",
        "model_id": next(iter(models)),
        "seeds": list(observed),
        "regimes": regimes,
        "consumer_gate": "PASS" if passing else "FAIL",
        "routing_stage": (
            "ENABLED_NEXT" if passing else "BLOCKED_CONSUMER_NOT_REPLICATED"
        ),
        "passing_regimes": passing,
        "source_files": [str(path) for path in paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = aggregate(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"consumer_gate": result["consumer_gate"]}, indent=2))
