"""Model-independent scalar aggregation across trials and seeds."""

from __future__ import annotations

import math
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from .state import atomic_write_json, read_json


def _numeric_metrics(value: dict) -> dict[str, float]:
    return {
        key: float(item)
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
    }


def aggregate_metrics(run_dir: str | Path) -> dict:
    """Aggregate scalar metrics, grouping seed-only variants together."""

    run_dir = Path(run_dir)
    groups = defaultdict(list)
    for trial_dir in sorted((run_dir / "trials").glob("*")):
        manifest = read_json(trial_dir / "experiment.json")
        metrics = read_json(trial_dir / "metric.json")
        status = read_json(trial_dir / "status.json", {})
        if not manifest or not metrics or status.get("state") != "SUCCEEDED":
            continue
        parameters = dict(manifest.get("parameters") or {})
        parameters.pop("seed", None)
        parameters = {key: value for key, value in parameters.items() if not key.startswith("_")}
        key = json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        groups[key].append(_numeric_metrics(metrics))

    grouped = {}
    for key, rows in groups.items():
        metric_names = sorted(set().union(*(row.keys() for row in rows)))
        summary = {}
        for metric in metric_names:
            values = [row[metric] for row in rows if metric in row]
            summary[metric] = {
                "count": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        grouped[key] = summary
    result = {"groups": grouped, "successful_trials": sum(len(rows) for rows in groups.values())}
    atomic_write_json(run_dir / "aggregate.json", result)
    return result
