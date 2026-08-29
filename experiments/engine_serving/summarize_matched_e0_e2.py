"""Summarize the cross-engine matched E0 selected-text versus E2 native-K/V run."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


CONDITIONS = ("e0_selected_text", "e2_native_kv")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numbers(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _timing_key(row: dict[str, object]) -> str:
    return "online_ttft_ms" if row.get("online_ttft_ms") is not None else "ttft_ms"


def _normalize(payload: dict[str, object], source: Path) -> list[dict[str, object]]:
    normalized = []
    for raw in payload["rows"]:
        row = dict(raw)
        ttft = float(row[_timing_key(row)])
        ingestion = row.get("one_time_ingestion_ms")
        if row.get("cold_end_to_end_ttft_ms") is None and row["reuse_state"] == "cold":
            row["cold_end_to_end_ttft_ms"] = ttft + float(ingestion or 0.0)
        row["online_ttft_ms"] = ttft
        row["engine"] = payload["engine"]
        row["model_id"] = payload["model_id"]
        row["source_file"] = source.as_posix()
        normalized.append(row)
    return normalized


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(
            str(row["engine"]),
            str(row["dataset"]),
            str(row["condition"]),
            str(row["reuse_state"]),
        )].append(row)
    result = []
    for (engine, dataset, condition, reuse), group in sorted(groups.items()):
        ttft = _numbers(group, "online_ttft_ms")
        completion = _numbers(group, "completion_latency_ms")
        result.append(
            {
                "engine": engine,
                "dataset": dataset,
                "condition": condition,
                "reuse_state": reuse,
                "examples": len(group),
                "exact_match": mean(_numbers(group, "exact_match")),
                "token_f1": mean(_numbers(group, "token_f1")),
                "visible_prompt_tokens": mean(_numbers(group, "visible_prompt_tokens")),
                "selected_native_tokens": mean(_numbers(group, "selected_native_tokens")),
                "selected_kv_mib": mean(_numbers(group, "selected_kv_bytes")) / 2**20,
                "one_time_ingestion_ms": (
                    mean(_numbers(group, "one_time_ingestion_ms"))
                    if _numbers(group, "one_time_ingestion_ms")
                    else None
                ),
                "cold_end_to_end_ttft_ms": (
                    mean(_numbers(group, "cold_end_to_end_ttft_ms"))
                    if _numbers(group, "cold_end_to_end_ttft_ms")
                    else None
                ),
                "ttft_p50_ms": _percentile(ttft, 0.50),
                "ttft_p95_ms": _percentile(ttft, 0.95),
                "ttft_p99_ms": _percentile(ttft, 0.99),
                "itl_mean_ms": mean(_numbers(group, "itl_ms")),
                "completion_p50_ms": _percentile(completion, 0.50),
                "completion_p95_ms": _percentile(completion, 0.95),
                "completion_p99_ms": _percentile(completion, 0.99),
            }
        )
    return result


def _parity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pairs: dict[tuple[str, str, str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["engine"]),
            str(row["dataset"]),
            str(row["example_id"]),
            str(row["reuse_state"]),
        )
        pairs[key][str(row["condition"])] = row
    grouped: dict[tuple[str, str], list[dict[str, dict[str, object]]]] = defaultdict(list)
    for (engine, dataset, _example, _reuse), pair in pairs.items():
        if set(pair) == set(CONDITIONS):
            grouped[(engine, dataset)].append(pair)
    result = []
    for (engine, dataset), complete in sorted(grouped.items()):
        exact_outputs = sum(
            pair[CONDITIONS[0]].get("output") == pair[CONDITIONS[1]].get("output")
            for pair in complete
        )
        result.append(
            {
                "engine": engine,
                "dataset": dataset,
                "paired_requests": len(complete),
                "exact_output_parity": exact_outputs / len(complete),
                "mean_f1_delta_e2_minus_e0": mean(
                    float(pair[CONDITIONS[1]]["token_f1"])
                    - float(pair[CONDITIONS[0]]["token_f1"])
                    for pair in complete
                ),
                "mean_absolute_f1_delta": mean(
                    abs(
                        float(pair[CONDITIONS[1]]["token_f1"])
                        - float(pair[CONDITIONS[0]]["token_f1"])
                    )
                    for pair in complete
                ),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, aggregates: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cold = [row for row in aggregates if row["reuse_state"] == "cold"]
    identities = sorted({(str(row["engine"]), str(row["dataset"])) for row in cold})
    lookup = {
        (str(row["engine"]), str(row["dataset"]), str(row["condition"])): row
        for row in cold
    }
    labels = [f"{engine}\n{dataset}" for engine, dataset in identities]
    visible_reduction = []
    f1_delta = []
    for engine, dataset in identities:
        e0 = lookup[(engine, dataset, CONDITIONS[0])]
        e2 = lookup[(engine, dataset, CONDITIONS[1])]
        visible_reduction.append(
            100.0
            * (float(e0["visible_prompt_tokens"]) - float(e2["visible_prompt_tokens"]))
            / float(e0["visible_prompt_tokens"])
        )
        f1_delta.append(float(e2["token_f1"]) - float(e0["token_f1"]))

    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.7), sharex=True)
    axes[0].bar(x, visible_reduction, color="#247a7a")
    axes[0].set_ylabel("Visible-token reduction (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, f1_delta, color=["#2f7d32" if value >= 0 else "#b4473d" for value in f1_delta])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Answer F1: E2 - E0")
    axes[1].set_xticks(x, labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = []
    expanded = []
    for value in args.inputs:
        matches = [Path(match) for match in glob.glob(str(value))]
        expanded.extend(matches or [value])
    for source in expanded:
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows.extend(_normalize(payload, source))
        sources.append(source.as_posix())
    aggregates = _aggregate(rows)
    parity = _parity(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matched_e0_e2_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "cross_engine_matched_e0_e2_summary_v1",
                "sources": sources,
                "aggregates": aggregates,
                "parity": parity,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "matched_e0_e2_summary.csv", aggregates)
    _write_csv(args.output_dir / "matched_e0_e2_parity.csv", parity)
    _plot(args.output_dir / "matched_e0_e2_summary.png", aggregates)


if __name__ == "__main__":
    main()
