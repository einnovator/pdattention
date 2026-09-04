"""Aggregate the five-seed Paper 3.2 MLX fixture experiment.

The powered runner writes one self-contained directory per seed.  This analyzer
keeps seeds as the replication unit and separately checks the selector-frozen
Selected Context/Native Memory pairs at the request level.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


SELECTED = "PRA_SELECTED_CONTEXT_NO_ADAPTOR"
NATIVE = "PRA_NATIVE_MEMORY_NO_ADAPTOR"
MEASURED = {"NO_PRA_STANDARD_RAG", SELECTED, NATIVE}
SUMMARY_METRICS = (
    "token_f1",
    "supporting_document_coverage",
    "physical_context_tokens",
    "visible_prompt_tokens",
    "selected_native_kv_tokens",
    "ttft_ms",
    "total_latency_ms",
    "ingestion_ms",
    "native_reuse",
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _metric(row: Mapping[str, object], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        for container_name in ("retrieval_context_metrics", "serving_metrics"):
            container = row.get(container_name)
            if isinstance(container, Mapping) and name in container:
                value = container[name]
                break
    return float(value) if value is not None else None


def _mean(values: Iterable[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return statistics.fmean(measured) if measured else None


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _pair_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["seed"],
        row["example_id"],
        row["candidate_count"],
        row["token_budget"],
        row["regime"],
        row["selector_profile"],
    )


def _parity(rows: list[dict[str, object]]) -> dict[str, object]:
    pairs: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        condition = str(row.get("condition"))
        if row.get("status") == "MEASURED" and condition in {SELECTED, NATIVE}:
            pairs[_pair_key(row)][condition] = row

    complete = [pair for pair in pairs.values() if set(pair) == {SELECTED, NATIVE}]
    candidate_matches = sum(
        pair[SELECTED]["candidate_receipt_id"] == pair[NATIVE]["candidate_receipt_id"]
        for pair in complete
    )
    receipt_matches = sum(
        pair[SELECTED]["selection_receipt_id"] == pair[NATIVE]["selection_receipt_id"]
        for pair in complete
    )
    interval_matches = sum(
        pair[SELECTED]["selected_intervals"] == pair[NATIVE]["selected_intervals"]
        for pair in complete
    )
    output_matches = sum(
        pair[SELECTED]["prediction"] == pair[NATIVE]["prediction"]
        for pair in complete
    )
    score_matches = sum(
        pair[SELECTED]["token_f1"] == pair[NATIVE]["token_f1"]
        for pair in complete
    )
    return {
        "complete_pairs": len(complete),
        "candidate_receipt_matches": candidate_matches,
        "selection_receipt_matches": receipt_matches,
        "selected_interval_matches": interval_matches,
        "exact_output_matches": output_matches,
        "token_f1_matches": score_matches,
        "all_pairs_transport_equivalent": bool(complete)
        and candidate_matches
        == receipt_matches
        == interval_matches
        == output_matches
        == score_matches
        == len(complete),
    }


def _seed_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "MEASURED" or row.get("condition") not in MEASURED:
            continue
        groups[(row["seed"], row["condition"], row["regime"])].append(row)

    summaries: list[dict[str, object]] = []
    for (seed, condition, regime), values in sorted(groups.items(), key=str):
        summary: dict[str, object] = {
            "seed": seed,
            "condition": condition,
            "regime": regime,
            "examples": len(values),
        }
        for metric in SUMMARY_METRICS:
            summary[metric] = _mean(_metric(row, metric) for row in values)
        summaries.append(summary)
    return summaries


def _aggregate(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        groups[(str(row["condition"]), str(row["regime"]))].append(row)

    result: list[dict[str, object]] = []
    for (condition, regime), values in sorted(groups.items()):
        summary: dict[str, object] = {
            "condition": condition,
            "regime": regime,
            "seeds": len(values),
            "examples_per_seed": int(values[0]["examples"]),
        }
        for metric in SUMMARY_METRICS:
            samples = [float(row[metric]) for row in values if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.fmean(samples) if samples else None
            summary[f"{metric}_seed_sd"] = statistics.stdev(samples) if len(samples) > 1 else None
        result.append(summary)
    return result


def _add_ratios(summary: dict[str, object], aggregate: list[dict[str, object]]) -> None:
    indexed = {(row["condition"], row["regime"]): row for row in aggregate}
    for regime in ("COLD", "WARM"):
        selected = indexed[(SELECTED, regime)]
        native = indexed[(NATIVE, regime)]
        for metric in ("ttft_ms", "total_latency_ms", "ingestion_ms"):
            left = float(selected[f"{metric}_mean"])
            right = float(native[f"{metric}_mean"])
            summary[f"native_over_selected_{regime.lower()}_{metric}"] = right / left


def _plot(path: Path, aggregate: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    indexed = {(row["condition"], row["regime"]): row for row in aggregate}
    labels = ["Selected text\ncold", "Native K/V\ncold", "Selected text\nwarm", "Native K/V\nwarm"]
    keys = [(SELECTED, "COLD"), (NATIVE, "COLD"), (SELECTED, "WARM"), (NATIVE, "WARM")]
    colors = ["#3b82f6", "#ef4444", "#60a5fa", "#f97316"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for axis, metric, title in (
        (axes[0], "ttft_ms", "Time to first token"),
        (axes[1], "total_latency_ms", "Total request latency"),
    ):
        means = [float(indexed[key][f"{metric}_mean"]) for key in keys]
        errors = [float(indexed[key][f"{metric}_seed_sd"] or 0.0) for key in keys]
        axis.bar(labels, means, yerr=errors, color=colors, capsize=3)
        axis.set_title(title)
        axis.set_ylabel("Milliseconds")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    fig.suptitle("Selector-frozen realization on Qwen3-1.7B-4bit (mean +/- seed SD)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for run_dir in sorted(args.input_root.glob("paper3_2_m1_fixture_seed*")):
        manifest = json.loads((run_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
        manifests.append(manifest)
        for row in _load_jsonl(run_dir / "condition_results.jsonl.gz"):
            row["seed"] = manifest["seed"]
            rows.append(row)

    if not rows:
        raise SystemExit(f"No seed runs found below {args.input_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries = _seed_summaries(rows)
    aggregate = _aggregate(seed_summaries)
    parity = _parity(rows)
    result: dict[str, object] = {
        "schema_version": "paper3.2-model-smoke-aggregate-v1",
        "replication_unit": "seed",
        "seeds": sorted(int(manifest["seed"]) for manifest in manifests),
        "unique_questions_per_seed": 15,
        "measured_rows": sum(row.get("status") == "MEASURED" for row in rows),
        "model": manifests[0]["model"],
        "model_revision": manifests[0]["model_revision"],
        "hardware": manifests[0]["hardware"],
        "parity": parity,
        "conditions": aggregate,
        "scope": "controlled fixture; not a natural-task quality claim",
    }
    _add_ratios(result, aggregate)
    _write_csv(args.output_dir / "per_seed_summary.csv", seed_summaries)
    _write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(args.output_dir / "model_smoke_transport", aggregate)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
