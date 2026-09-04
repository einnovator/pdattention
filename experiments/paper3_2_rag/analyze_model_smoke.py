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

    def counts(values: list[dict[str, dict[str, object]]]) -> dict[str, object]:
        candidate_matches = sum(
            pair[SELECTED]["candidate_receipt_id"]
            == pair[NATIVE]["candidate_receipt_id"]
            for pair in values
        )
        receipt_matches = sum(
            pair[SELECTED]["selection_receipt_id"]
            == pair[NATIVE]["selection_receipt_id"]
            for pair in values
        )
        interval_matches = sum(
            pair[SELECTED]["selected_intervals"] == pair[NATIVE]["selected_intervals"]
            for pair in values
        )
        output_matches = sum(
            pair[SELECTED]["prediction"] == pair[NATIVE]["prediction"]
            for pair in values
        )
        score_matches = sum(
            pair[SELECTED]["token_f1"] == pair[NATIVE]["token_f1"]
            for pair in values
        )
        return {
            "complete_pairs": len(values),
            "candidate_receipt_matches": candidate_matches,
            "selection_receipt_matches": receipt_matches,
            "selected_interval_matches": interval_matches,
            "exact_output_matches": output_matches,
            "token_f1_matches": score_matches,
            "all_pairs_transport_equivalent": bool(values)
            and candidate_matches
            == receipt_matches
            == interval_matches
            == output_matches
            == score_matches
            == len(values),
        }

    by_selector: dict[str, list[dict[str, dict[str, object]]]] = defaultdict(list)
    for pair in complete:
        by_selector[str(pair[SELECTED]["selector_profile"])].append(pair)
    result = counts(complete)
    result["by_selector_profile"] = {
        profile: counts(values) for profile, values in sorted(by_selector.items())
    }
    return result


def _seed_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "MEASURED" or row.get("condition") not in MEASURED:
            continue
        groups[
            (row["seed"], row["condition"], row["selector_profile"], row["regime"])
        ].append(row)

    summaries: list[dict[str, object]] = []
    for (seed, condition, selector_profile, regime), values in sorted(
        groups.items(), key=str
    ):
        summary: dict[str, object] = {
            "seed": seed,
            "condition": condition,
            "selector_profile": selector_profile,
            "regime": regime,
            "examples": len(values),
        }
        for metric in SUMMARY_METRICS:
            summary[metric] = _mean(_metric(row, metric) for row in values)
        summaries.append(summary)
    return summaries


def _aggregate(seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        groups[
            (
                str(row["condition"]),
                str(row["selector_profile"]),
                str(row["regime"]),
            )
        ].append(row)

    result: list[dict[str, object]] = []
    for (condition, selector_profile, regime), values in sorted(groups.items()):
        summary: dict[str, object] = {
            "condition": condition,
            "selector_profile": selector_profile,
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
    indexed = {
        (row["condition"], row["selector_profile"], row["regime"]): row
        for row in aggregate
    }
    profiles = sorted(
        str(row["selector_profile"])
        for row in aggregate
        if row["condition"] == SELECTED
    )
    ratios: dict[str, dict[str, float]] = {}
    for profile in profiles:
        profile_ratios: dict[str, float] = {}
        for regime in ("COLD", "WARM"):
            selected = indexed[(SELECTED, profile, regime)]
            native = indexed[(NATIVE, profile, regime)]
            for metric in ("ttft_ms", "total_latency_ms", "ingestion_ms"):
                left = float(selected[f"{metric}_mean"])
                right = float(native[f"{metric}_mean"])
                profile_ratios[f"native_over_selected_{regime.lower()}_{metric}"] = (
                    right / left
                )
        ratios[profile] = profile_ratios
    summary["realization_ratios"] = ratios


def _plot(
    path: Path,
    aggregate: list[dict[str, object]],
    *,
    model: str,
    seed_count: int,
    selector_profile: str,
) -> None:
    import matplotlib.pyplot as plt

    indexed = {
        (row["condition"], row["selector_profile"], row["regime"]): row
        for row in aggregate
    }
    labels = ["Selected text\ncold", "Native K/V\ncold", "Selected text\nwarm", "Native K/V\nwarm"]
    keys = [
        (SELECTED, selector_profile, "COLD"),
        (NATIVE, selector_profile, "COLD"),
        (SELECTED, selector_profile, "WARM"),
        (NATIVE, selector_profile, "WARM"),
    ]
    missing = [key for key in keys if key not in indexed]
    if missing:
        available = sorted(
            {str(row["selector_profile"]) for row in aggregate}
        )
        raise ValueError(
            f"Selector profile {selector_profile!r} has no complete selected/native "
            f"cold/warm series; missing={missing}, available={available}"
        )
    colors = ["#3b82f6", "#ef4444", "#60a5fa", "#f97316"]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for axis, metric, title in (
        (axes[0], "ttft_ms", "Time to first token"),
        (axes[1], "total_latency_ms", "Total request latency"),
    ):
        means = [float(indexed[key][f"{metric}_mean"]) for key in keys]
        errors = [float(indexed[key][f"{metric}_seed_sd"] or 0.0) for key in keys]
        axis.bar(
            labels,
            means,
            yerr=errors if seed_count > 1 else None,
            color=colors,
            capsize=3,
        )
        axis.set_title(title)
        axis.set_ylabel("Milliseconds")
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", labelsize=8)
    suffix = "mean +/- seed SD" if seed_count > 1 else "cohort mean"
    fig.suptitle(f"Selector-frozen realization on {model} ({suffix})")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-pattern", default="paper3_2_m1_fixture_seed*")
    parser.add_argument("--selector-profile", default="pra_generic")
    parser.add_argument(
        "--scope",
        default="controlled fixture; not a natural-task quality claim",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for run_dir in sorted(args.input_root.glob(args.run_pattern)):
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
    question_counts = [len(manifest.get("question_ids", ())) for manifest in manifests]
    result: dict[str, object] = {
        "schema_version": "paper3.2-model-run-aggregate-v1",
        "replication_unit": "seed",
        "seeds": sorted(int(manifest["seed"]) for manifest in manifests),
        "unique_questions_per_seed": question_counts[0]
        if len(set(question_counts)) == 1
        else question_counts,
        "measured_rows": sum(row.get("status") == "MEASURED" for row in rows),
        "model": manifests[0]["model"],
        "model_revision": manifests[0]["model_revision"],
        "hardware": manifests[0]["hardware"],
        "parity": parity,
        "conditions": aggregate,
        "scope": args.scope,
    }
    _add_ratios(result, aggregate)
    _write_csv(args.output_dir / "per_seed_summary.csv", seed_summaries)
    _write_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    (args.output_dir / "aggregate_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _plot(
        args.output_dir / "model_smoke_transport",
        aggregate,
        model=str(manifests[0]["model"]),
        seed_count=len(manifests),
        selector_profile=args.selector_profile,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
