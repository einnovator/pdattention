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


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = _numbers(rows, key)
    return mean(values) if values else None


def _ttft(row: dict[str, object]) -> float | None:
    value = row.get("online_ttft_ms")
    if value is None:
        value = row.get("ttft_ms")
    return None if value is None else float(value)


def _normalize(payload: dict[str, object], source: Path) -> list[dict[str, object]]:
    normalized = []
    for raw in payload["rows"]:
        if payload.get("schema_version") == "2.0":
            metrics = raw["metrics"]
            extra = raw.get("extra", {})
            selection = raw["selection"]
            serving = metrics["serving"]
            ingestion = metrics["ingestion"]
            total = serving["total_latency_ms"]
            usable = ingestion["time_to_usable_context_ms"]
            row = {
                "engine": payload["engine"],
                "model_id": payload["model_id"],
                "dataset": selection["dataset"],
                "example_id": selection["example_id"],
                "selection_id": selection["selection_id"],
                "condition": raw["condition"],
                "regime": raw["regime"],
                "request_ordinal": raw["request_ordinal"],
                "query_sha256": raw["query_sha256"],
                "output": raw["output"],
                "source_file": source.as_posix(),
                **metrics["quality"],
                **metrics["input"],
                **metrics["pra"],
                **ingestion,
                **serving,
                **metrics["reuse"],
                "cold_end_to_end_ttft_ms": (
                    float(usable) + float(serving["ttft_ms"])
                    if raw["regime"] == "cold_one_shot"
                    and usable is not None
                    and serving["ttft_ms"] is not None
                    else None
                ),
                "cold_end_to_end_completion_ms": (
                    float(usable) + float(total)
                    if raw["regime"] == "cold_one_shot"
                    and usable is not None
                    and total is not None
                    else None
                ),
                "concurrency_execution": extra.get("concurrency_execution"),
            }
            normalized.append(row)
            continue
        row = dict(raw)
        ttft = _ttft(row)
        ingestion = row.get("one_time_ingestion_ms")
        if (
            row.get("cold_end_to_end_ttft_ms") is None
            and row["reuse_state"] == "cold"
            and ttft is not None
        ):
            row["cold_end_to_end_ttft_ms"] = ttft + float(ingestion or 0.0)
        row["online_ttft_ms"] = ttft
        row["cold_end_to_end_completion_ms"] = (
            float(row["completion_latency_ms"]) + float(ingestion or 0.0)
            if row["reuse_state"] == "cold"
            else None
        )
        row["engine"] = payload["engine"]
        row["model_id"] = payload["model_id"]
        row["regime"] = (
            "cold_one_shot" if row["reuse_state"] == "cold" else "warm_repeated"
        )
        row["selection_id"] = f"{row['dataset']}:{row['example_id']}"
        row["request_ordinal"] = int(row.get("repeat", 0))
        row["query_sha256"] = "legacy-v1"
        row["total_latency_ms"] = row["completion_latency_ms"]
        row["selected_native_kv_tokens"] = row.get("selected_native_tokens", 0)
        row["active_detail_bytes"] = row.get("selected_kv_bytes", 0)
        row["retained_detail_bytes"] = row.get("selected_kv_bytes", 0)
        row["time_to_usable_context_ms"] = row.get("one_time_ingestion_ms")
        row["source_file"] = source.as_posix()
        normalized.append(row)
    return normalized


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(
            str(row["engine"]),
            str(row["dataset"]),
            str(row["regime"]),
            str(row["condition"]),
        )].append(row)
    result = []
    for (engine, dataset, regime, condition), group in sorted(groups.items()):
        ttft = _numbers(group, "ttft_ms")
        completion = _numbers(group, "total_latency_ms")
        result.append(
            {
                "engine": engine,
                "dataset": dataset,
                "regime": regime,
                "condition": condition,
                "examples": len(group),
                "exact_match": _mean(group, "exact_match"),
                "token_f1": _mean(group, "token_f1"),
                "gold_answer_logprob": _mean(group, "gold_answer_logprob"),
                "evidence_recall": _mean(group, "evidence_recall"),
                "candidate_tokens": _mean(group, "candidate_tokens"),
                "selected_source_tokens": _mean(group, "selected_source_tokens"),
                "visible_prompt_tokens": _mean(group, "visible_prompt_tokens"),
                "selected_native_kv_tokens": _mean(
                    group, "selected_native_kv_tokens"
                ),
                "active_detail_mib": (_mean(group, "active_detail_bytes") or 0) / 2**20,
                "retained_detail_mib": (
                    _mean(group, "retained_detail_bytes") or 0
                ) / 2**20,
                "text_preparation_ms": _mean(group, "text_preparation_ms"),
                "kv_encode_ms": _mean(group, "kv_encode_ms"),
                "index_construction_ms": _mean(group, "index_construction_ms"),
                "time_to_usable_context_ms": _mean(
                    group, "time_to_usable_context_ms"
                ),
                "cold_end_to_end_ttft_ms": _mean(
                    group, "cold_end_to_end_ttft_ms"
                ),
                "ttft_p50_ms": _percentile(ttft, 0.50),
                "ttft_p95_ms": _percentile(ttft, 0.95),
                "ttft_p99_ms": _percentile(ttft, 0.99),
                "itl_mean_ms": _mean(group, "itl_ms"),
                "tokens_per_second": _mean(group, "tokens_per_second"),
                "requests_per_second": _mean(group, "requests_per_second"),
                "cold_end_to_end_completion_ms": _mean(
                    group, "cold_end_to_end_completion_ms"
                ),
                "completion_p50_ms": _percentile(completion, 0.50),
                "completion_p95_ms": _percentile(completion, 0.95),
                "completion_p99_ms": _percentile(completion, 0.99),
                "ordinary_prefix_cache_hit_tokens": _mean(
                    group, "ordinary_prefix_cache_hit_tokens"
                ),
                "pra_hot_hit_rate": _mean(group, "pra_hot_hit"),
                "pra_warm_hit_rate": _mean(group, "pra_warm_hit"),
                "bytes_read": _mean(group, "bytes_read"),
                "bytes_promoted": _mean(group, "bytes_promoted"),
                "bytes_avoided": _mean(group, "bytes_avoided"),
                "duplicate_physical_kv_avoided_bytes": _mean(
                    group, "duplicate_physical_kv_avoided_bytes"
                ),
            }
        )
    return result


def _parity(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    pairs: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["engine"]),
            str(row["dataset"]),
            str(row["selection_id"]),
            str(row["regime"]),
            int(row["request_ordinal"]),
            str(row["query_sha256"]),
        )
        pairs[key][str(row["condition"])] = row
    grouped: dict[tuple[str, str, str], list[dict[str, dict[str, object]]]] = defaultdict(list)
    for (engine, dataset, _selection, regime, _ordinal, _query), pair in pairs.items():
        if set(pair) == set(CONDITIONS):
            grouped[(engine, dataset, regime)].append(pair)
    result = []
    for (engine, dataset, regime), complete in sorted(grouped.items()):
        exact_outputs = sum(
            pair[CONDITIONS[0]].get("output") == pair[CONDITIONS[1]].get("output")
            for pair in complete
        )
        result.append(
            {
                "engine": engine,
                "dataset": dataset,
                "regime": regime,
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


def _value(value: object, digits: int = 1) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def _write_tex(
    output_dir: Path,
    aggregates: list[dict[str, object]],
    parity: list[dict[str, object]],
) -> None:
    lookup = {
        (
            str(row["engine"]),
            str(row["dataset"]),
            str(row["regime"]),
            str(row["condition"]),
        ): row
        for row in aggregates
    }
    parity_lookup = {
        (str(row["engine"]), str(row["dataset"]), str(row["regime"])): row
        for row in parity
    }
    identities = sorted(parity_lookup)
    quality = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Engine & Dataset & Regime & Visible E0/E2 & Reduction & F1 E0/E2 & Exact parity \\",
        r"\midrule",
    ]
    latency = [
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Engine & Dataset & Regime & Completion E0/E2 & TTFT E0/E2 & E2 usable & E2 reuse MiB \\",
        r"\midrule",
    ]
    for engine, dataset, regime in identities:
        e0 = lookup[(engine, dataset, regime, CONDITIONS[0])]
        e2 = lookup[(engine, dataset, regime, CONDITIONS[1])]
        reduction = 100.0 * (
            float(e0["visible_prompt_tokens"])
            - float(e2["visible_prompt_tokens"])
        ) / float(e0["visible_prompt_tokens"])
        parity_value = 100.0 * float(
            parity_lookup[(engine, dataset, regime)]["exact_output_parity"]
        )
        quality.append(
            f"{engine} & {dataset} & {regime.replace('_', ' ')} & "
            f"{_value(e0['visible_prompt_tokens'], 0)}/{_value(e2['visible_prompt_tokens'], 0)} & "
            f"{reduction:.1f}\\% & "
            f"{_value(e0['token_f1'], 3)}/{_value(e2['token_f1'], 3)} & "
            f"{parity_value:.1f}\\% \\\\"
        )
        latency.append(
            f"{engine} & {dataset} & {regime.replace('_', ' ')} & "
            f"{_value(e0['completion_p50_ms'])}/{_value(e2['completion_p50_ms'])} & "
            f"{_value(e0['ttft_p50_ms'])}/{_value(e2['ttft_p50_ms'])} & "
            f"{_value(e2['time_to_usable_context_ms'])} & "
            f"{_value(float(e2['bytes_avoided'] or 0) / 2**20, 2)} \\\\"
        )
    quality.extend((r"\bottomrule", r"\end{tabular}"))
    latency.extend((r"\bottomrule", r"\end{tabular}"))
    (output_dir / "generated_matched_quality_table.tex").write_text(
        "\n".join(quality) + "\n", encoding="utf-8"
    )
    (output_dir / "generated_matched_latency_table.tex").write_text(
        "\n".join(latency) + "\n", encoding="utf-8"
    )


def _plot(path: Path, aggregates: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    cold = [row for row in aggregates if row["regime"] == "cold_one_shot"]
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
                "schema_version": "2.0",
                "experiment": "cross_engine_matched_e0_e2_summary_v2",
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
    _write_tex(args.output_dir, aggregates, parity)
    _plot(args.output_dir / "matched_e0_e2_summary.png", aggregates)


if __name__ == "__main__":
    main()
