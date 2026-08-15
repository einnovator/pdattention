"""Build paired findings, paper tables, and secondary Paper-3 figures."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper3_kv_materialization"
DISCOVERY = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation/gate3_discovery_selections.json"
DATASETS = ("musique", "2wikimultihopqa")


def _mean(values):
    return statistics.fmean(float(value) for value in values)


def _paired(rows, left: str, right: str):
    by = {(row["example_id"], row["condition"]): row for row in rows}
    identities = sorted({row["example_id"] for row in rows})
    pairs = [(by[identity, left], by[identity, right]) for identity in identities]
    return {
        "examples": len(pairs),
        "left": left,
        "right": right,
        "left_unique_kv_tokens": _mean(left_row["materialized_unique_tokens"] for left_row, _ in pairs),
        "right_unique_kv_tokens": _mean(right_row["materialized_unique_tokens"] for _, right_row in pairs),
        "mean_kv_reduction_fraction": _mean(
            1.0 - left_row["materialized_unique_tokens"] / max(right_row["materialized_unique_tokens"], 1)
            for left_row, right_row in pairs
        ),
        "mean_gold_logprob_delta": _mean(
            left_row["gold_mean_token_logprob"] - right_row["gold_mean_token_logprob"]
            for left_row, right_row in pairs
        ),
        "gold_logprob_wins_or_ties": sum(
            left_row["gold_mean_token_logprob"] >= right_row["gold_mean_token_logprob"]
            for left_row, right_row in pairs
        ),
        "mean_token_f1_delta": _mean(
            left_row["token_f1"] - right_row["token_f1"]
            for left_row, right_row in pairs
        ),
        "mean_accuracy_delta": _mean(
            left_row["normalized_answer_accuracy"] - right_row["normalized_answer_accuracy"]
            for left_row, right_row in pairs
        ),
        "mean_evidence_attention_delta": _mean(
            (left_row["evidence_attention_mass"] or 0.0)
            - (right_row["evidence_attention_mass"] or 0.0)
            for left_row, right_row in pairs
        ),
        "mean_ttft_delta_seconds": _mean(
            left_row["ttft_seconds"] - right_row["ttft_seconds"]
            for left_row, right_row in pairs
        ),
    }


def _factorial_pair(rows, selector: str, materialization: str):
    return {
        "selector": selector,
        "materialization": materialization,
        **_paired(
            rows,
            f"{selector}__{materialization}",
            f"{selector}__whole_parent",
        ),
        "conceptual_selected_parents": _mean(
            row["conceptual_selected_parents"]
            for row in rows
            if row["condition"] == f"{selector}__{materialization}"
        ),
        "evidence_recall": _mean(
            row["evidence_recall"]
            for row in rows
            if row["condition"] == f"{selector}__{materialization}"
        ),
    }


def _write_csv(path: Path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    value.rstrip().replace("\r", r"\r").replace("\n", r"\n")
                    if isinstance(value, str)
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def _merged_token_count(spans):
    ordered = sorted((int(start), int(end)) for start, end in spans if int(end) > int(start))
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _backfill_evidence_accounting(artifact, evidence_counts, json_path, csv_path):
    for row in artifact["rows"]:
        total = evidence_counts[(row["dataset"], row["example_id"])]
        row["evidence_source_tokens"] = total
        row["evidence_coverage"] = row["evidence_kv_tokens"] / max(total, 1)
        row["encoding_granularity_tokens"] = 256
        row["active_kv_fraction"] = row["materialized_unique_tokens"] / max(row["logical_source_tokens"], 1)
        row["cpu_reference_cache_bytes"] = row["logical_source_tokens"] * 28 * 4096
        row["gpu_reference_cache_bytes"] = 0
        row["h2d_kv_bytes"] = row["native_kv_bytes"]
    for aggregate in artifact["aggregates"]:
        values = [
            row for row in artifact["rows"]
            if row["dataset"] == aggregate["dataset"]
            and row["condition"] == aggregate["condition"]
        ]
        for key in (
            "evidence_source_tokens",
            "evidence_coverage",
            "requested_materialization_tokens",
            "deduplicated_materialization_tokens",
            "conceptual_selected_parents",
            "evidence_recall",
            "complete_evidence_recovery",
            "annotated_edge_recall",
            "active_kv_fraction",
            "cpu_reference_cache_bytes",
            "gpu_reference_cache_bytes",
            "h2d_kv_bytes",
            "encoding_granularity_tokens",
        ):
            samples = [row[key] for row in values if row.get(key) is not None]
            aggregate[key] = _mean(samples) if samples else None
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(
        csv_path,
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in artifact["rows"]
        ],
    )
    _write_csv(csv_path.with_name(csv_path.name.replace("_rows", "_aggregate")), artifact["aggregates"])


def _plots(validation, heldout, factorial):
    colors = {"musique": "#32688f", "2wikimultihopqa": "#c85c3d"}

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for dataset, color in colors.items():
        rows = [
            row for row in validation
            if row["dataset"] == dataset and row["condition"].startswith("M3_radius_")
        ]
        rows.sort(key=lambda row: int(row["condition"].rsplit("_", 1)[1]))
        tokens = [row["materialized_unique_tokens"] for row in rows]
        axes[0].plot(tokens, [row["evidence_attention_mass"] for row in rows], marker="o", color=color, label=f"{dataset}: evidence")
        axes[0].plot(tokens, [row["non_evidence_attention_mass"] for row in rows], marker="s", linestyle="--", color=color, alpha=0.75, label=f"{dataset}: non-evidence")
        axes[1].plot(tokens, [row["ttft_seconds"] for row in rows], marker="o", color=color, label=dataset)
    axes[0].set(xlabel="Materialized unique K/V tokens", ylabel="Mean answer-token attention mass")
    axes[1].set(xlabel="Materialized unique K/V tokens", ylabel="TTFT (seconds)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(RESULTS / f"materialization_dilution_latency.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    markers = {"whole_parent": "o", "local_atomic": "s", "budget_128": "^", "gist_local": "D"}
    for dataset, axis in zip(DATASETS, axes):
        for row in factorial:
            if row["dataset"] != dataset:
                continue
            selector, policy = row["condition"].split("__", 1)
            axis.scatter(
                row["conceptual_selected_parents"],
                row["materialized_unique_tokens"],
                marker=markers[policy],
                alpha=0.72,
                label=policy,
            )
        axis.set(title=dataset, xlabel="Conceptual selected parents", ylabel="Materialized unique K/V tokens")
        axis.grid(alpha=0.25)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(unique.values(), unique.keys(), loc="lower center", ncol=4)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(RESULTS / f"conceptual_breadth_physical_kv.{suffix}", dpi=180)
    plt.close(figure)


def main():
    validation_artifact = json.loads((RESULTS / "oracle_frontier_validation.json").read_text(encoding="utf-8"))
    heldout_artifact = json.loads((RESULTS / "oracle_frontier_heldout.json").read_text(encoding="utf-8"))
    factorial_artifact = json.loads((RESULTS / "selector_materialization_factorial.json").read_text(encoding="utf-8"))
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))["rows"]
    evidence_counts = {
        (row["dataset"], row["example_id"]): _merged_token_count(row["evidence_token_spans"])
        for row in discovery
    }
    _backfill_evidence_accounting(
        validation_artifact,
        evidence_counts,
        RESULTS / "oracle_frontier_validation.json",
        RESULTS / "oracle_frontier_validation_rows.csv",
    )
    _backfill_evidence_accounting(
        heldout_artifact,
        evidence_counts,
        RESULTS / "oracle_frontier_heldout.json",
        RESULTS / "oracle_frontier_heldout_rows.csv",
    )
    _backfill_evidence_accounting(
        factorial_artifact,
        evidence_counts,
        RESULTS / "selector_materialization_factorial.json",
        RESULTS / "selector_materialization_factorial_rows.csv",
    )
    validation = validation_artifact["rows"]
    heldout = heldout_artifact["rows"]
    factorial = factorial_artifact["rows"]
    oracle_pairs = []
    for dataset in DATASETS:
        rows = [row for row in heldout if row["dataset"] == dataset]
        for left in (
            "M0_native_gist",
            "M2_evidence_only",
            "M6_budget_128_equal",
            "M7_gist_selected_radius_0",
        ):
            oracle_pairs.append({"dataset": dataset, **_paired(rows, left, "M1_whole_parent")})
    factorial_pairs = []
    for dataset in DATASETS:
        rows = [row for row in factorial if row["dataset"] == dataset]
        for selector in ("one_shot", "graph_sparse", "graph_balanced", "graph_high"):
            for materialization in ("local_atomic", "budget_128", "gist_local"):
                factorial_pairs.append(
                    {"dataset": dataset, **_factorial_pair(rows, selector, materialization)}
                )
    summary = {
        "schema_version": "1.0",
        "validation_examples_per_dataset": 4,
        "heldout_examples_per_dataset": 8,
        "factorial_examples_per_dataset": 4,
        "selected_radius": json.loads((RESULTS / "oracle_policy_selection.json").read_text(encoding="utf-8")),
        "oracle_heldout_paired": oracle_pairs,
        "selector_materialization_paired": factorial_pairs,
        "scope_limits": {
            "validated_llm_judge": False,
            "direct_full_context": "omitted after 4 GB GPU OOM under eager dense attention",
            "serving_speed_claim": False,
            "statistical_scope": "descriptive pilot; no significance claim",
        },
    }
    (RESULTS / "paper3_findings.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(RESULTS / "oracle_heldout_paired.csv", oracle_pairs)
    _write_csv(RESULTS / "selector_materialization_paired.csv", factorial_pairs)
    _plots(validation_artifact["aggregates"], heldout, factorial)
    print(json.dumps({"oracle_pairs": len(oracle_pairs), "factorial_pairs": len(factorial_pairs)}, indent=2))


if __name__ == "__main__":
    main()
