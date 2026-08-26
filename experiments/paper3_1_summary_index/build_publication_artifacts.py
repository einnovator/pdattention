"""Freeze preselected Paper 3.1 rows into publication tables and figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper3_1_summary_index"
OUTPUT = RESULTS / "publication"

POLICIES = {
    "hotpotqa": ("test_teacher_hotpot", "teacher_8b_generic_1x32_summary_exact"),
    "qasper": ("test_subb", "subb_600m_retrieval_1x32_summary_exact"),
    "2wikimultihopqa": ("test_teacher_2wiki", "teacher_8b_retrieval_1x32_summary_bm25"),
    "musique": ("test_subb", "subb_600m_retrieval_1x32_summary_hybrid_a0.50"),
}
BASELINES = ("native_mean", "source_bm25", "rank16", "rank8_centroid8", "oracle_identity")
LABELS = {
    "native_mean": "Native mean",
    "source_bm25": "Source BM25",
    "rank16": "Rank-16 QK",
    "rank8_centroid8": "Rank-8 / 8 centroids",
    "oracle_identity": "Oracle identity",
    "summary": "Generated summary",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_headline() -> list[dict]:
    rows = []
    effects = []
    for dataset, (run_name, summary_condition) in POLICIES.items():
        run = RESULTS / run_name
        summary = _read_csv(run / "summary.csv")
        lookup = {(row["dataset"], row["condition"]): row for row in summary}
        for condition in (*BASELINES, summary_condition):
            source = lookup[(dataset, condition)]
            rows.append(
                {
                    "dataset": dataset,
                    "condition": "summary" if condition == summary_condition else condition,
                    "frozen_condition": condition,
                    "examples": int(source["examples"]),
                    "evidence_recall": float(source["evidence_recall"]),
                    "complete_recovery": float(source["complete_recovery"]),
                    "precision": float(source["precision"]),
                    "mrr": float(source["reciprocal_rank"]),
                    "routing_index_bytes_per_source": float(source["routing_index_bytes"]),
                    "embedding_index_bytes_per_source": float(source["embedding_index_bytes"]),
                    "summary_tokens_per_source": float(source["summary_tokens"]),
                    "ingestion_seconds_per_source": float(source["ingestion_seconds"]),
                    "routing_seconds_per_query": float(source["routing_seconds"]),
                    "native_kv_tokens_materialized": float(source["native_kv_tokens_materialized"]),
                }
            )
        paired = _read_csv(run / "paired_effects.csv")
        effect = next(
            row
            for row in paired
            if row["dataset"] == dataset and row["condition"] == summary_condition
        )
        effects.append(effect)
    _write_csv(OUTPUT / "headline.csv", rows)
    _write_csv(OUTPUT / "headline_paired_effects.csv", effects)
    return rows


def build_amortized_cost(rows: list[dict]) -> None:
    output = []
    lookup = {(row["dataset"], row["condition"]): row for row in rows}
    for row in rows:
        if row["condition"] != "summary":
            continue
        native = lookup[(row["dataset"], "native_mean")]
        lexical = lookup[(row["dataset"], "source_bm25")]
        native_saving = native["routing_seconds_per_query"] - row["routing_seconds_per_query"]
        lexical_saving = lexical["routing_seconds_per_query"] - row["routing_seconds_per_query"]
        for query_count in (1, 10, 100, 1000):
            output.append(
                {
                    "dataset": row["dataset"],
                    "queries_per_source": query_count,
                    "amortized_ingestion_seconds_per_query": row["ingestion_seconds_per_source"] / query_count,
                    "routing_seconds_per_query": row["routing_seconds_per_query"],
                    "routing_index_bytes_per_source": row["routing_index_bytes_per_source"],
                    "native_kv_tokens_materialized": row["native_kv_tokens_materialized"],
                    "routing_break_even_queries_vs_native_mean": (
                        row["ingestion_seconds_per_source"] / native_saving
                        if native_saving > 0
                        else "never"
                    ),
                    "routing_break_even_queries_vs_source_bm25": (
                        row["ingestion_seconds_per_source"] / lexical_saving
                        if lexical_saving > 0
                        else "never"
                    ),
                }
            )
    _write_csv(OUTPUT / "amortized_cost.csv", output)


def plot_headline(rows: list[dict]) -> None:
    datasets = tuple(POLICIES)
    conditions = ("native_mean", "source_bm25", "rank16", "rank8_centroid8", "summary", "oracle_identity")
    lookup = {(row["dataset"], row["condition"]): row for row in rows}
    width = 0.13
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    colors = ("#355070", "#6D597A", "#B56576", "#E56B6F", "#2A9D8F", "#8D99AE")
    centers = list(range(len(datasets)))
    for offset, (condition, color) in enumerate(zip(conditions, colors)):
        x = [center + (offset - 2.5) * width for center in centers]
        values = [lookup[(dataset, condition)]["evidence_recall"] for dataset in datasets]
        axis.bar(x, values, width=width, label=LABELS[condition], color=color)
    axis.set_xticks(centers, ["HotpotQA", "QASPER", "2Wiki", "MuSiQue"])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Evidence recall @ 4 parent chunks")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    figure.tight_layout()
    figure.savefig(OUTPUT / "headline_recall.png", dpi=200)
    figure.savefig(OUTPUT / "headline_recall.pdf")
    plt.close(figure)


def plot_omission() -> None:
    path = RESULTS / "omission" / "summary.csv"
    if not path.exists():
        return
    rows = _read_csv(path)
    profile = next(row["profile"] for row in rows)
    conditions = (
        "source_bm25",
        "salient_sentence_bm25",
        "entity_rare_bm25",
        f"{profile}_retrieval_summary_bm25",
        f"{profile}_retrieval_summary_bm25_shuffled",
    )
    fact_types = ("entity", "alias", "relation", "date_number", "rare_string")
    lookup = {(row["fact_type"], row["condition"]): row for row in rows}
    width = 0.16
    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    for offset, condition in enumerate(conditions):
        values = [float(lookup[(fact, condition)]["evidence_recall"]) for fact in fact_types]
        x = [index + (offset - 2) * width for index in range(len(fact_types))]
        axis.bar(x, values, width=width, label=condition.replace("_", " "))
    axis.set_xticks(range(len(fact_types)), [value.replace("_", " ") for value in fact_types])
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Evidence recall @ 1 chunk")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "omission_recall.png", dpi=200)
    figure.savefig(OUTPUT / "omission_recall.pdf")
    plt.close(figure)


def plot_geometry() -> None:
    path = RESULTS / "geometry_validation" / "summary.csv"
    if not path.exists():
        return
    rows = [
        row
        for row in _read_csv(path)
        if row["condition"].endswith("summary_bm25") and "shuffled" not in row["condition"]
    ]
    datasets = sorted({row["dataset"] for row in rows})
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    for dataset in datasets:
        local = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: int(row["condition"].split("_")[-3].split("x")[0]),
        )
        facets = [int(row["condition"].split("_")[-3].split("x")[0]) for row in local]
        axis.plot(facets, [float(row["evidence_recall"]) for row in local], marker="o", label=dataset)
    axis.set_xscale("log", base=2)
    axis.set_xticks((1, 2, 4, 8), ("1x32", "2x16", "4x8", "8x4"))
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Evidence recall @ 4")
    axis.set_xlabel("Equal 32-token summary geometry")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "geometry_recall.png", dpi=200)
    figure.savefig(OUTPUT / "geometry_recall.pdf")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = build_headline()
    build_amortized_cost(rows)
    plot_headline(rows)
    plot_omission()
    plot_geometry()


if __name__ == "__main__":
    main()
