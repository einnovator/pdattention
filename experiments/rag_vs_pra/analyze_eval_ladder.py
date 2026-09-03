"""Summarize matched-candidate artifacts and render publication plots."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Mapping


def _load(path: Path) -> Mapping[str, object]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def _reuse_curve(payload: Mapping[str, object], candidate_count: int, token_budget: int):
    selections = {
        row["selection_id"]: row for row in payload.get("selection_receipts", ())
    }
    rows = [
        row
        for row in payload["rows"]
        if row["candidate_count"] == candidate_count
        and row["token_budget"] == token_budget
        and row["condition"] in {"no_pra", "pra_no_adaptor"}
    ]
    examples = [row["example_id"] for row in payload["examples"]]
    by_key = {(row["example_id"], row["condition"]): row for row in rows}
    cumulative_visible = 0
    cumulative_native = 0
    seen_native_chunks: set[str] = set()
    curve = []
    for ordinal, example_id in enumerate(examples, 1):
        baseline = by_key[(example_id, "no_pra")]
        pra = by_key[(example_id, "pra_no_adaptor")]
        cumulative_visible += int(
            baseline["retrieval_context_metrics"]["physical_context_tokens"]
        )
        selection = selections[pra["selection_id"]]
        newly_materialized = 0
        for chunk in selection["selected_chunks"]:
            if chunk["chunk_id"] not in seen_native_chunks:
                seen_native_chunks.add(chunk["chunk_id"])
                newly_materialized += int(chunk["token_count"])
        cumulative_native += newly_materialized
        curve.append(
            {
                "query_count": ordinal,
                "standard_rag_cumulative_visible_tokens": cumulative_visible,
                "pra_cumulative_new_native_tokens": cumulative_native,
                "pra_unique_materialized_chunks": len(seen_native_chunks),
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-candidate-count", type=int, default=20)
    parser.add_argument("--reuse-token-budget", type=int, default=2048)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = {}
    csv_rows = []
    for path in args.inputs:
        payload = _load(path)
        key = f"{payload['dataset']}_{payload['stage']}"
        reuse = _reuse_curve(
            payload, args.reuse_candidate_count, args.reuse_token_budget
        )
        combined[key] = {
            "artifact": str(path).replace("\\", "/"),
            "evidence_tier": payload["evidence_tier"],
            "backend": payload["backend"],
            "summary": payload["summary"],
            "reuse_curve": reuse,
        }
        for row in payload["summary"]:
            csv_rows.append({"experiment": key, **row})

    summary_path = args.output_dir / "rag_eval_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    if csv_rows:
        with (args.output_dir / "rag_eval_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    import matplotlib.pyplot as plt

    colors = {"no_pra": "#1f5a94", "pra_no_adaptor": "#c4512d"}
    labels = {"no_pra": "Standard RAG", "pra_no_adaptor": "RAG + PRA"}
    natural_key = next((key for key in combined if key.startswith("multihoprag_fixed")), None)
    retrieval_key = next((key for key in combined if key.startswith("multihoprag_retrieval")), None)
    if natural_key:
        rows = combined[natural_key]["summary"]
        figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.5))
        for condition in colors:
            candidate_rows = sorted(
                (
                    row
                    for row in rows
                    if row["condition"] == condition and row["token_budget"] == 2048
                ),
                key=lambda row: row["candidate_count"],
            )
            axes[0].plot(
                [row["candidate_count"] for row in candidate_rows],
                [row["exact_match"] for row in candidate_rows],
                marker="o",
                color=colors[condition],
                label=labels[condition],
            )
            axes[1].plot(
                [row["candidate_count"] for row in candidate_rows],
                [row["supporting_document_coverage"] for row in candidate_rows],
                marker="o",
                color=colors[condition],
            )
            budget_rows = sorted(
                (
                    row
                    for row in rows
                    if row["condition"] == condition and row["candidate_count"] == 50
                ),
                key=lambda row: row["token_budget"],
            )
            axes[2].plot(
                [row["token_budget"] / 1024 for row in budget_rows],
                [row["exact_match"] for row in budget_rows],
                marker="o",
                color=colors[condition],
            )
        axes[0].set_title("Answer availability at 2K")
        axes[0].set_xlabel("Candidate documents")
        axes[0].set_ylabel("Answer present")
        axes[1].set_title("Evidence coverage at 2K")
        axes[1].set_xlabel("Candidate documents")
        axes[1].set_ylabel("Supporting-document coverage")
        axes[2].set_title("Budget curve at 50 candidates")
        axes[2].set_xlabel("Physical context budget (K tokens)")
        axes[2].set_ylabel("Answer present")
        for axis in axes:
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.2)
        axes[0].legend(frameon=False)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(args.output_dir / f"multihoprag_fixed_candidate_curves.{suffix}", dpi=180)
        plt.close(figure)

    if retrieval_key:
        curve = combined[retrieval_key]["reuse_curve"]
        figure, axis = plt.subplots(figsize=(6.4, 3.8))
        axis.plot(
            [row["query_count"] for row in curve],
            [row["standard_rag_cumulative_visible_tokens"] for row in curve],
            color=colors["no_pra"],
            label="Standard RAG visible text",
        )
        axis.plot(
            [row["query_count"] for row in curve],
            [row["pra_cumulative_new_native_tokens"] for row in curve],
            color=colors["pra_no_adaptor"],
            label="PRA newly materialized chunks",
        )
        axis.set_xlabel("Queries over persistent corpus")
        axis.set_ylabel("Cumulative tokens")
        axis.set_title("Multi-query context reuse (20 candidates, 2K budget)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(args.output_dir / f"multihoprag_reuse_curve.{suffix}", dpi=180)
        plt.close(figure)

    print(f"wrote summary tables and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
