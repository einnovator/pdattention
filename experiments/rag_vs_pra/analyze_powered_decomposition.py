"""Analyze powered RAG condition rows into auditable tables and plots."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from experiments.rag_vs_pra.datasets import load_multihop_rag
from pra_hf.rag_evaluation import (
    CandidateReceipt,
    ContextCondition,
    prepare_candidate_context,
)
from pra_hf.rag_powered import (
    paired_delta,
    qualification_gates,
    summarize_rows,
    write_csv,
    write_results,
)


def _load_rows(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _normalize_failure_classes(rows: Sequence[dict[str, object]]) -> None:
    """Apply the current stage taxonomy to rows from resumable older runners."""

    for row in rows:
        if row.get("status") != "MEASURED":
            continue
        gold = set(map(str, row.get("gold_document_ids", [])))
        candidates = set(map(str, row.get("candidate_document_ids", [])))
        selected = set(map(str, row.get("selected_document_ids", [])))
        condition = str(row["condition"])
        if not gold.issubset(candidates):
            row["failure_class"] = "FIRST_STAGE_RETRIEVAL_MISS"
        elif condition == ContextCondition.NO_PRA_STANDARD_RAG.value and not gold.issubset(selected):
            row["failure_class"] = "STANDARD_RAG_PACKING_MISS"
        elif condition != ContextCondition.NO_PRA_STANDARD_RAG.value and not gold.intersection(selected):
            row["failure_class"] = "PRA_SELECTOR_MISS"
        elif condition != ContextCondition.NO_PRA_STANDARD_RAG.value and not gold.issubset(selected):
            row["failure_class"] = "PRA_DISTRACTOR_SELECTION"


def _enrich_gold_chunk_recall(
    root: Path,
    rows: Sequence[dict[str, object]],
    *,
    dataset: str,
    cache_dir: Path,
) -> None:
    """Reconstruct exact gold chunks for resumable pre-schema result rows."""

    if dataset != "multihoprag":
        return
    documents, questions, _ = load_multihop_rag(cache_dir)
    documents_by_id = {document.document_id: document for document in documents}
    questions_by_id = {question.example_id: question for question in questions}
    cache: dict[str, set[str]] = {}
    for row in rows:
        if row.get("status") != "MEASURED":
            continue
        receipt_id = str(row["candidate_receipt_id"])
        relevant = cache.get(receipt_id)
        if relevant is None:
            receipt = CandidateReceipt.from_dict(
                json.loads(
                    (root / "candidate_receipts" / f"{receipt_id}.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
            question = questions_by_id[str(row["example_id"])]
            candidates = prepare_candidate_context(receipt, documents_by_id).chunks
            relevant = {
                chunk.chunk_id
                for chunk in candidates
                if (
                    any(
                        chunk.document_id == document_id
                        and chunk.start < end
                        and start < chunk.end
                        for start, end in spans
                    )
                    for document_id, spans in question.gold_spans.items()
                )
                or (
                    chunk.document_id in question.gold_document_ids
                    and not question.gold_spans.get(chunk.document_id)
                )
            }
            cache[receipt_id] = relevant
        selected = set(map(str, row.get("selected_chunk_ids", [])))
        metrics = row.get("retrieval_context_metrics")
        if isinstance(metrics, dict):
            metrics["gold_chunk_recall"] = (
                len(relevant.intersection(selected)) / len(relevant) if relevant else 0.0
            )


def _primary_rows(
    summaries: Sequence[Mapping[str, object]], candidate_count: int, token_budget: int
) -> list[Mapping[str, object]]:
    return [
        row
        for row in summaries
        if int(row["candidate_count"]) == candidate_count
        and int(row["token_budget"]) == token_budget
        and row.get("status") == "MEASURED"
    ]


def _failure_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], Counter[str]] = {}
    for row in rows:
        if row.get("status") != "MEASURED":
            continue
        key = (
            row["condition"],
            row["selector_profile"],
            row["candidate_count"],
            row["token_budget"],
            row["regime"],
        )
        groups.setdefault(key, Counter())[str(row.get("failure_class", "UNKNOWN"))] += 1
    result = []
    for key, counts in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        for failure, count in sorted(counts.items()):
            result.append(
                {
                    "condition": key[0],
                    "selector_profile": key[1],
                    "candidate_count": key[2],
                    "token_budget": key[3],
                    "regime": key[4],
                    "failure_class": failure,
                    "examples": count,
                }
            )
    return result


def _deltas(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    comparisons = (
        (
            "native_vs_selected_generic",
            ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value,
            ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value,
            "pra_generic",
        ),
        (
            "native_vs_selected_strong",
            ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value,
            ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value,
            "pra_strong_reranker",
        ),
    )
    result = []
    for name, left, right, selector in comparisons:
        for regime in ("COLD", "WARM"):
            for metric in (
                "exact_match",
                "token_f1",
                "official_multihop_rag_score",
                "visible_prompt_tokens",
                "ttft_ms",
                "total_latency_ms",
                "tokens_per_second",
                "peak_memory_bytes",
            ):
                value = paired_delta(
                    rows,
                    left_condition=left,
                    right_condition=right,
                    metric=metric,
                    selector_profile=selector,
                    regime=regime,
                )
                value["comparison"] = name
                result.append(value)
    return result


def _persistent_curve(
    rows: Sequence[Mapping[str, object]], candidate_count: int, token_budget: int
) -> list[dict[str, object]]:
    regime = (
        "PERSISTENT_CORPUS"
        if any(row.get("regime") == "PERSISTENT_CORPUS" for row in rows)
        else "COLD"
    )
    selected = [
        row
        for row in rows
        if row.get("status") == "MEASURED"
        and row["candidate_count"] == candidate_count
        and row["token_budget"] == token_budget
        and row["regime"] == regime
        and (
            (
                row["condition"] == ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value
                and row["selector_profile"] == "pra_generic"
            )
            or (
                row["condition"] == ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value
                and row["selector_profile"] == "pra_generic"
            )
        )
    ]
    selected.sort(key=lambda row: (str(row["example_id"]), str(row["condition"])))
    cumulative_visible = 0
    cumulative_unique_native = 0
    cumulative_wall = {
        ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value: 0.0,
        ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value: 0.0,
    }
    result = []
    for row in selected:
        condition = str(row["condition"])
        if condition == ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value:
            cumulative_visible += int(
                row.get("retrieval_context_metrics", {}).get("physical_context_tokens", 0)
            )
        else:
            serving = row.get("serving_metrics", {})
            cumulative_unique_native += int(
                serving.get("newly_materialized_tokens") or 0
            )
        serving = row.get("serving_metrics", {})
        # total_latency_ms already includes ingestion in the powered schema.
        cumulative_wall[condition] += float(serving.get("total_latency_ms") or 0.0)
        result.append(
            {
                "example_id": row["example_id"],
                "condition": condition,
                "cumulative_visible_tokens": cumulative_visible,
                "cumulative_unique_native_tokens": cumulative_unique_native,
                "cumulative_wall_ms": cumulative_wall[condition],
            }
        )
    return result


def _plots(
    summaries: Sequence[Mapping[str, object]],
    curve: Sequence[Mapping[str, object]],
    output: Path,
    candidate_count: int,
    token_budget: int,
) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    primary = [row for row in _primary_rows(summaries, candidate_count, token_budget) if row["regime"] == "COLD"]
    labels = [f"{row['selector_profile']}\n{str(row['condition']).replace('_NO_ADAPTOR', '')}" for row in primary]
    specs = (
        ("generated_quality", "token_f1", "Token F1", "data"),
        ("visible_tokens", "visible_prompt_tokens", "Visible prompt tokens", None),
        ("ttft_p95", "ttft_p95_ms", "TTFT p95 (ms)", None),
        ("completion_latency", "completion_latency_ms", "Completion latency (ms)", None),
        ("support_coverage", "supporting_document_coverage", "Supporting-document coverage", (0.0, 1.0)),
    )
    for filename, metric, ylabel, limits in specs:
        values = [float(row.get(metric) or 0.0) for row in primary]
        fig, axis = plt.subplots(figsize=(12, 5.5))
        axis.bar(range(len(values)), values, color="#1f6f78")
        axis.set_xticks(range(len(values)), labels, rotation=25, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if limits == "data":
            upper = max(values, default=0.0)
            axis.set_ylim(0.0, max(0.01, upper * 1.2))
        elif limits:
            axis.set_ylim(*limits)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)

    if curve:
        fig, axis = plt.subplots(figsize=(9, 5.5))
        visible = [row for row in curve if row["condition"] == ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value]
        native = [row for row in curve if row["condition"] == ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value]
        axis.plot(
            range(1, len(visible) + 1),
            [row["cumulative_visible_tokens"] for row in visible],
            label="PRA Selected Context repeated selected tokens",
        )
        axis.plot(
            range(1, len(native) + 1),
            [row["cumulative_unique_native_tokens"] for row in native],
            label="PRA Native Memory newly materialized chunk tokens",
        )
        axis.set_xlabel("Questions")
        axis.set_ylabel("Cumulative source tokens")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"cumulative_repeated_query_tokens.{suffix}", dpi=180)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(9, 5.5))
        for condition, label in (
            (ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR.value, "PRA Selected Context"),
            (ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR.value, "PRA Native Memory"),
        ):
            values = [row for row in curve if row["condition"] == condition]
            axis.plot(range(1, len(values) + 1), [row["cumulative_wall_ms"] for row in values], label=label)
        axis.set_xlabel("Questions")
        axis.set_ylabel("Cumulative wall time (ms)")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"cumulative_repeated_query_wall_time.{suffix}", dpi=180)
        plt.close(fig)


def _paper_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Condition & Selector & F1 & Official & Visible & Native & TTFT p95 & Latency \\",
        r"\midrule",
    ]
    for row in rows:
        condition = str(row["condition"]).replace("_", r"\_")
        selector = str(row["selector_profile"]).replace("_", r"\_")
        def fmt(name: str, digits: int = 3) -> str:
            value = row.get(name)
            return "--" if value is None else f"{float(value):.{digits}f}"
        lines.append(
            f"{condition} & {selector} & {fmt('token_f1')} & "
            f"{fmt('official_multihop_rag_score')} & {fmt('visible_prompt_tokens', 0)} & "
            f"{fmt('selected_native_kv_tokens', 0)} & {fmt('ttft_p95_ms', 1)} & "
            f"{fmt('total_latency_ms', 1)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _paper_quality_table(rows: Sequence[Mapping[str, object]]) -> str:
    measured = [
        row for row in rows if row.get("status") == "MEASURED" and row["regime"] == "COLD"
    ]
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Condition & Selector & Official & F1 & Answer avail. & Support & False docs & Tokens \\",
        r"\midrule",
    ]
    for row in measured:
        condition = str(row["condition"]).replace("_", r"\_")
        selector = str(row["selector_profile"]).replace("_", r"\_")

        def fmt(name: str, digits: int = 3) -> str:
            value = row.get(name)
            return "--" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"{condition} & {selector} & {fmt('official_multihop_rag_score')} & "
            f"{fmt('token_f1')} & {fmt('answer_string_availability')} & "
            f"{fmt('supporting_document_coverage')} & "
            f"{fmt('false_selected_document_fraction')} & "
            f"{fmt('physical_context_tokens', 0)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def _paper_runtime_table(rows: Sequence[Mapping[str, object]]) -> str:
    measured = [
        row
        for row in rows
        if row.get("status") == "MEASURED"
        and row["selector_profile"] == "pra_generic"
    ]
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Condition & Regime & TTFT p95 & Total & Ingest & Visible & Native & Reuse \\",
        r"\midrule",
    ]
    for row in measured:
        condition = str(row["condition"]).replace("_", r"\_")

        def fmt(name: str, digits: int = 1) -> str:
            value = row.get(name)
            return "--" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            f"{condition} & {row['regime']} & {fmt('ttft_p95_ms')} & "
            f"{fmt('total_latency_ms')} & {fmt('ingestion_ms')} & "
            f"{fmt('visible_prompt_tokens', 0)} & "
            f"{fmt('selected_native_kv_tokens', 0)} & {fmt('native_reuse', 3)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--primary-candidate-count", type=int, default=20)
    parser.add_argument("--primary-token-budget", type=int, default=2048)
    parser.add_argument("--minimum-examples", type=int, default=50)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    args = parser.parse_args()

    root = args.input_dir
    rows = _load_rows(root / "condition_results.jsonl.gz")
    manifest = json.loads((root / "cohort_manifest.json").read_text(encoding="utf-8"))
    _enrich_gold_chunk_recall(
        root,
        rows,
        dataset=str(manifest["dataset"]),
        cache_dir=args.cache_dir,
    )
    _normalize_failure_classes(rows)
    write_results(root / "condition_results.jsonl.gz", rows)
    summaries = summarize_rows(rows)
    failures = _failure_rows(rows)
    deltas = _deltas(rows)
    gates = qualification_gates(summaries, minimum_examples=args.minimum_examples)
    curve = _persistent_curve(rows, args.primary_candidate_count, args.primary_token_budget)

    write_csv(root / "summary.csv", summaries)
    write_csv(root / "failure_summary.csv", failures)
    write_csv(root / "condition_deltas.csv", deltas)
    write_csv(root / "persistent_corpus_curve.csv", curve)
    primary = _primary_rows(summaries, args.primary_candidate_count, args.primary_token_budget)
    (root / "paper_table.tex").write_text(_paper_table(primary), encoding="utf-8")
    (root / "paper_quality_table.tex").write_text(
        _paper_quality_table(primary), encoding="utf-8"
    )
    (root / "paper_runtime_table.tex").write_text(
        _paper_runtime_table(primary), encoding="utf-8"
    )
    canonical = {
        "schema_version": "pra-rag-powered-evidence-v1",
        "key": {
            "task_dataset": "MultiHop-RAG",
            "hardware_engine": f"{manifest['hardware']['machine']}/{manifest['engine']}",
            "model": manifest["model"],
            "model_revision": manifest["model_revision"],
            "precision": manifest["precision"],
            "profile": "RAG_POWERED_DECOMPOSITION",
        },
        "conditions": summaries,
        "deltas": deltas,
        "qualification_gates": gates,
        "provenance": {
            "run_id": manifest["run_id"],
            "git_commit": manifest["git_commit"],
            "cohort_manifest": "cohort_manifest.json",
            "condition_rows": "condition_results.jsonl.gz",
        },
    }
    (root / "canonical_evidence.json").write_text(
        json.dumps(canonical, indent=2, sort_keys=True), encoding="utf-8"
    )
    card_status = gates["card_gate"]
    (root / "hf_card_fragment.md").write_text(
        "## MultiHop-RAG powered qualification\n\n"
        f"**RAG QUALIFICATION: {card_status}**\n\n"
        "This row decomposes Standard RAG, PRA Selected Context, and selector-frozen "
        "PRA Native Memory. Bundle arms remain `NO_QUALIFIED_ADAPTER` unless an exact "
        "model/precision adapter passes the held-out gate.\n",
        encoding="utf-8",
    )
    _plots(
        summaries,
        curve,
        root / "plots",
        args.primary_candidate_count,
        args.primary_token_budget,
    )
    print(json.dumps({"rows": len(rows), "summaries": len(summaries), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
