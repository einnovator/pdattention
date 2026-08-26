"""Run the fixed-budget Paper 3.1 multi-index complementarity study.

The experiment replays frozen summary addresses and inherited Paper 2.8 native
Q/K selectors.  Lexical, typed extractive, summary, and learned-QK channels all
rank the same source-parent identities.  Every selected identity still resolves
to the original source span; no address view becomes answer evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import file_sha256
from experiments.paper2_8_qk_compression.run_gated_study import _project_native_queries
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper3_1_summary_index.ollama_sidecar import OllamaClient
from experiments.paper3_1_summary_index.run_study import (
    DEFAULT_DATA_ROOT,
    DEFAULT_FEATURE_ROOT,
    EMBEDDING_MODEL,
    SEED,
    _embedding_scores,
    _load_lowrank,
    _lowrank_parent_scores,
    _parent_mean_scores,
    _source_index,
    load_cases,
)
from pra_hf.multi_index import (
    agreement_priority_union,
    candidate_provenance,
    extract_typed_sidecars,
    normalized_score_fusion,
    rank_round_robin,
    reciprocal_rank_fusion_scores,
    reserved_slot_union,
)
from pra_hf.summary_index import (
    BM25SummaryScorer,
    SummaryIndex,
    SummaryIndexRecord,
    exact_summary_scores,
    lexical_terms,
    retrieval_metrics,
    source_sha256,
    stable_topk,
)


OUTPUT_ROOT = ROOT / "docs/papers/shared/results/paper3_1_summary_index/multi_index"
SOURCE_RESULTS = ROOT / "docs/papers/shared/results/paper3_1_summary_index"
CHANNELS = ("L", "E", "S", "QK")
K_VALUES = (2, 4, 8)
PRIMARY_K = 4
RRF_GRID = (10.0, 60.0)
FUSION_TEMPLATES = ("equal", "address_safe", "semantic", "summary", "qk")
DATASETS = ("hotpotqa", "qasper", "2wikimultihopqa", "musique")
SUMMARY_POLICY = {
    "hotpotqa": {
        "model": "llama3.1:8b",
        "prompt_id": "generic",
        "scorer": "exact",
        "source_run": "test_teacher_hotpot",
    },
    "qasper": {
        "model": "qwen3:0.6b",
        "prompt_id": "retrieval",
        "scorer": "exact",
        "source_run": "test_subb",
    },
    "2wikimultihopqa": {
        "model": "llama3.1:8b",
        "prompt_id": "retrieval",
        "scorer": "bm25",
        "source_run": "test_teacher_2wiki",
    },
    "musique": {
        "model": "qwen3:0.6b",
        "prompt_id": "retrieval",
        "scorer": "hybrid_a0.50",
        "source_run": "test_subb",
    },
}
COMBINATIONS = (
    ("L", "S"),
    ("L", "E"),
    ("L", "QK"),
    ("E", "S"),
    ("E", "QK"),
    ("S", "QK"),
    ("L", "E", "S"),
    ("L", "E", "QK"),
    ("L", "S", "QK"),
    ("E", "S", "QK"),
    ("L", "E", "S", "QK"),
)
RESERVED_POLICIES = {
    "L+S:L2_S2": ({"L": 2, "S": 2}, ("L", "S")),
    "E+S:E2_S2": ({"E": 2, "S": 2}, ("E", "S")),
    "L+QK:L2_QK2": ({"L": 2, "QK": 2}, ("L", "QK")),
    "E+QK:E2_QK2": ({"E": 2, "QK": 2}, ("E", "QK")),
    "L+E+S:L1_E1_S2": ({"L": 1, "E": 1, "S": 2}, ("L", "E", "S")),
    "L+E+QK:L1_E1_QK2": ({"L": 1, "E": 1, "QK": 2}, ("L", "E", "QK")),
    "L+S+QK:L1_S1_QK2": ({"L": 1, "S": 1, "QK": 2}, ("L", "S", "QK")),
    "E+S+QK:E1_S1_QK2": ({"E": 1, "S": 1, "QK": 2}, ("E", "S", "QK")),
    "L+E+S+QK:L1_E1_S1_QK1": (
        {"L": 1, "E": 1, "S": 1, "QK": 1},
        ("L", "E", "S", "QK"),
    ),
    "L+S+QK:L1_S1_QK2_lex1": ({"L": 1, "S": 1, "QK": 2}, ("L", "S", "QK")),
    "L+E+S+QK:L1_E1_QK2_lex2": (
        {"L": 1, "E": 1, "QK": 2},
        ("L", "E", "S", "QK"),
    ),
}


@dataclass
class CaseChannels:
    """Aligned channel scores and accounting for one immutable source."""

    case: object
    scores: dict[str, np.ndarray]
    costs: dict[str, dict[str, float | int | str]]
    summary_index: SummaryIndex
    typed_index: SummaryIndex
    typed_sidecars: tuple


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to create empty required artifact: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _summary_rows(split: str) -> dict[tuple[str, str], list[dict]]:
    runs = ("expanded_validation",) if split == "validation" else tuple(
        sorted({str(policy["source_run"]) for policy in SUMMARY_POLICY.values()})
    )
    output: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        path = SOURCE_RESULTS / run / "summary_addresses.jsonl"
        for row in _read_jsonl(path):
            dataset = str(row["dataset"])
            policy = SUMMARY_POLICY[dataset]
            if row["generation_model"] != policy["model"] or row["prompt_id"] != policy["prompt_id"]:
                continue
            output.setdefault((dataset, str(row["example_id"])), []).append(row)
    return output


def _load_aligned_cases(args, tokenizer, split: str) -> list:
    summaries = _summary_rows(split)
    case_args = SimpleNamespace(
        split=split,
        datasets=DATASETS,
        max_per_dataset=8,
        cache_dir=args.cache_dir,
        dataset_seed=args.dataset_seed,
        feature_root=args.feature_root,
        annotations=args.annotations,
        twowiki_dev=args.twowiki_dev,
        musique_dev=args.musique_dev,
    )
    cases, _ = load_cases(case_args, tokenizer)
    aligned = [case for case in cases if (case.dataset, case.example_id) in summaries]
    missing = sorted(set(summaries) - {(case.dataset, case.example_id) for case in aligned})
    if missing:
        raise ValueError(f"Summary/source parity failed for {len(missing)} identities: {missing[:3]}")
    return aligned


def _summary_index(case, rows: Sequence[dict]) -> SummaryIndex:
    ordered = sorted(rows, key=lambda row: int(str(row["chunk_id"]).split("-")[-1]))
    index = SummaryIndex(SummaryIndexRecord.from_dict(row) for row in ordered)
    index.assert_source_alignment(
        (
            case.uri,
            f"parent-{chunk_index}",
            int(span[0]),
            int(span[1]),
            source_sha256(text),
        )
        for chunk_index, (span, text) in enumerate(zip(case.feature["parent_spans"], case.chunk_texts))
    )
    return index


def _typed_index(case, sidecars: Sequence) -> SummaryIndex:
    return SummaryIndex(
        SummaryIndexRecord(
            uri=case.uri,
            chunk_id=f"parent-{index}",
            token_start=int(span[0]),
            token_end=int(span[1]),
            source_sha256=source_sha256(source),
            summary=sidecar.text,
            summary_token_count=len(lexical_terms(sidecar.text)),
            generation_model="deterministic-extractive-v1",
            prompt_id="typed-entity-rare",
        )
        for index, (source, span, sidecar) in enumerate(
            zip(case.chunk_texts, case.feature["parent_spans"], sidecars)
        )
    )


def _finite(values: Sequence[float]) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64)
    finite = output[np.isfinite(output)]
    floor = float(finite.min() - 1.0) if len(finite) else -1.0
    ceiling = float(finite.max() + 1.0) if len(finite) else 1.0
    return np.nan_to_num(output, nan=floor, neginf=floor, posinf=ceiling)


def _timed_bm25(index: SummaryIndex, query: str) -> tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    scorer = BM25SummaryScorer(index)
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    scores = scorer.score(query)
    return scores, build_seconds, time.perf_counter() - started


def _case_channels(
    case,
    summary_rows: Mapping[tuple[str, str], Sequence[dict]],
    lowrank_models,
    client: OllamaClient,
    device: torch.device,
) -> CaseChannels:
    source_index = _source_index(case)
    summary_index = _summary_index(case, summary_rows[(case.dataset, case.example_id)])

    started = time.perf_counter()
    sidecars = extract_typed_sidecars(case.chunk_texts)
    extract_seconds = time.perf_counter() - started
    typed_index = _typed_index(case, sidecars)
    typed_index.assert_source_alignment(
        (
            case.uri,
            f"parent-{chunk_index}",
            int(span[0]),
            int(span[1]),
            source_sha256(text),
        )
        for chunk_index, (span, text) in enumerate(zip(case.feature["parent_spans"], case.chunk_texts))
    )

    lexical, lexical_build, lexical_route = _timed_bm25(source_index, case.question)
    extractive, extractive_build, extractive_route = _timed_bm25(typed_index, case.question)

    policy = SUMMARY_POLICY[case.dataset]
    summary_embedding_bytes = 0
    summary_embedding_seconds = 0.0
    started = time.perf_counter()
    if policy["scorer"] == "exact":
        summary = exact_summary_scores(summary_index, case.question)
    elif policy["scorer"] == "bm25":
        summary = BM25SummaryScorer(summary_index).score(case.question)
    else:
        bm25 = BM25SummaryScorer(summary_index).score(case.question)
        embedding_started = time.perf_counter()
        embedding, width = _embedding_scores(client, summary_index, case.question)
        summary_embedding_seconds = time.perf_counter() - embedding_started
        summary_embedding_bytes = len(summary_index.records) * width * 4
        from pra_hf.summary_index import hybrid_scores

        summary = hybrid_scores(bm25, embedding, 0.5)
    summary_route = time.perf_counter() - started

    started = time.perf_counter()
    qk = _lowrank_parent_scores(
        case, lowrank_models[(case.dataset, 16)], centroids=None, device=device
    )
    qk_route = time.perf_counter() - started
    started = time.perf_counter()
    qk8 = _lowrank_parent_scores(
        case, lowrank_models[(case.dataset, 8)], centroids=8, device=device
    )
    qk8_route = time.perf_counter() - started
    started = time.perf_counter()
    native = _parent_mean_scores(case)
    native_route = time.perf_counter() - started

    generation_seconds = sum(
        float(row.get("generation_seconds", 0.0))
        for row in summary_rows[(case.dataset, case.example_id)]
    )
    parent_count = len(case.chunk_texts)
    costs = {
        "L": {
            "persistent_bytes": source_index.text_bytes,
            "ingestion_seconds": lexical_build,
            "routing_seconds": lexical_route,
            "ingestion_status": "measured_source_bm25_build",
        },
        "E": {
            "persistent_bytes": typed_index.text_bytes,
            "ingestion_seconds": extract_seconds + extractive_build,
            "routing_seconds": extractive_route,
            "ingestion_status": "measured_extract_and_bm25_build",
        },
        "S": {
            "persistent_bytes": summary_index.text_bytes + summary_embedding_bytes,
            "text_bytes": summary_index.text_bytes,
            "embedding_bytes": summary_embedding_bytes,
            "ingestion_seconds": generation_seconds + summary_embedding_seconds,
            "generation_seconds": generation_seconds,
            "embedding_seconds": summary_embedding_seconds,
            "routing_seconds": summary_route,
            "ingestion_status": "cached_generation_plus_measured_embedding",
        },
        "QK": {
            "persistent_bytes": parent_count * 8 * 32 * 16 * 4,
            "ingestion_seconds": 0.0,
            "routing_seconds": qk_route,
            "ingestion_status": "inherited_checkpoint_projection_not_remeasured",
        },
        "QK8": {
            "persistent_bytes": parent_count * 8 * 8 * 8 * 4,
            "ingestion_seconds": 0.0,
            "routing_seconds": qk8_route,
            "ingestion_status": "inherited_checkpoint_projection_not_remeasured",
        },
        "native_mean": {
            "persistent_bytes": parent_count * 1024 * 4,
            "ingestion_seconds": 0.0,
            "routing_seconds": native_route,
            "ingestion_status": "inherited_native_features",
        },
        "oracle": {
            "persistent_bytes": 0,
            "ingestion_seconds": 0.0,
            "routing_seconds": 0.0,
            "ingestion_status": "evaluation_identity_control",
        },
    }
    return CaseChannels(
        case=case,
        scores={
            "L": _finite(lexical),
            "E": _finite(extractive),
            "S": _finite(summary),
            "QK": _finite(qk),
            "QK8": _finite(qk8),
            "native_mean": _finite(native),
            "oracle": np.asarray(
                [float(index in case.positive_indices) for index in range(parent_count)]
            ),
        },
        costs=costs,
        summary_index=summary_index,
        typed_index=typed_index,
        typed_sidecars=sidecars,
    )


def _weights(template: str, channels: Sequence[str]) -> dict[str, float]:
    if template == "equal":
        return {name: 1.0 for name in channels}
    if template == "address_safe":
        return {name: 2.0 if name in {"L", "E"} else 1.0 for name in channels}
    if template == "semantic":
        return {name: 2.0 if name in {"S", "QK"} else 1.0 for name in channels}
    if template == "summary":
        return {name: 3.0 if name == "S" else 1.0 for name in channels}
    if template == "qk":
        return {name: 3.0 if name == "QK" else 1.0 for name in channels}
    raise ValueError(f"Unknown fusion template: {template}")


def _metrics_from_order(case, order: Sequence[int], k: int) -> dict[str, float | str | int]:
    selected = tuple(order[: min(k, len(order))])
    positives = set(case.positive_indices)
    recovered = positives.intersection(selected)
    first_rank = next((rank for rank, index in enumerate(order, start=1) if index in positives), None)
    materialized = sum(
        int(case.feature["parent_spans"][index][1])
        - int(case.feature["parent_spans"][index][0])
        for index in selected
    )
    return {
        "evidence_recall": len(recovered) / max(len(positives), 1),
        "complete_recovery": float(bool(positives) and recovered == positives),
        "precision": len(recovered) / max(len(selected), 1),
        "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        "selected_indices": " ".join(map(str, selected)),
        "recovered_indices": " ".join(map(str, sorted(recovered))),
        "positive_indices": " ".join(map(str, case.positive_indices)),
        "selected_chunks": len(selected),
        "native_kv_tokens_materialized": materialized,
    }


def _condition_row(
    item: CaseChannels,
    *,
    family: str,
    policy: str,
    channels: Sequence[str],
    order: Sequence[int],
    k: int,
    parameter: str = "",
) -> dict:
    local = {name: item.scores[name] for name in channels}
    provenance = candidate_provenance(local)
    metrics = _metrics_from_order(item.case, order, k)
    selected = [int(value) for value in str(metrics["selected_indices"]).split()]
    return {
        "dataset": item.case.dataset,
        "split": item.case.split,
        "example_id": item.case.example_id,
        "family": family,
        "policy": policy,
        "channels": "+".join(channels),
        "parameter": parameter,
        "k_total": k,
        **metrics,
        "candidate_chunks": len(order),
        "routing_index_bytes": sum(int(item.costs[name]["persistent_bytes"]) for name in channels),
        "routing_seconds": sum(float(item.costs[name]["routing_seconds"]) for name in channels),
        "selected_provenance_json": json.dumps(
            {str(index): provenance[index] for index in selected}, sort_keys=True
        ),
        "native_kv_rewritten": False,
    }


def _validation_choice(
    validation: Sequence[CaseChannels],
) -> tuple[dict, list[dict]]:
    policy: dict[str, dict[str, dict]] = {}
    rows = []
    for dataset in DATASETS:
        local_cases = [item for item in validation if item.case.dataset == dataset]
        policy[dataset] = {}
        for channels in COMBINATIONS:
            name = "+".join(channels)
            candidates = []
            for constant in RRF_GRID:
                values = []
                for item in local_cases:
                    scores = {channel: item.scores[channel] for channel in channels}
                    order = stable_topk(
                        reciprocal_rank_fusion_scores(scores, constant=constant),
                        len(item.case.chunk_texts),
                    )
                    values.append(_metrics_from_order(item.case, order, PRIMARY_K))
                candidates.append(
                    (
                        statistics.fmean(float(row["evidence_recall"]) for row in values),
                        statistics.fmean(float(row["complete_recovery"]) for row in values),
                        statistics.fmean(float(row["reciprocal_rank"]) for row in values),
                        -constant,
                        "rrf",
                        constant,
                    )
                )
            best_rrf = max(candidates)
            fusion_candidates = []
            for template in FUSION_TEMPLATES:
                values = []
                for item in local_cases:
                    scores = {channel: item.scores[channel] for channel in channels}
                    order = stable_topk(
                        normalized_score_fusion(scores, _weights(template, channels)),
                        len(item.case.chunk_texts),
                    )
                    values.append(_metrics_from_order(item.case, order, PRIMARY_K))
                fusion_candidates.append(
                    (
                        statistics.fmean(float(row["evidence_recall"]) for row in values),
                        statistics.fmean(float(row["complete_recovery"]) for row in values),
                        statistics.fmean(float(row["reciprocal_rank"]) for row in values),
                        -FUSION_TEMPLATES.index(template),
                        "fusion",
                        template,
                    )
                )
            best_fusion = max(fusion_candidates)
            policy[dataset][name] = {
                "rrf_constant": float(best_rrf[-1]),
                "fusion_template": str(best_fusion[-1]),
                "selection_split": "validation",
                "selection_examples": len(local_cases),
            }
            for candidate in (*candidates, *fusion_candidates):
                rows.append(
                    {
                        "dataset": dataset,
                        "channels": name,
                        "family": candidate[-2],
                        "parameter": candidate[-1],
                        "evidence_recall": candidate[0],
                        "complete_recovery": candidate[1],
                        "reciprocal_rank": candidate[2],
                        "selected": candidate[-1] in {best_rrf[-1], best_fusion[-1]},
                    }
                )
    return policy, rows


def _evaluate_test(
    items: Sequence[CaseChannels], validation_policy: Mapping,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    union_rows: list[dict] = []
    reserved_rows: list[dict] = []
    rrf_rows: list[dict] = []
    fusion_rows: list[dict] = []
    for item in items:
        count = len(item.case.chunk_texts)
        for k in K_VALUES:
            for channel in (*CHANNELS, "QK8", "native_mean", "oracle"):
                order = stable_topk(item.scores[channel], count)
                union_rows.append(
                    _condition_row(
                        item,
                        family="single",
                        policy=channel,
                        channels=(channel,),
                        order=order,
                        k=k,
                    )
                )
            for channels in COMBINATIONS:
                local = {channel: item.scores[channel] for channel in channels}
                name = "+".join(channels)
                union_rows.append(
                    _condition_row(
                        item,
                        family="union_round_robin",
                        policy=f"{name}:U0",
                        channels=channels,
                        order=rank_round_robin(local, count),
                        k=k,
                    )
                )
                union_rows.append(
                    _condition_row(
                        item,
                        family="union_agreement",
                        policy=f"{name}:U2",
                        channels=channels,
                        order=agreement_priority_union(local, count, candidate_pool=k),
                        k=k,
                        parameter=f"candidate_pool={k}",
                    )
                )
                frozen = validation_policy[item.case.dataset][name]
                constant = float(frozen["rrf_constant"])
                order = stable_topk(reciprocal_rank_fusion_scores(local, constant=constant), count)
                rrf_rows.append(
                    _condition_row(
                        item,
                        family="rrf",
                        policy=f"{name}:RRF",
                        channels=channels,
                        order=order,
                        k=k,
                        parameter=f"c={constant:g}",
                    )
                )
                template = str(frozen["fusion_template"])
                order = stable_topk(
                    normalized_score_fusion(local, _weights(template, channels)), count
                )
                fusion_rows.append(
                    _condition_row(
                        item,
                        family="normalized_fusion",
                        policy=f"{name}:fusion",
                        channels=channels,
                        order=order,
                        k=k,
                        parameter=f"template={template}",
                    )
                )
        for policy_name, (allocation, channels) in RESERVED_POLICIES.items():
            local = {channel: item.scores[channel] for channel in channels}
            reserved_rows.append(
                _condition_row(
                    item,
                    family="reserved_slots",
                    policy=policy_name,
                    channels=channels,
                    order=reserved_slot_union(local, allocation, k=count),
                    k=PRIMARY_K,
                    parameter=json.dumps(allocation, sort_keys=True),
                )
            )
    return union_rows, reserved_rows, rrf_rows, fusion_rows


def _channel_diagnostics(items: Sequence[CaseChannels]) -> tuple[list[dict], list[dict]]:
    hits = []
    overlap = []
    for item in items:
        local = {name: item.scores[name] for name in CHANNELS}
        provenance = candidate_provenance(local)
        selected = {name: set(stable_topk(scores, PRIMARY_K)) for name, scores in local.items()}
        positives = set(item.case.positive_indices)
        for candidate, channels in enumerate(provenance):
            support = [name for name in CHANNELS if candidate in selected[name]]
            if candidate not in positives:
                recovery_type = "non_evidence"
            elif not support:
                recovery_type = "missed_by_all"
            elif len(support) == 1:
                recovery_type = f"{support[0]}_only"
            else:
                recovery_type = "multi_channel"
            row = {
                "dataset": item.case.dataset,
                "example_id": item.case.example_id,
                "parent_index": candidate,
                "is_evidence": candidate in positives,
                "recovery_type": recovery_type,
                "support_channels": "+".join(support),
            }
            for name in CHANNELS:
                row[f"{name}_score"] = channels[name]["score"]
                row[f"{name}_rank"] = channels[name]["rank"]
                row[f"{name}_top4"] = candidate in selected[name]
            hits.append(row)
        for left, right in combinations(CHANNELS, 2):
            left_set, right_set = selected[left], selected[right]
            union = left_set | right_set
            overlap.append(
                {
                    "dataset": item.case.dataset,
                    "example_id": item.case.example_id,
                    "left_channel": left,
                    "right_channel": right,
                    "k_total": PRIMARY_K,
                    "selection_jaccard": len(left_set & right_set) / max(len(union), 1),
                    "left_evidence_recovered": len(left_set & positives),
                    "right_evidence_recovered": len(right_set & positives),
                    "shared_evidence_recovered": len(left_set & right_set & positives),
                    "left_only_evidence": len((left_set - right_set) & positives),
                    "right_only_evidence": len((right_set - left_set) & positives),
                }
            )
    return hits, overlap


def _retention(reference: Iterable[str], summary_terms: set[str]) -> tuple[int, int, float | str]:
    values = set(reference)
    if not values:
        return 0, 0, ""
    retained = sum(term.casefold() in summary_terms for term in values)
    return retained, len(values), retained / len(values)


def _coverage_rows(items: Sequence[CaseChannels]) -> list[dict]:
    rows = []
    for item in items:
        summary_ranking = stable_topk(item.scores["S"], len(item.case.chunk_texts))
        inverse_rank = {candidate: rank for rank, candidate in enumerate(summary_ranking, start=1)}
        query_terms = set(lexical_terms(item.case.question))
        for index in item.case.positive_indices:
            source_terms = tuple(lexical_terms(item.case.chunk_texts[index]))
            summary_terms = set(lexical_terms(item.summary_index.records[index].summary))
            sidecar = item.typed_sidecars[index]
            diagnostics = {
                "entity": tuple(term.casefold() for value in sidecar.entities for term in lexical_terms(value)),
                "number_date": tuple(term.casefold() for value in sidecar.numbers_dates for term in lexical_terms(value)),
                "rare_term": sidecar.rare_terms,
                "relation_term": sidecar.relation_terms,
                "evidence_key": tuple(term for term in source_terms if term in query_terms),
            }
            row = {
                "dataset": item.case.dataset,
                "example_id": item.case.example_id,
                "parent_index": index,
                "summary_target_rank": inverse_rank[index],
                "summary_target_recovered_at4": index in summary_ranking[:PRIMARY_K],
                "summary_token_count": item.summary_index.records[index].summary_token_count,
                "source_token_count": int(item.case.feature["parent_spans"][index][1])
                - int(item.case.feature["parent_spans"][index][0]),
                "compression_ratio": item.summary_index.records[index].summary_token_count
                / max(
                    int(item.case.feature["parent_spans"][index][1])
                    - int(item.case.feature["parent_spans"][index][0]),
                    1,
                ),
                "alias_reference_available": False,
                "alias_recall": "",
            }
            for name, reference in diagnostics.items():
                retained, total, recall = _retention(reference, summary_terms)
                row[f"{name}_retained"] = retained
                row[f"{name}_reference"] = total
                row[f"{name}_recall"] = recall
            rows.append(row)
    return rows


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0 + 1.0
        cursor = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | str:
    if len(left) < 3:
        return ""
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return ""
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _correlation_rows(coverage: Sequence[dict]) -> list[dict]:
    metrics = (
        "entity_recall",
        "number_date_recall",
        "rare_term_recall",
        "relation_term_recall",
        "evidence_key_recall",
        "compression_ratio",
        "summary_token_count",
    )
    outcomes = ("summary_target_rank", "summary_target_recovered_at4")
    rows = []
    for dataset in (*DATASETS, "pooled"):
        local = list(coverage) if dataset == "pooled" else [row for row in coverage if row["dataset"] == dataset]
        for metric in metrics:
            for outcome in outcomes:
                pairs = [
                    (float(row[metric]), float(row[outcome]))
                    for row in local
                    if row.get(metric, "") != ""
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "diagnostic": metric,
                        "routing_outcome": outcome,
                        "n": len(pairs),
                        "spearman_rho": _spearman(
                            [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                        )
                        if pairs
                        else "",
                        "unit": "positive_source_parent",
                    }
                )
    return rows


def _cost_rows(items: Sequence[CaseChannels]) -> list[dict]:
    rows = []
    stacks = {"L+E+QK": ("L", "E", "QK"), "L+E+S+QK": ("L", "E", "S", "QK")}
    for item in items:
        for channel, cost in item.costs.items():
            rows.append(
                {
                    "dataset": item.case.dataset,
                    "example_id": item.case.example_id,
                    "scope": "single_channel",
                    "address_views": channel,
                    "persistent_bytes": cost["persistent_bytes"],
                    "ingestion_seconds": cost["ingestion_seconds"],
                    "routing_seconds": cost["routing_seconds"],
                    "summary_generation_seconds": cost.get("generation_seconds", 0.0),
                    "summary_embedding_seconds": cost.get("embedding_seconds", 0.0),
                    "shared_backing_memory_copies": 1,
                    "ingestion_status": cost["ingestion_status"],
                }
            )
        for name, channels in stacks.items():
            rows.append(
                {
                    "dataset": item.case.dataset,
                    "example_id": item.case.example_id,
                    "scope": "combined_stack",
                    "address_views": name,
                    "persistent_bytes": sum(int(item.costs[c]["persistent_bytes"]) for c in channels),
                    "ingestion_seconds": sum(float(item.costs[c]["ingestion_seconds"]) for c in channels),
                    "routing_seconds": sum(float(item.costs[c]["routing_seconds"]) for c in channels),
                    "summary_generation_seconds": float(item.costs["S"]["generation_seconds"])
                    if "S" in channels
                    else 0.0,
                    "summary_embedding_seconds": float(item.costs["S"]["embedding_seconds"])
                    if "S" in channels
                    else 0.0,
                    "shared_backing_memory_copies": 1,
                    "ingestion_status": "component_sum; QK projection inherited",
                }
            )
    return rows


def run(args) -> dict:
    args.output_root.mkdir(parents=True, exist_ok=True)
    print("[setup] load tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    client = OllamaClient(args.ollama_url, timeout=args.ollama_timeout)
    print("[setup] reconstruct validation identities", flush=True)
    validation_cases = _load_aligned_cases(args, tokenizer, "validation")
    print("[setup] reconstruct test identities", flush=True)
    test_cases = _load_aligned_cases(args, tokenizer, "test")
    all_cases = [*validation_cases, *test_cases]
    device = torch.device(args.device)
    missing_queries = [case.feature for case in all_cases if "query_pre_query" not in case.feature]
    if missing_queries:
        print(f"[setup] project {len(missing_queries)} native queries on {device}", flush=True)
        _project_native_queries({"multi_index": missing_queries}, device)
    print("[setup] load inherited low-rank checkpoints", flush=True)
    lowrank_models = {
        (dataset, rank): _load_lowrank(args.feature_root, dataset, rank, device)
        for dataset in DATASETS
        for rank in (16, 8)
    }

    scored: dict[str, list[CaseChannels]] = {}
    for split, cases in (("validation", validation_cases), ("test", test_cases)):
        addresses = _summary_rows(split)
        scored[split] = []
        for number, case in enumerate(cases, start=1):
            scored[split].append(
                _case_channels(case, addresses, lowrank_models, client, device)
            )
            print(f"[{split} {number}/{len(cases)}] {case.dataset} {case.example_id}", flush=True)

    validation_policy, validation_rows = _validation_choice(scored["validation"])
    union_rows, reserved_rows, rrf_rows, fusion_rows = _evaluate_test(
        scored["test"], validation_policy
    )
    hit_rows, overlap_rows = _channel_diagnostics(scored["test"])
    coverage_rows = _coverage_rows(scored["test"])
    correlation_rows = _correlation_rows(coverage_rows)
    cost_rows = _cost_rows(scored["test"])

    _write_csv(args.output_root / "validation_policy_grid.csv", validation_rows)
    _write_json(args.output_root / "validation_policy.json", validation_policy)
    _write_csv(args.output_root / "multi_index_union_results.csv", union_rows)
    _write_csv(args.output_root / "multi_index_reserved_slot_results.csv", reserved_rows)
    _write_csv(args.output_root / "multi_index_rrf_results.csv", rrf_rows)
    _write_csv(args.output_root / "multi_index_fusion_results.csv", fusion_rows)
    _write_csv(args.output_root / "multi_index_channel_hits.csv", hit_rows)
    _write_csv(args.output_root / "multi_index_overlap.csv", overlap_rows)
    _write_csv(args.output_root / "multi_index_costs.csv", cost_rows)
    _write_csv(args.output_root / "summary_address_coverage.csv", coverage_rows)
    _write_csv(args.output_root / "summary_quality_retrieval_correlation.csv", correlation_rows)

    source_paths = {
        "expanded_validation": SOURCE_RESULTS / "expanded_validation" / "summary_addresses.jsonl",
        **{
            str(policy["source_run"]): SOURCE_RESULTS
            / str(policy["source_run"])
            / "summary_addresses.jsonl"
            for policy in SUMMARY_POLICY.values()
        },
    }
    manifest = {
        "schema_version": "1.0",
        "paper": "3.1",
        "experiment": "fixed_budget_multi_index",
        "selection_budget_primary": PRIMARY_K,
        "selection_budget_sweep": list(K_VALUES),
        "validation_examples": len(scored["validation"]),
        "test_examples": len(scored["test"]),
        "test_examples_by_dataset": {
            dataset: sum(item.case.dataset == dataset for item in scored["test"])
            for dataset in DATASETS
        },
        "summary_policy": SUMMARY_POLICY,
        "validation_policy": validation_policy,
        "summary_length_sensitivity_run": False,
        "summary_length_note": (
            "Frozen caches contain at most 32 generated tokens; longer 64/128-token outputs "
            "would require a new generation intervention and were not fabricated."
        ),
        "rouge_run": False,
        "rouge_note": "No independently authored chunk-level reference summaries exist.",
        "source_native_kv_rewritten": False,
        "materialization_budget_shared_across_policies": True,
        "source_model": MODEL_ID,
        "source_model_revision": MODEL_REVISION,
        "embedding_model": EMBEDDING_MODEL,
        "device": str(device),
        "source_artifacts": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in source_paths.items()
        },
        "implementation_sha256": hashlib.sha256(
            (ROOT / "src/pra_hf/multi_index.py").read_bytes()
        ).hexdigest(),
    }
    _write_json(args.output_root / "manifest.json", manifest)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:/git/rd/pdattention-paper2-8/data/.hf_cache"))
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=DEFAULT_DATA_ROOT / "2wiki/dev.json")
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=DEFAULT_DATA_ROOT / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument("--dataset-seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=float, default=900.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
