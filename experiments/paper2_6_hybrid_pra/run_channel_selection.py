"""Separate PRA root/successor channel choice and audit robust selection.

This follow-up consumes the frozen four-dataset feature cohorts from the
preceding Paper 2.6 study.  It evaluates two requested root chunks followed by
two requested successor chunks, preserving the same total budget of four and
stopping before native-K/V materialization or generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_6_hybrid_pra.run_channel_geometry import (  # noqa: E402
    DATASETS,
    LABELS,
    MODEL_ID,
    MODEL_REVISION,
    WEIGHTS,
    _load_cases,
    _pieces,
    _write_csv,
)
from experiments.paper2_6_hybrid_pra.run_study import (  # noqa: E402
    _records,
    _route_case,
    _token_spans,
)
from pra_hf.channel_geometry import (  # noqa: E402
    headroom_decomposition,
    jaccard,
    new_address_tokens,
    oracle_channel,
    precision_recall,
    reciprocal_rank_fusion,
    select_observable_channel,
    useful_address,
)
from pra_hf.hybrid_discovery import HybridDiscoveryPolicy, TokenNativeIndex  # noqa: E402
from pra_hf.iterative import IterativeGistRouter  # noqa: E402


ROOT_CHANNELS = ("gist", "exact", "bm25", "approx", "hybrid")
SUCCESSOR_CHANNELS = (
    "native_semantic",
    "exact_new_address",
    "bm25_state",
    "approximate_new_address",
    "hybrid_state",
)
ROOT_MODES = {
    "gist": "gist_only",
    "exact": "token_exact",
    "bm25": "bm25",
    "approx": "token_approx",
    "hybrid": "token_semantic_rerank",
}
SUCCESSOR_MODES = {
    "native_semantic": "gist_only",
    "exact_new_address": "token_exact",
    "bm25_state": "bm25",
    "approximate_new_address": "token_approx",
    "hybrid_state": "iterative_hybrid",
}


def _read_csv(path: Path) -> list[dict]:
    """Read generated rows and restore numeric fields for postprocessing."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value is None or value == "":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def _metrics(selected: list[str], gold: set[str], ranking: list[str]) -> dict[str, float]:
    precision, recall = precision_recall(selected, gold)
    rank = next((index for index, identity in enumerate(ranking, 1) if identity in gold), None)
    return {
        "recall": recall,
        "precision": precision,
        "mrr": 1.0 / rank if rank else 0.0,
        "best_gold_rank": rank if rank else len(ranking) + 1,
        "complete_recovery": float(gold.issubset(selected)),
        "distractors": len(set(selected) - gold),
    }


def _rank(scores: dict[str, float], excluded: set[str] = frozenset()) -> list[str]:
    return sorted(
        (identity for identity in scores if identity not in excluded),
        key=lambda identity: (-scores[identity], identity),
    )


def _score_gap(scores: dict[str, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else (values[0] if values else 0.0)


def _candidate_scores(index, token_index, tokenizer, query_ids, semantic, mode, hop=1):
    rows = token_index.score(
        query_ids,
        semantic,
        tokenizer,
        HybridDiscoveryPolicy(
            mode=mode,
            semantic_weight=WEIGHTS[0],
            token_weight=1.0 - WEIGHTS[0],
            later_semantic_weight=WEIGHTS[1],
            later_token_weight=1.0 - WEIGHTS[1],
        ),
        hop=hop,
        parent_id="__root__" if hop == 1 else "__state__",
    )
    return {
        identity: float(candidate.selected_score)
        for identity, candidate in zip(index.chunk_ids, rows)
    }, {identity: candidate for identity, candidate in zip(index.chunk_ids, rows)}


def _root_gold(feature: dict, index) -> set[str]:
    mask = feature.get("root_positive_mask", feature["positive_mask"])
    gold = {identity for identity, value in zip(index.chunk_ids, mask) if bool(value)}
    return gold or {
        identity for identity, value in zip(index.chunk_ids, feature["positive_mask"]) if bool(value)
    }


def _successor_gold(feature: dict, index, roots: set[str]) -> tuple[set[str], str]:
    if "successor_positive_mask" in feature:
        gold = {
            identity
            for identity, value in zip(index.chunk_ids, feature["successor_positive_mask"])
            if bool(value)
        }
        return gold, "annotated_dependency"
    all_gold = {
        identity for identity, value in zip(index.chunk_ids, feature["positive_mask"]) if bool(value)
    }
    return all_gold - roots, "unordered_evidence_remainder"


def _index_memory_bytes(token_index: TokenNativeIndex) -> int:
    return sum(
        8 * len(row.token_ids)
        + sum(len(value.encode("utf-8")) for value in row.normalized_tokens)
        + sum(len(value.encode("utf-8")) for value in row.bm25_terms)
        for row in token_index.records
    )


def _observable_features(tokenizer, example, token_index, root_scores, root_selected):
    query_ids = tokenizer(example["question"], add_special_tokens=False).input_ids
    query_tokens = _pieces(tokenizer, query_ids)
    idfs = [token_index.idf.get(token, 0.0) for token in query_tokens]
    rare_cutoff = 0.75 * max(token_index.idf.values(), default=0.0)
    rare = [value for value in idfs if value >= rare_cutoff]
    words = re.findall(r"\b\w+\b", example["question"])
    top_ids = [ranking[0] for ranking in root_selected.values()]
    return {
        "query_tokens": len(query_ids),
        "query_terms": len(query_tokens),
        "rare_token_count": len(rare),
        "query_rare_fraction": len(rare) / max(len(query_tokens), 1),
        "query_idf_mean": mean(idfs) if idfs else 0.0,
        "query_idf_max": max(idfs, default=0.0),
        "named_entity_count": max(0, len(re.findall(r"\b[A-Z][\w-]+\b", example["question"])) - 1),
        "numeric_marker": int(bool(re.search(r"\d", example["question"]))),
        "url_marker": int(bool(re.search(r"(?:https?://|www\.|\w+\.\w+/)", example["question"]))),
        "id_marker": int(bool(re.search(r"\b(?:id|doi|isbn|arxiv)\b", example["question"], re.I))),
        "question_markers": int(example["question"].count("?")),
        "exact_top_score": max(root_scores["exact"].values()),
        "exact_score_gap": _score_gap(root_scores["exact"]),
        "bm25_top_score": max(root_scores["bm25"].values()),
        "bm25_score_gap": _score_gap(root_scores["bm25"]),
        "approx_top_score": max(root_scores["approx"].values()),
        "approx_score_gap": _score_gap(root_scores["approx"]),
        "gist_top_score": max(root_scores["gist"].values()),
        "semantic_score_gap": _score_gap(root_scores["gist"]),
        "hybrid_score_gap": _score_gap(root_scores["hybrid"]),
        "channel_disagreement": len(set(top_ids)),
        "mean_selected_jaccard": mean(
            jaccard(root_selected[left], root_selected[right])
            for left_index, left in enumerate(ROOT_CHANNELS)
            for right in ROOT_CHANNELS[left_index + 1 :]
        ),
        "query_region_layout": int(any(value.casefold() in {"section", "table", "figure", "appendix"} for value in words)),
        "facet_disagreement": len(set(top_ids)),
    }


def _evaluate_case(tokenizer, feature, example, budget):
    index, all_gold = _records(feature, example)
    token_index = TokenNativeIndex.from_gist_index(index, tokenizer)
    router = IterativeGistRouter(index)
    query = feature["queries"]["question_exp_h2.0"].float()
    semantic, _ = router._scores(query.unsqueeze(0))
    query_ids = tokenizer(example["question"], add_special_tokens=False).input_ids
    root_gold = _root_gold(feature, index)
    root_rows, root_scores, root_candidates, root_selected = [], {}, {}, {}
    for channel in ROOT_CHANNELS:
        started = time.perf_counter()
        scores, candidates = _candidate_scores(
            index, token_index, tokenizer, query_ids, semantic[0], ROOT_MODES[channel]
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        ranking = _rank(scores)
        selected = ranking[:budget]
        root_scores[channel] = scores
        root_candidates[channel] = candidates
        root_selected[channel] = selected
        root_rows.append(
            {
                "split": feature["split"],
                "dataset": feature["dataset"],
                "example_id": feature["example_id"],
                "channel": channel,
                **_metrics(selected, root_gold, ranking),
                "selected_chunk_ids": "|".join(selected),
                "gold_chunk_ids": "|".join(sorted(root_gold)),
                "requested_chunks": budget,
                "comparisons": len(index.chunk_ids),
                "index_lookups": len(query_ids),
                "token_span_operations": (
                    len(query_ids) * sum(len(row.token_ids) for row in token_index.records)
                    if channel != "gist"
                    else 0
                ),
                "latency_ms": elapsed,
                "index_memory_bytes": _index_memory_bytes(token_index) if channel != "gist" else 0,
                "placement": "CPU exhaustive Python",
                "top_score": max(scores.values()),
                "score_gap": _score_gap(scores),
            }
        )

    rankings = {
        channel: {identity: rank for rank, identity in enumerate(_rank(root_scores[channel]), 1)}
        for channel in ROOT_CHANNELS
    }
    rrf_scores = reciprocal_rank_fusion(rankings)
    rrf_ranking = _rank(rrf_scores)
    root_rows.append(
        {
            "split": feature["split"], "dataset": feature["dataset"],
            "example_id": feature["example_id"], "channel": "rrf",
            **_metrics(rrf_ranking[:budget], root_gold, rrf_ranking),
            "selected_chunk_ids": "|".join(rrf_ranking[:budget]),
            "gold_chunk_ids": "|".join(sorted(root_gold)), "requested_chunks": budget,
            "comparisons": len(index.chunk_ids) * len(ROOT_CHANNELS), "index_lookups": 0,
            "token_span_operations": 0, "latency_ms": 0.0,
            "index_memory_bytes": _index_memory_bytes(token_index),
            "placement": "CPU rank aggregation", "top_score": max(rrf_scores.values()),
            "score_gap": _score_gap(rrf_scores),
        }
    )

    successor_rows = []
    successor_details = {}
    id_to_row = {identity: row for row, identity in enumerate(index.chunk_ids)}
    source_ids = tokenizer(example["source"], add_special_tokens=False).input_ids
    for root_channel in ROOT_CHANNELS:
        roots = set(root_selected[root_channel])
        successor_gold, mapping = _successor_gold(feature, index, roots & all_gold)
        if not successor_gold:
            continue
        frontier_rows = [id_to_row[identity] for identity in root_selected[root_channel]]
        frontier_semantic, _ = router._scores(index.gists[frontier_rows, 0])
        root_token_ids = [
            token_index.records[row].token_ids for row in frontier_rows
        ]
        for successor_channel in SUCCESSOR_CHANNELS:
            started = time.perf_counter()
            merged = {identity: float("-inf") for identity in index.chunk_ids}
            candidate_trace = {}
            for frontier_index, (row_index, state_ids) in enumerate(zip(frontier_rows, root_token_ids)):
                if successor_channel in {"bm25_state", "hybrid_state"}:
                    lexical_state = [*query_ids, *state_ids]
                else:
                    lexical_state = list(state_ids)
                scores, candidates = _candidate_scores(
                    index,
                    token_index,
                    tokenizer,
                    lexical_state,
                    frontier_semantic[frontier_index],
                    SUCCESSOR_MODES[successor_channel],
                    hop=2,
                )
                for identity, score in scores.items():
                    if score > merged[identity]:
                        merged[identity] = score
                        candidate_trace[identity] = candidates[identity]
            elapsed = (time.perf_counter() - started) * 1000.0
            ranking = _rank(merged, roots)
            selected = ranking[:budget]
            result = {
                "split": feature["split"], "dataset": feature["dataset"],
                "example_id": feature["example_id"], "root_channel": root_channel,
                "successor_channel": successor_channel, "mapping_semantics": mapping,
                **_metrics(selected, successor_gold, ranking),
                "root_recall": next(row["recall"] for row in root_rows if row["channel"] == root_channel),
                "root_valid": float(bool(roots & root_gold)),
                "path_recall": len((roots | set(selected)) & all_gold) / max(len(all_gold), 1),
                "path_gain": len(set(selected) & all_gold - roots) / max(len(all_gold), 1),
                "selected_root_ids": "|".join(root_selected[root_channel]),
                "selected_chunk_ids": "|".join(selected),
                "gold_chunk_ids": "|".join(sorted(successor_gold)),
                "requested_chunks": budget, "frontiers": len(frontier_rows),
                "comparisons": len(index.chunk_ids) * len(frontier_rows),
                "index_lookups": sum(len(values) for values in root_token_ids),
                "token_span_operations": (
                    sum(len(values) for values in root_token_ids)
                    * sum(len(row.token_ids) for row in token_index.records)
                    if successor_channel != "native_semantic" else 0
                ),
                "latency_ms": elapsed,
                "index_memory_bytes": _index_memory_bytes(token_index) if successor_channel != "native_semantic" else 0,
                "placement": "CPU exhaustive Python", "top_score": max(merged.values()),
                "score_gap": _score_gap({identity: score for identity, score in merged.items() if identity not in roots}),
            }
            successor_rows.append(result)
            successor_details[(root_channel, successor_channel)] = {
                "scores": merged, "ranking": ranking, "candidates": candidate_trace,
                "gold": successor_gold, "selected": selected,
            }
    observable = _observable_features(
        tokenizer, example, token_index, root_scores, root_selected
    )
    return {
        "root_rows": root_rows,
        "successor_rows": successor_rows,
        "observable": observable,
        "root_scores": root_scores,
        "root_candidates": root_candidates,
        "root_selected": root_selected,
        "successor_details": successor_details,
        "index": index,
        "token_index": token_index,
        "source_ids": source_ids,
        "query_ids": query_ids,
        "all_gold": all_gold,
        "root_gold": root_gold,
    }


def _best(rows, channel_field):
    order = ROOT_CHANNELS if channel_field == "channel" else SUCCESSOR_CHANNELS
    return max(rows, key=lambda row: (row["recall"], row["precision"], row["mrr"], -order.index(row[channel_field])))


def _aggregate(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [
        {**dict(zip(keys, key)), "examples": len(group), **{metric: mean(float(row[metric]) for row in group) for metric in metrics}}
        for key, group in sorted(groups.items())
    ]


def _fit_linear_selector(feature_rows, root_rows, seed):
    feature_names = [
        "query_tokens", "query_terms", "rare_token_count", "query_rare_fraction",
        "query_idf_mean", "query_idf_max", "named_entity_count", "numeric_marker",
        "url_marker", "id_marker", "question_markers", "exact_top_score",
        "exact_score_gap", "bm25_top_score", "bm25_score_gap", "approx_top_score",
        "approx_score_gap", "gist_top_score", "semantic_score_gap", "hybrid_score_gap",
        "channel_disagreement", "mean_selected_jaccard", "query_region_layout",
        "facet_disagreement",
    ]
    outcomes = defaultdict(list)
    for row in root_rows:
        if row["channel"] in ROOT_CHANNELS:
            outcomes[(row["split"], row["dataset"], row["example_id"])].append(row)
    labels = {key: _best(group, "channel")["channel"] for key, group in outcomes.items()}
    train = [row for row in feature_rows if row["split"] == "validation"]
    means = torch.tensor([mean(float(row[name]) for row in train) for name in feature_names])
    scales = torch.tensor([pstdev(float(row[name]) for row in train) or 1.0 for name in feature_names])
    x = torch.tensor([[float(row[name]) for name in feature_names] for row in train])
    x = (x - means) / scales
    y = torch.tensor([ROOT_CHANNELS.index(labels[(row["split"], row["dataset"], row["example_id"])]) for row in train])
    torch.manual_seed(seed)
    model = torch.nn.Linear(len(feature_names), len(ROOT_CHANNELS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.05)
    for _ in range(500):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    predictions = {}
    with torch.no_grad():
        for row in feature_rows:
            values = torch.tensor([float(row[name]) for name in feature_names])
            logits = model((values - means) / scales)
            predictions[(row["split"], row["dataset"], row["example_id"])] = ROOT_CHANNELS[int(logits.argmax())]
    return predictions, feature_names, float(loss.item())


def _successor_headroom(successor_rows, validation_root):
    """Decompose successor-channel headroom with the root policy held fixed."""
    fixed_root_rows = [
        row for row in successor_rows
        if row["root_channel"] == validation_root[row["dataset"]]
    ]
    validation_channel, heldout_channel = {}, {}
    for dataset in DATASETS:
        validation = _aggregate(
            [row for row in fixed_root_rows if row["split"] == "validation" and row["dataset"] == dataset],
            ("successor_channel",), ("recall", "precision", "mrr"),
        )
        heldout = _aggregate(
            [row for row in fixed_root_rows if row["split"] == "test" and row["dataset"] == dataset],
            ("successor_channel",), ("recall", "precision", "mrr"),
        )
        validation_channel[dataset] = _best(validation, "successor_channel")["successor_channel"]
        heldout_channel[dataset] = _best(heldout, "successor_channel")["successor_channel"]
    groups = defaultdict(dict)
    for row in fixed_root_rows:
        if row["split"] == "test":
            groups[(row["dataset"], row["example_id"])][row["successor_channel"]] = row
    output = []
    for (dataset, example_id), outcomes in groups.items():
        best = _best(list(outcomes.values()), "successor_channel")
        heldout = outcomes[heldout_channel[dataset]]
        validation = outcomes[validation_channel[dataset]]
        selection, instability = headroom_decomposition(
            best["recall"], heldout["recall"], validation["recall"]
        )
        output.append(
            {
                "dataset": dataset, "example_id": example_id,
                "fixed_root_channel": validation_root[dataset],
                "oracle_successor_channel": best["successor_channel"],
                "oracle_successor_recall": best["recall"],
                "best_heldout_successor_channel": heldout_channel[dataset],
                "best_heldout_successor_recall": heldout["recall"],
                "validation_successor_channel": validation_channel[dataset],
                "validation_successor_recall": validation["recall"],
                "selection_headroom": selection,
                "validation_instability": instability,
            }
        )
    return output, validation_channel, heldout_channel


def _taxonomy(case, winner, kind):
    candidates = case["root_candidates"][winner]
    selected = case["root_selected"][winner]
    gold = case["root_gold"]
    useful = [candidates[identity] for identity in selected if identity in gold]
    if kind == "approx":
        if any(row.normalized_exact_score > row.exact_span_score for row in useful):
            return "punctuation/case or BPE normalization"
        if any(row.approximate_score >= 0.5 and row.exact_span_score == 0 for row in useful):
            return "lexical near-match"
        if any(row.ordered_score > row.weighted_overlap_score for row in useful):
            return "partial entity mention"
        return "distributed subword overlap"
    if kind == "exact":
        if any(row.exact_span_score >= 0.75 for row in useful):
            return "technical phrase reuse"
        if any(row.entity_name_score > 0 for row in useful):
            return "identity-style reference"
        return "explicit evidence wording"
    if any(row.bm25_score >= 0.75 and row.weighted_overlap_score < 0.75 for row in useful):
        return "multiple moderately informative terms"
    if any(row.entity_name_score > 0 for row in useful):
        return "entity plus relation co-occurrence"
    return "distributed lexical evidence"


def _prior_musique_approx_wins(args, feature_rows):
    """Classify the strict approximate wins from the inherited four-chunk run."""
    path = args.output_dir.parent / "channel_geometry" / "channel_results_musique.csv"
    if not path.exists():
        return []
    rows = _read_csv(path)
    outcomes = defaultdict(dict)
    for row in rows:
        if row["split"] == "test":
            outcomes[row["example_id"]][row["channel"]] = row
    features = {
        row["example_id"]: row
        for row in feature_rows
        if row["split"] == "test" and row["dataset"] == "musique"
    }
    result = []
    for example_id, channels in outcomes.items():
        if not {"gist", "exact", "bm25", "approx"}.issubset(channels):
            continue
        approximate = float(channels["approx"]["evidence_recall"])
        alternatives = max(
            float(channels[channel]["evidence_recall"])
            for channel in ("gist", "exact", "bm25")
        )
        if approximate <= alternatives:
            continue
        feature = features.get(example_id, {})
        if float(feature.get("approx_top_score", 0)) > float(feature.get("exact_top_score", 0)) + 0.1:
            cause = "lexical near-match"
        elif float(feature.get("named_entity_count", 0)) > 0:
            cause = "partial entity mention"
        elif float(feature.get("rare_token_count", 0)) > 0:
            cause = "BPE/tokenization variation"
        else:
            cause = "distributed subword overlap"
        result.append(
            {
                "dataset": "musique", "example_id": example_id,
                "winner": "approx", "cause": cause,
                "recall_gain": approximate - alternatives,
                "analysis_budget": 4,
            }
        )
    return result


def _robust_feature(tokenizer, index, perturbation):
    entry = f"Meridian{index}"
    correct, wrong = f"Cobalt{index}", f"Copper{index}"
    mention = correct
    if perturbation == "case": mention = correct.upper()
    elif perturbation == "punctuation": mention = f"{correct},"
    elif perturbation == "typo": mention = f"Coblat{index}"
    elif perturbation == "near_entity_collision": wrong = f"Cobalto{index}"
    elif perturbation == "shared_prefix": correct, wrong, mention = f"Atlas{index}731", f"Atlas{index}730", f"Atlas{index}731"
    elif perturbation == "numeric_id_collision": correct, wrong, mention = f"ID-{index}73142", f"ID-{index}73124", f"ID-{index}73142"
    elif perturbation == "url_domain_overlap": correct, wrong, mention = f"api.example.org/v2/orders/{index}", f"api.example.org/v2/order/{index}", f"api.example.org/v2/orders/{index}"
    elif perturbation == "alias_synonym": correct, wrong, mention = f"New York City {index}", f"York City {index}", f"Big Apple {index}"
    elif perturbation == "same_name_wrong_entity": correct, wrong, mention = f"Mercury planet {index}", f"Mercury element {index}", f"Mercury {index}"
    elif perturbation == "same_class_wrong_instance": correct, wrong, mention = f"Cobalt Project {index}", f"Copper Project {index}", f"Project {index}"
    elif perturbation == "correct_entity_wrong_relation": correct, wrong, mention = f"Cobalt parent {index}", f"Cobalt rival {index}", f"Cobalt {index}"
    elif perturbation == "stale_alternate_alias": correct, wrong, mention = f"CurrentName{index}", f"OldName{index}", f"OldName{index}"
    elif perturbation == "two_plausible_references": correct, wrong, mention = f"Cobalt option {index}", f"Copper option {index}", f"Cobalt option {index} or Copper option {index}"
    chunks = [f"{entry} points to {mention}", f"{wrong} stores a plausible distractor", f"Unrelated {index} generic text", f"{correct} stores payload Zeta{index}"]
    if perturbation == "confidently_wrong":
        wrong, correct = f"Wrong{index}", f"Correct{index}"
        chunks = [f"{entry} points to {wrong}; {correct} is less prominent", f"{wrong} stores a plausible distractor", f"Unrelated {index} generic text", f"{correct} stores payload Zeta{index}"]
    source, spans = _token_spans(tokenizer, chunks)
    feature = {
        "split": "synthetic", "dataset": "robustness", "example_id": f"{perturbation}-{index}",
        "chunk_spans": spans,
        "memory_gists": torch.tensor([[1.,0.,0.,0.],[.8,.2,0.,0.],[.7,0.,.7,0.],[0.,1.,0.,0.]]),
        "positive_mask": torch.tensor([False,False,False,True]),
        "queries": {"question_exp_h2.0": torch.tensor([1.,0.,0.,0.])},
    }
    example = {"dataset":"robustness","id":feature["example_id"],"source":source,"question":f"Locate {entry}"}
    uri=f"benchmark://robustness/{feature['example_id']}"
    return feature, example, f"{uri}#chunk=3", f"{uri}#chunk=1"


def _robustness(tokenizer):
    perturbations = (
        "clean", "case", "punctuation", "typo", "alias_synonym",
        "confidently_wrong", "near_entity_collision", "shared_prefix",
        "numeric_id_collision", "url_domain_overlap", "same_name_wrong_entity",
        "same_class_wrong_instance", "correct_entity_wrong_relation",
        "stale_alternate_alias", "two_plausible_references",
    )
    modes = {"exact":"token_exact","approx":"token_approx","hybrid":"iterative_hybrid"}
    rows=[]
    for perturbation in perturbations:
        for index in range(20):
            feature,example,target,wrong=_robust_feature(tokenizer,index,perturbation)
            for channel,mode in modes.items():
                result,_=_route_case(tokenizer,feature,example,"H5_iterative_hybrid",mode,WEIGHTS,2)
                selected=set(result["selected_chunk_ids"].split("|"))
                confidence={value.rsplit("@",1)[0]:float(value.rsplit("@",1)[1]) for value in result["selected_confidence_pairs"].split("|") if "@" in value}
                wrong_conf=confidence.get(wrong,0.0)
                rows.append({"perturbation":perturbation,"example_id":feature["example_id"],"channel":channel,"target_recovery":float(target in selected),"wrong_target_recovery":float(wrong in selected),"wrong_target_confidence":wrong_conf,"abstention_at_0_5":float(max(confidence.values(),default=0)<.5),"retry_opportunity":float(target not in selected and wrong_conf<.6)})
    return rows


def _search_method_action_spec():
    """Return the stable discovery-method contract consumed by Paper 3.5."""
    shared_root = {
        "stage": "root",
        "required_state": ["question_token_ids", "question_hidden_state"],
        "required_index": ["identity_aligned_gist_index", "token_native_sidecar"],
        "parameters": {"top_k": [1, 2, 4], "chunk_tokens": [32]},
        "confidence_outputs": [
            "top_score", "top1_top2_gap", "effective_support",
            "top_candidate_agreement", "semantic_lexical_consistency",
        ],
        "cost_metrics": [
            "comparisons", "index_lookups", "token_span_operations",
            "latency_ms", "index_memory_bytes",
        ],
    }
    shared_successor = {
        "stage": "successor",
        "required_state": ["admitted_root_chunk", "current_path"],
        "required_index": ["identity_aligned_gist_index", "token_native_sidecar"],
        "parameters": {"top_k": [1, 2, 4]},
        "confidence_outputs": [
            "top_score", "top1_top2_gap", "effective_support",
            "current_path_consistency", "candidate_ambiguity",
        ],
        "cost_metrics": [
            "frontiers", "comparisons", "index_lookups",
            "token_span_operations", "latency_ms", "index_memory_bytes",
        ],
    }
    roots = {
        "semantic": {
            **shared_root,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[gist_only]",
            "legacy_channel": "gist",
            "required_index": ["identity_aligned_gist_index"],
            "known_failure_modes": ["semantic_collision", "lexical_identity_miss"],
        },
        "exact": {
            **shared_root,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[token_exact]",
            "legacy_channel": "exact",
            "known_failure_modes": ["typo_miss", "alias_miss", "same_name_ambiguity"],
        },
        "bm25": {
            **shared_root,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[bm25]",
            "legacy_channel": "bm25",
            "known_failure_modes": ["term_collision", "relation_ambiguity", "budget_dilution"],
        },
        "approximate": {
            **shared_root,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[token_approx]",
            "legacy_channel": "approx",
            "known_failure_modes": ["near_entity_collision", "shared_prefix_collision"],
        },
        "hybrid": {
            **shared_root,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[token_semantic_rerank]",
            "legacy_channel": "hybrid",
            "known_failure_modes": ["score_scale_mismatch", "budget_dilution"],
        },
    }
    successors = {
        "native_semantic": {
            **shared_successor,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[gist_only,hop=2]",
            "required_index": ["identity_aligned_gist_index"],
            "parameters": {"top_k": [1, 2, 4], "state": "admitted_chunk"},
            "known_failure_modes": ["semantic_collision", "new_address_blindness"],
        },
        "exact_new_address": {
            **shared_successor,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[token_exact,hop=2]",
            "parameters": {"top_k": [1, 2, 4], "state": "admitted_chunk"},
            "known_failure_modes": ["typo_miss", "alias_miss", "same_name_ambiguity"],
        },
        "bm25_state": {
            **shared_successor,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[bm25,hop=2]",
            "parameters": {"top_k": [1, 2, 4], "state": "question_plus_admitted_chunk"},
            "known_failure_modes": ["state_query_dilution", "term_collision"],
        },
        "approximate_new_address": {
            **shared_successor,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[token_approx,hop=2]",
            "parameters": {"top_k": [1, 2, 4], "state": "admitted_chunk"},
            "known_failure_modes": ["near_entity_collision", "shared_prefix_collision"],
        },
        "hybrid_state": {
            **shared_successor,
            "implementation_id": "pra_hf.hybrid_discovery:TokenNativeIndex.score[iterative_hybrid,hop=2]",
            "parameters": {"top_k": [1, 2, 4], "state": "question_plus_admitted_chunk"},
            "known_failure_modes": ["score_scale_mismatch", "distractor_expansion"],
        },
    }
    # Legacy keys remain aliases during the Paper 3.5 transition.
    for definition in [*roots.values(), *successors.values()]:
        definition["allowed_params"] = definition["parameters"]
        definition["confidence_signals"] = definition["confidence_outputs"]
        definition["failure_indicators"] = definition["known_failure_modes"]
    return {
        "schema_version": "2.0",
        "contract": "paper2.6-search-method-action-space",
        "root_successor_independent": True,
        "root_search_methods": roots,
        "successor_search_methods": successors,
        "materialization_performed": False,
    }


def _plots(output, root_summary, successor_summary, transitions, headroom, disagreement, taxonomy, useful_rows, postmortem, selector):
    output.mkdir(parents=True, exist_ok=True)
    colors=["#345995","#d1495b","#00798c","#edae49","#5f6f52"]
    for summary, channels, metric, stem, title in ((root_summary,ROOT_CHANNELS,"recall","root_recall","Root recall"),(successor_summary,SUCCESSOR_CHANNELS,"recall","successor_recall","Successor recall")):
        fig,ax=plt.subplots(figsize=(10.5,4.8),constrained_layout=True); width=.15
        for ci,channel in enumerate(channels):
            vals=[next((r[metric] for r in summary if r["dataset"]==d and r.get("channel",r.get("successor_channel"))==channel),0) for d in DATASETS]
            ax.bar([i+(ci-2)*width for i in range(4)],vals,width,label=channel.replace("_"," "),color=colors[ci])
        ax.set(xticks=range(4),xticklabels=[LABELS[d] for d in DATASETS],ylabel=title,ylim=(0,1)); ax.grid(axis="y",alpha=.25); ax.legend(ncol=3,fontsize=8)
        fig.savefig(output/f"{stem}.png",dpi=180); fig.savefig(output/f"{stem}.pdf"); plt.close(fig)
    matrix=torch.zeros((len(ROOT_CHANNELS),len(SUCCESSOR_CHANNELS)))
    for row in transitions: matrix[ROOT_CHANNELS.index(row["root_channel"]),SUCCESSOR_CHANNELS.index(row["successor_channel"])]=row["frequency"]
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True); im=ax.imshow(matrix,cmap="YlGnBu"); fig.colorbar(im,ax=ax,label="Held-out path frequency"); ax.set(xticks=range(5),xticklabels=[v.replace("_","\n") for v in SUCCESSOR_CHANNELS],yticks=range(5),yticklabels=ROOT_CHANNELS); ax.set(xlabel="Successor channel",ylabel="Root channel")
    fig.savefig(output/"channel_transition_heatmap.png",dpi=180); fig.savefig(output/"channel_transition_heatmap.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7),constrained_layout=True); x=range(4); hs=[mean(r["selection_headroom"] for r in headroom if r["dataset"]==d) for d in DATASETS]; hv=[mean(r["validation_instability"] for r in headroom if r["dataset"]==d) for d in DATASETS]; ax.bar(x,hs,label="true selection headroom"); ax.bar(x,hv,bottom=hs,label="validation instability"); ax.set(xticks=x,xticklabels=[LABELS[d] for d in DATASETS],ylabel="Recall difference"); ax.axhline(0,color="black",lw=.8); ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.savefig(output/"headroom_decomposition.png",dpi=180); fig.savefig(output/"headroom_decomposition.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.8),constrained_layout=True)
    for d in DATASETS:
        g=[r for r in disagreement if r["dataset"]==d and r["split"]=="test"]; ax.scatter([r["root_disagreement"] for r in g],[r["selection_headroom"] for r in g],label=LABELS[d],alpha=.7)
    ax.set(xlabel="Distinct root choices",ylabel="True oracle headroom"); ax.axhline(0,color="black",lw=.8); ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/"disagreement_true_headroom.png",dpi=180); fig.savefig(output/"disagreement_true_headroom.pdf"); plt.close(fig)
    counts=Counter(r["cause"] for r in taxonomy if r["dataset"]=="musique"); fig,ax=plt.subplots(figsize=(8,4.6),constrained_layout=True); ax.barh(list(counts),list(counts.values()),color="#edae49"); ax.set(xlabel="Approximate-win examples",title="MuSiQue approximate-win taxonomy")
    fig.savefig(output/"musique_approx_taxonomy.png",dpi=180); fig.savefig(output/"musique_approx_taxonomy.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True); groups=[("exposed only",[r for r in useful_rows if r["exposed"] and not r["useful_address"]]),("useful",[r for r in useful_rows if r["useful_address"]]),("none",[r for r in useful_rows if not r["exposed"]])]; ax.bar([g[0] for g in groups],[mean([r["iterative_gain"] for r in g[1]]) if g[1] else 0 for g in groups],color=["#9aa0a6","#00798c","#d1495b"]); ax.set(ylabel="Hybrid-state - native-semantic path recall"); ax.axhline(0,color="black",lw=.8)
    fig.savefig(output/"useful_address_gain.png",dpi=180); fig.savefig(output/"useful_address_gain.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    for kind,summary,field in (("root",root_summary,"channel"),("successor",successor_summary,"successor_channel")):
        ax.scatter([r["recall"] for r in summary],[r["precision"] for r in summary],label=kind,alpha=.75)
    ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/"root_successor_precision_recall.png",dpi=180); fig.savefig(output/"root_successor_precision_recall.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.6),constrained_layout=True); datasets=list(DATASETS); gold=[mean(r["unique_gold_from_lexical"] for r in postmortem if r["dataset"]==d and r["split"]=="test") for d in datasets]; dist=[mean(r["unique_distractors_from_lexical"] for r in postmortem if r["dataset"]==d and r["split"]=="test") for d in datasets]; x=range(4); ax.bar([i-.18 for i in x],gold,.36,label="unique gold"); ax.bar([i+.18 for i in x],dist,.36,label="unique distractors"); ax.set(xticks=x,xticklabels=[LABELS[d] for d in datasets],ylabel="Chunks per example"); ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.savefig(output/"hybrid_unique_contributions.png",dpi=180); fig.savefig(output/"hybrid_unique_contributions.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
    for policy,marker in (("best_heldout_fixed","o"),("simple_rules","s"),("linear","^"),("oracle","*")):
        g=[r for r in selector if r["policy"]==policy]; ax.scatter([r["recall"] for r in g],[r["precision"] for r in g],label=policy.replace("_"," "),marker=marker,s=85)
    ax.set(xlabel="Root recall",ylabel="Root precision",xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.2); ax.legend()
    fig.savefig(output/"selector_oracle_frontier.png",dpi=180); fig.savefig(output/"selector_oracle_frontier.pdf"); plt.close(fig)


def _postprocess_existing(args):
    """Recover plots and findings from completed, serialized measurements."""
    root_rows = _read_csv(args.output_dir / "root_channel_results.csv")
    successor_rows = _read_csv(args.output_dir / "successor_channel_results.csv")
    feature_rows = _read_csv(args.output_dir / "selector_observable_features.csv")
    root_test = [
        row for row in root_rows
        if row["split"] == "test" and row["channel"] in ROOT_CHANNELS
    ]
    root_summary = _aggregate(
        root_test, ("dataset", "channel"),
        ("recall", "precision", "mrr", "comparisons", "latency_ms"),
    )
    validation_fixed, heldout_fixed = {}, {}
    for dataset in DATASETS:
        validation = _aggregate(
            [
                row for row in root_rows
                if row["split"] == "validation"
                and row["dataset"] == dataset
                and row["channel"] in ROOT_CHANNELS
            ],
            ("channel",),
            ("recall", "precision", "mrr"),
        )
        validation_fixed[dataset] = _best(validation, "channel")["channel"]
        heldout_fixed[dataset] = _best(
            [row for row in root_summary if row["dataset"] == dataset], "channel"
        )["channel"]
    successor_summary = _aggregate(
        [
            row for row in successor_rows
            if row["split"] == "test"
            and row["root_channel"] == validation_fixed[row["dataset"]]
        ],
        ("dataset", "successor_channel"),
        ("recall", "precision", "mrr", "path_recall", "comparisons", "latency_ms"),
    )
    successor_headroom, validation_successor, heldout_successor = _successor_headroom(
        successor_rows, validation_fixed
    )
    _write_csv(
        args.output_dir / "successor_true_oracle_headroom.csv", successor_headroom
    )
    transitions = _read_csv(args.output_dir / "channel_transition_matrix.csv")
    headroom = _read_csv(args.output_dir / "channel_true_oracle_headroom.csv")
    disagreement = _read_csv(args.output_dir / "channel_disagreement_features.csv")
    taxonomy = (
        _read_csv(args.output_dir / "musique_approx_win_analysis.csv")
        + _read_csv(args.output_dir / "qasper_exact_win_analysis.csv")
        + _read_csv(args.output_dir / "bm25_win_analysis.csv")
    )
    if not any(row["dataset"] == "musique" for row in taxonomy):
        musique_rows = _prior_musique_approx_wins(args, feature_rows)
        _write_csv(args.output_dir / "musique_approx_win_analysis.csv", musique_rows)
        taxonomy.extend(musique_rows)
    useful_rows = _read_csv(args.output_dir / "iterative_useful_address.csv")
    postmortem = _read_csv(args.output_dir / "static_hybrid_postmortem.csv")
    selector = _read_csv(args.output_dir / "selector_results.csv")
    root_successor_table = _read_csv(args.output_dir / "root_successor_summary.csv")
    _, linear_features, linear_loss = _fit_linear_selector(
        feature_rows, root_rows, args.seed
    )
    _plots(
        args.output_dir, root_summary, successor_summary, transitions, headroom,
        disagreement, taxonomy, useful_rows, postmortem, selector,
    )
    findings_path = args.output_dir.parent / "channel_geometry" / "paper2_6_findings.json"
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    update = {
        "step_budget": args.step_budget,
        "total_budget": 2 * args.step_budget,
        "root_successor_summary": root_successor_table,
        "heldout_fixed": heldout_fixed,
        "validation_fixed": validation_fixed,
        "mean_true_selection_headroom": {
            dataset: mean(
                row["selection_headroom"] for row in headroom
                if row["dataset"] == dataset
            )
            for dataset in DATASETS
        },
        "mean_validation_instability": {
            dataset: mean(
                row["validation_instability"] for row in headroom
                if row["dataset"] == dataset
            )
            for dataset in DATASETS
        },
        "successor_validation_fixed": validation_successor,
        "successor_heldout_fixed": heldout_successor,
        "mean_successor_selection_headroom": {
            dataset: mean(
                row["selection_headroom"] for row in successor_headroom
                if row["dataset"] == dataset
            )
            for dataset in DATASETS
        },
        "mean_successor_validation_instability": {
            dataset: mean(
                row["validation_instability"] for row in successor_headroom
                if row["dataset"] == dataset
            )
            for dataset in DATASETS
        },
        "linear_selector_validation_loss": linear_loss,
        "linear_selector_features": linear_features,
        "generation_performed": False,
        "materialization_performed": False,
    }
    findings["channel_selection_iteration"] = update
    findings_path.write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    return update


def run(args):
    random.seed(args.seed); torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,local_files_only=args.local_files_only)
    cases=_load_cases(args); root_rows=[]; successor_rows=[]; feature_rows=[]; case_details={}
    for index,(feature,example) in enumerate(cases,1):
        detail=_evaluate_case(tokenizer,feature,example,args.step_budget); key=(feature["split"],feature["dataset"],feature["example_id"]); case_details[key]=detail; root_rows.extend(detail["root_rows"]); successor_rows.extend(detail["successor_rows"]); feature_rows.append({"split":key[0],"dataset":key[1],"example_id":key[2],**detail["observable"]}); print(f"[selection {index}/{len(cases)}] {key[1]} {key[2]}",flush=True)
    if {int(r["requested_chunks"]) for r in root_rows+successor_rows}!={args.step_budget}: raise AssertionError("Root/successor budgets differ.")
    root_test=[r for r in root_rows if r["split"]=="test" and r["channel"] in ROOT_CHANNELS]
    root_summary=_aggregate(root_test,("dataset","channel"),("recall","precision","mrr","comparisons","latency_ms"))
    validation_fixed={}; heldout_fixed={}
    for dataset in DATASETS:
        validation_fixed[dataset]=_best(_aggregate([r for r in root_rows if r["split"]=="validation" and r["dataset"]==dataset and r["channel"] in ROOT_CHANNELS],("channel",),("recall","precision","mrr")),"channel")["channel"]
        heldout_fixed[dataset]=_best([r for r in root_summary if r["dataset"]==dataset],"channel")["channel"]
    successor_test=[
        row for row in successor_rows
        if row["split"]=="test"
        and row["root_channel"]==validation_fixed[row["dataset"]]
    ]
    successor_summary=_aggregate(successor_test,("dataset","successor_channel"),("recall","precision","mrr","path_recall","comparisons","latency_ms"))
    successor_headroom,validation_successor,heldout_successor=_successor_headroom(successor_rows,validation_fixed)
    headroom=[]
    by_root=defaultdict(dict)
    for r in root_rows:
        if r["channel"] in ROOT_CHANNELS: by_root[(r["split"],r["dataset"],r["example_id"])][r["channel"]]=r
    for key,outcomes in by_root.items():
        if key[0]!="test": continue
        recalls={c:outcomes[c]["recall"] for c in ROOT_CHANNELS}; oracle_name,oracle_value=oracle_channel(recalls,ROOT_CHANNELS); held=heldout_fixed[key[1]]; valid=validation_fixed[key[1]]; hs,hv=headroom_decomposition(oracle_value,recalls[held],recalls[valid]); headroom.append({"dataset":key[1],"example_id":key[2],"oracle_channel":oracle_name,"oracle_recall":oracle_value,"best_heldout_fixed_channel":held,"best_heldout_fixed_recall":recalls[held],"validation_selected_channel":valid,"validation_selected_recall":recalls[valid],"selection_headroom":hs,"validation_instability":hv})
    transitions_by_example=[]
    for key,detail in case_details.items():
        if key[0]!="test" or not detail["successor_rows"]: continue
        root_best=_best([r for r in detail["root_rows"] if r["channel"] in ROOT_CHANNELS],"channel")["channel"]; options=[r for r in detail["successor_rows"] if r["root_channel"]==root_best]; successor_best=_best(options,"successor_channel"); transitions_by_example.append({"dataset":key[1],"example_id":key[2],"root_channel":root_best,"successor_channel":successor_best["successor_channel"],"path_gain":successor_best["path_gain"]})
    transitions=[]
    transition_groups=defaultdict(list)
    for row in transitions_by_example: transition_groups[(row["root_channel"],row["successor_channel"])].append(row)
    for (root_channel,successor_channel),group in transition_groups.items(): transitions.append({"root_channel":root_channel,"successor_channel":successor_channel,"frequency":len(group),"path_gain":mean(r["path_gain"] for r in group)})
    linear_predictions,linear_features,linear_loss=_fit_linear_selector(feature_rows,root_rows,args.seed)
    selector=[]
    for dataset in DATASETS:
        keys=[key for key in by_root if key[0]=="test" and key[1]==dataset]
        for policy in ("validation_fixed","best_heldout_fixed","simple_rules","linear","oracle"):
            chosen=[]
            for key in keys:
                outcomes=by_root[key]; features=next(r for r in feature_rows if (r["split"],r["dataset"],r["example_id"])==key)
                if policy=="validation_fixed": channel=validation_fixed[dataset]
                elif policy=="best_heldout_fixed": channel=heldout_fixed[dataset]
                elif policy=="simple_rules": channel=select_observable_channel(features)
                elif policy=="linear": channel=linear_predictions[key]
                else: channel=_best(list(outcomes.values()),"channel")["channel"]
                chosen.append(outcomes[channel])
            selector.append({"dataset":dataset,"policy":policy,"examples":len(chosen),"recall":mean(r["recall"] for r in chosen),"precision":mean(r["precision"] for r in chosen),"mrr":mean(r["mrr"] for r in chosen)})
    disagreement=[]; taxonomy=[]; postmortem=[]; useful_rows=[]; address_conf=[]
    headroom_lookup={(r["dataset"],r["example_id"]):r for r in headroom}
    for key,detail in case_details.items():
        rankings={c:_rank(detail["root_scores"][c]) for c in ROOT_CHANNELS}; top10=[set(v[:10]) for v in rankings.values()]; selected=detail["root_selected"]
        scores_by_identity=zip(*(detail["root_scores"][c].values() for c in ROOT_CHANNELS))
        h=headroom_lookup.get((key[1],key[2]),{})
        disagreement.append({"split":key[0],"dataset":key[1],"example_id":key[2],"selected_set_jaccard":mean(jaccard(selected[a],selected[b]) for i,a in enumerate(ROOT_CHANNELS) for b in ROOT_CHANNELS[i+1:]),"rank_overlap_top10":mean(jaccard(top10[i],top10[j]) for i in range(5) for j in range(i+1,5)),"score_disagreement":mean(pstdev(v) for v in scores_by_identity),"root_disagreement":len({v[0] for v in rankings.values()}),"selection_headroom":h.get("selection_headroom",0.0)})
        outcome={r["channel"]:r for r in detail["root_rows"] if r["channel"] in ROOT_CHANNELS}
        if key[0]=="test":
            if key[1]=="musique" and outcome["approx"]["recall"]>max(outcome[c]["recall"] for c in ROOT_CHANNELS if c!="approx"): taxonomy.append({"dataset":key[1],"example_id":key[2],"winner":"approx","cause":_taxonomy(detail,"approx","approx"),"recall_gain":outcome["approx"]["recall"]-max(outcome[c]["recall"] for c in ROOT_CHANNELS if c!="approx")})
            if key[1]=="qasper" and outcome["exact"]["recall"]>max(outcome[c]["recall"] for c in ROOT_CHANNELS if c!="exact"): taxonomy.append({"dataset":key[1],"example_id":key[2],"winner":"exact","cause":_taxonomy(detail,"exact","exact"),"recall_gain":outcome["exact"]["recall"]-max(outcome[c]["recall"] for c in ROOT_CHANNELS if c!="exact")})
            if key[1] in {"hotpotqa","2wikimultihopqa"} and outcome["bm25"]["recall"]>max(outcome[c]["recall"] for c in ROOT_CHANNELS if c!="bm25"): taxonomy.append({"dataset":key[1],"example_id":key[2],"winner":"bm25","cause":_taxonomy(detail,"bm25","bm25"),"recall_gain":outcome["bm25"]["recall"]-max(outcome[c]["recall"] for c in ROOT_CHANNELS if c!="bm25")})
        gist=set(selected["gist"]); lexical=set(selected["exact"])|set(selected["bm25"])|set(selected["approx"]); gold=detail["root_gold"]; hybrid=set(selected["hybrid"]); rrf=next(r for r in detail["root_rows"] if r["channel"]=="rrf")
        postmortem.append({"split":key[0],"dataset":key[1],"example_id":key[2],"unique_gold_from_lexical":len((lexical-gist)&gold),"unique_distractors_from_lexical":len((lexical-gist)-gold),"duplicate_budget_candidates":sum(len(set(selected[a])&set(selected[b])) for i,a in enumerate(ROOT_CHANNELS) for b in ROOT_CHANNELS[i+1:]),"hybrid_gold":len(hybrid&gold),"hybrid_distractors":len(hybrid-gold),"hybrid_precision":outcome["hybrid"]["precision"],"best_single_recall":max(outcome[c]["recall"] for c in ROOT_CHANNELS[:4]),"hybrid_recall":outcome["hybrid"]["recall"],"rrf_recall":rrf["recall"],"rrf_precision":rrf["precision"],"score_gap_mismatch":max(r["score_gap"] for r in outcome.values())-min(r["score_gap"] for r in outcome.values())})
        root_channel=validation_fixed[key[1]]; exact_detail=detail["successor_details"].get((root_channel,"exact_new_address")); hybrid_detail=detail["successor_details"].get((root_channel,"hybrid_state")); semantic_detail=detail["successor_details"].get((root_channel,"native_semantic"))
        if exact_detail and hybrid_detail and semantic_detail:
            root_ids=set(selected[root_channel]); qtokens=_pieces(tokenizer,detail["query_ids"]); root_tokens=set().union(*(_pieces(tokenizer,detail["token_index"].records[detail["index"].chunk_ids.index(i)].token_ids) for i in root_ids),set()); gold_tokens=set().union(*(_pieces(tokenizer,detail["token_index"].records[detail["index"].chunk_ids.index(i)].token_ids) for i in exact_detail["gold"]),set()); exposed=new_address_tokens(qtokens,root_tokens,gold_tokens); cutoff=.75*max(detail["token_index"].idf.values(),default=0); exposed={t for t in exposed if detail["token_index"].idf.get(t,0)>=cutoff}; candidate_counts={t:sum(t in row.normalized_tokens for row in detail["token_index"].records) for t in exposed}; gold_linked=bool(exposed); rank=int(next((i for i,v in enumerate(exact_detail["ranking"],1) if v in exact_detail["gold"]),len(exact_detail["ranking"])+1)); useful=useful_address(exposed=bool(exposed),gold_linked=gold_linked,successor_rank=rank,rank_limit=2*args.step_budget); gain=next(r["path_recall"] for r in detail["successor_rows"] if r["root_channel"]==root_channel and r["successor_channel"]=="hybrid_state")-next(r["path_recall"] for r in detail["successor_rows"] if r["root_channel"]==root_channel and r["successor_channel"]=="native_semantic"); useful_rows.append({"split":key[0],"dataset":key[1],"example_id":key[2],"root_channel":root_channel,"exposed":int(bool(exposed)),"gold_linked":int(gold_linked),"competitive_rank":int(rank<=2*args.step_budget),"useful_address":int(useful),"address_count":len(exposed),"minimum_candidate_count":min(candidate_counts.values(),default=0),"minimum_successor_rank":rank,"iterative_gain":gain})
            for token in exposed: address_conf.append({"split":key[0],"dataset":key[1],"example_id":key[2],"token":token,"idf":detail["token_index"].idf.get(token,0),"candidate_count":candidate_counts[token],"successor_rank":rank,"useful":int(useful)})
    if not any(row["dataset"] == "musique" for row in taxonomy):
        taxonomy.extend(_prior_musique_approx_wins(args, feature_rows))
    robustness=_robustness(tokenizer)
    root_successor_table=[]
    for dataset in DATASETS:
        roots=[r for r in root_summary if r["dataset"]==dataset]; successors=[r for r in successor_summary if r["dataset"]==dataset]; br=_best(roots,"channel"); bs=_best(successors,"successor_channel") if successors else None; root_successor_table.append({"dataset":dataset,"best_root_channel":br["channel"],"best_successor_channel":bs["successor_channel"] if bs else "not_defined","same_representation":int(bs is not None and br["channel"] in bs["successor_channel"]),"root_recall":br["recall"],"root_precision":br["precision"],"successor_recall":bs["recall"] if bs else 0,"successor_precision":bs["precision"] if bs else 0,"root_comparisons":br["comparisons"],"successor_comparisons":bs["comparisons"] if bs else 0})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    artifacts={"root_channel_results.csv":root_rows,"successor_channel_results.csv":successor_rows,"channel_transition_matrix.csv":transitions,"channel_true_oracle_headroom.csv":headroom,"successor_true_oracle_headroom.csv":successor_headroom,"validation_instability.csv":headroom,"channel_disagreement_features.csv":disagreement,"selector_observable_features.csv":feature_rows,"selector_results.csv":selector,"static_hybrid_postmortem.csv":postmortem,"iterative_useful_address.csv":useful_rows,"address_confidence.csv":address_conf,"wrong_reference_robustness_extended.csv":robustness,"root_successor_summary.csv":root_successor_table,"musique_approx_win_analysis.csv":[r for r in taxonomy if r["dataset"]=="musique"],"qasper_exact_win_analysis.csv":[r for r in taxonomy if r["dataset"]=="qasper"],"bm25_win_analysis.csv":[r for r in taxonomy if r["dataset"] in {"hotpotqa","2wikimultihopqa"}]}
    for name,rows in artifacts.items(): _write_csv(args.output_dir/name,rows)
    action_spec=_search_method_action_spec()
    (args.output_dir/"search_method_action_spec.json").write_text(json.dumps(action_spec,indent=2,sort_keys=True),encoding="utf-8")
    findings_path=args.output_dir.parent/"channel_geometry"/"paper2_6_findings.json"; findings=json.loads(findings_path.read_text(encoding="utf-8")); findings["channel_selection_iteration"]={"step_budget":args.step_budget,"total_budget":2*args.step_budget,"root_successor_summary":root_successor_table,"heldout_fixed":heldout_fixed,"validation_fixed":validation_fixed,"successor_validation_fixed":validation_successor,"successor_heldout_fixed":heldout_successor,"mean_true_selection_headroom":{d:mean(r["selection_headroom"] for r in headroom if r["dataset"]==d) for d in DATASETS},"mean_validation_instability":{d:mean(r["validation_instability"] for r in headroom if r["dataset"]==d) for d in DATASETS},"mean_successor_selection_headroom":{d:mean(r["selection_headroom"] for r in successor_headroom if r["dataset"]==d) for d in DATASETS},"mean_successor_validation_instability":{d:mean(r["validation_instability"] for r in successor_headroom if r["dataset"]==d) for d in DATASETS},"linear_selector_validation_loss":linear_loss,"linear_selector_features":linear_features,"generation_performed":False,"materialization_performed":False}; findings_path.write_text(json.dumps(findings,indent=2,sort_keys=True),encoding="utf-8")
    audit=["# Paper 2.6 channel-selection claim audit","","- Root and successor search each request two chunks; total conceptual budget remains four.","- Root and successor precision/recall are reported separately.","- True adaptive headroom is separated from validation-selection instability.","- The linear selector trains only on validation identities and observable features.","- Gold evidence geometry appears only in explanatory/taxonomy artifacts.","- RRF is the only additional fusion control.","- UsefulAddress requires exposure, a gold link, and successor rank <= 4.","- Costs are exhaustive Python research measurements, not serving claims.","- No native K/V is materialized and no generation/output metric is claimed."]
    (args.output_dir/"claim_audit.md").write_text("\n".join(audit)+"\n",encoding="utf-8")
    _plots(args.output_dir,root_summary,successor_summary,transitions,headroom,disagreement,taxonomy,useful_rows,postmortem,selector)
    return findings["channel_selection_iteration"]


def parse_args():
    parser=argparse.ArgumentParser(); parser.add_argument("--step-budget",type=int,default=2); parser.add_argument("--seed",type=int,default=20260811); parser.add_argument("--local-files-only",action="store_true"); parser.add_argument("--postprocess-only",action="store_true"); parser.add_argument("--cache-dir",type=Path,default=ROOT/"data/.hf_cache"); parser.add_argument("--paper2-feature-dir",type=Path,default=ROOT/"docs/papers/shared/results/paper2_hf/routing/learned_adapter"); natural=ROOT/"docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/natural_graph_features.pt"; data=ROOT/"data/.paper2_5_datasets"; parser.add_argument("--natural-features",type=Path,default=natural); parser.add_argument("--musique-dev",type=Path,default=data/"musique/data/musique_ans_v1.0_dev.jsonl"); parser.add_argument("--twowiki-dev",type=Path,default=data/"2wiki/dev.json"); parser.add_argument("--output-dir",type=Path,default=ROOT/"docs/papers/shared/results/paper2_6_hybrid_pra/channel_selection"); return parser.parse_args()


if __name__=="__main__":
    parsed = parse_args()
    print(json.dumps(
        _postprocess_existing(parsed) if parsed.postprocess_only else run(parsed),
        indent=2,
    ))
