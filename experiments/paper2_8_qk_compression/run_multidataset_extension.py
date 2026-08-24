"""Run the frozen Paper 2.8 transfer study on 2Wiki and MuSiQue."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_7_query_graph.helpers import file_sha256, write_csv, write_json
from experiments.paper2_8_qk_compression.run_gated_study import (
    MODEL_ID,
    MODEL_REVISION,
    SEEDS,
    _bootstrap,
    _case_tensors,
    _project_native_queries,
    _score_compact,
)
from experiments.paper2_8_qk_compression.run_low_rank_frontier import _fit_direct_router
from experiments.paper2_8_qk_compression.run_query_conditioned_study import _query_feature
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.hybrid_discovery import HybridDiscoveryPolicy, TokenNativeIndex
from pra_hf.multihop_routing_data import load_multihop_routing_examples
from pra_hf.qk_compression import (
    QueryConditionedLandmarkSelector,
    farthest_first_indices,
    gather_landmarks,
    kmeans_centroids,
    low_rank_response_scores,
    masked_mean_keys,
    response_metrics,
    routing_metrics,
    stable_topk_indices,
)


RESULT_ROOT = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"
OUTPUT_ROOT = RESULT_ROOT / "multi_dataset"
DATASET_DIR = {"2wikimultihopqa": "2wiki", "musique": "musique"}
DATASET_LABEL = {"2wikimultihopqa": "2WikiMultiHopQA", "musique": "MuSiQue"}
RANKS = (8, 16)
PRIMARY_BUDGET = 4
FUSION_WEIGHTS = (0.25, 0.5, 0.75)
LEXICAL_CONDITIONS = ("exact", "bm25", "approximate", "inherited_hybrid")


def _condition_label(condition: str) -> str:
    labels = {
        "native_mean": "native\nmean",
        "exact": "exact",
        "bm25": "BM25",
        "approximate": "approx.",
        "rank16_zero_shot": "r16\nzero-shot",
        "rank16_retrained": "r16\nretrained",
        "rank16_mixed": "r16\nmixed",
        "rank8_centroid8_retrained": "r8/c8\nretrained",
        "lexical_lowrank_retrained": "lexical + r16\nretrained",
    }
    return labels.get(condition, condition.replace("_", " "))


def _load_features(root: Path, dataset: str, split: str) -> list[dict]:
    path = root / DATASET_DIR[dataset] / f"native_qk_features_{split}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _key_scale(features: list[dict]) -> float:
    square_sum = 0.0
    count = 0
    for feature in features:
        values = feature["local_pre_key"].float()[feature["local_token_mask"]]
        square_sum += float(values.square().sum())
        count += values.numel()
    return math.sqrt(square_sum / max(count, 1))


def _training_cases(features: list[dict], scale: float, device: torch.device) -> list[dict]:
    cases = []
    for feature in features:
        query, keys, mask, positives = _case_tensors(feature, device)
        teacher = _score_compact(
            query, keys, mask, function="top_r_mean", head_reduction="mean"
        )
        cases.append(
            {
                "keys": feature["local_pre_key"].float().flatten(2) / scale,
                "query": _query_feature(query[0]).cpu(),
                "token_mask": feature["local_token_mask"],
                "positives": positives.cpu(),
                "teacher": teacher.cpu(),
            }
        )
    return cases


def _balanced(features_by_dataset: dict[str, list[dict]]) -> list[dict]:
    """Cycle smaller inherited cohorts to equal dataset sampling weight."""

    target = max(len(group) for group in features_by_dataset.values())
    output = []
    for dataset in sorted(features_by_dataset):
        group = features_by_dataset[dataset]
        output.extend(group[index % len(group)] for index in range(target))
    return output


def _checkpoint_payload(selector, *, regime, dataset, rank, seed, scale, seconds, args):
    return {
        "state_dict": {name: value.detach().cpu() for name, value in selector.state_dict().items()},
        "regime": regime,
        "dataset": dataset,
        "rank": rank,
        "seed": seed,
        "objective": "combined",
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "native_key_rms_scale": scale,
        "train_seconds": seconds,
    }


def _selector_from_checkpoint(checkpoint: dict, device: torch.device):
    selector = QueryConditionedLandmarkSelector(
        2048,
        feature_width=1024,
        rank=int(checkpoint["rank"]),
        use_salience=False,
        use_interaction=True,
    ).to(device)
    selector.load_state_dict(checkpoint["state_dict"])
    return selector.eval()


def _fit_models(
    features: list[dict],
    *,
    regime: str,
    dataset: str,
    output_dir: Path,
    device: torch.device,
    args,
):
    scale = _key_scale(features)
    cases = _training_cases(features, scale, device)
    output = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank in RANKS:
        for seed in SEEDS:
            path = output_dir / f"{regime}_{dataset}_r{rank}_seed{seed}.pt"
            if args.resume and path.exists():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            else:
                selector, _, seconds = _fit_direct_router(
                    cases,
                    rank=rank,
                    seed=seed,
                    steps=args.steps,
                    learning_rate=args.learning_rate,
                    device=device,
                )
                checkpoint = _checkpoint_payload(
                    selector,
                    regime=regime,
                    dataset=dataset,
                    rank=rank,
                    seed=seed,
                    scale=scale,
                    seconds=seconds,
                    args=args,
                )
                torch.save(checkpoint, path)
            output[(rank, seed)] = (_selector_from_checkpoint(checkpoint, device), checkpoint)
            print(f"[projection {regime} {dataset} r{rank} s{seed}]", flush=True)
    return output


def _load_zero_shot(device: torch.device):
    output = {}
    root = RESULT_ROOT / "low_rank_frontier/checkpoints"
    for rank in RANKS:
        for seed in SEEDS:
            checkpoint = torch.load(
                root / f"direct_lowrank_r{rank}_seed{seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            output[(rank, seed)] = (_selector_from_checkpoint(checkpoint, device), checkpoint)
    return output


def _semantic_unit(scores: torch.Tensor) -> torch.Tensor:
    low, high = scores.min(), scores.max()
    if float(high - low) <= 1e-12:
        return torch.zeros_like(scores)
    return 2.0 * (scores - low) / (high - low) - 1.0


def _token_index(tokenizer, example, feature) -> TokenNativeIndex:
    entry = SimpleNamespace(
        uri=f"benchmark://{feature['dataset']}/{feature['example_id']}",
        text=example.source,
        metadata={},
    )
    records = [
        (
            entry,
            SimpleNamespace(
                chunk_id=f"local-{index}", token_start=int(start), token_end=int(end)
            ),
        )
        for index, (start, end) in enumerate(feature["local_spans"])
    ]
    return TokenNativeIndex.from_gist_index(
        SimpleNamespace(records=records, layer_id=27), tokenizer
    )


def _lexical_scores(tokenizer, example, feature, semantic_scores):
    index = _token_index(tokenizer, example, feature)
    query_ids = tokenizer(example.question, add_special_tokens=False).input_ids
    output = {}
    modes = {
        "exact": "token_exact",
        "bm25": "bm25",
        "approximate": "token_approx",
        "inherited_hybrid": "iterative_hybrid",
    }
    for name, mode in modes.items():
        candidates = index.score(
            query_ids,
            _semantic_unit(semantic_scores),
            tokenizer,
            HybridDiscoveryPolicy(mode=mode),
            hop=1,
            parent_id="query",
        )
        output[name] = torch.tensor(
            [candidate.selected_score for candidate in candidates], dtype=torch.float32
        )
    return output


@torch.no_grad()
def _lowrank_scores(selector, checkpoint, feature, *, centroids: int | None, device):
    keys = feature["local_pre_key"].to(device).float().flatten(2)
    mask = feature["local_token_mask"].to(device)
    started = time.perf_counter()
    projected = selector.feature_projection(
        keys / float(checkpoint["native_key_rms_scale"])
    )
    if centroids:
        projected, mask = kmeans_centroids(projected, mask, centroids)
    query = selector.query_projection(
        _query_feature(feature["query_pre_query"]).to(device)
    )
    scores = low_rank_response_scores(
        query, projected, mask, function="top_r_mean", top_r=4
    )[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    milliseconds = 1000.0 * (time.perf_counter() - started)
    return scores.cpu(), milliseconds


def _normalize(scores: torch.Tensor) -> torch.Tensor:
    extent = scores.max() - scores.min()
    return (scores - scores.min()) / extent if float(extent) > 1e-12 else torch.zeros_like(scores)


def _secondary_budget(features: list[dict]) -> int:
    counts = sorted(int(feature["local_positive_mask"].sum()) for feature in features)
    percentile = counts[min(math.ceil(0.9 * len(counts)) - 1, len(counts) - 1)]
    return max(PRIMARY_BUDGET, min(8, percentile))


@torch.no_grad()
def _score_case(feature, example, tokenizer, models, device):
    query, keys, mask, _ = _case_tensors(feature, device)
    teacher = _score_compact(
        query, keys, mask, function="top_r_mean", head_reduction="mean"
    ).cpu()
    mean_keys = masked_mean_keys(keys, mask)
    mean_scores = _score_compact(
        query,
        mean_keys,
        torch.ones(len(keys), 1, dtype=torch.bool, device=device),
        function="top_r_mean",
        head_reduction="mean",
    ).cpu()
    farthest, farthest_mask = gather_landmarks(
        keys, farthest_first_indices(keys, mask, 2)
    )
    farthest_scores = _score_compact(
        query,
        farthest,
        farthest_mask,
        function="top_r_mean",
        head_reduction="mean",
    ).cpu()
    scores = {
        ("native_mean", -1): (mean_scores, 4096, 0.0, "control"),
        ("native_farthest_m2", -1): (farthest_scores, 8192, 0.0, "control"),
    }
    started = time.perf_counter()
    lexical = _lexical_scores(tokenizer, example, feature, mean_scores)
    lexical_ms = 1000.0 * (time.perf_counter() - started)
    for name, values in lexical.items():
        scores[(name, -1)] = (values, 0, lexical_ms / len(lexical), "control")
    for regime, selectors in models.items():
        for rank in RANKS:
            seed_scores = []
            for seed in SEEDS:
                selector, checkpoint = selectors[(rank, seed)]
                values, milliseconds = _lowrank_scores(
                    selector,
                    checkpoint,
                    feature,
                    centroids=None,
                    device=device,
                )
                condition = f"rank{rank}_{regime}"
                scores[(condition, seed)] = (
                    values,
                    32 * rank * 4,
                    milliseconds,
                    "seed",
                )
                seed_scores.append(values)
            condition = f"rank{rank}_{regime}"
            scores[(condition, -1)] = (
                torch.stack(seed_scores).mean(0),
                32 * rank * 4,
                0.0,
                "ensemble",
            )
        seed_scores = []
        for seed in SEEDS:
            selector, checkpoint = selectors[(8, seed)]
            values, milliseconds = _lowrank_scores(
                selector, checkpoint, feature, centroids=8, device=device
            )
            condition = f"rank8_centroid8_{regime}"
            scores[(condition, seed)] = (values, 8 * 8 * 4, milliseconds, "seed")
            seed_scores.append(values)
        scores[(f"rank8_centroid8_{regime}", -1)] = (
            torch.stack(seed_scores).mean(0),
            8 * 8 * 4,
            0.0,
            "ensemble",
        )
    return scores, teacher


def _select_validation_policy(score_cache, features_by_dataset, budgets):
    policies = {}
    for dataset, features in features_by_dataset.items():
        budget = budgets[dataset]
        best_lexical = max(
            LEXICAL_CONDITIONS,
            key=lambda condition: (
                statistics.fmean(
                    routing_metrics(
                        stable_topk_indices(
                            score_cache[feature["example_id"]][0][(condition, -1)][0],
                            budget,
                        ).tolist(),
                        feature["local_positive_mask"],
                        budget=budget,
                    )["evidence_recall"]
                    for feature in features
                ),
                condition,
            ),
        )
        policies[dataset] = {"best_lexical": best_lexical, "fusion": {}}
        for regime in ("zero_shot", "retrained", "mixed"):
            candidates = []
            for lexical in ("exact", "bm25", "approximate"):
                for lexical_weight in FUSION_WEIGHTS:
                    recalls, chains, mrrs = [], [], []
                    for feature in features:
                        values = score_cache[feature["example_id"]][0]
                        fused = lexical_weight * _normalize(values[(lexical, -1)][0]) + (
                            1.0 - lexical_weight
                        ) * _normalize(values[(f"rank16_{regime}", -1)][0])
                        route = routing_metrics(
                            stable_topk_indices(fused, budget).tolist(),
                            feature["local_positive_mask"],
                            budget=budget,
                        )
                        recalls.append(route["evidence_recall"])
                        chains.append(route["chain_completion"])
                        mrrs.append(route["mrr"])
                    candidates.append(
                        (
                            statistics.fmean(recalls),
                            statistics.fmean(chains),
                            statistics.fmean(mrrs),
                            lexical,
                            lexical_weight,
                        )
                    )
            selected = max(candidates)
            policies[dataset]["fusion"][regime] = {
                "lexical": selected[-2],
                "lexical_weight": selected[-1],
                "validation_recall": selected[0],
                "validation_chain_completion": selected[1],
            }
    return policies


def _add_derived_scores(scores, policy):
    best = policy["best_lexical"]
    scores[("best_lexical", -1)] = (*scores[(best, -1)][:3], "derived")
    for regime, config in policy["fusion"].items():
        lexical = scores[(config["lexical"], -1)][0]
        lowrank = scores[(f"rank16_{regime}", -1)][0]
        weight = float(config["lexical_weight"])
        fused = weight * _normalize(lexical) + (1.0 - weight) * _normalize(lowrank)
        scores[(f"lexical_lowrank_{regime}", -1)] = (fused, 32 * 16 * 4, 0.0, "fusion")


def _metric_row(feature, condition, seed, values, teacher, budget, index_bytes, latency, kind):
    selected = stable_topk_indices(values, budget).tolist()
    route = routing_metrics(selected, feature["local_positive_mask"], budget=budget)
    response = response_metrics(teacher, values)
    tokens = sum(int(feature["local_token_mask"][index].sum()) for index in selected)
    native_width = feature["local_pre_key"].shape[2] * feature["local_pre_key"].shape[3]
    return {
        "dataset": feature["dataset"],
        "split": feature["split"],
        "example_id": feature["example_id"],
        "condition": condition,
        "seed": seed,
        "row_kind": kind,
        "budget_chunks": budget,
        "candidate_chunks": len(values),
        "positive_chunks": int(feature["local_positive_mask"].sum()),
        "authored_evidence_items": int(feature["authored_evidence_items"]),
        "positive_chunk_ids": " ".join(
            map(
                str,
                torch.nonzero(
                    feature["local_positive_mask"], as_tuple=False
                ).flatten().tolist(),
            )
        ),
        "authored_chain_budget_feasible": int(feature["authored_evidence_items"])
        <= budget,
        "full_positive_span_budget_feasible": int(
            feature["local_positive_mask"].sum()
        )
        <= budget,
        "selected_chunks": " ".join(map(str, selected)),
        "index_bytes_per_chunk": index_bytes,
        "routing_setup_plus_score_ms": latency,
        "materialized_native_kv_tokens": tokens,
        "active_memory_fraction": tokens / max(int(feature["source_tokens"]), 1),
        "backing_native_kv_bytes": int(feature["source_tokens"]) * native_width * 2 * 2,
        "transfer_bytes": tokens * native_width * 2 * 2,
        **route,
        **{key: value for key, value in asdict(response).items() if key != "topk_overlap"},
        **{f"teacher_top{k}_overlap": value for k, value in response.topk_overlap.items()},
    }


def _summaries(rows):
    headline = [row for row in rows if row["row_kind"] != "seed"]
    grouped = defaultdict(list)
    for row in headline:
        grouped[(row["dataset"], row["split"], row["budget_chunks"], row["condition"])].append(row)
    metrics = (
        "evidence_recall",
        "evidence_precision",
        "any_evidence",
        "chain_completion",
        "exact_identity",
        "mrr",
        "teacher_top4_overlap",
        "spearman",
        "kl",
        "index_bytes_per_chunk",
        "routing_setup_plus_score_ms",
        "materialized_native_kv_tokens",
        "active_memory_fraction",
        "transfer_bytes",
    )
    summary = []
    for key, group in sorted(grouped.items()):
        summary.append(
            {
                "dataset": key[0],
                "split": key[1],
                "budget_chunks": key[2],
                "condition": key[3],
                "identities": len(group),
                **{
                    metric: statistics.fmean(float(row[metric]) for row in group)
                    for metric in metrics
                },
            }
        )
    seed_grouped = defaultdict(list)
    seed_rows = defaultdict(list)
    for row in rows:
        if row["row_kind"] == "seed" and row["split"] == "test":
            seed_grouped[(row["dataset"], row["budget_chunks"], row["condition"], row["seed"])].append(row)
            seed_rows[(row["dataset"], row["budget_chunks"], row["condition"])].append(row)
    per_seed = defaultdict(list)
    for (dataset, budget, condition, seed), group in seed_grouped.items():
        per_seed[(dataset, budget, condition)].append(
            (seed, statistics.fmean(float(row["evidence_recall"]) for row in group))
        )
    seed_summary = []
    for (dataset, budget, condition), values in sorted(per_seed.items()):
        scores = [value for _, value in values]
        seed_summary.append(
            {
                "dataset": dataset,
                "budget_chunks": budget,
                "condition": condition,
                "seeds": len(values),
                "mean": statistics.fmean(scores),
                "median": statistics.median(scores),
                "min": min(scores),
                "max": max(scores),
                "std": statistics.pstdev(scores),
                "topk_stability_jaccard": _mean_seed_selection_jaccard(
                    seed_rows[(dataset, budget, condition)]
                ),
            }
        )
    return summary, seed_summary


def _mean_seed_selection_jaccard(rows) -> float:
    """Average pairwise selected-set agreement across seeds and identities."""

    by_example = defaultdict(list)
    for row in rows:
        selected = frozenset(map(int, str(row["selected_chunks"]).split()))
        by_example[row["example_id"]].append(selected)
    similarities = []
    for selections in by_example.values():
        for left, right in combinations(selections, 2):
            similarities.append(len(left & right) / max(len(left | right), 1))
    return statistics.fmean(similarities) if similarities else 1.0


def _paired(rows, budgets):
    lookup = {
        (row["dataset"], row["budget_chunks"], row["condition"], row["example_id"]): float(row["evidence_recall"])
        for row in rows
        if row["split"] == "test" and row["row_kind"] != "seed"
    }
    comparisons = (
        ("rank16_zero_shot", "native_mean"),
        ("rank16_retrained", "native_mean"),
        ("rank16_mixed", "native_mean"),
        ("rank8_centroid8_retrained", "native_mean"),
        ("rank16_retrained", "best_lexical"),
        ("lexical_lowrank_retrained", "best_lexical"),
        ("lexical_lowrank_mixed", "best_lexical"),
    )
    output = []
    for dataset, dataset_budgets in budgets.items():
        ids = sorted({key[3] for key in lookup if key[0] == dataset})
        for budget in dataset_budgets:
            for left, right in comparisons:
                deltas = [
                    lookup[(dataset, budget, left, example_id)]
                    - lookup[(dataset, budget, right, example_id)]
                    for example_id in ids
                ]
                low, high = _bootstrap(deltas, 20260824 + budget)
                output.append(
                    {
                        "dataset": dataset,
                        "budget_chunks": budget,
                        "left": left,
                        "right": right,
                        "identities": len(deltas),
                        "mean_delta": statistics.fmean(deltas),
                        "ci95_low": low,
                        "ci95_high": high,
                        "better": sum(value > 0 for value in deltas),
                        "worse": sum(value < 0 for value in deltas),
                        "unchanged": sum(value == 0 for value in deltas),
                    }
                )
    return output


def _changed_selection(rows, policies):
    grouped = defaultdict(dict)
    for row in rows:
        if row["split"] == "test" and row["row_kind"] != "seed":
            grouped[(row["dataset"], row["budget_chunks"], row["example_id"])][row["condition"]] = row
    output = []
    for (dataset, budget, example_id), values in grouped.items():
        for regime in ("zero_shot", "retrained", "mixed"):
            lexical = values["best_lexical"]
            learned = values[f"rank16_{regime}"]
            lexical_ids = set(map(int, lexical["selected_chunks"].split()))
            learned_ids = set(map(int, learned["selected_chunks"].split()))
            positive_ids = set(map(int, lexical["positive_chunk_ids"].split()))
            output.append(
                {
                    "dataset": dataset,
                    "budget_chunks": budget,
                    "example_id": example_id,
                    "regime": regime,
                    "best_lexical_mode": policies[dataset]["best_lexical"],
                    "selection_jaccard": len(lexical_ids & learned_ids) / max(len(lexical_ids | learned_ids), 1),
                    "chunks_unique_to_lowrank": len(learned_ids - lexical_ids),
                    "evidence_unique_to_lowrank": len(
                        (learned_ids & positive_ids) - lexical_ids
                    ),
                    "evidence_unique_to_lexical": len(
                        (lexical_ids & positive_ids) - learned_ids
                    ),
                    "lowrank_recall_delta": float(learned["evidence_recall"]) - float(lexical["evidence_recall"]),
                }
            )
    return output


def _plots(summary, output_root):
    test = [row for row in summary if row["split"] == "test" and int(row["budget_chunks"]) == 4]
    conditions = [
        "native_mean",
        "exact",
        "bm25",
        "approximate",
        "rank16_zero_shot",
        "rank16_retrained",
        "rank16_mixed",
        "rank8_centroid8_retrained",
        "lexical_lowrank_retrained",
    ]
    lookup = {(row["dataset"], row["condition"]): row for row in test}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, dataset in zip(axes, sorted(DATASET_DIR)):
        values = [float(lookup[(dataset, condition)]["evidence_recall"]) for condition in conditions]
        axis.bar(range(len(conditions)), values, color="#2878b5")
        axis.set_xticks(range(len(conditions)), [_condition_label(value) for value in conditions], fontsize=7)
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_ylabel("Evidence recall@4")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_root / "recall_by_channel.pdf")
    figure.savefig(output_root / "recall_by_channel.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, dataset in zip(axes, sorted(DATASET_DIR)):
        values = [float(lookup[(dataset, condition)]["chain_completion"]) for condition in conditions]
        axis.bar(range(len(conditions)), values, color="#5b8e7d")
        axis.set_xticks(range(len(conditions)), [_condition_label(value) for value in conditions], fontsize=7)
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_ylabel("Complete evidence@4")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_root / "complete_evidence_by_channel.pdf")
    figure.savefig(output_root / "complete_evidence_by_channel.png", dpi=180)
    plt.close(figure)

    points = [
        row
        for row in test
        if row["condition"] in {"native_mean", "rank16_retrained", "rank8_retrained", "rank8_centroid8_retrained"}
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, sorted(DATASET_DIR)):
        for row in [value for value in points if value["dataset"] == dataset]:
            x = float(row["index_bytes_per_chunk"])
            y = float(row["evidence_recall"])
            axis.scatter(x, y, s=45)
            axis.annotate(row["condition"].replace("_retrained", ""), (x, y), fontsize=7)
        axis.set_xscale("log")
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_xlabel("Routing-index bytes/chunk")
        axis.set_ylabel("Evidence recall@4")
        axis.grid(alpha=0.25)
    figure.savefig(output_root / "recall_vs_index_bytes.pdf")
    figure.savefig(output_root / "recall_vs_index_bytes.png", dpi=180)
    plt.close(figure)

    transfer = ("rank16_zero_shot", "rank16_retrained", "rank16_mixed")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, sorted(DATASET_DIR)):
        axis.bar(
            range(len(transfer)),
            [float(lookup[(dataset, condition)]["evidence_recall"]) for condition in transfer],
            color=("#9e9e9e", "#2878b5", "#d65f5f"),
        )
        axis.set_xticks(range(len(transfer)), ("zero-shot", "retrained", "mixed"))
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_ylabel("Rank-16 evidence recall@4")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_root / "transfer_regimes.pdf")
    figure.savefig(output_root / "transfer_regimes.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, sorted(DATASET_DIR)):
        group = [
            row
            for row in test
            if row["dataset"] == dataset and row["condition"].startswith("rank")
        ]
        axis.scatter(
            [float(row["teacher_top4_overlap"]) for row in group],
            [float(row["evidence_recall"]) for row in group],
            color="#7b4f9d",
        )
        for row in group:
            axis.annotate(
                row["condition"].replace("_retrained", "-R").replace("_zero_shot", "-Z").replace("_mixed", "-M"),
                (float(row["teacher_top4_overlap"]), float(row["evidence_recall"])),
                fontsize=6,
            )
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_xlabel("Teacher top-4 overlap")
        axis.set_ylabel("Evidence recall@4")
        axis.grid(alpha=0.25)
    figure.savefig(output_root / "teacher_overlap_vs_recall.pdf")
    figure.savefig(output_root / "teacher_overlap_vs_recall.png", dpi=180)
    plt.close(figure)


def _changed_selection_plot(rows, output_root):
    datasets = sorted(DATASET_DIR)
    regimes = ("zero_shot", "retrained", "mixed")
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        group = [
            row
            for row in rows
            if row["dataset"] == dataset and int(row["budget_chunks"]) == 4
        ]
        lowrank = [
            statistics.fmean(
                float(row["evidence_unique_to_lowrank"])
                for row in group
                if row["regime"] == regime
            )
            for regime in regimes
        ]
        lexical = [
            statistics.fmean(
                float(row["evidence_unique_to_lexical"])
                for row in group
                if row["regime"] == regime
            )
            for regime in regimes
        ]
        x = list(range(len(regimes)))
        axis.bar([value - 0.18 for value in x], lowrank, 0.36, label="low-rank only", color="#2878b5")
        axis.bar([value + 0.18 for value in x], lexical, 0.36, label="lexical only", color="#d65f5f")
        axis.set_xticks(x, ("zero-shot", "retrained", "mixed"))
        axis.set_title(DATASET_LABEL[dataset])
        axis.set_ylabel("Unique evidence chunks/example")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output_root / "unique_evidence_by_channel.pdf")
    figure.savefig(output_root / "unique_evidence_by_channel.png", dpi=180)
    plt.close(figure)


def _gate_decisions(paired, summary):
    primary = [row for row in paired if int(row["budget_chunks"]) == PRIMARY_BUDGET]
    m1_rows = [
        row
        for row in primary
        if row["left"] in {"rank16_zero_shot", "rank16_retrained", "rank16_mixed"}
        and row["right"] == "native_mean"
    ]
    m1 = any(float(row["ci95_low"]) > 0 for row in m1_rows)
    m3_rows = [
        row
        for row in primary
        if row["left"] in {"lexical_lowrank_retrained", "lexical_lowrank_mixed"}
    ]
    m3 = any(float(row["ci95_low"]) > 0 for row in m3_rows)
    lookup = {
        (row["dataset"], row["condition"]): float(row["evidence_recall"])
        for row in summary
        if row["split"] == "test" and int(row["budget_chunks"]) == PRIMARY_BUDGET
    }
    retention = {}
    for dataset in DATASET_DIR:
        rank16_gain = lookup[(dataset, "rank16_retrained")] - lookup[(dataset, "native_mean")]
        compact_gain = lookup[(dataset, "rank8_centroid8_retrained")] - lookup[(dataset, "native_mean")]
        retention[dataset] = compact_gain / rank16_gain if rank16_gain > 0 else None
    m2 = any(value is not None and value >= 0.70 for value in retention.values())
    return {
        "M0_feature_parity": True,
        "M1_native_lowrank_transfer": m1,
        "M2_compression_point": m2,
        "M2_gain_retention": retention,
        "M3_channel_complementarity": m3,
        "M4_multi_dataset_generality": m1,
        "M5_generation_eligible": m1 or m3,
    }


def run(args):
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    examples = load_multihop_routing_examples(args.annotations, args.twowiki_dev, args.musique_dev)
    example_lookup = {(example.dataset, example.example_id): example for example in examples}
    validation = {dataset: _load_features(args.output_root, dataset, "validation") for dataset in DATASET_DIR}
    test = {dataset: _load_features(args.output_root, dataset, "test") for dataset in DATASET_DIR}
    old_validation = torch.load(
        RESULT_ROOT / "native_qk_features_validation.pt", map_location="cpu", weights_only=False
    )
    if any("query_pre_query" not in feature for feature in old_validation):
        _project_native_queries({"inherited_validation": old_validation}, device)
    old_groups = {
        dataset: [row for row in old_validation if row["dataset"] == dataset]
        for dataset in ("hotpotqa", "qasper")
    }
    models_by_dataset = {}
    for dataset in DATASET_DIR:
        checkpoint_dir = args.output_root / DATASET_DIR[dataset] / "checkpoints"
        models_by_dataset[dataset] = {
            "zero_shot": _load_zero_shot(device),
            "retrained": _fit_models(
                validation[dataset],
                regime="retrained",
                dataset=dataset,
                output_dir=checkpoint_dir,
                device=device,
                args=args,
            ),
        }
    mixed_features = _balanced({**old_groups, **validation})
    mixed_models = _fit_models(
        mixed_features,
        regime="mixed",
        dataset="four_dataset",
        output_dir=args.output_root / "mixed_projection/checkpoints",
        device=device,
        args=args,
    )
    for dataset in DATASET_DIR:
        models_by_dataset[dataset]["mixed"] = mixed_models

    validation_scores = {}
    for dataset, features in validation.items():
        for feature in features:
            example = example_lookup[(dataset, feature["example_id"])]
            validation_scores[feature["example_id"]] = _score_case(
                feature, example, tokenizer, models_by_dataset[dataset], device
            )
    secondary = {dataset: _secondary_budget(features) for dataset, features in validation.items()}
    calibration_budgets = secondary
    policies = _select_validation_policy(validation_scores, validation, calibration_budgets)
    dataset_budgets = {
        dataset: sorted({PRIMARY_BUDGET, secondary[dataset]}) for dataset in DATASET_DIR
    }
    rows = []
    for split, groups in (("validation", validation), ("test", test)):
        for dataset, features in groups.items():
            for index, feature in enumerate(features, start=1):
                feature["split"] = split
                if split == "validation":
                    scores, teacher = validation_scores[feature["example_id"]]
                else:
                    example = example_lookup[(dataset, feature["example_id"])]
                    scores, teacher = _score_case(
                        feature, example, tokenizer, models_by_dataset[dataset], device
                    )
                _add_derived_scores(scores, policies[dataset])
                for budget in dataset_budgets[dataset]:
                    for (condition, seed), (values, index_bytes, latency, kind) in scores.items():
                        rows.append(
                            _metric_row(
                                feature,
                                condition,
                                seed,
                                values,
                                teacher,
                                budget,
                                index_bytes,
                                latency,
                                kind,
                            )
                        )
                print(f"[evaluate {split} {dataset} {index}/{len(features)}]", flush=True)
    summary, seed_summary = _summaries(rows)
    paired = _paired(rows, dataset_budgets)
    changed = _changed_selection(rows, policies)
    gates = _gate_decisions(paired, summary)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "per_example.csv", rows)
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "seed_summary.csv", seed_summary)
    write_csv(args.output_root / "paired_effects.csv", paired)
    write_csv(args.output_root / "changed_selection.csv", changed)
    write_json(args.output_root / "validation_policy.json", policies)
    _plots(summary, args.output_root)
    _changed_selection_plot(changed, args.output_root)
    for dataset, directory in DATASET_DIR.items():
        dataset_dir = args.output_root / directory
        write_csv(dataset_dir / "per_example.csv", [row for row in rows if row["dataset"] == dataset])
        write_csv(dataset_dir / "summary.csv", [row for row in summary if row["dataset"] == dataset])
        write_csv(dataset_dir / "seed_summary.csv", [row for row in seed_summary if row["dataset"] == dataset])
        write_csv(dataset_dir / "paired_effects.csv", [row for row in paired if row["dataset"] == dataset])
        write_csv(dataset_dir / "changed_selection.csv", [row for row in changed if row["dataset"] == dataset])
    write_csv(
        args.output_root / "mixed_projection/summary.csv",
        [row for row in summary if "mixed" in row["condition"]],
    )
    write_csv(
        args.output_root / "hybrid_channel/summary.csv",
        [
            row
            for row in summary
            if row["condition"] in LEXICAL_CONDITIONS
            or row["condition"].startswith("lexical_lowrank")
        ],
    )
    write_csv(args.output_root / "hybrid_channel/changed_selection.csv", changed)
    feature_manifest = args.output_root / "feature_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_layer": 27,
        "backbone_frozen": True,
        "ranks": list(RANKS),
        "seeds": list(SEEDS),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "primary_budget_chunks": PRIMARY_BUDGET,
        "secondary_budget_chunks": secondary,
        "validation_policy": policies,
        "feature_manifest_sha256": file_sha256(feature_manifest),
        "old_validation_feature_sha256": file_sha256(RESULT_ROOT / "native_qk_features_validation.pt"),
        "test_used_for_selection": False,
        "mixed_projection_balanced_examples_per_dataset": 20,
        "native_kv_unchanged": True,
        "gates": gates,
    }
    write_json(args.output_root / "extension_manifest.json", manifest)
    return manifest


def parse_args():
    inherited = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl"
    )
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "2wiki/dev.json")
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=inherited / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
