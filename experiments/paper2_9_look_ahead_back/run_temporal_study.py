"""Run the gated Paper 2.9 temporal-query study on frozen Paper 2.8 indexes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_9_look_ahead_back.precompute_temporal_queries import (
    DATASETS,
    MODEL_ID,
    MODEL_REVISION,
    load_questions,
    load_source_features,
)
from experiments.paper2_8_qk_compression.run_multidataset_extension import (
    _lexical_scores,
)
from pra_hf.qk_compression import (
    kmeans_centroids,
    masked_mean_keys,
    qk_response_scores,
    routing_metrics,
    stable_topk_indices,
)
from pra_hf.temporal_routing import (
    delayed_commit_index,
    interaction_contrast,
    recency_weights,
    score_diagnostics,
    selection_churn,
    stride_update_mask,
    temporal_chunk_scores,
)


SEEDS = (11, 23, 37, 53, 71)
WINDOWS = (1, 2, 4, 8, 16)
REDUCERS = ("current", "mean", "recency", "late_max", "late_top_mean")
MEMORIES = ("native_mean", "rank16", "rank8_centroid8")
DELAYS = (1, 2, 4, 8)
STRIDES = (1, 2, 4, 8)
BUDGET = 4
RESULT_28 = ROOT / "docs/papers/shared/results/paper2_8_qk_compression"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize(scores: torch.Tensor) -> torch.Tensor:
    extent = scores.max() - scores.min()
    if float(extent) <= 1e-12:
        return torch.zeros_like(scores)
    return (scores - scores.min()) / extent


def source_rows(root: Path, dataset: str, split: str) -> list[dict]:
    groups = load_source_features(root)
    return groups[(dataset, split)]


def temporal_rows(root: Path, dataset: str, split: str) -> list[dict]:
    path = root / dataset / f"temporal_query_features_{split}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def aligned_rows(source: list[dict], temporal: list[dict]):
    indexed = {row["example_id"]: row for row in temporal}
    if len(indexed) != len(temporal):
        raise ValueError("Temporal cache contains duplicate identities.")
    output = []
    for row in source:
        identity = row["example_id"]
        if identity not in indexed:
            raise ValueError(f"Missing temporal row for {identity}")
        output.append((row, indexed[identity]))
    return output


def checkpoint_paths(dataset: str, rank: int) -> list[Path]:
    if dataset in {"hotpotqa", "qasper"}:
        directory = RESULT_28 / "low_rank_frontier/checkpoints"
        return [directory / f"direct_lowrank_r{rank}_seed{seed}.pt" for seed in SEEDS]
    directory_name = "2wiki" if dataset == "2wikimultihopqa" else "musique"
    directory = RESULT_28 / "multi_dataset" / directory_name / "checkpoints"
    return [
        directory / f"retrained_{dataset}_r{rank}_seed{seed}.pt" for seed in SEEDS
    ]


def load_checkpoints(dataset: str, rank: int, device: torch.device) -> list[dict]:
    output = []
    for path in checkpoint_paths(dataset, rank):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]
        output.append(
            {
                "seed": int(checkpoint["seed"]),
                "rank": int(checkpoint["rank"]),
                "scale": float(checkpoint["native_key_rms_scale"]),
                "wq": state["query_projection.weight"].to(device).float(),
                "wk": state["feature_projection.weight"].to(device).float(),
            }
        )
    return output


def projected_memory(
    source: dict,
    checkpoints: list[dict],
    *,
    centroids: int | None,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    keys = source["local_pre_key"].to(device).float().flatten(2)
    mask = source["local_token_mask"].to(device)
    output = []
    for checkpoint in checkpoints:
        values = F.linear(keys / checkpoint["scale"], checkpoint["wk"])
        row_mask = mask
        if centroids is not None:
            values, row_mask = kmeans_centroids(values, row_mask, centroids)
        output.append((values, row_mask))
    return output


def native_temporal_scores(
    queries: torch.Tensor,
    mean_keys: torch.Tensor,
    mean_mask: torch.Tensor,
    *,
    reducer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if reducer == "current":
        pooled = queries[-1:]
    elif reducer == "mean":
        pooled = queries.mean(dim=0, keepdim=True)
    elif reducer == "recency":
        weights = recency_weights(len(queries), device=queries.device).to(queries.dtype)
        pooled = torch.einsum("q,qhd->hd", weights, queries).unsqueeze(0)
    else:
        per_query = qk_response_scores(
            queries,
            mean_keys,
            mean_mask,
            function="top_r_mean",
            top_r=4,
            head_reduction="mean",
        )
        weights = recency_weights(len(queries), device=queries.device).to(per_query.dtype)
        scores = torch.einsum("q,qc->c", weights, per_query)
        return scores, per_query.unsqueeze(-1)
    scores = qk_response_scores(
        pooled,
        mean_keys,
        mean_mask,
        function="top_r_mean",
        top_r=4,
        head_reduction="mean",
    )[0]
    interactions = qk_response_scores(
        queries,
        mean_keys,
        mean_mask,
        function="top_r_mean",
        top_r=4,
        head_reduction="mean",
    ).unsqueeze(-1)
    return scores, interactions


class ExampleScorer:
    """Reuse one example's frozen memory projections across temporal policies."""

    def __init__(self, source, temporal, checkpoints, device):
        self.source = source
        self.temporal = temporal
        self.device = device
        self.checkpoints = checkpoints
        keys = source["local_pre_key"].to(device).float()
        mask = source["local_token_mask"].to(device)
        self.mean_keys = masked_mean_keys(keys, mask)
        self.mean_mask = torch.ones(
            len(keys), 1, dtype=torch.bool, device=device
        )
        self.memory = {}

    def routing_memory(self, memory: str):
        """Build each frozen routing index only when a policy first requests it."""
        if memory not in self.memory:
            rank = 16 if memory == "rank16" else 8
            self.memory[memory] = projected_memory(
                self.source,
                self.checkpoints[rank],
                centroids=None if memory == "rank16" else 8,
                device=self.device,
            )
        return self.memory[memory]

    def query_states(self, layer: int) -> torch.Tensor:
        return self.temporal["pre_query_by_layer"][str(layer)].to(
            self.device
        ).float()

    def score(
        self,
        memory: str,
        *,
        layer: int,
        start: int,
        stop: int,
        reducer: str,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        queries = self.query_states(layer)[start:stop]
        if len(queries) == 0:
            raise ValueError("Temporal routing window is empty.")
        if memory == "native_mean":
            scores, interactions = native_temporal_scores(
                queries, self.mean_keys, self.mean_mask, reducer=reducer
            )
            self.last_seed_scores = scores.unsqueeze(0)
            dots = len(queries) * len(self.mean_keys) * queries.shape[1]
            return scores, interactions, 4096, int(dots)
        rank = 16 if memory == "rank16" else 8
        flattened = queries.flatten(1)
        projected_queries = torch.stack(
            [F.linear(flattened, checkpoint["wq"]) for checkpoint in self.checkpoints[rank]]
        )
        routing_memory = self.routing_memory(memory)
        values = torch.stack([row[0] for row in routing_memory])
        masks = torch.stack([row[1] for row in routing_memory])
        interactions = torch.einsum("sqr,scmr->sqcm", projected_queries, values)
        interactions = interactions / math.sqrt(float(rank))
        interactions = interactions.masked_fill(~masks[:, None], float("-inf"))
        if reducer in {"current", "mean", "recency"}:
            if reducer == "current":
                pooled = projected_queries[:, -1]
            elif reducer == "mean":
                pooled = projected_queries.mean(dim=1)
            else:
                weights = recency_weights(
                    len(queries), device=queries.device
                ).to(projected_queries.dtype)
                pooled = torch.einsum("q,sqr->sr", weights, projected_queries)
            dots_by_mode = torch.einsum("sr,scmr->scm", pooled, values)
            dots_by_mode = dots_by_mode / math.sqrt(float(rank))
            dots_by_mode = dots_by_mode.masked_fill(~masks, float("-inf"))
            count = min(4, values.shape[2])
            top = dots_by_mode.topk(count, dim=-1).values
            finite = torch.isfinite(top)
            seed_scores = top.masked_fill(~finite, 0).sum(dim=-1)
            seed_scores = seed_scores / finite.sum(dim=-1).clamp_min(1)
        else:
            if reducer == "late_max":
                per_query = interactions.amax(dim=-1)
            else:
                count = min(4, values.shape[2])
                top = interactions.topk(count, dim=-1).values
                finite = torch.isfinite(top)
                per_query = top.masked_fill(~finite, 0).sum(dim=-1)
                per_query = per_query / finite.sum(dim=-1).clamp_min(1)
            weights = recency_weights(
                len(queries), device=queries.device
            ).to(per_query.dtype)
            seed_scores = torch.einsum("q,sqc->sc", weights, per_query)
        self.last_seed_scores = seed_scores
        representatives = 32 if memory == "rank16" else 8
        index_bytes = representatives * rank * 4
        dots = len(queries) * len(self.mean_keys) * representatives * rank
        return (
            seed_scores.mean(0),
            interactions.mean(0),
            index_bytes,
            int(dots),
        )

    @property
    def checkpoints(self):
        return self._checkpoints

    @checkpoints.setter
    def checkpoints(self, value):
        self._checkpoints = value


def build_scorer(source, temporal, checkpoints, device):
    return ExampleScorer(source, temporal, checkpoints, device)


def metric_row(
    source: dict,
    scores: torch.Tensor,
    interactions: torch.Tensor,
    *,
    dataset: str,
    split: str,
    condition: str,
    memory: str,
    reducer: str,
    layer: int,
    look_behind: int,
    index_bytes: int,
    routing_dots: int,
) -> dict:
    ranking = stable_topk_indices(scores, BUDGET).tolist()
    metrics = routing_metrics(
        ranking, source["local_positive_mask"], budget=BUDGET
    )
    diagnostics = score_diagnostics(scores, interactions, selected_chunk=ranking[0])
    return {
        "dataset": dataset,
        "split": split,
        "example_id": source["example_id"],
        "condition": condition,
        "memory": memory,
        "reducer": reducer,
        "layer": layer,
        "look_behind": look_behind,
        "budget_chunks": BUDGET,
        "selected_chunks": " ".join(map(str, ranking)),
        "index_bytes_per_chunk": index_bytes,
        "temporal_query_buffer_bytes": look_behind * (16 if memory != "native_mean" else 2048) * 4,
        "routing_dots": routing_dots,
        **metrics,
        **diagnostics.__dict__,
    }


def final_conditions() -> list[tuple[str, int, str, int]]:
    output = [
        (memory, 27, reducer, window)
        for memory in MEMORIES
        for reducer in REDUCERS
        for window in WINDOWS
    ]
    output.extend(
        ("rank16", layer, reducer, window)
        for layer in (0, 8, 18)
        for reducer in ("current", "late_max")
        for window in (1, 4, 8)
    )
    return output


def run_final_sweep(dataset, split, pairs, checkpoints, device):
    rows = []
    for index, (source, temporal) in enumerate(pairs, start=1):
        scorer = build_scorer(source, temporal, checkpoints, device)
        token_count = len(temporal["prompt_token_ids"])
        for memory, layer, reducer, window in final_conditions():
            start = max(0, token_count - window)
            scores, interactions, index_bytes, dots = scorer.score(
                memory,
                layer=layer,
                start=start,
                stop=token_count,
                reducer=reducer,
            )
            condition = f"{memory}_l{layer}_{reducer}_b{window}"
            rows.append(
                metric_row(
                    source,
                    scores,
                    interactions,
                    dataset=dataset,
                    split=split,
                    condition=condition,
                    memory=memory,
                    reducer=reducer,
                    layer=layer,
                    look_behind=window,
                    index_bytes=index_bytes,
                    routing_dots=dots,
                )
            )
        print(f"[final {dataset} {split} {index}/{len(pairs)}]", flush=True)
    return rows


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    metrics = (
        "evidence_recall",
        "evidence_precision",
        "any_evidence",
        "chain_completion",
        "mrr",
        "normalized_entropy",
        "top1_margin",
        "score_concentration",
    )
    output = []
    for key, group in sorted(grouped.items()):
        row = {name: value for name, value in zip(keys, key)}
        row["examples"] = len(group)
        for metric in metrics:
            if metric in group[0]:
                row[metric] = statistics.fmean(float(item[metric]) for item in group)
        output.append(row)
    return output


def select_policies(validation_rows: list[dict]) -> dict:
    policies = {}
    for dataset in sorted({row["dataset"] for row in validation_rows}):
        candidates = [
            row
            for row in aggregate(
                [
                    item
                    for item in validation_rows
                    if item["dataset"] == dataset
                    and item["split"] == "validation"
                    and item["memory"] == "rank16"
                    and item["layer"] == 27
                    and item["reducer"] != "current"
                ],
                ("condition", "reducer", "look_behind"),
            )
            if int(row["look_behind"]) > 1
        ]
        best = max(
            candidates,
            key=lambda row: (
                row["evidence_recall"],
                row["chain_completion"],
                -int(row["look_behind"]),
                row["reducer"] == "current",
            ),
        )
        policies[dataset] = {
            "condition": best["condition"],
            "memory": "rank16",
            "layer": 27,
            "reducer": best["reducer"],
            "look_behind": int(best["look_behind"]),
            "validation_evidence_recall": best["evidence_recall"],
        }
    return policies


def score_at(scorer, policy, anchor, *, future=0, delayed=0):
    endpoint = anchor + delayed
    start = max(0, anchor - policy["look_behind"] + 1)
    if delayed:
        start = max(0, endpoint - policy["look_behind"] + 1)
    stop = endpoint + future + 1
    return scorer.score(
        policy["memory"],
        layer=policy["layer"],
        start=start,
        stop=stop,
        reducer=policy["reducer"],
    )


def trajectory_rows(dataset, split, pairs, checkpoints, policy, thresholds, device):
    rows = []
    for example_index, (source, temporal) in enumerate(pairs, start=1):
        scorer = build_scorer(source, temporal, checkpoints, device)
        question_start, question_stop = temporal["question_span"]
        anchors = list(range(question_start, question_stop))
        score_cache = {}

        def cached(anchor):
            if anchor not in score_cache:
                score_cache[anchor] = score_at(scorer, policy, anchor)
            return score_cache[anchor]

        immediate_rankings, margins, entropies = [], [], []
        for anchor in anchors:
            scores, interactions, _, dots = cached(anchor)
            ranking = stable_topk_indices(scores, BUDGET)
            diagnostics = score_diagnostics(scores, interactions, selected_chunk=int(ranking[0]))
            immediate_rankings.append(ranking)
            margins.append(diagnostics.top1_margin)
            entropies.append(diagnostics.normalized_entropy)
            metrics = routing_metrics(
                ranking.tolist(), source["local_positive_mask"], budget=BUDGET
            )
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "example_id": source["example_id"],
                    "anchor": anchor - question_start,
                    "condition": "immediate",
                    "delay": 0,
                    "look_ahead": 0,
                    "router_call": 1,
                    "routing_dots": dots,
                    "churn": 0.0 if len(immediate_rankings) == 1 else selection_churn(immediate_rankings[-2], ranking),
                    **metrics,
                    "normalized_entropy": diagnostics.normalized_entropy,
                    "top1_margin": diagnostics.top1_margin,
                }
            )
        for amount in DELAYS:
            for anchor in anchors:
                if anchor + amount >= question_stop:
                    continue
                for name, future, delayed in (
                    (f"fixed_delay_{amount}", 0, amount),
                    (f"known_future_{amount}", amount, 0),
                ):
                    scores, _, _, dots = score_at(
                        scorer, policy, anchor, future=future, delayed=delayed
                    )
                    ranking = stable_topk_indices(scores, BUDGET)
                    metrics = routing_metrics(
                        ranking.tolist(), source["local_positive_mask"], budget=BUDGET
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "example_id": source["example_id"],
                            "anchor": anchor - question_start,
                            "condition": name,
                            "delay": amount if delayed else 0,
                            "look_ahead": future,
                            "router_call": 1,
                            "routing_dots": dots,
                            "churn": selection_churn(immediate_rankings[anchor - question_start], ranking),
                            **metrics,
                            "normalized_entropy": float("nan"),
                            "top1_margin": float("nan"),
                        }
                    )
        margin_tensor = torch.tensor(margins)
        entropy_tensor = torch.tensor(entropies)
        for policy_name, margin_threshold, entropy_threshold in (
            ("margin_threshold", thresholds["margin"], None),
            ("entropy_threshold", None, thresholds["entropy"]),
        ):
            for local_anchor, anchor in enumerate(anchors):
                chosen = delayed_commit_index(
                    margin_tensor,
                    entropy_tensor,
                    start=local_anchor,
                    maximum_delay=min(8, len(anchors) - local_anchor - 1),
                    margin_threshold=margin_threshold,
                    entropy_threshold=entropy_threshold,
                )
                ranking = immediate_rankings[chosen]
                metrics = routing_metrics(
                    ranking.tolist(), source["local_positive_mask"], budget=BUDGET
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "example_id": source["example_id"],
                        "anchor": local_anchor,
                        "condition": policy_name,
                        "delay": chosen - local_anchor,
                        "look_ahead": 0,
                        "router_call": 1,
                        "routing_dots": 0,
                        "churn": selection_churn(immediate_rankings[local_anchor], ranking),
                        **metrics,
                        "normalized_entropy": float(entropy_tensor[chosen]),
                        "top1_margin": float(margin_tensor[chosen]),
                    }
                )
        for stride in STRIDES:
            updates = stride_update_mask(len(anchors), stride)
            previous = None
            for local_anchor, anchor in enumerate(anchors):
                router_call = bool(updates[local_anchor])
                if router_call:
                    previous = immediate_rankings[local_anchor]
                metrics = routing_metrics(
                    previous.tolist(), source["local_positive_mask"], budget=BUDGET
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "example_id": source["example_id"],
                        "anchor": local_anchor,
                        "condition": f"stride_{stride}",
                        "delay": 0,
                        "look_ahead": 0,
                        "router_call": int(router_call),
                        "routing_dots": 0,
                        "churn": 0.0,
                        **metrics,
                        "normalized_entropy": float("nan"),
                        "top1_margin": float("nan"),
                    }
                )
        print(
            f"[trajectory {dataset} {split} {example_index}/{len(pairs)}]",
            flush=True,
        )
    return rows


def threshold_policy(validation_trajectory: list[dict], dataset: str) -> dict:
    rows = [
        row
        for row in validation_trajectory
        if row["dataset"] == dataset and row["condition"] == "immediate"
    ]
    return {
        "margin": statistics.median(float(row["top1_margin"]) for row in rows),
        "entropy": statistics.median(float(row["normalized_entropy"]) for row in rows),
    }


def lexical_hybrid_rows(dataset, split, pairs, checkpoints, policy, tokenizer, examples, device):
    rows = []
    for source, temporal in pairs:
        scorer = build_scorer(source, temporal, checkpoints, device)
        tokens = len(temporal["prompt_token_ids"])
        semantic, interactions, index_bytes, dots = scorer.score(
            "rank16",
            layer=27,
            start=max(0, tokens - policy["look_behind"]),
            stop=tokens,
            reducer=policy["reducer"],
        )
        example = examples[(dataset, source["example_id"])]
        lexical = _lexical_scores(
            tokenizer,
            SimpleNamespace(source=example["source"], question=example["question"]),
            source,
            semantic.detach().cpu(),
        )
        for channel in ("exact", "bm25"):
            lexical_scores = lexical[channel].to(device)
            for lexical_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                combined = (
                    lexical_weight * normalize(lexical_scores)
                    + (1.0 - lexical_weight) * normalize(semantic)
                )
                ranking = stable_topk_indices(combined, BUDGET).tolist()
                metrics = routing_metrics(
                    ranking, source["local_positive_mask"], budget=BUDGET
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "example_id": source["example_id"],
                        "condition": f"{channel}_w{lexical_weight:.2f}",
                        "lexical_channel": channel,
                        "lexical_weight": lexical_weight,
                        "selected_chunks": " ".join(map(str, ranking)),
                        "index_bytes_per_chunk": index_bytes,
                        "routing_dots": dots,
                        **metrics,
                    }
                )
    return rows


def select_hybrid(validation_rows, dataset):
    summary = aggregate(
        [row for row in validation_rows if row["dataset"] == dataset],
        ("condition", "lexical_channel", "lexical_weight"),
    )
    return max(
        summary,
        key=lambda row: (
            row["evidence_recall"],
            row["chain_completion"],
            -abs(float(row["lexical_weight"]) - 0.5),
        ),
    )


def bootstrap_effect(rows, left, right, *, seed=20260824, draws=5000):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["example_id"])][row["condition"]] = float(
            row["evidence_recall"]
        )
    effects = [
        values[left] - values[right]
        for values in grouped.values()
        if left in values and right in values
    ]
    if not effects:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(effects[rng.randrange(len(effects))] for _ in effects)
        for _ in range(draws)
    )
    return {
        "mean": statistics.fmean(effects),
        "ci_low": samples[int(0.025 * draws)],
        "ci_high": samples[min(int(0.975 * draws), draws - 1)],
        "n": len(effects),
    }


def inherited_parity_rows(dataset, pairs, checkpoints, device):
    """Reproduce Paper 2.8's dataset-specific seed/ensemble reporting semantics."""
    historical = {}
    if dataset in {"hotpotqa", "qasper"}:
        path = RESULT_28 / "low_rank_frontier/per_example.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if (
                    row["dataset"] == dataset
                    and row["method"] == "lowrank_all"
                    and int(row["rank"]) == 16
                    and int(row["m"]) == 32
                ):
                    historical[(row["example_id"], int(row["seed"]))] = row
        report_semantics = "five_seed_metric_mean"
    else:
        path = RESULT_28 / "multi_dataset/per_example.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if (
                    row["dataset"] == dataset
                    and row["split"] == "test"
                    and row["condition"] == "rank16_retrained"
                    and int(row["seed"]) == -1
                    and int(row["budget_chunks"]) == BUDGET
                ):
                    historical[(row["example_id"], -1)] = row
        report_semantics = "five_seed_score_ensemble"
    output = []
    for source, temporal in pairs:
        scorer = build_scorer(source, temporal, checkpoints, device)
        tokens = len(temporal["prompt_token_ids"])
        ensemble, _, _, _ = scorer.score(
            "rank16", layer=27, start=tokens - 1, stop=tokens, reducer="current"
        )
        if report_semantics == "five_seed_metric_mean":
            scored = zip(SEEDS, scorer.last_seed_scores)
        else:
            scored = ((-1, ensemble),)
        for seed, scores in scored:
            ranking = stable_topk_indices(scores, BUDGET).tolist()
            metrics = routing_metrics(
                ranking, source["local_positive_mask"], budget=BUDGET
            )
            expected = historical[(source["example_id"], int(seed))]
            expected_selection = " ".join(expected["selected_chunks"].split())
            actual_selection = " ".join(map(str, ranking))
            output.append(
                {
                    "dataset": dataset,
                    "example_id": source["example_id"],
                    "seed": seed,
                    "report_semantics": report_semantics,
                    "expected_selection": expected_selection,
                    "actual_selection": actual_selection,
                    "selection_match": expected_selection == actual_selection,
                    "expected_evidence_recall": float(expected["evidence_recall"]),
                    "actual_evidence_recall": metrics["evidence_recall"],
                    "recall_delta": metrics["evidence_recall"]
                    - float(expected["evidence_recall"]),
                }
            )
    return output


def summarize_trajectory(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["split"], row["condition"])].append(row)
    output = []
    for (dataset, split, condition), group in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "split": split,
                "condition": condition,
                "examples": len({row["example_id"] for row in group}),
                "decisions": len(group),
                "evidence_recall": statistics.fmean(float(row["evidence_recall"]) for row in group),
                "chain_completion": statistics.fmean(float(row["chain_completion"]) for row in group),
                "mean_delay": statistics.fmean(float(row["delay"]) for row in group),
                "median_delay": statistics.median(float(row["delay"]) for row in group),
                "router_calls_per_token": statistics.fmean(float(row["router_call"]) for row in group),
                "mean_churn": statistics.fmean(float(row["churn"]) for row in group),
            }
        )
    return output


def make_plots(output_root, final_summary, trajectory_summary, policies):
    colors = {"native_mean": "#5B6770", "rank16": "#007C91", "rank8_centroid8": "#D1495B"}
    datasets = list(policies)
    rows_count = math.ceil(len(datasets) / 2)
    fig, axes = plt.subplots(rows_count, 2, figsize=(10, 3.5 * rows_count), sharex=True, squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        for memory in MEMORIES:
            rows = [
                row for row in final_summary
                if row["dataset"] == dataset and row["split"] == "test"
                and row["memory"] == memory and row["layer"] == 27
                and row["reducer"] == policies[dataset]["reducer"]
            ]
            rows.sort(key=lambda row: int(row["look_behind"]))
            axis.plot(
                [int(row["look_behind"]) for row in rows],
                [row["evidence_recall"] for row in rows],
                marker="o", label=memory.replace("_", " "), color=colors[memory],
            )
        axis.set_title(dataset)
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    for axis in axes.flat:
        axis.set_xlabel("Causal query window B")
    axes[0, 0].set_ylabel("Evidence recall @ 4")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / "causal_window_sweep.png", dpi=180)
    fig.savefig(output_root / "causal_window_sweep.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(rows_count, 2, figsize=(10, 3.5 * rows_count), sharex=True, squeeze=False)
    for axis, dataset in zip(axes.flat, datasets):
        rows = [
            row for row in trajectory_summary
            if row["dataset"] == dataset and row["split"] == "test"
            and (row["condition"].startswith("fixed_delay_") or row["condition"].startswith("known_future_"))
        ]
        for prefix, label, color in (
            ("fixed_delay_", "causal delay", "#007C91"),
            ("known_future_", "known future", "#D1495B"),
        ):
            selected = [row for row in rows if row["condition"].startswith(prefix)]
            selected.sort(key=lambda row: int(row["condition"].split("_")[-1]))
            axis.plot(
                [int(row["condition"].split("_")[-1]) for row in selected],
                [row["evidence_recall"] for row in selected],
                marker="o", label=label, color=color,
            )
        axis.set_title(dataset)
        axis.grid(alpha=0.25)
    for axis in axes.flat:
        axis.set_xlabel("Observed delay / analysis future tokens")
    axes[0, 0].set_ylabel("Trajectory recall @ 4")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_root / "delay_vs_known_future.png", dpi=180)
    fig.savefig(output_root / "delay_vs_known_future.pdf")
    plt.close(fig)


def run(args):
    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    all_source = load_source_features(args.paper2_8_root)
    examples = load_questions(args, all_source)
    final_stage = args.output_root / ".final_stage.pt"
    final_rows = (
        torch.load(final_stage, map_location="cpu", weights_only=False)
        if args.resume and final_stage.exists()
        else []
    )
    completed_final = {
        (row["dataset"], row["split"])
        for row in final_rows
    }
    pair_cache = {}
    checkpoint_cache = {}
    for dataset in args.datasets:
        checkpoint_cache[dataset] = {
            rank: load_checkpoints(dataset, rank, device) for rank in (8, 16)
        }
        for split in ("validation", "test"):
            pairs = aligned_rows(
                all_source[(dataset, split)],
                temporal_rows(args.temporal_root, dataset, split),
            )
            pair_cache[(dataset, split)] = pairs
            if (dataset, split) not in completed_final:
                final_rows.extend(
                    run_final_sweep(
                        dataset, split, pairs, checkpoint_cache[dataset], device
                    )
                )
                torch.save(final_rows, final_stage)
    policies = select_policies(final_rows)
    final_summary = aggregate(
        final_rows,
        ("dataset", "split", "condition", "memory", "reducer", "layer", "look_behind"),
    )
    parity_rows = []
    for dataset in args.datasets:
        parity_rows.extend(
            inherited_parity_rows(
                dataset,
                pair_cache[(dataset, "test")],
                checkpoint_cache[dataset],
                device,
            )
        )
    parity_summary = []
    for dataset in args.datasets:
        rows = [row for row in parity_rows if row["dataset"] == dataset]
        parity_summary.append(
            {
                "dataset": dataset,
                "report_semantics": rows[0]["report_semantics"],
                "rows": len(rows),
                "selection_match_fraction": statistics.fmean(
                    float(row["selection_match"]) for row in rows
                ),
                "expected_evidence_recall": statistics.fmean(
                    row["expected_evidence_recall"] for row in rows
                ),
                "actual_evidence_recall": statistics.fmean(
                    row["actual_evidence_recall"] for row in rows
                ),
                "maximum_absolute_recall_delta": max(
                    abs(row["recall_delta"]) for row in rows
                ),
            }
        )
    write_csv(args.output_root / "final_per_example.csv", final_rows)
    write_csv(args.output_root / "final_summary.csv", final_summary)
    write_csv(args.output_root / "inherited_parity_per_example.csv", parity_rows)
    write_csv(args.output_root / "inherited_parity_summary.csv", parity_summary)
    (args.output_root / "temporal_policies.json").write_text(
        json.dumps(policies, indent=2, sort_keys=True), encoding="utf-8"
    )
    trajectory_stage = args.output_root / ".trajectory_stage.pt"
    if args.resume and trajectory_stage.exists():
        stage = torch.load(trajectory_stage, map_location="cpu", weights_only=False)
    else:
        stage = None
    if stage is not None and stage.get("policies") == policies:
        trajectory = stage["rows"]
        thresholds = stage["thresholds"]
    else:
        trajectory = []
        thresholds = {}
        for dataset in args.datasets:
            preliminary = trajectory_rows(
                dataset,
                "validation",
                pair_cache[(dataset, "validation")],
                checkpoint_cache[dataset],
                policies[dataset],
                {"margin": float("inf"), "entropy": float("-inf")},
                device,
            )
            thresholds[dataset] = threshold_policy(preliminary, dataset)
            trajectory.extend(
                trajectory_rows(
                    dataset,
                    "validation",
                    pair_cache[(dataset, "validation")],
                    checkpoint_cache[dataset],
                    policies[dataset],
                    thresholds[dataset],
                    device,
                )
            )
            trajectory.extend(
                trajectory_rows(
                    dataset,
                    "test",
                    pair_cache[(dataset, "test")],
                    checkpoint_cache[dataset],
                    policies[dataset],
                    thresholds[dataset],
                    device,
                )
            )
        torch.save(
            {"rows": trajectory, "thresholds": thresholds, "policies": policies},
            trajectory_stage,
        )
    trajectory_summary = summarize_trajectory(trajectory)
    write_csv(args.output_root / "trajectory_per_decision.csv", trajectory)
    write_csv(args.output_root / "trajectory_summary.csv", trajectory_summary)

    hybrid_stage = args.output_root / ".hybrid_stage.pt"
    if args.resume and hybrid_stage.exists():
        stage = torch.load(hybrid_stage, map_location="cpu", weights_only=False)
    else:
        stage = None
    if stage is not None and stage.get("temporal_policies") == policies:
        hybrid_rows = stage["rows"]
        hybrid_policies = stage["policies"]
    else:
        hybrid_rows = []
        hybrid_policies = {}
        for dataset in args.datasets:
            validation = lexical_hybrid_rows(
                dataset,
                "validation",
                pair_cache[(dataset, "validation")],
                checkpoint_cache[dataset],
                policies[dataset],
                tokenizer,
                examples,
                device,
            )
            policy = select_hybrid(validation, dataset)
            hybrid_policies[dataset] = policy
            hybrid_rows.extend(validation)
            test = lexical_hybrid_rows(
                dataset,
                "test",
                pair_cache[(dataset, "test")],
                checkpoint_cache[dataset],
                policies[dataset],
                tokenizer,
                examples,
                device,
            )
            hybrid_rows.extend(
                row for row in test if row["condition"] == policy["condition"]
            )
        torch.save(
            {
                "rows": hybrid_rows,
                "policies": hybrid_policies,
                "temporal_policies": policies,
            },
            hybrid_stage,
        )
    hybrid_summary = aggregate(
        hybrid_rows,
        ("dataset", "split", "condition", "lexical_channel", "lexical_weight"),
    )

    interactions = []
    effects = []
    for dataset in args.datasets:
        policy = policies[dataset]
        names = {
            "baseline": f"native_mean_l27_{policy['reducer']}_b1",
            "temporal_only": f"native_mean_l27_{policy['reducer']}_b{policy['look_behind']}",
            "memory_only": f"rank16_l27_{policy['reducer']}_b1",
            "combined": policy["condition"],
        }
        values = {}
        for role, condition in names.items():
            matched = [
                row for row in final_rows
                if row["dataset"] == dataset and row["split"] == "test"
                and row["condition"] == condition
            ]
            values[role] = statistics.fmean(row["evidence_recall"] for row in matched)
        interactions.append(
            {
                "dataset": dataset,
                **names,
                **values,
                "interaction_contrast": interaction_contrast(**values),
            }
        )
        for left, right in (
            (policy["condition"], names["memory_only"]),
            (names["memory_only"], "rank16_l27_current_b1"),
            (names["combined"], names["temporal_only"]),
        ):
            rows = [
                row for row in final_rows
                if row["dataset"] == dataset and row["split"] == "test"
                and row["condition"] in {left, right}
            ]
            effects.append(
                {"dataset": dataset, "left": left, "right": right, **bootstrap_effect(rows, left, right)}
            )

    make_plots(args.output_root, final_summary, trajectory_summary, policies)
    write_csv(args.output_root / "hybrid_per_example.csv", hybrid_rows)
    write_csv(args.output_root / "hybrid_summary.csv", hybrid_summary)
    write_csv(args.output_root / "interaction_contrasts.csv", interactions)
    write_csv(args.output_root / "paired_effects.csv", effects)
    manifest = {
        "schema_version": "1.0",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "native_kv_payload_unchanged": True,
        "materialization_budget_chunks": BUDGET,
        "datasets": list(args.datasets),
        "seeds": list(SEEDS),
        "windows": list(WINDOWS),
        "reducers": list(REDUCERS),
        "delays": list(DELAYS),
        "strides": list(STRIDES),
        "validation_selected_temporal_policies": policies,
        "validation_selected_thresholds": thresholds,
        "validation_selected_hybrid_policies": hybrid_policies,
        "inherited_parity": parity_summary,
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "temporal_feature_root": str(args.temporal_root.resolve()),
    }
    (args.output_root / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args():
    inherited = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/paper2_7_query_facets/annotations.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "2wiki/dev.json")
    parser.add_argument("--musique-dev", type=Path, default=inherited / "musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--paper2-8-root", type=Path, default=RESULT_28)
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_9_look_ahead_back",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_9_look_ahead_back",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
