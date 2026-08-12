"""Stable router evaluation and recall-sparsity report helpers."""

from __future__ import annotations

from typing import Iterable

import torch

from common.recall_sparsity import recall_sparsity_curve

from .router import PRARouter


def evaluate_router_features(
    router: PRARouter,
    features: Iterable[dict],
    *,
    query_strategy: str = "last",
    device: torch.device | str = "cpu",
) -> dict:
    """Evaluate evidence recall over frozen query/chunk feature records."""
    router = router.to(device).eval()
    rankings = []
    evidence = []
    token_lengths = []
    datasets: dict[str, int] = {}
    with torch.no_grad():
        for feature in features:
            query = feature["queries"][query_strategy].reshape(1, -1).to(device)
            memory = feature["memory_gists"].to(device)
            scores = router.scores(query, memory)[0]
            order = torch.argsort(scores, descending=True).cpu().tolist()
            rankings.append(order)
            positive = feature["positive_mask"].nonzero(as_tuple=False).flatten().tolist()
            evidence.append(set(int(index) for index in positive))
            spans = feature.get("chunk_spans")
            token_lengths.append(
                [int(spans[index][1] - spans[index][0]) for index in order]
                if spans is not None
                else [1 for _ in order]
            )
            dataset = str(feature.get("dataset", "unknown"))
            datasets[dataset] = datasets.get(dataset, 0) + 1
    report = recall_sparsity_curve(
        rankings,
        evidence,
        candidate_token_lengths=token_lengths,
        require_complete_endpoint=True,
    )
    reciprocal_ranks = []
    for order, positives in zip(rankings, evidence):
        first_positive = next(
            (rank for rank, candidate in enumerate(order, start=1) if candidate in positives),
            None,
        )
        reciprocal_ranks.append(1.0 / first_positive if first_positive is not None else 0.0)
    by_fraction = {float(row["fraction"]): row for row in report["curve"]}
    report["summary"] = {
        **{
            f"R@{int(fraction * 100)}%": by_fraction[fraction]["recall"]
            for fraction in (0.05, 0.10, 0.20, 0.30)
        },
        "f70": report["inverse"]["f70"],
        "f80": report["inverse"]["f80"],
        "f90": report["inverse"]["f90"],
        "AUC0-30": report["auc_0_30"],
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        **{
            f"R@{cutoff}": report["fixed_k"][str(cutoff)]["recall"]
            for cutoff in (3, 8, 16)
        },
    }
    report["dataset_counts"] = datasets
    report["query_strategy"] = query_strategy
    return report
