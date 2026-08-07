import math
from common.metrics import (
    RunningAverages,
    ThroughputTimer,
    cuda_memory_allocated,
    grad_norm,
    perplexity,
)


def ranking_metrics(selected: list[str], expected: set[str]) -> dict[str, float]:
    """Compute binary-relevance hit, recall, precision, MRR, and nDCG."""
    selected = list(dict.fromkeys(selected))
    relevant = [item in expected for item in selected]
    hits = sum(relevant)
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0)
    dcg = sum((1.0 / math.log2(rank + 1)) for rank, hit in enumerate(relevant, 1) if hit)
    ideal_hits = min(len(expected), len(selected))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    recall = hits / max(len(expected), 1)
    precision = hits / max(len(selected), 1)
    return {
        "hit_at_1": float(bool(relevant and relevant[0])),
        "hit_at_k": float(hits > 0),
        "recall_at_k": recall,
        "precision_at_k": precision,
        "f1_at_k": 2 * precision * recall / max(precision + recall, 1e-12),
        "mrr": reciprocal_rank,
        "ndcg": dcg / max(ideal_dcg, 1e-12) if expected else 0.0,
        "selected_count": float(len(selected)),
    }


def span_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(min(first[1], second[1]) - max(first[0], second[0]), 0)
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / max(union, 1)


def chunk_is_relevant(hit, target_ids: set[str], target_spans: list[dict], mode: str, iou: float) -> bool:
    if hit.chunk_id in target_ids:
        return True
    for span in target_spans:
        if span.get("uri") != hit.source_uri:
            continue
        target = (int(span["token_start"]), int(span["token_end"]))
        selected = (hit.token_start, hit.token_end)
        if mode == "any_overlap" and min(target[1], selected[1]) > max(target[0], selected[0]):
            return True
        if mode == "iou_threshold" and span_iou(target, selected) >= iou:
            return True
    return False


__all__ = [
    "RunningAverages",
    "ThroughputTimer",
    "chunk_is_relevant",
    "cuda_memory_allocated",
    "grad_norm",
    "perplexity",
    "ranking_metrics",
    "span_iou",
]
