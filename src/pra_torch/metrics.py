import math
import time
from dataclasses import dataclass, field

import torch


@dataclass
class RunningAverages:
    """Incrementally average scalar metric dictionaries."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)

    def update(self, metrics: dict[str, float], weight: float = 1.0) -> None:
        """Add scalar metrics with an optional sample/batch weight."""
        weight = float(weight)
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value) * weight
            self.counts[key] = self.counts.get(key, 0.0) + weight

    def compute(self) -> dict[str, float]:
        return {key: self.totals[key] / max(self.counts[key], 1) for key in self.totals}


def perplexity(loss: float) -> float:
    """Convert cross-entropy loss to a numerically capped perplexity."""
    try:
        return float(math.exp(min(loss, 20.0)))
    except OverflowError:
        return float("inf")


def grad_norm(parameters) -> float:
    """Compute the global L2 norm of existing parameter gradients."""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item() ** 2)
    return total ** 0.5


def cuda_memory_allocated(device: str) -> float:
    """Return allocated CUDA memory in MiB, or zero on CPU."""
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.memory_allocated() / (1024 * 1024))
    return 0.0


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


class ThroughputTimer:
    """Simple wall-clock timer for examples/sec and tokens/sec metrics."""

    def __init__(self):
        self.start = time.perf_counter()

    def rates(self, examples: int, tokens: int) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self.start, 1e-9)
        return {
            "examples_per_second": examples / elapsed,
            "tokens_per_second": tokens / elapsed,
        }
