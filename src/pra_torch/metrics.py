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
