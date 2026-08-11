"""Small evidence-supervised trainer for frozen-backbone PRA routing adapters."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Iterable

import torch

from .evaluation import evaluate_router_features
from .router import PRARouter


def load_feature_rows(paths: Iterable[str | Path]) -> list[dict]:
    """Load and concatenate trusted feature records produced by PRA extraction."""
    rows = []
    for path in paths:
        rows.extend(torch.load(path, map_location="cpu", weights_only=False))
    if not rows:
        raise ValueError("At least one routing feature row is required.")
    return rows


def train_router(
    train_features: list[dict],
    validation_features: list[dict],
    *,
    routing_width: int = 128,
    query_strategy: str = "last",
    steps: int = 512,
    learning_rate: float = 1e-3,
    seed: int = 53,
    device: torch.device | str = "cpu",
    metadata: dict | None = None,
) -> tuple[PRARouter, dict]:
    """Train asymmetric query/memory projections with a multi-positive loss."""
    if not train_features or not validation_features:
        raise ValueError("Training and validation feature sets must be non-empty.")
    random.seed(seed)
    torch.manual_seed(seed)
    input_width = int(train_features[0]["memory_gists"].shape[-1])
    router = PRARouter(
        input_width,
        routing_width,
        "asymmetric_linear",
        metadata=metadata,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=learning_rate)
    losses = []
    started = time.perf_counter()
    router.train()
    for _ in range(steps):
        feature = random.choice(train_features)
        query = feature["queries"][query_strategy].reshape(1, -1).to(device)
        memory = feature["memory_gists"].to(device)
        positive = feature["positive_mask"].to(device=device, dtype=torch.bool)
        scores = router.scores(query, memory)[0] / 0.07
        loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positive], dim=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    training_seconds = time.perf_counter() - started
    router.freeze()
    validation = evaluate_router_features(
        router, validation_features, query_strategy=query_strategy, device=device
    )
    metrics = {
        "steps": steps,
        "seed": seed,
        "learning_rate": learning_rate,
        "training_seconds": training_seconds,
        "mean_training_loss": sum(losses) / len(losses),
        "adapter_parameters": router.parameter_count,
        "validation": validation,
    }
    router.metadata.update(
        {
            "training_seed": seed,
            "training_steps": steps,
            "query_strategy": query_strategy,
            "metrics": validation["summary"],
        }
    )
    return router, metrics
