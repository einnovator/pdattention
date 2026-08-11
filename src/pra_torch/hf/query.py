"""Zero-parameter information-need representations for HF PRA routing."""

from __future__ import annotations

from collections.abc import Sequence

import torch


QUERY_LAST = "last"
QUERY_UNIFORM = "uniform"
QUERY_EXPONENTIAL = "exponential"
QUERY_LINEAR = "linear"
QUERY_QUESTION_MEAN = "question_mean"
QUERY_QUESTION_EXPONENTIAL = "question_exponential"

RUNTIME_QUERY_STRATEGIES = {
    QUERY_LAST,
    QUERY_UNIFORM,
    QUERY_EXPONENTIAL,
    QUERY_LINEAR,
}
QUERY_STRATEGIES = RUNTIME_QUERY_STRATEGIES | {
    QUERY_QUESTION_MEAN,
    QUERY_QUESTION_EXPONENTIAL,
}


def half_life_to_decay(half_life: float) -> float:
    """Convert a token half-life to the equivalent per-token decay factor."""
    if half_life <= 0:
        raise ValueError("query half-life must be positive.")
    return 2.0 ** (-1.0 / float(half_life))


def token_span_from_offsets(
    offsets: Sequence[Sequence[int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Map a known character interval to its overlapping token interval."""
    if char_start < 0 or char_end <= char_start:
        raise ValueError("Question character bounds must form a non-empty interval.")
    overlapping = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > char_start and int(start) < char_end
    ]
    if not overlapping:
        raise ValueError("Question span does not overlap the retained prompt tokens.")
    return overlapping[0], overlapping[-1] + 1


def _span_rows(
    states: torch.Tensor,
    strategy: str,
    window: int | None,
    token_spans: Sequence[tuple[int, int]] | None,
) -> list[torch.Tensor]:
    batch, tokens, _ = states.shape
    if strategy.startswith("question_"):
        if token_spans is None or len(token_spans) != batch:
            raise ValueError("Question query strategies require one token span per batch row.")
        rows = []
        for row, (start, end) in zip(states, token_spans):
            start, end = int(start), int(end)
            if start < 0 or end <= start or end > tokens:
                raise ValueError("Question token spans must fit the query-state sequence.")
            rows.append(row[start:end])
        return rows
    if window is None or window <= 0:
        raise ValueError("Recent query strategies require a positive window.")
    start = max(0, tokens - int(window))
    return [row[start:] for row in states]


def aggregate_query_states(
    states: torch.Tensor,
    strategy: str = QUERY_LAST,
    *,
    window: int | None = None,
    half_life: float | None = None,
    token_spans: Sequence[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Pool ``[B,T,D]`` attention-input states into routing queries ``[B,D]``.

    Weighting is normalized before pooling. Recent-state weights are defined
    backward from the newest available token; question strategies apply the
    same rule within each explicitly supplied question span.
    """
    if states.ndim != 3 or states.shape[1] == 0:
        raise ValueError("Query states must have shape [batch,tokens,width] with tokens.")
    if strategy not in QUERY_STRATEGIES:
        raise ValueError(f"Unsupported query strategy: {strategy}")
    if strategy == QUERY_LAST:
        return states[:, -1, :]

    rows = _span_rows(states, strategy, window, token_spans)
    pooled = []
    for row in rows:
        length = row.shape[0]
        if strategy in {QUERY_UNIFORM, QUERY_QUESTION_MEAN}:
            weights = row.new_ones(length)
        elif strategy in {QUERY_EXPONENTIAL, QUERY_QUESTION_EXPONENTIAL}:
            if half_life is None:
                raise ValueError("Exponential query strategies require a half-life.")
            decay = half_life_to_decay(float(half_life))
            distance = torch.arange(length - 1, -1, -1, device=row.device, dtype=row.dtype)
            weights = torch.pow(row.new_tensor(decay), distance)
        elif strategy == QUERY_LINEAR:
            weights = torch.arange(1, length + 1, device=row.device, dtype=row.dtype)
        else:  # pragma: no cover - guarded by QUERY_STRATEGIES
            raise AssertionError(strategy)
        weights = weights / weights.sum()
        pooled.append((row * weights.unsqueeze(-1)).sum(dim=0))
    return torch.stack(pooled)


def streaming_exponential_query(
    states: torch.Tensor,
    half_life: float,
) -> torch.Tensor:
    """Compute the normalized full-history exponential query by O(1)-state EMA."""
    if states.ndim != 3 or states.shape[1] == 0:
        raise ValueError("Query states must have shape [batch,tokens,width] with tokens.")
    decay = half_life_to_decay(half_life)
    numerator = torch.zeros_like(states[:, 0, :])
    denominator = states.new_zeros((states.shape[0], 1))
    for token in states.unbind(dim=1):
        numerator = decay * numerator + token
        denominator = decay * denominator + 1.0
    return numerator / denominator
