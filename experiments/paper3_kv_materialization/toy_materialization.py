"""Pure helpers for the controlled Paper 3 materialization experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from pra_torch.controlled_local_sa import SPECIAL_TOKENS, ControlledExample
from pra_torch.materialization import (
    LogicalDomainBounds,
    LogicalInterval,
    allocate_interval_budget,
    evidence_centered_interval,
    union_intervals,
)


@dataclass(frozen=True)
class ToyPolicy:
    """A physical disclosure condition applied after oracle parent selection."""

    name: str
    mode: str
    radius: int | None = None
    budget: int | None = None
    allocation: str | None = None
    wrong_memory: bool = False
    whole_fact: bool = False
    whole_parent: bool = False
    region_groups: int | None = None


def label_metrics(logits: torch.Tensor, answer_id: int) -> dict[str, float | int]:
    """Measure one controlled next-token prediction over the eight label classes."""
    label_start = len(SPECIAL_TOKENS)
    label_logits = logits[label_start : label_start + 8].float()
    label_index = int(answer_id) - label_start
    if not 0 <= label_index < 8:
        raise ValueError("answer_id must name one of the eight controlled labels")
    correct = label_logits[label_index]
    alternatives = label_logits.clone()
    alternatives[label_index] = -torch.inf
    maximum_wrong = alternatives.max()
    probabilities = label_logits.softmax(dim=-1)
    target = F.one_hot(
        torch.tensor(label_index, device=logits.device), num_classes=8
    ).to(probabilities.dtype)
    probability = probabilities[label_index]
    return {
        "correct_logit": float(correct.detach().cpu()),
        "max_wrong_logit": float(maximum_wrong.detach().cpu()),
        "correct_margin": float((correct - maximum_wrong).detach().cpu()),
        "correct_probability": float(probability.detach().cpu()),
        "nll": float((-probability.clamp_min(1e-12).log()).detach().cpu()),
        "prediction_entropy": float(
            (-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
            .detach()
            .cpu()
        ),
        "brier_score": float((probabilities - target).square().sum().detach().cpu()),
        "correct": int(int(label_logits.argmax()) == label_index),
        "full_vocabulary_correct": int(int(logits.argmax()) == int(answer_id)),
    }


def source_and_fact_spans(
    example: ControlledExample,
) -> tuple[tuple[int, ...], dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Recover source tokens plus five-token fact and three-token semantic cores."""
    raw_query_tokens = len(example.query_input_ids) - 1
    source = tuple(example.full_input_ids[:-raw_query_tokens])
    fact_spans: dict[str, tuple[int, int]] = {}
    core_spans: dict[str, tuple[int, int]] = {}
    cursor = 1  # The source begins with [BOS].
    for reference in example.references:
        end = cursor + len(reference.token_ids)
        fact_spans[reference.uri] = (cursor, end)
        core_spans[reference.uri] = (cursor + 1, end - 1)
        cursor = end + example.evidence_gap
    if cursor != len(source):
        raise AssertionError(
            f"reconstructed source has {cursor} tokens, expected {len(source)}"
        )
    return source, fact_spans, core_spans


def _regions(
    spans: Sequence[tuple[int, int]],
    *,
    groups: int | None,
) -> list[tuple[int, int]]:
    """Optionally merge ordered evidence spans into 1, 2, or 4 bounding regions."""
    ordered = sorted((int(start), int(end)) for start, end in spans)
    if groups is None or groups >= len(ordered):
        return ordered
    if groups < 1:
        raise ValueError("groups must be positive")
    result = []
    for group in range(groups):
        left = math.floor(group * len(ordered) / groups)
        right = math.floor((group + 1) * len(ordered) / groups)
        members = ordered[left:right]
        if members:
            result.append((members[0][0], members[-1][1]))
    return result


def policy_intervals(
    policy: ToyPolicy,
    *,
    domain: str,
    source_tokens: int,
    evidence_core_spans: Sequence[tuple[int, int]],
    evidence_fact_spans: Sequence[tuple[int, int]],
    wrong_core_spans: Sequence[tuple[int, int]],
) -> list[LogicalInterval]:
    """Resolve a policy into deterministic source-relative physical intervals."""
    bounds = LogicalDomainBounds(domain, 0, int(source_tokens))
    if policy.whole_parent:
        return [LogicalInterval(domain, 0, int(source_tokens))]
    if policy.whole_fact:
        return [
            LogicalInterval(domain, start, end, start + 1, end - 1)
            for start, end in evidence_fact_spans
        ]
    cores = wrong_core_spans if policy.wrong_memory else evidence_core_spans
    cores = _regions(cores, groups=policy.region_groups)
    radius = int(policy.radius or 0)
    intervals = [
        evidence_centered_interval(
            domain,
            start,
            end,
            radius_left=radius,
            radius_right=radius,
            bounds=bounds,
        )
        for start, end in cores
    ]
    if policy.budget is not None:
        intervals = allocate_interval_budget(
            intervals,
            total_budget=int(policy.budget),
            strategy=str(policy.allocation),
            minimum_per_region=1,
        )
    return intervals


def materialized_positions(
    policy: ToyPolicy,
    intervals: Sequence[LogicalInterval],
    *,
    source_tokens: int,
) -> list[int | None]:
    """Return physical memory positions in the same order as materialized K/V."""
    if policy.mode == "none":
        return []
    if policy.mode == "native_gist_only":
        return [None]
    positions: list[int | None] = []
    if policy.mode == "gist_plus_logical_intervals":
        positions.append(None)
    if policy.whole_parent and policy.mode == "selected_chunks":
        positions.extend(range(source_tokens))
    else:
        for interval in union_intervals(intervals):
            positions.extend(range(interval.start, interval.end))
    return positions


def attention_partition(
    per_head_weights: Iterable[Iterable[float]],
    positions: Sequence[int | None],
    *,
    evidence_positions: set[int],
    distractor_positions: set[int],
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    """Partition final-query shared-softmax mass by physical token provenance."""
    weights = torch.tensor(
        [list(head) for head in per_head_weights], dtype=torch.float64
    )
    if weights.numel() == 0:
        return {
            "memory_attention_mass": 0.0,
            "evidence_attention_mass": 0.0,
            "surrounding_attention_mass": 0.0,
            "distractor_attention_mass": 0.0,
            "native_attention_mass": 1.0,
            "attention_mass_sum": 1.0,
            "attention_entropy": 0.0,
            "effective_support": 1.0,
        }, []
    if weights.shape[-1] != len(positions):
        raise ValueError("attention width does not match materialized positions")
    masks = {
        "evidence": torch.tensor(
            [position in evidence_positions for position in positions]
        ),
        "distractor": torch.tensor(
            [position in distractor_positions for position in positions]
        ),
    }
    masks["surrounding"] = ~(masks["evidence"] | masks["distractor"])
    masses = {
        name: float(weights[:, mask].sum(dim=-1).mean())
        for name, mask in masks.items()
    }
    memory = float(weights.sum(dim=-1).mean())
    native = max(1.0 - memory, 0.0)
    partition = torch.tensor(
        [masses["evidence"], masses["surrounding"], masses["distractor"], native],
        dtype=torch.float64,
    )
    entropy = float(
        (-(partition * partition.clamp_min(1e-12).log()).sum()).item()
    )
    summary = {
        "memory_attention_mass": memory,
        "evidence_attention_mass": masses["evidence"],
        "surrounding_attention_mass": masses["surrounding"],
        "distractor_attention_mass": masses["distractor"],
        "native_attention_mass": native,
        "attention_mass_sum": memory + native,
        "attention_entropy": entropy,
        "effective_support": math.exp(entropy),
    }
    head_rows = []
    for head, row in enumerate(weights):
        head_rows.append(
            {
                "head": head,
                "evidence_attention_mass": float(row[masks["evidence"]].sum()),
                "surrounding_attention_mass": float(row[masks["surrounding"]].sum()),
                "distractor_attention_mass": float(row[masks["distractor"]].sum()),
                "native_attention_mass": max(1.0 - float(row.sum()), 0.0),
            }
        )
    return summary, head_rows


def representation_portability(
    contextual_values: torch.Tensor, isolated_values: torch.Tensor
) -> dict[str, float]:
    """Compare aligned native V states with and without surrounding context."""
    if contextual_values.shape != isolated_values.shape:
        raise ValueError("aligned contextual and isolated values must have equal shape")
    left = contextual_values.float().reshape(-1, contextual_values.shape[-1])
    right = isolated_values.float().reshape(-1, isolated_values.shape[-1])
    cosine = F.cosine_similarity(left, right, dim=-1)
    similarity = float(cosine.mean().detach().cpu())
    return {
        "value_cosine_similarity": similarity,
        "representation_change": 1.0 - similarity,
        "contextual_value_norm": float(left.norm(dim=-1).mean().detach().cpu()),
        "isolated_value_norm": float(right.norm(dim=-1).mean().detach().cpu()),
    }


def retention_vs_parent(
    margin: float, baseline_margin: float, parent_margin: float
) -> float | None:
    """Return retained parent gain when the parent actually improves baseline."""
    gain = float(parent_margin) - float(baseline_margin)
    if gain <= 0.0:
        return None
    return (float(margin) - float(baseline_margin)) / gain


def minimum_sufficient_radius(
    rows: Sequence[dict],
    *,
    target: float = 0.95,
) -> dict[str, float | int | str | None]:
    """Select the smallest validation radius retaining a target parent gain."""
    by_policy = {str(row["policy"]): row for row in rows}
    baseline = float(by_policy["T0_none"]["correct_margin"])
    parent = float(by_policy["T7_whole_parent"]["correct_margin"])
    candidates = []
    for radius in (0, 2, 4, 8, 16):
        row = by_policy[f"T{1 if radius == 0 else {2: 2, 4: 3, 8: 4, 16: 5}[radius]}_radius_{radius}"]
        retention = retention_vs_parent(
            float(row["correct_margin"]), baseline, parent
        )
        candidates.append((radius, retention, row))
    feasible = [
        (radius, retention, row)
        for radius, retention, row in candidates
        if retention is not None and retention >= target
    ]
    if not feasible:
        return {
            "status": "unresolved",
            "radius": None,
            "materialized_tokens": None,
            "retention": None,
            "parent_gain": parent - baseline,
        }
    radius, retention, row = min(feasible, key=lambda value: value[0])
    return {
        "status": "supported",
        "radius": radius,
        "materialized_tokens": int(row["materialized_tokens"]),
        "retention": float(retention),
        "parent_gain": parent - baseline,
    }
