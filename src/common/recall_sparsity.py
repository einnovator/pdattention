"""Dataset-independent recall curves over selected chunks and exact KV-token fractions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Hashable, Sequence


DEFAULT_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0)
DEFAULT_FIXED_K = (1, 3, 8, 16)


def _validate_inputs(
    rankings: Sequence[Sequence[Hashable]],
    evidence_ids: Sequence[set[Hashable]],
    candidate_sizes: Sequence[int] | None,
    candidate_token_lengths: Sequence[Sequence[int]] | None,
) -> None:
    if not rankings or len(rankings) != len(evidence_ids):
        raise ValueError("Rankings and evidence_ids must contain the same non-zero example count.")
    if candidate_sizes is not None and len(candidate_sizes) != len(rankings):
        raise ValueError("candidate_sizes must contain one value per ranking.")
    if candidate_token_lengths is not None and len(candidate_token_lengths) != len(rankings):
        raise ValueError("candidate_token_lengths must contain one row per ranking.")
    for index, (ranking, evidence) in enumerate(zip(rankings, evidence_ids)):
        if not ranking or not evidence:
            raise ValueError(f"Example {index} must have candidates and evidence.")
        if len(set(ranking)) != len(ranking):
            raise ValueError(f"Example {index} ranking contains duplicate identities.")
        if candidate_sizes is not None and int(candidate_sizes[index]) != len(ranking):
            raise ValueError(f"Example {index} candidate size does not match its ranking.")
        if candidate_token_lengths is not None:
            lengths = candidate_token_lengths[index]
            if len(lengths) != len(ranking) or any(int(value) <= 0 for value in lengths):
                raise ValueError(f"Example {index} requires one positive token length per candidate.")


def _aggregate_at_cutoffs(
    rankings: Sequence[Sequence[Hashable]],
    evidence_ids: Sequence[set[Hashable]],
    cutoffs: Sequence[int],
    candidate_token_lengths: Sequence[Sequence[int]] | None,
) -> dict[str, float | None]:
    recalls = []
    any_recalls = []
    all_recalls = []
    selected_fractions = []
    kv_fractions = []
    for index, (ranking, evidence, cutoff) in enumerate(zip(rankings, evidence_ids, cutoffs)):
        selected = set(ranking[: int(cutoff)])
        hits = len(selected & evidence)
        recalls.append(hits / len(evidence))
        any_recalls.append(float(hits > 0))
        all_recalls.append(float(hits == len(evidence)))
        selected_fractions.append(min(int(cutoff), len(ranking)) / len(ranking))
        if candidate_token_lengths is not None:
            lengths = candidate_token_lengths[index]
            kv_fractions.append(sum(lengths[: int(cutoff)]) / sum(lengths))
    return {
        "recall": statistics.fmean(recalls),
        "any_evidence_recall": statistics.fmean(any_recalls),
        "all_evidence_recall": statistics.fmean(all_recalls),
        "selected_chunk_fraction": statistics.fmean(selected_fractions),
        "selected_kv_token_fraction": (
            statistics.fmean(kv_fractions) if kv_fractions else None
        ),
    }


def _normalized_sparse_auc(curve: Sequence[dict], limit: float = 0.30) -> float:
    points = [(0.0, 0.0)]
    points.extend(
        (float(row["fraction"]), float(row["recall"]))
        for row in curve
        if float(row["fraction"]) <= limit
    )
    if not points or points[-1][0] != limit:
        raise ValueError(f"Recall curve must include fraction {limit:.2f} for sparse AUC.")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    return area / limit


def recall_sparsity_curve(
    rankings: Sequence[Sequence[Hashable]],
    evidence_ids: Sequence[set[Hashable]],
    *,
    candidate_sizes: Sequence[int] | None = None,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    fixed_k: Sequence[int] = DEFAULT_FIXED_K,
    candidate_token_lengths: Sequence[Sequence[int]] | None = None,
    require_complete_endpoint: bool = False,
) -> dict:
    """Aggregate evidence coverage as each example selects ``ceil(f * N_i)`` candidates.

    ``candidate_token_lengths`` must follow each ranked candidate order. When omitted, the
    function intentionally leaves the native-KV fraction undefined rather than approximating it
    from chunk count.
    """
    _validate_inputs(rankings, evidence_ids, candidate_sizes, candidate_token_lengths)
    normalized_fractions = sorted({float(value) for value in fractions})
    if not normalized_fractions or any(value <= 0.0 or value > 1.0 for value in normalized_fractions):
        raise ValueError("Fractions must lie in (0, 1].")
    curve = []
    for fraction in normalized_fractions:
        cutoffs = [max(1, math.ceil(fraction * len(ranking))) for ranking in rankings]
        curve.append(
            {
                "fraction": fraction,
                **_aggregate_at_cutoffs(
                    rankings, evidence_ids, cutoffs, candidate_token_lengths
                ),
            }
        )
    recalls = [float(row["recall"]) for row in curve]
    if any(left > right + 1e-12 for left, right in zip(recalls, recalls[1:])):
        raise AssertionError("Recall-sparsity curve is not monotonic.")
    endpoint_complete = bool(curve[-1]["fraction"] == 1.0 and abs(recalls[-1] - 1.0) <= 1e-12)
    if require_complete_endpoint and not endpoint_complete:
        raise AssertionError("Selecting 100% of indexed candidates does not recover all evidence.")

    inverse = {}
    for target in (0.70, 0.80, 0.90, 0.95):
        inverse[f"f{int(target * 100)}"] = next(
            (
                float(row["selected_chunk_fraction"])
                for row in curve
                if float(row["recall"]) >= target
            ),
            None,
        )
    fixed = {}
    for cutoff in sorted({int(value) for value in fixed_k}):
        if cutoff <= 0:
            raise ValueError("Fixed-k values must be positive.")
        fixed[str(cutoff)] = _aggregate_at_cutoffs(
            rankings,
            evidence_ids,
            [min(cutoff, len(ranking)) for ranking in rankings],
            candidate_token_lengths,
        )
    return {
        "examples": len(rankings),
        "curve": curve,
        "inverse": inverse,
        "auc_0_30": _normalized_sparse_auc(curve),
        "fixed_k": fixed,
        "endpoint_complete": endpoint_complete,
        "evidence_semantics": "fraction of mapped evidence identities recovered",
        "kv_fraction_exact": candidate_token_lengths is not None,
    }
