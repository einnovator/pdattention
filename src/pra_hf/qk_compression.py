"""Native Q/K response teachers and compact landmark controllers.

Paper 2.8 treats a routing gist as an approximation to a frozen attention
layer's query response.  This module deliberately does not materialize values:
it scores candidate chunks from pre-RoPE query/key tensors and returns token
indices that a caller can persist as routing landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

import torch
from torch import nn


TEACHER_FUNCTIONS = ("max", "top_r_mean", "logsumexp", "attention_mass")
LANDMARK_TRAINING_OBJECTIVES = (
    "oracle_imitation",
    "listwise",
    "combined",
    "decision_aware",
)


def gqa_head_map(query_heads: int, key_heads: int, *, device=None) -> torch.Tensor:
    """Map each query head to its grouped-query-attention K/V head."""
    if query_heads <= 0 or key_heads <= 0 or query_heads % key_heads:
        raise ValueError("Query heads must be a positive multiple of K/V heads.")
    return torch.arange(query_heads, device=device) // (query_heads // key_heads)


def _canonical_queries(queries: torch.Tensor) -> torch.Tensor:
    if queries.ndim == 2:
        queries = queries.unsqueeze(0)
    if queries.ndim != 3 or queries.shape[-1] == 0:
        raise ValueError("Queries must have shape [queries,query_heads,head_dim].")
    return queries


def _canonical_keys(
    keys: torch.Tensor, token_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if keys.ndim != 4 or keys.shape[1] == 0 or keys.shape[-1] == 0:
        raise ValueError("Keys must have shape [chunks,tokens,kv_heads,head_dim].")
    if token_mask is None:
        token_mask = torch.ones(keys.shape[:2], dtype=torch.bool, device=keys.device)
    if token_mask.shape != keys.shape[:2]:
        raise ValueError("Token mask must have shape [chunks,tokens].")
    token_mask = token_mask.to(device=keys.device, dtype=torch.bool)
    if not bool(token_mask.any(dim=1).all()):
        raise ValueError("Every chunk must contain at least one valid key token.")
    return keys, token_mask


def token_query_key_dots(
    queries: torch.Tensor,
    keys: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scaled native dots ``[Q,C,T,Hq]`` and a ``[C,T]`` mask."""
    queries = _canonical_queries(queries)
    keys, token_mask = _canonical_keys(keys, token_mask)
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("Query and key head dimensions must match.")
    mapping = gqa_head_map(queries.shape[1], keys.shape[2], device=keys.device)
    compatible_keys = keys[:, :, mapping, :]
    dots = torch.einsum("qhd,cthd->qcth", queries, compatible_keys)
    return dots / sqrt(float(keys.shape[-1])), token_mask


def reduce_token_responses(
    dots: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    function: str = "max",
    top_r: int = 4,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reduce ``[Q,C,T,H]`` token responses to per-head ``[Q,C,H]`` scores."""
    if function not in TEACHER_FUNCTIONS:
        raise ValueError(f"Unsupported teacher function: {function}")
    if dots.ndim != 4 or token_mask.shape != dots.shape[1:3]:
        raise ValueError("Dots and token mask are not aligned.")
    if top_r <= 0 or temperature <= 0:
        raise ValueError("top_r and temperature must be positive.")
    mask = token_mask.unsqueeze(0).unsqueeze(-1)
    masked = dots.masked_fill(~mask, float("-inf"))
    if function == "max":
        return masked.amax(dim=2)
    if function == "top_r_mean":
        count = min(int(top_r), dots.shape[2])
        values = masked.topk(count, dim=2).values
        finite = torch.isfinite(values)
        return values.masked_fill(~finite, 0).sum(dim=2) / finite.sum(dim=2).clamp_min(1)
    if function == "logsumexp":
        normalizer = token_mask.sum(dim=1).clamp_min(1).log().view(1, -1, 1)
        return temperature * torch.logsumexp(masked / temperature, dim=2) - (
            temperature * normalizer
        )
    weights = torch.softmax(masked / temperature, dim=2)
    return (weights * masked.masked_fill(~mask, 0)).sum(dim=2)


def qk_response_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    function: str = "max",
    top_r: int = 4,
    temperature: float = 1.0,
    head_reduction: str = "mean",
) -> torch.Tensor:
    """Score each query/chunk pair, returning ``[queries,chunks]``."""
    dots, token_mask = token_query_key_dots(queries, keys, token_mask)
    per_head = reduce_token_responses(
        dots,
        token_mask,
        function=function,
        top_r=top_r,
        temperature=temperature,
    )
    if head_reduction == "mean":
        return per_head.mean(dim=-1)
    if head_reduction == "max":
        return per_head.amax(dim=-1)
    raise ValueError("head_reduction must be 'mean' or 'max'.")


def masked_mean_keys(keys: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Build one synthetic mean key per chunk as ``[C,1,Hkv,D]``."""
    keys, token_mask = _canonical_keys(keys, token_mask)
    weights = token_mask.to(keys.dtype).unsqueeze(-1).unsqueeze(-1)
    return (keys * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1)


def gather_landmarks(
    keys: torch.Tensor,
    indices: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather per-chunk token indices into padded compact key streams."""
    if len(indices) != keys.shape[0] or not indices:
        raise ValueError("Landmark indices must provide one non-empty row per chunk.")
    width = max(len(row) for row in indices)
    if width <= 0:
        raise ValueError("Every chunk requires at least one landmark.")
    output = keys.new_zeros((keys.shape[0], width, keys.shape[2], keys.shape[3]))
    mask = torch.zeros((keys.shape[0], width), dtype=torch.bool, device=keys.device)
    for chunk, row in enumerate(indices):
        if not row:
            raise ValueError("Every chunk requires at least one landmark.")
        chosen = torch.as_tensor(row, dtype=torch.long, device=keys.device)
        if bool(((chosen < 0) | (chosen >= keys.shape[1])).any()):
            raise IndexError("Landmark token index is outside its chunk.")
        output[chunk, : len(row)] = keys[chunk].index_select(0, chosen)
        mask[chunk, : len(row)] = True
    return output, mask


def last_token_indices(token_mask: torch.Tensor, m: int = 1) -> list[list[int]]:
    """Select the last ``m`` valid native tokens in each chunk."""
    if m <= 0:
        raise ValueError("m must be positive.")
    output = []
    for row in token_mask:
        valid = torch.nonzero(row, as_tuple=False).flatten().tolist()
        output.append(valid[-min(m, len(valid)) :])
    return output


def random_token_indices(
    token_mask: torch.Tensor, m: int, *, generator: torch.Generator
) -> list[list[int]]:
    """Select a deterministic random native subset for each chunk."""
    if m <= 0:
        raise ValueError("m must be positive.")
    output = []
    for row in token_mask.cpu():
        valid = torch.nonzero(row, as_tuple=False).flatten()
        order = torch.randperm(len(valid), generator=generator)[: min(m, len(valid))]
        output.append(valid.index_select(0, order).tolist())
    return output


def farthest_first_indices(
    keys: torch.Tensor, token_mask: torch.Tensor, m: int
) -> list[list[int]]:
    """Select diverse native tokens using cosine farthest-first traversal."""
    if m <= 0:
        raise ValueError("m must be positive.")
    output: list[list[int]] = []
    for chunk, row_mask in zip(keys, token_mask):
        valid = torch.nonzero(row_mask, as_tuple=False).flatten()
        vectors = chunk.index_select(0, valid).flatten(1).float()
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
        centroid = torch.nn.functional.normalize(vectors.mean(dim=0), dim=0)
        selected = [int(torch.argmax(vectors @ centroid).item())]
        while len(selected) < min(m, len(valid)):
            similarity = vectors @ vectors[selected].T
            nearest = similarity.amax(dim=1)
            nearest[selected] = float("inf")
            selected.append(int(torch.argmin(nearest).item()))
        output.append(valid[selected].tolist())
    return output


def greedy_qk_landmarks(
    queries: torch.Tensor,
    keys: torch.Tensor,
    token_mask: torch.Tensor,
    m: int,
    *,
    function: str,
    top_r: int = 4,
    temperature: float = 1.0,
    head_reduction: str = "mean",
) -> list[list[int]]:
    """Build a query-aware, non-deployable native-landmark upper bound.

    At each step, the candidate whose compact response has the lowest mean
    squared error to the full-token teacher response is retained.  Selection is
    performed independently per chunk but can use an ensemble of ``Q`` queries.
    """
    if m <= 0:
        raise ValueError("m must be positive.")
    teacher = qk_response_scores(
        queries,
        keys,
        token_mask,
        function=function,
        top_r=top_r,
        temperature=temperature,
        head_reduction=head_reduction,
    )
    if function == "max":
        dots, _ = token_query_key_dots(queries, keys, token_mask)
        selections = []
        for chunk_index, row_mask in enumerate(token_mask):
            valid_mask = row_mask.clone()
            selected: list[int] = []
            running = None
            target = teacher[:, chunk_index]
            chunk_dots = dots[:, chunk_index]
            for _ in range(min(m, int(row_mask.sum().item()))):
                candidate_heads = (
                    chunk_dots
                    if running is None
                    else torch.maximum(running.unsqueeze(1), chunk_dots)
                )
                if head_reduction == "mean":
                    estimates = candidate_heads.mean(dim=-1)
                elif head_reduction == "max":
                    estimates = candidate_heads.amax(dim=-1)
                else:
                    raise ValueError("head_reduction must be 'mean' or 'max'.")
                losses = ((estimates - target.unsqueeze(1)) ** 2).mean(dim=0)
                losses = losses.masked_fill(~valid_mask, float("inf"))
                token = int(torch.argmin(losses).item())
                selected.append(token)
                valid_mask[token] = False
                running = (
                    chunk_dots[:, token]
                    if running is None
                    else torch.maximum(running, chunk_dots[:, token])
                )
            selections.append(selected)
        return selections
    selections: list[list[int]] = []
    for chunk_index, row_mask in enumerate(token_mask):
        valid = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
        selected: list[int] = []
        target = teacher[:, chunk_index]
        for _ in range(min(m, len(valid))):
            best_token, best_loss = None, float("inf")
            for token in valid:
                if token in selected:
                    continue
                trial = selected + [token]
                compact = keys[chunk_index : chunk_index + 1, trial]
                compact_mask = torch.ones(
                    (1, len(trial)), dtype=torch.bool, device=keys.device
                )
                estimate = qk_response_scores(
                    queries,
                    compact,
                    compact_mask,
                    function=function,
                    top_r=top_r,
                    temperature=temperature,
                    head_reduction=head_reduction,
                )[:, 0]
                loss = float(torch.mean((target - estimate) ** 2).item())
                if loss < best_loss or (loss == best_loss and token < int(best_token)):
                    best_token, best_loss = token, loss
            if best_token is None:  # pragma: no cover - valid is non-empty
                break
            selected.append(best_token)
        selections.append(selected)
    return selections


def landmark_features(keys: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Return key-only selector features ``[C,T,8]`` without query leakage."""
    keys, token_mask = _canonical_keys(keys, token_mask)
    flat = keys.float().flatten(2)
    norm = flat.norm(dim=-1)
    normalized = torch.nn.functional.normalize(flat, dim=-1)
    weights = token_mask.to(flat.dtype).unsqueeze(-1)
    centroid = (flat * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    centroid = torch.nn.functional.normalize(centroid, dim=-1)
    centroid_cosine = torch.einsum("ctd,cd->ct", normalized, centroid)
    previous = torch.roll(normalized, shifts=1, dims=1)
    local_change = 1.0 - (normalized * previous).sum(dim=-1)
    local_change[:, 0] = 0
    head_norm = keys.float().norm(dim=-1)
    positions = torch.arange(keys.shape[1], device=keys.device, dtype=flat.dtype)
    positions = positions / max(keys.shape[1] - 1, 1)
    position = positions.unsqueeze(0).expand(keys.shape[0], -1)
    reverse_position = 1.0 - position
    features = torch.stack(
        (
            norm,
            centroid_cosine,
            local_change,
            head_norm.mean(dim=-1),
            head_norm.std(dim=-1, unbiased=False),
            position,
            reverse_position,
            token_mask.to(flat.dtype),
        ),
        dim=-1,
    )
    return features.masked_fill(~token_mask.unsqueeze(-1), 0)


class NativeLandmarkSelector(nn.Module):
    """Tiny key-only MLP that scores native token landmarks independently."""

    def __init__(self, feature_width: int = 8, hidden_width: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, features: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if features.shape[:-1] != token_mask.shape:
            raise ValueError("Selector features and token mask are not aligned.")
        return self.network(features).squeeze(-1).masked_fill(~token_mask, float("-inf"))

    @torch.no_grad()
    def select(
        self, keys: torch.Tensor, token_mask: torch.Tensor, m: int
    ) -> list[list[int]]:
        scores = self(landmark_features(keys, token_mask), token_mask)
        selections = []
        for row, mask in zip(scores, token_mask):
            count = min(int(m), int(mask.sum().item()))
            selections.append(row.topk(count).indices.sort().values.tolist())
        return selections


class QueryConditionedLandmarkSelector(nn.Module):
    """Score token landmarks from cached key features and the current native query.

    The low-rank interaction is bounded by ``rank`` and leaves the frozen
    transformer's query/key projections unchanged. Inputs are ``[...,C,T,F]``
    key features, ``[...,Q]`` flattened query features, and ``[...,C,T]`` masks.
    """

    def __init__(
        self,
        query_width: int,
        *,
        feature_width: int = 8,
        hidden_width: int = 32,
        rank: int = 16,
    ) -> None:
        super().__init__()
        if query_width <= 0 or feature_width <= 0 or hidden_width <= 0 or rank <= 0:
            raise ValueError("Selector widths and rank must be positive.")
        self.query_width = int(query_width)
        self.feature_width = int(feature_width)
        self.rank = int(rank)
        self.salience = nn.Sequential(
            nn.Linear(feature_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        self.query_projection = nn.Linear(query_width, rank, bias=False)
        self.feature_projection = nn.Linear(feature_width, rank, bias=False)
        self.interaction_scale = rank**-0.5

    def forward(
        self,
        features: torch.Tensor,
        query_features: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if features.shape[:-1] != token_mask.shape:
            raise ValueError("Selector features and token mask are not aligned.")
        if features.shape[-1] != self.feature_width:
            raise ValueError("Unexpected selector feature width.")
        if query_features.shape[-1] != self.query_width:
            raise ValueError("Unexpected query feature width.")
        if features.shape[:-3] != query_features.shape[:-1]:
            raise ValueError("Query batch dimensions do not match selector features.")
        salience = self.salience(features).squeeze(-1)
        query = self.query_projection(query_features)
        singleton_dims = features.ndim - query.ndim
        query = query.reshape(
            *query.shape[:-1], *([1] * singleton_dims), query.shape[-1]
        )
        interaction = (
            self.feature_projection(features) * query
        ).sum(dim=-1) * self.interaction_scale
        return (salience + interaction).masked_fill(~token_mask, float("-inf"))


def differentiable_landmark_scores(
    selector_logits: torch.Tensor,
    token_responses: torch.Tensor,
    token_mask: torch.Tensor,
    m: int,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Approximate chunk response using the selector's hard top-``m`` support.

    Top-k membership is piecewise constant, while softmax weights on the chosen
    logits carry gradients. Final evaluation must gather native keys and call
    :func:`qk_response_scores`; this surrogate is only a training objective.
    """
    if selector_logits.shape != token_responses.shape or token_mask.shape != selector_logits.shape:
        raise ValueError("Selector logits, token responses, and masks must align.")
    if m <= 0 or temperature <= 0:
        raise ValueError("m and temperature must be positive.")
    count = min(int(m), selector_logits.shape[-1])
    top_logits, top_indices = selector_logits.topk(count, dim=-1)
    top_responses = token_responses.gather(-1, top_indices)
    finite = torch.isfinite(top_logits)
    weights = torch.softmax(top_logits / temperature, dim=-1)
    weights = weights.masked_fill(~finite, 0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (weights * top_responses).sum(dim=-1)


def landmark_training_loss(
    objective: str,
    selector_logits: torch.Tensor,
    token_responses: torch.Tensor,
    token_mask: torch.Tensor,
    positive_chunks: torch.Tensor,
    *,
    m: int,
    teacher_scores: torch.Tensor | None = None,
    oracle_targets: torch.Tensor | None = None,
    budget: int = 4,
    margin: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one of the prespecified Paper 2.8 selector objectives.

    ``listwise`` ranks evidence chunks, ``combined`` adds full-QK response
    distillation, and ``decision_aware`` upweights negatives nearest the final
    top-``budget`` routing boundary. All tensors may include leading batch axes.
    """
    if objective not in LANDMARK_TRAINING_OBJECTIVES:
        raise ValueError(f"Unsupported landmark objective: {objective}")
    if selector_logits.shape != token_mask.shape:
        raise ValueError("Selector logits and masks must align.")
    if positive_chunks.shape != token_mask.shape[:-1]:
        raise ValueError("Positive-chunk mask must omit only the token axis.")
    if budget <= 0:
        raise ValueError("budget must be positive.")
    chunk_scores = differentiable_landmark_scores(
        selector_logits, token_responses, token_mask, m
    )
    candidate_mask = token_mask.any(dim=-1)
    flat_scores = chunk_scores.reshape(-1, chunk_scores.shape[-1])
    flat_candidates = candidate_mask.reshape_as(flat_scores)
    flat_positives = positive_chunks.reshape_as(flat_scores) & flat_candidates

    listwise_terms = []
    response_terms = []
    boundary_terms = []
    teacher_flat = (
        teacher_scores.reshape_as(flat_scores) if teacher_scores is not None else None
    )
    for index, (scores, candidates, positives) in enumerate(
        zip(flat_scores, flat_candidates, flat_positives)
    ):
        valid_scores = scores[candidates]
        valid_positives = positives[candidates]
        if bool(valid_positives.any()):
            target = valid_positives.to(valid_scores.dtype)
            target = target / target.sum()
            listwise_terms.append(-(target * torch.log_softmax(valid_scores, dim=-1)).sum())
        if teacher_flat is not None:
            target_distribution = torch.softmax(teacher_flat[index][candidates], dim=-1)
            response_terms.append(
                torch.nn.functional.kl_div(
                    torch.log_softmax(valid_scores, dim=-1),
                    target_distribution,
                    reduction="sum",
                )
            )
        negative_mask = ~valid_positives
        if bool(valid_positives.any()) and bool(negative_mask.any()):
            boundary_rank = min(int(budget), len(valid_scores)) - 1
            threshold = valid_scores.detach().topk(boundary_rank + 1).values[-1]
            negative_scores = valid_scores[negative_mask]
            boundary_count = min(max(2 * int(budget), 1), len(negative_scores))
            nearest = (negative_scores.detach() - threshold).abs().topk(
                boundary_count, largest=False
            ).indices
            boundary_negatives = negative_scores.index_select(0, nearest)
            positive_scores = valid_scores[valid_positives]
            boundary_terms.append(
                torch.nn.functional.softplus(
                    margin + boundary_negatives.unsqueeze(0) - positive_scores.unsqueeze(1)
                ).mean()
            )

    zero = selector_logits[torch.isfinite(selector_logits)].sum() * 0
    components = {
        "oracle": zero,
        "listwise": torch.stack(listwise_terms).mean() if listwise_terms else zero,
        "response": torch.stack(response_terms).mean() if response_terms else zero,
        "boundary": torch.stack(boundary_terms).mean() if boundary_terms else zero,
    }
    if objective == "oracle_imitation":
        if oracle_targets is None or oracle_targets.shape != token_mask.shape:
            raise ValueError("Oracle imitation requires aligned token targets.")
        valid_logits = selector_logits[token_mask]
        valid_targets = oracle_targets[token_mask].to(valid_logits.dtype)
        positives = valid_targets.sum().clamp_min(1)
        positive_weight = ((valid_targets.numel() - positives) / positives).clamp(max=20)
        components["oracle"] = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_logits, valid_targets, pos_weight=positive_weight
        )
        total = components["oracle"]
    elif objective == "listwise":
        total = components["listwise"]
    elif objective == "combined":
        if teacher_scores is None:
            raise ValueError("Combined training requires full-QK teacher scores.")
        total = components["listwise"] + components["response"]
    else:
        total = components["listwise"] + 2.0 * components["boundary"]
    return total, components


@dataclass(frozen=True)
class ResponseMetrics:
    """Per-example agreement between full and compressed chunk responses."""

    mae: float
    rmse: float
    spearman: float
    kendall: float
    kl: float
    topk_overlap: dict[int, float]


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), device=values.device, dtype=torch.float32)
    return ranks


def response_metrics(
    teacher: torch.Tensor,
    estimate: torch.Tensor,
    *,
    topk: Iterable[int] = (1, 2, 4, 8),
) -> ResponseMetrics:
    """Measure score, ranking, distribution, and top-k selection preservation."""
    teacher = teacher.flatten().float()
    estimate = estimate.flatten().float()
    if teacher.shape != estimate.shape or teacher.numel() == 0:
        raise ValueError("Teacher and estimate must be equally sized non-empty vectors.")
    error = estimate - teacher
    if teacher.numel() == 1:
        spearman = kendall = 1.0
    else:
        left, right = _rankdata(teacher), _rankdata(estimate)
        spearman = float(torch.corrcoef(torch.stack((left, right)))[0, 1].item())
        first, second = torch.triu_indices(
            len(teacher), len(teacher), offset=1, device=teacher.device
        )
        products = (teacher[first] - teacher[second]) * (
            estimate[first] - estimate[second]
        )
        concordant = int((products > 0).sum().item())
        discordant = int((products < 0).sum().item())
        kendall = (concordant - discordant) / max(concordant + discordant, 1)
    teacher_log = torch.log_softmax(teacher, dim=0)
    estimate_log = torch.log_softmax(estimate, dim=0)
    kl = torch.sum(teacher_log.exp() * (teacher_log - estimate_log))
    overlaps = {}
    for requested in topk:
        count = min(int(requested), len(teacher))
        expected = set(teacher.topk(count).indices.tolist())
        actual = set(estimate.topk(count).indices.tolist())
        overlaps[int(requested)] = len(expected & actual) / count
    return ResponseMetrics(
        mae=float(error.abs().mean().item()),
        rmse=float(torch.sqrt(torch.mean(error.square())).item()),
        spearman=spearman,
        kendall=kendall,
        kl=float(kl.item()),
        topk_overlap=overlaps,
    )


def routing_metrics(
    ranking: Sequence[int],
    positive_mask: torch.Tensor,
    *,
    budget: int,
) -> dict[str, float]:
    """Compute Paper 2.5/2.6-style evidence metrics at a fixed chunk budget."""
    selected = list(ranking[: min(int(budget), len(ranking))])
    positives = set(torch.nonzero(positive_mask, as_tuple=False).flatten().tolist())
    hits = [index in positives for index in selected]
    true_positive = sum(hits)
    groups: list[set[int]] = []
    for index in sorted(positives):
        if not groups or index != max(groups[-1]) + 1:
            groups.append(set())
        groups[-1].add(index)
    reciprocal_rank = next(
        (1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0
    )
    return {
        "evidence_recall": true_positive / max(len(positives), 1),
        "evidence_precision": true_positive / max(len(selected), 1),
        "any_evidence": float(true_positive > 0),
        "chain_completion": float(
            bool(groups) and all(bool(group.intersection(selected)) for group in groups)
        ),
        "exact_identity": float(bool(positives) and set(selected) == positives),
        "mrr": reciprocal_rank,
        "requested_chunks": float(len(selected)),
    }
