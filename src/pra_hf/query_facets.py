"""Bounded root discovery from contextual query facets and native Q/K heads.

The functions in this module operate only on states captured from one complete
query forward pass. They never tokenize or encode a window independently. The
semantic path preserves the Paper-2 learned query-to-memory projection; the
native path preserves real query and K/V head identities, including GQA.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .native_closure import gqa_query_to_kv_heads


_REDUCTIONS = {"max", "mean", "top_m_mean"}


@dataclass(frozen=True)
class QueryFacetProvenance:
    """Identify one global or local view in the original contextual sequence."""

    kind: str
    token_start: int
    token_end: int
    family: str = "global"
    scale: int | None = None


@dataclass(frozen=True)
class QueryFacetSet:
    """Contextual query facets and optional native heads.

    ``hidden`` has shape ``[facets, hidden_width]``. ``native_query`` is either
    absent or has shape ``[facets, query_heads, head_dim]``. When present, the
    global final-token query is the first row. Local-only controls intentionally
    omit it; all remaining rows pool already contextualized states.
    """

    hidden: torch.Tensor
    native_query: torch.Tensor | None
    provenance: tuple[QueryFacetProvenance, ...]

    def __post_init__(self) -> None:
        if self.hidden.ndim != 2 or self.hidden.shape[0] == 0:
            raise ValueError("Facet hidden states must have shape [facets,width].")
        if len(self.provenance) != self.hidden.shape[0]:
            raise ValueError("Facet provenance must align with facet states.")
        global_rows = [
            index for index, row in enumerate(self.provenance) if row.kind == "global"
        ]
        if global_rows and global_rows != [0]:
            raise ValueError("A global query facet must be the first and only global row.")
        if self.native_query is not None:
            if self.native_query.ndim != 3:
                raise ValueError(
                    "Native facet queries must have shape [facets,heads,head_dim]."
                )
            if self.native_query.shape[0] != self.hidden.shape[0]:
                raise ValueError("Semantic and native query facets must align.")


@dataclass(frozen=True)
class FacetScoreResult:
    """Reduced parent scores plus facet/head provenance and measured work.

    ``component_scores`` has shape ``[facets,heads,parents]``. Semantic
    scoring uses one head; native scoring retains every real query head.
    """

    scores: torch.Tensor
    component_scores: torch.Tensor
    winning_facet: torch.Tensor
    winning_head: torch.Tensor
    comparisons: int

    def __post_init__(self) -> None:
        if self.scores.ndim != 1:
            raise ValueError("Reduced scores must have shape [parents].")
        if self.component_scores.ndim != 3:
            raise ValueError("Component scores must have shape [facets,heads,parents].")
        parents = self.scores.shape[0]
        if self.component_scores.shape[2] != parents:
            raise ValueError("Component and reduced parent scores must align.")
        if self.winning_facet.shape != (parents,) or self.winning_head.shape != (
            parents,
        ):
            raise ValueError("Winning facet/head provenance must have shape [parents].")


@dataclass(frozen=True)
class BoundedParentSelection:
    """One globally bounded parent set after optional per-head nominations."""

    parent_indices: tuple[int, ...]
    nominated_parent_indices: tuple[int, ...]
    deduplicated_candidates: int
    final_budget: int


def contextual_window_spans(
    question_span: tuple[int, int],
    window: int,
    stride: int,
) -> tuple[tuple[int, int], ...]:
    """Cover one question span with deterministic overlapping token windows."""
    start, end = map(int, question_span)
    if start < 0 or end <= start:
        raise ValueError("Question span must be non-empty and non-negative.")
    if window <= 0 or stride <= 0:
        raise ValueError("Query window and stride must be positive.")
    length = end - start
    if length <= window:
        return ((start, end),)
    starts = list(range(start, end - window + 1, stride))
    final_start = end - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple((offset, offset + window) for offset in starts)


def clip_query_support(
    token_count: int,
    *,
    support_span: tuple[int, int] | None = None,
    max_support_tokens: int | None = None,
) -> tuple[int, int]:
    """Return a bounded suffix of an explicit query-support region.

    Tokens before this region remain causally represented in its contextual
    states, but cannot issue independent retrieval nominations.
    """
    if token_count <= 0:
        raise ValueError("A query sequence must contain at least one token.")
    start, end = support_span or (0, token_count)
    start, end = int(start), int(end)
    if start < 0 or end <= start or end > token_count:
        raise ValueError("Query-support span must fit the non-empty sequence.")
    if max_support_tokens is not None:
        if max_support_tokens <= 0:
            raise ValueError("max_support_tokens must be positive when provided.")
        start = max(start, end - int(max_support_tokens))
    return start, end


def deterministic_phrase_spans(
    token_texts: Sequence[str],
    support_span: tuple[int, int],
    *,
    neighborhood: int = 4,
) -> tuple[tuple[int, int, str], ...]:
    """Create parser-free clause and relation-neighborhood token spans."""
    start, end = clip_query_support(len(token_texts), support_span=support_span)
    if neighborhood <= 0:
        raise ValueError("Phrase neighborhood must be positive.")
    spans: list[tuple[int, int, str]] = []
    clause_start = start
    for index in range(start, end):
        if re.search(r"[?!.;:\n]", token_texts[index]):
            if index + 1 > clause_start:
                spans.append((clause_start, index + 1, "clause"))
            clause_start = index + 1
    if clause_start < end:
        spans.append((clause_start, end, "clause"))

    cues = {
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "before", "after", "between", "both", "from", "with", "than", "during",
    }
    left_radius = max(1, neighborhood // 2)
    for index in range(start, end):
        normalized = re.sub(r"[^a-z]", "", token_texts[index].lower())
        if normalized in cues:
            left = max(start, index - left_radius)
            right = min(end, left + neighborhood)
            left = max(start, right - neighborhood)
            spans.append((left, right, "relation_neighborhood"))
    return tuple(dict.fromkeys(spans))


def build_span_query_facets(
    hidden_states: torch.Tensor,
    spans: Sequence[tuple[int, int] | tuple[int, int, str]],
    *,
    include_global: bool = True,
    family: str = "span",
    native_query: torch.Tensor | None = None,
) -> QueryFacetSet:
    """Pool deterministic spans from one contextual state sequence."""
    if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
        raise ValueError("Hidden states must have shape [tokens,width].")
    if native_query is not None and (
        native_query.ndim != 3 or native_query.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError("Native query states must align with hidden states.")
    rows = [hidden_states[-1]] if include_global else []
    native_rows = (
        [native_query[-1]] if include_global and native_query is not None else []
    )
    provenance = (
        [QueryFacetProvenance("global", hidden_states.shape[0] - 1, hidden_states.shape[0])]
        if include_global
        else []
    )
    seen: set[tuple[int, int]] = set()
    for item in spans:
        start, end = int(item[0]), int(item[1])
        label = str(item[2]) if len(item) == 3 else family
        if start < 0 or end <= start or end > hidden_states.shape[0]:
            raise ValueError("Facet span must fit the contextual state sequence.")
        if (start, end) in seen:
            continue
        seen.add((start, end))
        rows.append(hidden_states[start:end].mean(dim=0))
        if native_query is not None:
            native_rows.append(native_query[start:end].mean(dim=0))
        provenance.append(
            QueryFacetProvenance("local", start, end, label, end - start)
        )
    if not rows:
        raise ValueError("At least one global or local query facet is required.")
    return QueryFacetSet(
        hidden=torch.stack(rows),
        native_query=torch.stack(native_rows) if native_query is not None else None,
        provenance=tuple(provenance),
    )


def build_multiscale_query_facets(
    hidden_states: torch.Tensor,
    support_span: tuple[int, int],
    *,
    windows: Sequence[int] = (2, 4, 8, 16),
    include_global: bool = True,
    native_query: torch.Tensor | None = None,
) -> QueryFacetSet:
    """Union overlapping contextual windows at fixed resolutions."""
    spans: list[tuple[int, int, str]] = []
    for window in windows:
        if window <= 0:
            raise ValueError("Multiscale windows must be positive.")
        for start, end in contextual_window_spans(
            support_span, int(window), max(1, int(window) // 2)
        ):
            spans.append((start, end, f"window_{int(window)}"))
    return build_span_query_facets(
        hidden_states,
        spans,
        include_global=include_global,
        family="multiscale",
        native_query=native_query,
    )


def build_token_query_facets(
    hidden_states: torch.Tensor,
    support_span: tuple[int, int],
    *,
    include_global: bool = True,
    native_query: torch.Tensor | None = None,
) -> QueryFacetSet:
    """Retain each contextual support token as one finest-scale facet."""
    start, end = clip_query_support(
        hidden_states.shape[0], support_span=support_span
    )
    return build_span_query_facets(
        hidden_states,
        [(index, index + 1, "token") for index in range(start, end)],
        include_global=include_global,
        family="token",
        native_query=native_query,
    )


def build_contextual_query_facets(
    hidden_states: torch.Tensor,
    question_span: tuple[int, int],
    *,
    window: int,
    stride: int,
    include_global: bool = True,
    native_query: torch.Tensor | None = None,
) -> QueryFacetSet:
    """Derive global+local facets from one fully contextualized query sequence.

    ``hidden_states`` is ``[tokens,width]`` and ``native_query`` is optionally
    ``[tokens,query_heads,head_dim]`` from the same model pass. No encoder or
    callback is accepted, making independent window re-encoding impossible.
    """
    if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
        raise ValueError("Hidden states must have shape [tokens,width].")
    if native_query is not None:
        if native_query.ndim != 3 or native_query.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "Native query states must have shape [tokens,heads,head_dim]."
            )
    spans = contextual_window_spans(question_span, window, stride)
    if spans[-1][1] > hidden_states.shape[0]:
        raise ValueError("Question windows must fit the contextual state sequence.")

    return build_span_query_facets(
        hidden_states,
        [(start, end, f"window_{window}") for start, end in spans],
        include_global=include_global,
        family="window",
        native_query=native_query,
    )


def global_query_facet(
    hidden_states: torch.Tensor,
    native_query: torch.Tensor | None = None,
) -> QueryFacetSet:
    """Return only the exact existing final-token root representation."""
    if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
        raise ValueError("Hidden states must have shape [tokens,width].")
    if native_query is not None and (
        native_query.ndim != 3 or native_query.shape[0] != hidden_states.shape[0]
    ):
        raise ValueError("Native query states must align with hidden states.")
    return QueryFacetSet(
        hidden=hidden_states[-1:].clone(),
        native_query=(native_query[-1:].clone() if native_query is not None else None),
        provenance=(
            QueryFacetProvenance(
                "global", hidden_states.shape[0] - 1, hidden_states.shape[0]
            ),
        ),
    )


def _top_m_mean(values: torch.Tensor, dimension: int, top_m: int) -> torch.Tensor:
    if top_m <= 0:
        raise ValueError("top_m must be positive.")
    count = min(top_m, values.shape[dimension])
    return torch.topk(values, k=count, dim=dimension).values.mean(dim=dimension)


def _reduce_components(
    components: torch.Tensor,
    *,
    facet_reduction: str,
    head_reduction: str,
    top_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if facet_reduction not in {"max", "top_m_mean"}:
        raise ValueError(f"Unsupported facet reduction: {facet_reduction}")
    if head_reduction not in _REDUCTIONS:
        raise ValueError(f"Unsupported head reduction: {head_reduction}")
    if head_reduction == "max":
        per_facet = components.max(dim=1).values
    elif head_reduction == "mean":
        per_facet = components.mean(dim=1)
    else:
        per_facet = _top_m_mean(components, 1, top_m)
    if facet_reduction == "max":
        reduced = per_facet.max(dim=0).values
    else:
        reduced = _top_m_mean(per_facet, 0, top_m)
    flattened = components.reshape(-1, components.shape[-1])
    winners = flattened.argmax(dim=0)
    winning_facet = winners // components.shape[1]
    winning_head = winners % components.shape[1]
    return reduced, winning_facet, winning_head


def score_semantic_query_facets(
    projected_query_facets: torch.Tensor,
    projected_parent_memory: torch.Tensor,
    *,
    facet_reduction: str = "max",
    top_m: int = 2,
) -> FacetScoreResult:
    """Cosine-score projected semantic facets against fixed parent gists."""
    if projected_query_facets.ndim != 2 or projected_parent_memory.ndim != 2:
        raise ValueError("Semantic query and memory tensors must be rank two.")
    if projected_query_facets.shape[1] != projected_parent_memory.shape[1]:
        raise ValueError("Semantic query and memory widths must match.")
    query = F.normalize(projected_query_facets.float(), dim=-1)
    memory = F.normalize(projected_parent_memory.float(), dim=-1)
    components = (query @ memory.T).unsqueeze(1)
    scores, winning_facet, _ = _reduce_components(
        components,
        facet_reduction=facet_reduction,
        head_reduction="mean",
        top_m=top_m,
    )
    return FacetScoreResult(
        scores=scores,
        component_scores=components,
        winning_facet=winning_facet,
        winning_head=torch.full_like(winning_facet, -1),
        comparisons=components.shape[0] * components.shape[2],
    )


def pool_parent_native_keys(
    local_keys: torch.Tensor,
    token_mask: torch.Tensor,
    local_parent_indices: torch.Tensor,
    parent_count: int,
) -> torch.Tensor:
    """Pool fixed local native keys into ``[parents,kv_heads,head_dim]``.

    The implementation packs every valid token and uses ``index_add_``; it does
    not loop over parents, local regions, tokens, or heads.
    """
    if local_keys.ndim != 4:
        raise ValueError("Local keys must have shape [locals,tokens,kv_heads,head_dim].")
    if token_mask.shape != local_keys.shape[:2]:
        raise ValueError("Token mask must align with local native keys.")
    if local_parent_indices.shape != (local_keys.shape[0],):
        raise ValueError("Every local key region needs one parent index.")
    if parent_count <= 0:
        raise ValueError("parent_count must be positive.")
    if local_parent_indices.numel() and (
        int(local_parent_indices.min()) < 0
        or int(local_parent_indices.max()) >= parent_count
    ):
        raise ValueError("Local parent indices are out of range.")

    locals_, tokens, kv_heads, head_dim = local_keys.shape
    parent_rows = (
        local_parent_indices.to(local_keys.device)
        .unsqueeze(1)
        .expand(locals_, tokens)
        .reshape(-1)
    )
    valid = token_mask.to(local_keys.device).reshape(-1)
    packed_keys = local_keys.reshape(-1, kv_heads, head_dim)[valid].float()
    packed_parents = parent_rows[valid]
    sums = packed_keys.new_zeros((parent_count, kv_heads, head_dim))
    counts = packed_keys.new_zeros((parent_count, 1, 1))
    sums.index_add_(0, packed_parents, packed_keys)
    counts.index_add_(
        0,
        packed_parents,
        packed_keys.new_ones((packed_keys.shape[0], 1, 1)),
    )
    if bool((counts == 0).any()):
        raise ValueError("Every parent must contain at least one valid native key.")
    return sums / counts


def score_native_query_facets(
    native_query_facets: torch.Tensor,
    parent_native_keys: torch.Tensor,
    *,
    facet_reduction: str = "max",
    head_reduction: str = "max",
    top_m: int = 4,
) -> FacetScoreResult:
    """Score real query heads only against their GQA-compatible parent K heads."""
    if native_query_facets.ndim != 3 or parent_native_keys.ndim != 3:
        raise ValueError(
            "Native query/key tensors must be [facets,Hq,Dh] and [parents,Hkv,Dh]."
        )
    if native_query_facets.shape[-1] != parent_native_keys.shape[-1]:
        raise ValueError("Native query and key head widths must match.")
    query_heads = native_query_facets.shape[1]
    kv_heads = parent_native_keys.shape[1]
    mapping = gqa_query_to_kv_heads(query_heads, kv_heads).to(
        parent_native_keys.device
    )
    compatible_keys = parent_native_keys[:, mapping, :]
    components = torch.einsum(
        "fhd,phd->fhp",
        native_query_facets.float(),
        compatible_keys.float(),
    ) / math.sqrt(native_query_facets.shape[-1])
    scores, winning_facet, winning_head = _reduce_components(
        components,
        facet_reduction=facet_reduction,
        head_reduction=head_reduction,
        top_m=top_m,
    )
    return FacetScoreResult(
        scores=scores,
        component_scores=components,
        winning_facet=winning_facet,
        winning_head=winning_head,
        comparisons=int(components.numel()),
    )


def _deterministic_topk(
    scores: torch.Tensor,
    k: int,
    candidates: Sequence[int] | None = None,
) -> list[int]:
    pool = range(scores.numel()) if candidates is None else candidates
    finite = [index for index in pool if math.isfinite(float(scores[index]))]
    finite.sort(key=lambda index: (-float(scores[index]), int(index)))
    return [int(index) for index in finite[: max(0, k)]]


def select_bounded_parents(
    result: FacetScoreResult,
    budget: int,
    *,
    per_head_nomination_k: int = 0,
) -> BoundedParentSelection:
    """Deduplicate all nominations before enforcing one final parent budget."""
    parents = result.scores.numel()
    final_budget = min(max(0, int(budget)), parents)
    nominated: set[int] = set()
    if per_head_nomination_k > 0:
        # Facets share each real head's nomination allowance; they do not each
        # receive an independent final parent budget.
        per_head = result.component_scores.max(dim=0).values
        for head_scores in per_head:
            nominated.update(
                _deterministic_topk(head_scores, per_head_nomination_k)
            )
    if nominated:
        selected = _deterministic_topk(
            result.scores, final_budget, sorted(nominated)
        )
        if len(selected) < final_budget:
            remainder = [index for index in range(parents) if index not in selected]
            selected.extend(
                _deterministic_topk(
                    result.scores, final_budget - len(selected), remainder
                )
            )
    else:
        selected = _deterministic_topk(result.scores, final_budget)
    return BoundedParentSelection(
        parent_indices=tuple(selected),
        nominated_parent_indices=tuple(sorted(nominated)),
        deduplicated_candidates=len(nominated) if nominated else parents,
        final_budget=final_budget,
    )


def target_rank_metrics(
    scores: torch.Tensor,
    targets: set[int],
    selected: set[int],
    *,
    cutoffs: Sequence[int] = (1, 2, 4, 8),
) -> dict[str, float | int | None]:
    """Measure best-target rank, margins, entropy, and bounded-set retrieval."""
    if scores.ndim != 1:
        raise ValueError("Target metrics require one score per parent.")
    valid_targets = sorted(
        index
        for index in targets
        if 0 <= index < scores.numel() and math.isfinite(float(scores[index]))
    )
    if not valid_targets:
        raise ValueError("At least one finite target parent is required.")
    target = max(valid_targets, key=lambda index: (float(scores[index]), -index))
    target_score = float(scores[target])
    finite = [
        index for index in range(scores.numel()) if math.isfinite(float(scores[index]))
    ]
    rank = 1 + sum(float(scores[index]) > target_score for index in finite)
    distractors = [index for index in finite if index not in targets]
    distractor = max(
        distractors,
        key=lambda index: (float(scores[index]), -index),
        default=None,
    )
    distractor_score = (
        float(scores[distractor]) if distractor is not None else float("-inf")
    )
    probabilities = torch.softmax(scores[finite].float(), dim=0)
    entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()
    )
    intersection, union = selected & targets, selected | targets
    output: dict[str, float | int | None] = {
        "target_parent": target,
        "target_rank": rank,
        "mrr": 1.0 / rank,
        "target_score": target_score,
        "best_distractor_parent": distractor,
        "best_distractor_score": distractor_score,
        "oracle_margin": target_score - distractor_score,
        "score_entropy": entropy,
        "target_present": float(bool(intersection)),
        "oracle_recall": len(intersection) / len(targets),
        "oracle_precision": len(intersection) / len(selected) if selected else 0.0,
        "oracle_jaccard": len(intersection) / len(union) if union else 1.0,
        "false_positive_parent_count": len(selected - targets),
    }
    for cutoff in cutoffs:
        output[f"recall_at_{int(cutoff)}"] = float(rank <= int(cutoff))
    return output
