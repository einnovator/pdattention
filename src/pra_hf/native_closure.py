"""Tokenwise native-QK propagation after bounded semantic narrowing.

The classes here operate only on routing tensors. They never own or request a
native K/V payload; selected parent identities cross that boundary only after
closure has stopped.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .iterative import (
    IterativeGistRouter,
    IterativeRoutingResult,
    RetrievalEdge,
    RetrievalGraph,
    RetrievalNode,
)


_TOKEN_REDUCTIONS = {"max", "top_m_mean"}
_HEAD_REDUCTIONS = {"max", "mean", "top_m_mean"}
_TRANSITION_MODES = {"topk", "threshold"}


@dataclass(frozen=True)
class NativeQKRoutingConfig:
    """Bound semantic narrowing and one native local Q-to-K propagation hop."""

    max_unique_parents: int
    candidate_pool_fraction: float = 0.20
    initial_parent_count: int | None = None
    branch_top_k: int | None = None
    root_anchor_alpha: float = 0.25
    token_reduction: str = "max"
    head_reduction: str = "max"
    top_m: int = 4
    transition_mode: str = "topk"
    threshold_lambda: float = 1.0

    def __post_init__(self) -> None:
        if self.max_unique_parents < 0:
            raise ValueError("max_unique_parents must be non-negative.")
        if not 0.0 < self.candidate_pool_fraction <= 1.0:
            raise ValueError("candidate_pool_fraction must lie in (0, 1].")
        if self.initial_parent_count is not None and self.initial_parent_count < 0:
            raise ValueError("initial_parent_count must be non-negative.")
        if self.branch_top_k is not None and self.branch_top_k < 0:
            raise ValueError("branch_top_k must be non-negative.")
        if not 0.0 <= self.root_anchor_alpha <= 1.0:
            raise ValueError("root_anchor_alpha must lie in [0, 1].")
        if self.token_reduction not in _TOKEN_REDUCTIONS:
            raise ValueError(f"Unsupported token reduction: {self.token_reduction}")
        if self.head_reduction not in _HEAD_REDUCTIONS:
            raise ValueError(f"Unsupported head reduction: {self.head_reduction}")
        if self.transition_mode not in _TRANSITION_MODES:
            raise ValueError(f"Unsupported transition mode: {self.transition_mode}")
        if self.top_m <= 0:
            raise ValueError("top_m must be positive.")


@dataclass(frozen=True)
class NativeQKIndex:
    """Aligned semantic local gists and contextual tokenwise pre-RoPE Q/K.

    Semantic parent/local tensors are ``[P,Dr]`` and ``[L,Dr]``. Native query
    tensors are ``[L,T,Hq,Dh]`` and keys are ``[L,T,Hkv,Dh]``. ``token_mask``
    marks valid positions in the padded local windows.
    """

    parent_ids: tuple[str, ...]
    parent_spans: tuple[tuple[int, int], ...]
    parent_memory_gists: torch.Tensor
    local_spans: tuple[tuple[int, int], ...]
    local_parent_indices: torch.Tensor
    local_memory_gists: torch.Tensor
    local_query_gists: torch.Tensor
    local_pre_query: torch.Tensor
    local_pre_key: torch.Tensor
    token_mask: torch.Tensor
    layer_id: int = 27

    def __post_init__(self) -> None:
        parents, locals_ = len(self.parent_ids), len(self.local_spans)
        if len(self.parent_spans) != parents:
            raise ValueError("Parent identities and spans must align.")
        if self.parent_memory_gists.ndim != 2 or self.parent_memory_gists.shape[0] != parents:
            raise ValueError("Parent memory gists must have shape [parents,width].")
        if self.local_memory_gists.ndim != 2 or self.local_memory_gists.shape[0] != locals_:
            raise ValueError("Local memory gists must have shape [locals,width].")
        if self.local_query_gists.shape != self.local_memory_gists.shape:
            raise ValueError("Local semantic query and memory tensors must align.")
        if self.local_parent_indices.shape != (locals_,):
            raise ValueError("local_parent_indices must have shape [locals].")
        if self.local_pre_query.ndim != 4 or self.local_pre_query.shape[0] != locals_:
            raise ValueError("Native queries must have shape [locals,tokens,Hq,head_dim].")
        if self.local_pre_key.ndim != 4 or self.local_pre_key.shape[0] != locals_:
            raise ValueError("Native keys must have shape [locals,tokens,Hkv,head_dim].")
        if self.local_pre_query.shape[1] != self.local_pre_key.shape[1]:
            raise ValueError("Native query/key windows must have equal padded length.")
        if self.local_pre_query.shape[-1] != self.local_pre_key.shape[-1]:
            raise ValueError("Native query/key head widths must match.")
        if self.token_mask.shape != self.local_pre_query.shape[:2]:
            raise ValueError("token_mask must have shape [locals,tokens].")
        if self.local_pre_query.shape[2] % self.local_pre_key.shape[2]:
            raise ValueError("Query heads must be divisible by native K/V heads.")
        if locals_ and (
            int(self.local_parent_indices.min()) < 0
            or int(self.local_parent_indices.max()) >= parents
        ):
            raise ValueError("Every local region must map to a valid parent.")

    @property
    def device(self) -> torch.device:
        return self.parent_memory_gists.device


def gqa_query_to_kv_heads(num_query_heads: int, num_kv_heads: int) -> torch.Tensor:
    """Map each query head to its model-compatible grouped-query K/V head."""
    if num_query_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("Head counts must be positive.")
    if num_query_heads % num_kv_heads:
        raise ValueError("Query heads must be divisible by K/V heads for GQA.")
    return torch.arange(num_query_heads) // (num_query_heads // num_kv_heads)


@dataclass(frozen=True)
class NativeQKScores:
    """Reduced source-by-candidate scores plus strongest token/head identities."""

    scores: torch.Tensor
    query_token: torch.Tensor
    key_token: torch.Tensor
    query_head: torch.Tensor
    kv_head: torch.Tensor
    dot_products: int


def native_local_qk_scores(
    source_query: torch.Tensor,
    candidate_key: torch.Tensor,
    source_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    token_reduction: str = "max",
    head_reduction: str = "max",
    top_m: int = 4,
) -> NativeQKScores:
    """Score all packed local pairs with batched model-compatible GQA dots.

    No Python loop traverses token pairs or heads. Reductions are transparent:
    max or Top-M token-pair mean per compatible query head, followed by max,
    mean, or Top-M mean across query heads.
    """
    if token_reduction not in _TOKEN_REDUCTIONS:
        raise ValueError(f"Unsupported token reduction: {token_reduction}")
    if head_reduction not in _HEAD_REDUCTIONS:
        raise ValueError(f"Unsupported head reduction: {head_reduction}")
    if source_query.ndim != 4 or candidate_key.ndim != 4:
        raise ValueError("Native Q/K tensors must be rank four.")
    if source_query.shape[-1] != candidate_key.shape[-1]:
        raise ValueError("Native Q/K head widths must match.")
    if source_mask.shape != source_query.shape[:2]:
        raise ValueError("source_mask does not align with source queries.")
    if candidate_mask.shape != candidate_key.shape[:2]:
        raise ValueError("candidate_mask does not align with candidate keys.")
    q_heads, kv_heads = source_query.shape[2], candidate_key.shape[2]
    mapping = gqa_query_to_kv_heads(q_heads, kv_heads).to(candidate_key.device)
    mapped_key = candidate_key[:, :, mapping, :]
    logits = torch.einsum(
        "fahd,cbhd->fcabh",
        source_query.float(),
        mapped_key.float(),
    ) / math.sqrt(source_query.shape[-1])
    valid = source_mask[:, None, :, None, None] & candidate_mask[None, :, None, :, None]
    logits = logits.masked_fill(~valid, float("-inf"))
    flat_all = logits.reshape(logits.shape[0], logits.shape[1], -1)
    strongest = flat_all.argmax(dim=-1)
    token_pairs_per_head = logits.shape[2] * logits.shape[3]
    query_head = strongest % q_heads
    pair_index = strongest // q_heads
    query_token = pair_index // logits.shape[3]
    key_token = pair_index % logits.shape[3]
    if token_reduction == "max":
        per_head = logits.amax(dim=(2, 3))
    else:
        token_flat = logits.permute(0, 1, 4, 2, 3).reshape(
            logits.shape[0], logits.shape[1], q_heads, token_pairs_per_head
        )
        count = min(top_m, token_pairs_per_head)
        strongest_pairs = torch.topk(token_flat, k=count, dim=-1).values
        finite_pairs = torch.isfinite(strongest_pairs)
        per_head = strongest_pairs.masked_fill(~finite_pairs, 0.0).sum(dim=-1) / (
            finite_pairs.sum(dim=-1).clamp_min(1)
        )
    if head_reduction == "max":
        reduced = per_head.max(dim=-1).values
    elif head_reduction == "mean":
        reduced = per_head.mean(dim=-1)
    else:
        reduced = torch.topk(per_head, k=min(top_m, q_heads), dim=-1).values.mean(dim=-1)
    kv_head = mapping[query_head]
    source_tokens = source_mask.sum(dim=1, dtype=torch.int64)
    candidate_tokens = candidate_mask.sum(dim=1, dtype=torch.int64)
    dots = int((source_tokens[:, None] * candidate_tokens[None, :]).sum().item()) * q_heads
    return NativeQKScores(
        reduced, query_token, key_token, query_head, kv_head, dots
    )


class NativeLocalQKRouter:
    """Use semantic gists for narrowing and native token Q/K for propagation."""

    def __init__(self, index: NativeQKIndex):
        self.index = index

    def _best_root_local(self, direct_local: torch.Tensor, parent: int) -> int:
        rows = torch.nonzero(self.index.local_parent_indices == parent).flatten()
        return int(rows[torch.argmax(direct_local[rows])])

    def route(
        self,
        root_query: torch.Tensor,
        config: NativeQKRoutingConfig,
        *,
        example_id: str | None = None,
        evidence_parent_ids: set[str] | None = None,
    ) -> IterativeRoutingResult:
        """Return final parent identities without touching post-RoPE payload K/V."""
        root = F.normalize(root_query.reshape(-1).to(self.index.device).float(), dim=-1)
        pm = F.normalize(self.index.parent_memory_gists.float(), dim=-1)
        lm = F.normalize(self.index.local_memory_gists.float(), dim=-1)
        lq = F.normalize(self.index.local_query_gists.float(), dim=-1)
        direct_parent, direct_local = pm @ root, lm @ root
        graph = RetrievalGraph(
            example_id,
            self.index.layer_id,
            {
                "node_id": "__root__",
                "representation_type": "semantic_gist",
                "projection_type": "query",
            },
            budget=asdict(config),
        )
        budget = min(config.max_unique_parents, len(self.index.parent_ids))
        if budget == 0:
            graph.stop_reason = "zero_limit"
            return IterativeRoutingResult((), tuple(direct_parent.cpu().tolist()), graph)
        initial_count = config.initial_parent_count
        if initial_count is None:
            initial_count = max(1, math.ceil(budget / 2))
        initial_count = min(initial_count, budget)
        first = IterativeGistRouter._topk(direct_parent, initial_count)
        visited = set(first)
        frontier_locals = [self._best_root_local(direct_local, parent) for parent in first]

        for parent, local in zip(first, frontier_locals):
            parent_id = self.index.parent_ids[parent]
            score = float(direct_parent[parent])
            affinity = (score + 1.0) / 2.0
            node_id = f"{parent_id}#local={local}"
            graph.nodes.append(RetrievalNode(
                node_id=node_id,
                reference_uri=example_id or "memory",
                hop=1,
                parent_ids=["__root__"],
                direct_query_score=score,
                edge_score=score,
                path_score=affinity,
                winning_gist_index=local,
                evidence=parent_id in evidence_parent_ids if evidence_parent_ids is not None else None,
                parent_chunk_id=parent_id,
                local_span=self.index.local_spans[local],
                resolution_level="local",
                representation_type="semantic_gist",
                projection_type="root_query",
            ))
            graph.edges.append(RetrievalEdge(
                "__root__", node_id, 1, score, score, affinity, True,
                edge_type="root_to_local", projection_type="query_to_memory",
                score=score, target_span=self.index.local_spans[local],
            ))

        semantic_comparisons = int(direct_parent.numel() + direct_local.numel())
        candidate_sets: list[list[int]] = []
        candidate_ranks: list[dict[int, int]] = []
        pool_size = max(1, math.ceil(len(self.index.parent_ids) * config.candidate_pool_fraction))
        for local in frontier_locals:
            semantic = lm @ lq[local]
            semantic_comparisons += int(semantic.numel())
            parent_semantic = direct_parent.new_full(
                (len(self.index.parent_ids),), float("-inf")
            )
            for parent in range(len(self.index.parent_ids)):
                rows = torch.nonzero(
                    self.index.local_parent_indices == parent
                ).flatten()
                parent_semantic[parent] = semantic[rows].max()
            if visited:
                parent_semantic[list(visited)] = float("-inf")
            narrowed_parents = IterativeGistRouter._topk(parent_semantic, pool_size)
            rows = torch.nonzero(
                torch.isin(
                    self.index.local_parent_indices,
                    torch.tensor(narrowed_parents, device=self.index.device),
                )
            ).flatten().tolist()
            candidate_sets.append(rows)
            ranks = {parent: rank + 1 for rank, parent in enumerate(narrowed_parents)}
            candidate_ranks.append({
                row: ranks[int(self.index.local_parent_indices[row])] for row in rows
            })
        candidate_union = sorted({row for rows in candidate_sets for row in rows})
        proposed_native = 0
        native_dots = 0
        candidate_tokens = 0
        native_thresholds: list[float] = []
        proposals: dict[int, tuple] = {}
        if candidate_union and frontier_locals and len(visited) < budget:
            source_rows = torch.tensor(frontier_locals, device=self.index.device)
            candidate_rows = torch.tensor(candidate_union, device=self.index.device)
            scored = native_local_qk_scores(
                self.index.local_pre_query[source_rows],
                self.index.local_pre_key[candidate_rows],
                self.index.token_mask[source_rows],
                self.index.token_mask[candidate_rows],
                token_reduction=config.token_reduction,
                head_reduction=config.head_reduction,
                top_m=config.top_m,
            )
            native_dots = scored.dot_products
            candidate_tokens = int(self.index.token_mask[candidate_rows].sum().item())
            remaining = budget - len(visited)
            branch_cap = config.branch_top_k if config.branch_top_k is not None else remaining
            branch_cap = min(branch_cap, remaining)
            for source_row, source_local in enumerate(frontier_locals):
                allowed = torch.tensor(
                    [candidate in candidate_ranks[source_row] for candidate in candidate_union],
                    device=self.index.device,
                    dtype=torch.bool,
                )
                values = scored.scores[source_row].masked_fill(~allowed, float("-inf"))
                threshold = None
                if config.transition_mode == "threshold":
                    finite = values[torch.isfinite(values)]
                    threshold = float(
                        finite.mean() + config.threshold_lambda * finite.std(unbiased=False)
                    ) if finite.numel() else float("inf")
                    values = values.masked_fill(values < threshold, float("-inf"))
                    native_thresholds.append(threshold)
                rows = IterativeGistRouter._topk(values, branch_cap)
                proposed_native += len(rows)
                for packed_row in rows:
                    local = candidate_union[packed_row]
                    parent = int(self.index.local_parent_indices[local])
                    if parent in visited:
                        continue
                    raw = float(scored.scores[source_row, packed_row])
                    native_affinity = float(torch.sigmoid(scored.scores[source_row, packed_row]))
                    direct_affinity = (float(direct_parent[parent]) + 1.0) / 2.0
                    anchored = (
                        config.root_anchor_alpha * direct_affinity
                        + (1.0 - config.root_anchor_alpha) * native_affinity
                    )
                    source_parent = int(self.index.local_parent_indices[source_local])
                    source_path = (float(direct_parent[source_parent]) + 1.0) / 2.0
                    path = source_path * anchored
                    proposal = (
                        path, source_row, source_local, local, raw, anchored,
                        int(scored.query_token[source_row, packed_row]),
                        int(scored.key_token[source_row, packed_row]),
                        int(scored.query_head[source_row, packed_row]),
                        int(scored.kv_head[source_row, packed_row]),
                        candidate_ranks[source_row][local], threshold,
                    )
                    if parent not in proposals or proposal[0] > proposals[parent][0]:
                        proposals[parent] = proposal
            ranking = direct_parent.new_full((len(self.index.parent_ids),), float("-inf"))
            for parent, proposal in proposals.items():
                ranking[parent] = proposal[0]
            accepted_parents = IterativeGistRouter._topk(ranking, remaining)
            for parent in accepted_parents:
                (
                    path, source_row, source_local, local, raw, anchored,
                    query_token, key_token, query_head, kv_head, semantic_rank, threshold,
                ) = proposals[parent]
                parent_id = self.index.parent_ids[parent]
                source_parent = int(self.index.local_parent_indices[source_local])
                source_id = f"{self.index.parent_ids[source_parent]}#local={source_local}"
                node_id = f"{parent_id}#local={local}"
                graph.nodes.append(RetrievalNode(
                    node_id=node_id,
                    reference_uri=example_id or "memory",
                    hop=2,
                    parent_ids=[source_id],
                    direct_query_score=float(direct_parent[parent]),
                    edge_score=raw,
                    path_score=path,
                    winning_gist_index=local,
                    evidence=parent_id in evidence_parent_ids if evidence_parent_ids is not None else None,
                    parent_chunk_id=parent_id,
                    local_span=self.index.local_spans[local],
                    resolution_level="local",
                    representation_type="pre_rope_native_qk",
                    projection_type="native_query_to_key",
                ))
                graph.edges.append(RetrievalEdge(
                    source_id, node_id, 2, raw, anchored, path, True,
                    edge_type="native_qk", representation_type="pre_rope_native_qk",
                    projection_type="native_query_to_key", head_id=query_head,
                    query_head=query_head, kv_head=kv_head, score=raw,
                    threshold=threshold,
                    source_span=self.index.local_spans[source_local],
                    target_span=self.index.local_spans[local],
                    semantic_candidate_rank=semantic_rank,
                ))
                visited.add(parent)

        # Thresholding can accept fewer transitions. Semantic fill preserves the
        # identical final materialization budget required by the experiment.
        if len(visited) < budget:
            fill_scores = direct_parent.clone()
            if visited:
                fill_scores[list(visited)] = float("-inf")
            for parent in IterativeGistRouter._topk(fill_scores, budget - len(visited)):
                local = self._best_root_local(direct_local, parent)
                parent_id = self.index.parent_ids[parent]
                score = float(direct_parent[parent])
                graph.nodes.append(RetrievalNode(
                    node_id=f"{parent_id}#local={local}",
                    reference_uri=example_id or "memory",
                    hop=2,
                    parent_ids=["__root__"],
                    direct_query_score=score,
                    edge_score=score,
                    path_score=(score + 1.0) / 2.0,
                    winning_gist_index=local,
                    selection_reason="semantic_budget_fill",
                    evidence=parent_id in evidence_parent_ids if evidence_parent_ids is not None else None,
                    parent_chunk_id=parent_id,
                    local_span=self.index.local_spans[local],
                    resolution_level="local",
                    representation_type="semantic_gist",
                    projection_type="root_query",
                ))
                visited.add(parent)
        graph.stop_reason = (
            "unique_budget" if len(visited) == budget else "no_new_parents"
        )
        candidate_parents = {
            int(self.index.local_parent_indices[row]) for row in candidate_union
        }
        graph.costs = {
            "semantic_gist_comparisons": semantic_comparisons,
            "native_qk_dot_products": native_dots,
            "native_qk_comparisons": native_dots,
            "candidate_parents": len(candidate_parents),
            "candidate_local_regions": len(candidate_union),
            "candidate_tokens": candidate_tokens,
            "proposed_native_transitions": proposed_native,
            "accepted_native_transitions": sum(
                edge.edge_type == "native_qk" for edge in graph.edges
            ),
            "unique_parents_selected": len(visited),
            "final_kv_tokens": sum(
                self.index.parent_spans[parent][1] - self.index.parent_spans[parent][0]
                for parent in visited
            ),
            "native_threshold_mean": (
                sum(native_thresholds) / len(native_thresholds) if native_thresholds else None
            ),
            "kv_materializations_during_closure": 0,
        }
        return IterativeRoutingResult(
            tuple(sorted(visited)), tuple(direct_parent.cpu().tolist()), graph
        )
