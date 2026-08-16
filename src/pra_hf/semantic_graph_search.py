"""Bounded search over a frozen memory graph with a terminal query goal.

The search treats query facets only as a boundary condition. Native memory-to-
memory scores generate and admit intermediate successors; semantic query-facet
scores are consulted only after a successor has entered the search workspace.
The module contains no task labels and never materializes native K/V payloads.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .native_closure import native_local_qk_scores


_STRATEGIES = {"breadth_first", "best_first", "beam"}


@dataclass(frozen=True)
class SemanticGraphSearchConfig:
    """Resource and score limits for one first-path search."""

    successor_k: int
    max_visited_parents: int | None
    edge_threshold: float
    goal_threshold: float
    max_hops: int = 4
    strategy: str = "breadth_first"
    beam_width: int = 4
    max_expanded_nodes: int = 64
    minimum_goal_depth: int = 1
    different_facet_goal: bool = False

    def __post_init__(self) -> None:
        if self.successor_k <= 0:
            raise ValueError("successor_k must be positive.")
        if self.max_visited_parents is not None and self.max_visited_parents <= 0:
            raise ValueError("max_visited_parents must be positive or None.")
        if self.max_hops < 0:
            raise ValueError("max_hops must be non-negative.")
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"Unsupported search strategy: {self.strategy}")
        if self.beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        if self.max_expanded_nodes <= 0:
            raise ValueError("max_expanded_nodes must be positive.")
        if self.minimum_goal_depth <= 0:
            raise ValueError("minimum_goal_depth must be positive.")


@dataclass(frozen=True)
class SearchState:
    """One admitted parent and the deterministic path that reached it."""

    parent_index: int
    depth: int
    path: tuple[int, ...]
    path_quality: float
    entry_facet: int | None


@dataclass(frozen=True)
class SearchDecision:
    """Provenance for one native Top-K proposal and its admission outcome."""

    source_parent: int
    candidate_parent: int
    hop: int
    native_rank: int
    native_score: float
    passed_edge_threshold: bool
    duplicate: bool
    cycle: bool
    admitted: bool
    goal_best_facet: int | None
    goal_score: float | None
    goal_triggered: bool
    path: tuple[int, ...]
    path_quality: float


@dataclass(frozen=True)
class SemanticGraphSearchResult:
    """First-path result, graph provenance, and routing-only resource counts."""

    roots: tuple[int, ...]
    visited: tuple[int, ...]
    terminal_parent: int | None
    terminal_facet: int | None
    path: tuple[int, ...]
    goal_triggered: bool
    stop_reason: str
    decisions: tuple[SearchDecision, ...]
    nodes_expanded: int
    raw_proposals: int
    edge_admitted_proposals: int
    duplicate_proposals: int
    cycles_prevented: int
    peak_frontier: int
    goal_tests: int
    goal_comparisons: int
    peak_candidate_tensor_bytes: int
    cpu_dedup_seconds: float
    goal_test_seconds: float
    search_seconds: float


@dataclass(frozen=True)
class NativeParentAdjacency:
    """Parent-level max reduction of one packed local native-QK product."""

    scores: torch.Tensor
    dot_products: int
    local_pair_count: int


def build_native_parent_adjacency(
    local_pre_query: torch.Tensor,
    local_pre_key: torch.Tensor,
    token_mask: torch.Tensor,
    local_parent_indices: torch.Tensor,
    parent_count: int,
    *,
    token_reduction: str = "top_m_mean",
    head_reduction: str = "top_m_mean",
    top_m: int = 4,
    source_batch_size: int = 8,
) -> NativeParentAdjacency:
    """Score every local pair once and max-reduce to ``[parents,parents]``.

    Source-local blocks bound the temporary token/head product while every
    block still scores all target locals in one tensorized operation. Scatter
    reduction maps local pairs to parent identities; no query facet or task
    label participates.
    """
    if local_pre_query.ndim != 4 or local_pre_key.ndim != 4:
        raise ValueError("Local native Q/K tensors must be rank four.")
    locals_ = int(local_pre_query.shape[0])
    if local_pre_key.shape[0] != locals_ or token_mask.shape != local_pre_query.shape[:2]:
        raise ValueError("Local native tensors and token mask must align.")
    if local_parent_indices.shape != (locals_,):
        raise ValueError("local_parent_indices must have shape [locals].")
    if parent_count <= 0:
        raise ValueError("parent_count must be positive.")
    if source_batch_size <= 0:
        raise ValueError("source_batch_size must be positive.")
    parent_indices = local_parent_indices.to(local_pre_query.device)
    if int(parent_indices.min()) < 0 or int(parent_indices.max()) >= parent_count:
        raise ValueError("Local parent index is outside parent_count.")
    reduced = local_pre_query.new_full(
        (parent_count * parent_count,), float("-inf"), dtype=torch.float32
    )
    dot_products = 0
    target = parent_indices[None, :]
    for start in range(0, locals_, source_batch_size):
        end = min(start + source_batch_size, locals_)
        scored = native_local_qk_scores(
            local_pre_query[start:end],
            local_pre_key,
            token_mask[start:end],
            token_mask,
            token_reduction=token_reduction,
            head_reduction=head_reduction,
            top_m=top_m,
        )
        source = parent_indices[start:end, None]
        flat_parent_pairs = (source * parent_count + target).reshape(-1)
        reduced.scatter_reduce_(
            0,
            flat_parent_pairs,
            scored.scores.reshape(-1),
            reduce="amax",
            include_self=True,
        )
        dot_products += scored.dot_products
    adjacency = reduced.reshape(parent_count, parent_count)
    adjacency.fill_diagonal_(float("-inf"))
    return NativeParentAdjacency(
        adjacency,
        dot_products,
        locals_ * locals_,
    )


def _validate_inputs(
    edge_scores: torch.Tensor,
    goal_scores: torch.Tensor,
    roots: Sequence[int],
    config: SemanticGraphSearchConfig,
    entry_facets: Mapping[int, int] | None,
) -> tuple[int, tuple[int, ...]]:
    if edge_scores.ndim != 2 or edge_scores.shape[0] != edge_scores.shape[1]:
        raise ValueError("edge_scores must have shape [parents,parents].")
    parents = int(edge_scores.shape[0])
    if goal_scores.ndim != 2 or goal_scores.shape[1] != parents:
        raise ValueError("goal_scores must have shape [facets,parents].")
    if goal_scores.shape[0] == 0:
        raise ValueError("At least one query facet is required.")
    ordered_roots = tuple(dict.fromkeys(int(root) for root in roots))
    if not ordered_roots:
        raise ValueError("At least one root parent is required.")
    if min(ordered_roots) < 0 or max(ordered_roots) >= parents:
        raise ValueError("Root parent is outside the score matrices.")
    if (
        config.max_visited_parents is not None
        and len(ordered_roots) > config.max_visited_parents
    ):
        raise ValueError("Root set exceeds max_visited_parents.")
    if entry_facets:
        for root, facet in entry_facets.items():
            if int(root) not in ordered_roots:
                raise ValueError("entry_facets contains a non-root parent.")
            if not 0 <= int(facet) < goal_scores.shape[0]:
                raise ValueError("Entry facet is outside goal_scores.")
    return parents, ordered_roots


def _topk_successors(scores: torch.Tensor, source: int, k: int) -> tuple[list[int], int]:
    """Return deterministic native successors without consulting goal scores."""
    values = scores[source].clone()
    values[source] = float("-inf")
    probe = torch.topk(values, min(k, values.numel()), largest=True, sorted=False)
    finite_probe = probe.values[torch.isfinite(probe.values)]
    count = int(finite_probe.numel())
    if count == 0:
        return [], 0
    # torch.topk does not specify which index wins a boundary tie. Recover all
    # scores above its boundary, then fill the tie from ascending parent IDs.
    boundary = finite_probe.min()
    strict = torch.nonzero(values > boundary, as_tuple=False).flatten()
    ties = torch.nonzero(values == boundary, as_tuple=False).flatten()
    order = torch.cat((strict, ties[: count - strict.numel()]))
    selected_order = torch.argsort(values[order], descending=True, stable=True)
    order = order[selected_order]
    tensor_bytes = values.numel() * values.element_size()
    tensor_bytes += probe.values.numel() * probe.values.element_size()
    tensor_bytes += probe.indices.numel() * probe.indices.element_size()
    tensor_bytes += order.numel() * order.element_size()
    return [int(index) for index in order.tolist()], tensor_bytes


def _goal(
    scores: torch.Tensor,
    parent: int,
    entry_facet: int | None,
    different_facet: bool,
) -> tuple[float, int]:
    values = scores[:, parent].clone()
    if different_facet and entry_facet is not None:
        values[entry_facet] = float("-inf")
    if not bool(torch.isfinite(values).any()):
        return float("-inf"), -1
    facet = int(torch.argmax(values))
    return float(values[facet]), facet


def _proposal_priority(
    state: SearchState,
    native_rank: int,
    native_score: float,
    strategy: str,
) -> tuple:
    quality = min(state.path_quality, native_score)
    if strategy == "breadth_first":
        return (state.depth + 1, native_rank, state.parent_index)
    return (-quality, state.depth + 1, native_rank, state.parent_index)


def search_semantic_graph(
    edge_scores: torch.Tensor,
    goal_scores: torch.Tensor,
    roots: Sequence[int],
    config: SemanticGraphSearchConfig,
    *,
    entry_facets: Mapping[int, int] | None = None,
) -> SemanticGraphSearchResult:
    """Find the first admitted memory node that reconnects to any query facet.

    ``edge_scores[P,P]`` drives every intermediate decision. ``goal_scores[F,P]``
    is read only after edge admission, so a low-query-similarity bridge cannot
    be filtered before it has a chance to expand.
    """
    parents, ordered_roots = _validate_inputs(
        edge_scores, goal_scores, roots, config, entry_facets
    )
    started = time.perf_counter()
    visited_order = list(ordered_roots)
    visited = set(ordered_roots)
    initial = [
        SearchState(
            parent_index=root,
            depth=0,
            path=(root,),
            path_quality=float("inf"),
            entry_facet=(int(entry_facets[root]) if entry_facets and root in entry_facets else None),
        )
        for root in ordered_roots
    ]
    frontier = list(initial)
    decisions: list[SearchDecision] = []
    nodes_expanded = 0
    raw_proposals = 0
    edge_admitted = 0
    duplicate_proposals = 0
    cycles_prevented = 0
    peak_frontier = len(frontier)
    goal_tests = 0
    peak_candidate_bytes = 0
    cpu_dedup_seconds = 0.0
    goal_test_seconds = 0.0
    terminal: SearchState | None = None
    terminal_facet: int | None = None
    stop_reason = "frontier_exhausted"

    while frontier:
        if nodes_expanded >= config.max_expanded_nodes:
            stop_reason = "max_expanded_nodes"
            break
        if config.strategy == "best_first":
            frontier.sort(key=lambda state: (-state.path_quality, state.depth, state.parent_index))
            expanding = [frontier.pop(0)]
        else:
            expanding, frontier = frontier, []
        peak_frontier = max(peak_frontier, len(expanding) + len(frontier))
        proposals: list[tuple[tuple, SearchState, int, int, float]] = []
        for state in expanding:
            if state.depth >= config.max_hops:
                continue
            if nodes_expanded >= config.max_expanded_nodes:
                break
            nodes_expanded += 1
            candidates, candidate_bytes = _topk_successors(
                edge_scores, state.parent_index, config.successor_k
            )
            peak_candidate_bytes = max(peak_candidate_bytes, candidate_bytes)
            raw_proposals += len(candidates)
            for rank, candidate in enumerate(candidates, start=1):
                score = float(edge_scores[state.parent_index, candidate])
                passed = score >= config.edge_threshold
                duplicate = candidate in visited
                cycle = candidate in state.path
                if duplicate:
                    duplicate_proposals += 1
                if cycle:
                    cycles_prevented += 1
                if not passed or duplicate:
                    decisions.append(
                        SearchDecision(
                            state.parent_index,
                            candidate,
                            state.depth + 1,
                            rank,
                            score,
                            passed,
                            duplicate,
                            cycle,
                            False,
                            None,
                            None,
                            False,
                            state.path + (candidate,),
                            min(state.path_quality, score),
                        )
                    )
                    continue
                proposals.append(
                    (
                        _proposal_priority(state, rank, score, config.strategy),
                        state,
                        candidate,
                        rank,
                        score,
                    )
                )

        dedup_started = time.perf_counter()
        proposals.sort(key=lambda row: (row[0], row[2]))
        unique: list[tuple[tuple, SearchState, int, int, float]] = []
        proposed_parents: set[int] = set()
        for proposal in proposals:
            candidate = proposal[2]
            if candidate in proposed_parents:
                duplicate_proposals += 1
                _, state, _, rank, score = proposal
                decisions.append(
                    SearchDecision(
                        state.parent_index,
                        candidate,
                        state.depth + 1,
                        rank,
                        score,
                        True,
                        True,
                        candidate in state.path,
                        False,
                        None,
                        None,
                        False,
                        state.path + (candidate,),
                        min(state.path_quality, score),
                    )
                )
                continue
            proposed_parents.add(candidate)
            unique.append(proposal)
        cpu_dedup_seconds += time.perf_counter() - dedup_started

        remaining = parents - len(visited)
        if config.max_visited_parents is not None:
            remaining = min(remaining, config.max_visited_parents - len(visited))
        if config.strategy == "beam":
            remaining = min(remaining, config.beam_width)
        if remaining <= 0:
            stop_reason = "visited_budget"
            break

        admitted_states: list[SearchState] = []
        for _, state, candidate, rank, score in unique[:remaining]:
            new_state = SearchState(
                parent_index=candidate,
                depth=state.depth + 1,
                path=state.path + (candidate,),
                path_quality=min(state.path_quality, score),
                entry_facet=state.entry_facet,
            )
            visited.add(candidate)
            visited_order.append(candidate)
            edge_admitted += 1
            goal_started = time.perf_counter()
            goal_score, goal_facet = _goal(
                goal_scores,
                candidate,
                state.entry_facet,
                config.different_facet_goal,
            )
            goal_test_seconds += time.perf_counter() - goal_started
            goal_tests += 1
            triggered = (
                new_state.depth >= config.minimum_goal_depth
                and goal_score >= config.goal_threshold
            )
            decisions.append(
                SearchDecision(
                    state.parent_index,
                    candidate,
                    new_state.depth,
                    rank,
                    score,
                    True,
                    False,
                    False,
                    True,
                    goal_facet,
                    goal_score,
                    triggered,
                    new_state.path,
                    new_state.path_quality,
                )
            )
            if triggered:
                terminal = new_state
                terminal_facet = goal_facet
                stop_reason = "goal"
                break
            admitted_states.append(new_state)
        for _, state, candidate, rank, score in unique[remaining:]:
            decisions.append(
                SearchDecision(
                    state.parent_index,
                    candidate,
                    state.depth + 1,
                    rank,
                    score,
                    True,
                    False,
                    False,
                    False,
                    None,
                    None,
                    False,
                    state.path + (candidate,),
                    min(state.path_quality, score),
                )
            )
        if terminal is not None:
            break
        if config.strategy == "best_first":
            frontier.extend(admitted_states)
        else:
            frontier = admitted_states
        peak_frontier = max(peak_frontier, len(frontier))
        if not frontier and any(state.depth >= config.max_hops for state in expanding):
            stop_reason = "max_hops"
        elif config.max_visited_parents is not None and len(visited) >= config.max_visited_parents:
            stop_reason = "visited_budget"

    elapsed = time.perf_counter() - started
    return SemanticGraphSearchResult(
        roots=ordered_roots,
        visited=tuple(visited_order),
        terminal_parent=terminal.parent_index if terminal else None,
        terminal_facet=terminal_facet,
        path=terminal.path if terminal else (),
        goal_triggered=terminal is not None,
        stop_reason=stop_reason,
        decisions=tuple(decisions),
        nodes_expanded=nodes_expanded,
        raw_proposals=raw_proposals,
        edge_admitted_proposals=edge_admitted,
        duplicate_proposals=duplicate_proposals,
        cycles_prevented=cycles_prevented,
        peak_frontier=peak_frontier,
        goal_tests=goal_tests,
        goal_comparisons=goal_tests * int(goal_scores.shape[0]),
        peak_candidate_tensor_bytes=peak_candidate_bytes,
        cpu_dedup_seconds=cpu_dedup_seconds,
        goal_test_seconds=goal_test_seconds,
        search_seconds=elapsed,
    )
