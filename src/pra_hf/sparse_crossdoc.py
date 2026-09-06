"""Auditable oracle plans for sparse cross-document contextualization.

Paper 3.3 represents each physical interaction as a
``[layer, query-head, target-token, source-token]`` edge. Plans may rank those
edges observationally by teacher attention or hierarchically by measured
document-pair, layer, or layer/head intervention utility.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from .rag_causal_decomposition import TokenBoundary


def _digest_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _document_boundaries(lengths: Sequence[int]) -> tuple[TokenBoundary, ...]:
    if len(lengths) < 2 or any(int(length) <= 0 for length in lengths):
        raise ValueError("oracle extraction requires at least two non-empty records")
    boundaries: list[TokenBoundary] = []
    cursor = 0
    for length in lengths:
        boundaries.append(TokenBoundary(cursor, cursor + int(length)))
        cursor += int(length)
    return tuple(boundaries)


def _cross_document_token_pairs(
    boundaries: Sequence[TokenBoundary],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    source_records: list[np.ndarray] = []
    target_records: list[np.ndarray] = []
    for target_record, target_boundary in enumerate(boundaries):
        for source_record in range(target_record):
            source_boundary = boundaries[source_record]
            target_tokens = np.arange(
                target_boundary.start, target_boundary.end, dtype=np.int32
            )
            source_tokens = np.arange(
                source_boundary.start, source_boundary.end, dtype=np.int32
            )
            targets.append(np.repeat(target_tokens, source_tokens.size))
            sources.append(np.tile(source_tokens, target_tokens.size))
            target_records.append(
                np.full(
                    target_tokens.size * source_tokens.size, target_record, np.int16
                )
            )
            source_records.append(
                np.full(
                    target_tokens.size * source_tokens.size, source_record, np.int16
                )
            )
    return (
        np.concatenate(sources),
        np.concatenate(targets),
        np.concatenate(source_records),
        np.concatenate(target_records),
    )


@dataclass(frozen=True)
class CrossDocumentOracleGraph:
    """Compressed teacher graph over causal cross-record token pairs.

    ``edge_scores`` has shape ``[L,H,E]``. Each score is the attention
    probability of one physical layer/head/source/target edge. Only causal
    cross-record token pairs are retained, rather than the full dense tensor
    ``[L,H,T,T]``.
    """

    record_ids: tuple[str, ...]
    document_boundaries: tuple[TokenBoundary, ...]
    source_tokens: np.ndarray
    target_tokens: np.ndarray
    source_records: np.ndarray
    target_records: np.ndarray
    edge_scores: np.ndarray
    selection_receipt_id: str
    model_revision: str
    schema_version: str = "paper3.3-crossdoc-oracle-graph-v1"

    def __post_init__(self) -> None:
        edge_count = int(self.source_tokens.size)
        arrays = (
            self.target_tokens,
            self.source_records,
            self.target_records,
        )
        if edge_count <= 0 or any(int(array.size) != edge_count for array in arrays):
            raise ValueError("oracle graph token-pair arrays are inconsistent")
        if self.edge_scores.ndim != 3 or self.edge_scores.shape[2] != edge_count:
            raise ValueError("edge scores must have shape [layers, heads, token pairs]")
        if len(self.record_ids) != len(self.document_boundaries):
            raise ValueError("record identities do not match document boundaries")

    @property
    def layer_count(self) -> int:
        return int(self.edge_scores.shape[0])

    @property
    def head_count(self) -> int:
        return int(self.edge_scores.shape[1])

    @property
    def token_pair_count(self) -> int:
        return int(self.source_tokens.size)

    @property
    def logical_edge_count(self) -> int:
        return self.layer_count * self.token_pair_count

    @property
    def physical_edge_count(self) -> int:
        return self.logical_edge_count * self.head_count

    @property
    def attention_mass(self) -> float:
        return float(self.edge_scores.astype(np.float64).sum())

    @property
    def graph_digest(self) -> str:
        metadata = json.dumps(
            {
                "schema_version": self.schema_version,
                "record_ids": self.record_ids,
                "document_boundaries": [
                    {"start": row.start, "end": row.end}
                    for row in self.document_boundaries
                ],
                "selection_receipt_id": self.selection_receipt_id,
                "model_revision": self.model_revision,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return _digest_bytes(
            metadata,
            np.ascontiguousarray(self.source_tokens).tobytes(),
            np.ascontiguousarray(self.target_tokens).tobytes(),
            np.ascontiguousarray(self.edge_scores).tobytes(),
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "graph_digest": self.graph_digest,
            "selection_receipt_id": self.selection_receipt_id,
            "model_revision": self.model_revision,
            "record_ids": list(self.record_ids),
            "document_boundaries": [
                {"start": row.start, "end": row.end} for row in self.document_boundaries
            ],
            "layers": self.layer_count,
            "heads": self.head_count,
            "cross_document_token_pairs_per_layer": self.token_pair_count,
            "dense_logical_edges": self.logical_edge_count,
            "dense_physical_head_edges": self.physical_edge_count,
            "cross_document_attention_mass": self.attention_mass,
        }

    def save(self, path: Path) -> None:
        """Persist the compressed graph without expanding Python edge objects."""

        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": self.schema_version,
            "record_ids": list(self.record_ids),
            "document_boundaries": [
                {"start": row.start, "end": row.end} for row in self.document_boundaries
            ],
            "selection_receipt_id": self.selection_receipt_id,
            "model_revision": self.model_revision,
        }
        np.savez_compressed(
            path,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            source_tokens=self.source_tokens,
            target_tokens=self.target_tokens,
            source_records=self.source_records,
            target_records=self.target_records,
            edge_scores=self.edge_scores,
        )

    @classmethod
    def load(cls, path: Path) -> "CrossDocumentOracleGraph":
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            return cls(
                record_ids=tuple(metadata["record_ids"]),
                document_boundaries=tuple(
                    TokenBoundary(int(row["start"]), int(row["end"]))
                    for row in metadata["document_boundaries"]
                ),
                source_tokens=payload["source_tokens"].copy(),
                target_tokens=payload["target_tokens"].copy(),
                source_records=payload["source_records"].copy(),
                target_records=payload["target_records"].copy(),
                edge_scores=payload["edge_scores"].copy(),
                selection_receipt_id=str(metadata["selection_receipt_id"]),
                model_revision=str(metadata["model_revision"]),
                schema_version=str(metadata["schema_version"]),
            )


class CrossDocumentAttentionCollector:
    """Stream packed-teacher attention into a compressed oracle graph."""

    def __init__(
        self,
        document_lengths: Sequence[int],
        *,
        record_ids: Sequence[str],
        selection_receipt_id: str,
        model_revision: str,
    ) -> None:
        self.boundaries = _document_boundaries(document_lengths)
        self.record_ids = tuple(record_ids)
        if len(self.record_ids) != len(self.boundaries):
            raise ValueError("record identities must align with document lengths")
        (
            self.source_tokens,
            self.target_tokens,
            self.source_records,
            self.target_records,
        ) = _cross_document_token_pairs(self.boundaries)
        self.selection_receipt_id = selection_receipt_id
        self.model_revision = model_revision
        self._layer_scores: list[np.ndarray] = []

    def observe(self, layer_index: int, attention_probabilities: object) -> None:
        """Consume one ``[B,H,T,T]`` or ``[H,T,T]`` host attention tensor."""

        probabilities = np.asarray(attention_probabilities, dtype=np.float32)
        if probabilities.ndim == 4:
            probabilities = probabilities.mean(axis=0)
        if probabilities.ndim != 3:
            raise ValueError("attention probabilities must have shape [B,H,T,T]")
        expected_tokens = self.boundaries[-1].end
        if probabilities.shape[1:] != (expected_tokens, expected_tokens):
            raise ValueError("attention token dimensions do not match record geometry")
        if layer_index != len(self._layer_scores):
            raise ValueError("attention layers must be observed in execution order")
        selected = probabilities[:, self.target_tokens, self.source_tokens]
        self._layer_scores.append(selected.astype(np.float32))

    def finalize(self) -> CrossDocumentOracleGraph:
        if not self._layer_scores:
            raise ValueError("no teacher attention layers were observed")
        return CrossDocumentOracleGraph(
            record_ids=self.record_ids,
            document_boundaries=self.boundaries,
            source_tokens=self.source_tokens,
            target_tokens=self.target_tokens,
            source_records=self.source_records,
            target_records=self.target_records,
            edge_scores=np.stack(self._layer_scores).astype(np.float32),
            selection_receipt_id=self.selection_receipt_id,
            model_revision=self.model_revision,
        )


@dataclass(frozen=True)
class SparseInteractionPlan:
    """Request-local mask over physical layer/head/token-pair edges."""

    graph_digest: str
    selection_receipt_id: str
    mode: str
    target: float
    selected_mask: np.ndarray
    token_pair_count: int
    layer_count: int
    head_count: int
    retained_attention_mass: float
    schema_version: str = "paper3.3-sparse-interaction-plan-v1"

    def __post_init__(self) -> None:
        expected = (self.layer_count, self.head_count, self.token_pair_count)
        if self.selected_mask.shape != expected or self.selected_mask.dtype != np.bool_:
            raise ValueError(f"selected mask must be bool with shape {expected}")

    @property
    def selected_logical_edges(self) -> int:
        return int(np.any(self.selected_mask, axis=1).sum())

    @property
    def selected_physical_head_edges(self) -> int:
        return int(self.selected_mask.sum())

    @property
    def selected_logical_edge_fraction(self) -> float:
        denominator = self.layer_count * self.token_pair_count
        return float(self.selected_logical_edges / denominator)

    @property
    def selected_physical_edge_fraction(self) -> float:
        denominator = self.layer_count * self.head_count * self.token_pair_count
        return float(self.selected_physical_head_edges / denominator)

    @property
    def plan_digest(self) -> str:
        return _digest_bytes(
            json.dumps(self.to_dict(include_digest=False), sort_keys=True).encode(
                "ascii"
            ),
            np.ascontiguousarray(self.selected_mask).tobytes(),
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "graph_digest": self.graph_digest,
            "selection_receipt_id": self.selection_receipt_id,
            "mode": self.mode,
            "target": self.target,
            "granularity": "layer_head_token_pair",
            "selected_logical_edges": self.selected_logical_edges,
            "selected_physical_head_edges": self.selected_physical_head_edges,
            "selected_logical_edge_fraction": self.selected_logical_edge_fraction,
            "selected_physical_edge_fraction": self.selected_physical_edge_fraction,
            "retained_attention_mass": self.retained_attention_mass,
            "layer_count": self.layer_count,
            "head_count": self.head_count,
            "token_pair_count_per_layer": self.token_pair_count,
        }
        if include_digest:
            result["plan_digest"] = self.plan_digest
        return result

    def mask_for_layer(
        self,
        layer_index: int,
        *,
        base_mask: Sequence[Sequence[bool]],
        source_tokens: np.ndarray,
        target_tokens: np.ndarray,
    ) -> np.ndarray:
        """Return ``[H,T,T]`` visibility with selected cross-record edges added."""

        base = np.asarray(base_mask, dtype=np.bool_)
        if base.ndim != 2 or base.shape[0] != base.shape[1]:
            raise ValueError("base attention mask must have shape [tokens, tokens]")
        if layer_index < 0 or layer_index >= self.layer_count:
            raise ValueError("layer index is outside the interaction plan")
        mask = np.broadcast_to(base, (self.head_count, *base.shape)).copy()
        heads, pairs = np.nonzero(self.selected_mask[layer_index])
        if pairs.size:
            mask[heads, target_tokens[pairs], source_tokens[pairs]] = True
        return mask


def ranked_physical_indices(graph: CrossDocumentOracleGraph) -> np.ndarray:
    """Rank all physical edges once using stable teacher-attention order."""

    scores = graph.edge_scores.reshape(-1)
    return np.argsort(-scores, kind="stable")


def _plan(
    graph: CrossDocumentOracleGraph,
    *,
    mode: str,
    target: float,
    selected: np.ndarray,
) -> SparseInteractionPlan:
    scores = graph.edge_scores.reshape(-1).astype(np.float64)
    total_mass = float(scores.sum())
    retained = (
        float(scores[selected].sum() / total_mass)
        if total_mass and selected.size
        else 0.0
    )
    selected_mask = np.zeros(scores.size, dtype=np.bool_)
    selected_mask[selected] = True
    return SparseInteractionPlan(
        graph_digest=graph.graph_digest,
        selection_receipt_id=graph.selection_receipt_id,
        mode=mode,
        target=float(target),
        selected_mask=selected_mask.reshape(graph.edge_scores.shape),
        token_pair_count=graph.token_pair_count,
        layer_count=graph.layer_count,
        head_count=graph.head_count,
        retained_attention_mass=retained,
    )


def top_attention_edge_plan(
    graph: CrossDocumentOracleGraph,
    fraction: float,
    *,
    ranked: np.ndarray | None = None,
) -> SparseInteractionPlan:
    """Keep the highest teacher-attention physical edges."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("edge fraction must be in [0, 1]")
    count = int(math.ceil(graph.physical_edge_count * fraction))
    selected = (ranked if ranked is not None else ranked_physical_indices(graph))[
        :count
    ]
    return _plan(graph, mode="TOP_ATTENTION", target=fraction, selected=selected)


def cumulative_attention_mass_plan(
    graph: CrossDocumentOracleGraph,
    mass_fraction: float,
    *,
    ranked: np.ndarray | None = None,
) -> SparseInteractionPlan:
    """Keep the smallest ranked edge set reaching a teacher-mass target."""

    if not 0.0 <= mass_fraction <= 1.0:
        raise ValueError("mass fraction must be in [0, 1]")
    if mass_fraction == 0.0:
        selected = np.asarray([], dtype=np.int64)
    else:
        edge_order = ranked if ranked is not None else ranked_physical_indices(graph)
        scores = graph.edge_scores.reshape(-1).astype(np.float64)
        cumulative = np.cumsum(scores[edge_order])
        threshold = float(scores.sum()) * mass_fraction
        count = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        selected = edge_order[:count]
    return _plan(
        graph, mode="CUMULATIVE_ATTENTION_MASS", target=mass_fraction, selected=selected
    )


def interaction_localization(graph: CrossDocumentOracleGraph) -> dict[str, object]:
    """Summarize where teacher cross-record mass occurs without causal claims."""

    total = max(graph.attention_mass, np.finfo(np.float64).tiny)
    layer_mass = graph.edge_scores.astype(np.float64).sum(axis=(1, 2))
    head_mass = graph.edge_scores.astype(np.float64).sum(axis=2)
    pair_rows = []
    for target in range(len(graph.record_ids)):
        for source in range(target):
            edge_mask = (graph.source_records == source) & (
                graph.target_records == target
            )
            mass = float(graph.edge_scores[:, :, edge_mask].astype(np.float64).sum())
            pair_rows.append(
                {
                    "source_record_index": source,
                    "target_record_index": target,
                    "source_record_id": graph.record_ids[source],
                    "target_record_id": graph.record_ids[target],
                    "attention_mass": mass,
                    "attention_mass_fraction": mass / total,
                }
            )
    pair_rows.sort(
        key=lambda row: (-float(row["attention_mass"]), str(row["source_record_id"]))
    )
    layer_rows = [
        {
            "layer": index,
            "attention_mass": float(value),
            "attention_mass_fraction": float(value / total),
        }
        for index, value in enumerate(layer_mass)
    ]
    head_rows = [
        {
            "layer": layer,
            "head": head,
            "attention_mass": float(head_mass[layer, head]),
            "attention_mass_fraction": float(head_mass[layer, head] / total),
        }
        for layer in range(graph.layer_count)
        for head in range(graph.head_count)
    ]
    head_rows.sort(
        key=lambda row: (
            -float(row["attention_mass"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    return {
        "schema_version": "paper3.3-interaction-localization-v1",
        "graph_digest": graph.graph_digest,
        "total_cross_document_attention_mass": graph.attention_mass,
        "layers": layer_rows,
        "top_layer_heads": head_rows[: min(32, len(head_rows))],
        "record_pairs": pair_rows,
    }


InteractionGroupKind = Literal["document_pair", "layer", "layer_head"]
InteractionGroupKey = tuple[int, ...]


def interaction_group_keys(
    graph: CrossDocumentOracleGraph,
    kind: InteractionGroupKind,
) -> tuple[InteractionGroupKey, ...]:
    """Enumerate causal groups from coarse record pairs to physical heads."""

    if kind == "document_pair":
        return tuple(
            (source, target)
            for target in range(len(graph.record_ids))
            for source in range(target)
        )
    if kind == "layer":
        return tuple((layer,) for layer in range(graph.layer_count))
    if kind == "layer_head":
        return tuple(
            (layer, head)
            for layer in range(graph.layer_count)
            for head in range(graph.head_count)
        )
    raise ValueError(f"unsupported interaction group kind: {kind}")


def interaction_group_mask(
    graph: CrossDocumentOracleGraph,
    kind: InteractionGroupKind,
    key: InteractionGroupKey,
) -> np.ndarray:
    """Return the physical ``[L,H,E]`` mask belonging to one group."""

    mask = np.zeros(graph.edge_scores.shape, dtype=np.bool_)
    if kind == "document_pair":
        if len(key) != 2:
            raise ValueError("document-pair keys require source and target indices")
        source, target = key
        pairs = (graph.source_records == source) & (graph.target_records == target)
        mask[:, :, pairs] = True
    elif kind == "layer":
        if len(key) != 1 or not 0 <= key[0] < graph.layer_count:
            raise ValueError("layer key is outside the graph")
        mask[key[0], :, :] = True
    elif kind == "layer_head":
        if (
            len(key) != 2
            or not 0 <= key[0] < graph.layer_count
            or not 0 <= key[1] < graph.head_count
        ):
            raise ValueError("layer/head key is outside the graph")
        mask[key[0], key[1], :] = True
    else:
        raise ValueError(f"unsupported interaction group kind: {kind}")
    if not mask.any():
        raise ValueError(f"interaction group {kind}:{key} contains no physical edges")
    return mask


def interaction_group_ablation_plan(
    graph: CrossDocumentOracleGraph,
    kind: InteractionGroupKind,
    key: InteractionGroupKey,
) -> SparseInteractionPlan:
    """Keep the packed teacher graph except for one causally tested group."""

    removed = interaction_group_mask(graph, kind, key).reshape(-1)
    selected = np.flatnonzero(~removed)
    return _plan(
        graph,
        mode=f"ABLATE_{kind.upper()}",
        target=float(removed.mean()),
        selected=selected,
    )


def ranked_physical_indices_by_group_utility(
    graph: CrossDocumentOracleGraph,
    kind: InteractionGroupKind,
    utilities: Mapping[InteractionGroupKey, float],
    *,
    combination: Literal["lexicographic", "utility_x_attention"] = "lexicographic",
) -> np.ndarray:
    """Rank physical edges by causal group utility with attention as refinement.

    ``lexicographic`` first orders groups by utility, then edges within each
    group by teacher attention. ``utility_x_attention`` assigns each edge the
    non-negative group utility multiplied by its teacher attention. The latter
    is a finite-difference-times-attention proxy, not an autograd claim.
    """

    keys = interaction_group_keys(graph, kind)
    missing = [key for key in keys if key not in utilities]
    if missing:
        raise ValueError(f"missing intervention utilities for {kind}: {missing[:4]}")
    scores = graph.edge_scores.reshape(-1)
    if combination == "lexicographic":
        ranked_groups = sorted(keys, key=lambda key: (-float(utilities[key]), key))
        ranked: list[np.ndarray] = []
        for key in ranked_groups:
            indices = np.flatnonzero(
                interaction_group_mask(graph, kind, key).reshape(-1)
            )
            order = np.argsort(-scores[indices], kind="stable")
            ranked.append(indices[order])
        return np.concatenate(ranked).astype(np.int64, copy=False)
    if combination == "utility_x_attention":
        combined = np.empty(scores.shape, dtype=np.float32)
        for key in keys:
            indices = np.flatnonzero(
                interaction_group_mask(graph, kind, key).reshape(-1)
            )
            combined[indices] = max(0.0, float(utilities[key])) * scores[indices]
        if not np.any(combined > 0):
            return ranked_physical_indices(graph)
        return np.argsort(-combined, kind="stable")
    raise ValueError(f"unsupported utility/attention combination: {combination}")


def ranked_edge_plan(
    graph: CrossDocumentOracleGraph,
    fraction: float,
    *,
    ranked: np.ndarray,
    mode: str,
) -> SparseInteractionPlan:
    """Materialize a physical-edge budget from an externally defined ranking."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("edge fraction must be in [0, 1]")
    if ranked.ndim != 1 or ranked.size != graph.physical_edge_count:
        raise ValueError("ranked physical edge indices do not cover the graph")
    if (
        np.any(ranked < 0)
        or np.any(ranked >= graph.physical_edge_count)
        or np.unique(ranked).size != graph.physical_edge_count
    ):
        raise ValueError("ranked physical edge indices must be a permutation")
    count = int(math.ceil(graph.physical_edge_count * fraction))
    return _plan(graph, mode=mode, target=fraction, selected=ranked[:count])


def selected_interaction_localization(
    graph: CrossDocumentOracleGraph,
    plan: SparseInteractionPlan,
) -> dict[str, object]:
    """Describe where a sparse plan spends its physical interaction budget."""

    if plan.graph_digest != graph.graph_digest:
        raise ValueError("plan and graph identities differ")
    selected = plan.selected_mask
    total = max(plan.selected_physical_head_edges, 1)
    layer_counts = selected.sum(axis=(1, 2), dtype=np.int64)
    head_counts = selected.sum(axis=2, dtype=np.int64)
    layers = [
        {
            "layer": layer,
            "selected_physical_head_edges": int(count),
            "selected_fraction": float(count / total),
        }
        for layer, count in enumerate(layer_counts)
        if count
    ]
    layer_heads = [
        {
            "layer": layer,
            "head": head,
            "selected_physical_head_edges": int(head_counts[layer, head]),
            "selected_fraction": float(head_counts[layer, head] / total),
        }
        for layer in range(graph.layer_count)
        for head in range(graph.head_count)
        if head_counts[layer, head]
    ]
    pairs = []
    for source, target in interaction_group_keys(graph, "document_pair"):
        edge_mask = (graph.source_records == source) & (graph.target_records == target)
        count = int(selected[:, :, edge_mask].sum())
        if count:
            pairs.append(
                {
                    "source_record_index": source,
                    "target_record_index": target,
                    "source_record_id": graph.record_ids[source],
                    "target_record_id": graph.record_ids[target],
                    "selected_physical_head_edges": count,
                    "selected_fraction": float(count / total),
                }
            )
    layers.sort(
        key=lambda row: (-int(row["selected_physical_head_edges"]), int(row["layer"]))
    )
    layer_heads.sort(
        key=lambda row: (
            -int(row["selected_physical_head_edges"]),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    pairs.sort(
        key=lambda row: (
            -int(row["selected_physical_head_edges"]),
            int(row["source_record_index"]),
            int(row["target_record_index"]),
        )
    )
    return {
        "schema_version": "paper3.3-selected-interaction-localization-v1",
        "graph_digest": graph.graph_digest,
        "plan_digest": plan.plan_digest,
        "mode": plan.mode,
        "target_percentage": 100.0 * plan.target,
        "selected_physical_head_edges": plan.selected_physical_head_edges,
        "top_layers": layers[: min(16, len(layers))],
        "layer_heads": layer_heads,
        "top_layer_heads": layer_heads[: min(32, len(layer_heads))],
        "record_pairs": pairs,
    }
