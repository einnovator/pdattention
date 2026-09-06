"""Auditable oracle plans for sparse cross-document contextualization.

The initial Paper 3.3 oracle deliberately selects layer/token-pair edges while
retaining all attention heads for each selected pair. This is coarser than the
eventual learned policy, but it establishes whether sparse, on-manifold host
attention has enough task headroom before training a selector.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
                np.full(target_tokens.size * source_tokens.size, target_record, np.int16)
            )
            source_records.append(
                np.full(target_tokens.size * source_tokens.size, source_record, np.int16)
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

    ``layer_scores`` has shape ``[L, E]``. Each score is attention probability
    summed across heads for one causal source/target token pair. ``head_mass``
    has shape ``[L, H]`` and supports head localization without retaining the
    prohibitive dense tensor ``[L, H, T, T]``.
    """

    record_ids: tuple[str, ...]
    document_boundaries: tuple[TokenBoundary, ...]
    source_tokens: np.ndarray
    target_tokens: np.ndarray
    source_records: np.ndarray
    target_records: np.ndarray
    layer_scores: np.ndarray
    head_mass: np.ndarray
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
        if self.layer_scores.ndim != 2 or self.layer_scores.shape[1] != edge_count:
            raise ValueError("layer scores must have shape [layers, token-pair edges]")
        if self.head_mass.ndim != 2 or self.head_mass.shape[0] != self.layer_count:
            raise ValueError("head mass must have shape [layers, heads]")
        if len(self.record_ids) != len(self.document_boundaries):
            raise ValueError("record identities do not match document boundaries")

    @property
    def layer_count(self) -> int:
        return int(self.layer_scores.shape[0])

    @property
    def head_count(self) -> int:
        return int(self.head_mass.shape[1])

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
        return float(self.layer_scores.astype(np.float64).sum())

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
            np.ascontiguousarray(self.layer_scores).tobytes(),
            np.ascontiguousarray(self.head_mass).tobytes(),
        )

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "graph_digest": self.graph_digest,
            "selection_receipt_id": self.selection_receipt_id,
            "model_revision": self.model_revision,
            "record_ids": list(self.record_ids),
            "document_boundaries": [
                {"start": row.start, "end": row.end}
                for row in self.document_boundaries
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
                {"start": row.start, "end": row.end}
                for row in self.document_boundaries
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
            layer_scores=self.layer_scores,
            head_mass=self.head_mass,
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
                layer_scores=payload["layer_scores"].copy(),
                head_mass=payload["head_mass"].copy(),
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
        self._head_mass: list[np.ndarray] = []

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
        self._layer_scores.append(selected.sum(axis=0, dtype=np.float32))
        self._head_mass.append(selected.sum(axis=1, dtype=np.float64))

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
            layer_scores=np.stack(self._layer_scores).astype(np.float32),
            head_mass=np.stack(self._head_mass).astype(np.float64),
            selection_receipt_id=self.selection_receipt_id,
            model_revision=self.model_revision,
        )


@dataclass(frozen=True)
class SparseInteractionPlan:
    """Request-local layer/token-pair oracle plan replayed across all heads."""

    graph_digest: str
    selection_receipt_id: str
    mode: str
    target: float
    selected_flat_indices: tuple[int, ...]
    token_pair_count: int
    layer_count: int
    head_count: int
    retained_attention_mass: float
    selected_logical_edge_fraction: float
    schema_version: str = "paper3.3-sparse-interaction-plan-v1"

    @property
    def selected_logical_edges(self) -> int:
        return len(self.selected_flat_indices)

    @property
    def selected_physical_head_edges(self) -> int:
        return self.selected_logical_edges * self.head_count

    @property
    def plan_digest(self) -> str:
        return _digest_bytes(
            json.dumps(self.to_dict(include_digest=False), sort_keys=True).encode("ascii")
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": self.schema_version,
            "graph_digest": self.graph_digest,
            "selection_receipt_id": self.selection_receipt_id,
            "mode": self.mode,
            "target": self.target,
            "granularity": "layer_token_pair_all_heads",
            "selected_logical_edges": self.selected_logical_edges,
            "selected_physical_head_edges": self.selected_physical_head_edges,
            "selected_logical_edge_fraction": self.selected_logical_edge_fraction,
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
        start = layer_index * self.token_pair_count
        stop = start + self.token_pair_count
        local = np.asarray(
            [index - start for index in self.selected_flat_indices if start <= index < stop],
            dtype=np.int64,
        )
        if local.size:
            mask[:, target_tokens[local], source_tokens[local]] = True
        return mask


def _ranked_flat_indices(graph: CrossDocumentOracleGraph) -> np.ndarray:
    scores = graph.layer_scores.reshape(-1).astype(np.float64)
    indices = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((indices, -scores))


def _plan(
    graph: CrossDocumentOracleGraph,
    *,
    mode: str,
    target: float,
    selected: np.ndarray,
) -> SparseInteractionPlan:
    scores = graph.layer_scores.reshape(-1).astype(np.float64)
    total_mass = float(scores.sum())
    retained = float(scores[selected].sum() / total_mass) if total_mass and selected.size else 0.0
    return SparseInteractionPlan(
        graph_digest=graph.graph_digest,
        selection_receipt_id=graph.selection_receipt_id,
        mode=mode,
        target=float(target),
        selected_flat_indices=tuple(int(value) for value in np.sort(selected)),
        token_pair_count=graph.token_pair_count,
        layer_count=graph.layer_count,
        head_count=graph.head_count,
        retained_attention_mass=retained,
        selected_logical_edge_fraction=float(selected.size / graph.logical_edge_count),
    )


def top_attention_edge_plan(
    graph: CrossDocumentOracleGraph, fraction: float
) -> SparseInteractionPlan:
    """Keep the highest teacher-attention layer/token-pair edges."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("edge fraction must be in [0, 1]")
    count = int(math.ceil(graph.logical_edge_count * fraction))
    selected = _ranked_flat_indices(graph)[:count]
    return _plan(graph, mode="TOP_ATTENTION", target=fraction, selected=selected)


def cumulative_attention_mass_plan(
    graph: CrossDocumentOracleGraph, mass_fraction: float
) -> SparseInteractionPlan:
    """Keep the smallest ranked edge set reaching a teacher-mass target."""

    if not 0.0 <= mass_fraction <= 1.0:
        raise ValueError("mass fraction must be in [0, 1]")
    if mass_fraction == 0.0:
        selected = np.asarray([], dtype=np.int64)
    else:
        ranked = _ranked_flat_indices(graph)
        scores = graph.layer_scores.reshape(-1).astype(np.float64)
        cumulative = np.cumsum(scores[ranked])
        threshold = float(scores.sum()) * mass_fraction
        count = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        selected = ranked[:count]
    return _plan(
        graph, mode="CUMULATIVE_ATTENTION_MASS", target=mass_fraction, selected=selected
    )


def interaction_localization(graph: CrossDocumentOracleGraph) -> dict[str, object]:
    """Summarize where teacher cross-record mass occurs without causal claims."""

    total = max(graph.attention_mass, np.finfo(np.float64).tiny)
    layer_mass = graph.layer_scores.astype(np.float64).sum(axis=1)
    head_mass = graph.head_mass.astype(np.float64)
    pair_rows = []
    for target in range(len(graph.record_ids)):
        for source in range(target):
            edge_mask = (graph.source_records == source) & (graph.target_records == target)
            mass = float(graph.layer_scores[:, edge_mask].astype(np.float64).sum())
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
    pair_rows.sort(key=lambda row: (-float(row["attention_mass"]), str(row["source_record_id"])))
    layer_rows = [
        {"layer": index, "attention_mass": float(value), "attention_mass_fraction": float(value / total)}
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
    head_rows.sort(key=lambda row: (-float(row["attention_mass"]), int(row["layer"]), int(row["head"])))
    return {
        "schema_version": "paper3.3-interaction-localization-v1",
        "graph_digest": graph.graph_digest,
        "total_cross_document_attention_mass": graph.attention_mass,
        "layers": layer_rows,
        "top_layer_heads": head_rows[: min(32, len(head_rows))],
        "record_pairs": pair_rows,
    }
