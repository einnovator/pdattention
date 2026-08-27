"""Hugging Face bridge for engine-neutral PRA execution policies."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pra_torch.execution import (
    PRARequestExecutionContext,
    PRASelectedIdentity,
    PRASelectionController,
    PRASelectionLayerScope,
)

if TYPE_CHECKING:
    import torch

    from pra_torch.core import PreparedPRAMemory
    from pra_torch.hf.adapter_base import PRAHFAttentionAdapter
    from pra_torch.hf.injection import PRAHFModel
    from pra_torch.memory import SelectedChunk


def selected_to_identity(selected: "SelectedChunk") -> PRASelectedIdentity:
    """Drop layer-native K/V while preserving stable routing provenance."""

    return PRASelectedIdentity(
        reference_uri=selected.reference_uri,
        chunk_id=selected.chunk_id,
        token_start=selected.token_start,
        token_end=selected.token_end,
        reference_score=selected.reference_score,
        chunk_score=selected.chunk_score,
        winning_gist_index=selected.winning_gist_index,
        metadata={
            "selection_source_layer": selected.layer_id,
            "reference_rank": selected.reference_rank,
            "rank_within_reference": selected.rank_within_reference,
        },
    )


def selected_rows_to_identities(rows):
    """Convert row-local native selections to immutable logical identities."""

    return tuple(tuple(selected_to_identity(item) for item in row) for row in rows)


class PRAHFExecutionBridge:
    """Apply dynamic policy semantics while leaving family adapters thin.

    The bridge is request-owned. Attention modules hold it only for the duration
    of one serialized HF generation call, which prevents policy and selection
    state from leaking into later requests.
    """

    def __init__(
        self,
        handle: "PRAHFModel",
        context: PRARequestExecutionContext,
        *,
        active_layers: tuple[int, ...],
        routing_layer: int,
    ) -> None:
        self.handle = handle
        self.context = context
        self.active_layers = tuple(sorted(set(int(layer) for layer in active_layers)))
        self.routing_layer = int(routing_layer)
        self.controller = PRASelectionController()
        self._native_rows: dict[tuple[int, int], list[list[SelectedChunk]]] = {}
        self._rankings: dict[tuple[int, int], list[list[dict]]] = {}

    def _route(self, adapter: "PRAHFAttentionAdapter", routing_states: "torch.Tensor"):
        started = time.perf_counter()
        if routing_states.ndim == 4:
            query = adapter.pra_core.prepare_pra_query(routing_states)
        elif routing_states.ndim == 2:
            query = routing_states
        else:
            raise ValueError("HF routing states must be [B,H,T,Dh] or [B,D].")
        selected, rankings = adapter.pra_core.route_memory(query)
        return selected, rankings, time.perf_counter() - started

    def _selected_for_layer(self, plan, layer_id: int):
        source = plan.source_layer
        if source is None:
            raise RuntimeError("A dynamic HF selection plan requires a source layer.")
        source_rows = self._native_rows[(plan.epoch_id, int(source))]
        if plan.layer_scope == PRASelectionLayerScope.PER_LAYER or source == layer_id:
            return source_rows
        return self.handle.map_chunk_identities_to_layers(source_rows, {layer_id})[layer_id]

    def prepare_memory(
        self,
        *,
        adapter: "PRAHFAttentionAdapter",
        query: "torch.Tensor",
        routing_query_states: "torch.Tensor",
        direct_tokens: int,
    ) -> "PreparedPRAMemory":
        """Select logical identities and materialize this layer's native K/V."""

        layer_id = adapter.layer_idx
        policy = self.context.policy
        self.context.phase = "prefill" if self.context.token_index == 0 else "decode"
        if (
            policy.selection_layer_scope == PRASelectionLayerScope.SHARED
            and layer_id < self.routing_layer
        ):
            return adapter.pra_core.prepare_selected_memory(
                query,
                [[] for _ in range(int(query.shape[0]))],
                direct_tokens=direct_tokens,
            )

        routed: dict[str, object] = {}

        def route(current_layer: int):
            native, rankings, duration = self._route(adapter, routing_query_states)
            routed["native"] = native
            routed["rankings"] = rankings
            routed["duration"] = duration
            return selected_rows_to_identities(native)

        before_epoch = self.context._epoch
        plan = self.controller.selection_for(
            context=self.context,
            layer_id=layer_id,
            phase=self.context.phase,
            token_index=self.context.token_index,
            route=route,
        )
        if self.context._epoch != before_epoch:
            self._native_rows[(plan.epoch_id, layer_id)] = routed["native"]
            self._rankings[(plan.epoch_id, layer_id)] = routed["rankings"]
            if self.context.trace and self.context.trace[-1].get("event") == "selection":
                self.context.trace[-1]["routing_seconds"] = float(routed["duration"])
        elif plan.layer_scope == PRASelectionLayerScope.PER_LAYER:
            # A reused per-layer plan may have been assembled over several
            # epochs. Find the latest native rows retained for this layer.
            candidates = [
                (epoch, rows)
                for (epoch, source_layer), rows in self._native_rows.items()
                if source_layer == layer_id and epoch <= plan.epoch_id
            ]
            if candidates:
                self._native_rows[(plan.epoch_id, layer_id)] = max(candidates)[1]

        selected = self._selected_for_layer(plan, layer_id)
        rankings = self._rankings.get((plan.epoch_id, plan.source_layer), None)
        prepared = adapter.pra_core.prepare_selected_memory(
            query,
            selected,
            direct_tokens=direct_tokens,
            rankings=rankings,
            routing_duration_seconds=float(routed.get("duration", 0.0)),
        )
        self.context.trace.append(
            {
                "event": "materialization",
                "epoch_id": plan.epoch_id,
                "phase": self.context.phase,
                "token_index": self.context.token_index,
                "layer_id": layer_id,
                "selected_tokens": sum(prepared.selected_lengths),
                "cache_hit": False,
            }
        )
        if layer_id == self.active_layers[-1]:
            self.context.token_index += 1
        return prepared
