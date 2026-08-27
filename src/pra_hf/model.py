"""High-level model and reference lifecycle for PRA-HF."""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from pra_torch.hf import inject_pra
from pra_torch.execution import (
    PRAExecutionCapabilities,
    PRAExecutionPolicy,
    PRARequestExecutionContext,
    PRASelectionLayerScope,
    PRASelectionPlan,
    PRASelectionStage,
    resolve_execution_policy,
    resolve_routing_layer,
)
from pra_torch.memory import SelectedChunk

from .config import PRAConfig
from .hybrid_discovery import TokenNativeIndex
from .iterative import (
    GistIndex,
    HierarchicalGistIndex,
    HierarchicalLocalGistRouter,
    IterativeGistRouter,
    IterativeRoutingResult,
)
from .memory_adapter import PRAMemoryAdapter
from .router import PRARouter
from .hf_execution import PRAHFExecutionBridge, selected_rows_to_identities


@dataclass(frozen=True)
class ReferenceHandle:
    """Stable user-facing identity and size summary for one cached reference."""

    id: str
    uri: str
    tokens: int
    chunks: int


@dataclass(frozen=True)
class GenerationResult:
    """Generated text plus the routing and systems summary for one request."""

    text: str
    prompt_tokens: int
    generated_tokens: int
    latency_seconds: float
    stats: dict[str, Any]


@dataclass(frozen=True)
class RoutingResult:
    """Production PRA selection without autoregressive generation."""

    prompt_tokens: int
    selected: tuple[dict[str, Any], ...]
    rankings: tuple[dict[str, Any], ...]
    query_encoding_seconds: float
    routing_seconds: float
    stats: dict[str, Any]


class PRAForCausalLM:
    """A supported frozen Hugging Face causal LM with bounded PRA memory."""

    def __init__(
        self,
        model,
        tokenizer,
        config: PRAConfig,
        router: PRARouter | None = None,
        memory_adapter: PRAMemoryAdapter | None = None,
        pra_execution_policy: PRAExecutionPolicy | dict[str, object] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        if config.routing_mode == "local_iterative" and router is None:
            raise ValueError(
                "local_iterative routing requires a routing adapter with aligned W_q/W_m projections."
            )
        layer_count = len(model.model.layers)
        self.routing_layer, self.consumption_layers = config.resolved_layers(model.config)
        self._handle = inject_pra(
            model,
            config.to_internal(model.config),
            routing_projection=router,
        )
        self.router = router
        self.memory_adapter: PRAMemoryAdapter | None = None
        self._references: dict[str, ReferenceHandle] = {}
        self._last_stats: dict[str, Any] = {}
        self._execution_policy = pra_execution_policy
        self._execution_lock = threading.RLock()
        self._handle.set_memory_enabled(False)
        if memory_adapter is not None:
            self._install_memory_adapter(memory_adapter)
            self._validate_memory_adapter_compatibility()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        routing_adapter: str | Path | None = None,
        memory_adapter: str | Path | None = None,
        pra_config: PRAConfig | dict[str, Any] | None = None,
        pra_execution_policy: PRAExecutionPolicy | dict[str, object] | None = None,
        tokenizer_name_or_path: str | None = None,
        **model_kwargs,
    ) -> "PRAForCausalLM":
        """Load a supported Qwen, Llama, or Gemma 3 LM and inject PRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        config = (
            pra_config
            if isinstance(pra_config, PRAConfig)
            else PRAConfig.from_dict(pra_config or {})
        )
        model_kwargs.setdefault("attn_implementation", "eager")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        tokenizer_kwargs = {}
        if "revision" in model_kwargs:
            tokenizer_kwargs["revision"] = model_kwargs["revision"]
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path or model_name_or_path, **tokenizer_kwargs
        )
        router = PRARouter.from_pretrained(routing_adapter) if routing_adapter else None
        conditional = (
            PRAMemoryAdapter.from_pretrained(memory_adapter)
            if memory_adapter
            else None
        )
        instance = cls(
            model,
            tokenizer,
            config,
            router,
            conditional,
            pra_execution_policy,
        )
        instance._validate_router_compatibility(
            model_name_or_path, model_kwargs.get("revision")
        )
        instance._validate_memory_adapter_compatibility(
            model_name_or_path, model_kwargs.get("revision")
        )
        return instance

    @classmethod
    def from_model(
        cls,
        model,
        tokenizer,
        *,
        pra_config: PRAConfig | None = None,
        router: PRARouter | None = None,
        memory_adapter: PRAMemoryAdapter | None = None,
        pra_execution_policy: PRAExecutionPolicy | dict[str, object] | None = None,
    ) -> "PRAForCausalLM":
        """Wrap an already-loaded model; useful for offline tests and custom loading."""
        return cls(
            model,
            tokenizer,
            pra_config or PRAConfig(),
            router,
            memory_adapter,
            pra_execution_policy,
        )

    @property
    def device(self) -> torch.device:
        return self._handle.device

    @property
    def execution_capabilities(self) -> PRAExecutionCapabilities:
        """Report the policy combinations implemented by the HF reference path."""

        return PRAExecutionCapabilities(
            engine="huggingface_eager",
            request_selection=True,
            phase_selection=False,
            token_selection=True,
            shared_layer_selection=True,
            per_layer_selection=True,
            request_materialization=True,
            layer_materialization=True,
            token_materialization=True,
            keep_residency=True,
            layer_lifetime_residency=True,
            external_kv=True,
        )

    def set_execution_policy(
        self, policy: PRAExecutionPolicy | dict[str, object] | None = None, **overrides
    ) -> None:
        """Set the model-level default used by subsequent independent requests."""

        if policy is not None and overrides:
            raise ValueError("Pass a policy object/mapping or keyword overrides, not both.")
        self._execution_policy = overrides or policy

    def _resolve_execution_policy(self, request_policy=None):
        return resolve_execution_policy(
            request_policy=request_policy,
            model_policy=self._execution_policy,
            capabilities=self.execution_capabilities,
            active_layers=self.consumption_layers,
            configured_routing_layer=self.routing_layer,
        )

    def _validate_router_compatibility(
        self,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> None:
        if self.router is None:
            return
        hidden = int(self.model.config.hidden_size)
        if self.router.input_width != hidden:
            raise ValueError(
                f"Router input width {self.router.input_width} does not match model hidden size {hidden}."
            )
        expected = self.router.metadata.get("base_model")
        if expected and model_name and str(expected) != str(model_name):
            raise ValueError(f"Router expects base model {expected!r}, received {model_name!r}.")
        expected_revision = self.router.metadata.get("base_model_revision")
        if expected_revision and revision and str(expected_revision) != str(revision):
            raise ValueError(
                f"Router expects base revision {expected_revision!r}, received {revision!r}."
            )

    def _validate_memory_adapter_compatibility(
        self,
        model_name: str | None = None,
        revision: str | None = None,
    ) -> None:
        if self.memory_adapter is None:
            return
        metadata = self.memory_adapter.metadata
        expected = metadata.get("base_model")
        if expected and model_name and str(expected) != str(model_name):
            raise ValueError(
                f"Memory adapter expects base model {expected!r}, received {model_name!r}."
            )
        expected_revision = metadata.get("base_model_revision")
        if expected_revision and revision and str(expected_revision) != str(revision):
            raise ValueError(
                "Memory adapter expects base revision "
                f"{expected_revision!r}, received {revision!r}."
            )
        expected_family = metadata.get("model_family")
        family = next(iter(self._handle.adapters.values())).family
        if expected_family and str(expected_family) != str(family):
            raise ValueError(
                f"Memory adapter expects family {expected_family!r}, received {family!r}."
            )

    def _install_memory_adapter(self, adapter: PRAMemoryAdapter) -> None:
        adapter.apply(self._handle)
        self.memory_adapter = adapter

    def load_router(self, directory: str | Path) -> None:
        """Load a router before reference ingestion so cached gists use its space."""
        if not self._handle.cache.is_empty():
            raise RuntimeError("Clear references before replacing the routing adapter.")
        router = PRARouter.from_pretrained(directory, device=self.device)
        self.router = router
        self._validate_router_compatibility()
        self._handle.routing_projection = router
        for adapter in self._handle.adapters.values():
            adapter.__dict__["routing_projection"] = router

    def load_memory_adapter(self, directory: str | Path) -> None:
        """Load conditional memory-use weights before reference ingestion."""

        if not self._handle.cache.is_empty():
            raise RuntimeError("Clear references before replacing the memory adapter.")
        adapter = PRAMemoryAdapter.from_pretrained(directory, device=self.device)
        self._install_memory_adapter(adapter)
        self._validate_memory_adapter_compatibility()

    def add_reference(
        self,
        reference: str,
        *,
        text: str | None = None,
        uri: str | None = None,
    ) -> ReferenceHandle:
        """Add text, a local text file, or explicit ``(uri, text)`` memory."""
        if text is not None:
            if uri is not None:
                raise ValueError("Pass the URI as reference or with uri=, not both.")
            uri, content = reference, text
        else:
            candidate = Path(reference)
            is_file = False
            if "\n" not in reference and len(reference) < 512:
                try:
                    is_file = candidate.is_file()
                except OSError:
                    is_file = False
            if is_file:
                return self.add_reference_file(candidate, uri=uri)
            content = reference
            uri = uri or f"memory://{uuid.uuid4().hex}"
        if not content.strip():
            raise ValueError("Reference text cannot be empty.")
        encoded = self.tokenizer(content, return_tensors="pt", add_special_tokens=False)
        entry = self._handle.add_reference(uri, encoded.input_ids, text=content)
        chunks = len(entry.layer_memory[self.routing_layer].chunks)
        handle = ReferenceHandle(uri, uri, int(encoded.input_ids.shape[1]), chunks)
        self._references[uri] = handle
        return handle

    def add_reference_file(
        self,
        path: str | Path,
        *,
        uri: str | None = None,
        encoding: str = "utf-8",
    ) -> ReferenceHandle:
        """Read and index one local text file."""
        path = Path(path).resolve()
        return self.add_reference(
            path.read_text(encoding=encoding), uri=uri or path.as_uri()
        )

    def remove_reference(self, reference: str | ReferenceHandle) -> None:
        """Remove one reference and invalidate the exact routing index."""
        uri = reference.uri if isinstance(reference, ReferenceHandle) else reference
        if uri not in self._references and not any(
            entry.uri == uri for entry in self._handle.cache.all_entries()
        ):
            raise KeyError(f"Unknown PRA reference: {uri}")
        self._handle.cache.invalidate(uri)
        self._references.pop(uri, None)

    def clear_references(self) -> None:
        """Drop explicit references and implicit prompt-head memory."""
        self._handle.cache.clear()
        self._references.clear()
        self._handle.configure_memory_layers(set())

    def enable(self) -> None:
        self.config.enabled = True

    def disable(self) -> None:
        self.config.enabled = False
        self._handle.configure_memory_layers(set())

    def _selected_from_rankings(
        self,
        rankings: list[list[dict]],
        *,
        layer_id: int | None = None,
    ) -> list[list[SelectedChunk]]:
        layer_id = self.routing_layer if layer_id is None else int(layer_id)
        entries = {entry.uri: entry for entry in self._handle.cache.all_entries()}
        rows: list[list[SelectedChunk]] = []
        for ranking in rankings:
            candidates = []
            for reference in ranking:
                entry = entries[reference["reference_uri"]]
                chunks = {
                    chunk.chunk_id: chunk
                    for chunk in entry.layer_memory[layer_id].chunks
                }
                for chunk_row in reference["chunks"]:
                    candidates.append((reference, chunk_row, entry, chunks[chunk_row["chunk_id"]]))
            candidates.sort(key=lambda item: (-float(item[1]["chunk_score"]), item[1]["chunk_id"]))
            if self.config.selected_fraction is not None:
                cutoff = max(1, math.ceil(self.config.selected_fraction * len(candidates)))
            else:
                cutoff = min(self.config.top_k, len(candidates))
            selected = []
            for reference, chunk_row, entry, chunk in candidates[:cutoff]:
                selected.append(
                    SelectedChunk(
                        entry=entry,
                        chunk=chunk,
                        reference_score=float(reference["reference_score"]),
                        chunk_score=float(chunk_row["chunk_score"]),
                        layer_id=layer_id,
                        reference_rank=int(reference["reference_rank"]),
                        rank_within_reference=int(chunk_row["chunk_rank"]),
                        winning_gist_index=chunk_row.get("winning_gist_index"),
                        winning_gist_score=chunk_row.get("winning_gist_score"),
                        gist_count=int(chunk_row.get("gist_count", 1)),
                        metadata={"selection_policy": self.config.selection_policy},
                    )
                )
            rows.append(selected)
        return rows

    def _iterative_rankings(
        self,
        index: GistIndex,
        results: list[IterativeRoutingResult],
    ) -> list[list[dict]]:
        """Expose complete root-query rankings in the existing diagnostic shape."""
        rows = []
        for result in results:
            grouped: dict[str, list[tuple[int, float]]] = {}
            for candidate_index, ((entry, _), score) in enumerate(
                zip(index.records, result.direct_scores)
            ):
                grouped.setdefault(entry.uri, []).append((candidate_index, score))
            references = sorted(
                grouped,
                key=lambda uri: (-max(score for _, score in grouped[uri]), uri),
            )
            ranking = []
            for reference_rank, uri in enumerate(references, start=1):
                chunks = sorted(
                    grouped[uri],
                    key=lambda row: (-row[1], index.records[row[0]][1].chunk_id),
                )
                ranking.append(
                    {
                        "reference_uri": uri,
                        "reference_rank": reference_rank,
                        "reference_score": float(chunks[0][1]),
                        "chunks": [
                            {
                                "chunk_id": index.records[candidate_index][1].chunk_id,
                                "chunk_rank": chunk_rank,
                                "chunk_score": float(score),
                                "token_start": index.records[candidate_index][1].token_start,
                                "token_end": index.records[candidate_index][1].token_end,
                                "gist_count": int(
                                    index.records[candidate_index][1].routing_gist.k.shape[0]
                                ),
                            }
                            for chunk_rank, (candidate_index, score) in enumerate(chunks, start=1)
                        ],
                    }
                )
            rows.append(ranking)
        return rows

    @torch.no_grad()
    def _route_once(
        self, input_ids, attention_mask, position_ids, *, routing_layer: int | None = None
    ):
        routing_layer = self.routing_layer if routing_layer is None else int(routing_layer)
        adapter = self._handle.adapters[routing_layer]
        self._handle.configure_memory_layers(set())
        adapter.begin_capture(position_ids)
        started = time.perf_counter()
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        query_seconds = time.perf_counter() - started
        captured = adapter.consume_capture()
        query = adapter._routing_query_states(
            captured.hidden_states, captured.pre_query, captured.post_query
        )
        started = time.perf_counter()
        retrieval_graphs = []
        if self.config.routing_mode == "local_iterative":
            index = HierarchicalGistIndex.from_entries(
                self._handle.cache.all_entries(),
                routing_layer,
                device=query.device,
                dtype=query.dtype,
            )
            iterative = HierarchicalLocalGistRouter(index)
            results = [
                iterative.route(row, self.config.iterative_config)
                for row in query
            ]
            selected = [iterative.selected_chunks(result) for result in results]
            # Parent root scores retain the same diagnostic ranking contract.
            simple_index = GistIndex.from_entries(
                self._handle.cache.all_entries(), routing_layer,
                device=query.device, dtype=query.dtype,
            )
            rankings = self._iterative_rankings(simple_index, results)
            retrieval_graphs = [result.graph.to_dict() for result in results]
        elif self.config.routing_mode in {
            "iterative",
            "token_iterative",
            "hybrid_iterative",
        }:
            index = GistIndex.from_entries(
                self._handle.cache.all_entries(),
                routing_layer,
                device=query.device,
                dtype=query.dtype,
            )
            iterative = IterativeGistRouter(index)
            if self.config.routing_mode == "iterative":
                results = iterative.route_batch(query, self.config.iterative_config)
            else:
                token_index = TokenNativeIndex.from_gist_index(index, self.tokenizer)
                results = []
                for row_index, row in enumerate(query):
                    prompt_ids = input_ids[row_index][attention_mask[row_index].bool()]
                    results.append(
                        iterative.route(
                            row,
                            self.config.iterative_config,
                            token_index=token_index,
                            root_token_ids=prompt_ids.detach().cpu().tolist(),
                            tokenizer=self.tokenizer,
                            discovery_policy=self.config.hybrid_discovery_policy,
                        )
                    )
            selected = [iterative.selected_chunks(result) for result in results]
            rankings = self._iterative_rankings(index, results)
            retrieval_graphs = [result.graph.to_dict() for result in results]
        else:
            _, rankings = adapter.pra_core.route_memory(query)
            selected = self._selected_from_rankings(rankings, layer_id=routing_layer)
        routing_seconds = time.perf_counter() - started
        fixed = self._handle.map_chunk_identities_to_layers(
            selected, self.consumption_layers
        )
        self._handle.configure_memory_layers(
            set(self.consumption_layers), fixed_selections=fixed
        )
        # The graph is a retrieval artifact until fixed identities have been
        # mapped to layer-native payloads.  Mark only this post-mapping state as
        # materialized so Paper 3 can distinguish selection from K/V activation.
        for graph in retrieval_graphs:
            for node in graph["nodes"]:
                if node["final_selected"]:
                    node["materialized"] = True
        return selected, rankings, query_seconds, routing_seconds, retrieval_graphs

    @torch.no_grad()
    def _route_request_per_layer(
        self,
        input_ids,
        attention_mask,
        position_ids,
        context: PRARequestExecutionContext,
    ):
        """Capture once, independently route every layer, then freeze each plan."""

        if self.config.routing_mode != "one_shot":
            raise ValueError(
                "REQUEST+PER_LAYER currently supports one_shot routing only; "
                "no fallback to shared routing was applied."
            )
        self._handle.configure_memory_layers(set())
        adapters = {
            layer: self._handle.adapters[layer] for layer in self.consumption_layers
        }
        for adapter in adapters.values():
            adapter.begin_capture(position_ids)
        started = time.perf_counter()
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        query_seconds = time.perf_counter() - started
        fixed = {}
        rankings_by_layer = {}
        routing_seconds = 0.0
        for layer, adapter in adapters.items():
            captured = adapter.consume_capture()
            query = adapter._routing_query_states(
                captured.hidden_states, captured.pre_query, captured.post_query
            )
            started = time.perf_counter()
            selected, rankings = adapter.pra_core.route_memory(query)
            routing_seconds += time.perf_counter() - started
            fixed[layer] = selected
            rankings_by_layer[layer] = rankings
        plan = PRASelectionPlan(
            selection_stage=PRASelectionStage.REQUEST,
            layer_scope=PRASelectionLayerScope.PER_LAYER,
            source_layer=None,
            epoch_id=context.next_epoch(),
            per_layer_rows={
                layer: selected_rows_to_identities(rows)
                for layer, rows in fixed.items()
            },
            phase="request",
            token_index=0,
            routing_seconds=routing_seconds,
        )
        context.record_plan(plan)
        context.trace[-1]["routing_operations"] = len(fixed)
        self._handle.configure_memory_layers(
            set(self.consumption_layers), fixed_selections=fixed
        )
        canonical_layer = (
            self.routing_layer if self.routing_layer in fixed else self.consumption_layers[0]
        )
        return (
            fixed[canonical_layer],
            rankings_by_layer[canonical_layer],
            query_seconds,
            routing_seconds,
            [],
        )

    def _record_request_shared_plan(
        self,
        context: PRARequestExecutionContext,
        selected: list[list[SelectedChunk]],
        routing_seconds: float,
    ) -> None:
        plan = PRASelectionPlan(
            selection_stage=PRASelectionStage.REQUEST,
            layer_scope=PRASelectionLayerScope.SHARED,
            source_layer=self.routing_layer,
            epoch_id=context.next_epoch(),
            shared_rows=selected_rows_to_identities(selected),
            phase="request",
            token_index=0,
            routing_seconds=routing_seconds,
        )
        context.record_plan(plan)

    @torch.no_grad()
    def route(self, prompt: str) -> RoutingResult:
        """Run production query encoding, routing, and native-K/V selection only.

        Hosts use this route-only path when a compact result descriptor points to
        exact backing state and generation must wait for an auditable retrieval
        decision. It applies the same configured selection budget as ``generate``.
        """

        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prepared = self._handle.prepare_long_prompt(encoded.input_ids.to(self.device))
        input_ids = prepared.input_ids.to(self.device)
        attention_mask = prepared.attention_mask.to(self.device)
        position_ids = prepared.position_ids.to(self.device)
        selected: list[list[SelectedChunk]] = [[]]
        rankings: list[list[dict]] = [[]]
        query_seconds = routing_seconds = 0.0
        retrieval_graphs: list[dict[str, Any]] = []
        if self.config.enabled and not self._handle.cache.is_empty():
            selected, rankings, query_seconds, routing_seconds, retrieval_graphs = (
                self._route_once(input_ids, attention_mask, position_ids)
            )
        else:
            self._handle.configure_memory_layers(set())
        candidate_tokens = sum(
            chunk.token_count
            for entry in self._handle.cache.all_entries()
            for memory in [entry.layer_memory.get(self.routing_layer)]
            if memory is not None
            for chunk in memory.chunks
        )
        candidate_count = sum(
            len(reference["chunks"]) for reference in rankings[0]
        ) if rankings else 0
        selected_tokens = sum(hit.selected_token_count for hit in selected[0])
        diagnostics = self._handle.diagnostics_by_layer()
        routing_diagnostics = diagnostics.get(self.routing_layer, {})
        materialized_tokens = int(
            routing_diagnostics.get("memory_tokens_materialized", 0)
        )
        self._last_stats = {
            "selection_policy": self.config.selection_policy,
            "candidate_chunks": candidate_count,
            "requested_chunks": len(selected[0]),
            "requested_chunk_fraction": len(selected[0]) / max(candidate_count, 1),
            "candidate_kv_tokens": candidate_tokens,
            "requested_kv_tokens": selected_tokens,
            "requested_kv_token_fraction": selected_tokens / max(candidate_tokens, 1),
            "materialized_kv_tokens": materialized_tokens,
            "materialized_kv_token_fraction": materialized_tokens / max(candidate_tokens, 1),
            "selected": [hit.as_trace_dict() for hit in selected[0]],
            "query_encoding_seconds": query_seconds,
            "routing_seconds": routing_seconds,
            "retrieval_graphs": retrieval_graphs,
            "generation_seconds": 0.0,
            "diagnostics_by_layer": diagnostics,
            "head_tokens": prepared.head_tokens,
        }
        return RoutingResult(
            prompt_tokens=int(input_ids.shape[1]),
            selected=tuple(self._last_stats["selected"]),
            rankings=tuple(rankings[0]),
            query_encoding_seconds=query_seconds,
            routing_seconds=routing_seconds,
            stats=dict(self._last_stats),
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        return_details: bool = False,
        pra_policy: PRAExecutionPolicy | dict[str, object] | None = None,
        **generation_kwargs,
    ) -> str | GenerationResult:
        """Generate with an isolated request override of the execution policy."""

        with self._execution_lock:
            return self._generate_locked(
                prompt,
                max_new_tokens=max_new_tokens,
                return_details=return_details,
                pra_policy=pra_policy,
                **generation_kwargs,
            )

    @torch.no_grad()
    def _generate_locked(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        return_details: bool,
        pra_policy: PRAExecutionPolicy | dict[str, object] | None,
        **generation_kwargs,
    ) -> str | GenerationResult:
        """Execute one serialized HF request after resolving policy precedence."""

        resolved = self._resolve_execution_policy(pra_policy)
        context = PRARequestExecutionContext(resolved)
        routing_layer = resolve_routing_layer(
            resolved.policy,
            self.consumption_layers,
            self.routing_layer,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prepared = self._handle.prepare_long_prompt(encoded.input_ids.to(self.device))
        input_ids = prepared.input_ids.to(self.device)
        attention_mask = prepared.attention_mask.to(self.device)
        position_ids = prepared.position_ids.to(self.device)
        selected: list[list[SelectedChunk]] = [[]]
        rankings: list[list[dict]] = [[]]
        query_seconds = routing_seconds = 0.0
        retrieval_graphs: list[dict[str, Any]] = []
        bridge = None
        if self.config.enabled and not self._handle.cache.is_empty():
            if resolved.policy.selection_stage == PRASelectionStage.REQUEST:
                if (
                    resolved.policy.selection_layer_scope
                    == PRASelectionLayerScope.SHARED
                ):
                    (
                        selected,
                        rankings,
                        query_seconds,
                        routing_seconds,
                        retrieval_graphs,
                    ) = self._route_once(
                        input_ids,
                        attention_mask,
                        position_ids,
                        routing_layer=routing_layer,
                    )
                    self._record_request_shared_plan(
                        context, selected, routing_seconds
                    )
                else:
                    (
                        selected,
                        rankings,
                        query_seconds,
                        routing_seconds,
                        retrieval_graphs,
                    ) = self._route_request_per_layer(
                        input_ids, attention_mask, position_ids, context
                    )
            else:
                if self.config.routing_mode != "one_shot":
                    raise ValueError(
                        "Dynamic execution policies currently support one_shot "
                        "routing only; no request-level fallback was applied."
                    )
                self._handle.configure_memory_layers(set(self.consumption_layers))
                bridge = PRAHFExecutionBridge(
                    self._handle,
                    context,
                    active_layers=self.consumption_layers,
                    routing_layer=routing_layer,
                )
                self._handle.set_execution_bridge(bridge)
        else:
            self._handle.configure_memory_layers(set())
        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            if major < 7:
                # Transformers auto-compiles some generation caches, but Triton
                # supports only compute capability 7.0 and newer.
                generation_kwargs.setdefault("disable_compile", True)
        started = time.perf_counter()
        try:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
            )
        finally:
            if bridge is not None:
                self._handle.set_execution_bridge(None)
                self._handle.configure_memory_layers(set())
        latency = time.perf_counter() - started
        generated = output[:, input_ids.shape[1] :]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        candidate_count = sum(len(ref["chunks"]) for ref in rankings[0]) if rankings else 0
        candidate_tokens = sum(
            chunk.token_count
            for entry in self._handle.cache.all_entries()
            for memory in [entry.layer_memory.get(routing_layer)]
            if memory is not None
            for chunk in memory.chunks
        )
        if selected and selected[0]:
            selected_tokens = sum(hit.selected_token_count for hit in selected[0])
            selected_trace = [hit.as_trace_dict() for hit in selected[0]]
        elif context.selection_plan is not None:
            logical = context.selection_plan.rows_for(
                routing_layer
                if context.selection_plan.layer_scope
                == PRASelectionLayerScope.PER_LAYER
                else routing_layer
            )[0]
            selected_tokens = sum(item.token_end - item.token_start for item in logical)
            selected_trace = [
                {
                    "reference_uri": item.reference_uri,
                    "chunk_id": item.chunk_id,
                    "token_start": item.token_start,
                    "token_end": item.token_end,
                }
                for item in logical
            ]
        else:
            selected_tokens = 0
            selected_trace = []
        if bridge is not None:
            routing_seconds = sum(
                float(row.get("routing_seconds", 0.0))
                for row in context.trace
                if row.get("event") == "selection"
            )
        diagnostics = self._handle.diagnostics_by_layer()
        routing_diagnostics = diagnostics.get(routing_layer, {})
        materialized_tokens = int(
            routing_diagnostics.get("memory_tokens_materialized", 0)
        )
        self._last_stats = {
            "selection_policy": self.config.selection_policy,
            "candidate_chunks": candidate_count,
            "requested_chunks": len(selected[0]),
            "requested_chunk_fraction": len(selected[0]) / max(candidate_count, 1),
            "candidate_kv_tokens": candidate_tokens,
            "requested_kv_tokens": selected_tokens,
            "requested_kv_token_fraction": selected_tokens / max(candidate_tokens, 1),
            "materialized_kv_tokens": materialized_tokens,
            "materialized_kv_token_fraction": materialized_tokens
            / max(candidate_tokens, 1),
            "selected": selected_trace,
            "query_encoding_seconds": query_seconds,
            "routing_seconds": routing_seconds,
            "retrieval_graphs": retrieval_graphs,
            "generation_seconds": latency,
            "diagnostics_by_layer": diagnostics,
            "head_tokens": prepared.head_tokens,
            "pra_execution": context.summary(),
        }
        result = GenerationResult(
            text=text,
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=int(generated.shape[1]),
            latency_seconds=latency,
            stats=self._last_stats,
        )
        return result if return_details else result.text

    def chat(self, messages: Iterable[dict[str, str]], **generation_kwargs):
        """Format chat messages with the base tokenizer and call ``generate``."""
        messages = list(messages)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{row['role']}: {row['content']}" for row in messages)
        return self.generate(prompt, **generation_kwargs)

    def stats(self) -> dict[str, Any]:
        """Return reference inventory, memory economics, and the latest request trace."""
        entries = self._handle.cache.all_entries()
        route_chunks = [
            chunk
            for entry in entries
            for memory in [entry.layer_memory.get(self.routing_layer)]
            if memory is not None
            for chunk in memory.chunks
        ]
        routing_bytes = sum(
            int(chunk.metadata.get("routing_gist_bytes", 0)) for chunk in route_chunks
        )
        detail_bytes = sum(
            int(chunk.metadata.get("detail_kv_bytes", 0))
            for entry in entries
            for memory in entry.layer_memory.values()
            for chunk in memory.chunks
        )
        return {
            "enabled": self.config.enabled,
            "family": next(iter(self._handle.adapters.values())).family,
            "routing_layer": self.routing_layer,
            "consumption_layers": list(self.consumption_layers),
            "router_parameters": self.router.parameter_count if self.router else 0,
            "memory_adapter_parameters": (
                self.memory_adapter.parameter_count if self.memory_adapter else 0
            ),
            "memory_adapter": (
                self.memory_adapter.artifact_config() if self.memory_adapter else None
            ),
            "references": [asdict(handle) for handle in self._references.values()],
            "routing_index_bytes": routing_bytes,
            "resident_detail_kv_bytes": detail_bytes,
            "max_native_operation_tokens": self._handle.max_native_operation_tokens,
            "native_limit_violations": self._handle.native_limit_violations,
            "last_request": self._last_stats,
        }
