"""High-level model and reference lifecycle for PRA-HF."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from pra_torch.hf import inject_pra
from pra_torch.memory import SelectedChunk

from .config import PRAConfig
from .iterative import (
    GistIndex,
    HierarchicalGistIndex,
    HierarchicalLocalGistRouter,
    IterativeGistRouter,
    IterativeRoutingResult,
)
from .memory_adapter import PRAMemoryAdapter
from .router import PRARouter


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


class PRAForCausalLM:
    """A supported frozen Hugging Face causal LM with bounded PRA memory."""

    def __init__(
        self,
        model,
        tokenizer,
        config: PRAConfig,
        router: PRARouter | None = None,
        memory_adapter: PRAMemoryAdapter | None = None,
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
        instance = cls(model, tokenizer, config, router, conditional)
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
    ) -> "PRAForCausalLM":
        """Wrap an already-loaded model; useful for offline tests and custom loading."""
        return cls(
            model,
            tokenizer,
            pra_config or PRAConfig(),
            router,
            memory_adapter,
        )

    @property
    def device(self) -> torch.device:
        return self._handle.device

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

    def _selected_from_rankings(self, rankings: list[list[dict]]) -> list[list[SelectedChunk]]:
        entries = {entry.uri: entry for entry in self._handle.cache.all_entries()}
        rows: list[list[SelectedChunk]] = []
        for ranking in rankings:
            candidates = []
            for reference in ranking:
                entry = entries[reference["reference_uri"]]
                chunks = {
                    chunk.chunk_id: chunk
                    for chunk in entry.layer_memory[self.routing_layer].chunks
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
                        layer_id=self.routing_layer,
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
    def _route_once(self, input_ids, attention_mask, position_ids):
        adapter = self._handle.adapters[self.routing_layer]
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
                self.routing_layer,
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
                self._handle.cache.all_entries(), self.routing_layer,
                device=query.device, dtype=query.dtype,
            )
            rankings = self._iterative_rankings(simple_index, results)
            retrieval_graphs = [result.graph.to_dict() for result in results]
        elif self.config.routing_mode == "iterative":
            index = GistIndex.from_entries(
                self._handle.cache.all_entries(),
                self.routing_layer,
                device=query.device,
                dtype=query.dtype,
            )
            iterative = IterativeGistRouter(index)
            results = iterative.route_batch(query, self.config.iterative_config)
            selected = [iterative.selected_chunks(result) for result in results]
            rankings = self._iterative_rankings(index, results)
            retrieval_graphs = [result.graph.to_dict() for result in results]
        else:
            _, rankings = adapter.pra_core.route_memory(query)
            selected = self._selected_from_rankings(rankings)
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
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        return_details: bool = False,
        **generation_kwargs,
    ) -> str | GenerationResult:
        """Route references once, then generate with bounded layer-native memory."""
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
            selected, rankings, query_seconds, routing_seconds, retrieval_graphs = self._route_once(
                input_ids, attention_mask, position_ids
            )
        else:
            self._handle.configure_memory_layers(set())
        if self.device.type == "cuda":
            major, _ = torch.cuda.get_device_capability(self.device)
            if major < 7:
                # Transformers auto-compiles some generation caches, but Triton
                # supports only compute capability 7.0 and newer.
                generation_kwargs.setdefault("disable_compile", True)
        started = time.perf_counter()
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )
        latency = time.perf_counter() - started
        generated = output[:, input_ids.shape[1] :]
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        candidate_count = sum(len(ref["chunks"]) for ref in rankings[0]) if rankings else 0
        candidate_tokens = sum(
            chunk.token_count
            for entry in self._handle.cache.all_entries()
            for memory in [entry.layer_memory.get(self.routing_layer)]
            if memory is not None
            for chunk in memory.chunks
        )
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
            "materialized_kv_token_fraction": materialized_tokens
            / max(candidate_tokens, 1),
            "selected": [hit.as_trace_dict() for hit in selected[0]],
            "query_encoding_seconds": query_seconds,
            "routing_seconds": routing_seconds,
            "retrieval_graphs": retrieval_graphs,
            "generation_seconds": latency,
            "diagnostics_by_layer": diagnostics,
            "head_tokens": prepared.head_tokens,
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
