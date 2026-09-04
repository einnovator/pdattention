"""Run the powered MultiHop-RAG Standard-RAG/PRA decomposition.

The first-stage receipt is shared by every selector. Each PRA selector is then
frozen once and executed as both visible selected text and detached native K/V.
Bundle arms remain explicit unavailable records until an exact MultiHop-RAG
qualified adapter exists for the immutable model identity.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping, Sequence

from experiments.rag_vs_pra.datasets import (
    controlled_fixture,
    load_multihop_rag,
    select_cohort,
)
from pra_hf.rag_evaluation import (
    CandidateReceipt,
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    PRAHybridSelector,
    RAGFailureClass,
    RAGQuestion,
    SelectionReceipt,
    StandardRAGSelector,
    context_metrics,
    failure_classification,
    make_candidate_receipt,
    packed_context_from_ranking,
    prepare_candidate_context,
)
from pra_hf.rag_powered import (
    answer_metrics,
    answer_normalization_diagnostics,
    official_multihop_rag_score,
    write_results,
)


SCHEMA_VERSION = "pra-rag-powered-v1"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _hardware() -> dict[str, object]:
    result: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if platform.system() == "Darwin":
        for key, command in {
            "chip": ["sysctl", "-n", "machdep.cpu.brand_string"],
            "memory_bytes": ["sysctl", "-n", "hw.memsize"],
            "model_identifier": ["sysctl", "-n", "hw.model"],
        }.items():
            value = subprocess.run(command, capture_output=True, text=True, check=False)
            if value.returncode == 0:
                text = value.stdout.strip()
                result[key] = int(text) if key == "memory_bytes" else text
    return result


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("mlx", "mlx-lm", "torch", "transformers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def _resolve_hf_revision(model_id: str, revision: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(model_id, revision=revision).sha)
    except Exception as exc:
        raise RuntimeError(
            f"cannot resolve immutable Hugging Face revision for {model_id}@{revision}"
        ) from exc


class ProbeBackend:
    """Dependency-free execution path for contract and artifact smoke tests."""

    name = "evidence_probe_v2"
    publishable_answer_quality = False

    @staticmethod
    def token_count(text: str) -> int:
        return len(text.split())

    def answer(
        self,
        question: RAGQuestion,
        context: str,
        condition: ContextCondition,
        *,
        selection_receipt_id: str,
        regime: str,
        selected_texts: Sequence[str] = (),
    ) -> tuple[str, Mapping[str, object]]:
        started = time.perf_counter()
        prediction = next(
            (answer for answer in question.answers if answer.casefold() in context.casefold()),
            "unknown",
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        native = condition in {
            ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
            ContextCondition.PRA_NATIVE_MEMORY_BUNDLE,
        }
        tokens = self.token_count(context)
        return prediction, {
            "ttft_ms": elapsed,
            "itl_ms": 0.0,
            "prefill_ms": elapsed,
            "completion_latency_ms": 0.0,
            "total_latency_ms": elapsed,
            "generated_tokens": len(prediction.split()),
            "tokens_per_second": len(prediction.split()) / max(elapsed / 1000.0, 1e-9),
            "output_tokens_per_second": len(prediction.split()) / max(elapsed / 1000.0, 1e-9),
            "requests_per_second": 1000.0 / max(elapsed, 1e-9),
            "visible_prompt_tokens": 0 if native else tokens,
            "selected_native_kv_tokens": tokens if native else 0,
            "newly_materialized_tokens": 0 if regime == "WARM" and native else tokens,
            "visible_reuse": 0.0,
            "native_reuse": float(regime == "WARM" and native),
            "ingestion_ms": 0.0,
            "active_detail_bytes": 0,
            "retained_detail_bytes": 0,
            "kv_bytes": 0,
            "peak_memory_bytes": None,
            "temporary_allocation_bytes": None,
            "cache_key": selection_receipt_id,
            "native_state_fingerprint": None,
        }

    def release(self, selection_receipt_id: str) -> None:
        return None


class PersistentMLXBackend:
    """MLX selected-text/native-KV executor with explicit cold/warm lifecycle."""

    publishable_answer_quality = True

    def __init__(
        self,
        model_id: str,
        revision: str,
        max_new_tokens: int,
        *,
        native_cache_unit: str = "selection",
    ) -> None:
        from mlx_lm import load

        self.name = f"mlx_native:{model_id}@{revision}"
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        if native_cache_unit not in {"selection", "chunk"}:
            raise ValueError("native_cache_unit must be selection or chunk")
        self.native_cache_unit = native_cache_unit
        self.model, self.tokenizer = load(model_id, revision=revision)
        self._native_memories: dict[str, object] = {}

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @staticmethod
    def _is_native(condition: ContextCondition) -> bool:
        return condition in {
            ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
            ContextCondition.PRA_NATIVE_MEMORY_BUNDLE,
        }

    def _query(self, question: RAGQuestion) -> str:
        return (
            "/no_think\nUse only the available evidence. Give only the shortest "
            f"supported answer.\nQuestion: {question.question}\nAnswer:"
        )

    def _generate(self, query_tokens, cache):
        import mlx.core as mx
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler

        started = time.perf_counter()
        arrivals: list[float] = []
        generated: list[int] = []
        for token, _ in generate_step(
            mx.array(query_tokens, dtype=mx.int32),
            self.model,
            max_tokens=self.max_new_tokens,
            prompt_cache=cache,
            sampler=make_sampler(temp=0),
        ):
            generated.append(int(token))
            arrivals.append((time.perf_counter() - started) * 1000.0)
        elapsed = (time.perf_counter() - started) * 1000.0
        ttft = arrivals[0] if arrivals else elapsed
        itl = (
            statistics.fmean(right - left for left, right in zip(arrivals, arrivals[1:]))
            if len(arrivals) > 1
            else 0.0
        )
        return self.tokenizer.decode(generated).strip(), {
            "ttft_ms": ttft,
            "itl_ms": itl,
            "completion_latency_ms": max(elapsed - ttft, 0.0),
            "total_latency_ms": elapsed,
            "generated_tokens": len(generated),
            "tokens_per_second": len(generated) / max(elapsed / 1000.0, 1e-9),
            "output_tokens_per_second": len(generated) / max(elapsed / 1000.0, 1e-9),
        }

    def answer(
        self,
        question: RAGQuestion,
        context: str,
        condition: ContextCondition,
        *,
        selection_receipt_id: str,
        regime: str,
        selected_texts: Sequence[str] = (),
    ) -> tuple[str, Mapping[str, object]]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from pra_hf.rag_mlx_native import (
            combine_native_memories,
            encode_native_memory,
            make_native_prompt_cache,
        )

        if regime not in {"COLD", "WARM", "PERSISTENT_CORPUS"}:
            raise ValueError("regime must be COLD, WARM, or PERSISTENT_CORPUS")
        source_tokens = list(
            self.tokenizer.encode(context.rstrip() + "\n\n", add_special_tokens=False)
        )
        query_tokens = list(
            self.tokenizer.encode(self._query(question), add_special_tokens=False)
        )
        native = self._is_native(condition)
        cache_hit = False
        cache_hits = 0
        cache_lookups = 0
        newly_materialized_tokens = len(source_tokens)
        materialized_native_tokens = len(source_tokens)
        if self.native_cache_unit == "selection":
            cache_hit = native and selection_receipt_id in self._native_memories
            if regime == "COLD" and cache_hit:
                del self._native_memories[selection_receipt_id]
                cache_hit = False
        if (
            regime == "WARM"
            and native
            and self.native_cache_unit == "selection"
            and not cache_hit
        ):
            raise RuntimeError("warm native execution has no retained cold memory")

        reset_peak = getattr(mx, "reset_peak_memory", None)
        if reset_peak is not None:
            reset_peak()
        ingestion_started = time.perf_counter()
        if native:
            if self.native_cache_unit == "chunk":
                if not selected_texts:
                    raise ValueError("chunk-resident native execution requires selected texts")
                memories = []
                newly_materialized_tokens = 0
                materialized_native_tokens = 0
                for text in selected_texts:
                    tokens = list(
                        self.tokenizer.encode(text, add_special_tokens=False)
                    )
                    chunk_key = _digest(
                        {
                            "model": self.model_id,
                            "revision": self.revision,
                            "tokens": tokens,
                        }
                    )
                    cache_lookups += 1
                    materialized_native_tokens += len(tokens)
                    chunk_memory = self._native_memories.get(chunk_key)
                    if chunk_memory is None:
                        chunk_memory = encode_native_memory(self.model, tokens)
                        self._native_memories[chunk_key] = chunk_memory
                        newly_materialized_tokens += len(tokens)
                    else:
                        cache_hits += 1
                    memories.append(chunk_memory)
                memory = combine_native_memories(memories)
                cache_hit = cache_hits == cache_lookups
            else:
                cache_lookups = 1
                if cache_hit:
                    memory = self._native_memories[selection_receipt_id]
                    cache_hits = 1
                    newly_materialized_tokens = 0
                else:
                    memory = encode_native_memory(self.model, source_tokens)
                    self._native_memories[selection_receipt_id] = memory
            cache = make_native_prompt_cache(self.model, memory)
            native_bytes = int(memory.nbytes)
        else:
            cache = make_prompt_cache(self.model)
            self.model(mx.array(source_tokens, dtype=mx.int32)[None], cache=cache)
            mx.eval([layer.state for layer in cache])
            native_bytes = 0
        ingestion_ms = (time.perf_counter() - ingestion_started) * 1000.0
        prediction, serving = self._generate(query_tokens, cache)
        decode_total_ms = float(serving["total_latency_ms"])
        decode_ttft_ms = float(serving["ttft_ms"])
        generated_tokens = int(serving["generated_tokens"])
        total_latency_ms = ingestion_ms + decode_total_ms
        peak = int(getattr(mx, "get_peak_memory", lambda: 0)()) or None
        active = int(getattr(mx, "get_active_memory", lambda: 0)()) or None
        source_count = len(source_tokens)
        return prediction, {
            **serving,
            "ttft_ms": ingestion_ms + decode_ttft_ms,
            "prefill_ms": ingestion_ms + decode_ttft_ms,
            "total_latency_ms": total_latency_ms,
            "tokens_per_second": generated_tokens / max(total_latency_ms / 1000.0, 1e-9),
            "output_tokens_per_second": generated_tokens / max(decode_total_ms / 1000.0, 1e-9),
            "requests_per_second": 1000.0 / max(total_latency_ms, 1e-9),
            "ingestion_ms": ingestion_ms,
            "native_encode_ms": ingestion_ms if native and not cache_hit else 0.0,
            "visible_text_ingestion_ms": ingestion_ms if not native else 0.0,
            "active_detail_bytes": native_bytes,
            "retained_detail_bytes": (
                sum(int(value.nbytes) for value in self._native_memories.values())
                if native
                else 0
            ),
            "kv_bytes": native_bytes,
            "visible_prompt_tokens": (
                len(query_tokens) if native else source_count + len(query_tokens)
            ),
            "selected_native_kv_tokens": materialized_native_tokens if native else 0,
            "newly_materialized_tokens": newly_materialized_tokens,
            "visible_reuse": 0.0,
            "native_reuse": (
                cache_hits / cache_lookups if native and cache_lookups else 0.0
            ),
            "native_cache_hits": cache_hits,
            "native_cache_lookups": cache_lookups,
            "native_cache_unit": self.native_cache_unit,
            "peak_memory_bytes": peak,
            "active_memory_bytes": active,
            "temporary_allocation_bytes": (
                max(peak - active, 0)
                if peak is not None and active is not None
                else None
            ),
            "cache_key": selection_receipt_id,
            "native_state_fingerprint": (
                _digest(
                    {
                        "selection_receipt_id": selection_receipt_id,
                        "model_id": self.model_id,
                        "model_revision": self.revision,
                        "source_tokens": materialized_native_tokens,
                        "position_policy": "source_length_query_base_post_rope_kv",
                        "consumer_profile": "all_layers",
                        "cache_unit": self.native_cache_unit,
                    }
                )
                if native
                else None
            ),
        }

    def release(self, selection_receipt_id: str) -> None:
        if self.native_cache_unit == "selection":
            self._native_memories.pop(selection_receipt_id, None)


def _row(
    *,
    question: RAGQuestion,
    receipt: CandidateReceipt,
    selection: SelectionReceipt,
    context,
    backend,
    candidate_count: int,
    token_budget: int,
    selector_profile: str,
    regime: str,
) -> dict[str, object]:
    selection.validate_context(context)
    prediction, serving = backend.answer(
        question,
        context.text,
        context.condition,
        selection_receipt_id=selection.receipt_id,
        regime=regime,
        selected_texts=tuple(row.chunk.text for row in context.chunks),
    )
    exact, token_f1 = answer_metrics(prediction, question.answers)
    normalization = answer_normalization_diagnostics(prediction, question.answers)
    metrics = context_metrics(question, receipt, context)
    answer_format_ok = bool(normalization["answer_format_ok"])
    return {
        "example_id": question.example_id,
        "question_type": question.question_type,
        "condition": context.condition.value,
        "selector_profile": selector_profile,
        "candidate_count": candidate_count,
        "token_budget": token_budget,
        "regime": regime,
        "status": "MEASURED",
        "candidate_receipt_id": receipt.receipt_id,
        "selection_receipt_id": selection.receipt_id,
        "candidate_document_ids": list(receipt.candidate_document_ids),
        "gold_document_ids": sorted(question.gold_document_ids),
        "selected_document_ids": list(context.selected_document_ids),
        "selected_chunk_ids": list(context.selected_chunk_ids),
        "selected_intervals": [asdict(interval) for interval in selection.intervals],
        "prediction": prediction,
        "gold_answers": list(question.answers),
        "exact_match": exact,
        "token_f1": token_f1,
        "official_multihop_rag_score": official_multihop_rag_score(
            prediction, question.answers
        ),
        "answer_normalization": normalization,
        "answer_quality_publishable": backend.publishable_answer_quality,
        "failure_class": failure_classification(
            question=question,
            receipt=receipt,
            context=context,
            answer_correct=bool(exact),
            answer_format_ok=answer_format_ok,
        ),
        "selector_name": context.selector_name,
        "selector_latency_ms": context.selector_latency_ms,
        "index_build_ms": context.index_build_ms,
        "bundle_id": context.bundle_id,
        "bundle_revision": context.bundle_revision,
        "retrieval_context_metrics": metrics,
        "serving_metrics": dict(serving),
        "resource_metrics": {
            key: serving.get(key)
            for key in (
                "active_detail_bytes",
                "retained_detail_bytes",
                "kv_bytes",
                "peak_memory_bytes",
                "temporary_allocation_bytes",
            )
        },
    }


def _bundle_unavailable_row(
    *,
    question: RAGQuestion,
    receipt: CandidateReceipt,
    condition: ContextCondition,
    candidate_count: int,
    token_budget: int,
    regime: str,
) -> dict[str, object]:
    return {
        "example_id": question.example_id,
        "question_type": question.question_type,
        "condition": condition.value,
        "selector_profile": "bundle_exact_qualified",
        "candidate_count": candidate_count,
        "token_budget": token_budget,
        "regime": regime,
        "status": "NO_QUALIFIED_ADAPTER",
        "candidate_receipt_id": receipt.receipt_id,
        "selection_receipt_id": None,
        "bundle_id": None,
        "bundle_revision": None,
        "failure_class": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fixture", "multihoprag"), default="multihoprag")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--candidate-counts", type=_parse_ints, default=(20, 50))
    parser.add_argument("--token-budgets", type=_parse_ints, default=(2048, 4096))
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--backend", choices=("probe", "mlx-native"), default="probe")
    parser.add_argument("--model", default="mlx-community/Qwen3-4B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--regimes",
        choices=("cold-warm", "persistent-corpus"),
        default="cold-warm",
    )
    parser.add_argument(
        "--native-cache-unit", choices=("selection", "chunk"), default="selection"
    )
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument("--reranker-device", default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=16)
    parser.add_argument("--skip-strong", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.native_cache_unit == "chunk" and args.regimes != "persistent-corpus":
        parser.error("chunk-native caching requires --regimes persistent-corpus")
    if args.native_cache_unit == "selection" and args.regimes == "persistent-corpus":
        parser.error("persistent-corpus regime requires --native-cache-unit chunk")

    if args.dataset == "fixture":
        documents, questions, dataset_metadata = controlled_fixture(seed=args.seed)
    else:
        documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(questions, max_examples=args.max_examples, seed=args.seed)
    documents_by_id = {document.document_id: document for document in documents}
    retriever = FirstStageBM25(documents)
    model_revision = (
        "fixture" if args.backend == "probe" else _resolve_hf_revision(args.model, args.revision)
    )
    reranker_revision = (
        "skipped"
        if args.skip_strong
        else _resolve_hf_revision(args.reranker, args.reranker_revision)
    )
    backend = (
        ProbeBackend()
        if args.backend == "probe"
        else PersistentMLXBackend(
            args.model,
            model_revision,
            args.max_new_tokens,
            native_cache_unit=args.native_cache_unit,
        )
    )
    strong_selector = (
        None
        if args.skip_strong
        else CrossEncoderRAGSelector(
            model_id=args.reranker,
            revision=reranker_revision,
            device=args.reranker_device,
            batch_size=args.reranker_batch_size,
            name_prefix="cross_encoder",
        )
    )
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    output = args.output
    (output / "candidate_receipts").mkdir(parents=True, exist_ok=True)
    (output / "selection_receipts").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    candidate_receipts: dict[str, CandidateReceipt] = {}
    selection_receipts: dict[str, SelectionReceipt] = {}
    regimes = (
        ("COLD", "WARM")
        if args.regimes == "cold-warm"
        else ("PERSISTENT_CORPUS",)
    )
    started_run = time.time()
    for question_index, question in enumerate(questions, 1):
        print(f"[{question_index}/{len(questions)}] {question.example_id}", flush=True)
        for candidate_count in args.candidate_counts:
            receipt = make_candidate_receipt(
                dataset=args.dataset,
                dataset_revision=dataset_metadata["dataset_revision"],
                corpus_revision=dataset_metadata["corpus_revision"],
                corpus_sha256=dataset_metadata["corpus_sha256"],
                question=question,
                retriever=retriever,
                candidate_count=candidate_count,
                chunker=chunker,
                ensure_gold=False,
                seed=args.seed,
            )
            candidate_receipts[receipt.receipt_id] = receipt
            prepared = prepare_candidate_context(
                receipt, documents_by_id, token_count=backend.token_count
            )
            selectors: list[tuple[str, object, tuple[ContextCondition, ...]]] = [
                (
                    "standard_bm25",
                    StandardRAGSelector(),
                    (ContextCondition.NO_PRA_STANDARD_RAG,),
                ),
                (
                    "pra_generic",
                    PRAHybridSelector(),
                    (
                        ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
                        ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
                    ),
                ),
            ]
            if not args.skip_strong:
                assert strong_selector is not None
                selectors.extend(
                    (
                        (
                            "strong_conventional_reranker",
                            strong_selector,
                            (ContextCondition.NO_PRA_STANDARD_RAG,),
                        ),
                        (
                            "pra_strong_reranker",
                            strong_selector,
                            (
                                ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
                                ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
                            ),
                        ),
                    )
                )
            ranking_cache: dict[int, tuple[object, float]] = {}
            for selector_profile, selector, conditions in selectors:
                cached_ranking = ranking_cache.get(id(selector))
                if cached_ranking is None:
                    rank_started = time.perf_counter()
                    ranking = selector.rank(question.question, prepared.chunks)
                    selector_ms = (time.perf_counter() - rank_started) * 1000.0
                    ranking_cache[id(selector)] = (ranking, selector_ms)
                else:
                    ranking, selector_ms = cached_ranking
                for token_budget in args.token_budgets:
                    base = packed_context_from_ranking(
                        condition=conditions[0],
                        selector_name=selector.name,
                        ranked=ranking,
                        prepared=prepared,
                        token_budget=token_budget,
                        selector_latency_ms=selector_ms,
                    )
                    selection = SelectionReceipt.from_context(
                        candidate_receipt_id=receipt.receipt_id,
                        example_id=question.example_id,
                        context=base,
                        selector_revision=selector.name,
                    )
                    selection_receipts[selection.receipt_id] = selection
                    for regime in regimes:
                        for condition in conditions:
                            context = replace(base, condition=condition)
                            rows.append(
                                _row(
                                    question=question,
                                    receipt=receipt,
                                    selection=selection,
                                    context=context,
                                    backend=backend,
                                    candidate_count=candidate_count,
                                    token_budget=token_budget,
                                    selector_profile=selector_profile,
                                    regime=regime,
                                )
                            )
                        if len(conditions) == 1:
                            # Conventional RAG has no native counterpart. Warm
                            # labels model/runtime warmth without claiming reuse.
                            continue
                    backend.release(selection.receipt_id)
            for token_budget in args.token_budgets:
                for regime in regimes:
                    for condition in (
                        ContextCondition.PRA_SELECTED_CONTEXT_BUNDLE,
                        ContextCondition.PRA_NATIVE_MEMORY_BUNDLE,
                    ):
                        rows.append(
                            _bundle_unavailable_row(
                                question=question,
                                receipt=receipt,
                                condition=condition,
                                candidate_count=candidate_count,
                                token_budget=token_budget,
                                regime=regime,
                            )
                        )

    for receipt in candidate_receipts.values():
        path = output / "candidate_receipts" / f"{receipt.receipt_id}.json"
        path.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    for receipt in selection_receipts.values():
        path = output / "selection_receipts" / f"{receipt.receipt_id}.json"
        path.write_text(json.dumps(receipt.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    write_results(output / "condition_results.jsonl.gz", rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "powered_rag_model_decomposition",
        "run_id": _digest(
            {
                "questions": [question.example_id for question in questions],
                "dataset": dataset_metadata,
                "model": args.model,
                "model_revision": model_revision,
                "reranker": args.reranker,
                "reranker_revision": reranker_revision,
                "seed": args.seed,
            }
        )[:20],
        "started_unix": started_run,
        "completed_unix": time.time(),
        "dataset": args.dataset,
        "dataset_metadata": dict(dataset_metadata),
        "question_ids": [question.example_id for question in questions],
        "candidate_counts": list(args.candidate_counts),
        "token_budgets": list(args.token_budgets),
        "chunker": asdict(chunker),
        "retriever": {"name": "bm25", "revision": retriever.revision},
        "reranker": {
            "model_id": args.reranker,
            "revision": reranker_revision,
            "status": "SKIPPED" if args.skip_strong else "MEASURED",
        },
        "backend": backend.name,
        "model": args.model,
        "model_revision": model_revision,
        "precision": "4bit" if "4bit" in args.model.lower() else "model_default",
        "engine": "probe" if args.backend == "probe" else "mlx",
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0,
            "prompt": "shortest_supported_answer_v2_no_think",
            "prompt_template": (
                "/no_think\nUse only the available evidence. Give only the shortest "
                "supported answer.\nQuestion: {question}\nAnswer:"
            ),
        },
        "seed": args.seed,
        "bundle_status": "NO_QUALIFIED_ADAPTER",
        "regimes": list(regimes),
        "native_cache_unit": args.native_cache_unit,
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "git_commit": _git_commit(),
        "candidate_receipt_count": len(candidate_receipts),
        "selection_receipt_count": len(selection_receipts),
        "condition_row_count": len(rows),
    }
    (output / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
