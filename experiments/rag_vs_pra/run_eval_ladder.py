"""Run controlled and MultiHop-RAG matched-candidate evaluation grids."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import re
import statistics
import string
import time
from dataclasses import asdict
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
    FirstStageBM25,
    PRAHybridSelector,
    RankedChunk,
    RAGDocument,
    RAGQuestion,
    StandardRAGSelector,
    context_metrics,
    failure_classification,
    make_candidate_receipt,
    packed_context_from_ranking,
    prepare_candidate_context,
)


SCHEMA_VERSION = "1.0"


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalize(text: str) -> str:
    value = text.casefold()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _answer_metrics(
    prediction: str, answers: Sequence[str]
) -> tuple[float, float, float]:
    best_exact = 0.0
    best_f1 = 0.0
    task_score = 0.0
    predicted = _normalize(prediction)
    predicted_tokens = predicted.split()
    for answer in answers:
        gold = _normalize(answer)
        gold_tokens = gold.split()
        best_exact = max(best_exact, float(predicted == gold))
        task_score = max(
            task_score,
            float(bool(set(predicted_tokens).intersection(gold_tokens))),
        )
        if not predicted_tokens or not gold_tokens:
            score = float(predicted_tokens == gold_tokens)
        else:
            common = sum(
                min(predicted_tokens.count(token), gold_tokens.count(token))
                for token in set(predicted_tokens)
            )
            precision = common / len(predicted_tokens)
            recall = common / len(gold_tokens)
            score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, score)
    return best_exact, best_f1, task_score


class EvidenceProbeBackend:
    """Deterministic L0/L1 probe of whether selected context exposes the answer."""

    name = "evidence_availability_probe_v1"
    publishable_answer_quality = False

    def answer(
        self, question: RAGQuestion, context: str, condition: ContextCondition
    ) -> tuple[str, Mapping[str, object]]:
        started = time.perf_counter()
        prediction = next(
            (answer for answer in question.answers if answer.casefold() in context.casefold()),
            "unknown",
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return prediction, {
            "ttft_ms": elapsed_ms,
            "total_latency_ms": elapsed_ms,
            "generated_tokens": len(prediction.split()),
            "tokens_per_second": len(prediction.split()) / max(elapsed_ms / 1000.0, 1e-9),
        }


class HFTextBackend:
    """Visible-text model baseline; native PRA execution is a separate engine run."""

    publishable_answer_quality = True

    def __init__(self, model_id: str, revision: str, device: str, max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf_text:{model_id}@{revision}"
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        self.device = device

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def answer(
        self, question: RAGQuestion, context: str, condition: ContextCondition
    ) -> tuple[str, Mapping[str, object]]:
        import torch

        prompt = (
            "Use only the supplied evidence. Give only the shortest supported answer.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {question.question}\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        first_ms = None
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        generated = output[0, inputs.input_ids.shape[1] :]
        prediction = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return prediction, {
            "ttft_ms": first_ms,
            "total_latency_ms": elapsed_ms,
            "generated_tokens": int(generated.shape[0]),
            "tokens_per_second": int(generated.shape[0]) / max(elapsed_ms / 1000.0, 1e-9),
        }

    def warmup(self, question: RAGQuestion, contexts: Mapping[ContextCondition, str]) -> None:
        """Compile/load the visible decode path before timing benchmark rows."""

        original = self.max_new_tokens
        self.max_new_tokens = 1
        try:
            self.answer(question, contexts[ContextCondition.NO_PRA], ContextCondition.NO_PRA)
        finally:
            self.max_new_tokens = original


class HFNativeBackend:
    """Execute selected text visibly for RAG and as detached native K/V for PRA."""

    publishable_answer_quality = True

    def __init__(
        self,
        model_id: str,
        revision: str,
        device: str,
        max_new_tokens: int,
        consumption_layers: int,
        max_native_tokens: int,
    ) -> None:
        import torch

        from pra_hf import PRAConfig, PRAForCausalLM

        self.name = f"hf_native:{model_id}@{revision}"
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.device = device
        config = PRAConfig(
            routing_layer=-1,
            consumption_layers=tuple(range(-consumption_layers, 0)),
            address_layers=(-1,),
            detail_kv_layers=tuple(range(-consumption_layers, 0)),
            chunk_tokens=32,
            selected_fraction=1.0,
            max_materialized_tokens=max_native_tokens,
            materialization_profile="paper8_full_record_diagnostic",
            reference_device="gpu" if device == "cuda" else "cpu",
        )
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.pra = PRAForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            pra_config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.pra.model.to(device).eval()

    @property
    def tokenizer(self):
        return self.pra.tokenizer

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _query(self, question: RAGQuestion) -> str:
        return (
            "Use only the available evidence. Give only the shortest supported answer.\n"
            f"Question: {question.question}\nAnswer:"
        )

    def answer(
        self, question: RAGQuestion, context: str, condition: ContextCondition
    ) -> tuple[str, Mapping[str, object]]:
        import torch

        query = self._query(question)
        native = condition is ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR
        context = context.rstrip() + "\n\n"
        encode_ms = 0.0
        native_bytes = 0
        if native:
            from pra_hf.native_geometry import FrozenNativeAnchor, FrozenNativeSelection

            self.pra.enable()
            self.pra.clear_references()
            started = time.perf_counter()
            context_id = hashlib.sha256(context.encode("utf-8")).hexdigest()[:20]
            handle = self.pra.add_reference(f"memory://rag/{context_id}", text=context)
            if self.device == "cuda":
                torch.cuda.synchronize()
            encode_ms = (time.perf_counter() - started) * 1000.0
            entry = self.pra._handle.cache.get(handle.uri)
            if entry is None:
                raise RuntimeError("native RAG reference encoding did not create a cache entry")
            chunk = entry.layer_memory[self.pra.routing_layer].chunks[0]
            frozen = FrozenNativeSelection(
                (FrozenNativeAnchor(handle.uri, chunk.chunk_id, 0, handle.tokens),)
            )
            plan = self.pra.plan_native_materialization(frozen, full_selected_record=True)
            native_bytes = sum(
                hit.chunk.token_kv.k.numel() * hit.chunk.token_kv.k.element_size()
                + hit.chunk.token_kv.v.numel() * hit.chunk.token_kv.v.element_size()
                for selected in plan.selections_by_layer.values()
                for hit in selected
                if hit.chunk.token_kv is not None
            )
            prompt = query
        else:
            self.pra.disable()
            plan = None
            prompt = context + query
        if self.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            if native:
                generated = self.pra.generate_with_native_plan(
                    prompt,
                    plan,
                    max_new_tokens=self.max_new_tokens,
                    return_details=True,
                    do_sample=False,
                )
                prediction = generated.text.strip()
                generated_tokens = generated.generated_tokens
            else:
                encoded = self.tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=False
                ).to(self.device)
                output = self.pra.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                tokens = output[:, encoded.input_ids.shape[1] :]
                generated_tokens = int(tokens.shape[1])
                prediction = self.tokenizer.decode(
                    tokens[0], skip_special_tokens=True
                ).strip()
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return prediction, {
            "ttft_ms": None,
            "total_latency_ms": elapsed_ms,
            "generated_tokens": generated_tokens,
            "tokens_per_second": generated_tokens / max(elapsed_ms / 1000.0, 1e-9),
            "native_encode_ms": encode_ms,
            "active_detail_bytes": native_bytes,
            "visible_prompt_tokens": self.token_count(prompt),
            "selected_native_kv_tokens": self.token_count(context) if native else 0,
        }

    def warmup(self, question: RAGQuestion, contexts: Mapping[ContextCondition, str]) -> None:
        """Warm visible and detached-K/V paths symmetrically before measurement."""

        original = self.max_new_tokens
        self.max_new_tokens = 1
        try:
            self.answer(question, contexts[ContextCondition.NO_PRA], ContextCondition.NO_PRA)
            self.answer(
                question,
                contexts[ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR],
                ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
            )
        finally:
            self.max_new_tokens = original


class MLXNativeBackend:
    """Matched selected-text and native-K/V execution using MLX-LM caches."""

    publishable_answer_quality = True

    def __init__(self, model_id: str, revision: str, max_new_tokens: int) -> None:
        from mlx_lm import load

        self.name = f"mlx_native:{model_id}@{revision}"
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load(model_id, revision=revision)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

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
        arrivals = []
        generated = []
        for token, _ in generate_step(
            mx.array(query_tokens, dtype=mx.int32),
            self.model,
            max_tokens=self.max_new_tokens,
            prompt_cache=cache,
            sampler=make_sampler(temp=0),
        ):
            generated.append(int(token))
            arrivals.append((time.perf_counter() - started) * 1000.0)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return self.tokenizer.decode(generated).strip(), {
            "ttft_ms": arrivals[0] if arrivals else elapsed_ms,
            "itl_ms": (
                sum(right - left for left, right in zip(arrivals, arrivals[1:]))
                / (len(arrivals) - 1)
                if len(arrivals) > 1
                else 0.0
            ),
            "total_latency_ms": elapsed_ms,
            "generated_tokens": len(generated),
            "tokens_per_second": len(generated) / max(elapsed_ms / 1000.0, 1e-9),
        }

    def answer(
        self, question: RAGQuestion, context: str, condition: ContextCondition
    ) -> tuple[str, Mapping[str, object]]:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from pra_mlx.native import encode_native_memory, make_native_prompt_cache

        source_tokens = list(
            self.tokenizer.encode(context.rstrip() + "\n\n", add_special_tokens=False)
        )
        query_tokens = list(
            self.tokenizer.encode(self._query(question), add_special_tokens=False)
        )
        native = condition is ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR
        started = time.perf_counter()
        if native:
            memory = encode_native_memory(self.model, source_tokens)
            cache = make_native_prompt_cache(self.model, memory)
            native_bytes = memory.nbytes
        else:
            cache = make_prompt_cache(self.model)
            self.model(mx.array(source_tokens, dtype=mx.int32)[None], cache=cache)
            mx.eval([layer.state for layer in cache])
            native_bytes = 0
        ingestion_ms = (time.perf_counter() - started) * 1000.0
        prediction, serving = self._generate(query_tokens, cache)
        return prediction, {
            **serving,
            "native_encode_ms": ingestion_ms if native else 0.0,
            "visible_text_ingestion_ms": ingestion_ms if not native else 0.0,
            "active_detail_bytes": native_bytes,
            "visible_prompt_tokens": len(query_tokens) if native else len(source_tokens) + len(query_tokens),
            "selected_native_kv_tokens": len(source_tokens) if native else 0,
        }

    def warmup(self, question: RAGQuestion, contexts: Mapping[ContextCondition, str]) -> None:
        original = self.max_new_tokens
        self.max_new_tokens = 1
        try:
            self.answer(question, contexts[ContextCondition.NO_PRA], ContextCondition.NO_PRA)
            self.answer(
                question,
                contexts[ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR],
                ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
            )
        finally:
            self.max_new_tokens = original


def _condition_row(
    *,
    question: RAGQuestion,
    receipt: CandidateReceipt,
    context,
    backend,
    candidate_count: int,
    token_budget: int,
) -> dict[str, object]:
    prediction, serving = backend.answer(question, context.text, context.condition)
    exact, f1, task_score = _answer_metrics(prediction, question.answers)
    metrics = context_metrics(question, receipt, context)
    failure = failure_classification(
        question=question,
        receipt=receipt,
        context=context,
        answer_correct=bool(exact),
    )
    return {
        "example_id": question.example_id,
        "question_type": question.question_type,
        "condition": context.condition.value,
        "candidate_count": candidate_count,
        "token_budget": token_budget,
        "receipt_id": receipt.receipt_id,
        "candidate_document_ids": list(receipt.candidate_document_ids),
        "gold_document_ids": sorted(question.gold_document_ids),
        "selected_document_ids": list(context.selected_document_ids),
        "selected_chunks": [
            {
                "chunk_id": row.chunk.chunk_id,
                "document_id": row.chunk.document_id,
                "start": row.chunk.start,
                "end": row.chunk.end,
                "token_count": row.chunk.token_count,
                "rank": row.rank,
                "score": row.score,
                "channel_ranks": dict(row.channel_ranks),
            }
            for row in context.chunks
        ],
        "answer": prediction,
        "gold_answers": list(question.answers),
        "exact_match": exact,
        "token_f1": f1,
        "dataset_task_score": task_score,
        "answer_quality_publishable": backend.publishable_answer_quality,
        "failure_class": failure,
        "selector": context.selector_name,
        "selector_latency_ms": context.selector_latency_ms,
        "index_build_ms": context.index_build_ms,
        "bundle_id": context.bundle_id,
        "bundle_revision": context.bundle_revision,
        "retrieval_context_metrics": metrics,
        "serving_metrics": dict(serving),
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (row["condition"], row["candidate_count"], row["token_budget"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (condition, candidate_count, token_budget), values in sorted(groups.items()):
        context_rows = [value["retrieval_context_metrics"] for value in values]
        serving_rows = [value["serving_metrics"] for value in values]
        summaries.append(
            {
                "condition": condition,
                "candidate_count": candidate_count,
                "token_budget": token_budget,
                "examples": len(values),
                "exact_match": statistics.fmean(float(value["exact_match"]) for value in values),
                "token_f1": statistics.fmean(float(value["token_f1"]) for value in values),
                "dataset_task_score": statistics.fmean(
                    float(value["dataset_task_score"]) for value in values
                ),
                "supporting_document_coverage": statistics.fmean(
                    float(value["supporting_document_coverage"]) for value in context_rows
                ),
                "supporting_span_coverage": statistics.fmean(
                    float(value["supporting_span_coverage"]) for value in context_rows
                ),
                "false_selected_document_fraction": statistics.fmean(
                    float(value["false_selected_document_fraction"]) for value in context_rows
                ),
                "logical_candidate_tokens": statistics.fmean(
                    float(value["logical_candidate_tokens"]) for value in context_rows
                ),
                "physical_context_tokens": statistics.fmean(
                    float(value["physical_context_tokens"]) for value in context_rows
                ),
                "materialization_avoidance": statistics.fmean(
                    float(value["materialization_avoidance"]) for value in context_rows
                ),
                "total_latency_ms": statistics.fmean(
                    float(value["total_latency_ms"]) for value in serving_rows
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("fixture", "multihoprag"), required=True)
    parser.add_argument("--stage", choices=("fixed", "retrieval"), default="fixed")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--candidate-counts", type=_parse_ints, default=(5, 10, 20, 50))
    parser.add_argument("--token-budgets", type=_parse_ints, default=(2048, 4096, 8192, 16384))
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--backend",
        choices=("probe", "hf-text", "hf-native", "mlx-native"),
        default="probe",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--consumption-layers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "fixture":
        documents, questions, dataset_metadata = controlled_fixture(seed=args.seed)
    else:
        documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    questions = select_cohort(questions, max_examples=args.max_examples, seed=args.seed)
    documents_by_id = {document.document_id: document for document in documents}
    retriever = FirstStageBM25(documents)
    if args.backend == "probe":
        backend = EvidenceProbeBackend()
    elif args.backend == "hf-text":
        backend = HFTextBackend(args.model, args.revision, args.device, args.max_new_tokens)
    elif args.backend == "hf-native":
        backend = HFNativeBackend(
            args.model,
            args.revision,
            args.device,
            args.max_new_tokens,
            args.consumption_layers,
            max(args.token_budgets),
        )
    else:
        backend = MLXNativeBackend(args.model, args.revision, args.max_new_tokens)
    token_count = getattr(backend, "token_count", None)
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    rows = []
    receipts: dict[str, dict[str, object]] = {}
    selection_receipts: dict[str, dict[str, object]] = {}
    warmed = False
    for question in questions:
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
                ensure_gold=args.stage == "fixed",
                seed=args.seed,
            )
            receipts[receipt.receipt_id] = receipt.to_dict()
            kwargs = {"token_count": token_count} if token_count is not None else {}
            prepared = prepare_candidate_context(receipt, documents_by_id, **kwargs)
            baseline_selector = StandardRAGSelector()
            pra_selector = PRAHybridSelector()
            started = time.perf_counter()
            baseline_ranking = baseline_selector.rank(question.question, prepared.chunks)
            baseline_latency_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            pra_ranking = pra_selector.rank(question.question, prepared.chunks)
            pra_latency_ms = (time.perf_counter() - started) * 1000.0
            oracle_ranking = tuple(
                RankedChunk(chunk, 1.0, rank, {"oracle": rank})
                for rank, chunk in enumerate(
                    (
                        chunk
                        for chunk in prepared.chunks
                        if chunk.document_id in question.gold_document_ids
                    ),
                    1,
                )
            )
            for token_budget in args.token_budgets:
                baseline = packed_context_from_ranking(
                    condition=ContextCondition.NO_PRA,
                    selector_name=baseline_selector.name,
                    ranked=baseline_ranking,
                    prepared=prepared,
                    token_budget=token_budget,
                    selector_latency_ms=baseline_latency_ms,
                )
                selected_context = packed_context_from_ranking(
                    condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
                    selector_name=pra_selector.name,
                    ranked=pra_ranking,
                    prepared=prepared,
                    token_budget=token_budget,
                    selector_latency_ms=pra_latency_ms,
                )
                native_memory = packed_context_from_ranking(
                    condition=ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR,
                    selector_name=pra_selector.name,
                    ranked=pra_ranking,
                    prepared=prepared,
                    token_budget=token_budget,
                    selector_latency_ms=pra_latency_ms,
                )
                oracle = packed_context_from_ranking(
                    condition=ContextCondition.ORACLE_GOLD_DOCUMENTS,
                    selector_name="oracle_gold_documents",
                    ranked=oracle_ranking,
                    prepared=prepared,
                    token_budget=token_budget,
                    selector_latency_ms=0.0,
                )
                if not warmed and hasattr(backend, "warmup"):
                    backend.warmup(
                        question,
                        {
                            ContextCondition.NO_PRA: baseline.text,
                            ContextCondition.PRA_NATIVE_MEMORY_NO_ADAPTOR: native_memory.text,
                        },
                    )
                    warmed = True
                for context in (baseline, selected_context, native_memory, oracle):
                    row = _condition_row(
                            question=question,
                            receipt=receipt,
                            context=context,
                            backend=backend,
                            candidate_count=candidate_count,
                            token_budget=token_budget,
                        )
                    selection = {
                        "receipt_id": row["receipt_id"],
                        "condition": row["condition"],
                        "token_budget": row["token_budget"],
                        "selected_document_ids": row.pop("selected_document_ids"),
                        "selected_chunks": row.pop("selected_chunks"),
                    }
                    selection_id = _digest(selection)
                    selection_receipts[selection_id] = {
                        "selection_id": selection_id,
                        **selection,
                    }
                    row["selection_id"] = selection_id
                    row.pop("candidate_document_ids")
                    row.pop("gold_document_ids")
                    rows.append(row)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "rag_vs_pra_matched_candidate_ladder",
        "evidence_tier": (
            "CONTROLLED_NON_PUBLISHABLE" if args.dataset == "fixture" else "NATURAL_FIXED_CANDIDATE"
        ),
        "dataset": args.dataset,
        "stage": args.stage,
        "dataset_metadata": dict(dataset_metadata),
        "retriever": {
            "name": "bm25",
            "revision": retriever.revision,
            "index_sha256": retriever.index_sha256,
        },
        "chunker": asdict(chunker),
        "candidate_counts": list(args.candidate_counts),
        "token_budgets": list(args.token_budgets),
        "seed": args.seed,
        "backend": backend.name,
        "model": args.model if args.backend != "probe" else None,
        "model_revision": args.revision if args.backend != "probe" else None,
        "hardware": platform.platform(),
        "conditions": {
            "NO_PRA_STANDARD_RAG": "standard RAG: global BM25 chunk packing into visible context",
            "PRA_SELECTED_CONTEXT_NO_ADAPTOR": "generic PRA selection rendered as visible context",
            "PRA_NATIVE_MEMORY_NO_ADAPTOR": "the same generic PRA selection realized as native K/V",
            "PRA_SELECTED_CONTEXT_BUNDLE": {
                "state": "NO_QUALIFIED_ADAPTER",
                "note": "No immutable document-RAG routing adaptor is qualified for this cohort.",
            },
            "PRA_NATIVE_MEMORY_BUNDLE": {
                "state": "NO_QUALIFIED_ADAPTER",
                "note": "No immutable document-RAG native-memory adaptor is qualified for this cohort.",
            },
            "oracle_gold_documents": "research-only diagnostic; excluded from headline deltas",
        },
        "receipts": list(receipts.values()),
        "examples": [
            {
                "example_id": question.example_id,
                "question": question.question,
                "gold_answers": list(question.answers),
                "gold_document_ids": sorted(question.gold_document_ids),
                "gold_spans": {
                    document_id: [list(span) for span in spans]
                    for document_id, spans in question.gold_spans.items()
                },
                "question_type": question.question_type,
            }
            for question in questions
        ],
        "selection_receipts": list(selection_receipts.values()),
        "rows": rows,
        "summary": _summary(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2) + "\n"
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wt", encoding="utf-8") as stream:
            stream.write(serialized)
    else:
        args.output.write_text(serialized, encoding="utf-8")
    print(f"wrote {len(rows)} rows and {len(receipts)} frozen receipts to {args.output}")


if __name__ == "__main__":
    main()
