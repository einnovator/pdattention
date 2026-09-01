"""Evaluate MLX native PRA on natural QA, quantization, and session pressure."""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


SEEDS = (11, 23, 37, 53, 71)


@dataclass(frozen=True)
class QADocument:
    """One independently routable document or paper paragraph."""

    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class QAExample:
    dataset: str
    example_id: str
    question: str
    answer: str
    source: str
    source_scope: str
    documents: tuple[QADocument, ...] = ()
    evidence_document_ids: frozenset[str] = frozenset()


def _normalize(text: str) -> str:
    value = text.lower()
    value = "".join(char for char in value if char not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _metrics(prediction: str, answer: str) -> tuple[float, float]:
    predicted = _normalize(prediction)
    gold = _normalize(answer)
    exact = float(predicted == gold)
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    common = sum(
        min(predicted_tokens.count(token), gold_tokens.count(token))
        for token in set(predicted_tokens)
    )
    if not predicted_tokens or not gold_tokens:
        return exact, float(predicted_tokens == gold_tokens)
    precision = common / len(predicted_tokens)
    recall = common / len(gold_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return exact, f1


def _ordered_context(row: Mapping[str, object]) -> str:
    supporting = {
        str(title)
        for title, _ in zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
        )
    }
    documents = [
        (str(title), " ".join(map(str, sentences)))
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
    ]
    documents.sort(key=lambda item: (item[0] not in supporting, item[0]))
    return "\n\n".join(f"Document: {title}\n{text}" for title, text in documents)


def _multihop_documents(row: Mapping[str, object]) -> tuple[QADocument, ...]:
    """Preserve the dataset's candidate order for routing experiments."""

    return tuple(
        QADocument(str(index), str(title), " ".join(map(str, sentences)))
        for index, (title, sentences) in enumerate(
            zip(row["context"]["title"], row["context"]["sentences"])
        )
    )


def _hotpot_examples(cache_dir: Path) -> list[QAExample]:
    from datasets import load_dataset

    rows = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        cache_dir=str(cache_dir),
    )
    examples = []
    for row in rows:
        if not str(row.get("answer", "")).strip():
            continue
        documents = _multihop_documents(row)
        supporting = set(map(str, row["supporting_facts"]["title"]))
        examples.append(
            QAExample(
                "hotpotqa",
                str(row["id"]),
                str(row["question"]),
                str(row["answer"]),
                _ordered_context(row),
                "supporting_documents_first_plus_distractors",
                documents,
                frozenset(
                    document.document_id
                    for document in documents
                    if document.title in supporting
                ),
            )
        )
    return examples


def _qasper_answer(answer: Mapping[str, object]) -> str | None:
    if answer.get("unanswerable"):
        return None
    if answer.get("yes_no") is not None:
        return "yes" if answer["yes_no"] else "no"
    spans = [str(value).strip() for value in answer.get("extractive_spans", ())]
    spans = [value for value in spans if value]
    if spans:
        return "; ".join(spans)
    free = str(answer.get("free_form_answer") or "").strip()
    return free or None


def _qasper_examples(cache_dir: Path) -> list[QAExample]:
    from data.native_kv_benchmarks import load_qasper_papers

    examples = []
    for paper_id, paper in load_qasper_papers(
        "validation", cache_dir=cache_dir
    ).items():
        abstract = str(paper.get("abstract", ""))
        documents = []
        if abstract:
            documents.append(QADocument("abstract", "Abstract", abstract))
        for section_index, section in enumerate(paper.get("full_text", [])):
            section_name = str(section.get("section_name", f"Section {section_index + 1}"))
            for paragraph_index, paragraph in enumerate(section.get("paragraphs", [])):
                text = str(paragraph).strip()
                if text:
                    documents.append(
                        QADocument(
                            f"section-{section_index}-paragraph-{paragraph_index}",
                            section_name,
                            text,
                        )
                    )
        for qa in paper.get("qas", []):
            candidates = []
            evidence = []
            for annotation in qa.get("answers", []):
                answer = annotation.get("answer", {})
                value = _qasper_answer(answer)
                if value:
                    candidates.append(value)
                    evidence.extend(
                        str(item).strip()
                        for item in answer.get("evidence", ())
                        if str(item).strip()
                    )
            if not candidates or not evidence:
                continue
            answer = min(candidates, key=lambda value: (len(value.split()), value))
            if len(answer.split()) > 12:
                continue
            source = "Evidence:\n" + "\n".join(dict.fromkeys(evidence))
            if abstract:
                source += "\n\nPaper abstract:\n" + abstract
            normalized_evidence = tuple(_normalize(value) for value in evidence)
            evidence_document_ids = frozenset(
                document.document_id
                for document in documents
                if any(
                    value
                    and (
                        value in _normalize(document.text)
                        or _normalize(document.text) in value
                    )
                    for value in normalized_evidence
                )
            )
            if not evidence_document_ids:
                continue
            examples.append(
                QAExample(
                    "qasper",
                    f"{paper_id}-{qa.get('question_id', len(examples))}",
                    str(qa.get("question", "")),
                    answer,
                    source,
                    "annotated_evidence_first_plus_abstract",
                    tuple(documents),
                    evidence_document_ids,
                )
            )
    return examples


def _twowiki_examples(cache_dir: Path) -> list[QAExample]:
    from datasets import load_dataset

    rows = load_dataset(
        "parquet",
        data_files={
            "dev": (
                "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/"
                "resolve/main/dev.parquet"
            )
        },
        split="dev",
        cache_dir=str(cache_dir),
    )
    examples = []
    for row in rows:
        context = row["context"]
        if isinstance(context, str):
            context = json.loads(context)
        if isinstance(context, Mapping):
            titles, sentences = context["title"], context["sentences"]
        else:
            titles = [item[0] for item in context]
            sentences = [item[1] for item in context]
        facts = row["supporting_facts"]
        if isinstance(facts, str):
            facts = json.loads(facts)
        supporting = (
            set(map(str, facts["title"]))
            if isinstance(facts, Mapping)
            else {str(item[0]) for item in facts}
        )
        routable_documents = tuple(
            QADocument(str(index), str(title), " ".join(map(str, values)))
            for index, (title, values) in enumerate(zip(titles, sentences))
        )
        documents = [(document.title, document.text) for document in routable_documents]
        documents.sort(key=lambda item: (item[0] not in supporting, item[0]))
        source = "\n\n".join(
            f"Document: {title}\n{text}" for title, text in documents
        )
        examples.append(
            QAExample(
                "2wikimultihopqa",
                str(row.get("_id", row.get("id", len(examples)))),
                str(row["question"]),
                str(row["answer"]),
                source,
                "supporting_documents_first_plus_distractors",
                routable_documents,
                frozenset(
                    document.document_id
                    for document in routable_documents
                    if document.title in supporting
                ),
            )
        )
    return examples


def _examples(dataset: str, cache_dir: Path) -> list[QAExample]:
    if dataset == "qasper":
        return _qasper_examples(cache_dir)
    if dataset == "hotpotqa":
        return _hotpot_examples(cache_dir)
    return _twowiki_examples(cache_dir)


def _generate(model, tokenizer, query, cache, max_tokens: int):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    generated = []
    text = ""
    for token, _ in generate_step(
        mx.array(query, dtype=mx.int32),
        model,
        max_tokens=max_tokens,
        prompt_cache=cache,
        sampler=make_sampler(temp=0),
    ):
        generated.append(int(token))
        text = tokenizer.decode(generated)
    return text.strip(), (time.perf_counter() - started) * 1000.0


def _answer_logprob(model, query, answer, cache) -> float:
    import mlx.core as mx

    input_ids = list(query) + list(answer)
    logits = model(mx.array(input_ids, dtype=mx.int32)[None], cache=cache)
    start = len(query) - 1
    total = mx.array(0.0, dtype=mx.float32)
    for offset, token_id in enumerate(answer):
        row = logits[0, start + offset].astype(mx.float32)
        total = total + row[int(token_id)] - mx.logsumexp(row)
    mx.eval(total)
    return float(total.item())


def _bounded_source(tokenizer, source: str, limit: int) -> list[int]:
    tokens = list(tokenizer.encode(source, add_special_tokens=False))
    return tokens[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), required=True
    )
    parser.add_argument("--examples-per-seed", type=int, default=4)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx.native import (
        dequantize_native_memory,
        encode_native_memory,
        make_native_prompt_cache,
        quantize_native_memory,
    )

    model, tokenizer = load(args.model, revision=args.revision)
    candidates = _examples(args.dataset, args.cache_dir)
    rows = []
    pressure_rows = []
    session_rows = []
    for seed in SEEDS:
        cohort = list(candidates)
        random.Random(seed).shuffle(cohort)
        cohort = cohort[: args.examples_per_seed]
        prepared = []
        for example in cohort:
            source = _bounded_source(tokenizer, example.source, args.max_source_tokens)
            query_text = (
                "Answer the question using the available evidence. Give only the "
                f"short answer.\nQuestion: {example.question}\nAnswer:"
            )
            query = list(tokenizer.encode(query_text, add_special_tokens=False))
            answer = list(
                tokenizer.encode(" " + example.answer, add_special_tokens=False)
            )
            memory = encode_native_memory(model, source)
            quantized = quantize_native_memory(memory)
            prepared.append((example, source, query, answer, memory, quantized))

        for index, item in enumerate(prepared):
            example, source, query, answer, memory, quantized = item
            shuffled = prepared[(index + 1) % len(prepared)][4]
            restored_started = time.perf_counter()
            restored = dequantize_native_memory(quantized)
            dequantize_ms = (time.perf_counter() - restored_started) * 1000.0

            def cache_for(condition: str):
                if condition == "ordinary_split":
                    cache = make_prompt_cache(model)
                    # MLX is lazy. Synchronize the reusable source cache before
                    # request timing so ordinary E0 and pre-encoded native E2
                    # are compared at the same warm lifecycle state.
                    encoded = model(mx.array(source, dtype=mx.int32)[None], cache=cache)
                    mx.eval(encoded)
                    return cache
                if condition == "native_fp":
                    return make_native_prompt_cache(model, memory)
                if condition == "native_int8_resident":
                    return make_native_prompt_cache(model, restored)
                if condition == "native_shuffled":
                    return make_native_prompt_cache(model, shuffled)
                return make_prompt_cache(model)

            for condition in (
                "ordinary_split",
                "native_fp",
                "native_int8_resident",
                "native_shuffled",
                "no_memory",
            ):
                active_memory = (
                    shuffled if condition == "native_shuffled" else memory
                )
                logprob = _answer_logprob(model, query, answer, cache_for(condition))
                output, latency_ms = _generate(
                    model,
                    tokenizer,
                    query,
                    cache_for(condition),
                    args.max_new_tokens,
                )
                exact, f1 = _metrics(output, example.answer)
                resident_bytes = (
                    quantized.nbytes
                    if condition == "native_int8_resident"
                    else active_memory.nbytes
                    if condition in {"native_fp", "native_shuffled"}
                    else 0
                )
                rows.append(
                    {
                        "dataset": example.dataset,
                        "seed": seed,
                        "example_id": example.example_id,
                        "condition": condition,
                        "source_scope": example.source_scope,
                        "source_tokens": len(source),
                        "query_tokens": len(query),
                        "gold_answer": example.answer,
                        "output": output,
                        "exact_match": exact,
                        "token_f1": f1,
                        "gold_answer_logprob": logprob,
                        "completion_latency_ms": latency_ms,
                        "resident_selected_kv_bytes": resident_bytes,
                        "active_materialized_kv_bytes": (
                            active_memory.nbytes
                            if condition.startswith("native_")
                            else 0
                        ),
                        "dequantize_ms": (
                            dequantize_ms
                            if condition == "native_int8_resident"
                            else 0.0
                        ),
                        "storage_compression_ratio": (
                            memory.nbytes / quantized.nbytes
                            if condition == "native_int8_resident"
                            else 1.0
                        ),
                    }
                )

        native_bytes = [item[4].nbytes for item in prepared]
        quantized_bytes = [item[5].nbytes for item in prepared]
        for count in (1, 2, len(prepared)):
            count = min(count, len(prepared))
            pressure_rows.append(
                {
                    "dataset": args.dataset,
                    "seed": seed,
                    "resident_resources": count,
                    "full_precision_resident_bytes": sum(native_bytes[:count]),
                    "int8_resident_bytes": sum(quantized_bytes[:count]),
                    "bytes_saved": sum(native_bytes[:count]) - sum(quantized_bytes[:count]),
                    "mlx_active_memory_bytes": int(
                        getattr(mx, "get_active_memory", lambda: 0)()
                    ),
                    "mlx_peak_memory_bytes": int(
                        getattr(mx, "get_peak_memory", lambda: 0)()
                    ),
                }
            )

        if prepared:
            example, _, query, _, memory, _ = prepared[0]
            cache = make_native_prompt_cache(model, memory)
            for turn in range(1, 9):
                output, latency_ms = _generate(
                    model, tokenizer, query, cache, args.max_new_tokens
                )
                exact, f1 = _metrics(output, example.answer)
                session_rows.append(
                    {
                        "dataset": args.dataset,
                        "seed": seed,
                        "turn": turn,
                        "example_id": example.example_id,
                        "exact_match": exact,
                        "token_f1": f1,
                        "latency_ms": latency_ms,
                        "local_cache_offset": int(cache[0].local_offset),
                        "selected_memory_tokens": memory.source_tokens,
                        "selected_memory_bytes": memory.nbytes,
                    }
                )

    payload = {
        "schema_version": "1.0",
        "experiment": "paper6_2_mlx_natural_answer_quality_pressure_v1",
        "evidence_tier": "NATURAL_QA_ORACLE_EVIDENCE_MATERIALIZATION",
        "engine": "mlx-lm",
        "engine_version": getattr(mlx_lm, "__version__", "unknown"),
        "model_id": args.model,
        "model_revision": args.revision,
        "dataset": args.dataset,
        "seeds": list(SEEDS),
        "examples_per_seed": args.examples_per_seed,
        "max_source_tokens": args.max_source_tokens,
        "quantization": (
            "symmetric int8 per-head residency; dequantized to model-native K/V "
            "before attention"
        ),
        "rows": rows,
        "pressure_rows": pressure_rows,
        "session_rows": session_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
