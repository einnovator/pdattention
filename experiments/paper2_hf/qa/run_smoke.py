"""Run one HotpotQA and one QASPER zero-shot PRA smoke example."""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.native_kv_benchmarks import load_qasper_papers
from experiments.paper2_hf.common.artifacts import runtime_metadata, write_artifacts
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_torch.hf import PRAHFConfig, inject_pra


def normalize(text: str) -> list[str]:
    """Apply the conventional lowercase alphanumeric QA normalization."""
    return re.findall(r"[a-z0-9]+", text.lower())


def answer_metrics(prediction: str, answer: str) -> dict[str, float]:
    """Return token F1 and normalized exact match for one free-form answer."""
    predicted = normalize(prediction)
    expected = normalize(answer)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / max(len(predicted), 1)
    recall = overlap / max(len(expected), 1)
    return {
        "em": float(predicted == expected),
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "answer_contained": float(bool(expected) and " ".join(expected) in " ".join(predicted)),
    }


def evidence_token_spans(tokenizer, source: str, evidence: list[str]) -> list[tuple[int, int]]:
    """Map exact evidence character spans through tokenizer offset metadata."""
    encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.offset_mapping
    spans = []
    for text in evidence:
        char_start = source.find(text)
        if char_start < 0:
            continue
        char_end = char_start + len(text)
        overlapping = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
        if overlapping:
            spans.append((overlapping[0], overlapping[-1] + 1))
    return spans


def hotpot_example(cache_dir: Path) -> dict:
    """Load a deterministic unrestricted HotpotQA validation example."""
    row = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        cache_dir=str(cache_dir),
    )[0]
    supporting = {
        (str(title), int(sentence_id))
        for title, sentence_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
    }
    source_segments = []
    evidence = []
    for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        for sentence_id, sentence in enumerate(sentences):
            segment = f"{title}: {str(sentence).strip()}"
            source_segments.append(segment)
            if (str(title), sentence_id) in supporting:
                evidence.append(segment)
    return {
        "dataset": "hotpotqa",
        "id": str(row["id"]),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
        "source": "\n".join(source_segments),
        "evidence": evidence,
    }


def qasper_example(cache_dir: Path) -> dict:
    """Choose the first answerable yes/no QASPER item with textual evidence."""
    papers = load_qasper_papers("validation", cache_dir=cache_dir)
    for paper_id, paper in papers.items():
        paragraphs = [str(paper.get("abstract", ""))]
        for section in paper.get("full_text", []):
            paragraphs.extend(str(value) for value in section.get("paragraphs", []))
        for qa in paper.get("qas", []):
            for annotation in qa.get("answers", []):
                answer = annotation.get("answer", {})
                evidence = [str(value) for value in answer.get("evidence", []) if str(value).strip()]
                if answer.get("yes_no") is None or not evidence:
                    continue
                # Preserve the paper context while ensuring the annotated span is present.
                source = "\n".join(dict.fromkeys([*evidence, *paragraphs]))
                return {
                    "dataset": "qasper",
                    "id": f"{paper_id}:{qa.get('question_id', '')}",
                    "question": str(qa["question"]),
                    "answer": "yes" if answer["yes_no"] else "no",
                    "source": source,
                    "evidence": evidence,
                }
    raise RuntimeError("No answerable QASPER yes/no example with evidence was found.")


def prompt_ids(tokenizer, question: str, *, context: str | None = None, max_tokens: int = 256):
    """Create one Qwen chat prompt, left-truncated so the answer request survives."""
    content = "Answer briefly and directly."
    if context:
        content += f"\nContext:\n{context}"
    content += f"\nQuestion: {question}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_tokens)
    tokenizer.truncation_side = previous
    return encoded


@torch.no_grad()
def generate_answer(model, tokenizer, encoded, device, new_tokens: int) -> tuple[str, float]:
    """Generate only the continuation and return synchronized wall time."""
    encoded = encoded.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(**encoded, max_new_tokens=new_tokens, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    continuation = output[0, encoded.input_ids.shape[1] :]
    return tokenizer.decode(continuation, skip_special_tokens=True).strip(), time.perf_counter() - started


def run_condition(handle, tokenizer, example, condition: str, device, new_tokens: int) -> dict:
    """Evaluate truncation, dense text, oracle text-RAG, or routed native-KV."""
    handle.cache.clear()
    handle.set_memory_enabled(False)
    context = None
    if condition == "dense":
        context = example["source"]
    elif condition == "text_rag_oracle":
        context = "\n".join(example["evidence"])
    elif condition == "pra":
        source_ids = tokenizer(example["source"], return_tensors="pt", add_special_tokens=False).input_ids
        handle.add_reference(f"benchmark://{example['dataset']}/{example['id']}", source_ids, text=example["source"])
        handle.set_memory_enabled(True)
    encoded = prompt_ids(tokenizer, example["question"], context=context)
    prediction, duration = generate_answer(handle.model, tokenizer, encoded, device, new_tokens)
    result = {
        "condition": condition,
        "prompt_tokens": int(encoded.input_ids.shape[1]),
        "prediction": prediction,
        "duration_seconds": duration,
        **answer_metrics(prediction, example["answer"]),
    }
    if condition == "pra":
        adapter = next(iter(handle.adapters.values()))
        selected = [hit for row in adapter.last_selected_chunks for hit in row]
        source_ids = tokenizer(example["source"], add_special_tokens=False).input_ids
        evidence_spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
        selected_spans = [(hit.logical_start, hit.logical_end) for hit in selected]
        covered = sum(
            any(max(start, selected_start) < min(end, selected_end) for selected_start, selected_end in selected_spans)
            for start, end in evidence_spans
        )
        result.update(
            {
                "source_tokens": len(source_ids),
                "selected_spans": selected_spans,
                "evidence_spans": evidence_spans,
                "routing_recall": covered / max(len(evidence_spans), 1),
                "diagnostics": handle.diagnostics_by_layer(),
            }
        )
    return result


def run(args) -> dict:
    """Load one checkpoint and execute the two-dataset, four-condition smoke matrix."""
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(-1,),
            model_max_context_tokens=256,
            max_prompt_direct_tokens=160,
            encoding_block_tokens=64,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=96,
            top_k_references=1,
            top_k_chunks_per_reference=3,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
        ),
    )
    examples = [hotpot_example(args.cache_dir), qasper_example(args.cache_dir / "qasper")]
    rows = []
    for example in examples:
        conditions = [
            run_condition(handle, tokenizer, example, condition, device, args.new_tokens)
            for condition in ("truncation", "dense", "text_rag_oracle", "pra")
        ]
        rows.append(
            {
                "dataset": example["dataset"],
                "id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "conditions": conditions,
            }
        )
    return {
        "runtime": runtime_metadata(),
        "protocol": "unrestricted pretrained QA smoke; oracle evidence text-RAG is an upper control",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "examples": rows,
        "max_native_operation_tokens": handle.max_native_operation_tokens,
        "native_limit_violations": handle.native_limit_violations,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "qa",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    artifact = run(arguments)
    json_path, csv_path = write_artifacts(artifact, arguments.output_dir, "qwen3_0_6b_qa_smoke")
    print(json_path)
    print(csv_path)
