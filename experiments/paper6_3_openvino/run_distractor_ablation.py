"""Measure OpenVINO selected-text quality under controlled distractor dilution."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from experiments.engine_serving.run_openai_natural_e0 import (
    _aggregate,
    _bounded_text,
    _load_serving_helpers,
    _quality,
)
from experiments.paper6_3_openvino.run_natural_e0 import (
    _decoded_text,
    _history,
    _mean,
    _metric,
    _rss_bytes,
)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with",
}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS and len(token) > 1
    }


def document_blocks(text: str) -> list[str]:
    """Split the portable manifest's non-selected source into documents."""

    return [
        block.strip()
        for block in re.split(r"\n\s*\n(?=Document:)", text)
        if block.strip()
    ]


def ranked_distractors(entry: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Return deterministic relevant-first and irrelevant-first document orders."""

    query_terms = _terms(f"{entry['question']} {entry['answer']}")
    scored = []
    for ordinal, block in enumerate(document_blocks(str(entry["distractor_source"]))):
        block_terms = _terms(block)
        overlap = len(query_terms & block_terms)
        score = overlap / math.sqrt(max(len(block_terms), 1))
        scored.append((score, overlap, ordinal, block))
    relevant = [item[3] for item in sorted(scored, key=lambda item: (-item[0], -item[1], item[2]))]
    irrelevant = [item[3] for item in sorted(scored, key=lambda item: (item[1] > 0, item[0], item[2]))]
    return relevant, irrelevant


def contexts(
    entry: Mapping[str, Any],
    tokenizer: Any,
    *,
    selected_limit: int,
    full_limit: int,
    distractor_counts: tuple[int, ...],
) -> dict[str, tuple[str, int, int]]:
    """Build evidence-only and count-controlled distractor conditions."""

    selected, selected_tokens = _bounded_text(
        tokenizer, str(entry["selected_source"]), selected_limit
    )
    result = {"evidence_only": (selected, selected_tokens, 0)}
    relevant, irrelevant = ranked_distractors(entry)
    for mode, blocks in (("relevant", relevant), ("irrelevant", irrelevant)):
        for count in distractor_counts:
            combined = "\n\n".join((selected, *blocks[:count]))
            bounded, total_tokens = _bounded_text(tokenizer, combined, full_limit)
            result[f"{mode}_distractors_k{count}"] = (
                bounded,
                total_tokens,
                max(total_tokens - selected_tokens, 0),
            )
    return result


def _prompt(question: str, evidence: str) -> str:
    return (
        f"Evidence:\n{evidence}\n\nQuestion: {question}\n"
        "Answer briefly using only the evidence."
    )


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    import openvino_genai as genai
    from transformers import AutoTokenizer

    helpers = _load_serving_helpers()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.datasets)
    counters: Counter[str] = Counter()
    entries = []
    for entry in manifest["entries"]:
        dataset = str(entry["dataset"])
        if requested and dataset not in requested:
            continue
        if args.max_examples_per_dataset > 0 and counters[dataset] >= args.max_examples_per_dataset:
            continue
        counters[dataset] += 1
        entries.append(entry)

    scheduler = genai.SchedulerConfig()
    scheduler.enable_prefix_caching = True
    scheduler.dynamic_split_fuse = True
    scheduler.max_num_seqs = max(1, args.max_num_seqs)
    scheduler.max_num_batched_tokens = args.max_num_batched_tokens
    scheduler.cache_size = args.cache_size_gb
    rss_before_compile = _rss_bytes()
    compile_started = time.perf_counter()
    pipe = genai.LLMPipeline(args.model, args.device, scheduler_config=scheduler)
    compile_ms = (time.perf_counter() - compile_started) * 1000.0
    rss_after_compile = _rss_bytes()

    config = genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    distractor_counts = tuple(sorted(set(args.distractor_counts)))
    for entry in entries:
        selected_contexts = contexts(
            entry,
            tokenizer,
            selected_limit=args.max_selected_tokens,
            full_limit=args.max_full_tokens,
            distractor_counts=distractor_counts,
        )
        for condition, (evidence, source_tokens, distractor_tokens) in selected_contexts.items():
            prompt = _prompt(str(entry["question"]), evidence)
            for repeat in range(args.repeats):
                request_started = time.perf_counter()
                result = pipe.generate(
                    _history(genai, [{"role": "user", "content": prompt}]), config
                )
                wall_ms = (time.perf_counter() - request_started) * 1000.0
                output = _decoded_text(result)
                exact, f1, containment = _quality(output, str(entry["answer"]))
                perf = result.perf_metrics
                rows.append(
                    {
                        "dataset": entry["dataset"],
                        "seed": entry["seed"],
                        "example_id": entry["example_id"],
                        "selection_id": entry["selection_id"],
                        "condition": condition,
                        "repeat": repeat,
                        "cache_state": "COLD" if repeat == 0 else "WARM_REPEAT",
                        "answer": entry["answer"],
                        "source_tokens": source_tokens,
                        "selected_source_tokens": min(
                            source_tokens, args.max_selected_tokens
                        ),
                        "distractor_tokens": distractor_tokens,
                        "evidence_recall_at_4": entry["evidence_recall_at_4"],
                        "exact_match": exact,
                        "token_f1": f1,
                        "answer_containment": containment,
                        "gold_answer_log_probability": None,
                        "gold_answer_log_probability_status": "NOT_MEASURED",
                        "output_text": output,
                        "ttft_ms": _metric(perf, "get_ttft"),
                        "mean_itl_ms": _metric(perf, "get_tpot"),
                        "completion_latency_ms": wall_ms,
                        "prompt_tokens": _metric(perf, "get_num_input_tokens"),
                        "completion_tokens": _metric(perf, "get_num_generated_tokens"),
                        "cached_tokens": None,
                        "tokenization_ms": _metric(perf, "get_tokenization_duration"),
                        "detokenization_ms": _metric(perf, "get_detokenization_duration"),
                        "generation_duration_ms": _metric(perf, "get_generate_duration"),
                        "rss_bytes": _rss_bytes(),
                    }
                )
    elapsed = time.perf_counter() - started
    aggregates = []
    conditions = sorted({str(row["condition"]) for row in rows})
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        for condition in conditions:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            if not selected:
                continue
            base = dict(_aggregate(helpers, selected))
            base.update(
                {
                    "mean_distractor_tokens": _mean(selected, "distractor_tokens"),
                    "mean_tokenization_ms": _mean(selected, "tokenization_ms"),
                    "mean_detokenization_ms": _mean(selected, "detokenization_ms"),
                    "mean_rss_bytes": _mean(selected, "rss_bytes"),
                    "successful_requests_per_second": (
                        float(base["answer_containment"])
                        * len(selected)
                        / max(
                            sum(float(row["completion_latency_ms"]) for row in selected)
                            / 1000.0,
                            1e-9,
                        )
                    ),
                }
            )
            aggregates.append({"dataset": dataset, "condition": condition, **base})
    return {
        "schema_version": "1.0",
        "benchmark": "paper6_3_openvino_distractor_ablation_v1",
        "evidence_tier": "NATURAL_QA_DISTRACTOR_ABLATION",
        "measurement_status": "MEASURED",
        "integration_level": "E0_SELECTED_TEXT",
        "selector_frozen": True,
        "engine": "openvino-genai",
        "engine_version": importlib.metadata.version("openvino-genai"),
        "model_id": str(args.model),
        "device": args.device,
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "distractor_counts": list(distractor_counts),
        "conditions": conditions,
        "repeats": args.repeats,
        "max_selected_tokens": args.max_selected_tokens,
        "max_full_tokens": args.max_full_tokens,
        "max_new_tokens": args.max_new_tokens,
        "compile_ms": compile_ms,
        "rss_before_compile_bytes": rss_before_compile,
        "rss_after_compile_bytes": rss_after_compile,
        "elapsed_seconds": elapsed,
        "request_throughput_s": len(rows) / max(elapsed, 1e-9),
        "aggregates": aggregates,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "docs/papers/shared/results/portable_e0_qa_manifest_expanded.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=("qasper", "hotpotqa", "2wikimultihopqa"),
        default=(),
    )
    parser.add_argument("--max-examples-per-dataset", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--distractor-counts", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--max-selected-tokens", type=int, default=384)
    parser.add_argument("--max-full-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--cache-size-gb", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    payload = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aggregates": payload["aggregates"]}))


if __name__ == "__main__":
    main()
