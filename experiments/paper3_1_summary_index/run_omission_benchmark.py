"""Measure whether generated routing summaries retain low-salience facts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper3_1_summary_index.ollama_sidecar import (
    JSONLGenerationCache,
    OllamaClient,
    PROMPT_VERSION,
    generate_cached,
)
from experiments.paper3_1_summary_index.run_study import MODEL_PROFILES
from pra_hf.summary_index import (
    BM25SummaryScorer,
    SummaryIndex,
    SummaryIndexRecord,
    lexical_terms,
    retrieval_metrics,
    source_sha256,
)


OUTPUT_ROOT = ROOT / "docs/papers/shared/results/paper3_1_summary_index/omission"


@dataclass(frozen=True)
class OmissionCase:
    example_id: str
    fact_type: str
    query: str
    target: str
    chunks: tuple[str, ...]
    positive_index: int


def _fact(fact_type: str, index: int) -> tuple[str, str, str]:
    values = {
        "entity": (
            f"Dr. Ilyra Venn-{index}",
            "Which researcher signed the calibration note?",
            "The calibration note was signed by {value}.",
        ),
        "alias": (
            f"Northglass-{index}",
            "What alias was assigned to the archive?",
            "In a margin note, the archive received the alias {value}.",
        ),
        "relation": (
            f"Orchid relay superseded Cinder node {index}",
            "Which relay superseded the Cinder node?",
            "A maintenance footnote states that {value}.",
        ),
        "date_number": (
            f"17-{index:02d}-2041 at 06:{index:02d}",
            "When was the backup seal verified?",
            "The backup seal was verified on {value}.",
        ),
        "rare_string": (
            f"VXQ-{index:02d}-LUMEN",
            "What literal checksum label identifies the pilot?",
            "The pilot checksum label, printed once in the appendix, is {value}.",
        ),
    }
    value, query, sentence = values[fact_type]
    return value, query, sentence.format(value=value)


def build_cases(count_per_type: int, seed: int) -> list[OmissionCase]:
    """Create topical chunks whose query target appears once as a minor detail."""

    rng = random.Random(seed)
    cases = []
    themes = (
        "A municipal observatory upgraded its weather instruments and published a long operational review.",
        "A university archive reorganized its collections and described the catalog migration.",
        "A transit authority documented routine maintenance across several regional stations.",
        "A laboratory reported a broad replication campaign and ordinary equipment changes.",
        "A standards committee summarized its annual meeting and administrative resolutions.",
        "A museum described conservation work, visitor programs, and storage improvements.",
    )
    for fact_type in ("entity", "alias", "relation", "date_number", "rare_string"):
        for index in range(count_per_type):
            target, query, low_salience = _fact(fact_type, index)
            chunks = []
            positive = rng.randrange(len(themes))
            for chunk_index, theme in enumerate(themes):
                detail = (
                    low_salience
                    if chunk_index == positive
                    else f"A minor appendix instead records routine batch {index}-{chunk_index}."
                )
                chunks.append(
                    f"{theme} The main discussion concerns scheduling, staffing, procurement, "
                    f"and common procedural outcomes. {detail} The report closes with general recommendations."
                )
            cases.append(
                OmissionCase(
                    example_id=f"{fact_type}-{index:03d}",
                    fact_type=fact_type,
                    query=query,
                    target=target,
                    chunks=tuple(chunks),
                    positive_index=positive,
                )
            )
    return cases


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clip_text(tokenizer, text: str, budget: int) -> tuple[str, int]:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids[:budget]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip(), len(token_ids)


def _salient_sentence(text: str) -> str:
    """Use the opening sentence as a cheap salience-biased extractive address."""

    return re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]


def _entity_rare_terms(chunks: tuple[str, ...], chunk_index: int) -> str:
    """Extract named-looking spans and corpus-rare terms without a language model."""

    chunk = chunks[chunk_index]
    document_frequency = {}
    for candidate in chunks:
        for term in set(lexical_terms(candidate)):
            document_frequency[term] = document_frequency.get(term, 0) + 1
    named = re.findall(
        r"\b(?:[A-Z][\w-]*(?:\s+[A-Z][\w-]*)+|[A-Z]{2,}[\w-]*|[\w]+-\d[\w-]*)\b",
        chunk,
    )
    rare = [
        term
        for term in lexical_terms(chunk)
        if document_frequency.get(term, 0) == 1 and (len(term) >= 5 or any(char.isdigit() for char in term))
    ]
    return " ".join(dict.fromkeys([*named, *rare])) or chunk


def _control_index(
    case: OmissionCase,
    tokenizer,
    texts: list[str],
    *,
    token_budget: int,
    prompt_id: str,
) -> SummaryIndex:
    records = []
    for index, (chunk, address) in enumerate(zip(case.chunks, texts)):
        clipped, count = _clip_text(tokenizer, address, token_budget)
        records.append(
            SummaryIndexRecord(
                uri=f"omission://{case.example_id}",
                chunk_id=f"chunk-{index}",
                token_start=index * 256,
                token_end=(index + 1) * 256,
                source_sha256=source_sha256(chunk),
                summary=clipped,
                summary_token_count=count,
                generation_model="extractive-control",
                prompt_id=prompt_id,
            )
        )
    return SummaryIndex(records)


def run(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    client = OllamaClient(args.ollama_url, args.ollama_timeout)
    cases = build_cases(args.count_per_type, args.seed)
    rows = []
    address_rows = []
    model_metadata = {}
    for profile in args.profiles:
        model = MODEL_PROFILES[profile]
        info = client.model_info(model)
        model_metadata[profile] = {
            "model": model,
            "modified_at": info.get("modified_at"),
            "details": info.get("details", {}),
            "model_info_sha256": hashlib.sha256(
                json.dumps(info, sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        items = [
            (f"{case.example_id}:chunk-{index}", chunk)
            for case in cases
            for index, chunk in enumerate(case.chunks)
        ]
        cache = JSONLGenerationCache(args.output_root / "summary_cache" / f"{profile}_{args.prompt}.jsonl")
        generated = generate_cached(
            client,
            cache,
            items,
            model=model,
            prompt_id=args.prompt,
            token_budget=args.token_budget,
            facet_count=1,
            seed=args.seed,
            batch_size=args.batch_size,
            structured_batches=False,
        )
        source_by_id = dict(items)
        address_rows.extend(
            {
                "profile": profile,
                "model": model,
                "prompt": args.prompt,
                "summary_prompt_version": PROMPT_VERSION,
                "item_id": item.item_id,
                "source_sha256": source_sha256(source_by_id[item.item_id]),
                "summary": item.summary,
                "facets": [
                    {"label": label, "text": text} for label, text in item.facets
                ],
                "prompt_eval_tokens": item.prompt_eval_tokens,
                "eval_tokens": item.eval_tokens,
                "generation_seconds": item.generation_seconds,
                "generation_mode": item.generation_mode,
                "raw_response_sha256": item.raw_response_sha256,
            }
            for item in generated
        )
        by_id = {row.item_id: row for row in generated}
        for case in cases:
            records = []
            source_records = []
            ingestion_seconds = 0.0
            for index, chunk in enumerate(case.chunks):
                item = by_id[f"{case.example_id}:chunk-{index}"]
                summary_ids = tokenizer(item.summary, add_special_tokens=False).input_ids[: args.token_budget]
                summary = tokenizer.decode(summary_ids, skip_special_tokens=True).strip()
                common = {
                    "uri": f"omission://{case.example_id}",
                    "chunk_id": f"chunk-{index}",
                    "token_start": index * 256,
                    "token_end": (index + 1) * 256,
                    "source_sha256": source_sha256(chunk),
                }
                records.append(
                    SummaryIndexRecord(
                        **common,
                        summary=summary,
                        summary_token_count=len(summary_ids),
                        generation_model=model,
                        prompt_id=args.prompt,
                    )
                )
                source_records.append(
                    SummaryIndexRecord(
                        **common,
                        summary=chunk,
                        summary_token_count=len(tokenizer(chunk, add_special_tokens=False).input_ids),
                        generation_model="none",
                        prompt_id="source-control",
                    )
                )
                ingestion_seconds += item.generation_seconds
            summary_index = SummaryIndex(records)
            source_index = SummaryIndex(source_records)
            salient_index = _control_index(
                case,
                tokenizer,
                [_salient_sentence(chunk) for chunk in case.chunks],
                token_budget=args.token_budget,
                prompt_id="salient-sentence",
            )
            entity_rare_index = _control_index(
                case,
                tokenizer,
                [_entity_rare_terms(case.chunks, index) for index in range(len(case.chunks))],
                token_budget=args.token_budget,
                prompt_id="entity-rare",
            )
            for condition, index in (
                ("source_bm25", source_index),
                ("salient_sentence_bm25", salient_index),
                ("entity_rare_bm25", entity_rare_index),
                (f"{profile}_{args.prompt}_summary_bm25", summary_index),
                (
                    f"{profile}_{args.prompt}_summary_bm25_shuffled",
                    summary_index.shuffled_addresses(args.seed),
                ),
            ):
                metrics = retrieval_metrics(
                    BM25SummaryScorer(index).score(case.query),
                    [case.positive_index],
                    k=args.selection_budget,
                )
                target_terms = set(lexical_terms(case.target))
                retained_terms = set(
                    lexical_terms(" ".join(index.records[case.positive_index].address_texts))
                )
                rows.append(
                    {
                        "profile": profile,
                        "model": model,
                        "prompt": args.prompt,
                        "example_id": case.example_id,
                        "fact_type": case.fact_type,
                        "condition": condition,
                        "evidence_recall": metrics["evidence_recall"],
                        "complete_recovery": metrics["complete_recovery"],
                        "target_literal_retained": float(target_terms <= retained_terms),
                        "target_term_retention": len(target_terms & retained_terms) / max(len(target_terms), 1),
                        "summary_index_bytes": index.text_bytes,
                        "ingestion_seconds": ingestion_seconds if "summary" in condition else 0.0,
                        "native_kv_rewritten": False,
                    }
                )
    grouped = {}
    for row in rows:
        key = (row["profile"], row["fact_type"], row["condition"])
        grouped.setdefault(key, []).append(row)
    summary = [
        {
            "profile": key[0],
            "fact_type": key[1],
            "condition": key[2],
            "examples": len(group),
            "evidence_recall": statistics.fmean(row["evidence_recall"] for row in group),
            "target_literal_retained": statistics.fmean(row["target_literal_retained"] for row in group),
            "target_term_retention": statistics.fmean(row["target_term_retention"] for row in group),
            "summary_index_bytes": statistics.fmean(row["summary_index_bytes"] for row in group),
            "ingestion_seconds": statistics.fmean(row["ingestion_seconds"] for row in group),
        }
        for key, group in sorted(grouped.items())
    ]
    _write_csv(args.output_root / "per_example.csv", rows)
    _write_csv(args.output_root / "summary.csv", summary)
    with (args.output_root / "summary_addresses.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        for row in address_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "1.0",
        "seed": args.seed,
        "count_per_fact_type": args.count_per_type,
        "fact_types": ["entity", "alias", "relation", "date_number", "rare_string"],
        "chunks_per_example": 6,
        "selection_budget_chunks": args.selection_budget,
        "summary_token_budget": args.token_budget,
        "profiles": list(args.profiles),
        "prompt": args.prompt,
        "summary_prompt_version": PROMPT_VERSION,
        "model_metadata": model_metadata,
        "native_kv_rewritten": False,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="+", choices=tuple(MODEL_PROFILES), default=("teacher_8b",))
    parser.add_argument("--prompt", choices=("generic", "retrieval"), default="retrieval")
    parser.add_argument("--count-per-type", type=int, default=10)
    parser.add_argument("--selection-budget", type=int, default=1)
    parser.add_argument("--token-budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=float, default=900.0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
