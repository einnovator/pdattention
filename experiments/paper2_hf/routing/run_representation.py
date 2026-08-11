"""Compare zero-parameter Qwen routing spaces on annotated QA evidence.

The runner is deliberately ranking-only. It captures a matched question query,
scores every cached source chunk, and reports evidence retrieval and systems
costs without using generated answers as a proxy for routing quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.native_kv_benchmarks import load_qasper_papers
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_smoke import evidence_token_spans, prompt_ids
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    CENTERED_ROPE_KEY,
    PRAHFConfig,
    canonical_routing_representation,
    inject_pra,
)


REPRESENTATIONS = (
    "post_rope_key",
    "pre_rope_key",
    CENTERED_ROPE_KEY,
    ATTENTION_INPUT_HIDDEN_STATE,
    "hidden_state",
)
DEFAULT_TOP_K = (3, 8, 16)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def _hotpot_examples(cache_dir: Path, count: int, seed: int) -> list[dict]:
    rows = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        cache_dir=str(cache_dir),
    )
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    examples = []
    for index in indices:
        row = rows[index]
        supporting = {
            (str(title), int(sentence_id))
            for title, sentence_id in zip(
                row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
            )
        }
        segments, evidence = [], []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            for sentence_id, sentence in enumerate(sentences):
                segment = f"{title}: {str(sentence).strip()}"
                segments.append(segment)
                if (str(title), sentence_id) in supporting:
                    evidence.append(segment)
        if evidence:
            examples.append(
                {
                    "dataset": "hotpotqa",
                    "id": str(row["id"]),
                    "question": str(row["question"]),
                    "source": "\n".join(segments),
                    "evidence": evidence,
                }
            )
        if len(examples) == count:
            return examples
    raise RuntimeError(f"HotpotQA yielded only {len(examples)} usable examples.")


def _qasper_examples(cache_dir: Path, count: int, seed: int) -> list[dict]:
    papers = load_qasper_papers("validation", cache_dir=cache_dir)
    candidates = []
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
                candidates.append(
                    {
                        "dataset": "qasper",
                        "id": f"{paper_id}:{qa.get('question_id', '')}",
                        "question": str(qa["question"]),
                        # Keep the exact smoke-study construction protocol.
                        "source": "\n".join(dict.fromkeys([*evidence, *paragraphs])),
                        "evidence": evidence,
                    }
                )
                break
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"QASPER yielded only {len(candidates)} usable examples.")
    return candidates[:count]


def load_examples(cache_dir: Path, count: int, seed: int) -> list[dict]:
    """Load matched deterministic validation subsets from both QA sources."""
    return [
        *_hotpot_examples(cache_dir, count, seed),
        *_qasper_examples(cache_dir / "qasper", count, seed + 1),
    ]


def _configure(
    handle,
    representation: str,
    chunk_size: int,
    gist_mode: str,
    gist_count: int,
    center_policy: str,
) -> None:
    handle.cache.clear()
    representation = canonical_routing_representation(representation)
    handle.hf_config.routing_representation = representation
    handle.hf_config.routing_chunk_tokens = int(chunk_size)
    handle.hf_config.gist_mode = gist_mode
    handle.hf_config.gists_per_chunk = int(gist_count)
    handle.hf_config.centered_rope_center_policy = center_policy
    handle.pra_config.fixed_chunk_tokens = int(chunk_size)
    handle.pra_config.gist_mode = gist_mode
    handle.pra_config.gists_per_chunk = int(gist_count)
    for adapter in handle.adapters.values():
        adapter.routing_representation = representation


@torch.no_grad()
def _capture_query(handle, tokenizer, example: dict, device: torch.device):
    encoded = prompt_ids(tokenizer, example["question"], max_tokens=128).to(device)
    positions = torch.arange(encoded.input_ids.shape[1], device=device).unsqueeze(0)
    adapter = next(iter(handle.adapters.values()))
    handle.set_memory_enabled(False)
    adapter.begin_capture(positions)
    handle.model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        position_ids=positions,
        use_cache=False,
    )
    captured = adapter.consume_capture()
    representation = handle.hf_config.routing_representation
    native_query = adapter.pra_core.prepare_pra_query(captured.post_query)
    if representation in {"post_rope_key", CENTERED_ROPE_KEY}:
        query = adapter.pra_core.prepare_pra_query(captured.post_query)
    elif representation == "pre_rope_key":
        query = adapter.pra_core.prepare_pra_query(captured.pre_query)
    elif representation == ATTENTION_INPUT_HIDDEN_STATE:
        query = captured.hidden_states[:, -1, :]
    else:
        raise ValueError(representation)
    return query, native_query, int(encoded.input_ids.shape[1])


def _overlaps(span: tuple[int, int], selected: list[tuple[int, int]]) -> bool:
    return any(max(span[0], start) < min(span[1], end) for start, end in selected)


def _primitive_topk_seconds(scores: list[float], k: int, device: torch.device) -> float:
    values = torch.tensor(scores, device=device)
    repeats = 25
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        torch.topk(values, k=min(k, len(scores)), sorted=True)
    _synchronize(device)
    return (time.perf_counter() - started) / repeats


def _ranking_row(
    *,
    handle,
    example: dict,
    source_tokens: int,
    evidence_spans: list[tuple[int, int]],
    query: torch.Tensor,
    native_query: torch.Tensor,
    direct_tokens: int,
    representation: str,
    chunk_size: int,
    gist_mode: str,
    gist_count: int,
    center_policy: str,
    top_k: int,
    index_build_seconds: float,
    warm_repeats: int,
    seed: int,
) -> dict:
    adapter = next(iter(handle.adapters.values()))
    handle.pra_config.top_k_chunks_per_reference = int(top_k)
    handle.pra_config.collect_routing_metrics = False
    _synchronize(query.device)
    started = time.perf_counter()
    for _ in range(warm_repeats):
        handle.cache.search(query, adapter.layer_idx, handle.pra_config)
    _synchronize(query.device)
    warm_seconds = (time.perf_counter() - started) / warm_repeats

    handle.pra_config.collect_routing_metrics = True
    selected = handle.cache.search(query, adapter.layer_idx, handle.pra_config)[0]
    rankings = handle.cache.last_rankings(adapter.layer_idx)[0][0]["chunks"]
    ranked_spans = [(int(row["token_start"]), int(row["token_end"])) for row in rankings]
    evidence_flags = [
        any(_overlaps(span, [ranked_span]) for span in evidence_spans)
        for ranked_span in ranked_spans
    ]
    evidence_ranks = [index + 1 for index, flag in enumerate(evidence_flags) if flag]
    best_rank = min(evidence_ranks, default=None)
    scores = [float(row["chunk_score"]) for row in rankings]
    positions = [((start + end) / 2) / max(source_tokens, 1) for start, end in ranked_spans]

    chunks_by_id = {
        chunk.chunk_id: chunk
        for entry in handle.cache.all_entries()
        for chunk in entry.layer_memory[adapter.layer_idx].chunks
    }
    winning_gist_spans = []
    winning_gist_evidence_flags = []
    for ranking in rankings:
        chunk = chunks_by_id[ranking["chunk_id"]]
        local_spans = chunk.routing_gist.metadata.get("segment_token_spans", [])
        winner = ranking.get("winning_gist_index")
        if winner is None or winner >= len(local_spans):
            winning_gist_spans.append(None)
            winning_gist_evidence_flags.append(False)
            continue
        local_start, local_end = local_spans[int(winner)]
        winner_span = (
            int(chunk.logical_start) + int(local_start),
            int(chunk.logical_start) + int(local_end),
        )
        winning_gist_spans.append(winner_span)
        winning_gist_evidence_flags.append(_overlaps(winner_span, evidence_spans))

    retained, budget = adapter.pra_core.budget_selected_memory(
        selected,
        direct_tokens=direct_tokens,
        routing_candidates=len(rankings),
    )
    selected_spans = [(int(hit.logical_start), int(hit.logical_end)) for hit in selected]
    selected_ids = [hit.chunk_id for hit in selected]
    materialized_spans = [(int(hit.logical_start), int(hit.logical_end)) for hit in retained]
    materialized_ids = [hit.chunk_id for hit in retained]
    selected_positions = [
        ((start + end) / 2) / max(source_tokens, 1) for start, end in selected_spans
    ]
    gist_bytes = sum(
        int(chunk.metadata.get("routing_gist_bytes", 0))
        for entry in handle.cache.all_entries()
        for chunk in entry.layer_memory[adapter.layer_idx].chunks
    )
    kv_bytes = sum(
        int(chunk.metadata.get("detail_kv_bytes", 0))
        for entry in handle.cache.all_entries()
        for chunk in entry.layer_memory[adapter.layer_idx].chunks
    )
    selected_kv_bytes = sum(
        hit.chunk.token_kv.k.numel() * hit.chunk.token_kv.k.element_size()
        + hit.chunk.token_kv.v.numel() * hit.chunk.token_kv.v.element_size()
        for hit in retained
    )
    actual_gist_counts = [
        int(chunk.routing_gist.k.shape[0]) for chunk in chunks_by_id.values()
    ]
    native_query_vector = native_query[0].to(query.device, torch.float32)
    native_token_max_scores = []
    native_token_mean_scores = []
    for ranking in rankings:
        chunk = chunks_by_id[ranking["chunk_id"]]
        token_keys = (
            chunk.token_kv.k.transpose(1, 2)
            .contiguous()
            .view(chunk.token_count, -1)
            .to(query.device, torch.float32)
        )
        token_scores = token_keys @ native_query_vector
        native_token_max_scores.append(float(token_scores.max().item()))
        native_token_mean_scores.append(float(token_scores.mean().item()))

    def recall_at(cutoff: int) -> float:
        return float(any(evidence_flags[:cutoff]))

    def all_recall_at(cutoff: int) -> float:
        spans = ranked_spans[:cutoff]
        return float(bool(evidence_spans) and all(_overlaps(span, spans) for span in evidence_spans))

    return {
        "dataset": example["dataset"],
        "example_id": example["id"],
        "seed": seed,
        "source_tokens": source_tokens,
        "candidate_chunks": len(rankings),
        "routing_chunk_size": chunk_size,
        "routing_representation": representation,
        "gist_mode": gist_mode,
        "gist_count": gist_count,
        "center_policy": center_policy,
        "mean_actual_gists_per_chunk": statistics.fmean(actual_gist_counts),
        "candidate_gists": sum(actual_gist_counts),
        "top_k": top_k,
        "evidence_token_spans": evidence_spans,
        "evidence_chunk_ids": [
            rankings[index]["chunk_id"] for index, flag in enumerate(evidence_flags) if flag
        ],
        "selected_chunk_ids": selected_ids,
        "selected_spans": selected_spans,
        "materialized_chunk_ids": materialized_ids,
        "materialized_spans": materialized_spans,
        "scores": scores,
        "normalized_source_positions": positions,
        "winning_gist_spans": winning_gist_spans,
        "winning_gist_evidence_flags": winning_gist_evidence_flags,
        "best_evidence_rank": best_rank,
        "mrr": 1.0 / best_rank if best_rank else 0.0,
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_8": recall_at(8),
        "recall_at_16": recall_at(16),
        "any_evidence_recall": recall_at(top_k),
        "all_evidence_recall": all_recall_at(top_k),
        "winning_gist_evidence_recall": float(
            any(winning_gist_evidence_flags[:top_k])
        ),
        "target_coverage": (
            sum(_overlaps(span, selected_spans) for span in evidence_spans)
            / max(len(evidence_spans), 1)
        ),
        "materialized_target_coverage": (
            sum(_overlaps(span, materialized_spans) for span in evidence_spans)
            / max(len(evidence_spans), 1)
        ),
        "selected_fraction": len(selected) / max(len(rankings), 1),
        "materialized_fraction": len(retained) / max(len(rankings), 1),
        "mean_selected_normalized_position": (
            statistics.fmean(selected_positions) if selected_positions else None
        ),
        "median_selected_normalized_position": (
            statistics.median(selected_positions) if selected_positions else None
        ),
        "score_position_correlation": _correlation(positions, scores),
        "native_token_max_score_correlation": _correlation(
            scores, native_token_max_scores
        ),
        "native_token_mean_score_correlation": _correlation(
            scores, native_token_mean_scores
        ),
        "selected_chunks": len(selected),
        "chunks_materialized": len(retained),
        "materialized_tokens": int(budget["memory_tokens_materialized"]),
        "active_kv_bytes": selected_kv_bytes,
        "cpu_to_gpu_selected_kv_bytes": selected_kv_bytes if query.is_cuda else 0,
        "routing_gist_bytes": gist_bytes,
        "detail_kv_bytes": kv_bytes,
        "extra_routing_cache_fraction": gist_bytes / max(kv_bytes, 1),
        "packed_index_build_seconds": index_build_seconds,
        "warm_routing_topk_seconds": warm_seconds,
        "topk_primitive_seconds": _primitive_topk_seconds(scores, top_k, query.device),
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    scalar_keys = sorted(
        {key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))}
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in scalar_keys} for row in rows)


def aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate matched examples by routing, gist, chunk-size, and top-k settings."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["dataset"],
                row["routing_representation"],
                row["routing_chunk_size"],
                row["gist_mode"],
                row["gist_count"],
                row.get("center_policy", "exact"),
                row["top_k"],
            )
        ].append(row)
    output = []
    metrics = (
        "recall_at_1",
        "recall_at_3",
        "recall_at_8",
        "recall_at_16",
        "any_evidence_recall",
        "all_evidence_recall",
        "winning_gist_evidence_recall",
        "mrr",
        "target_coverage",
        "materialized_target_coverage",
        "selected_fraction",
        "materialized_fraction",
        "mean_selected_normalized_position",
        "score_position_correlation",
        "native_token_max_score_correlation",
        "native_token_mean_score_correlation",
        "materialized_tokens",
        "active_kv_bytes",
        "extra_routing_cache_fraction",
        "mean_actual_gists_per_chunk",
        "candidate_gists",
        "packed_index_build_seconds",
        "warm_routing_topk_seconds",
        "topk_primitive_seconds",
    )
    for key, values in sorted(grouped.items()):
        record = {
            "dataset": key[0],
            "routing_representation": key[1],
            "routing_chunk_size": key[2],
            "gist_mode": key[3],
            "gist_count": key[4],
            "center_policy": key[5],
            "top_k": key[6],
            "examples": len(values),
        }
        for metric in metrics:
            samples = [float(row[metric]) for row in values if row.get(metric) is not None]
            record[metric] = statistics.fmean(samples) if samples else None
        output.append(record)
    return output


def _plot(aggregates: list[dict], output_dir: Path, stem: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    display_names = {
        ATTENTION_INPUT_HIDDEN_STATE: "attention input",
        CENTERED_ROPE_KEY: "centered RoPE key",
        "hidden_state": "attention input",
    }
    markers = {
        "post_rope_key": "o",
        "pre_rope_key": "s",
        CENTERED_ROPE_KEY: "D",
        ATTENTION_INPUT_HIDDEN_STATE: "^",
        "hidden_state": "^",
    }
    for (dataset, representation, gist_count), values in sorted(
        _group(aggregates, "dataset", "routing_representation", "gist_count").items()
    ):
        values = sorted(values, key=lambda row: row["selected_fraction"])
        axis.plot(
            [row["selected_fraction"] for row in values],
            [row["any_evidence_recall"] for row in values],
            marker=markers[representation],
            label=f"{dataset}: {display_names.get(representation, representation)}, G={gist_count}",
        )
    axis.set_xlabel("Selected chunk fraction")
    axis.set_ylabel("Any-evidence recall")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}_recall_vs_selected_fraction.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    labels, correlations = [], []
    for (dataset, representation, gist_count), values in sorted(
        _group(aggregates, "dataset", "routing_representation", "gist_count").items()
    ):
        samples = [row["score_position_correlation"] for row in values if row["score_position_correlation"] is not None]
        labels.append(
            f"{dataset}\n{display_names.get(representation, representation)}\nG={gist_count}"
        )
        correlations.append(statistics.fmean(samples) if samples else 0.0)
    axis.bar(range(len(labels)), correlations, color=["#4472c4", "#70ad47", "#ed7d31"] * 2)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Score-position correlation")
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}_position_bias.{suffix}", dpi=180)
    plt.close(figure)

    diagnostic_rows = [row for row in aggregates if int(row["top_k"]) == 3]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    labels = [
        f"{row['dataset']}\n{display_names.get(row['routing_representation'], row['routing_representation'])}\nG={row['gist_count']}"
        for row in diagnostic_rows
    ]
    maximum = [row["native_token_max_score_correlation"] or 0.0 for row in diagnostic_rows]
    mean = [row["native_token_mean_score_correlation"] or 0.0 for row in diagnostic_rows]
    x = torch.arange(len(labels), dtype=torch.float32).numpy()
    axis.bar(x - 0.2, maximum, width=0.4, label="native token-QK maximum")
    axis.bar(x + 0.2, mean, width=0.4, label="native token-QK mean")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Chunk-score correlation")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylim(-1.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{stem}_token_qk_correlation.{suffix}", dpi=180)
    plt.close(figure)


def _group(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def run(args) -> dict:
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
            max_prompt_direct_tokens=128,
            encoding_block_tokens=128,
            routing_chunk_tokens=min(args.chunk_sizes),
            centered_rope_center_policy=args.center_policy,
            gist_mode=args.gist_mode,
            gists_per_chunk=min(args.gist_counts),
            max_materialized_memory_tokens=128,
            top_k_references=1,
            top_k_chunks_per_reference=max(args.top_k),
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
    )
    examples = load_examples(args.cache_dir, args.examples_per_dataset, args.seed)
    checkpoint = args.output_dir / f"{args.stem}.checkpoint.json"
    rows = []
    if args.resume and checkpoint.exists():
        rows = json.loads(checkpoint.read_text(encoding="utf-8")).get("rows", [])
    completed = {
        (
            row["dataset"],
            row["example_id"],
            row["routing_representation"],
            row["routing_chunk_size"],
            row.get("gist_mode", "mean"),
            row.get("gist_count", 1),
            row.get("center_policy", "exact"),
            row["top_k"],
        )
        for row in rows
    }

    for representation in args.representations:
        for chunk_size in args.chunk_sizes:
            for gist_count in args.gist_counts:
                for example_index, example in enumerate(examples, start=1):
                    keys = [
                        (
                            example["dataset"],
                            example["id"],
                            representation,
                            chunk_size,
                            args.gist_mode,
                            gist_count,
                            args.center_policy,
                            top_k,
                        )
                        for top_k in args.top_k
                    ]
                    if all(key in completed for key in keys):
                        continue
                    _configure(
                        handle,
                        representation,
                        chunk_size,
                        args.gist_mode,
                        gist_count,
                        args.center_policy,
                    )
                    source = tokenizer(
                        example["source"], return_tensors="pt", add_special_tokens=False
                    ).input_ids
                    source_tokens = int(source.shape[1])
                    evidence_spans = evidence_token_spans(
                        tokenizer, example["source"], example["evidence"]
                    )
                    handle.add_reference(
                        f"benchmark://{example['dataset']}/{example['id']}",
                        source,
                        text=example["source"],
                    )
                    query, native_query, direct_tokens = _capture_query(
                        handle, tokenizer, example, device
                    )
                    adapter = next(iter(handle.adapters.values()))
                    _synchronize(device)
                    started = time.perf_counter()
                    handle.cache.prepare_routing_index(
                        adapter.layer_idx, query, force_rebuild=True
                    )
                    _synchronize(device)
                    index_build_seconds = time.perf_counter() - started
                    for top_k, key in zip(args.top_k, keys):
                        if key in completed:
                            continue
                        rows.append(
                            _ranking_row(
                                handle=handle,
                                example=example,
                                source_tokens=source_tokens,
                                evidence_spans=evidence_spans,
                                query=query,
                                native_query=native_query,
                                direct_tokens=direct_tokens,
                                representation=representation,
                                chunk_size=chunk_size,
                                gist_mode=args.gist_mode,
                                gist_count=gist_count,
                                center_policy=args.center_policy,
                                top_k=top_k,
                                index_build_seconds=index_build_seconds,
                                warm_repeats=args.warm_repeats,
                                seed=args.seed,
                            )
                        )
                        completed.add(key)
                    _write_json(checkpoint, {"rows": rows})
                    print(
                        f"[{example_index}/{len(examples)}] {representation} "
                        f"chunk={chunk_size} {args.gist_mode} G={gist_count} "
                        f"{example['dataset']} {example['id']}",
                        flush=True,
                    )

    aggregates = aggregate(rows)
    artifact = {
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "protocol": "frozen Qwen evidence-ranking evaluation; no answer generation",
        "seed": args.seed,
        "examples_per_dataset": args.examples_per_dataset,
        "representations": list(args.representations),
        "chunk_sizes": list(args.chunk_sizes),
        "gist_mode": args.gist_mode,
        "gist_counts": list(args.gist_counts),
        "center_policy": args.center_policy,
        "top_k": list(args.top_k),
        "rows": rows,
        "aggregates": aggregates,
        "max_native_operation_tokens": handle.max_native_operation_tokens,
        "native_limit_violations": handle.native_limit_violations,
    }
    _write_json(args.output_dir / f"{args.stem}.json", artifact)
    _write_csv(args.output_dir / f"{args.stem}.csv", rows)
    _write_csv(args.output_dir / f"{args.stem}_aggregate.csv", aggregates)
    _plot(aggregates, args.output_dir, args.stem)
    return artifact


def _csv_tuple(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--examples-per-dataset", type=int, default=8)
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--chunk-sizes", default="32")
    parser.add_argument("--gist-mode", default="mean")
    parser.add_argument("--gist-counts", default="1")
    parser.add_argument(
        "--center-policy",
        choices=("exact", "floor", "ceil"),
        default="exact",
    )
    parser.add_argument("--top-k", default=",".join(map(str, DEFAULT_TOP_K)))
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stem", default="qwen_routing_representation")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "routing",
    )
    args = parser.parse_args()
    args.representations = _csv_tuple(args.representations, str)
    args.chunk_sizes = _csv_tuple(args.chunk_sizes, int)
    args.gist_counts = _csv_tuple(args.gist_counts, int)
    args.top_k = _csv_tuple(args.top_k, int)
    invalid = set(args.representations) - set(REPRESENTATIONS)
    if invalid:
        parser.error(f"Unsupported representations: {sorted(invalid)}")
    args.representations = tuple(
        dict.fromkeys(
            canonical_routing_representation(value)
            for value in args.representations
        )
    )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    print(arguments.output_dir / f"{arguments.stem}.json")
