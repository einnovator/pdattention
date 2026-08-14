"""Audit the frozen-Qwen oracle gap without changing PRA materialization.

The runner reproduces the canonical HotpotQA/QASPER oracle controls, proves
that annotation-derived parent identities survive every requested consumer
layer, and then measures three possible explanations for the remaining gap:
annotation sufficiency, bounded-reference representation mismatch, and
softmax support dilution by non-evidence tokens in selected parents.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_multilayer_pra import (
    _generate_timed,
    layer_schedules,
)
from experiments.paper2_hf.qa.run_oracle_memory_use import (
    _answer_ids,
    _hidden_deltas,
    _oracle_selections,
    _prompt,
    _teacher_forced,
)
from experiments.paper2_hf.qa.run_smoke import answer_metrics
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import PRAHFConfig, inject_pra


def _mean(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _find_all(text: str, needle: str) -> list[int]:
    """Return every exact occurrence without using answer-side information."""
    starts = []
    cursor = 0
    while needle:
        found = text.find(needle, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + max(len(needle), 1)
    return starts


def token_span_from_offsets(
    offsets: list[tuple[int, int]] | list[list[int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    """Map one half-open character span to intersecting half-open token indices."""
    overlap = [
        index
        for index, (start, end) in enumerate(offsets)
        if int(end) > char_start and int(start) < char_end
    ]
    return (overlap[0], overlap[-1] + 1) if overlap else None


def build_evidence_span_audit(tokenizer, example: dict) -> dict:
    """Resolve annotation text through source characters and tokenizer offsets."""
    source = example["source"]
    encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(pair) for pair in encoded.offset_mapping]
    annotations = example.get("evidence_annotations") or [
        {"evidence_index": index, "text": text}
        for index, text in enumerate(example["evidence"])
    ]
    rows = []
    for index, (text, annotation) in enumerate(zip(example["evidence"], annotations)):
        starts = _find_all(source, text)
        chosen = starts[0] if starts else None
        token_span = (
            token_span_from_offsets(offsets, chosen, chosen + len(text))
            if chosen is not None
            else None
        )
        rows.append(
            {
                "evidence_index": index,
                "annotation": annotation,
                "evidence_text": text,
                "char_occurrences": [[start, start + len(text)] for start in starts],
                "selected_char_span": (
                    [chosen, chosen + len(text)] if chosen is not None else None
                ),
                "token_span": list(token_span) if token_span is not None else None,
                "representable": token_span is not None,
            }
        )
    missing = [row["evidence_index"] for row in rows if not row["representable"]]
    return {
        "source_characters": len(source),
        "source_tokens": len(encoded.input_ids),
        "annotations": rows,
        "all_representable": not missing,
        "missing_evidence_indices": missing,
        "token_spans": [row["token_span"] for row in rows if row["token_span"]],
    }


def materialized_source_positions(selected) -> tuple[list[int], int]:
    """Reconstruct physical source positions after overlap deduplication."""
    positions = []
    duplicate_tokens = 0
    covered_end_by_uri: dict[str, int] = {}
    ordered = sorted(
        selected,
        key=lambda hit: (hit.reference_uri, hit.token_start, hit.chunk_id),
    )
    for hit in ordered:
        covered_end = covered_end_by_uri.get(hit.reference_uri, hit.token_start)
        overlap = min(max(covered_end - hit.token_start, 0), hit.selected_token_count)
        duplicate_tokens += overlap
        positions.extend(range(hit.logical_start + overlap, hit.logical_end))
        covered_end_by_uri[hit.reference_uri] = max(covered_end, hit.token_end)
    return positions, duplicate_tokens


def audit_materialized_selection(selected, evidence_spans) -> dict:
    """Prove selected native-K/V positions cover every annotated token span."""
    positions, duplicates = materialized_source_positions(selected)
    position_set = set(positions)
    evidence_positions = {
        position for start, end in evidence_spans for position in range(start, end)
    }
    coverage = [
        all(position in position_set for position in range(start, end))
        for start, end in evidence_spans
    ]
    return {
        "requested_chunk_ids": [hit.chunk_id for hit in selected],
        "selected_token_spans": [[hit.logical_start, hit.logical_end] for hit in selected],
        "active_native_kv_tokens": len(positions),
        "materialized_source_token_ids": positions,
        "deduplicated_overlap_tokens": duplicates,
        "evidence_span_covered": coverage,
        "all_evidence_covered": bool(coverage) and all(coverage),
        "extra_non_evidence_tokens": len(position_set.difference(evidence_positions)),
    }


def counterfactual_softmax_diagnostic(
    weights: torch.Tensor,
    query_positions: list[int],
    evidence_keys: list[int],
    distractor_keys: list[int],
    local_keys: list[int],
) -> dict:
    """Remove D from already-computed probabilities and renormalize over E+H."""
    valid_queries = [index for index in query_positions if index < weights.shape[-2]]
    groups = {
        "evidence": [index for index in evidence_keys if index < weights.shape[-1]],
        "distractor": [index for index in distractor_keys if index < weights.shape[-1]],
        "local": [index for index in local_keys if index < weights.shape[-1]],
    }
    if not valid_queries:
        return {}
    probs = weights[0, :, valid_queries, :].float().clamp_min(0)

    def mass(indices):
        if not indices:
            return torch.zeros(probs.shape[:2], device=probs.device)
        return probs[..., indices].sum(dim=-1)

    evidence = mass(groups["evidence"])
    distractor = mass(groups["distractor"])
    local = mass(groups["local"])
    actual_entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    retained = groups["evidence"] + groups["local"]
    retained_probs = probs[..., retained] if retained else probs[..., :0]
    retained_probs = retained_probs / retained_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    counterfactual_entropy = -(
        retained_probs * retained_probs.clamp_min(1e-12).log()
    ).sum(dim=-1)
    recovered_evidence = evidence / (evidence + local).clamp_min(1e-12)
    top_evidence = (
        probs[..., groups["evidence"]].max(dim=-1).values
        if groups["evidence"]
        else torch.zeros_like(evidence)
    )
    counterfactual_top = top_evidence / (evidence + local).clamp_min(1e-12)

    per_head = []
    for head in range(probs.shape[0]):
        per_head.append(
            {
                "head": head,
                "evidence_mass": float(evidence[head].mean().item()),
                "distractor_mass": float(distractor[head].mean().item()),
                "local_mass": float(local[head].mean().item()),
                "attention_entropy": float(actual_entropy[head].mean().item()),
                "counterfactual_evidence_mass": float(
                    recovered_evidence[head].mean().item()
                ),
                "counterfactual_entropy": float(
                    counterfactual_entropy[head].mean().item()
                ),
                "top_evidence_weight": float(top_evidence[head].mean().item()),
                "counterfactual_top_evidence_weight": float(
                    counterfactual_top[head].mean().item()
                ),
            }
        )
    return {
        key: _mean([row[key] for row in per_head])
        for key in per_head[0]
        if key != "head"
    } | {"per_head": per_head}


def oracle_attention_trace(adapter, query_positions, evidence_spans) -> dict:
    """Partition oracle-visible support into E, D, and ordinary local/head H."""
    weights = adapter.last_attention_weights
    if weights is None:
        return {}
    positions, _ = materialized_source_positions(adapter.last_selected_chunks[0])
    evidence_set = {
        position for start, end in evidence_spans for position in range(start, end)
    }
    evidence_keys = [index for index, position in enumerate(positions) if position in evidence_set]
    distractor_keys = [index for index, position in enumerate(positions) if position not in evidence_set]
    memory_width = int(adapter.last_diagnostics.get("hf_memory_width", len(positions)))
    local_keys = list(range(memory_width, weights.shape[-1]))
    return counterfactual_softmax_diagnostic(
        weights,
        query_positions,
        evidence_keys,
        distractor_keys,
        local_keys,
    )


def align_evidence_token_ids(
    source_ids: list[int],
    evidence_spans: list[tuple[int, int]],
    prompt_ids: list[int],
    context_positions: list[int],
) -> dict:
    """Align identical BPE tokens between source evidence and direct-text context."""
    context_ids = [prompt_ids[index] for index in context_positions]
    used_context: set[int] = set()
    pairs = []
    expected = 0
    for start, end in evidence_spans:
        needle = source_ids[start:end]
        expected += len(needle)
        exact_start = None
        for candidate in range(len(context_ids) - len(needle) + 1):
            candidate_positions = context_positions[candidate : candidate + len(needle)]
            if any(position in used_context for position in candidate_positions):
                continue
            if context_ids[candidate : candidate + len(needle)] == needle:
                exact_start = candidate
                break
        if exact_start is not None:
            for offset in range(len(needle)):
                prompt_position = context_positions[exact_start + offset]
                pairs.append((start + offset, prompt_position))
                used_context.add(prompt_position)
            continue
        matcher = difflib.SequenceMatcher(a=needle, b=context_ids, autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                prompt_position = context_positions[block.b + offset]
                if prompt_position not in used_context:
                    pairs.append((start + block.a + offset, prompt_position))
                    used_context.add(prompt_position)
    pairs.sort()
    return {
        "pairs": [[source, prompt] for source, prompt in pairs],
        "expected_evidence_tokens": expected,
        "aligned_tokens": len(pairs),
        "alignment_fraction": len(pairs) / max(expected, 1),
    }


@torch.no_grad()
def _capture_positions(handle, input_ids, attention_mask, positions, wanted_positions):
    """Capture pre/post-RoPE K and V at ordered token positions for every layer."""
    handle.configure_memory_layers(set())
    for adapter in handle.adapters.values():
        adapter.begin_capture(positions)
    handle.model(
        input_ids=input_ids.to(handle.device),
        attention_mask=attention_mask.to(handle.device),
        position_ids=positions.to(handle.device),
        use_cache=False,
    )
    wanted = torch.tensor(wanted_positions, device=handle.device, dtype=torch.long)
    result = {}
    for layer, adapter in handle.adapters.items():
        capture = adapter.consume_capture()
        result[layer] = {
            "pre_key": capture.pre_key[0, :, wanted, :].float().cpu(),
            "post_key": capture.detail_kv.k[0, :, wanted, :].float().cpu(),
            "value": capture.detail_kv.v[0, :, wanted, :].float().cpu(),
        }
    return result


@torch.no_grad()
def capture_reference_positions(handle, source_ids, wanted_source_positions):
    """Replay the exact bounded publication protocol and retain requested tokens."""
    ordered = sorted(set(int(position) for position in wanted_source_positions))
    by_layer = {
        layer: {"pre_key": [], "post_key": [], "value": []}
        for layer in handle.adapters
    }
    retained_positions = []
    block_tokens = int(handle.hf_config.encoding_block_tokens)
    for block_start in range(0, source_ids.shape[1], block_tokens):
        block_end = min(block_start + block_tokens, source_ids.shape[1])
        selected = [position for position in ordered if block_start <= position < block_end]
        if not selected:
            continue
        block_ids = source_ids[:, block_start:block_end].to(handle.device)
        positions = torch.arange(block_start, block_end, device=handle.device).unsqueeze(0)
        local = [position - block_start for position in selected]
        captured = _capture_positions(
            handle,
            block_ids,
            torch.ones_like(block_ids),
            positions,
            local,
        )
        retained_positions.extend(selected)
        for layer in handle.adapters:
            for key in by_layer[layer]:
                by_layer[layer][key].append(captured[layer][key])
    for layer in by_layer:
        for key, parts in by_layer[layer].items():
            by_layer[layer][key] = torch.cat(parts, dim=1) if parts else torch.empty(0)
    return retained_positions, by_layer


def _tensor_similarity(reference: torch.Tensor, observed: torch.Tensor) -> dict:
    reference = reference.float()
    observed = observed.float()
    relative = float((observed - reference).norm() / reference.norm().clamp_min(1e-12))
    cosine = float(F.cosine_similarity(reference.flatten(), observed.flatten(), dim=0).item())
    per_head = []
    for head in range(reference.shape[0]):
        per_head.append(
            {
                "head": head,
                "relative_error": float(
                    (observed[head] - reference[head]).norm()
                    / reference[head].norm().clamp_min(1e-12)
                ),
                "cosine": float(
                    F.cosine_similarity(
                        reference[head].flatten(), observed[head].flatten(), dim=0
                    ).item()
                ),
            }
        )
    return {"relative_error": relative, "cosine": cosine, "per_head": per_head}


def representation_audit(reference_capture, direct_capture, source_positions, alignment) -> list[dict]:
    """Compare aligned evidence representations under reference and direct context."""
    index_by_source = {position: index for index, position in enumerate(source_positions)}
    ref_indices = [index_by_source[source] for source, _ in alignment["pairs"]]
    rows = []
    for layer in sorted(reference_capture):
        row = {"layer": layer, "aligned_tokens": len(ref_indices)}
        for key, prefix in (("pre_key", "pre_k"), ("post_key", "post_k"), ("value", "v")):
            metrics = _tensor_similarity(
                reference_capture[layer][key][:, ref_indices, :],
                direct_capture[layer][key],
            )
            row[f"{prefix}_relative_error"] = metrics["relative_error"]
            row[f"{prefix}_cosine"] = metrics["cosine"]
            row[f"{prefix}_per_head"] = metrics["per_head"]
        rows.append(row)
    return rows


def cache_fidelity_audit(entry, fresh_capture, source_positions) -> list[dict]:
    """Compare cached post-RoPE K/native V against a fresh publication replay."""
    index_by_source = {position: index for index, position in enumerate(source_positions)}
    rows = []
    for layer, memory in sorted(entry.layer_memory.items()):
        cached = {}
        for chunk in sorted(memory.chunks, key=lambda item: item.token_start):
            for local, position in enumerate(range(chunk.logical_start, chunk.logical_end)):
                cached.setdefault(
                    position,
                    (
                        chunk.token_kv.k[0, :, local, :].float().cpu(),
                        chunk.token_kv.v[0, :, local, :].float().cpu(),
                    ),
                )
        comparable = [position for position in source_positions if position in cached]
        fresh_indices = [index_by_source[position] for position in comparable]
        cached_k = torch.stack([cached[position][0] for position in comparable], dim=1)
        cached_v = torch.stack([cached[position][1] for position in comparable], dim=1)
        fresh_k = fresh_capture[layer]["post_key"][:, fresh_indices, :]
        fresh_v = fresh_capture[layer]["value"][:, fresh_indices, :]
        rows.append(
            {
                "layer": layer,
                "tokens": len(comparable),
                "post_k_max_abs_error": float((cached_k - fresh_k).abs().max().item()),
                "v_max_abs_error": float((cached_v - fresh_v).abs().max().item()),
                "post_k_cosine": _tensor_similarity(cached_k, fresh_k)["cosine"],
                "v_cosine": _tensor_similarity(cached_v, fresh_v)["cosine"],
            }
        )
    return rows


def parent_context(example: dict) -> str:
    """Return annotation-containing parents without consulting the gold answer."""
    parents = example.get("parent_paragraphs") or example["evidence"]
    return "\n".join(dict.fromkeys(parents))


@torch.no_grad()
def score_native_control(handle, tokenizer, question, answer, context, max_tokens, device):
    """Score a native prompt with PRA disabled and no retained attention tensors."""
    handle.configure_memory_layers(set())
    prompt_ids, prompt_mask, context_tokens = _prompt(
        tokenizer, question, context=context, max_tokens=max_tokens
    )
    answer_ids = _answer_ids(tokenizer, answer)
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    prediction_positions = list(range(prompt_ids.shape[1] - 1, full_ids.shape[1] - 1))
    output = handle.model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
    logits = output.logits[:, prediction_positions, :].float()
    targets = answer_ids.to(device)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    first = logits[0, 0]
    target = int(targets[0, 0])
    competitor = first.clone()
    competitor[target] = float("-inf")
    return {
        "prompt_tokens": int(prompt_ids.shape[1]),
        "context_tokens": len(context_tokens),
        "gold_sequence_logprob": float(token_log_probs.sum().item()),
        "gold_mean_token_logprob": float(token_log_probs.mean().item()),
        "gold_first_token_probability": float(first.softmax(dim=-1)[target].item()),
        "gold_first_token_rank": int((first > first[target]).sum().item()) + 1,
        "gold_first_token_margin": float((first[target] - competitor.max()).item()),
    }


def _context_complete(tokenizer, question: str, context: str | None, max_tokens: int) -> bool:
    prompt_ids, _, _ = _prompt(tokenizer, question, context=context, max_tokens=1_000_000)
    return int(prompt_ids.shape[1]) <= max_tokens


def _oracle_condition(
    handle,
    tokenizer,
    example,
    evidence_spans,
    entry,
    layers,
    baseline_hidden,
    args,
    device,
    *,
    generate: bool = True,
):
    oracle = {
        layer: [_oracle_selections(entry, layer, evidence_spans)] for layer in layers
    }
    handle.configure_memory_layers(set(layers), fixed_selections=oracle)
    prompt_ids, prompt_mask, context_tokens = _prompt(
        tokenizer, example["question"], max_tokens=args.prompt_tokens
    )
    answer_ids = _answer_ids(tokenizer, example["answer"])
    scored, hidden = _teacher_forced(
        handle,
        tokenizer,
        prompt_ids,
        prompt_mask,
        answer_ids,
        context_tokens,
        evidence_spans,
        device,
        retain_attention_weights=True,
    )
    prediction_positions = list(
        range(prompt_ids.shape[1] - 1, prompt_ids.shape[1] + answer_ids.shape[1] - 1)
    )
    layer_audits = {}
    attention = {}
    for layer in layers:
        adapter = handle.adapters[layer]
        selection = audit_materialized_selection(
            adapter.last_selected_chunks[0], evidence_spans
        )
        if not selection["all_evidence_covered"]:
            raise AssertionError(
                f"Oracle evidence missing after materialization at layer {layer}: {selection}"
            )
        layer_audits[str(layer)] = selection
        attention[str(layer)] = oracle_attention_trace(
            adapter, prediction_positions, evidence_spans
        )
    handle.set_attention_diagnostics(False)
    generated = (
        _generate_timed(handle, tokenizer, prompt_ids, prompt_mask, device, args.new_tokens)
        if generate
        else {}
    )
    return {
        **scored,
        **generated,
        **(
            answer_metrics(generated["generated_answer"], example["answer"])
            if generate
            else {}
        ),
        "layers": list(layers),
        "layer_materialization_audit": layer_audits,
        "attention_support_by_layer": attention,
        "hidden_state_delta_by_layer": (
            _hidden_deltas(hidden, baseline_hidden) if baseline_hidden is not None else []
        ),
    }


def _direct_attention_audit(
    handle,
    tokenizer,
    example,
    prompt_ids,
    prompt_mask,
    context_positions,
    evidence_spans,
    device,
) -> dict[str, dict]:
    """Capture native direct-evidence attention using the same gold positions."""
    handle.configure_memory_layers(set())
    answer_ids = _answer_ids(tokenizer, example["answer"])
    _teacher_forced(
        handle,
        tokenizer,
        prompt_ids,
        prompt_mask,
        answer_ids,
        context_positions,
        evidence_spans,
        device,
        retain_attention_weights=True,
    )
    queries = list(
        range(prompt_ids.shape[1] - 1, prompt_ids.shape[1] + answer_ids.shape[1] - 1)
    )
    evidence_keys = list(context_positions)
    rows = {}
    for layer, adapter in handle.adapters.items():
        weights = adapter.last_attention_weights
        if weights is None:
            continue
        evidence_set = set(evidence_keys)
        local_keys = [index for index in range(weights.shape[-1]) if index not in evidence_set]
        rows[str(layer)] = counterfactual_softmax_diagnostic(
            weights,
            queries,
            evidence_keys,
            [],
            local_keys,
        )
    handle.set_attention_diagnostics(False)
    return rows


def _run_encoding_sweep(handle, tokenizer, examples, example_artifacts, args, device):
    """Vary bounded publication context while preserving 32-token oracle parents."""
    if not args.encoding_sweep:
        return []
    original_block_tokens = int(handle.hf_config.encoding_block_tokens)
    rows = []
    selected_examples = []
    for dataset in sorted({example["dataset"] for example in examples}):
        selected_examples.extend(
            [example for example in examples if example["dataset"] == dataset][
                : args.sweep_examples_per_dataset
            ]
        )
    artifacts_by_id = {
        (row["dataset"], row["example_id"]): row for row in example_artifacts
    }
    try:
        for example in selected_examples:
            source_ids = tokenizer(
                example["source"], return_tensors="pt", add_special_tokens=False
            ).input_ids
            span_audit = build_evidence_span_audit(tokenizer, example)
            evidence_spans = [tuple(span) for span in span_audit["token_spans"]]
            direct_ids, direct_mask, direct_context = _prompt(
                tokenizer,
                example["question"],
                context="\n".join(example["evidence"]),
                max_tokens=args.direct_text_tokens,
            )
            alignment = align_evidence_token_ids(
                source_ids[0].tolist(),
                evidence_spans,
                direct_ids[0].tolist(),
                direct_context,
            )
            direct_positions = [prompt for _, prompt in alignment["pairs"]]
            direct_capture = _capture_positions(
                handle,
                direct_ids.to(device),
                direct_mask.to(device),
                torch.arange(direct_ids.shape[1], device=device).unsqueeze(0),
                direct_positions,
            )
            baseline = artifacts_by_id[(example["dataset"], example["id"])]["controls"][
                "no_context"
            ]["gold_mean_token_logprob"]
            baseline_parent_ids = None
            for block_tokens in args.encoding_sweep:
                handle.hf_config.encoding_block_tokens = int(block_tokens)
                handle.cache.clear()
                entry = handle.add_reference(
                    f"benchmark://{example['dataset']}/{example['id']}/encoding-{block_tokens}",
                    source_ids,
                    text=example["source"],
                )
                condition = _oracle_condition(
                    handle,
                    tokenizer,
                    example,
                    evidence_spans,
                    entry,
                    layer_schedules(len(handle.adapters))["last_half"],
                    None,
                    args,
                    device,
                    generate=False,
                )
                last_layer = str(max(condition["layers"]))
                parent_ids = condition["layer_materialization_audit"][last_layer][
                    "selected_token_spans"
                ]
                if baseline_parent_ids is None:
                    baseline_parent_ids = parent_ids
                if parent_ids != baseline_parent_ids:
                    raise AssertionError(
                        "Encoding-context sweep changed final oracle materialization."
                    )
                source_positions, reference_capture = capture_reference_positions(
                    handle,
                    source_ids,
                    [source for source, _ in alignment["pairs"]],
                )
                representation = representation_audit(
                    reference_capture,
                    direct_capture,
                    source_positions,
                    alignment,
                )
                late = [
                    row
                    for row in representation
                    if row["layer"] in condition["layers"]
                ]
                rows.append(
                    {
                        "dataset": example["dataset"],
                        "example_id": example["id"],
                        "encoding_block_tokens": int(block_tokens),
                        "oracle_last_half_delta_vs_no_context": condition[
                            "gold_mean_token_logprob"
                        ]
                        - baseline,
                        "late_pre_k_cosine": _mean([row["pre_k_cosine"] for row in late]),
                        "late_post_k_cosine": _mean([row["post_k_cosine"] for row in late]),
                        "late_v_cosine": _mean([row["v_cosine"] for row in late]),
                        "materialized_parent_spans": parent_ids,
                    }
                )
                print(
                    f"[encoding sweep] {example['dataset']} block={block_tokens} "
                    f"delta={rows[-1]['oracle_last_half_delta_vs_no_context']:+.3f}",
                    flush=True,
                )
    finally:
        handle.hf_config.encoding_block_tokens = original_block_tokens
    return rows


def _rectangular_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_control_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    aggregates = []
    for (dataset, condition), values in sorted(grouped.items()):
        aggregates.append(
            {
                "dataset": dataset,
                "condition": condition,
                "examples": len(values),
                "gold_mean_token_logprob": _mean(
                    [row["gold_mean_token_logprob"] for row in values]
                ),
                "delta_vs_no_context": _mean(
                    [row["delta_vs_no_context"] for row in values]
                ),
                "gold_first_token_rank": _mean(
                    [row["gold_first_token_rank"] for row in values]
                ),
                "gold_first_token_margin": _mean(
                    [row["gold_first_token_margin"] for row in values]
                ),
                "f1": _mean([row.get("f1") for row in values]),
                "em": _mean([row.get("em") for row in values]),
                "context_complete_fraction": _mean(
                    [float(row.get("context_complete", True)) for row in values]
                ),
            }
        )
    return aggregates


def _aggregate_layer_rows(examples: list[dict], key: str, condition: str) -> list[dict]:
    grouped = defaultdict(list)
    for example in examples:
        for layer, row in example[condition][key].items():
            grouped[(example["dataset"], int(layer))].append(row)
    output = []
    for (dataset, layer), values in sorted(grouped.items()):
        scalar_keys = [
            name for name, value in values[0].items() if not isinstance(value, (dict, list))
        ]
        output.append(
            {"dataset": dataset, "condition": condition, "layer": layer}
            | {name: _mean([row.get(name) for row in values]) for name in scalar_keys}
        )
    return output


def _plots(output_dir: Path, controls, representation, attention, divergence) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    control_order = [
        "exact_evidence",
        "parent_paragraphs",
        "full_context",
        "oracle_last_half",
        "oracle_all",
    ]
    display_names = {
        "exact_evidence": "Exact evidence",
        "parent_paragraphs": "Parent context",
        "full_context": "Full context",
        "oracle_last_half": "Oracle last 14",
        "oracle_all": "Oracle all 28",
    }
    for dataset in sorted({row["dataset"] for row in controls}):
        selected = {row["condition"]: row for row in controls if row["dataset"] == dataset}
        labels = [name for name in control_order if name in selected]
        axes[0, 0].plot(
            [display_names[name] for name in labels],
            [selected[name]["delta_vs_no_context"] for name in labels],
            marker="o",
            label="HotpotQA" if dataset == "hotpotqa" else "QASPER",
        )
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_ylabel("gold mean logP delta")
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].legend()

    for dataset in sorted({row["dataset"] for row in representation}):
        selected = [row for row in representation if row["dataset"] == dataset]
        axes[0, 1].plot(
            [row["layer"] for row in selected],
            [row["v_cosine"] for row in selected],
            label=f"{'HotpotQA' if dataset == 'hotpotqa' else 'QASPER'} V",
        )
        axes[0, 1].plot(
            [row["layer"] for row in selected],
            [row["pre_k_cosine"] for row in selected],
            linestyle="--",
            label=f"{'HotpotQA' if dataset == 'hotpotqa' else 'QASPER'} K pre-RoPE",
        )
    axes[0, 1].set_ylabel("aligned cosine")
    axes[0, 1].set_xlabel("layer")
    axes[0, 1].legend(fontsize=7)

    labels = []
    evidence, distractor, local = [], [], []
    for dataset in sorted({row["dataset"] for row in attention}):
        rows = [row for row in attention if row["dataset"] == dataset]
        for condition in ("oracle_last_half", "oracle_all"):
            values = [row for row in rows if row["condition"] == condition]
            dataset_name = "HotpotQA" if dataset == "hotpotqa" else "QASPER"
            band = "last 14" if condition == "oracle_last_half" else "all 28"
            labels.append(f"{dataset_name}\n{band}")
            evidence.append(_mean([row["evidence_mass"] for row in values]) or 0)
            distractor.append(_mean([row["distractor_mass"] for row in values]) or 0)
            local.append(_mean([row["local_mass"] for row in values]) or 0)
    x = range(len(labels))
    axes[1, 0].bar(x, evidence, label="evidence E")
    axes[1, 0].bar(x, distractor, bottom=evidence, label="selected distractor D")
    axes[1, 0].bar(
        x,
        local,
        bottom=[left + middle for left, middle in zip(evidence, distractor)],
        label="local/head H",
    )
    axes[1, 0].set_xticks(list(x), labels)
    axes[1, 0].set_ylabel("attention mass")
    axes[1, 0].legend(fontsize=7)

    for dataset in sorted({row["dataset"] for row in divergence}):
        for condition in ("oracle_last_half", "oracle_all"):
            selected = [
                row
                for row in divergence
                if row["dataset"] == dataset and row["condition"] == condition
            ]
            axes[1, 1].plot(
                [row["layer"] for row in selected],
                [row["relative_l2_delta"] for row in selected],
                label=(
                    f"{'HotpotQA' if dataset == 'hotpotqa' else 'QASPER'} "
                    f"{'last 14' if condition == 'oracle_last_half' else 'all 28'}"
                ),
            )
    axes[1, 1].set_xlabel("layer")
    axes[1, 1].set_ylabel("hidden relative L2 vs no context")
    axes[1, 1].legend(fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"oracle_gap_diagnostics.{suffix}", dpi=180)
    plt.close(figure)


def _encoding_plot(output_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for dataset in sorted({row["dataset"] for row in rows}):
        selected = sorted(
            [row for row in rows if row["dataset"] == dataset],
            key=lambda row: row["encoding_block_tokens"],
        )
        x = [row["encoding_block_tokens"] for row in selected]
        axes[0].plot(
            x,
            [row["oracle_last_half_delta_vs_no_context"] for row in selected],
            marker="o",
            label="HotpotQA" if dataset == "hotpotqa" else "QASPER",
        )
        axes[1].plot(
            x,
            [row["late_v_cosine"] for row in selected],
            marker="o",
            label=f"{'HotpotQA' if dataset == 'hotpotqa' else 'QASPER'} V",
        )
        axes[1].plot(
            x,
            [row["late_pre_k_cosine"] for row in selected],
            marker="s",
            linestyle="--",
            label=f"{'HotpotQA' if dataset == 'hotpotqa' else 'QASPER'} K",
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("oracle last-14 gold logP delta")
    axes[1].set_ylabel("direct/reference cosine")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("reference encoding block tokens")
        axis.legend(fontsize=7)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"oracle_gap_encoding_sweep.{suffix}", dpi=180)
    plt.close(figure)


def run(args) -> dict:
    torch.manual_seed(args.seed)
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layer_count = int(model.config.num_hidden_layers)
    schedules = layer_schedules(layer_count)
    all_layers = schedules["all"]
    last_half = schedules["last_half"]
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=all_layers,
            model_max_context_tokens=args.native_tokens,
            max_prompt_direct_tokens=args.prompt_tokens,
            encoding_block_tokens=args.encoding_block_tokens,
            routing_chunk_tokens=args.routing_chunk_tokens,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=args.top_k,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
    )
    examples = load_split_examples(
        args.cache_dir, args.examples_per_dataset, args.example_offset, args.seed
    )
    example_artifacts = []
    control_rows = []
    for number, example in enumerate(examples, start=1):
        handle.cache.clear()
        span_audit = build_evidence_span_audit(tokenizer, example)
        if not span_audit["all_representable"]:
            raise AssertionError(f"Unrepresentable evidence: {example['id']} {span_audit}")
        evidence_spans = [tuple(span) for span in span_audit["token_spans"]]
        source_ids = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        entry = handle.add_reference(
            f"benchmark://{example['dataset']}/{example['id']}",
            source_ids,
            text=example["source"],
        )
        no_prompt_ids, no_prompt_mask, no_context_tokens = _prompt(
            tokenizer, example["question"], max_tokens=args.prompt_tokens
        )
        answer_ids = _answer_ids(tokenizer, example["answer"])
        handle.configure_memory_layers(set())
        no_scored, baseline_hidden = _teacher_forced(
            handle,
            tokenizer,
            no_prompt_ids,
            no_prompt_mask,
            answer_ids,
            no_context_tokens,
            evidence_spans,
            device,
        )
        no_generated = _generate_timed(
            handle, tokenizer, no_prompt_ids, no_prompt_mask, device, args.new_tokens
        )
        no_row = {
            "dataset": example["dataset"],
            "example_id": example["id"],
            "condition": "no_context",
            **{key: value for key, value in no_scored.items() if key != "attention_by_layer"},
            **no_generated,
            **answer_metrics(no_generated["generated_answer"], example["answer"]),
            "context_complete": True,
            "delta_vs_no_context": 0.0,
        }
        controls = {"no_context": no_row}
        contexts = {
            "exact_evidence": ("\n".join(example["evidence"]), args.direct_text_tokens),
            "parent_paragraphs": (parent_context(example), args.full_context_tokens),
            "full_context": (example.get("full_context", example["source"]), args.full_context_tokens),
        }
        for name, (context, limit) in contexts.items():
            complete = _context_complete(tokenizer, example["question"], context, limit)
            if name == "full_context" and not complete:
                continue
            row = score_native_control(
                handle,
                tokenizer,
                example["question"],
                example["answer"],
                context,
                limit,
                device,
            )
            generated_metrics = {}
            if row["prompt_tokens"] <= args.generation_control_limit:
                prompt_ids, prompt_mask, _ = _prompt(
                    tokenizer, example["question"], context=context, max_tokens=limit
                )
                generated = _generate_timed(
                    handle, tokenizer, prompt_ids, prompt_mask, device, args.new_tokens
                )
                generated_metrics = generated | answer_metrics(
                    generated["generated_answer"], example["answer"]
                )
            controls[name] = {
                "dataset": example["dataset"],
                "example_id": example["id"],
                "condition": name,
                **row,
                **generated_metrics,
                "context_complete": complete,
                "delta_vs_no_context": row["gold_mean_token_logprob"]
                - no_scored["gold_mean_token_logprob"],
            }

        oracle_last_half = _oracle_condition(
            handle,
            tokenizer,
            example,
            evidence_spans,
            entry,
            last_half,
            baseline_hidden,
            args,
            device,
        )
        oracle_all = _oracle_condition(
            handle,
            tokenizer,
            example,
            evidence_spans,
            entry,
            all_layers,
            baseline_hidden,
            args,
            device,
        )
        for name, value in (("oracle_last_half", oracle_last_half), ("oracle_all", oracle_all)):
            controls[name] = {
                "dataset": example["dataset"],
                "example_id": example["id"],
                "condition": name,
                **{key: item for key, item in value.items() if not isinstance(item, (dict, list))},
                "context_complete": True,
                "delta_vs_no_context": value["gold_mean_token_logprob"]
                - no_scored["gold_mean_token_logprob"],
            }

        direct_prompt_ids, direct_prompt_mask, direct_context_positions = _prompt(
            tokenizer,
            example["question"],
            context="\n".join(example["evidence"]),
            max_tokens=args.direct_text_tokens,
        )
        alignment = align_evidence_token_ids(
            source_ids[0].tolist(),
            evidence_spans,
            direct_prompt_ids[0].tolist(),
            direct_context_positions,
        )
        if not alignment["pairs"]:
            raise AssertionError(f"No direct/reference evidence alignment for {example['id']}")
        oracle_positions, _ = materialized_source_positions(
            _oracle_selections(entry, all_layers[-1], evidence_spans)
        )
        wanted_source = sorted(
            set(oracle_positions).union(source for source, _ in alignment["pairs"])
        )
        source_positions, reference_capture = capture_reference_positions(
            handle, source_ids, wanted_source
        )
        direct_positions = [prompt for _, prompt in alignment["pairs"]]
        direct_capture = _capture_positions(
            handle,
            direct_prompt_ids.to(device),
            direct_prompt_mask.to(device),
            torch.arange(direct_prompt_ids.shape[1], device=device).unsqueeze(0),
            direct_positions,
        )
        direct_attention = _direct_attention_audit(
            handle,
            tokenizer,
            example,
            direct_prompt_ids,
            direct_prompt_mask,
            direct_context_positions,
            evidence_spans,
            device,
        )
        representation = representation_audit(
            reference_capture,
            direct_capture,
            source_positions,
            alignment,
        )
        fidelity = cache_fidelity_audit(entry, reference_capture, source_positions)
        if max(
            max(row["post_k_max_abs_error"], row["v_max_abs_error"])
            for row in fidelity
        ) > args.fidelity_tolerance:
            raise AssertionError(f"Cached native K/V fidelity failed for {example['id']}")

        example_artifacts.append(
            {
                "dataset": example["dataset"],
                "example_id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "prompt_construction": "Qwen chat template; Answer briefly and directly; question retained by left truncation",
                "oracle_construction": "annotation text -> exact source char span -> tokenizer span -> every intersecting 32-token parent",
                "answer_leakage": False,
                "span_audit": span_audit,
                "alignment": alignment,
                "direct_attention_by_layer": direct_attention,
                "controls": controls,
                "oracle_last_half": oracle_last_half,
                "oracle_all": oracle_all,
                "representation_by_layer": representation,
                "cache_fidelity_by_layer": fidelity,
            }
        )
        control_rows.extend(controls.values())
        print(
            f"[{number}/{len(examples)}] {example['dataset']} {example['id']} "
            f"direct={controls['exact_evidence']['delta_vs_no_context']:+.3f} "
            f"last14={controls['oracle_last_half']['delta_vs_no_context']:+.3f} "
            f"all={controls['oracle_all']['delta_vs_no_context']:+.3f}",
            flush=True,
        )

    encoding_sweep = _run_encoding_sweep(
        handle, tokenizer, examples, example_artifacts, args, device
    )
    control_aggregates = _aggregate_control_rows(control_rows)
    representation_rows = []
    fidelity_rows = []
    divergence_rows = []
    for example in example_artifacts:
        for row in example["representation_by_layer"]:
            representation_rows.append(
                {"dataset": example["dataset"], "example_id": example["example_id"]} | row
            )
        for row in example["cache_fidelity_by_layer"]:
            fidelity_rows.append(
                {"dataset": example["dataset"], "example_id": example["example_id"]} | row
            )
        for condition in ("oracle_last_half", "oracle_all"):
            for row in example[condition]["hidden_state_delta_by_layer"]:
                divergence_rows.append(
                    {
                        "dataset": example["dataset"],
                        "example_id": example["example_id"],
                        "condition": condition,
                    }
                    | row
                )
    representation_aggregates = []
    for (dataset, layer), values in sorted(
        _group_rows(representation_rows, ("dataset", "layer")).items()
    ):
        representation_aggregates.append(
            {
                "dataset": dataset,
                "layer": layer,
                **{
                    key: _mean([row[key] for row in values])
                    for key in (
                        "aligned_tokens",
                        "pre_k_relative_error",
                        "pre_k_cosine",
                        "post_k_relative_error",
                        "post_k_cosine",
                        "v_relative_error",
                        "v_cosine",
                    )
                },
            }
        )
    attention_rows = []
    for condition in ("oracle_last_half", "oracle_all"):
        attention_rows.extend(
            _aggregate_layer_rows(
                example_artifacts, "attention_support_by_layer", condition
            )
        )
    direct_grouped = defaultdict(list)
    for example in example_artifacts:
        for layer, row in example["direct_attention_by_layer"].items():
            direct_grouped[(example["dataset"], int(layer))].append(row)
    for (dataset, layer), values in sorted(direct_grouped.items()):
        scalar_keys = [
            name for name, value in values[0].items() if not isinstance(value, (dict, list))
        ]
        attention_rows.append(
            {
                "dataset": dataset,
                "condition": "direct_evidence",
                "layer": layer,
            }
            | {name: _mean([row.get(name) for row in values]) for name in scalar_keys}
        )
    divergence_aggregates = []
    for keys, values in sorted(
        _group_rows(divergence_rows, ("dataset", "condition", "layer")).items()
    ):
        divergence_aggregates.append(
            {
                "dataset": keys[0],
                "condition": keys[1],
                "layer": keys[2],
                "relative_l2_delta": _mean([row["relative_l2_delta"] for row in values]),
                "cosine_distance": _mean([row["cosine_distance"] for row in values]),
            }
        )
    historical = _historical_reproduction(control_aggregates)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "Paper 2 oracle-gap audit; fixed oracle payload and frozen Qwen",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_split": "validation",
        "seed": args.seed,
        "example_offset": args.example_offset,
        "examples_per_dataset": args.examples_per_dataset,
        "settings": {
            "prompt_tokens": args.prompt_tokens,
            "direct_text_tokens": args.direct_text_tokens,
            "full_context_tokens": args.full_context_tokens,
            "encoding_block_tokens": args.encoding_block_tokens,
            "routing_chunk_tokens": args.routing_chunk_tokens,
            "routing_chunk_overlap_tokens": 0,
            "memory_tokens": args.memory_tokens,
            "top_k": args.top_k,
            "position_state": "native source-position post-RoPE K; no rebinding",
            "kv_residency": "cpu with layer-native transfer",
            "attention_kernel": "Qwen eager attention",
        },
        "layer_audit": {
            "model_layer_count": layer_count,
            "injected_layer_ids": list(sorted(handle.adapters)),
            "all_layer_consumer_ids": list(all_layers),
            "last_half_consumer_ids": list(last_half),
            "all_layers_verified": tuple(sorted(handle.adapters)) == all_layers,
        },
        "no_answer_leakage": {
            "oracle_inputs": ["evidence annotation text", "source text", "token offsets"],
            "answer_used_for_selection": False,
        },
        "canonical_reproduction": historical,
        "control_aggregates": control_aggregates,
        "representation_aggregates": representation_aggregates,
        "attention_aggregates": attention_rows,
        "divergence_aggregates": divergence_aggregates,
        "encoding_sweep": encoding_sweep,
        "examples": example_artifacts,
        "native_limit_violations": handle.native_limit_violations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle_gap_audit.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _rectangular_csv(args.output_dir / "oracle_gap_controls.csv", control_aggregates)
    _rectangular_csv(
        args.output_dir / "oracle_gap_representation.csv", representation_aggregates
    )
    _rectangular_csv(args.output_dir / "oracle_gap_attention.csv", attention_rows)
    _rectangular_csv(args.output_dir / "oracle_gap_divergence.csv", divergence_aggregates)
    _rectangular_csv(args.output_dir / "oracle_gap_kv_fidelity.csv", fidelity_rows)
    _rectangular_csv(args.output_dir / "oracle_gap_encoding_sweep.csv", encoding_sweep)
    _plots(
        args.output_dir,
        control_aggregates,
        representation_aggregates,
        attention_rows,
        divergence_aggregates,
    )
    _encoding_plot(args.output_dir, encoding_sweep)
    return artifact


def _group_rows(rows: list[dict], keys: tuple[str, ...]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def _historical_reproduction(aggregates: list[dict]) -> dict:
    """Compare canonical conditions with the frozen prior layer-depth artifact."""
    historical_path = (
        ROOT
        / "docs"
        / "papers"
        / "shared"
        / "results"
        / "paper2_hf"
        / "multilayer_pra"
        / "oracle_layer_depth.json"
    )
    if not historical_path.exists():
        return {"available": False}
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    prior = {
        (row["dataset"], row["condition"]): row["gold_mean_logprob_delta_vs_none"]
        for row in historical["aggregates"]
        if row["dataset"] != "combined"
    }
    mapping = {
        "exact_evidence": "direct_text_oracle",
        "oracle_last_half": "oracle_last_half",
        "oracle_all": "oracle_all",
    }
    rows = []
    for row in aggregates:
        prior_name = mapping.get(row["condition"])
        if prior_name is None or (row["dataset"], prior_name) not in prior:
            continue
        expected = prior[(row["dataset"], prior_name)]
        observed = row["delta_vs_no_context"]
        rows.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "historical_delta": expected,
                "reproduced_delta": observed,
                "absolute_difference": abs(observed - expected),
            }
        )
    return {
        "available": True,
        "historical_artifact": str(historical_path.relative_to(ROOT)),
        "rows": rows,
        "max_absolute_difference": max(
            (row["absolute_difference"] for row in rows), default=None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--full-context-tokens", type=int, default=2048)
    parser.add_argument("--generation-control-limit", type=int, default=640)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--encoding-block-tokens", type=int, default=128)
    parser.add_argument("--encoding-sweep", default="32,64,128,256,512")
    parser.add_argument("--sweep-examples-per-dataset", type=int, default=1)
    parser.add_argument("--routing-chunk-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--fidelity-tolerance", type=float, default=1e-5)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs"
        / "papers"
        / "shared"
        / "results"
        / "paper2_hf"
        / "oracle_gap_audit",
    )
    args = parser.parse_args()
    args.encoding_sweep = tuple(
        int(value.strip())
        for value in args.encoding_sweep.split(",")
        if value.strip()
    )
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["control_aggregates"], indent=2, sort_keys=True))
