"""Test whether a frozen supported HF model uses native-K/V evidence once supplied.

The experiment separates selection from use. Learned routing, evidence-oracle
routing, and layer-depth controls all enter the same fixed-selection boundary;
the standard PRA budgeter and Qwen attention kernel handle the resulting K/V.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_smoke import answer_metrics, evidence_token_spans
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import PRAHFConfig, inject_pra, load_hf_routing_projection
from pra_torch.memory import SelectedChunk
from pra_hf import PRARouter


@dataclass(frozen=True)
class Condition:
    """One causal-memory intervention, with absolute decoder layer IDs."""

    name: str
    kind: str
    layers: tuple[int, ...] = ()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[float | None]) -> float | None:
    samples = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(samples) if samples else None


def _prompt(
    tokenizer,
    question: str,
    *,
    context: str | None = None,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Render a bounded chat prompt and return direct-context token positions."""
    content = "Answer briefly and directly."
    context_marker = None
    if context:
        context_marker = f"Context:\n{context}"
        content += f"\n{context_marker}"
    content += f"\nQuestion: {question}"
    if getattr(tokenizer, "chat_template", None):
        template_args = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                enable_thinking=False,
                **template_args,
            )
        except TypeError:
            # Llama templates do not expose Qwen's optional thinking switch.
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], **template_args
            )
    else:
        rendered = content
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_tokens,
    )
    tokenizer.truncation_side = previous
    offsets = encoded.pop("offset_mapping")[0].tolist()
    context_tokens: list[int] = []
    if context_marker:
        marker_start = rendered.rfind(context_marker)
        char_start = marker_start + len("Context:\n")
        char_end = char_start + len(context or "")
        context_tokens = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end
        ]
    return encoded.input_ids, encoded.attention_mask, context_tokens


def _answer_ids(tokenizer, answer: str) -> torch.Tensor:
    ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] == 0:
        raise ValueError("Gold answer produced no tokens.")
    return ids


def _oracle_selections(
    entry,
    layer_id: int,
    evidence_spans: list[tuple[int, int]],
) -> list[SelectedChunk]:
    """Force every parent chunk intersecting at least one annotated evidence span."""
    chunks = [
        chunk
        for chunk in entry.layer_memory[layer_id].chunks
        if any(
            max(chunk.logical_start, start) < min(chunk.logical_end, end)
            for start, end in evidence_spans
        )
    ]
    return [
        SelectedChunk(
            entry=entry,
            chunk=chunk,
            reference_score=1.0,
            chunk_score=1.0 - rank * 1e-6,
            layer_id=layer_id,
            reference_rank=1,
            rank_within_reference=rank + 1,
            metadata={"selection_source": "annotated_evidence_oracle"},
        )
        for rank, chunk in enumerate(chunks)
    ]


def _attention_mass(
    weights: torch.Tensor | None,
    query_positions: list[int],
    key_positions,
) -> float | None:
    if weights is None or not query_positions:
        return None
    valid_queries = [index for index in query_positions if index < weights.shape[-2]]
    if not valid_queries:
        return None
    selected = weights[0, :, valid_queries, :]
    if isinstance(key_positions, slice):
        selected = selected[..., key_positions]
    else:
        valid_keys = [index for index in key_positions if index < weights.shape[-1]]
        if not valid_keys:
            return 0.0
        selected = selected[..., valid_keys]
    return float(selected.float().sum(dim=-1).mean().item())


def _memory_attention_trace(adapter, weights, query_positions, evidence_spans) -> dict:
    memory_width = int(adapter.last_diagnostics.get("hf_memory_width", 0))
    selected = sorted(
        adapter.last_selected_chunks[0],
        key=lambda hit: (hit.reference_uri, hit.token_start, hit.chunk_id),
    )
    cursor = 0
    chunk_rows = []
    evidence_key_positions = []
    for hit in selected:
        length = hit.selected_token_count
        chunk_keys = list(range(cursor, cursor + length))
        chunk_rows.append(
            {
                "chunk_id": hit.chunk_id,
                "logical_span": [hit.logical_start, hit.logical_end],
                "attention_mass": _attention_mass(weights, query_positions, chunk_keys),
            }
        )
        for start, end in evidence_spans:
            overlap_start = max(start, hit.logical_start)
            overlap_end = min(end, hit.logical_end)
            if overlap_start < overlap_end:
                evidence_key_positions.extend(
                    range(
                        cursor + overlap_start - hit.logical_start,
                        cursor + overlap_end - hit.logical_start,
                    )
                )
        cursor += length
    return {
        "memory_attention_mass": _attention_mass(
            weights, query_positions, slice(0, memory_width)
        ),
        "evidence_attention_mass": _attention_mass(
            weights, query_positions, sorted(set(evidence_key_positions))
        ),
        "chunks": chunk_rows,
    }


def _decoder_hidden_tensor(layer_output) -> torch.Tensor:
    """Extract hidden states from common Hugging Face decoder-layer returns."""
    if isinstance(layer_output, torch.Tensor):
        return layer_output
    if isinstance(layer_output, (tuple, list)) and layer_output:
        if isinstance(layer_output[0], torch.Tensor):
            return layer_output[0]
    candidate = getattr(layer_output, "last_hidden_state", None)
    if isinstance(candidate, torch.Tensor):
        return candidate
    raise TypeError(
        f"Unsupported decoder-layer output for diagnostics: {type(layer_output).__name__}"
    )


def _teacher_forced(
    handle,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    answer_ids: torch.Tensor,
    context_tokens: list[int],
    evidence_spans,
    device: torch.device,
    *,
    retain_attention_weights: bool = False,
) -> tuple[dict, list[torch.Tensor]]:
    """Score the gold continuation and capture attention/hidden-state diagnostics."""
    prompt_tokens = int(prompt_ids.shape[1])
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    prediction_positions = list(range(prompt_tokens - 1, full_ids.shape[1] - 1))
    layer_outputs: dict[int, torch.Tensor] = {}
    hooks = [
        layer.register_forward_hook(
            lambda _module, _inputs, output, layer_id=layer_id: layer_outputs.__setitem__(
                layer_id, _decoder_hidden_tensor(output).detach()
            )
        )
        for layer_id, layer in enumerate(handle.model.model.layers)
    ]
    handle.set_attention_diagnostics(True)
    _synchronize(device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            output = handle.model(
                input_ids=full_ids,
                attention_mask=full_mask,
                use_cache=False,
            )
    except Exception:
        handle.set_attention_diagnostics(False)
        raise
    finally:
        for hook in hooks:
            hook.remove()
    _synchronize(device)
    duration = time.perf_counter() - started
    logits = output.logits[:, prediction_positions, :].float()
    targets = answer_ids.to(device)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
    first_logits = logits[0, 0]
    first_target = int(targets[0, 0])
    first_probability = float(first_logits.softmax(dim=-1)[first_target].item())
    first_rank = int((first_logits > first_logits[first_target]).sum().item()) + 1
    competing_logits = first_logits.clone()
    competing_logits[first_target] = float("-inf")
    first_margin = float((first_logits[first_target] - competing_logits.max()).item())
    hidden_signatures = [
        layer_outputs[layer][0, prediction_positions, :].float().mean(dim=0).cpu()
        for layer in range(len(handle.model.model.layers))
    ]
    layer_attention = {}
    for layer_id, adapter in handle.adapters.items():
        if not adapter.memory_enabled:
            continue
        layer_attention[str(layer_id)] = _memory_attention_trace(
            adapter,
            adapter.last_attention_weights,
            prediction_positions,
            evidence_spans,
        )
    direct_mass = None
    if context_tokens:
        direct_mass = _attention_mass(
            handle.adapters[max(handle.adapters)].last_attention_weights,
            prediction_positions,
            context_tokens,
        )
    if not retain_attention_weights:
        handle.set_attention_diagnostics(False)
    return (
        {
            "gold_token_logprobs": token_log_probs.cpu().tolist(),
            "gold_sequence_logprob": float(token_log_probs.sum().item()),
            "gold_mean_token_logprob": float(token_log_probs.mean().item()),
            "gold_first_token_probability": first_probability,
            "gold_first_token_rank": first_rank,
            "gold_first_token_margin": first_margin,
            "teacher_forced_seconds": duration,
            "direct_context_attention_mass": direct_mass,
            "attention_by_layer": layer_attention,
        },
        hidden_signatures,
    )


def _generate(handle, tokenizer, prompt_ids, prompt_mask, device, new_tokens) -> tuple[str, float]:
    encoded = {
        "input_ids": prompt_ids.to(device),
        "attention_mask": prompt_mask.to(device),
    }
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = handle.model.generate(
            **encoded,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
        )
    _synchronize(device)
    continuation = output[0, prompt_ids.shape[1] :]
    return (
        tokenizer.decode(continuation, skip_special_tokens=True).strip(),
        time.perf_counter() - started,
    )


@torch.no_grad()
def _route_once(handle, tokenizer, question, route_layer, prompt_tokens, device) -> dict:
    """Select parent identities once at one layer for native-K/V replay elsewhere."""
    prompt_ids, prompt_mask, _ = _prompt(
        tokenizer, question, max_tokens=prompt_tokens
    )
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    positions = torch.arange(prompt_ids.shape[1], device=device).unsqueeze(0)
    adapter = handle.adapters[route_layer]
    handle.configure_memory_layers(set())
    adapter.begin_capture(positions)
    _synchronize(device)
    query_started = time.perf_counter()
    handle.model(
        input_ids=prompt_ids,
        attention_mask=prompt_mask,
        position_ids=positions,
        use_cache=False,
    )
    _synchronize(device)
    captured = adapter.consume_capture()
    query = adapter._routing_query_states(
        captured.hidden_states, captured.pre_query, captured.post_query
    )
    routing_started = time.perf_counter()
    selected, rankings = adapter.pra_core.route_memory(query)
    _synchronize(device)
    return {
        "selected": selected,
        "rankings": rankings,
        "query_encoding_seconds": routing_started - query_started,
        "routing_seconds": time.perf_counter() - routing_started,
        "routing_layer": route_layer,
    }


def _hidden_deltas(signatures, baseline) -> list[dict]:
    rows = []
    for layer, (current, reference) in enumerate(zip(signatures, baseline)):
        rows.append(
            {
                "layer": layer,
                "relative_l2_delta": float(
                    (current - reference).norm()
                    / reference.norm().clamp_min(1e-12)
                ),
                "cosine_distance": float(1.0 - F.cosine_similarity(current, reference, dim=0)),
            }
        )
    return rows


def _condition_rows(handle, tokenizer, example, entry, evidence_spans, conditions, args, device):
    rows = []
    baseline_hidden = None
    routed = (
        _route_once(
            handle,
            tokenizer,
            example["question"],
            max(handle.adapters),
            args.prompt_tokens,
            device,
        )
        if any(condition.kind == "router" for condition in conditions)
        else None
    )
    oracle_by_layer = {
        layer: [_oracle_selections(entry, layer, evidence_spans)]
        for layer in handle.adapters
    }
    for condition in conditions:
        context = "\n".join(example["evidence"]) if condition.kind == "direct" else None
        prompt_ids, prompt_mask, context_tokens = _prompt(
            tokenizer,
            example["question"],
            context=context,
            max_tokens=args.direct_text_tokens if context else args.prompt_tokens,
        )
        answer_ids = _answer_ids(tokenizer, example["answer"])
        if condition.kind == "oracle":
            fixed = {layer: oracle_by_layer[layer] for layer in condition.layers}
        elif condition.kind == "router":
            if routed is None:
                raise RuntimeError("Learned routing requires a route-once result.")
            fixed = handle.map_chunk_identities_to_layers(
                routed["selected"], condition.layers
            )
        else:
            fixed = None
        handle.configure_memory_layers(set(condition.layers), fixed_selections=fixed)
        scored, hidden = _teacher_forced(
            handle,
            tokenizer,
            prompt_ids,
            prompt_mask,
            answer_ids,
            context_tokens,
            evidence_spans,
            device,
        )
        if condition.name == "no_memory":
            baseline_hidden = hidden
        prediction, generation_seconds = _generate(
            handle, tokenizer, prompt_ids, prompt_mask, device, args.new_tokens
        )
        diagnostics = handle.diagnostics_by_layer()
        selected = {
            str(layer): [hit.as_trace_dict() for hit in adapter.last_selected_chunks[0]]
            for layer, adapter in handle.adapters.items()
            if adapter.memory_enabled
        }
        memory_tokens = sum(
            int(values.get("retrieved_physical_kv_tokens", 0))
            for layer, values in diagnostics.items()
            if layer in condition.layers
        )
        budget_rejections = sum(
            int(values.get("chunks_budget_rejected", 0))
            for layer, values in diagnostics.items()
            if layer in condition.layers
        )
        if condition.kind == "oracle":
            evidence_guaranteed = all(
                {hit.chunk_id for hit in oracle_by_layer[layer][0]}.issubset(
                    {hit["chunk_id"] for hit in selected.get(str(layer), [])}
                )
                for layer in condition.layers
            )
        else:
            evidence_guaranteed = condition.kind == "direct"
        memory_mass = _mean(
            [
                trace["memory_attention_mass"]
                for trace in scored["attention_by_layer"].values()
            ]
        )
        evidence_mass = _mean(
            [
                trace["evidence_attention_mass"]
                for trace in scored["attention_by_layer"].values()
            ]
        )
        rows.append(
            {
                "dataset": example["dataset"],
                "example_id": example["id"],
                "condition": condition.name,
                "kind": condition.kind,
                "pra_layers": list(condition.layers),
                "question": example["question"],
                "answer": example["answer"],
                "answer_tokens": int(answer_ids.shape[1]),
                "prompt_tokens": int(prompt_ids.shape[1]),
                "generated_answer": prediction,
                "generation_seconds": generation_seconds,
                "active_memory_kv_tokens": memory_tokens,
                "evidence_parent_chunks": max(
                    (len(oracle_by_layer[layer][0]) for layer in condition.layers),
                    default=0,
                ),
                "evidence_guaranteed": evidence_guaranteed,
                "chunks_budget_rejected": budget_rejections,
                "native_limit_violations": handle.native_limit_violations,
                "route_once_query_encoding_seconds": (
                    routed["query_encoding_seconds"]
                    if routed is not None and condition.kind == "router"
                    else 0.0
                ),
                "route_once_routing_seconds": (
                    routed["routing_seconds"]
                    if routed is not None and condition.kind == "router"
                    else 0.0
                ),
                "route_once_routing_layer": (
                    routed["routing_layer"]
                    if routed is not None and condition.kind == "router"
                    else None
                ),
                "memory_attention_mass": memory_mass,
                "evidence_attention_mass": evidence_mass,
                "selected_by_layer": selected,
                "diagnostics_by_layer": diagnostics,
                "hidden_state_delta_by_layer": (
                    []
                    if baseline_hidden is None or condition.kind in {"none", "direct"}
                    else _hidden_deltas(hidden, baseline_hidden)
                ),
                **scored,
                **answer_metrics(prediction, example["answer"]),
            }
        )
    baseline = next(row for row in rows if row["condition"] == "no_memory")
    for row in rows:
        row["gold_logprob_delta_vs_none"] = (
            row["gold_sequence_logprob"] - baseline["gold_sequence_logprob"]
        )
        row["gold_mean_logprob_delta_vs_none"] = (
            row["gold_mean_token_logprob"] - baseline["gold_mean_token_logprob"]
        )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for dataset in ("combined", "hotpotqa", "qasper"):
        for condition in dict.fromkeys(row["condition"] for row in rows):
            values = [
                row
                for row in rows
                if row["condition"] == condition
                and (dataset == "combined" or row["dataset"] == dataset)
            ]
            output.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "examples": len(values),
                    **{
                        metric: _mean([row.get(metric) for row in values])
                        for metric in (
                            "gold_sequence_logprob",
                            "gold_mean_token_logprob",
                            "gold_logprob_delta_vs_none",
                            "gold_mean_logprob_delta_vs_none",
                            "gold_first_token_probability",
                            "gold_first_token_rank",
                            "memory_attention_mass",
                            "evidence_attention_mass",
                            "direct_context_attention_mass",
                            "f1",
                            "em",
                            "answer_contained",
                            "active_memory_kv_tokens",
                            "evidence_parent_chunks",
                            "evidence_guaranteed",
                            "chunks_budget_rejected",
                            "native_limit_violations",
                            "teacher_forced_seconds",
                            "generation_seconds",
                            "route_once_query_encoding_seconds",
                            "route_once_routing_seconds",
                        )
                    },
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(aggregates: list[dict], output_dir: Path) -> None:
    rows = [row for row in aggregates if row["dataset"] in {"hotpotqa", "qasper"}]
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    lookup = {(row["dataset"], row["condition"]): row for row in rows}
    x = list(range(len(conditions)))
    width = 0.38
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for offset, dataset, color in (
        (-width / 2, "hotpotqa", "#2f6b9a"),
        (width / 2, "qasper", "#d9782d"),
    ):
        axes[0].bar(
            [value + offset for value in x],
            [
                lookup[(dataset, condition)]["gold_mean_logprob_delta_vs_none"]
                for condition in conditions
            ],
            width,
            label=dataset,
            color=color,
        )
        axes[1].bar(
            [value + offset for value in x],
            [
                lookup[(dataset, condition)]["memory_attention_mass"] or 0.0
                for condition in conditions
            ],
            width,
            label=dataset,
            color=color,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Gold mean-token log-probability delta")
    axes[1].set_ylabel("Attention mass on PRA memory")
    for axis in axes:
        axis.set_xticks(x, conditions, rotation=28, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"oracle_memory_use.{suffix}", dpi=180)
    plt.close(figure)


def run(args) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    generation_compile_disabled = False
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 7:
        model.generation_config.disable_compile = True
        generation_compile_disabled = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = (
        PRARouter.from_pretrained(args.checkpoint, device=device)
        if args.checkpoint.is_dir()
        else load_hf_routing_projection(args.checkpoint, device=device)
    )
    layer_count = int(model.config.num_hidden_layers)
    layer_types = tuple(getattr(model.config, "layer_types", ()) or ())
    eligible_layers = (
        tuple(
            layer_id
            for layer_id, layer_type in enumerate(layer_types)
            if layer_type == "full_attention"
        )
        if layer_types
        else tuple(range(layer_count))
    )
    if not eligible_layers:
        raise ValueError("The host model exposes no PRA-eligible attention layers.")
    last_eight = eligible_layers[-8:]
    last_four = last_eight[-4:]
    learned_layers = last_eight[-min(args.learned_depth, len(last_eight)) :]
    early_late = (
        eligible_layers[len(eligible_layers) // 3],
        eligible_layers[-1],
    )
    requested = set(args.conditions)
    required_layers = [*last_four, *early_late]
    if "learned_router" in requested:
        required_layers.extend(learned_layers)
    if "oracle_last_8" in requested:
        required_layers.extend(last_eight)
    injected_layers = tuple(sorted(set(required_layers)))
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=injected_layers,
            model_max_context_tokens=args.native_tokens,
            max_prompt_direct_tokens=args.prompt_tokens,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=3,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    conditions = (
        Condition("no_memory", "none"),
        Condition("learned_router", "router", learned_layers),
        Condition("oracle_last_1", "oracle", (eligible_layers[-1],)),
        Condition("oracle_last_2", "oracle", last_four[-2:]),
        Condition("oracle_last_4", "oracle", last_four),
        Condition("oracle_last_8", "oracle", last_eight),
        Condition("oracle_early_late", "oracle", early_late),
        Condition("direct_text_oracle", "direct"),
    )
    conditions = tuple(condition for condition in conditions if condition.name in requested)
    if "no_memory" not in {condition.name for condition in conditions}:
        raise ValueError("no_memory is required to compute paired deltas.")
    examples = load_split_examples(
        args.cache_dir, args.examples_per_dataset, args.example_offset, args.seed
    )
    rows = []
    for index, example in enumerate(examples, start=1):
        handle.cache.clear()
        source = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
        entry = handle.add_reference(
            f"benchmark://{example['dataset']}/{example['id']}",
            source,
            text=example["source"],
        )
        example_rows = _condition_rows(
            handle, tokenizer, example, entry, spans, conditions, args, device
        )
        rows.extend(example_rows)
        print(
            f"[{index}/{len(examples)}] {example['dataset']} {example['id']} "
            + " ".join(
                f"{row['condition']}={row['gold_mean_logprob_delta_vs_none']:+.3f}"
                for row in example_rows
            ),
            flush=True,
        )
    aggregates = _aggregate(rows)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "frozen HF causal memory-use intervention; teacher-forced and generated QA",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "generation_compile_disabled": generation_compile_disabled,
        "eligible_attention_layers": list(eligible_layers),
        "checkpoint": str(args.checkpoint.resolve().relative_to(ROOT)),
        "adapter_parameters": projection.parameter_count,
        "seed": args.seed,
        "example_offset": args.example_offset,
        "examples_per_dataset": args.examples_per_dataset,
        "routing_chunk_tokens": 32,
        "top_k_chunks": 3,
        "routing_policy": "route once at the final layer; replay parent identities with target-layer native K/V",
        "learned_depth": args.learned_depth,
        "native_context_tokens": args.native_tokens,
        "memory_budget_tokens": args.memory_tokens,
        "injected_layers": list(injected_layers),
        "conditions": [condition.__dict__ for condition in conditions],
        "rows": rows,
        "aggregates": aggregates,
        "max_native_operation_tokens": handle.max_native_operation_tokens,
        "native_limit_violations": handle.native_limit_violations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "oracle_memory_use.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    flat_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    _write_csv(args.output_dir / "oracle_memory_use.csv", flat_rows)
    _write_csv(args.output_dir / "oracle_memory_use_aggregate.csv", aggregates)
    _plot(aggregates, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    result_dir = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--learned-depth", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            result_dir
            / "routing"
            / "learned_adapter"
            / "checkpoints"
            / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt"
        ),
    )
    parser.add_argument(
        "--conditions",
        default=(
            "no_memory,learned_router,oracle_last_1,oracle_last_2,"
            "oracle_last_4,oracle_early_late,direct_text_oracle"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=result_dir / "oracle_memory_use")
    args = parser.parse_args()
    args.conditions = tuple(value.strip() for value in args.conditions.split(",") if value.strip())
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))
