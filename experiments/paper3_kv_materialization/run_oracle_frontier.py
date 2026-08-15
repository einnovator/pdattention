"""Measure frozen-model output as oracle-selected native K/V detail is reduced.

Gate 1 fixes annotated evidence identities and varies only disclosure. Gate 2
reuses frozen Paper-2.5 selected spans and compares whole routing parents with
smaller logical intervals. Every row is checkpointed before the next decode.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_hf import PRAConfig, PRAForCausalLM
from pra_hf.natural_reasoning_graph import load_2wiki, load_musique
from pra_hf.output_validation import (
    MATERIALIZATION_BANDS,
    deterministic_answer_metrics,
    fixed_chunks_for_spans,
    merge_spans,
    selected_span_metrics,
)
from pra_torch.materialization import (
    LogicalDomainBounds,
    LogicalInterval,
    allocate_interval_budget,
    evidence_centered_interval,
    union_intervals,
)


RADII = (0, 4, 8, 16, 32, 64)
FIXED_BUDGETS = (64, 128, 256)
SELECTORS = ("one_shot", "graph_sparse", "graph_balanced", "graph_high")


@dataclass(frozen=True)
class Policy:
    """One frozen disclosure condition independent of conceptual selection."""

    name: str
    mode: str
    radius_left: int | None = None
    radius_right: int | None = None
    kv_budget: int | None = None
    allocation: str | None = None
    direct_full_context: bool = False


class _TokenClock(StoppingCriteria):
    def __init__(self, device: torch.device):
        self.device = device
        self.timestamps: list[float] = []

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timestamps.append(time.perf_counter())
        return False


def _policies(
    phase: str,
    selected_radius: int | None = None,
    *,
    study: str = "pilot",
) -> tuple[Policy, ...]:
    if study == "confirmation":
        if phase == "validation":
            return (
                Policy("M_none", "none"),
                Policy("M0_native_gist", "native_gist_only"),
                Policy("M1_whole_parent", "selected_chunks"),
                *(
                    Policy(
                        f"M3_radius_{radius}",
                        "logical_intervals",
                        radius,
                        radius,
                    )
                    for radius in (0, 2, 4, 8)
                ),
            )
        if phase == "heldout":
            if selected_radius is None:
                raise ValueError("heldout confirmation requires a validation-selected radius")
            selected = (
                Policy(
                    f"M3_selected_radius_{selected_radius}",
                    "logical_intervals",
                    selected_radius,
                    selected_radius,
                ),
            ) if selected_radius not in {0, 2} else ()
            return (
                Policy("M_none", "none"),
                Policy("M0_native_gist", "native_gist_only"),
                Policy("M1_whole_parent", "selected_chunks"),
                Policy("M2_evidence_only", "logical_intervals", 0, 0),
                Policy("M3_radius_2", "logical_intervals", 2, 2),
                *selected,
            )
        raise ValueError(f"unsupported phase: {phase}")
    if study != "pilot":
        raise ValueError(f"unsupported study: {study}")
    if phase == "validation":
        return (
            Policy("M_none", "none"),
            Policy("M0_native_gist", "native_gist_only"),
            Policy("M1_whole_parent", "selected_chunks"),
            *(Policy(f"M3_radius_{radius}", "logical_intervals", radius, radius) for radius in RADII),
            Policy("M5_asymmetric_l8_r32", "logical_intervals", 8, 32),
            *(
                Policy(
                    f"M6_budget_{budget}_equal",
                    "logical_intervals",
                    64,
                    64,
                    budget,
                    "equal",
                )
                for budget in FIXED_BUDGETS
            ),
            Policy("M7_gist_local_r16", "gist_plus_logical_intervals", 16, 16),
        )
    if phase == "heldout":
        if selected_radius is None:
            raise ValueError("heldout requires a validation-selected radius")
        return (
            Policy("M_none", "none"),
            Policy("M0_native_gist", "native_gist_only"),
            Policy("M1_whole_parent", "selected_chunks"),
            Policy("M2_evidence_only", "logical_intervals", 0, 0),
            Policy(
                f"M3_selected_radius_{selected_radius}",
                "logical_intervals",
                selected_radius,
                selected_radius,
            ),
            Policy("M6_budget_128_equal", "logical_intervals", 64, 64, 128, "equal"),
            Policy(
                f"M7_gist_selected_radius_{selected_radius}",
                "gist_plus_logical_intervals",
                selected_radius,
                selected_radius,
            ),
        )
    raise ValueError(f"unsupported phase: {phase}")


def _prompt(tokenizer, question: str, *, context: str | None = None):
    content = "Return only the brief answer, without explanation."
    if context is not None:
        content += f"\nContext:\n{context}"
    content += f"\nQuestion: {question.strip()}"
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        rendered = content + "\nAnswer:"
    return tokenizer(rendered, return_tensors="pt", add_special_tokens=True)


def _extract_answer(text: str) -> str:
    clean = str(text).replace("<think>", "").replace("</think>", "").strip()
    for line in clean.splitlines():
        line = line.strip()
        if line:
            return line.split(":", 1)[1].strip() if line.casefold().startswith("answer:") else line
    return clean


def _intervals(uri: str, spans, source_tokens: int, policy: Policy) -> list[LogicalInterval]:
    bounds = LogicalDomainBounds(uri, 0, source_tokens)
    values = [
        evidence_centered_interval(
            uri,
            int(start),
            int(end),
            radius_left=int(policy.radius_left or 0),
            radius_right=int(policy.radius_right or 0),
            bounds=bounds,
        )
        for start, end in spans
        if int(end) > int(start)
    ]
    if policy.kv_budget is not None:
        values = allocate_interval_budget(
            values,
            total_budget=policy.kv_budget,
            strategy=str(policy.allocation),
            minimum_per_region=1,
        )
    return values


def _attach_intervals(selected, intervals):
    if not selected:
        raise ValueError("oracle evidence did not map to a routing parent")
    payload = [
        {
            "domain": interval.domain,
            "start": interval.start,
            "end": interval.end,
            "evidence_start": interval.evidence_start,
            "evidence_end": interval.evidence_end,
            "score": interval.score,
        }
        for interval in intervals
    ]
    return [
        replace(
            hit,
            metadata={
                **hit.metadata,
                **({"materialization_intervals": payload} if index == 0 else {}),
            },
        )
        for index, hit in enumerate(selected)
    ]


def _materialized_positions(policy: Policy, selected, intervals) -> list[int | None]:
    if policy.mode == "native_gist_only":
        return [None] * len(selected)
    if policy.mode in {"logical_intervals", "gist_plus_logical_intervals"}:
        positions: list[int | None] = []
        if policy.mode.startswith("gist_plus"):
            positions.extend([None] * len(selected))
        for interval in union_intervals(intervals):
            positions.extend(range(interval.start, interval.end))
        return positions
    positions = []
    seen: set[int] = set()
    for hit in sorted(selected, key=lambda item: (item.source_uri, item.logical_start, item.chunk_id)):
        for position in range(hit.logical_start, hit.logical_end):
            if position not in seen:
                seen.add(position)
                positions.append(position)
    return positions


def _attention_metrics(pra, answer_query_positions, memory_positions, evidence_spans, layers):
    evidence = {
        position
        for start, end in evidence_spans
        for position in range(int(start), int(end))
    }
    rows = []
    for layer in layers:
        adapter = pra._handle.adapters[layer]
        weights = adapter.last_attention_weights
        if weights is None:
            continue
        width = int(adapter.last_diagnostics.get("hf_memory_width", 0))
        if width != len(memory_positions):
            raise AssertionError(
                f"layer {layer}: attention memory width {width} != provenance {len(memory_positions)}"
            )
        queries = [index for index in answer_query_positions if index < weights.shape[-2]]
        selected = weights[0, :, queries, :].float()
        memory = selected[..., :width]
        evidence_indices = [
            index for index, position in enumerate(memory_positions) if position in evidence
        ]
        non_evidence_indices = [
            index for index, position in enumerate(memory_positions) if position not in evidence
        ]
        probability = selected.clamp_min(1e-12)
        rows.append(
            {
                "memory_attention_mass": float(memory.sum(dim=-1).mean().cpu()),
                "evidence_attention_mass": float(
                    memory[..., evidence_indices].sum(dim=-1).mean().cpu()
                ) if evidence_indices else 0.0,
                "non_evidence_attention_mass": float(
                    memory[..., non_evidence_indices].sum(dim=-1).mean().cpu()
                ) if non_evidence_indices else 0.0,
                "attention_entropy": float(
                    (-(probability * probability.log()).sum(dim=-1)).mean().cpu()
                ),
            }
        )
    return {
        key: statistics.fmean(row[key] for row in rows) if rows else None
        for key in (
            "memory_attention_mass",
            "evidence_attention_mass",
            "non_evidence_attention_mass",
            "attention_entropy",
        )
    }


def _teacher_forced(pra, tokenizer, encoded, answer: str, device, memory_positions, evidence_spans, layers):
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    answer_ids = answer_ids.to(encoded.input_ids.device)
    prompt_tokens = int(encoded.input_ids.shape[1])
    full_ids = torch.cat((encoded.input_ids, answer_ids), dim=1).to(device)
    full_mask = torch.ones_like(full_ids)
    prediction_positions = list(range(prompt_tokens - 1, full_ids.shape[1] - 1))
    pra._handle.set_attention_diagnostics(bool(layers))
    started = time.perf_counter()
    with torch.no_grad():
        output = pra.model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    logits = output.logits[:, prediction_positions, :].float()
    targets = answer_ids.to(device)
    token_logprobs = F.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)[0]
    metrics = _attention_metrics(
        pra, prediction_positions, memory_positions, evidence_spans, layers
    ) if layers else {
        "memory_attention_mass": None,
        "evidence_attention_mass": None,
        "non_evidence_attention_mass": None,
        "attention_entropy": None,
    }
    diagnostics = pra._handle.diagnostics_by_layer()
    pra._handle.set_attention_diagnostics(False)
    return {
        "gold_sequence_logprob": float(token_logprobs.sum().cpu()),
        "gold_mean_token_logprob": float(token_logprobs.mean().cpu()),
        "teacher_forced_seconds": duration,
        "answer_tokens": int(answer_ids.shape[1]),
        "diagnostics_by_layer": diagnostics,
        **metrics,
    }


def _generate(model, tokenizer, encoded, device, max_new_tokens: int):
    encoded = encoded.to(device)
    clock = _TokenClock(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([clock]),
            disable_compile=(device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 7),
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ended = time.perf_counter()
    generated = output[0, encoded.input_ids.shape[1] :]
    intervals = [right - left for left, right in zip(clock.timestamps, clock.timestamps[1:])]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        "generated_answer": _extract_answer(raw),
        "raw_answer": raw,
        "generated_tokens": int(generated.numel()),
        "ttft_seconds": clock.timestamps[0] - started if clock.timestamps else ended - started,
        "tpot_seconds": statistics.fmean(intervals) if intervals else 0.0,
        "total_generation_seconds": ended - started,
        "tokens_per_second": int(generated.numel()) / max(ended - started, 1e-12),
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    }


def _mean(values):
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def _aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["dataset"], row["condition"])].append(row)
    metrics = (
        "exact_match", "token_f1", "answer_contained", "normalized_answer_accuracy",
        "gold_mean_token_logprob", "gold_mean_logprob_delta_vs_none",
        "materialized_unique_tokens", "native_kv_token_states", "native_kv_bytes",
        "kv_reduction_vs_whole", "evidence_kv_tokens", "non_evidence_kv_tokens",
        "evidence_density", "memory_attention_mass", "evidence_attention_mass",
        "non_evidence_attention_mass", "attention_entropy", "ttft_seconds",
        "tpot_seconds", "total_generation_seconds", "interval_resolution_seconds",
        "logical_gather_seconds", "logical_h2d_seconds", "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes", "cross_shard_interval_count",
        "conceptual_selected_parents", "evidence_recall",
        "complete_evidence_recovery", "annotated_edge_recall",
        "evidence_source_tokens", "evidence_coverage",
        "requested_materialization_tokens", "deduplicated_materialization_tokens",
        "active_kv_fraction", "cpu_reference_cache_bytes", "gpu_reference_cache_bytes",
        "h2d_kv_bytes", "encoding_granularity_tokens",
    )
    return [
        {
            "phase": key[0], "dataset": key[1], "condition": key[2], "examples": len(values),
            **{metric: _mean([row.get(metric) for row in values]) for metric in metrics},
        }
        for key, values in sorted(grouped.items())
    ]


def _write_csv(path: Path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    value.rstrip().replace("\r", r"\r").replace("\n", r"\n")
                    if isinstance(value, str)
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        )


def _append(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _load_checkpoint(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_radius(aggregates):
    selected = {}
    for dataset in ("musique", "2wikimultihopqa"):
        rows = {row["condition"]: row for row in aggregates if row["dataset"] == dataset}
        whole = rows["M1_whole_parent"]["gold_mean_token_logprob"]
        candidates = [
            row
            for name, row in rows.items()
            if name.startswith("M3_radius_")
        ]
        candidates.sort(key=lambda row: int(row["condition"].rsplit("_", 1)[1]))
        feasible = [row for row in candidates if row["gold_mean_token_logprob"] >= whole - 0.05]
        winner = min(
            feasible or candidates,
            key=lambda row: (
                row["materialized_unique_tokens"] if feasible else -row["gold_mean_token_logprob"],
                int(row["condition"].rsplit("_", 1)[1]),
            ),
        )
        selected[dataset] = int(winner["condition"].rsplit("_", 1)[1])
    return {
        "selection_partition": "validation",
        "criterion": "smallest radius within 0.05 mean gold-token nats of whole-parent; otherwise best log-probability",
        "selected_radius": selected,
    }


def _plots(aggregates, output_dir: Path):
    validation = [row for row in aggregates if row["phase"] == "validation"]
    if validation:
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.3))
        colors = {"musique": "#32688f", "2wikimultihopqa": "#c85c3d"}
        for dataset, color in colors.items():
            rows = [row for row in validation if row["dataset"] == dataset and row["condition"].startswith("M3_radius_")]
            rows.sort(key=lambda row: int(row["condition"].rsplit("_", 1)[1]))
            radii = [int(row["condition"].rsplit("_", 1)[1]) for row in rows]
            axes[0].plot([row["materialized_unique_tokens"] for row in rows], [row["gold_mean_token_logprob"] for row in rows], marker="o", label=dataset, color=color)
            axes[1].plot(radii, [row["gold_mean_token_logprob"] for row in rows], marker="o", label=dataset, color=color)
            axes[2].plot([row["materialized_unique_tokens"] for row in rows], [row["evidence_density"] for row in rows], marker="o", label=dataset, color=color)
        axes[0].set(xlabel="Materialized unique K/V tokens", ylabel="Gold mean-token log probability")
        axes[1].set(xlabel="Context radius", ylabel="Gold mean-token log probability")
        axes[2].set(xlabel="Materialized unique K/V tokens", ylabel="Evidence density")
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[0].legend()
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(output_dir / f"oracle_materialization_frontier.{suffix}", dpi=180)
        plt.close(figure)


def _examples(args, discovery_rows):
    all_examples = load_musique(args.musique_dev) + load_2wiki(args.twowiki_dev)
    by_id = {example.example_id: example for example in all_examples}
    partition = "validation" if args.phase == "validation" else "test"
    opposite = "test" if partition == "validation" else "validation"
    ids = {}
    for dataset in ("musique", "2wikimultihopqa"):
        selected = sorted({
            row["example_id"] for row in discovery_rows
            if row["dataset"] == dataset and row["partition"] == partition
        })[: args.examples_per_dataset]
        reserved = {
            row["example_id"] for row in discovery_rows
            if row["dataset"] == dataset and row["partition"] == opposite
        }
        if len(selected) < args.examples_per_dataset:
            extension = sorted(
                example.example_id
                for example in all_examples
                if example.dataset == dataset
                and example.example_id not in reserved
                and example.example_id not in selected
            )
            selected.extend(extension[: args.examples_per_dataset - len(selected)])
        ids[dataset] = selected
    return [by_id[identity] for dataset in ids for identity in ids[dataset]]


def _annotation_geometry(tokenizer, example):
    """Map dataset-truth character spans to the exact reference-token domain."""
    encoded = tokenizer(
        example.source,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    spans = []
    for node in example.nodes:
        if node.text_span is None:
            continue
        char_start, char_end = map(int, node.text_span)
        positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if int(end) > char_start and int(start) < char_end
        ]
        if positions:
            spans.append((positions[0], positions[-1] + 1))
    spans = sorted(set(spans))
    if not spans:
        raise ValueError(f"No mapped annotation evidence for {example.dataset}/{example.example_id}")
    return {
        "evidence_token_spans": spans,
        "source_tokens": len(encoded["input_ids"]),
        "geometry_source": "dataset_annotation_character_spans",
    }


def _row_metrics(diagnostics, layers):
    active = [diagnostics[layer] for layer in layers]
    if not active:
        return {key: 0.0 for key in (
            "materialized_unique_tokens", "native_kv_token_states", "native_kv_bytes",
            "interval_resolution_seconds", "logical_gather_seconds", "logical_h2d_seconds",
            "cross_shard_interval_count",
        )}
    unique = int(round(float(active[0].get("materialized_native_kv_tokens", active[0].get("memory_tokens_materialized", 0)))))
    return {
        "materialized_unique_tokens": unique,
        "native_kv_token_states": unique * len(layers),
        "native_kv_bytes": sum(
            float(
                row.get(
                    "materialized_native_kv_bytes",
                    row.get("retrieved_kv_transfer_bytes", 0),
                )
            )
            for row in active
        ),
        "interval_resolution_seconds": sum(float(row.get("interval_resolution_seconds", 0)) for row in active),
        "logical_gather_seconds": sum(
            float(row.get("logical_gather_seconds", row.get("materialization_duration_seconds", 0)))
            for row in active
        ),
        "logical_h2d_seconds": sum(
            float(row.get("logical_h2d_seconds", row.get("selected_kv_transfer_duration_seconds", 0)))
            for row in active
        ),
        "cross_shard_interval_count": sum(float(row.get("cross_shard_interval_count", 0)) for row in active),
    }


def run(args):
    device = torch.device(args.device)
    discovery_manifest = json.loads(args.discovery.read_text(encoding="utf-8"))
    discovery_rows = discovery_manifest["rows"]
    discovery = {(row["dataset"], row["example_id"], row["selection"]): row for row in discovery_rows}
    selection = json.loads(args.policy_selection.read_text(encoding="utf-8")) if args.policy_selection.exists() else None
    if args.phase == "heldout" and selection is None:
        raise ValueError("run validation before heldout")
    selected_radius = selection["selected_radius"] if selection else {}
    bands = {band.name: band for band in MATERIALIZATION_BANDS}
    inherited = json.loads(args.band_selection.read_text(encoding="utf-8"))["selected_bands"]
    examples = _examples(args, discovery_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = getattr(args, "artifact_prefix", "oracle_frontier")
    study = getattr(args, "study", "pilot")
    checkpoint = args.output_dir / f"{artifact_prefix}_{args.phase}_checkpoint.jsonl"
    rows = _load_checkpoint(checkpoint)
    completed = {(row["dataset"], row["example_id"], row["condition"]) for row in rows}

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, attn_implementation="eager",
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pra = PRAForCausalLM.from_model(
        model,
        tokenizer,
        pra_config=PRAConfig(
            routing_layer=27,
            consumption_layers=tuple(range(28)),
            chunk_tokens=args.parent_tokens,
            selected_fraction=None,
            top_k=1,
            max_direct_context=args.prompt_tokens,
            native_operation_limit=args.native_limit,
            max_materialized_tokens=args.native_limit - args.prompt_tokens,
            context_safety_reserve_tokens=0,
            encoding_block_tokens=args.encoding_tokens,
            reference_device="cpu",
            pin_reference_memory=device.type == "cuda",
            non_blocking_transfer=device.type == "cuda",
        ),
    )

    for example_index, example in enumerate(examples, start=1):
        dataset = example.dataset
        policies = _policies(
            args.phase,
            selected_radius.get(dataset),
            study=study,
        )
        pending = [policy for policy in policies if (dataset, example.example_id, policy.name) not in completed]
        if not pending:
            continue
        canonical = discovery.get((dataset, example.example_id, "oracle_evidence"))
        if canonical is None:
            canonical = _annotation_geometry(tokenizer, example)
        evidence_spans = [tuple(map(int, span)) for span in canonical["evidence_token_spans"]]
        source_tokens = int(canonical["source_tokens"])
        band = bands[inherited[dataset]]
        pra.clear_references()
        uri = f"benchmark://{dataset}/{example.example_id}"
        pra.add_reference(example.source, uri=uri)
        entry = pra._handle.cache.get(uri)
        if entry is None:
            raise AssertionError("reference cache entry missing after encoding")
        oracle_selected = fixed_chunks_for_spans(
            entry, routing_layer=pra.routing_layer, selected_spans=evidence_spans,
            selection_name="oracle_evidence",
        )
        whole_tokens = len({position for hit in oracle_selected for position in range(hit.logical_start, hit.logical_end)})

        for policy in pending:
            interval_plan = []
            layers = () if policy.mode == "none" else band.layers
            encoded = _prompt(tokenizer, example.question, context=example.source if policy.direct_full_context else None)
            if encoded.input_ids.shape[1] > args.native_limit:
                raise ValueError("direct full-context control exceeds native limit")
            if layers:
                interval_plan = _intervals(uri, evidence_spans, source_tokens, policy) if policy.mode in {"logical_intervals", "gist_plus_logical_intervals"} else []
                selected = _attach_intervals(oracle_selected, interval_plan) if interval_plan else oracle_selected
                mapped = pra._handle.map_chunk_identities_to_layers([selected], layers)
                pra._handle.pra_config.detail_materialization = policy.mode
                pra._handle.configure_memory_layers(set(layers), fixed_selections=mapped)
                positions = _materialized_positions(policy, selected, interval_plan)
            else:
                pra._handle.configure_memory_layers(set())
                positions = []
            teacher = _teacher_forced(
                pra, tokenizer, encoded, example.answer, device, positions, evidence_spans, layers
            )
            generation = _generate(pra.model, tokenizer, encoded, device, args.max_new_tokens)
            physical = _row_metrics(teacher["diagnostics_by_layer"], layers)
            span_metrics = selected_span_metrics(
                [tuple(position for position in (interval.start, interval.end)) for interval in union_intervals(interval_plan)]
                if interval_plan else (
                    [(hit.logical_start, hit.logical_end) for hit in oracle_selected]
                    if layers and policy.mode != "native_gist_only" else []
                ),
                evidence_spans,
                source_tokens,
            )
            evidence_tokens = int(span_metrics["evidence_kv_tokens"])
            non_evidence_tokens = max(0, int(physical["materialized_unique_tokens"]) - evidence_tokens)
            row = {
                "phase": args.phase,
                "dataset": dataset,
                "example_id": example.example_id,
                "question_type": example.question_type,
                "annotated_hops": example.annotated_hops,
                "condition": policy.name,
                "selection_policy": "annotated_evidence_oracle" if layers else "none",
                "oracle_labels_used": bool(layers),
                "geometry_source": canonical.get(
                    "geometry_source", "frozen_paper2_5_discovery_manifest"
                ),
                "materialization_policy": policy.mode,
                "radius_left": policy.radius_left,
                "radius_right": policy.radius_right,
                "kv_budget": policy.kv_budget,
                "budget_allocation": policy.allocation,
                "materialization_band": band.name if layers else "none",
                "materialization_layers": list(layers),
                "logical_source_tokens": source_tokens,
                "encoding_granularity_tokens": args.encoding_tokens,
                "cpu_reference_cache_bytes": source_tokens * int(pra.model.config.num_hidden_layers) * 2 * int(pra.model.config.num_key_value_heads) * int(pra.model.config.head_dim) * 2,
                "gpu_reference_cache_bytes": 0,
                "conceptual_selected_parents": len(oracle_selected) if layers else 0,
                "requested_materialization_tokens": sum(interval.token_count for interval in interval_plan) if interval_plan else (len(oracle_selected) if policy.mode == "native_gist_only" else whole_tokens if layers else 0),
                "deduplicated_materialization_tokens": int(physical["materialized_unique_tokens"]),
                "active_kv_fraction": int(physical["materialized_unique_tokens"]) / max(source_tokens, 1),
                "h2d_kv_bytes": physical["native_kv_bytes"],
                "whole_parent_tokens": whole_tokens,
                "kv_reduction_vs_whole": 1.0 - int(physical["materialized_unique_tokens"]) / max(whole_tokens, 1) if layers else 1.0,
                "evidence_kv_tokens": evidence_tokens,
                "evidence_source_tokens": int(span_metrics["evidence_source_tokens"]),
                "evidence_coverage": evidence_tokens / max(int(span_metrics["evidence_source_tokens"]), 1),
                "non_evidence_kv_tokens": non_evidence_tokens,
                "evidence_density": evidence_tokens / max(int(physical["materialized_unique_tokens"]), 1),
                "intervals": [[interval.start, interval.end] for interval in interval_plan],
                "reference_answer": example.answer,
                **physical,
                **teacher,
                **generation,
                **deterministic_answer_metrics(generation["generated_answer"], example.answer),
            }
            row.pop("diagnostics_by_layer", None)
            _append(checkpoint, row)
            rows.append(row)
            completed.add((dataset, example.example_id, policy.name))
        # Loop-local selected hits retain their owning cache entry. Release those
        # references before encoding the next source so long cohorts do not hold
        # two complete layer-native caches at once on memory-constrained GPUs.
        pra.clear_references()
        entry = oracle_selected = selected = mapped = None
        teacher = generation = encoded = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[{args.phase} {example_index}/{len(examples)}] {dataset} {example.example_id} rows={len(pending)}", flush=True)

    baselines = {(row["dataset"], row["example_id"]): row for row in rows if row["condition"] == "M_none"}
    whole = {(row["dataset"], row["example_id"]): row for row in rows if row["condition"] == "M1_whole_parent"}
    for row in rows:
        baseline = baselines[(row["dataset"], row["example_id"])]
        row["gold_mean_logprob_delta_vs_none"] = row["gold_mean_token_logprob"] - baseline["gold_mean_token_logprob"]
        if (row["dataset"], row["example_id"]) in whole:
            row["gold_mean_logprob_delta_vs_whole"] = row["gold_mean_token_logprob"] - whole[(row["dataset"], row["example_id"])]["gold_mean_token_logprob"]
    aggregates = _aggregate(rows)
    if args.phase == "validation":
        args.policy_selection.write_text(json.dumps(_select_radius(aggregates), indent=2, sort_keys=True), encoding="utf-8")
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "protocol": (
            "toy-motivated pretrained confirmation"
            if study == "confirmation"
            else "oracle-first frozen Qwen native-K/V materialization frontier"
        ),
        "study": study,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "phase": args.phase,
        "examples_per_dataset": args.examples_per_dataset,
        "G_encode": args.encoding_tokens,
        "G_search": args.parent_tokens,
        "G_materialize": "logical policy intervals",
        "inherited_materialization_bands": inherited,
        "radii": list(RADII),
        "fixed_kv_budgets": list(FIXED_BUDGETS),
        "rows": rows,
        "aggregates": aggregates,
    }
    (args.output_dir / f"{artifact_prefix}_{args.phase}.json").write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.output_dir / f"{artifact_prefix}_{args.phase}_rows.csv", [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows])
    _write_csv(args.output_dir / f"{artifact_prefix}_{args.phase}_aggregate.csv", aggregates)
    combined = []
    for phase in ("validation", "heldout"):
        path = args.output_dir / f"{artifact_prefix}_{phase}.json"
        if path.exists():
            combined.extend(json.loads(path.read_text(encoding="utf-8"))["aggregates"])
    _plots(combined, args.output_dir)
    return artifact


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    external_data = ROOT / "data/.paper2_5_datasets"
    inherited = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation"
    output = ROOT / "docs/papers/shared/results/paper3_kv_materialization"
    parser.add_argument("--phase", choices=("validation", "heldout"), required=True)
    parser.add_argument("--study", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--artifact-prefix", default="oracle_frontier")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--examples-per-dataset", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--native-limit", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--encoding-tokens", type=int, default=256)
    parser.add_argument("--parent-tokens", type=int, default=32)
    parser.add_argument("--musique-dev", type=Path, default=external_data / "musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=external_data / "2wiki/dev.json")
    parser.add_argument("--discovery", type=Path, default=inherited / "gate3_discovery_selections.json")
    parser.add_argument("--band-selection", type=Path, default=inherited / "gate3_materialization_band_selection.json")
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--policy-selection", type=Path, default=output / "oracle_policy_selection.json")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"phase": result["phase"], "rows": len(result["rows"])}, indent=2))
