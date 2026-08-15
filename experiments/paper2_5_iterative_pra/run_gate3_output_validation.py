"""Generate answers from frozen Paper 2.5 discovery and native-K/V sets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

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
    native_kv_accounting,
    selected_span_metrics,
)


PRIMARY_SELECTIONS = (
    "one_shot",
    "graph_sparse",
    "graph_balanced",
    "graph_high",
    "oracle_evidence",
)
SWEEP_SELECTIONS = ("graph_balanced", "oracle_evidence")
SEARCH_USE_BANDS = ("late_1", "layer_12", "all_28")


class _TokenClock(StoppingCriteria):
    """Record synchronized completion time after every greedy decode token."""

    def __init__(self, device: torch.device):
        self.device = device
        self.timestamps: list[float] = []

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.timestamps.append(time.perf_counter())
        return False


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if "active_native_kv_token_states" not in row:
                    generated_states = int(row.get("generated_tokens", 0)) * 28
                    row["active_native_kv_token_states"] = int(
                        row.get("native_kv_token_states", 0)
                    ) + int(row.get("direct_context_kv_token_states", 0)) + generated_states
                    row["active_native_kv_bytes"] = int(
                        row.get("native_kv_bytes", 0)
                    ) + int(row.get("direct_context_kv_bytes", 0)) + generated_states * 4096
                    row["peak_local_kv_tokens"] = int(row.get("prompt_tokens", 0)) + int(
                        row.get("generated_tokens", 0)
                    )
                row.setdefault(
                    "component_timing_scope",
                    "final decode forward, summed over active PRA layers",
                )
                rows.append(row)
    return rows


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _row_key(row: dict) -> tuple:
    return (
        row["phase"],
        row["dataset"],
        row["example_id"],
        row["condition"],
        row.get("selection"),
        row.get("materialization_band"),
        row.get("search_layer"),
    )


def _examples(args, identities: set[str]) -> dict[str, object]:
    rows = load_musique(args.musique_dev) + load_2wiki(args.twowiki_dev)
    selected = {row.example_id: row for row in rows if row.example_id in identities}
    if set(selected) != identities:
        missing = sorted(identities - set(selected))
        raise ValueError(f"local datasets are missing frozen identities: {missing[:3]}")
    return selected


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
    """Use the first nonempty generated line under a frozen brief-answer prompt."""
    clean = str(text).replace("<think>", "").replace("</think>", "").strip()
    for line in clean.splitlines():
        line = line.strip()
        if line:
            if line.casefold().startswith("answer:"):
                line = line.split(":", 1)[1].strip()
            return line
    return clean


def _head_dim(model) -> int:
    """Return the native attention head width across supported HF configs."""
    configured = getattr(model.config, "head_dim", None)
    if configured is not None:
        return int(configured)
    return int(model.config.hidden_size) // int(model.config.num_attention_heads)


def _attention_summary(pra: PRAForCausalLM, evidence_spans) -> dict[str, float | None]:
    rows = []
    evidence_spans = [tuple(map(int, span)) for span in evidence_spans]
    for layer, adapter in pra._handle.adapters.items():
        if not adapter.memory_enabled or adapter.last_attention_weights is None:
            continue
        weights = adapter.last_attention_weights.detach().float()
        if weights.ndim != 4:
            continue
        final = weights[0, :, -1, :].mean(dim=0)
        memory_width = int(adapter.last_diagnostics.get("hf_memory_width", 0))
        selected = adapter.last_selected_chunks[0]
        evidence_mask = []
        for hit in selected:
            evidence_mask.extend(
                any(start <= token < end for start, end in evidence_spans)
                for token in range(hit.logical_start, hit.logical_end)
            )
        if memory_width != len(evidence_mask) or memory_width > final.numel():
            raise AssertionError(
                f"layer {layer} attention width does not match selected native K/V"
            )
        memory = final[:memory_width]
        mask = torch.tensor(evidence_mask, device=memory.device, dtype=torch.bool)
        probability = final.clamp_min(1e-12)
        entropy = float(-(probability * probability.log()).sum().cpu())
        rows.append(
            {
                "evidence_attention_mass": float(memory[mask].sum().cpu()) if mask.any() else 0.0,
                "non_evidence_attention_mass": float(memory[~mask].sum().cpu()) if (~mask).any() else 0.0,
                "local_attention_mass": float(final[memory_width:].sum().cpu()),
                "attention_entropy": entropy,
                "normalized_attention_entropy": entropy / max(math.log(final.numel()), 1e-12),
                "attention_output_norm": float(
                    adapter.last_diagnostics.get("attention_output_norm", 0.0)
                ),
            }
        )
    keys = (
        "evidence_attention_mass",
        "non_evidence_attention_mass",
        "local_attention_mass",
        "attention_entropy",
        "normalized_attention_entropy",
        "attention_output_norm",
    )
    return {
        key: statistics.fmean(float(row[key]) for row in rows) if rows else None
        for key in keys
    }


def _generate(model, tokenizer, encoded, device, max_new_tokens: int):
    encoded = encoded.to(device)
    clock = _TokenClock(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
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
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    first = clock.timestamps[0] - started if clock.timestamps else ended - started
    intervals = [right - left for left, right in zip(clock.timestamps, clock.timestamps[1:])]
    return {
        "raw_answer": raw,
        "generated_answer": _extract_answer(raw),
        "prompt_tokens": int(encoded.input_ids.shape[1]),
        "generated_tokens": int(generated.numel()),
        "ttft_seconds": first,
        "tpot_seconds": statistics.fmean(intervals) if intervals else 0.0,
        "total_generation_seconds": ended - started,
        "tokens_per_second": int(generated.numel()) / max(ended - started, 1e-12),
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }


def _direct_row(pra, tokenizer, example, discovery, *, full_context, device, max_new_tokens):
    pra._handle.configure_memory_layers(set())
    pra._handle.set_attention_diagnostics(False)
    encoded = _prompt(tokenizer, example.question, context=example.source if full_context else None)
    if encoded.input_ids.shape[1] > pra.config.native_operation_limit:
        raise ValueError("direct full context exceeds the frozen native limit")
    generated = _generate(pra.model, tokenizer, encoded, device, max_new_tokens)
    source_tokens = int(discovery["source_tokens"])
    if full_context:
        span_metrics = selected_span_metrics(
            ((0, source_tokens),), discovery["evidence_token_spans"], source_tokens
        )
    else:
        span_metrics = selected_span_metrics(
            (), discovery["evidence_token_spans"], source_tokens
        )
    prompt_tokens = int(generated["prompt_tokens"])
    peak_local_tokens = prompt_tokens + int(generated["generated_tokens"])
    prompt_kv = native_kv_accounting(
        unique_tokens=prompt_tokens,
        materialization_layers=tuple(range(int(pra.model.config.num_hidden_layers))),
        kv_heads=int(pra.model.config.num_key_value_heads),
        head_dim=_head_dim(pra.model),
        element_size=2,
    )
    return {
        **generated,
        **deterministic_answer_metrics(generated["generated_answer"], example.answer),
        **span_metrics,
        **native_kv_accounting(
            unique_tokens=0,
            materialization_layers=(),
            kv_heads=int(pra.model.config.num_key_value_heads),
            head_dim=_head_dim(pra.model),
            element_size=2,
        ),
        "direct_context_kv_tokens": prompt_tokens,
        "direct_context_kv_token_states": prompt_kv["native_kv_token_states"],
        "direct_context_kv_bytes": prompt_kv["native_kv_bytes"],
        "peak_local_kv_tokens": peak_local_tokens,
        "active_native_kv_token_states": peak_local_tokens * int(pra.model.config.num_hidden_layers),
        "active_native_kv_bytes": (
            peak_local_tokens * int(pra.model.config.num_hidden_layers) * 4096
        ),
        "active_kv_fraction": 0.0,
        "materialization_seconds": 0.0,
        "selected_kv_transfer_seconds": 0.0,
        "memory_attention_seconds": 0.0,
        "component_timing_scope": "not applicable; PRA memory disabled",
        "evidence_attention_mass": None,
        "non_evidence_attention_mass": None,
        "local_attention_mass": None,
        "attention_entropy": None,
        "normalized_attention_entropy": None,
        "attention_output_norm": None,
    }


def _memory_row(
    pra,
    tokenizer,
    example,
    discovery,
    band,
    entry,
    *,
    device,
    max_new_tokens,
):
    selected = fixed_chunks_for_spans(
        entry,
        routing_layer=pra.routing_layer,
        selected_spans=discovery["selected_spans"],
        selection_name=discovery["selection"],
    )
    mapped = pra._handle.map_chunk_identities_to_layers([selected], band.layers)
    pra._handle.configure_memory_layers(set(band.layers), fixed_selections=mapped)
    pra._handle.set_attention_diagnostics(True)
    encoded = _prompt(tokenizer, example.question)
    generated = _generate(pra.model, tokenizer, encoded, device, max_new_tokens)
    diagnostics = pra._handle.diagnostics_by_layer()
    active = {layer: diagnostics[layer] for layer in band.layers}
    if any(not row for row in active.values()):
        raise AssertionError("an intended PRA layer did not execute")
    actual_by_layer = {
        int(round(float(row["memory_tokens_materialized"]))) for row in active.values()
    }
    if len(actual_by_layer) != 1:
        raise AssertionError("fixed evidence materialized a different token count by layer")
    actual_tokens = actual_by_layer.pop()
    if actual_tokens != sum(hit.selected_token_count for hit in selected):
        raise AssertionError("the materialization budget changed the frozen evidence identity")
    for row in active.values():
        if int(row["hf_query_heads"]) != int(pra.model.config.num_attention_heads):
            raise AssertionError("query-head shape changed in PRA execution")
        if int(row["hf_native_kv_heads"]) != int(pra.model.config.num_key_value_heads):
            raise AssertionError("native GQA K/V-head shape changed in PRA execution")
    selected_spans = [(hit.logical_start, hit.logical_end) for hit in selected]
    span_metrics = selected_span_metrics(
        selected_spans, discovery["evidence_token_spans"], discovery["source_tokens"]
    )
    timing = {
        "materialization_seconds": sum(
            float(row.get("materialization_duration_seconds", 0.0)) for row in active.values()
        ),
        "selected_kv_transfer_seconds": sum(
            float(row.get("selected_kv_transfer_duration_seconds", 0.0))
            for row in active.values()
        ),
        "memory_attention_seconds": sum(
            float(row.get("memory_attention_duration_seconds", 0.0)) for row in active.values()
        ),
    }
    external_kv = native_kv_accounting(
        unique_tokens=actual_tokens,
        materialization_layers=band.layers,
        kv_heads=int(pra.model.config.num_key_value_heads),
        head_dim=_head_dim(pra.model),
        element_size=2,
    )
    prompt_tokens = int(generated["prompt_tokens"])
    peak_local_tokens = prompt_tokens + int(generated["generated_tokens"])
    direct_states = prompt_tokens * int(pra.model.config.num_hidden_layers)
    direct_bytes = (
        direct_states
        * 2
        * int(pra.model.config.num_key_value_heads)
        * _head_dim(pra.model)
        * 2
    )
    return {
        **generated,
        **deterministic_answer_metrics(generated["generated_answer"], example.answer),
        **span_metrics,
        **external_kv,
        "direct_context_kv_tokens": prompt_tokens,
        "direct_context_kv_token_states": direct_states,
        "direct_context_kv_bytes": direct_bytes,
        "peak_local_kv_tokens": peak_local_tokens,
        "active_native_kv_token_states": (
            external_kv["native_kv_token_states"]
            + peak_local_tokens * int(pra.model.config.num_hidden_layers)
        ),
        "active_native_kv_bytes": (
            external_kv["native_kv_bytes"]
            + peak_local_tokens * int(pra.model.config.num_hidden_layers) * 4096
        ),
        "active_kv_fraction": actual_tokens / max(int(discovery["source_tokens"]), 1),
        **timing,
        "component_timing_scope": "final decode forward, summed over active PRA layers",
        **_attention_summary(pra, discovery["evidence_token_spans"]),
    }


def _specs(phase: str, partition: str, dataset: str, selected_bands: dict | None):
    bands = {band.name: band for band in MATERIALIZATION_BANDS}
    if phase == "layer_sweep":
        if partition != "validation":
            return []
        specs = [
            (f"{selection}__{band.name}", selection, band)
            for selection in SWEEP_SELECTIONS
            for band in MATERIALIZATION_BANDS
        ]
        specs.extend(
            (f"graph_balanced_l12__{name}", "graph_balanced_l12", bands[name])
            for name in SEARCH_USE_BANDS
        )
        return specs
    if phase == "heldout":
        if partition != "test" or selected_bands is None:
            return []
        band = bands[selected_bands[dataset]]
        return [
            ("native_bounded", "none", None),
            ("one_shot", "one_shot", band),
            ("graph_sparse", "graph_sparse", band),
            ("graph_balanced", "graph_balanced", band),
            ("graph_high", "graph_high", band),
            ("oracle_evidence", "oracle_evidence", band),
            ("native_full_context", "full_source", None),
        ]
    raise ValueError(f"unknown phase: {phase}")


def _select_bands(rows: list[dict]) -> dict[str, object]:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    band_sizes = {band.name: len(band.layers) for band in MATERIALIZATION_BANDS}
    for dataset in ("musique", "2wikimultihopqa"):
        for band in MATERIALIZATION_BANDS:
            values = [
                row
                for row in rows
                if row["phase"] == "layer_sweep"
                and row["dataset"] == dataset
                and row["selection"] == "graph_balanced"
                and row["materialization_band"] == band.name
            ]
            if not values:
                continue
            by_dataset[dataset].append(
                {
                    "dataset": dataset,
                    "band": band.name,
                    "layer_count": len(band.layers),
                    "rows": len(values),
                    "mean_token_f1": statistics.fmean(row["token_f1"] for row in values),
                    "mean_normalized_accuracy": statistics.fmean(
                        row["normalized_answer_accuracy"] for row in values
                    ),
                }
            )
    selected = {}
    for dataset, values in by_dataset.items():
        for row in values:
            row["objective"] = 0.5 * (
                row["mean_token_f1"] + row["mean_normalized_accuracy"]
            )
        winner = max(
            values,
            key=lambda row: (
                row["objective"],
                -band_sizes[row["band"]],
                -next(i for i, band in enumerate(MATERIALIZATION_BANDS) if band.name == row["band"]),
            ),
        )
        selected[dataset] = winner["band"]
    if set(selected) != {"musique", "2wikimultihopqa"}:
        raise ValueError("layer sweep is incomplete; no held-out band may be selected")
    return {
        "selection_partition": "validation",
        "objective": "0.5 * mean token F1 + 0.5 * mean normalized answer accuracy",
        "selection_condition": "graph_balanced only; oracle outputs are diagnostic",
        "tie_break": "fewer layers, then predeclared band order",
        "selected_bands": selected,
        "candidates": [row for values in by_dataset.values() for row in values],
    }


def _write_artifact(args, rows, discovery_manifest, band_selection=None):
    artifact = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "decoding": {
            "mode": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "answer_extraction": "first nonempty generated line",
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
        "native_limit_tokens": args.native_limit,
        "latency_protocol": {
            "ttft_tpot_total": "synchronized end-to-end greedy generation",
            "pra_components": "final decode forward, summed over active PRA layers",
            "serving_claim": False,
        },
        "atomic_materialization_tokens": args.atomic_tokens,
        "discovery_frozen_before_generation": discovery_manifest["frozen_before_generation"],
        "oracle_labels_available_to_non_oracle_conditions": False,
        "band_selection": band_selection,
        "rows": rows,
    }
    (args.output_dir / "gate3_generation_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(args.output_dir / "gate3_generation_rows.csv", rows)
    return artifact


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    discovery_manifest = json.loads(args.discovery.read_text(encoding="utf-8"))
    discovery_rows = discovery_manifest["rows"]
    discovery = {
        (row["dataset"], row["example_id"], row["selection"]): row
        for row in discovery_rows
    }
    identities = {row["example_id"] for row in discovery_rows}
    examples = _examples(args, identities)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "gate3_generation_checkpoint.jsonl"
    rows = _load_rows(checkpoint)
    completed = {_row_key(row) for row in rows}

    band_selection = None
    if args.band_selection.exists():
        band_selection = json.loads(args.band_selection.read_text(encoding="utf-8"))
    if args.phase == "heldout" and band_selection is None:
        raise ValueError("run layer_sweep before heldout generation")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    torch.manual_seed(11)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(11)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = PRAConfig(
        routing_layer=27,
        consumption_layers=tuple(range(28)),
        chunk_tokens=args.atomic_tokens,
        selected_fraction=None,
        top_k=1,
        max_direct_context=256,
        native_operation_limit=args.native_limit,
        max_materialized_tokens=args.native_limit - 128,
        context_safety_reserve_tokens=0,
        encoding_block_tokens=256,
        reference_device="cpu",
        pin_reference_memory=device.type == "cuda",
        non_blocking_transfer=device.type == "cuda",
    )
    pra = PRAForCausalLM.from_model(model, tokenizer, pra_config=config)

    phase_part = "validation" if args.phase == "layer_sweep" else "test"
    phase_ids = sorted(
        {
            row["example_id"]
            for row in discovery_rows
            if row["partition"] == phase_part
        }
    )
    if args.max_examples is not None:
        phase_ids = phase_ids[: args.max_examples]
    for example_index, example_id in enumerate(phase_ids, start=1):
        example = examples[example_id]
        dataset = example.dataset
        specs = _specs(
            args.phase,
            phase_part,
            dataset,
            band_selection["selected_bands"] if band_selection else None,
        )
        pending = []
        for condition, selection, band in specs:
            search_layer = (
                discovery[(dataset, example_id, selection)]["search_layer"]
                if selection not in {"none", "full_source"}
                else None
            )
            key = (
                args.phase,
                dataset,
                example_id,
                condition,
                selection,
                band.name if band else "none",
                search_layer,
            )
            if key not in completed:
                pending.append((condition, selection, band, search_layer))
        if not pending:
            continue

        pra.clear_references()
        uri = f"benchmark://{dataset}/{example_id}"
        handle = pra.add_reference(example.source, uri=uri)
        entry = pra._handle.cache.get(uri)
        if entry is None:
            raise AssertionError("reference cache entry disappeared after encoding")
        source_ids = tokenizer(example.source, add_special_tokens=False).input_ids
        canonical = discovery[(dataset, example_id, "one_shot")]
        if len(source_ids) != int(canonical["source_tokens"]):
            raise AssertionError("generation source tokenization differs from discovery")

        for condition, selection, band, search_layer in pending:
            selected_row = (
                discovery[(dataset, example_id, selection)]
                if selection not in {"none", "full_source"}
                else canonical
            )
            if selection == "none":
                measured = _direct_row(
                    pra,
                    tokenizer,
                    example,
                    selected_row,
                    full_context=False,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            elif selection == "full_source":
                measured = _direct_row(
                    pra,
                    tokenizer,
                    example,
                    selected_row,
                    full_context=True,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                measured = _memory_row(
                    pra,
                    tokenizer,
                    example,
                    selected_row,
                    band,
                    entry,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            row = {
                "phase": args.phase,
                "dataset": dataset,
                "example_id": example_id,
                "partition": phase_part,
                "question": example.question,
                "reference_answer": example.answer,
                "question_type": selected_row["question_type"],
                "annotated_hops": selected_row["annotated_hops"],
                "graph_type": selected_row["graph_type"],
                "condition": condition,
                "selection": selection,
                "oracle_condition": selection == "oracle_evidence",
                "K": selected_row["K"] if selection not in {"none", "full_source"} else None,
                "B": selected_row["B"] if selection not in {"none", "full_source"} else None,
                "H": selected_row["H"] if selection not in {"none", "full_source"} else None,
                "search_layer": search_layer,
                "materialization_band": band.name if band else "none",
                "materialization_layers": list(band.layers) if band else [],
                "conceptual_selected_parents": (
                    selected_row["conceptual_active_parents"]
                    if selection not in {"none", "full_source"}
                    else 0
                ),
                "root_recall": selected_row["root_recall"] if selection not in {"none", "full_source"} else None,
                "oracle_evidence_recall": selected_row["oracle_evidence_recall"] if selection not in {"none", "full_source"} else (1.0 if selection == "full_source" else 0.0),
                "later_evidence_recall": selected_row["later_evidence_recall"] if selection not in {"none", "full_source"} else (1.0 if selection == "full_source" else 0.0),
                "complete_evidence_recovery": selected_row["complete_evidence_recovery"] if selection not in {"none", "full_source"} else int(selection == "full_source"),
                "annotated_edge_recall": selected_row["annotated_edge_recall"] if selection not in {"none", "full_source"} else None,
                "complete_path_survival": selected_row["complete_path_survival"] if selection not in {"none", "full_source"} else int(selection == "full_source"),
                "native_recovery_depth": selected_row["native_recovery_depth"] if selection not in {"none", "full_source"} else None,
                "visited_parents": selected_row["visited_parents"] if selection not in {"none", "full_source"} else 0,
                "routing_search_seconds": selected_row["search_seconds"] if selection not in {"none", "full_source"} else 0.0,
                "source_tokens": selected_row["source_tokens"],
                "max_native_operation_tokens": pra._handle.max_native_operation_tokens,
                "native_limit_violations": pra._handle.native_limit_violations,
                **measured,
            }
            if row["oracle_condition"] and not condition.startswith("oracle_evidence"):
                raise AssertionError("oracle identity leaked into a non-oracle condition")
            _append_row(checkpoint, row)
            rows.append(row)
            completed.add(_row_key(row))
        print(
            f"[output-{args.phase} {example_index}/{len(phase_ids)}] "
            f"{dataset} {example_id} rows={len(pending)}",
            flush=True,
        )

    if args.phase == "layer_sweep" and args.max_examples is None:
        band_selection = _select_bands(rows)
        args.band_selection.write_text(
            json.dumps(band_selection, indent=2, sort_keys=True), encoding="utf-8"
        )
    return _write_artifact(args, rows, discovery_manifest, band_selection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = ROOT / "data/.paper2_5_datasets"
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/output_validation"
    parser.add_argument("--phase", choices=("layer_sweep", "heldout"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--native-limit", type=int, default=4096)
    parser.add_argument("--atomic-tokens", type=int, default=16)
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Debug-only cap applied after the deterministic example ordering.",
    )
    parser.add_argument("--musique-dev", type=Path, default=data / "musique/data/musique_ans_v1.0_dev.jsonl")
    parser.add_argument("--twowiki-dev", type=Path, default=data / "2wiki/dev.json")
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--discovery", type=Path, default=output / "gate3_discovery_selections.json")
    parser.add_argument("--band-selection", type=Path, default=output / "gate3_materialization_band_selection.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    print(json.dumps({"rows": len(result["rows"]), "phase": arguments.phase}, indent=2))
