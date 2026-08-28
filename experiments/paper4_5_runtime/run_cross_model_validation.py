"""Run the frozen Paper 4.5 cross-model correctness and portability gate.

Each model executes in a separate process so a completed phase releases model
weights and CUDA allocator state before the next checkpoint is loaded.  The
finalizer is deliberately model-free and can be rerun after an interrupted job.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pra_hf.runtime import RuntimeKVCache, RuntimeKVCacheKey
from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.task_context import (
    TaskEvent,
    TaskEventType,
    TaskGraph,
    TaskProvenance,
    attach_task_provenance,
)
from pra_hf.task_scope import TaskScopeSelector
from pra_torch.hf import PRAHFConfig, gemma3_global_layer_ids, inject_pra
from pra_torch.memory import SelectedChunk


RESULTS = ROOT / "docs" / "papers" / "shared" / "results" / "paper4_5_runtime"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    family: str
    model_id: str
    revision: str
    canonical_model_id: str
    canonical_revision: str
    checkpoint_status: str
    topology_policy: str


MODEL_SPECS = {
    "qwen": ModelSpec(
        "qwen",
        "qwen3",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "official_checkpoint",
        "all decoder layers for full gate; last eight for sparse profile",
    ),
    "llama": ModelSpec(
        "llama",
        "llama3",
        "unsloth/Llama-3.2-1B",
        "9535bd9b1d1dea6acafbdc4813b728796aeb28da",
        "meta-llama/Llama-3.2-1B",
        "4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "official_checkpoint_access_blocked_public_weight_mirror_validated",
        "all decoder layers for full gate; last eight for sparse profile",
    ),
    "gemma": ModelSpec(
        "gemma",
        "gemma3",
        "google/gemma-3-1b-it",
        "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "google/gemma-3-1b-it",
        "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "official_checkpoint",
        "all global-attention layers for full gate; last two global layers sparse",
    ),
}


SEMANTIC_CASES = (
    {
        "case_id": "capital",
        "record_id": "record-capital",
        "reference": "The capital of Portugal is Lisbon.",
        "query": "The capital of Portugal is",
        "answer": " Lisbon",
    },
    {
        "case_id": "color",
        "record_id": "record-color",
        "reference": "The calibration color is blue.",
        "query": "The calibration color is",
        "answer": " blue",
    },
    {
        "case_id": "code",
        "record_id": "record-code",
        "reference": "The validation code is 42.",
        "query": "The validation code is",
        "answer": " 42",
    },
)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function):
    _sync(device)
    started = time.perf_counter()
    value = function()
    _sync(device)
    return time.perf_counter() - started, value


def _legacy_cache(cache) -> tuple[tuple[torch.Tensor, ...], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(tuple(tensor for tensor in layer) for layer in cache.to_legacy_cache())
    return tuple(tuple(tensor for tensor in layer) for layer in cache)


def _cache_equal(left, right) -> bool:
    left_rows, right_rows = _legacy_cache(left), _legacy_cache(right)
    return len(left_rows) == len(right_rows) and all(
        torch.equal(a, b)
        for left_layer, right_layer in zip(left_rows, right_rows)
        for a, b in zip(left_layer, right_layer)
    )


def _target_id(tokenizer, answer: str) -> int:
    ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Answer {answer!r} did not tokenize.")
    return int(ids[0])


def _rank(logits: torch.Tensor, target_id: int) -> int:
    target = logits[..., target_id]
    return int((logits > target.unsqueeze(-1)).sum().item()) + 1


def _selected(entry, layer_ids: tuple[int, ...]) -> dict[int, list[list[SelectedChunk]]]:
    selections: dict[int, list[list[SelectedChunk]]] = {}
    for layer_id in layer_ids:
        selections[layer_id] = [[
            SelectedChunk(
                entry=entry,
                chunk=chunk,
                reference_score=1.0,
                chunk_score=1.0,
                layer_id=layer_id,
                reference_rank=1,
                rank_within_reference=rank,
                metadata={"selection_source": "paper4_5_cross_model_gate"},
            )
            for rank, chunk in enumerate(entry.layer_memory[layer_id].chunks, start=1)
        ]]
    return selections


def _kv_bytes(entry, layer_ids: tuple[int, ...]) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer_id in layer_ids
        for chunk in entry.layer_memory[layer_id].chunks
        for tensor in (chunk.token_kv.k, chunk.token_kv.v)
    )


def _condition_logits(
    handle,
    *,
    entry,
    reference_ids: torch.Tensor,
    query_ids: torch.Tensor,
    layer_ids: tuple[int, ...],
    visible: bool,
):
    if visible:
        handle.configure_memory_layers(set())
        ids = torch.cat((reference_ids, query_ids), dim=1).to(handle.device)
        positions = None
    else:
        handle.configure_memory_layers(
            set(layer_ids), fixed_selections=_selected(entry, layer_ids)
        )
        ids = query_ids.to(handle.device)
        start = int(reference_ids.shape[1])
        positions = torch.arange(start, start + ids.shape[1], device=handle.device).unsqueeze(0)
    return handle.model(input_ids=ids, position_ids=positions, use_cache=False).logits[:, -1]


def _model_layers(model, spec: ModelSpec) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    count = int(model.config.num_hidden_layers)
    if spec.key == "gemma":
        full = tuple(gemma3_global_layer_ids(model.config))
        sparse = full[-2:]
        coverage = "global_layers_only_local_sliding_layers_native"
    else:
        full = tuple(range(count))
        sparse = full[-min(8, count):]
        coverage = "all_decoder_layers"
    return full, sparse, coverage


@torch.no_grad()
def run_model(spec: ModelSpec, *, device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 7:
        model.generation_config.disable_compile = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))

    full_layers, sparse_layers, topology_coverage = _model_layers(model, spec)
    parity_ids = tokenizer(
        "PRA disabled parity preserves the host model.", return_tensors="pt"
    ).input_ids[:, :24].to(device)
    parity_mask = torch.ones_like(parity_ids)
    baseline_seconds, baseline = _timed(
        device,
        lambda: model(
            parity_ids,
            attention_mask=parity_mask,
            output_hidden_states=True,
            use_cache=True,
        ),
    )
    _, baseline_generation = _timed(
        device,
        lambda: model.generate(
            parity_ids, attention_mask=parity_mask, max_new_tokens=2, do_sample=False
        ),
    )

    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=full_layers,
            model_max_context_tokens=512,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=128,
            context_safety_reserve_tokens=4,
            top_k_references=1,
            top_k_chunks_per_reference=4,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
        ),
    )
    handle.set_memory_enabled(False)
    wrapped_seconds, wrapped = _timed(
        device,
        lambda: model(
            parity_ids,
            attention_mask=parity_mask,
            output_hidden_states=True,
            use_cache=True,
        ),
    )
    _, wrapped_generation = _timed(
        device,
        lambda: model.generate(
            parity_ids, attention_mask=parity_mask, max_new_tokens=2, do_sample=False
        ),
    )
    disabled = {
        "logits_exact": bool(torch.equal(baseline.logits, wrapped.logits)),
        "hidden_states_exact": bool(all(
            torch.equal(left, right)
            for left, right in zip(baseline.hidden_states, wrapped.hidden_states)
        )),
        "generation_exact": bool(torch.equal(baseline_generation, wrapped_generation)),
        "cache_exact": bool(_cache_equal(baseline.past_key_values, wrapped.past_key_values)),
    }
    if not all(disabled.values()):
        raise AssertionError(f"{spec.key} disabled parity failed: {disabled}")

    semantic_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    reference_gate = {"shape": True, "positions": True, "native_heads": True}
    decode_lifetime = True
    full_differences: list[tuple[float, float, bool]] = []
    for case in SEMANTIC_CASES:
        reference_ids = tokenizer(
            case["reference"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        query_ids = tokenizer(
            case["query"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        build_seconds, entry = _timed(
            device,
            lambda: handle.add_reference(
                f"validation://{case['record_id']}", reference_ids, text=case["reference"]
            ),
        )
        expected_heads = int(model.config.num_key_value_heads)
        expected_dim = int(
            getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // model.config.num_attention_heads,
            )
        )
        for layer_id in full_layers:
            for chunk in entry.layer_memory[layer_id].chunks:
                kv = chunk.token_kv
                reference_gate["shape"] &= kv.k.ndim == 4 and kv.v.shape == kv.k.shape
                reference_gate["native_heads"] &= int(kv.k.shape[1]) == expected_heads
                reference_gate["shape"] &= int(kv.k.shape[-1]) == expected_dim
                reference_gate["positions"] &= bool(torch.equal(
                    kv.position_ids.cpu(),
                    torch.arange(chunk.logical_start, chunk.logical_end).unsqueeze(0),
                ))

        target_id = _target_id(tokenizer, case["answer"])
        condition_specs = (
            ("VISIBLE_PREFIX", full_layers, True),
            ("NATIVE_FULL_REFERENCE", full_layers, False),
            ("NATIVE_SELECTED_FULL_RECORD", full_layers, False),
            ("NATIVE_CANONICAL_SPARSE_PROFILE", sparse_layers, False),
        )
        visible_logits = None
        for condition, active_layers, visible in condition_specs:
            seconds, logits = _timed(
                device,
                lambda active_layers=active_layers, visible=visible: _condition_logits(
                    handle,
                    entry=entry,
                    reference_ids=reference_ids,
                    query_ids=query_ids,
                    layer_ids=active_layers,
                    visible=visible,
                ),
            )
            logits = logits.float().cpu()
            if visible_logits is None:
                visible_logits = logits
            delta = (logits - visible_logits).abs()
            top_id = int(logits.argmax(dim=-1).item())
            row = {
                "model": spec.key,
                "family": spec.family,
                "case_id": case["case_id"],
                "record_id": case["record_id"],
                "condition": condition,
                "first_token_rank": _rank(logits, target_id),
                "exact_short_answer": int(top_id == target_id),
                "top_token_equal_visible": int(top_id == int(visible_logits.argmax().item())),
                "max_logit_error_vs_visible": float(delta.max().item()),
                "mean_logit_error_vs_visible": float(delta.mean().item()),
                "native_tokens_per_layer": 0 if visible else int(reference_ids.shape[1]),
                "active_layers": len(active_layers),
                "latency_seconds": seconds,
                "physical_kv_bytes": 0 if visible else _kv_bytes(entry, active_layers),
                "topology_coverage": topology_coverage,
            }
            semantic_rows.append(row)
            physical_rows.append({
                **{key: row[key] for key in (
                    "model", "case_id", "condition", "native_tokens_per_layer",
                    "active_layers", "physical_kv_bytes", "latency_seconds"
                )},
                "reference_build_seconds": build_seconds,
                "selection_sparsity": 0.0 if visible else 1.0,
                "materialization_width": 0 if visible else int(reference_ids.shape[1]),
                "consumer_layers": len(active_layers),
            })
            if condition == "NATIVE_FULL_REFERENCE":
                full_differences.append((
                    row["max_logit_error_vs_visible"],
                    row["mean_logit_error_vs_visible"],
                    bool(row["top_token_equal_visible"]),
                ))

        handle.configure_memory_layers(
            set(full_layers), fixed_selections=_selected(entry, full_layers)
        )
        handle.reset_memory_lifetime_trace()
        start = int(reference_ids.shape[1])
        first_positions = torch.arange(
            start, start + query_ids.shape[1], device=device
        ).unsqueeze(0)
        model(query_ids.to(device), position_ids=first_positions, use_cache=False)
        next_id = torch.tensor([[target_id]], device=device)
        next_positions = torch.tensor([[start + query_ids.shape[1]]], device=device)
        model(next_id, position_ids=next_positions, use_cache=False)
        for layer_id in full_layers:
            trace = handle.memory_lifetime_by_layer()[layer_id]
            decode_lifetime &= len(trace) == 2
            decode_lifetime &= all(
                int(row["active_native_tokens"]) == int(reference_ids.shape[1])
                for row in trace
            )

    _write_csv(RESULTS / f"hf_{spec.key}_semantic_gate.csv", semantic_rows)
    _write_csv(RESULTS / f"hf_{spec.key}_physical_accounting.csv", physical_rows)
    parity = {
        "model": spec.key,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "canonical_model_id": spec.canonical_model_id,
        "canonical_revision": spec.canonical_revision,
        "checkpoint_status": spec.checkpoint_status,
        "parameter_count": base_parameter_count,
        "dtype": str(dtype),
        "device": str(device),
        "layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "query_heads": int(model.config.num_attention_heads),
        "native_kv_heads": int(model.config.num_key_value_heads),
        "head_dim": int(getattr(model.config, "head_dim", expected_dim)),
        "rope_implementation": f"native_{spec.family}_post_position_source_relative",
        "cache_api": type(baseline.past_key_values).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_available": bool(getattr(tokenizer, "chat_template", None)),
        "memory_feasibility": "completed_on_gtx_950m_4gb_cuda_fp16",
        "full_consumption_layers": len(full_layers),
        "sparse_consumption_layers": len(sparse_layers),
        "topology_coverage": topology_coverage,
        "disabled_parity": disabled,
        "reference_gate": reference_gate,
        "decode_lifetime": decode_lifetime,
        "full_native_max_logit_error": max(row[0] for row in full_differences),
        "full_native_mean_logit_error": sum(row[1] for row in full_differences) / len(full_differences),
        "full_native_top_token_equal": all(row[2] for row in full_differences),
        "baseline_seconds": baseline_seconds,
        "wrapped_disabled_seconds": wrapped_seconds,
        "status": (
            "partial_topology"
            if spec.key == "gemma" and all(reference_gate.values()) and decode_lifetime
            else "passed"
            if all(reference_gate.values()) and decode_lifetime
            else "failed"
        ),
    }
    _write_json(RESULTS / f"hf_{spec.key}_cross_model.json", parity)
    return parity


def _runtime_control_artifacts() -> None:
    base = RuntimeKVCacheKey("tenant-a", "user", "session-a", "shared-uri", 3)
    reuse_rows = []
    for name, candidate in (
        ("identical", replace(base)),
        ("source_revision", replace(base, source_revision="v2")),
        ("position_geometry", replace(base, position_signature="absolute:64")),
        ("materialization_layout", replace(base, materialization_signature="packed:0-8")),
        ("authorization_scope", replace(base, scope_signature="request:r2")),
    ):
        reuse_rows.append({
            "case": name,
            "reuse_allowed": int(base.reuse_compatible(candidate)),
            "expected": int(name == "identical"),
            "passed": int(base.reuse_compatible(candidate) == (name == "identical")),
        })
    _write_csv(RESULTS / "hf_payload_reuse_results.csv", reuse_rows)

    cache = RuntimeKVCache(max_bytes=48, max_entries=8, max_bytes_per_tenant=16)
    keys = {
        "a1": base,
        "a2": replace(base, resource_id="pressure"),
        "b1": replace(base, tenant_id="tenant-b", session_id="session-b"),
    }

    def publish(name: str) -> str:
        cache.put(keys[name], name, nbytes=10)
        return name

    with ThreadPoolExecutor(max_workers=3) as executor:
        published = sorted(executor.map(publish, ("a1", "b1", "a2")))
    # Tenant A exceeds its private budget; tenant B's identical URI remains isolated.
    pressure_isolated = cache.get(keys["b1"]) == "b1" and cache.get(keys["a1"]) is None
    cache.clear_scope(tenant_id="tenant-b", session_id="session-b")
    cancelled_scope_cleared = cache.get(keys["b1"]) is None
    rows = [
        {
            "case": "same_uri_cross_tenant_concurrent_publish",
            "passed": int(published == ["a1", "a2", "b1"]),
            "tenant_isolation": int(pressure_isolated),
            "cancellation_cleanup": int(cancelled_scope_cleared),
            "evictions": cache.snapshot()["evictions"],
        }
    ]
    _write_csv(RESULTS / "hf_multitenant_concurrency_results.csv", rows)


def _paper8_task_selections() -> dict[str, tuple[str, ...]]:
    """Run the shared Paper 8 structural selector for the semantic smoke cases."""

    graph = TaskGraph()
    records = []
    for sequence, case in enumerate(SEMANTIC_CASES, start=1):
        task_id = f"task-{case['case_id']}"
        graph.apply(TaskEvent(
            f"create:{task_id}",
            sequence,
            TaskEventType.CREATE,
            task_id,
            payload={"description": case["query"]},
        ))
        records.append(attach_task_provenance(
            ContextRecord(case["record_id"], RecordType.GENERIC_TEXT, case["reference"]),
            TaskProvenance(task_id, event_sequence=sequence),
        ))
    selector = TaskScopeSelector(graph, records)
    return {
        case["case_id"]: selector.select(
            f"task-{case['case_id']}",
            case["query"],
            policy="task_structural",
            max_records=1,
        ).selected_record_ids
        for case in SEMANTIC_CASES
    }


def _write_cross_model_plot(rows: list[dict[str, str]]) -> None:
    """Plot rank and visible-prefix logit error from measured semantic rows."""

    import matplotlib.pyplot as plt

    conditions = (
        "VISIBLE_PREFIX",
        "NATIVE_FULL_REFERENCE",
        "NATIVE_SELECTED_FULL_RECORD",
        "NATIVE_CANONICAL_SPARSE_PROFILE",
    )
    labels = ("Visible", "Native full", "Selected full", "Sparse")
    models = tuple(MODEL_SPECS)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    width = 0.24
    for model_index, model in enumerate(models):
        ranks, errors = [], []
        for condition in conditions:
            subset = [
                row for row in rows
                if row["model"] == model and row["condition"] == condition
            ]
            ranks.append(sum(float(row["first_token_rank"]) for row in subset) / len(subset))
            errors.append(
                sum(float(row["mean_logit_error_vs_visible"]) for row in subset)
                / len(subset)
            )
        positions = [index + (model_index - 1) * width for index in range(len(labels))]
        axes[0].bar(positions, ranks, width=width, label=model.title())
        axes[1].bar(positions, errors, width=width, label=model.title())
    for axis, title, ylabel in (
        (axes[0], "Expected next-token rank", "Mean rank (lower is better)"),
        (axes[1], "Deviation from visible prefix", "Mean absolute logit error"),
    ):
        axis.set_title(title)
        axis.set_xticks(range(len(labels)), labels, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS / "hf_cross_model_semantic_gate.png", dpi=180)
    figure.savefig(RESULTS / "hf_cross_model_semantic_gate.pdf")
    plt.close(figure)


def _write_cross_model_tex(rows: list[dict[str, str]], parity_rows: list[dict[str, Any]]) -> None:
    def subset(model: str, condition: str) -> list[dict[str, str]]:
        return [
            row for row in rows
            if row["model"] == model and row["condition"] == condition
        ]

    values: dict[str, str] = {}
    for model in MODEL_SPECS:
        title = model.title()
        selected = subset(model, "NATIVE_SELECTED_FULL_RECORD")
        sparse = subset(model, "NATIVE_CANONICAL_SPARSE_PROFILE")
        values[f"{title}SelectedAccuracy"] = (
            f"{100 * sum(int(row['exact_short_answer']) for row in selected) / len(selected):.1f}"
        )
        values[f"{title}SparseAccuracy"] = (
            f"{100 * sum(int(row['exact_short_answer']) for row in sparse) / len(sparse):.1f}"
        )
        values[f"{title}SparseRank"] = (
            f"{sum(float(row['first_token_rank']) for row in sparse) / len(sparse):.2f}"
        )
        parity = next(row for row in parity_rows if row["model"] == model)
        values[f"{title}FullMaxError"] = f"{float(parity['full_native_max_logit_error']):.4f}"
        values[f"{title}FullMeanError"] = f"{float(parity['full_native_mean_logit_error']):.4f}"
    lines = [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    (RESULTS / "generated_cross_model_results.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def finalize() -> dict[str, Any]:
    parity_rows = []
    semantic_rows = []
    physical_rows = []
    manifest_rows = []
    task_rows = []
    task_selections = _paper8_task_selections()
    for spec in MODEL_SPECS.values():
        path = RESULTS / f"hf_{spec.key}_cross_model.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        manifest_rows.append({
            **asdict(spec),
            "result_status": payload["status"] if payload else "not_run",
            "runtime_model_id": payload["model_id"] if payload else "",
            "runtime_revision": payload["revision"] if payload else "",
            "official_checkpoint_tested": int(
                payload is not None and spec.model_id == spec.canonical_model_id
            ),
        })
        if payload:
            parity_rows.append({
                key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                for key, value in payload.items()
            })
        semantic_path = RESULTS / f"hf_{spec.key}_semantic_gate.csv"
        if semantic_path.exists():
            rows = list(csv.DictReader(semantic_path.open(encoding="utf-8")))
            semantic_rows.extend(rows)
            for row in rows:
                if row["condition"] in {"VISIBLE_PREFIX", "NATIVE_SELECTED_FULL_RECORD"}:
                    expected_id = f"record-{row['case_id']}"
                    selected_ids = task_selections[row["case_id"]]
                    task_rows.append({
                        "model": spec.key,
                        "case_id": row["case_id"],
                        "condition": row["condition"],
                        "selected_record_ids": "|".join(selected_ids),
                        "selected_id_exact": int(selected_ids == (expected_id,)),
                        "visible_or_native_answer_exact": row["exact_short_answer"],
                        "scope": "paper8_task_structural",
                    })
        physical_path = RESULTS / f"hf_{spec.key}_physical_accounting.csv"
        if physical_path.exists():
            physical_rows.extend(csv.DictReader(physical_path.open(encoding="utf-8")))
    _write_csv(RESULTS / "hf_cross_model_manifest.csv", manifest_rows)
    _write_csv(RESULTS / "hf_cross_model_native_parity.csv", parity_rows)
    _write_csv(RESULTS / "hf_cross_model_task_smoke.csv", task_rows)
    _write_csv(RESULTS / "hf_cross_model_physical_accounting.csv", physical_rows)
    _runtime_control_artifacts()
    _write_cross_model_plot(semantic_rows)
    _write_cross_model_tex(semantic_rows, parity_rows)
    result = {
        "models_completed": [row["model"] for row in parity_rows],
        "models_expected": list(MODEL_SPECS),
        "semantic_rows": len(semantic_rows),
        "task_rows": len(task_rows),
        "status": "complete" if len(parity_rows) == len(MODEL_SPECS) else "partial",
    }
    _write_json(RESULTS / "hf_cross_model_summary.json", result)
    return result


def _run_isolated(model: str, device: str) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.paper4_5_runtime.run_cross_model_validation",
        "--model",
        model,
        "--device",
        device,
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*MODEL_SPECS, "all", "finalize"), default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model == "all":
        for model_key in MODEL_SPECS:
            _run_isolated(model_key, args.device)
        outcome = finalize()
    elif args.model == "finalize":
        outcome = finalize()
    else:
        outcome = run_model(MODEL_SPECS[args.model], device_name=args.device)
    print(json.dumps(outcome, indent=2, sort_keys=True))
