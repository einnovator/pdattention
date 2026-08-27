"""Run the Paper 8 production-PRA oracle-scope iteration.

The experiment changes task admission before the frozen Paper-7 native hybrid
router. It records selection, typed materialization, and model consumption as
separate stages. JSONL checkpoints make the bounded CUDA run restartable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.task_production_cases import (
    DAGShape,
    ProductionTaskCase,
    TaskConfusability,
    dag_shape_case,
    join_capacity_case,
    production_task_cases,
)
from pra_hf import PRAConfig, PRAForCausalLM
from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    DeploymentTopology,
    StoragePolicy,
    TypeContextPolicy,
)
from pra_hf.context_records import ContextRecord, RecordType
from pra_hf.context_store import RecordScope
from pra_hf.progressive_context import ProgressiveContextRuntime
from pra_hf.session_service import LocalSessionService
from pra_hf.task_context import TaskGraph, TaskProvenance
from pra_hf.task_scope import TaskScopePolicy, TaskScopeSelector
from pra_hf.typed_context import CompressorRegistry


RESULTS = ROOT / "docs/papers/shared/results/paper8_tasks/production_pra"
FIGURES = ROOT / "docs/papers/shared/figures/paper8_tasks"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SEEDS = (11, 23, 37, 53, 71)
SCOPE_POLICIES = (
    TaskScopePolicy.SESSION,
    TaskScopePolicy.TASK_LOCAL,
    TaskScopePolicy.TASK_STRUCTURAL,
)
PROTOCOL = "paper8-production-pra-oracle-scope-v1"


@dataclass
class IndexedScope:
    progressive: ProgressiveContextRuntime
    temporary_store: tempfile.TemporaryDirectory
    source_to_runtime: dict[str, str]
    runtime_to_source: dict[str, str]
    index_seconds: float

    def close(self) -> None:
        self.progressive.runtime.store.close()
        self.temporary_store.cleanup()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _payload_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=True, default=str
    )


def _record_task_id(record: ContextRecord) -> str | None:
    provenance = TaskProvenance.from_record(record)
    return provenance.task_id if provenance else None


def _scope_partition(case: ProductionTaskCase, policy: TaskScopePolicy):
    return TaskScopeSelector(TaskGraph(case.graph), case.records).partition(
        case.active_task_id, policy=policy
    )


def _scope_label(policy: TaskScopePolicy, *, lexical: bool = False) -> str:
    suffix = "_LEXICAL" if lexical else ""
    return {
        TaskScopePolicy.SESSION: "PRA_SESSION",
        TaskScopePolicy.TASK_LOCAL: "PRA_TASK_LOCAL",
        TaskScopePolicy.TASK_STRUCTURAL: "PRA_TASK_STRUCT",
        TaskScopePolicy.TASK_ADAPTIVE: "PRA_TASK_ADAPTIVE",
    }[policy] + suffix


def _load_model(device: str) -> PRAForCausalLM:
    target = torch.device(device)
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    pra = PRAForCausalLM.from_pretrained(
        MODEL_ID,
        pra_config=PRAConfig(
            routing_layer=27,
            consumption_layers=(27,),
            chunk_tokens=32,
            chunk_overlap_tokens=8,
            selected_fraction=None,
            top_k=8,
            max_direct_context=128,
            native_operation_limit=512,
            max_materialized_tokens=256,
            context_safety_reserve_tokens=4,
            encoding_block_tokens=128,
            reference_device="cpu",
            routing_mode="hybrid_iterative",
            routing_depth=2,
            branch_top_k=4,
            beam_size=32,
            max_unique_chunks=8,
        ),
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    pra.model.to(target).eval()
    return pra


def _index_scope(
    pra: PRAForCausalLM,
    case: ProductionTaskCase,
    policy: TaskScopePolicy,
) -> IndexedScope:
    partition = _scope_partition(case, policy)
    temporary = tempfile.TemporaryDirectory(prefix="paper8-pra-")
    policies = {
        record_type: TypeContextPolicy(unit_limit=2)
        for record_type in RecordType
    }
    runtime = AdaptiveContextRuntime(
        RecordScope("paper8-production", f"{case.case_id}:{policy.value}"),
        ContextPolicy(
            topology=DeploymentTopology.SAME_PROCESS,
            storage=StoragePolicy.ON_DEMAND,
            local_store=Path(temporary.name),
            record_policies=policies,
            persistent_store=False,
            max_native_index_tokens=4096,
            max_native_index_bytes=1_048_576,
        ),
    )
    progressive = ProgressiveContextRuntime(runtime, chunk_tokens=32)
    source_to_runtime = {}
    runtime_to_source = {}
    for record in partition.candidate_records:
        task_id = _record_task_id(record)
        adapted = progressive.ingest(
            record.payload,
            record_type=record.record_type,
            provenance={
                "source": "paper8_production_task_case",
                "source_record_id": record.record_id,
                "task_id": task_id,
                "case_id": case.case_id,
            },
        )
        source_to_runtime[record.record_id] = adapted.record_id
        runtime_to_source[adapted.record_id] = record.record_id
    progressive.registry.pra_model = pra
    started = time.perf_counter()
    for runtime_id in runtime_to_source:
        progressive.register_backing_record(runtime_id)
    index_seconds = time.perf_counter() - started
    return IndexedScope(
        progressive,
        temporary,
        source_to_runtime,
        runtime_to_source,
        index_seconds,
    )


def _ranking_rows(routing, bundle: IndexedScope) -> list[dict[str, object]]:
    by_uri = {
        handle.uri: bundle.runtime_to_source[runtime_id]
        for runtime_id, handle in bundle.progressive.registry.backing_reference_handles.items()
    }
    rows = []
    for reference in routing.rankings:
        uri = str(reference["reference_uri"])
        for chunk in reference["chunks"]:
            rows.append({
                "source_record_id": by_uri.get(uri, ""),
                "runtime_record_id": next((
                    runtime_id
                    for runtime_id, handle in bundle.progressive.registry.backing_reference_handles.items()
                    if handle.uri == uri
                ), ""),
                "reference_uri": uri,
                "reference_score": float(reference["reference_score"]),
                "reference_rank": int(reference["reference_rank"]),
                "chunk_id": str(chunk["chunk_id"]),
                "chunk_score": float(chunk["chunk_score"]),
                "logical_start": int(chunk.get("logical_start", chunk.get("token_start", 0))),
                "logical_end": int(chunk.get("logical_end", chunk.get("token_end", 0))),
                "gist_count": int(chunk.get("gist_count", 1)),
            })
    rows.sort(key=lambda row: (-float(row["chunk_score"]), str(row["chunk_id"])))
    return rows


def _selected_rows(stats: Mapping[str, object], bundle: IndexedScope) -> list[dict[str, object]]:
    by_uri = {
        handle.uri: (runtime_id, bundle.runtime_to_source[runtime_id])
        for runtime_id, handle in bundle.progressive.registry.backing_reference_handles.items()
    }
    rows = []
    for rank, selected in enumerate(stats.get("selected", ()), start=1):
        uri = str(selected["reference_uri"])
        if uri not in by_uri:
            continue
        runtime_id, source_id = by_uri[uri]
        start = int(selected.get("logical_start", selected.get("token_start", 0)))
        stop = int(selected.get("logical_end", selected.get("token_end", 0)))
        payload = bundle.progressive.runtime.store.get(
            runtime_id, scope=bundle.progressive.runtime.scope
        )
        token_ids = bundle.progressive.registry.pra_model.tokenizer(
            _payload_text(payload), add_special_tokens=False
        ).input_ids
        text = bundle.progressive.registry.pra_model.tokenizer.decode(
            token_ids[start:stop], skip_special_tokens=True
        )
        rows.append({
            "selected_rank": rank,
            "source_record_id": source_id,
            "runtime_record_id": runtime_id,
            "reference_uri": uri,
            "chunk_id": str(selected["chunk_id"]),
            "chunk_score": float(selected.get("chunk_score", 0.0)),
            "reference_score": float(selected.get("reference_score", 0.0)),
            "logical_start": start,
            "logical_end": stop,
            "token_count": max(0, stop - start),
            "text": text,
        })
    return rows


def _materialize(
    bundle: IndexedScope,
    case: ProductionTaskCase,
    rows: Sequence[Mapping[str, object]],
) -> tuple[set[str], float, int]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["runtime_record_id"])].append({
            "chunk_id": row["chunk_id"],
            "logical_start": row["logical_start"],
            "logical_end": row["logical_end"],
            "text": row["text"],
        })
    started = time.perf_counter()
    payload_bytes = 0
    sources = set()
    for runtime_id, chunks in grouped.items():
        result = bundle.progressive.materialize_backing_chunks(
            runtime_id,
            case.query,
            chunks,
            selection_policy="PRA_NATIVE_HYBRID",
        )
        payload_bytes += result.payload_bytes
        sources.add(bundle.runtime_to_source[runtime_id])
    return sources, time.perf_counter() - started, payload_bytes


def _normal(value: str) -> str:
    return "".join(character for character in value.upper() if not character.isspace())


def _answer_metrics(case: ProductionTaskCase, generated: str) -> dict[str, object]:
    normalized = _normal(generated)
    pieces = case.expected_answer.split("+")
    distractor = any(_normal(value) in normalized for value in case.distractor_answers)
    evidence_use = sum(_normal(value) in normalized for value in pieces) / len(pieces)
    return {
        "generated_text": generated,
        "answer_correct": int(evidence_use == 1.0 and not distractor),
        "required_evidence_use": evidence_use,
        "constraint_violation": int(distractor),
        "format_compliant": int("ANSWER=" in generated.upper()),
    }


def _chat_prompt(pra: PRAForCausalLM, query: str, context: str = "") -> str:
    content = (
        ("Authoritative typed task records:\n" + context + "\n\n") if context else ""
    ) + query
    messages = [
        {
            "role": "system",
            "content": (
                "Answer from the active task's authoritative evidence. Output exactly one "
                "line beginning ANSWER= and no explanation."
            ),
        },
        {"role": "user", "content": content},
    ]
    try:
        return pra.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return pra.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _direct_generate(
    pra: PRAForCausalLM,
    prompt: str,
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    """Generate a full visible-context control without PRA head truncation."""

    pra.clear_references()
    encoded = pra.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(pra.device)
    attention_mask = encoded.attention_mask.to(pra.device)
    started = time.perf_counter()
    output = pra.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pra.tokenizer.eos_token_id,
        disable_compile=(pra.device.type == "cuda" and torch.cuda.get_device_capability(pra.device)[0] < 7),
    )
    elapsed = time.perf_counter() - started
    generated = output[:, input_ids.shape[1]:]
    return {
        "text": pra.tokenizer.decode(generated[0], skip_special_tokens=True),
        "prompt_tokens": int(input_ids.shape[1]),
        "generated_tokens": int(generated.shape[1]),
        "generation_seconds": elapsed,
    }


def _lexical_row(case: ProductionTaskCase, policy: TaskScopePolicy, top_k: int) -> dict[str, object]:
    selection = TaskScopeSelector(TaskGraph(case.graph), case.records).select(
        case.active_task_id,
        case.query,
        policy=policy,
        max_records=top_k,
    )
    selected_ids = set(selection.selected_record_ids)
    required = set(case.required_record_ids)
    relevant_tasks = {
        _record_task_id(record)
        for record in case.records
        if record.record_id in required
    } | {case.active_task_id}
    wrong = {
        record.record_id
        for record in selection.selected_records
        if _record_task_id(record) not in relevant_tasks
    }
    return {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "seed": case.seed,
        "scenario": case.scenario.value,
        "confusability": case.confusability.value,
        "condition": _scope_label(policy, lexical=True),
        "router": "lexical_recency_control",
        "scope_policy": policy.value,
        "candidate_records": len(selection.candidate_records),
        "selected_records": len(selected_ids),
        "required_record_recall": len(required & selected_ids) / len(required),
        "cross_task_contamination": len(wrong) / max(len(selected_ids), 1),
        "evidence_selected": int(required <= selected_ids),
        "evidence_materialized": 0,
        "active_native_tokens": 0,
        "routing_seconds": selection.scope_seconds,
    }


def _native_run(
    pra: PRAForCausalLM,
    case: ProductionTaskCase,
    policy: TaskScopePolicy,
    *,
    top_k: int = 8,
    generate: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    pra.clear_references()
    pra.config.routing_mode = "hybrid_iterative"
    pra.config.top_k = top_k
    pra.config.max_unique_chunks = top_k
    bundle = _index_scope(pra, case, policy)
    partition = _scope_partition(case, policy)
    try:
        prompt = _chat_prompt(pra, case.query)
        if generate:
            generated = pra.generate(
                prompt,
                max_new_tokens=max(24, 8 * len(case.required_record_ids)),
                do_sample=False,
                return_details=True,
                pad_token_id=pra.tokenizer.eos_token_id,
            )
            stats = generated.stats
            prompt_tokens = generated.prompt_tokens
            generated_text = generated.text
            generation_seconds = generated.latency_seconds
            routing_seconds = float(stats["routing_seconds"])
            query_seconds = float(stats["query_encoding_seconds"])
            ranking = []
        else:
            routing = pra.route(prompt)
            stats = routing.stats
            prompt_tokens = routing.prompt_tokens
            generated_text = ""
            generation_seconds = 0.0
            routing_seconds = routing.routing_seconds
            query_seconds = routing.query_encoding_seconds
            ranking = _ranking_rows(routing, bundle)
        selected = _selected_rows(stats, bundle)
        inventory_before_materialization = pra.stats()
        materialized_sources, materialization_seconds, materialized_bytes = _materialize(
            bundle, case, selected
        )
        selected_sources = {str(row["source_record_id"]) for row in selected}
        required = set(case.required_record_ids)
        selected_text = "\n".join(str(row["text"]) for row in selected)
        pieces = case.expected_answer.split("+")
        evidence_visible = all(_normal(piece) in _normal(selected_text) for piece in pieces)
        relevant_tasks = {
            _record_task_id(record)
            for record in case.records
            if record.record_id in required
        } | {case.active_task_id}
        wrong = {
            source_id
            for source_id in selected_sources
            if _record_task_id(next(
                record for record in case.records if record.record_id == source_id
            )) not in relevant_tasks
        }
        model_stats = _answer_metrics(case, generated_text) if generate else {}
        inventory_after_materialization = pra.stats()
        row = {
            "protocol": PROTOCOL,
            "case_id": case.case_id,
            "seed": case.seed,
            "scenario": case.scenario.value,
            "confusability": case.confusability.value,
            "condition": _scope_label(policy),
            "router": "paper7_native_hybrid",
            "scope_policy": policy.value,
            "oracle_task_graph": 1,
            "candidate_records": len(partition.candidate_records),
            "selected_records": len(selected_sources),
            "selected_chunks": len(selected),
            "required_records": len(required),
            "required_record_recall": len(required & selected_sources) / len(required),
            "cross_task_contamination": len(wrong) / max(len(selected_sources), 1),
            "evidence_selected": int(required <= selected_sources),
            "evidence_materialized": int(required <= materialized_sources and evidence_visible),
            "evidence_available": int(evidence_visible),
            "prompt_tokens": prompt_tokens,
            "candidate_native_tokens": int(stats.get("candidate_kv_tokens", 0)),
            "requested_native_tokens": int(stats.get("requested_kv_tokens", 0)),
            "active_native_tokens": int(stats.get("materialized_kv_tokens", 0)),
            "routing_index_bytes": int(inventory_before_materialization["routing_index_bytes"]),
            "resident_detail_kv_bytes": int(inventory_before_materialization["resident_detail_kv_bytes"]),
            "post_materialization_routing_index_bytes": int(
                inventory_after_materialization["routing_index_bytes"]
            ),
            "post_materialization_resident_detail_kv_bytes": int(
                inventory_after_materialization["resident_detail_kv_bytes"]
            ),
            "index_seconds": bundle.index_seconds,
            "query_encoding_seconds": query_seconds,
            "routing_seconds": routing_seconds,
            "materialization_seconds": materialization_seconds,
            "materialized_bytes": materialized_bytes,
            "generation_seconds": generation_seconds,
            "top_k": top_k,
            **model_stats,
        }
        trace = [{
            "protocol": PROTOCOL,
            "case_id": case.case_id,
            "condition": _scope_label(policy),
            "task_id": _record_task_id(next(
                record for record in case.records
                if record.record_id == selected_row["source_record_id"]
            )),
            "record_id": selected_row["source_record_id"],
            "backing_chunk_id": selected_row["chunk_id"],
            "routing_score": selected_row["chunk_score"],
            "routing_channel": "native_hybrid",
            "materialization_depth": "native_kv+selected_detail",
            "active_native_tokens": selected_row["token_count"],
            "selected_rank": selected_row["selected_rank"],
            "selected_text": selected_row["text"],
        } for selected_row in selected]
        return row, trace, ranking
    finally:
        bundle.close()
        pra.clear_references()


def _full_model_run(
    pra: PRAForCausalLM,
    case: ProductionTaskCase,
    condition: str,
) -> dict[str, object]:
    pra.clear_references()
    if condition == "FULL_SESSION":
        records = case.records
    elif condition == "FULL_TASK_SCOPE":
        records = _scope_partition(case, TaskScopePolicy.TASK_STRUCTURAL).candidate_records
    elif condition == "COMPACT_ONLY":
        records = case.records
    else:
        raise ValueError(condition)
    if condition == "COMPACT_ONLY":
        compressor = CompressorRegistry()
        context = "\n".join(
            _payload_text(
                compressor.compress(record.record_type, record.payload, unit_limit=2).compact_payload
            )
            for record in records
        )
    else:
        context = "\n".join(
            f"[{record.record_id}] {_payload_text(record.payload)}"
            for record in records
        )
    prompt = _chat_prompt(pra, case.query, context)
    result = _direct_generate(
        pra,
        prompt,
        max_new_tokens=max(24, 8 * len(case.required_record_ids)),
    )
    required = set(case.required_record_ids)
    available = required <= {record.record_id for record in records}
    return {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "seed": case.seed,
        "scenario": case.scenario.value,
        "confusability": case.confusability.value,
        "condition": condition,
        "oracle_task_graph": 1,
        "evidence_selected": int(available),
        "evidence_materialized": int(available),
        "evidence_available": int(available and all(
            _normal(piece) in _normal(context)
            for piece in case.expected_answer.split("+")
        )),
        "prompt_tokens": result["prompt_tokens"],
        "active_native_tokens": 0,
        "routing_seconds": 0.0,
        "materialization_seconds": 0.0,
        "generation_seconds": result["generation_seconds"],
        **_answer_metrics(case, str(result["text"])),
    }


def _native_visible_model_run(
    pra: PRAForCausalLM,
    case: ProductionTaskCase,
    policy: TaskScopePolicy,
    *,
    top_k: int = 8,
) -> dict[str, object]:
    """Route natively, then consume the exact selected detail as visible text."""

    route_row, trace, _ = _native_run(
        pra, case, policy, top_k=top_k, generate=False
    )
    context = "\n".join(
        f"[{row['record_id']}] {row['selected_text']}" for row in trace
    )
    prompt = _chat_prompt(pra, case.query, context)
    generated = _direct_generate(
        pra,
        prompt,
        max_new_tokens=max(24, 8 * len(case.required_record_ids)),
    )
    return {
        **route_row,
        "condition": _scope_label(policy) + "_VISIBLE",
        "consumption_mode": "typed_selected_detail",
        "prompt_tokens": generated["prompt_tokens"],
        "generation_seconds": generated["generation_seconds"],
        **_answer_metrics(case, str(generated["text"])),
    }


def _run_join_model_capacity(pra: PRAForCausalLM) -> None:
    """Measure bounded model success where fan-in and K remain practical."""

    rows = []
    for fan_in in (2, 4, 8):
        case = join_capacity_case(fan_in, seed=11)
        for budget in (2, 4, 8):
            row = _native_visible_model_run(
                pra,
                case,
                TaskScopePolicy.TASK_STRUCTURAL,
                top_k=budget,
            )
            rows.append({
                "case_id": case.case_id,
                "seed": case.seed,
                "fan_in": fan_in,
                "budget_chunks": budget,
                "evidence_available": row["evidence_available"],
                "required_record_recall": row["required_record_recall"],
                "model_success": row["answer_correct"],
                "required_evidence_use": row["required_evidence_use"],
                "selected_native_tokens": row["requested_native_tokens"],
                "prompt_tokens": row["prompt_tokens"],
            })
            print(
                f"join-model fan_in={fan_in} K={budget} success={row['answer_correct']}",
                flush=True,
            )
    _write_csv(RESULTS / "join_capacity_model_results.csv", rows)


def _route_main(pra: PRAForCausalLM, cases: Sequence[ProductionTaskCase]) -> None:
    checkpoint = RESULTS / "production_scope_checkpoint.jsonl"
    existing = _read_jsonl(checkpoint)
    keys = {(row["case_id"], row["condition"]) for row in existing}
    traces = _read_jsonl(RESULTS / "production_pra_routing_trace.jsonl")
    rankings = _read_jsonl(RESULTS / "production_ranking_checkpoint.jsonl")
    for case in cases:
        for policy in SCOPE_POLICIES:
            lexical = _lexical_row(case, policy, top_k=8)
            key = (case.case_id, lexical["condition"])
            if key not in keys:
                _append_jsonl(checkpoint, lexical)
                existing.append(lexical)
                keys.add(key)
            condition = _scope_label(policy)
            if (case.case_id, condition) in keys:
                continue
            row, selected_trace, ranking = _native_run(pra, case, policy, top_k=8)
            _append_jsonl(checkpoint, row)
            existing.append(row)
            keys.add((case.case_id, condition))
            for trace in selected_trace:
                _append_jsonl(RESULTS / "production_pra_routing_trace.jsonl", trace)
                traces.append(trace)
            _append_jsonl(RESULTS / "production_ranking_checkpoint.jsonl", {
                "case_id": case.case_id,
                "condition": condition,
                "rows": ranking,
            })
            rankings.append({"case_id": case.case_id, "condition": condition, "rows": ranking})
            print(f"route {case.case_id} {condition}", flush=True)
    _write_csv(RESULTS / "production_pra_scope_results.csv", existing)


def _route_join_capacity(pra: PRAForCausalLM) -> None:
    output = []
    for fan_in in (2, 4, 8, 16, 32):
        for seed in SEEDS:
            case = join_capacity_case(fan_in, seed=seed)
            _, _, ranking = _native_run(
                pra,
                case,
                TaskScopePolicy.TASK_STRUCTURAL,
                top_k=min(32, fan_in + 1),
            )
            required = set(case.required_record_ids)
            for budget in (2, 4, 8, 16, 32):
                chosen = ranking[:budget]
                selected = {str(row["source_record_id"]) for row in chosen}
                output.append({
                    "protocol": PROTOCOL,
                    "case_id": case.case_id,
                    "seed": seed,
                    "fan_in": fan_in,
                    "budget_chunks": budget,
                    "predecessor_recall": len(required & selected) / len(required),
                    "join_complete": int(required <= selected),
                    "selected_native_tokens": sum(
                        int(row["logical_end"]) - int(row["logical_start"])
                        for row in chosen
                    ),
                    "candidate_records": len(case.records),
                    "model_success": "",
                    "accounting_source": "encoded_native_chunk_spans",
                })
            print(f"join fan_in={fan_in} seed={seed}", flush=True)
    _write_csv(RESULTS / "join_capacity_curve.csv", output)


def _route_dag_shapes(pra: PRAForCausalLM) -> None:
    rows = []
    for shape in DAGShape:
        for seed in SEEDS:
            case = dag_shape_case(shape, seed=seed)
            graph = TaskGraph(case.graph)
            tasks = graph.tasks
            edges = sum(len(task.depends_on) + len(task.after) for task in tasks.values())
            for policy in (TaskScopePolicy.SESSION, TaskScopePolicy.TASK_STRUCTURAL):
                row, _, _ = _native_run(pra, case, policy, top_k=8)
                rows.append({
                    **row,
                    "dag_shape": shape.value,
                    "task_count": len(tasks),
                    "edge_count": edges,
                    "edge_density": edges / max(len(tasks) * (len(tasks) - 1) / 2, 1),
                    "structural_closure_size": len(graph.structural_closure(case.active_task_id)),
                })
            print(f"dag {shape.value} seed={seed}", flush=True)
    _write_csv(RESULTS / "dag_shape_results.csv", rows)


def _run_model(pra: PRAForCausalLM, cases: Sequence[ProductionTaskCase]) -> None:
    checkpoint = RESULTS / "oracle_model_checkpoint.jsonl"
    rows = _read_jsonl(checkpoint)
    keys = {(row["case_id"], row["condition"]) for row in rows}
    conditions = (
        "FULL_SESSION",
        "FULL_TASK_SCOPE",
        "COMPACT_ONLY",
        "PRA_SESSION",
        "PRA_TASK_LOCAL",
        "PRA_TASK_STRUCT",
        "PRA_SESSION_VISIBLE",
        "PRA_TASK_LOCAL_VISIBLE",
        "PRA_TASK_STRUCT_VISIBLE",
    )
    policy_by_condition = {
        "PRA_SESSION": TaskScopePolicy.SESSION,
        "PRA_TASK_LOCAL": TaskScopePolicy.TASK_LOCAL,
        "PRA_TASK_STRUCT": TaskScopePolicy.TASK_STRUCTURAL,
    }
    visible_policy_by_condition = {
        "PRA_SESSION_VISIBLE": TaskScopePolicy.SESSION,
        "PRA_TASK_LOCAL_VISIBLE": TaskScopePolicy.TASK_LOCAL,
        "PRA_TASK_STRUCT_VISIBLE": TaskScopePolicy.TASK_STRUCTURAL,
    }
    for case in cases:
        for condition in conditions:
            if (case.case_id, condition) in keys:
                continue
            if condition in visible_policy_by_condition:
                row = _native_visible_model_run(
                    pra, case, visible_policy_by_condition[condition]
                )
            elif condition in policy_by_condition:
                row, _, _ = _native_run(
                    pra,
                    case,
                    policy_by_condition[condition],
                    top_k=8,
                    generate=True,
                )
            else:
                row = _full_model_run(pra, case, condition)
            _append_jsonl(checkpoint, row)
            rows.append(row)
            keys.add((case.case_id, condition))
            print(
                f"model {case.case_id} {condition} correct={row['answer_correct']}",
                flush=True,
            )
    _write_csv(RESULTS / "oracle_model_task_results.csv", rows)
    decomposition = [{
        "case_id": row["case_id"],
        "condition": row["condition"],
        "evidence_selected": row.get("evidence_selected", 0),
        "evidence_materialized": row.get("evidence_materialized", 0),
        "evidence_available": row.get("evidence_available", 0),
        "answer_correct": row.get("answer_correct", 0),
        "required_evidence_use": row.get("required_evidence_use", 0),
        "answer_correct_given_evidence": (
            row.get("answer_correct", 0) if row.get("evidence_available", 0) else ""
        ),
    } for row in rows]
    _write_csv(RESULTS / "evidence_to_answer_decomposition.csv", decomposition)
    if len(cases) == len(production_task_cases()):
        _run_join_model_capacity(pra)


def _session_replay_artifact(cases: Sequence[ProductionTaskCase]) -> None:
    rows = []
    with tempfile.TemporaryDirectory(prefix="paper8-replay-") as directory:
        service = LocalSessionService(Path(directory))
        for case in cases:
            state = service.create_session("paper8-user", case.case_id)
            state = state._next(tasks=case.graph)
            service.save_session(state, expected_version=1)
            for record in case.records:
                state = service.append_record("paper8-user", case.case_id, record)
            before = TaskScopeSelector(TaskGraph(state.tasks), state.records).partition(
                case.active_task_id, policy=TaskScopePolicy.TASK_STRUCTURAL
            )
            restored = LocalSessionService(Path(directory)).get_session(
                "paper8-user", case.case_id
            )
            after = TaskScopeSelector(TaskGraph(restored.tasks), restored.records).partition(
                case.active_task_id, policy=TaskScopePolicy.TASK_STRUCTURAL
            )
            rows.append({
                "case_id": case.case_id,
                "selection_equal": int(before.admitted_record_ids == after.admitted_record_ids),
                "task_scope_equal": int(before.admitted_task_ids == after.admitted_task_ids),
                "provenance_equal": int(
                    [record.selection_provenance for record in state.records]
                    == [record.selection_provenance for record in restored.records]
                ),
                "model_cache_persisted": 0,
            })
    _write_csv(RESULTS / "session_replay_equivalence.csv", rows)


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return statistics.fmean(values) if values else math.nan


def _bootstrap_delta(
    left: Sequence[float], right: Sequence[float], *, seed: int = 2026
) -> tuple[float, float, float]:
    if len(left) != len(right) or not left:
        return math.nan, math.nan, math.nan
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    draws = [
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(5000)
    ]
    draws.sort()
    return statistics.fmean(differences), draws[125], draws[4874]


def _postprocess() -> None:
    import matplotlib.pyplot as plt

    route_rows = _read_jsonl(RESULTS / "production_scope_checkpoint.jsonl")
    model_rows = _read_jsonl(RESULTS / "oracle_model_checkpoint.jsonl")
    ranking_rows = _read_jsonl(RESULTS / "production_ranking_checkpoint.jsonl")
    if not route_rows:
        raise RuntimeError("Run the route phase before postprocessing.")

    conf_rows = []
    for conf in TaskConfusability:
        session = [
            row for row in route_rows
            if row["confusability"] == conf.value and row["condition"] == "PRA_SESSION"
        ]
        structural = [
            row for row in route_rows
            if row["confusability"] == conf.value and row["condition"] == "PRA_TASK_STRUCT"
        ]
        by_case_session = {row["case_id"]: row for row in session}
        by_case_struct = {row["case_id"]: row for row in structural}
        ids = sorted(set(by_case_session) & set(by_case_struct))
        recall_delta = _bootstrap_delta(
            [float(by_case_struct[key]["required_record_recall"]) for key in ids],
            [float(by_case_session[key]["required_record_recall"]) for key in ids],
        )
        contamination_delta = _bootstrap_delta(
            [float(by_case_session[key]["cross_task_contamination"]) for key in ids],
            [float(by_case_struct[key]["cross_task_contamination"]) for key in ids],
        )
        evidence_delta = _bootstrap_delta(
            [float(by_case_struct[key]["evidence_available"]) for key in ids],
            [float(by_case_session[key]["evidence_available"]) for key in ids],
        )
        model_session = {
            row["case_id"]: row for row in model_rows
            if row["confusability"] == conf.value and row["condition"] == "PRA_SESSION_VISIBLE"
        }
        model_struct = {
            row["case_id"]: row for row in model_rows
            if row["confusability"] == conf.value and row["condition"] == "PRA_TASK_STRUCT_VISIBLE"
        }
        model_ids = sorted(set(model_session) & set(model_struct))
        model_delta = _bootstrap_delta(
            [float(model_struct[key]["answer_correct"]) for key in model_ids],
            [float(model_session[key]["answer_correct"]) for key in model_ids],
        )
        conf_rows.append({
            "confusability": conf.value,
            "n": len(ids),
            "task_aware_recall_delta": recall_delta[0],
            "task_aware_recall_delta_ci_low": recall_delta[1],
            "task_aware_recall_delta_ci_high": recall_delta[2],
            "task_aware_evidence_delta": evidence_delta[0],
            "task_aware_evidence_delta_ci_low": evidence_delta[1],
            "task_aware_evidence_delta_ci_high": evidence_delta[2],
            "session_minus_struct_contamination": contamination_delta[0],
            "contamination_delta_ci_low": contamination_delta[1],
            "contamination_delta_ci_high": contamination_delta[2],
            "task_aware_model_accuracy_delta": model_delta[0],
            "model_accuracy_delta_ci_low": model_delta[1],
            "model_accuracy_delta_ci_high": model_delta[2],
        })
    _write_csv(RESULTS / "semantic_confusability_results.csv", conf_rows)

    ranking_by_key = {
        (row["case_id"], row["condition"]): row["rows"] for row in ranking_rows
    }
    cases = {case.case_id: case for case in production_task_cases()}
    frontier_specs = (
        ("session_shallow", "PRA_SESSION", 2),
        ("structural_selected", "PRA_TASK_STRUCT", 4),
        ("structural_native", "PRA_TASK_STRUCT", 8),
        ("local_deep", "PRA_TASK_LOCAL", 8),
    )
    frontier = []
    for case_id, case in cases.items():
        required = set(case.required_record_ids)
        for name, condition, budget in frontier_specs:
            ranking = ranking_by_key.get((case_id, condition), [])[:budget]
            selected = {row["source_record_id"] for row in ranking}
            tokens = sum(int(row["logical_end"]) - int(row["logical_start"]) for row in ranking)
            frontier.append({
                "case_id": case_id,
                "configuration": name,
                "scope_condition": condition,
                "chunk_budget": budget,
                "required_record_recall": len(required & selected) / len(required),
                "selected_native_tokens": tokens,
                "prompt_tokens": next((
                    int(row.get("prompt_tokens", 0)) for row in route_rows
                    if row["case_id"] == case_id and row["condition"] == condition
                ), 0),
                "accounting_source": "production_encoded_chunk_spans",
            })
    _write_csv(RESULTS / "real_scope_detail_frontier.csv", frontier)
    native_rows = [
        {key: row.get(key, "") for key in (
            "case_id", "condition", "candidate_native_tokens", "requested_native_tokens",
            "active_native_tokens", "routing_index_bytes", "resident_detail_kv_bytes",
            "prompt_tokens", "index_seconds", "routing_seconds", "materialization_seconds",
        )}
        for row in route_rows if row.get("router") == "paper7_native_hybrid"
    ]
    _write_csv(RESULTS / "real_native_kv_accounting.csv", native_rows)

    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [row["confusability"].upper() for row in conf_rows]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].bar(labels, [row["task_aware_evidence_delta"] for row in conf_rows], color="#2A9D8F")
    axes[0].set_ylabel("Structural - session evidence")
    axes[1].bar(labels, [row["session_minus_struct_contamination"] for row in conf_rows], color="#E9C46A")
    axes[1].set_ylabel("Session - structural contamination")
    axes[2].bar(labels, [row["task_aware_model_accuracy_delta"] for row in conf_rows], color="#4C78A8")
    axes[2].set_ylabel("Structural - session accuracy")
    for axis in axes:
        axis.set_xlabel("Task confusability")
        axis.axhline(0, color="black", linewidth=0.7)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"production_confusability.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    production_conditions = ("PRA_SESSION", "PRA_TASK_LOCAL", "PRA_TASK_STRUCT")
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.35))
    short_labels = ("Session", "Local", "Structural")
    axes[0].bar(
        short_labels,
        [_mean([row for row in route_rows if row["condition"] == condition], "evidence_available") for condition in production_conditions],
        color=("#E76F51", "#4C78A8", "#2A9D8F"),
    )
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Exact selected evidence availability")
    axes[1].bar(
        short_labels,
        [_mean([row for row in route_rows if row["condition"] == condition], "requested_native_tokens") for condition in production_conditions],
        color=("#E76F51", "#4C78A8", "#2A9D8F"),
    )
    axes[1].set_ylabel("Requested native K/V tokens")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"production_scope_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    frontier_groups = defaultdict(list)
    for row in frontier:
        frontier_groups[row["configuration"]].append(row)
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for name, rows in frontier_groups.items():
        axis.scatter(
            _mean(rows, "selected_native_tokens"),
            _mean(rows, "required_record_recall"),
            s=70,
            label=name.replace("_", " "),
        )
    axis.set_xlabel("Mean selected native chunk tokens")
    axis.set_ylabel("Required-record recall")
    axis.set_ylim(-0.03, 1.03)
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(FIGURES / f"production_scope_detail_frontier.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)

    join_path = RESULTS / "join_capacity_curve.csv"
    if join_path.is_file():
        with join_path.open(encoding="utf-8") as handle:
            joins = list(csv.DictReader(handle))
        join_model_path = RESULTS / "join_capacity_model_results.csv"
        if join_model_path.is_file():
            with join_model_path.open(encoding="utf-8") as handle:
                model_join = list(csv.DictReader(handle))
            model_by_key = {
                (row["seed"], row["fan_in"], row["budget_chunks"]): row
                for row in model_join
            }
            for row in joins:
                model = model_by_key.get((row["seed"], row["fan_in"], row["budget_chunks"]))
                if model is not None:
                    row["model_success"] = model["model_success"]
            _write_csv(RESULTS / "join_capacity_curve.csv", joins)
        figure, axis = plt.subplots(figsize=(6.6, 3.8))
        for budget in (2, 4, 8, 16, 32):
            points = []
            for fan_in in (2, 4, 8, 16, 32):
                subset = [row for row in joins if int(row["budget_chunks"]) == budget and int(row["fan_in"]) == fan_in]
                points.append(_mean(subset, "predecessor_recall"))
            axis.plot((2, 4, 8, 16, 32), points, marker="o", label=f"K={budget}")
        axis.set_xscale("log", base=2)
        axis.set_xticks((2, 4, 8, 16, 32), labels=("2", "4", "8", "16", "32"))
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Join fan-in")
        axis.set_ylabel("Predecessor recall")
        axis.legend(ncol=3, fontsize=8)
        figure.tight_layout()
        for suffix in ("pdf", "png"):
            figure.savefig(FIGURES / f"production_join_capacity.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(figure)

    route_summary = {
        condition: {
            "required_record_recall": _mean(
                [row for row in route_rows if row["condition"] == condition],
                "required_record_recall",
            ),
            "cross_task_contamination": _mean(
                [row for row in route_rows if row["condition"] == condition],
                "cross_task_contamination",
            ),
            "evidence_available": _mean(
                [row for row in route_rows if row["condition"] == condition],
                "evidence_available",
            ),
            "requested_native_tokens": _mean(
                [row for row in route_rows if row["condition"] == condition],
                "requested_native_tokens",
            ),
        }
        for condition in ("PRA_SESSION", "PRA_TASK_LOCAL", "PRA_TASK_STRUCT")
    }
    model_accuracy = {
        condition: _mean(
            [row for row in model_rows if row["condition"] == condition],
            "answer_correct",
        )
        for condition in (
            "FULL_SESSION", "FULL_TASK_SCOPE", "COMPACT_ONLY",
            "PRA_SESSION", "PRA_TASK_LOCAL", "PRA_TASK_STRUCT",
            "PRA_SESSION_VISIBLE", "PRA_TASK_LOCAL_VISIBLE", "PRA_TASK_STRUCT_VISIBLE",
        )
    }
    model_evidence_use = {
        condition: _mean(
            [row for row in model_rows if row["condition"] == condition],
            "required_evidence_use",
        )
        for condition in model_accuracy
    }
    summary = {
        "protocol": PROTOCOL,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "seeds": list(SEEDS),
        "main_cases": len(cases),
        "route_rows": len(route_rows),
        "model_rows": len(model_rows),
        "confusability": conf_rows,
        "route": route_summary,
        "model_accuracy": model_accuracy,
        "model_evidence_use": model_evidence_use,
    }
    (RESULTS / "production_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    macros = {
        "ProductionCases": len(cases),
        "ProductionSessionEvidence": route_summary["PRA_SESSION"]["evidence_available"] * 100,
        "ProductionLocalEvidence": route_summary["PRA_TASK_LOCAL"]["evidence_available"] * 100,
        "ProductionStructuralEvidence": route_summary["PRA_TASK_STRUCT"]["evidence_available"] * 100,
        "ProductionSessionContamination": route_summary["PRA_SESSION"]["cross_task_contamination"] * 100,
        "ProductionSessionNativeTokens": route_summary["PRA_SESSION"]["requested_native_tokens"],
        "ProductionStructuralNativeTokens": route_summary["PRA_TASK_STRUCT"]["requested_native_tokens"],
        "ProductionFullSessionAccuracy": model_accuracy["FULL_SESSION"] * 100,
        "ProductionFullTaskAccuracy": model_accuracy["FULL_TASK_SCOPE"] * 100,
        "ProductionVisibleSessionAccuracy": model_accuracy["PRA_SESSION_VISIBLE"] * 100,
        "ProductionVisibleLocalAccuracy": model_accuracy["PRA_TASK_LOCAL_VISIBLE"] * 100,
        "ProductionVisibleStructuralAccuracy": model_accuracy["PRA_TASK_STRUCT_VISIBLE"] * 100,
        "ProductionDirectNativeAccuracy": max(
            model_accuracy["PRA_SESSION"],
            model_accuracy["PRA_TASK_LOCAL"],
            model_accuracy["PRA_TASK_STRUCT"],
        ) * 100,
        "ProductionLowEvidenceDelta": conf_rows[0]["task_aware_evidence_delta"] * 100,
        "ProductionMediumEvidenceDelta": conf_rows[1]["task_aware_evidence_delta"] * 100,
        "ProductionHighEvidenceDelta": conf_rows[2]["task_aware_evidence_delta"] * 100,
    }
    macro_lines = ["% Generated by run_production_pra.py; do not edit by hand."]
    for name, value in macros.items():
        rendered = str(value) if isinstance(value, int) else f"{value:.1f}"
        macro_lines.append(f"\\newcommand{{\\{name}}}{{{rendered}}}")
    (RESULTS / "generated_production_pra_results.tex").write_text(
        "\n".join(macro_lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("route", "model", "postprocess", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--skip-join-dag", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = list(production_task_cases())
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    _session_replay_artifact(cases)
    pra = None
    if args.phase in {"route", "model", "all"}:
        pra = _load_model(args.device)
    if args.phase in {"route", "all"}:
        _route_main(pra, cases)
        if not args.skip_join_dag and args.case_limit is None:
            _route_join_capacity(pra)
            _route_dag_shapes(pra)
    if args.phase in {"model", "all"}:
        _run_model(pra, cases)
    if args.phase in {"postprocess", "all"}:
        _postprocess()


if __name__ == "__main__":
    main()
