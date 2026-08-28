"""Localize Paper-8 native consumption failures with monotonic HF controls.

The runner freezes the pinned Qwen checkpoint, removes routing from parity
conditions, and moves from complete record scope to the canonical sparse
materialization inherited from Paper 3. Results checkpoint after every case so
the old local GPU can resume safely after interruption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.task_production_cases import ProductionTaskCase, production_task_cases
from pra_hf import (
    FrozenNativeAnchor,
    FrozenNativeSelection,
    PRAConfig,
    PRAForCausalLM,
    evidence_token_intervals,
)
from pra_hf.task_scope import TaskScopePolicy

from experiments.paper8_tasks.run_production_pra import (
    MODEL_ID,
    MODEL_REVISION,
    RESULTS,
    _answer_metrics,
    _chat_prompt,
    _direct_generate,
    _index_scope,
    _payload_text,
    _read_jsonl,
    _scope_partition,
)


PROTOCOL = "paper8-native-bug-ablation-v2"
ALL_LAYERS = tuple(range(28))
SPARSE_LAYERS = (11, 17, 23, 27)
WINDOWS = (512, 256, 128, 96, 64, 32)
OUT = RESULTS / "native_bug_ablation"
CHECKPOINT = OUT / "checkpoint.jsonl"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append(row: Mapping[str, object]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _load(device: str) -> PRAForCausalLM:
    target = torch.device(device)
    dtype = torch.float16 if target.type == "cuda" else torch.float32
    pra = PRAForCausalLM.from_pretrained(
        MODEL_ID,
        pra_config=PRAConfig(
            routing_layer=27,
            consumption_layers=ALL_LAYERS,
            chunk_tokens=32,
            chunk_overlap_tokens=8,
            selected_fraction=None,
            top_k=8,
            max_direct_context=512,
            native_operation_limit=4096,
            max_materialized_tokens=3584,
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


def _rank(logits: torch.Tensor, token_id: int) -> int:
    row = logits[0].float()
    return int((row > row[token_id]).sum().item()) + 1


def _tensor_hash(tensor: torch.Tensor) -> str:
    values = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()[:16]


def _anchor_for_entry(entry, *, span: tuple[int, int] | None = None) -> FrozenNativeAnchor:
    chunks = sorted(
        entry.layer_memory[27].chunks,
        key=lambda row: (row.logical_start, row.logical_end, row.chunk_id),
    )
    start, end = span if span is not None else (chunks[0].logical_start, chunks[0].logical_end)
    source = next(
        (row for row in chunks if row.logical_start <= start < row.logical_end),
        chunks[0],
    )
    return FrozenNativeAnchor(entry.uri, source.chunk_id, int(start), int(end), 1.0, 1.0)


def _anchors_for_sources(pra, bundle, source_ids: Iterable[str]) -> FrozenNativeSelection:
    anchors = []
    for source_id in source_ids:
        runtime_id = bundle.source_to_runtime[source_id]
        uri = bundle.progressive.registry.backing_reference_handles[runtime_id].uri
        entry = pra._handle.cache.get(uri)
        if entry is None:
            raise RuntimeError(f"Missing active cache entry for {source_id}.")
        anchors.append(_anchor_for_entry(entry))
    return FrozenNativeSelection(tuple(anchors))


def _all_scope_sources(bundle) -> tuple[str, ...]:
    return tuple(bundle.source_to_runtime)


def _selected_trace(case_id: str) -> list[dict[str, object]]:
    rows = _read_jsonl(RESULTS / "production_pra_routing_trace.jsonl")
    selected = [
        row for row in rows
        if row.get("case_id") == case_id and row.get("condition") == "PRA_TASK_STRUCT"
    ]
    selected.sort(key=lambda row: int(row["selected_rank"]))
    if not selected:
        raise RuntimeError(f"No frozen production selection exists for {case_id}.")
    return selected


def _visible_context(case: ProductionTaskCase, policy: TaskScopePolicy) -> str:
    records = _scope_partition(case, policy).candidate_records
    return "\n".join(_payload_text(record.payload) for record in records)


def _expected_token(pra, case: ProductionTaskCase) -> int:
    values = pra.tokenizer(
        "ANSWER=" + case.expected_answer,
        add_special_tokens=False,
    ).input_ids
    return int(values[0])


def _visible_logits(pra, prompt: str) -> torch.Tensor:
    pra._handle.configure_memory_layers(set())
    encoded = pra.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        return pra.model(
            input_ids=encoded.input_ids.to(pra.device),
            attention_mask=encoded.attention_mask.to(pra.device),
            use_cache=False,
        ).logits[:, -1, :].detach()


def _plan_row(
    pra,
    case: ProductionTaskCase,
    condition: str,
    frozen: FrozenNativeSelection,
    *,
    width: int | None,
    full: bool,
    layers: tuple[int, ...],
    expected_token: int,
) -> dict[str, object]:
    plan = pra.plan_native_materialization(
        frozen,
        target_span_tokens=width,
        full_selected_record=full,
        consumption_layers=layers,
    )
    prompt = _chat_prompt(pra, case.query)
    logits = pra.native_next_token_logits(prompt, plan)
    rank = _rank(logits, expected_token)
    generated = pra.generate_with_native_plan(
        prompt,
        plan,
        max_new_tokens=max(24, 8 * len(case.required_record_ids)),
        do_sample=False,
        return_details=True,
        pad_token_id=pra.tokenizer.eos_token_id,
    )
    lifetime = generated.stats["memory_lifetime_by_layer"]
    active_calls = [
        row
        for layer in layers
        for row in lifetime.get(layer, ())
        if int(row["active_native_tokens"]) > 0
    ]
    return {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "condition": condition,
        "scope": "native",
        "selected_reference_count": len({row.reference_uri for row in frozen.anchors}),
        "raw_interval_count": plan.raw_interval_count,
        "merged_interval_count": len(plan.intervals),
        "raw_selected_tokens": plan.raw_native_tokens,
        "unique_native_tokens": plan.unique_native_tokens,
        "overlap_removed_tokens": plan.overlap_removed_tokens,
        "duplication_ratio": plan.duplication_ratio,
        "query_position_offset": plan.query_position_offset,
        "layers": json.dumps(layers),
        "layer_count": len(layers),
        "target_span_tokens": "full" if full else width,
        "expected_token_id": expected_token,
        "expected_token_rank": rank,
        "decode_calls_with_memory": len(active_calls),
        "decode_lifetime_pass": int(
            bool(active_calls)
            and all(int(row["active_native_tokens"]) > 0 for row in active_calls)
        ),
        "generated_text": generated.text,
        "generation_seconds": generated.latency_seconds,
        **_answer_metrics(case, generated.text),
    }


def _visible_row(
    pra,
    case: ProductionTaskCase,
    condition: str,
    policy: TaskScopePolicy,
) -> dict[str, object]:
    prompt = _chat_prompt(pra, case.query, _visible_context(case, policy))
    expected = _expected_token(pra, case)
    rank = _rank(_visible_logits(pra, prompt), expected)
    generated = _direct_generate(
        pra,
        prompt,
        max_new_tokens=max(24, 8 * len(case.required_record_ids)),
    )
    return {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "condition": condition,
        "scope": "visible",
        "expected_token_id": expected,
        "expected_token_rank": rank,
        "generated_text": generated["text"],
        "generation_seconds": generated["generation_seconds"],
        **_answer_metrics(case, str(generated["text"])),
    }


def _prefix_equivalence(pra) -> list[dict[str, object]]:
    """Run exact ordinary-prefix/native-KV parity on the pinned checkpoint."""

    pra.clear_references()
    reference_text = "OWNER_CODE = ZYRA\n"
    query = "Question: What is OWNER_CODE?\nAnswer:"
    reference = pra.tokenizer(reference_text, return_tensors="pt", add_special_tokens=False)
    entry = pra._handle.add_reference(
        "diagnostic://owner-code",
        reference.input_ids.to(pra.device),
        text=reference_text,
    )
    frozen = FrozenNativeSelection((_anchor_for_entry(entry),))
    plan = pra.plan_native_materialization(
        frozen,
        full_selected_record=True,
        consumption_layers=ALL_LAYERS,
    )
    visible_ids = pra.tokenizer(
        reference_text + query,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(pra.device)
    pra._handle.configure_memory_layers(set())
    with torch.no_grad():
        visible_logits = pra.model(visible_ids, use_cache=False).logits[:, -1, :]
    native_logits = pra.native_next_token_logits(query, plan)
    expected = int(pra.tokenizer(" ZYRA", add_special_tokens=False).input_ids[0])
    rows = [{
        "condition": "E0_ONE_FULL_REFERENCE",
        "reference_tokens": int(reference.input_ids.shape[1]),
        "query_tokens": int(visible_ids.shape[1] - reference.input_ids.shape[1]),
        "query_position_offset": plan.query_position_offset,
        "max_logit_error": float((visible_logits - native_logits).abs().max().cpu()),
        "mean_logit_error": float((visible_logits - native_logits).abs().mean().cpu()),
        "visible_expected_rank": _rank(visible_logits, expected),
        "native_expected_rank": _rank(native_logits, expected),
        "top_token_equal": int(visible_logits.argmax(-1).item() == native_logits.argmax(-1).item()),
    }]

    positions = torch.arange(reference.input_ids.shape[1], device=pra.device).unsqueeze(0)
    for adapter in pra._handle.adapters.values():
        adapter.begin_capture(positions)
    pra._handle.configure_memory_layers(set())
    with torch.no_grad():
        pra.model(
            reference.input_ids.to(pra.device),
            position_ids=positions,
            use_cache=False,
        )
    for layer, adapter in pra._handle.adapters.items():
        captured = adapter.consume_capture().detail_kv
        chunks = sorted(entry.layer_memory[layer].chunks, key=lambda row: row.logical_start)
        cached_k = torch.cat([row.token_kv.k.to(pra.device) for row in chunks], dim=2)
        cached_v = torch.cat([row.token_kv.v.to(pra.device) for row in chunks], dim=2)
        rows.append({
            "condition": "KV_IDENTITY",
            "layer": layer,
            "token_ids_hash": hashlib.sha256(
                bytes(int(value) % 256 for value in reference.input_ids[0].tolist())
            ).hexdigest()[:16],
            "k_hash": _tensor_hash(cached_k),
            "v_hash": _tensor_hash(cached_v),
            "k_shape": json.dumps(list(cached_k.shape)),
            "v_shape": json.dumps(list(cached_v.shape)),
            "kv_heads": int(cached_k.shape[1]),
            "positions": json.dumps(chunks[0].token_kv.position_ids.tolist()),
            "max_k_error": float((cached_k - captured.k).abs().max().cpu()),
            "mean_k_error": float((cached_k - captured.k).abs().mean().cpu()),
            "max_v_error": float((cached_v - captured.v).abs().max().cpu()),
            "mean_v_error": float((cached_v - captured.v).abs().mean().cpu()),
        })
    pra.clear_references()
    return rows


def _paper3_oracle(pra, bundle, case: ProductionTaskCase) -> FrozenNativeSelection:
    anchors = []
    by_annotation = {row.record_id: row for row in case.evidence_annotations}
    for source_id in case.required_record_ids:
        annotation = by_annotation[source_id]
        runtime_id = bundle.source_to_runtime[source_id]
        uri = bundle.progressive.registry.backing_reference_handles[runtime_id].uri
        entry = pra._handle.cache.get(uri)
        payload = bundle.progressive.runtime.store.get(
            runtime_id,
            scope=bundle.progressive.runtime.scope,
        )
        spans = evidence_token_intervals(
            pra.tokenizer,
            _payload_text(payload),
            answer=annotation.answer,
            semantic_anchors=annotation.semantic_anchors,
        )
        anchors.append(_anchor_for_entry(entry, span=spans.semantic))
    return FrozenNativeSelection(tuple(anchors))


def _run_case(pra, case: ProductionTaskCase) -> list[dict[str, object]]:
    rows = [
        _visible_row(pra, case, "VISIBLE_FULL_SESSION", TaskScopePolicy.SESSION),
        _visible_row(pra, case, "VISIBLE_FULL_TASK_SCOPE", TaskScopePolicy.TASK_STRUCTURAL),
    ]
    expected = _expected_token(pra, case)

    pra.clear_references()
    session = _index_scope(pra, case, TaskScopePolicy.SESSION)
    try:
        rows.append(_plan_row(
            pra,
            case,
            "NATIVE_FULL_SESSION",
            _anchors_for_sources(pra, session, _all_scope_sources(session)),
            width=None,
            full=True,
            layers=ALL_LAYERS,
            expected_token=expected,
        ))
    finally:
        session.close()

    pra.clear_references()
    task = _index_scope(pra, case, TaskScopePolicy.TASK_STRUCTURAL)
    try:
        all_scope = _anchors_for_sources(pra, task, _all_scope_sources(task))
        required = _anchors_for_sources(pra, task, case.required_record_ids)
        selected = pra.freeze_native_selection(_selected_trace(case.case_id))
        selected_sources = {
            task.runtime_to_source[runtime_id]
            for runtime_id, handle in task.progressive.registry.backing_reference_handles.items()
            if handle.uri in {row.reference_uri for row in selected.anchors}
        }
        selected_plus_required = _anchors_for_sources(
            pra,
            task,
            tuple(dict.fromkeys((*selected_sources, *case.required_record_ids))),
        )
        conditions = [
            ("NATIVE_FULL_TASK_SCOPE", all_scope, None, True, ALL_LAYERS),
            ("A1_REQUIRED_RECORDS_FULL", required, None, True, ALL_LAYERS),
            ("A2_SELECTED_RECORDS_FULL", selected, None, True, ALL_LAYERS),
            ("A3_SELECTED_PLUS_REQUIRED_FULL", selected_plus_required, None, True, ALL_LAYERS),
            *((f"A4_A5_SELECTED_WINDOW_{width}", selected, width, False, ALL_LAYERS) for width in WINDOWS),
            ("A6_PAPER3_EVIDENCE_RADIUS0_ALL28", _paper3_oracle(pra, task, case), None, False, ALL_LAYERS),
            ("LAYER_ABLATION_FULL_SELECTED_SPARSE4", selected, None, True, SPARSE_LAYERS),
            ("LAYER_ABLATION_FULL_SELECTED_LATE1", selected, None, True, (27,)),
        ]
        for name, frozen, width, full, layers in conditions:
            rows.append(_plan_row(
                pra,
                case,
                name,
                frozen,
                width=width,
                full=full,
                layers=layers,
                expected_token=expected,
            ))
    finally:
        task.close()
        pra.clear_references()
    return rows


def _publish(rows: Sequence[Mapping[str, object]], prefix_rows: Sequence[Mapping[str, object]]) -> None:
    parity_names = {
        "VISIBLE_FULL_SESSION",
        "NATIVE_FULL_SESSION",
        "VISIBLE_FULL_TASK_SCOPE",
        "NATIVE_FULL_TASK_SCOPE",
    }
    _write_csv(OUT / "native_full_scope_parity.csv", [row for row in rows if row["condition"] in parity_names])
    _write_csv(OUT / "native_prefix_equivalence.csv", list(prefix_rows))
    _write_csv(OUT / "native_progressive_removal.csv", [row for row in rows if str(row["condition"]).startswith(("A1_", "A2_", "A3_", "A4_", "A6_", "LAYER_"))])
    _write_csv(OUT / "native_interval_dedup.csv", [row for row in rows if row.get("scope") == "native"])
    _write_csv(OUT / "native_decode_lifetime.csv", [row for row in rows if row.get("scope") == "native"])
    _write_csv(OUT / "native_next_token_logits.csv", [row for row in rows if "expected_token_rank" in row])
    _write_csv(OUT / "native_position_rebinding_audit.csv", list(prefix_rows))
    _write_csv(OUT / "native_materialization_canonical_paper3.csv", [row for row in rows if str(row["condition"]).startswith("A6_")])
    summary = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        values = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            "examples": len(values),
            "answer_accuracy": sum(float(row.get("answer_correct", 0)) for row in values) / len(values),
            "mean_expected_token_rank": sum(float(row.get("expected_token_rank", 0)) for row in values) / len(values),
            "mean_unique_native_tokens": sum(float(row.get("unique_native_tokens", 0)) for row in values) / len(values),
        }
    manifest = {
        "protocol": PROTOCOL,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "paper3_profile_provenance": {
            "source_branch": "research/paper3-kv-materialization@94d5446",
            "routing_parent_tokens": 32,
            "routing_overlap_tokens": 0,
            "selection": "annotated evidence oracle",
            "materialization": "validation-selected radius 0",
            "consumer_band_transfer": "MuSiQue all_28; no Paper-8-specific retuning",
        },
        "summary": summary,
    }
    (OUT / "native_bug_ablation_summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pra = _load(args.device)
    prefix_rows = _prefix_equivalence(pra)
    completed = set()
    rows = _read_jsonl(CHECKPOINT) if args.resume else []
    if not args.resume and CHECKPOINT.exists():
        CHECKPOINT.unlink()
    completed.update(str(row["case_id"]) for row in rows)
    cases = production_task_cases()
    if args.limit is not None:
        cases = cases[: args.limit]
    for index, case in enumerate(cases, start=1):
        if case.case_id in completed:
            continue
        case_rows = _run_case(pra, case)
        for row in case_rows:
            _append(row)
        rows.extend(case_rows)
        print(f"[{index}/{len(cases)}] {case.case_id}: {len(case_rows)} conditions", flush=True)
    _publish(rows, prefix_rows)
    print(json.dumps({"rows": len(rows), "prefix_rows": len(prefix_rows), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
