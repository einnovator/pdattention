"""Replay frozen Paper-8 routing across native materialization geometries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.task_production_cases import ProductionTaskCase, production_task_cases
from pra_hf import evidence_token_intervals, intervals_cover
from pra_hf.task_scope import TaskScopePolicy

from experiments.paper8_tasks.run_production_pra import (
    FIGURES,
    MODEL_ID,
    MODEL_REVISION,
    RESULTS,
    _answer_metrics,
    _chat_prompt,
    _direct_generate,
    _index_scope,
    _load_model,
    _normal,
    _payload_text,
    _read_jsonl,
)


PROTOCOL = "paper8-native-consumption-geometry-v1"
CHECKPOINT = RESULTS / "native_geometry_checkpoint.jsonl"
LATE_ONLY = (27,)
SPARSE_MULTI = (11, 17, 23, 27)
ALL_LAYERS = tuple(sorted(set((*LATE_ONLY, *SPARSE_MULTI))))
WIDTHS = (32, 64, 128, 256, 512)


def _append_jsonl(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _trace_by_case() -> dict[str, list[dict[str, object]]]:
    rows = _read_jsonl(RESULTS / "production_pra_routing_trace.jsonl")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "PRA_TASK_STRUCT":
            grouped[str(row["case_id"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["selected_rank"]))
    return grouped


def _source_uri(bundle, source_id: str) -> str:
    runtime_id = bundle.source_to_runtime[source_id]
    return bundle.progressive.registry.backing_reference_handles[runtime_id].uri


def _annotations(pra, bundle, case: ProductionTaskCase):
    rows = {}
    for annotation in case.evidence_annotations:
        runtime_id = bundle.source_to_runtime[annotation.record_id]
        payload = bundle.progressive.runtime.store.get(
            runtime_id, scope=bundle.progressive.runtime.scope
        )
        text = _payload_text(payload)
        rows[annotation.record_id] = evidence_token_intervals(
            pra.tokenizer,
            text,
            answer=annotation.answer,
            semantic_anchors=annotation.semantic_anchors,
        )
    return rows


def _identity_digest(frozen) -> str:
    payload = json.dumps(frozen.source_identity, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _condition_name(width: int | None, layers: tuple[int, ...], full: bool) -> str:
    depth = "LATE_ONLY" if layers == LATE_ONLY else "SPARSE_MULTI"
    span = "FULL_SELECTED_RECORD" if full else f"W{width}"
    return f"NATIVE_{span}_{depth}"


def _geometry_row(
    pra,
    bundle,
    case: ProductionTaskCase,
    frozen,
    annotated,
    *,
    width: int | None,
    layers: tuple[int, ...],
    full: bool,
) -> dict[str, object]:
    plan = pra.plan_native_materialization(
        frozen,
        target_span_tokens=width,
        full_selected_record=full,
        consumption_layers=layers,
    )
    answer_covered = []
    semantic_covered = []
    full_covered = []
    for annotation in case.evidence_annotations:
        uri = _source_uri(bundle, annotation.record_id)
        spans = annotated[annotation.record_id]
        answer_covered.append(intervals_cover(plan.intervals, uri, spans.answer))
        semantic_covered.append(intervals_cover(plan.intervals, uri, spans.semantic))
        full_covered.append(intervals_cover(plan.intervals, uri, spans.full_record))
    prompt = _chat_prompt(pra, case.query)
    generated = pra.generate_with_native_plan(
        prompt,
        plan,
        max_new_tokens=max(24, 8 * len(case.required_record_ids)),
        do_sample=False,
        return_details=True,
        pad_token_id=pra.tokenizer.eos_token_id,
    )
    return {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "seed": case.seed,
        "scenario": case.scenario.value,
        "confusability": case.confusability.value,
        "condition": _condition_name(width, layers, full),
        "routing_frozen": 1,
        "frozen_selection_digest": _identity_digest(frozen),
        "selected_record_correct": 1,
        "answer_bearing_chunk_selected": int(all(answer_covered)),
        "answer_token_present": int(all(answer_covered)),
        "minimal_semantic_interval_covered": int(all(semantic_covered)),
        "contextually_sufficient_evidence": int(all(semantic_covered)),
        "full_record_covered": int(all(full_covered)),
        "target_span_tokens": "full" if full else width,
        "consumption_depth": "late_only" if layers == LATE_ONLY else "sparse_multi",
        "consumption_layers": json.dumps(layers),
        "selected_intervals": len(plan.intervals),
        "requested_native_tokens": plan.unique_native_tokens,
        "unique_materialized_native_tokens": plan.unique_native_tokens,
        "active_native_tokens": plan.unique_native_tokens * len(layers),
        "resident_detail_kv_bytes": generated.stats["resident_detail_kv_bytes"],
        "prompt_tokens": generated.prompt_tokens,
        "generation_seconds": generated.latency_seconds,
        **_answer_metrics(case, generated.text),
    }


def _visible_full_selected_row(pra, bundle, case, frozen) -> dict[str, object]:
    selected_uris = {row.reference_uri for row in frozen.anchors}
    uri_to_source = {
        handle.uri: bundle.runtime_to_source[runtime_id]
        for runtime_id, handle in bundle.progressive.registry.backing_reference_handles.items()
    }
    source_ids = [uri_to_source[uri] for uri in sorted(selected_uris)]
    context = []
    for source_id in source_ids:
        runtime_id = bundle.source_to_runtime[source_id]
        payload = bundle.progressive.runtime.store.get(
            runtime_id, scope=bundle.progressive.runtime.scope
        )
        context.append(_payload_text(payload))
    generated = _direct_generate(
        pra,
        _chat_prompt(pra, case.query, "\n".join(context)),
        max_new_tokens=max(24, 8 * len(case.required_record_ids)),
    )
    return {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "seed": case.seed,
        "scenario": case.scenario.value,
        "confusability": case.confusability.value,
        "condition": "VISIBLE_FULL_SELECTED_RECORD",
        "routing_frozen": 1,
        "frozen_selection_digest": _identity_digest(frozen),
        "selected_record_correct": 1,
        "answer_bearing_chunk_selected": 1,
        "answer_token_present": 1,
        "minimal_semantic_interval_covered": 1,
        "contextually_sufficient_evidence": 1,
        "full_record_covered": 1,
        "target_span_tokens": "visible_full",
        "consumption_depth": "visible",
        "consumption_layers": "[]",
        "selected_intervals": len(selected_uris),
        "requested_native_tokens": 0,
        "unique_materialized_native_tokens": 0,
        "active_native_tokens": 0,
        "resident_detail_kv_bytes": 0,
        "prompt_tokens": generated["prompt_tokens"],
        "generation_seconds": generated["generation_seconds"],
        **_answer_metrics(case, str(generated["text"])),
    }


def run(device: str) -> list[dict[str, object]]:
    cases = production_task_cases()
    traces = _trace_by_case()
    existing = _read_jsonl(CHECKPOINT)
    keys = {(row["case_id"], row["condition"]) for row in existing}
    pra = _load_model(
        device,
        consumption_layers=ALL_LAYERS,
        max_materialized_tokens=512,
    )
    conditions = [
        *((width, LATE_ONLY, False) for width in WIDTHS),
        (None, LATE_ONLY, True),
        (128, SPARSE_MULTI, False),
        (256, SPARSE_MULTI, False),
        (None, SPARSE_MULTI, True),
    ]
    for case in cases:
        selected_trace = traces.get(case.case_id, [])
        selected_ids = {str(row["record_id"]) for row in selected_trace}
        eligible = set(case.required_record_ids) <= selected_ids
        if not eligible:
            for width, layers, full in conditions:
                condition = _condition_name(width, layers, full)
                if (case.case_id, condition) not in keys:
                    row = {
                        "protocol": PROTOCOL,
                        "case_id": case.case_id,
                        "seed": case.seed,
                        "scenario": case.scenario.value,
                        "confusability": case.confusability.value,
                        "condition": condition,
                        "routing_frozen": 1,
                        "selected_record_correct": 0,
                        "answer_correct": 0,
                        "excluded_from_conditional": 1,
                    }
                    _append_jsonl(CHECKPOINT, row)
                    existing.append(row)
                    keys.add((case.case_id, condition))
            continue
        bundle = _index_scope(pra, case, TaskScopePolicy.TASK_STRUCTURAL)
        try:
            frozen = pra.freeze_native_selection(selected_trace)
            annotated = _annotations(pra, bundle, case)
            for width, layers, full in conditions:
                condition = _condition_name(width, layers, full)
                if (case.case_id, condition) in keys:
                    continue
                row = _geometry_row(
                    pra, bundle, case, frozen, annotated,
                    width=width, layers=layers, full=full,
                )
                _append_jsonl(CHECKPOINT, row)
                existing.append(row)
                keys.add((case.case_id, condition))
                print(
                    f"geometry {case.case_id} {condition} "
                    f"tokens={row['unique_materialized_native_tokens']} "
                    f"correct={row['answer_correct']}",
                    flush=True,
                )
            visible = "VISIBLE_FULL_SELECTED_RECORD"
            if (case.case_id, visible) not in keys:
                row = _visible_full_selected_row(pra, bundle, case, frozen)
                _append_jsonl(CHECKPOINT, row)
                existing.append(row)
                keys.add((case.case_id, visible))
        finally:
            bundle.close()
            pra.clear_references()
    return existing


def postprocess(rows: Sequence[Mapping[str, object]]) -> None:
    complete = [row for row in rows if "excluded_from_conditional" not in row]
    native = [row for row in complete if str(row["condition"]).startswith("NATIVE_")]
    _write_csv(RESULTS / "native_materialization_width_results.csv", [
        row for row in native if row["consumption_depth"] == "late_only"
    ])
    _write_csv(RESULTS / "native_consumption_depth_results.csv", native)
    _write_csv(RESULTS / "native_semantic_interval_coverage.csv", [{
        key: row[key]
        for key in (
            "case_id", "condition", "selected_record_correct",
            "answer_token_present", "minimal_semantic_interval_covered",
            "full_record_covered", "answer_correct",
        )
    } for row in native])
    summary = []
    for condition in sorted({str(row["condition"]) for row in complete}):
        values = [row for row in complete if row["condition"] == condition]
        all_rows = [row for row in rows if row["condition"] == condition]
        sufficient = [row for row in values if row.get("contextually_sufficient_evidence") == 1]
        summary.append({
            "condition": condition,
            "n_total": len(all_rows),
            "n_correctly_routed": len(values),
            "mean_unique_native_tokens": statistics.mean(
                float(row.get("unique_materialized_native_tokens", 0)) for row in values
            ),
            "unconditional_end_to_end_accuracy": sum(float(row.get("answer_correct", 0)) for row in all_rows) / len(all_rows),
            "conditional_consumption_accuracy": sum(float(row.get("answer_correct", 0)) for row in values) / len(values),
            "semantic_sufficiency_rate": sum(float(row.get("contextually_sufficient_evidence", 0)) for row in values) / len(values),
            "accuracy_given_semantic_sufficiency": (
                sum(float(row.get("answer_correct", 0)) for row in sufficient) / len(sufficient)
                if sufficient else 0.0
            ),
        })
    _write_csv(RESULTS / "native_geometry_conditional_accuracy.csv", summary)
    (RESULTS / "native_geometry_summary.json").write_text(
        json.dumps({
            "protocol": PROTOCOL,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "rows": summary,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    by_condition = {row["condition"]: row for row in summary}
    macros = {
        "NativeGeometryCases": int(by_condition["NATIVE_W32_LATE_ONLY"]["n_total"]),
        "NativeGeometryWidthThirtyTwoTokens": by_condition["NATIVE_W32_LATE_ONLY"]["mean_unique_native_tokens"],
        "NativeGeometryFullTokens": by_condition["NATIVE_FULL_SELECTED_RECORD_LATE_ONLY"]["mean_unique_native_tokens"],
        "NativeGeometryWidthThirtyTwoSufficiency": 100 * by_condition["NATIVE_W32_LATE_ONLY"]["semantic_sufficiency_rate"],
        "NativeGeometryFullSufficiency": 100 * by_condition["NATIVE_FULL_SELECTED_RECORD_LATE_ONLY"]["semantic_sufficiency_rate"],
        "NativeGeometryLateAccuracy": 100 * by_condition["NATIVE_FULL_SELECTED_RECORD_LATE_ONLY"]["conditional_consumption_accuracy"],
        "NativeGeometrySparseAccuracy": 100 * by_condition["NATIVE_FULL_SELECTED_RECORD_SPARSE_MULTI"]["conditional_consumption_accuracy"],
        "NativeGeometryVisibleAccuracy": 100 * by_condition["VISIBLE_FULL_SELECTED_RECORD"]["conditional_consumption_accuracy"],
    }
    (RESULTS / "generated_native_geometry_results.tex").write_text(
        "\n".join(
            f"\\newcommand{{\\{name}}}{{{value:.1f}}}"
            if isinstance(value, float) else f"\\newcommand{{\\{name}}}{{{value}}}"
            for name, value in macros.items()
        ) + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    late = [row for row in summary if row["condition"].endswith("LATE_ONLY")]
    order = {f"NATIVE_W{width}_LATE_ONLY": index for index, width in enumerate(WIDTHS)}
    order["NATIVE_FULL_SELECTED_RECORD_LATE_ONLY"] = len(WIDTHS)
    late.sort(key=lambda row: order[row["condition"]])
    x = [float(row["mean_unique_native_tokens"]) for row in late]
    y = [100 * float(row["conditional_consumption_accuracy"]) for row in late]
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    axis.plot(x, y, marker="o", label="late-only native")
    sparse = [row for row in summary if row["condition"].endswith("SPARSE_MULTI")]
    sparse.sort(key=lambda row: float(row["mean_unique_native_tokens"]))
    axis.plot(
        [float(row["mean_unique_native_tokens"]) for row in sparse],
        [100 * float(row["conditional_consumption_accuracy"]) for row in sparse],
        marker="s",
        label="sparse multi-layer native",
    )
    visible = next(row for row in summary if row["condition"] == "VISIBLE_FULL_SELECTED_RECORD")
    axis.axhline(
        100 * float(visible["conditional_consumption_accuracy"]),
        color="#555555", linestyle="--", label="visible selected-record ceiling",
    )
    axis.set_xlabel("Mean unique materialized native tokens")
    axis.set_ylabel("Conditional answer accuracy (%)")
    axis.set_ylim(-3, 103)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / "native_consumption_geometry.png", dpi=180)
    figure.savefig(FIGURES / "native_consumption_geometry.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--postprocess-only", action="store_true")
    args = parser.parse_args()
    rows = _read_jsonl(CHECKPOINT) if args.postprocess_only else run(args.device)
    postprocess(rows)


if __name__ == "__main__":
    main()
