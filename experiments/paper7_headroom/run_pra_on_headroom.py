"""Run frozen native PRA routing on a focused official Headroom evaluation subset."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from pra_hf import PRAConfig, PRAForCausalLM


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/papers/shared/results/paper7_records/headroom_cross_eval"
OFFICIAL_PYTHON = Path(r"D:\git\rd\.venv-headroom-037\Scripts\python.exe")
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _evidence_span(tokenizer: Any, source: str, target: str) -> tuple[int, int] | None:
    """Map a case-insensitive text match to source-token coordinates."""
    if not target:
        return None
    char_start = source.casefold().find(target.casefold())
    if char_start < 0:
        return None
    char_stop = char_start + len(target)
    encoded = tokenizer(source, add_special_tokens=False, return_offsets_mapping=True)
    covered = [
        index
        for index, (start, stop) in enumerate(encoded.offset_mapping)
        if stop > char_start and start < char_stop
    ]
    return (covered[0], covered[-1] + 1) if covered else None


def _ranked(routing: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reference in routing.rankings:
        for chunk in reference["chunks"]:
            rows.append({
                "reference_uri": reference["reference_uri"],
                "reference_score": float(reference["reference_score"]),
                **dict(chunk),
            })
    rows.sort(key=lambda row: (-float(row["chunk_score"]), str(row["chunk_id"])))
    return rows


def _covers(rows: Sequence[Mapping[str, Any]], span: tuple[int, int] | None) -> bool:
    if span is None:
        return False
    covered: set[int] = set()
    for row in rows:
        start = int(row.get("logical_start", row.get("token_start", 0)))
        stop = int(row.get("logical_end", row.get("token_end", 0)))
        covered.update(range(max(start, span[0]), min(stop, span[1])))
    return all(position in covered for position in range(*span))


def _selected_tokens(rows: Sequence[Mapping[str, Any]]) -> int:
    covered: set[int] = set()
    for row in rows:
        start = int(row.get("logical_start", row.get("token_start", 0)))
        stop = int(row.get("logical_end", row.get("token_end", 0)))
        covered.update(range(start, stop))
    return len(covered)


def _prepare_official_cases(
    *, refresh: bool = False
) -> tuple[Path, dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    cases = OUTPUT / "headroom_eval_cases.json"
    if not cases.is_file():
        subprocess.run([
            str(OFFICIAL_PYTHON),
            str(Path(__file__).with_name("export_headroom_eval_cases.py")),
            "--output",
            str(cases),
            "--n",
            "8",
        ], cwd=ROOT, check=True)
    profiles = {
        "HEADROOM_OFFICIAL_DEFAULT": (OUTPUT / "headroom_eval_default_raw.json", ()),
        "HEADROOM_OFFICIAL_TUNED": (
            OUTPUT / "headroom_eval_tuned_raw.json",
            ("--max-items", "4", "--without-compaction"),
        ),
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    for condition, (path, extra) in profiles.items():
        if refresh or not path.is_file():
            subprocess.run([
                str(OFFICIAL_PYTHON),
                str(Path(__file__).with_name("headroom_eval_worker.py")),
                "--input",
                str(cases),
                "--output",
                str(path),
                *extra,
            ], cwd=ROOT, check=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[condition] = payload["rows"]
        failures = payload.get("loader_failures", failures)
    return cases, loaded, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases_path, official, loader_failures = _prepare_official_cases(refresh=args.refresh)
    case_payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = list(case_payload["rows"])
    if args.case_limit is not None:
        cases = cases[: args.case_limit]

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pra = PRAForCausalLM.from_pretrained(
        MODEL_ID,
        pra_config=PRAConfig(
            routing_layer=27,
            consumption_layers=(27,),
            chunk_tokens=32,
            chunk_overlap_tokens=8,
            selected_fraction=None,
            top_k=8,
            max_direct_context=96,
            native_operation_limit=512,
            max_materialized_tokens=256,
            context_safety_reserve_tokens=4,
            encoding_block_tokens=128,
            reference_device="cpu",
            routing_depth=2,
            branch_top_k=4,
            beam_size=8,
            max_unique_chunks=8,
        ),
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    pra.model.to(device).eval()

    checkpoint = OUTPUT / "pra_on_headroom_checkpoint.csv"
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint.is_file() and not args.refresh:
        with checkpoint.open(encoding="utf-8-sig", newline="") as handle:
            completed = {
                f"{row['dataset']}::{row['case_id']}": dict(row)
                for row in csv.DictReader(handle)
            }
    for index, case in enumerate(cases, start=1):
        key = f"{case['dataset']}::{case['case_id']}"
        if key in completed:
            continue
        context = str(case["context"])
        truth = str(case.get("evidence_target", case["ground_truth"]))
        truth_span = _evidence_span(pra.tokenizer, context, truth)
        pra.clear_references()
        ingestion_started = time.perf_counter()
        handle = pra.add_reference(f"headroom-eval://{case['dataset']}/{case['case_id']}", text=context)
        ingestion_seconds = time.perf_counter() - ingestion_started
        pra.config.routing_mode = "hybrid_iterative"
        routing_started = time.perf_counter()
        routing = pra.route(str(case["query"]))
        routing_seconds = time.perf_counter() - routing_started
        ranked = _ranked(routing)
        top4 = ranked[:4]
        top8 = ranked[:8]
        row = {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "query": case["query"],
            "ground_truth": truth,
            "source_tokens": int(handle.tokens),
            "backing_contains_evidence": int(truth_span is not None),
            "pra_recall_at_4": int(_covers(top4, truth_span)),
            "pra_recall_at_8": int(_covers(top8, truth_span)),
            "pra_active_tokens_at_4": _selected_tokens(top4),
            "pra_active_tokens_at_8": _selected_tokens(top8),
            "pra_ingestion_seconds": ingestion_seconds,
            "pra_routing_seconds": routing_seconds,
            "pra_requested_kv_tokens": int(routing.stats["requested_kv_tokens"]),
            "status": "supported",
        }
        completed[key] = row
        _write_csv(checkpoint, list(completed.values()))
        print(
            f"[{index}/{len(cases)}] {case['dataset']}/{case['case_id']} "
            f"R4={row['pra_recall_at_4']} R8={row['pra_recall_at_8']}",
            flush=True,
        )

    official_by_condition = {
        condition: {f"{row['dataset']}::{row['case_id']}": row for row in rows}
        for condition, rows in official.items()
    }
    output_rows: list[dict[str, Any]] = []
    for case in cases:
        key = f"{case['dataset']}::{case['case_id']}"
        pra_row = completed[key]
        output_rows.append({
            "condition": "PRA_FROZEN",
            "dataset": case["dataset"],
            "case_id": case["case_id"],
            "task_success": int(pra_row["pra_recall_at_4"]),
            "evidence_eligible": int(pra_row["backing_contains_evidence"]),
            "evidence_recall_at_4": int(pra_row["pra_recall_at_4"]),
            "evidence_recall_at_8": int(pra_row["pra_recall_at_8"]),
            "initial_visible_tokens": 0,
            "active_tokens": int(pra_row["pra_active_tokens_at_4"]),
            "backing_tokens": int(pra_row["source_tokens"]),
            "ingestion_seconds": float(pra_row["pra_ingestion_seconds"]),
            "retrieval_seconds": float(pra_row["pra_routing_seconds"]),
            "index_bytes": "",
            "status": pra_row["status"],
            "notes": "native hybrid, top-four chunks",
        })
        output_rows.append({
            "condition": "FULL_BACKING",
            "dataset": case["dataset"],
            "case_id": case["case_id"],
            "task_success": int(pra_row["backing_contains_evidence"]),
            "evidence_eligible": int(pra_row["backing_contains_evidence"]),
            "evidence_recall_at_4": int(pra_row["backing_contains_evidence"]),
            "evidence_recall_at_8": int(pra_row["backing_contains_evidence"]),
            "initial_visible_tokens": int(pra_row["source_tokens"]),
            "active_tokens": int(pra_row["source_tokens"]),
            "backing_tokens": int(pra_row["source_tokens"]),
            "ingestion_seconds": 0.0,
            "retrieval_seconds": 0.0,
            "index_bytes": len(str(case["context"]).encode("utf-8")),
            "status": "supported",
            "notes": "exact context ceiling",
        })
        for condition in ("HEADROOM_OFFICIAL_DEFAULT", "HEADROOM_OFFICIAL_TUNED"):
            row = official_by_condition[condition][key]
            output_rows.append({
                "condition": condition,
                "dataset": case["dataset"],
                "case_id": case["case_id"],
                "task_success": int(row["evidence_visible_initially"]),
                "evidence_eligible": int(row["evidence_eligible"]),
                "evidence_recall_at_4": int(row["evidence_visible_initially"]),
                "evidence_recall_at_8": int(row["evidence_visible_after_retrieve"]),
                "initial_visible_tokens": int(row["compressed_tokens"]),
                "active_tokens": int(row["compressed_tokens"]),
                "backing_tokens": int(row["original_tokens"]),
                "ingestion_seconds": float(row["compression_seconds"]),
                "retrieval_seconds": float(row["retrieval_seconds"]),
                "index_bytes": int(row["compressed_bytes"]),
                "status": row["status"],
                "notes": f"{row['profile']}; oracle_retrieve={row['evidence_visible_after_retrieve']}",
            })
    recovered_datasets = {
        failure["dataset"] for failure in loader_failures if failure.get("recovered") == "true"
    }
    for failure in loader_failures:
        if failure["dataset"] in recovered_datasets:
            continue
        for condition in (
            "PRA_FROZEN",
            "HEADROOM_OFFICIAL_DEFAULT",
            "HEADROOM_OFFICIAL_TUNED",
            "FULL_BACKING",
            "COMPACT_ONLY",
        ):
            output_rows.append({
                "condition": condition,
                "dataset": failure["dataset"],
                "case_id": "UNSUPPORTED",
                "task_success": "",
                "evidence_eligible": "",
                "evidence_recall_at_4": "",
                "evidence_recall_at_8": "",
                "initial_visible_tokens": "",
                "active_tokens": "",
                "backing_tokens": "",
                "ingestion_seconds": "",
                "retrieval_seconds": "",
                "index_bytes": "",
                "status": "unsupported_loader",
                "notes": failure["error"],
            })
    for dataset in sorted({str(case["dataset"]) for case in cases}):
        output_rows.append({
            "condition": "COMPACT_ONLY",
            "dataset": dataset,
            "case_id": "NOT_APPLICABLE",
            "task_success": "",
            "evidence_eligible": "",
            "evidence_recall_at_4": "",
            "evidence_recall_at_8": "",
            "initial_visible_tokens": "",
            "active_tokens": "",
            "backing_tokens": "",
            "ingestion_seconds": "",
            "retrieval_seconds": "",
            "index_bytes": "",
            "status": "not_applicable",
            "notes": "official external cases do not define the Paper 7 typed compact-only baseline",
        })
    _write_csv(OUTPUT / "pra_on_headroom_results.csv", output_rows)
    print(f"wrote {len(output_rows)} cross-dataset rows")


if __name__ == "__main__":
    main()
