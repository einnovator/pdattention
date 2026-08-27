"""Measure compact versus full-backing production PRA addressability."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.full_pra_context_cases import FullPRAContextCase, full_pra_context_cases
from pra_hf import PRAConfig, PRAForCausalLM
from pra_hf.adaptive_context_runtime import (
    AdaptiveContextRuntime,
    ContextPolicy,
    CursorPolicy,
    DeploymentTopology,
    StoragePolicy,
    TypeContextPolicy,
)
from pra_hf.context_records import RecordType
from pra_hf.context_store import RecordScope
from pra_hf.progressive_context import ProgressiveContextRuntime
from pra_hf.typed_context import CompressorRegistry


DEFAULT_OUTPUT = (
    ROOT / "docs/papers/shared/results/paper7_records/full_pra_calibrated"
)
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
K_VALUES = (1, 2, 4, 8)
PROTOCOL = "paper7-full-pra-reachability-v1"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=True, default=str
    )


def _runtime(case: FullPRAContextCase, output: Path) -> tuple[ProgressiveContextRuntime, str]:
    policies = {record_type: TypeContextPolicy(unit_limit=3) for record_type in RecordType}
    runtime = AdaptiveContextRuntime(
        RecordScope("paper7-full-pra", case.case_id),
        ContextPolicy(
            topology=DeploymentTopology.SAME_PROCESS,
            storage=StoragePolicy.ON_DEMAND,
            local_store=output / ".stores" / case.case_id,
            record_policies=policies,
            cursor_policy=CursorPolicy(page_size=4, max_page_size=16),
            persistent_store=False,
        ),
    )
    progressive = ProgressiveContextRuntime(runtime, chunk_tokens=32)
    record = progressive.ingest(
        case.payload,
        record_type=case.record_type,
        capabilities=case.capabilities,
        provenance={"source": "paper7_full_pra_fixture", "partition": case.partition},
    )
    return progressive, record.record_id


def _subsequence_span(values: Sequence[int], target: Sequence[int]) -> tuple[int, int] | None:
    if not target:
        return None
    for start in range(len(values) - len(target) + 1):
        if list(values[start : start + len(target)]) == list(target):
            return start, start + len(target)
    return None


def _evidence_span(tokenizer, text: str, marker: str) -> tuple[int, int] | None:
    values = tokenizer(text, add_special_tokens=False).input_ids
    target = tokenizer(marker, add_special_tokens=False).input_ids
    return _subsequence_span(values, target)


def _ranked_chunks(routing) -> list[dict[str, object]]:
    rows = []
    for reference in routing.rankings:
        for chunk in reference["chunks"]:
            rows.append({
                "reference_uri": reference["reference_uri"],
                "reference_score": float(reference["reference_score"]),
                **dict(chunk),
            })
    rows.sort(key=lambda row: (-float(row["chunk_score"]), str(row["chunk_id"])))
    return rows


def _covers_span(
    rows: Sequence[Mapping[str, object]], span: tuple[int, int] | None
) -> bool:
    if span is None:
        return False
    covered = set()
    for row in rows:
        start = int(row.get("logical_start", row.get("token_start", 0)))
        stop = int(row.get("logical_end", row.get("token_end", 0)))
        covered.update(range(max(start, span[0]), min(stop, span[1])))
    return all(position in covered for position in range(*span))


def _decode_chunks(tokenizer, source: str, rows: Sequence[Mapping[str, object]]) -> list[dict]:
    token_ids = tokenizer(source, add_special_tokens=False).input_ids
    result = []
    for row in rows:
        start = int(row.get("logical_start", row.get("token_start", 0)))
        stop = int(row.get("logical_end", row.get("token_end", 0)))
        result.append({
            "chunk_id": str(row["chunk_id"]),
            "logical_start": start,
            "logical_end": stop,
            "text": tokenizer.decode(token_ids[start:stop], skip_special_tokens=True),
            "chunk_score": float(row["chunk_score"]),
        })
    return result


def _marker_visible(chunks: Sequence[Mapping[str, object]], marker: str) -> bool:
    """Require the exact evidence marker in the materialized chunk payload."""

    visible = "\n".join(str(chunk.get("text", "")) for chunk in chunks)
    return marker.casefold() in visible.casefold()


def _lexical_chunks(tokenizer, source: str, query: str, chunk_tokens: int = 32) -> list[dict]:
    source_ids = tokenizer(source, add_special_tokens=False).input_ids
    query_ids = set(tokenizer(query, add_special_tokens=False).input_ids)
    rows = []
    for ordinal, start in enumerate(range(0, len(source_ids), chunk_tokens)):
        stop = min(start + chunk_tokens, len(source_ids))
        score = len(query_ids & set(source_ids[start:stop]))
        rows.append({
            "chunk_id": f"lexical-{ordinal}",
            "chunk_score": float(score),
            "logical_start": start,
            "logical_end": stop,
        })
    rows.sort(key=lambda row: (-float(row["chunk_score"]), str(row["chunk_id"])))
    return rows


def _route_variant(pra: PRAForCausalLM, query: str, mode: str):
    pra.config.routing_mode = mode
    started = time.perf_counter()
    routing = pra.route(query)
    return routing, time.perf_counter() - started


def _consume_variant(pra: PRAForCausalLM, query: str, mode: str) -> dict[str, object]:
    pra.config.routing_mode = mode
    result = pra.generate(
        query, max_new_tokens=1, do_sample=False, return_details=True,
    )
    return {
        "requested_kv_tokens": int(result.stats["requested_kv_tokens"]),
        "materialized_kv_tokens": int(result.stats["materialized_kv_tokens"]),
        "active_kv_fraction": float(result.stats["materialized_kv_token_fraction"]),
        "generation_seconds": float(result.stats["generation_seconds"]),
    }


def _run_case(
    pra: PRAForCausalLM,
    case: FullPRAContextCase,
    output: Path,
    *,
    verify_consumption: bool = False,
) -> dict[str, object]:
    progressive, record_id = _runtime(case, output)
    record = progressive.runtime.records[record_id]
    compact = _json_text(record.compact_view())
    backing = _json_text(case.payload)
    compact_span = _evidence_span(pra.tokenizer, compact, case.evidence_marker)
    backing_span = _evidence_span(pra.tokenizer, backing, case.evidence_marker)
    result: dict[str, object] = {
        "protocol": PROTOCOL,
        "case_id": case.case_id,
        "partition": case.partition,
        "case_class": case.case_class,
        "omission_stratum": case.omission_stratum.value,
        "record_type": case.record_type.value,
        "query": case.query,
        "evidence_marker": case.evidence_marker,
        "compact_trigger_literal": int(compact_span is not None),
        "compact_semantic_cue": int(
            bool(case.semantic_cue)
            and str(case.semantic_cue).casefold() in compact.casefold()
        ),
        "backing_trigger_literal": int(backing_span is not None),
        "compact_bytes": len(compact.encode("utf-8")),
        "full_backing_bytes": len(backing.encode("utf-8")),
    }

    pra.clear_references()
    compact_handle = pra.add_reference(
        f"record://{case.case_id}/views/summary", text=compact
    )
    compact_routing, compact_wall = _route_variant(pra, case.query, "one_shot")
    compact_ranked = _ranked_chunks(compact_routing)
    result["PRA_COMPACT"] = {
        "ranked": compact_ranked,
        "recall": {
            str(k): int(_covers_span(compact_ranked[:k], compact_span))
            for k in K_VALUES
        },
        "source_tokens": compact_handle.tokens,
        "routing_seconds": compact_routing.routing_seconds,
        "query_seconds": compact_routing.query_encoding_seconds,
        "wall_seconds": compact_wall,
    }

    lexical = _lexical_chunks(pra.tokenizer, backing, case.query)
    result["PRA_FALLBACK"] = {
        "ranked": lexical,
        "recall": {
            str(k): int(_covers_span(lexical[:k], backing_span))
            for k in K_VALUES
        },
        "selected_chunks": _decode_chunks(pra.tokenizer, backing, lexical[:8]),
    }
    result["PRA_FALLBACK"]["evidence_visible_at_4"] = int(
        _marker_visible(result["PRA_FALLBACK"]["selected_chunks"][:4], case.evidence_marker)
    )

    pra.clear_references()
    progressive.registry.pra_model = pra
    handle = progressive.register_backing_record(record_id)
    result["index_metrics"] = progressive.registry.backing_index_metrics[record_id]
    for label, mode in (
        ("PRA_NATIVE_SEMANTIC", "one_shot"),
        ("PRA_NATIVE_HYBRID", "hybrid_iterative"),
    ):
        routing, wall = _route_variant(pra, case.query, mode)
        ranked = _ranked_chunks(routing)
        result[label] = {
            "ranked": ranked,
            "recall": {
                str(k): int(_covers_span(ranked[:k], backing_span))
                for k in K_VALUES
            },
            "selected_chunks": _decode_chunks(pra.tokenizer, backing, ranked[:8]),
            "source_tokens": handle.tokens,
            "routing_seconds": routing.routing_seconds,
            "query_seconds": routing.query_encoding_seconds,
            "wall_seconds": wall,
            "requested_kv_tokens": routing.stats["requested_kv_tokens"],
        }
        result[label]["evidence_visible_at_4"] = int(
            _marker_visible(result[label]["selected_chunks"][:4], case.evidence_marker)
        )
        if verify_consumption:
            result[label]["consumption"] = _consume_variant(pra, case.query, mode)
        else:
            requested = int(result[label]["requested_kv_tokens"])
            result[label]["consumption"] = {
                "requested_kv_tokens": requested,
                "materialized_kv_tokens": requested,
                "active_kv_fraction": requested / max(int(handle.tokens), 1),
                "generation_seconds": 0.0,
                "inferred_from_requested": True,
            }
    progressive.runtime.store.close()
    return result


def _load_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_checkpoint(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _summarize(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    cases = {case.case_id: case for case in full_pra_context_cases()}
    compressor = CompressorRegistry()
    compression_seconds = {}
    for case_id in {str(row["case_id"]) for row in rows}:
        case = cases[case_id]
        started = time.perf_counter()
        compressor.compress(case.record_type, case.payload, unit_limit=3)
        compression_seconds[case_id] = time.perf_counter() - started
    policies = (
        "PRA_COMPACT", "PRA_FALLBACK", "PRA_NATIVE_SEMANTIC", "PRA_NATIVE_HYBRID"
    )
    reachability = []
    for row in rows:
        for policy in policies:
            values = row[policy]
            for k in K_VALUES:
                reachability.append({
                    "case_id": row["case_id"],
                    "partition": row["partition"],
                    "case_class": row["case_class"],
                    "omission_stratum": row["omission_stratum"],
                    "policy": policy,
                    "k": k,
                    "backing_recall": values["recall"][str(k)],
                    "routing_seconds": values.get("routing_seconds", 0.0),
                    "query_seconds": values.get("query_seconds", 0.0),
                    "source_tokens": values.get("source_tokens", 0),
                })
    _write_csv(output / "pra_backing_reachability.csv", reachability)

    aggregate = []
    grouped = defaultdict(list)
    for row in reachability:
        grouped[(row["partition"], row["omission_stratum"], row["policy"], row["k"])].append(row)
    for (partition, stratum, policy, k), values in sorted(grouped.items()):
        aggregate.append({
            "partition": partition,
            "omission_stratum": stratum,
            "policy": policy,
            "k": k,
            "n": len(values),
            "recall": statistics.fmean(float(value["backing_recall"]) for value in values),
            "routing_seconds": statistics.fmean(float(value["routing_seconds"]) for value in values),
            "query_seconds": statistics.fmean(float(value["query_seconds"]) for value in values),
        })
    _write_csv(output / "pra_native_routing_variants.csv", aggregate)

    validation = [row for row in aggregate if row["partition"] == "validation" and row["k"] == 4]
    choices = []
    for policy in ("PRA_NATIVE_SEMANTIC", "PRA_NATIVE_HYBRID"):
        values = [row for row in validation if row["policy"] == policy]
        total = sum(int(row["n"]) for row in values)
        choices.append((
            sum(float(row["recall"]) * int(row["n"]) for row in values) / total,
            -sum(float(row["routing_seconds"]) * int(row["n"]) for row in values) / total,
            policy,
        ))
    selected_policy = max(choices)[2]
    addressability = []
    for row in rows:
        addressability.append({
            "case_id": row["case_id"],
            "partition": row["partition"],
            "case_class": row["case_class"],
            "omission_stratum": row["omission_stratum"],
            "compact_trigger_literal": row["compact_trigger_literal"],
            "compact_semantic_cue": row["compact_semantic_cue"],
            "backing_trigger_literal": row["backing_trigger_literal"],
            "pra_compact_recall_at_4": row["PRA_COMPACT"]["recall"]["4"],
            "pra_fallback_recall_at_4": row["PRA_FALLBACK"]["recall"]["4"],
            "pra_native_recall_at_4": row[selected_policy]["recall"]["4"],
            "fallback_evidence_visible_at_4": row["PRA_FALLBACK"]["evidence_visible_at_4"],
            "native_evidence_visible_at_4": row[selected_policy]["evidence_visible_at_4"],
            "fallback_selected_chunks": json.dumps(row["PRA_FALLBACK"]["selected_chunks"][:4]),
            "native_selected_chunks": json.dumps(row[selected_policy]["selected_chunks"][:4]),
            "native_selection_policy": selected_policy,
            "native_requested_kv_tokens": row[selected_policy]["requested_kv_tokens"],
            "native_materialized_kv_tokens": row[selected_policy]["consumption"]["materialized_kv_tokens"],
            "native_consumption_verified": int(
                not row[selected_policy]["consumption"].get("inferred_from_requested", False)
            ),
            "indexing_seconds": row["index_metrics"]["ingestion_seconds"],
            "agent_compression_seconds": compression_seconds[str(row["case_id"])],
            "routing_seconds": row[selected_policy]["routing_seconds"],
            "routing_index_bytes": row["index_metrics"]["routing_index_bytes"],
            "resident_detail_kv_bytes": row["index_metrics"]["resident_detail_kv_bytes"],
            "full_backing_bytes": row["full_backing_bytes"],
            "compact_bytes": row["compact_bytes"],
        })
    _write_csv(output / "compact_vs_backing_addressability.csv", addressability)
    (output / "pra_native_selection.json").write_text(
        json.dumps({
            "selected_policy": selected_policy,
            "selection_rule": "maximum validation Recall@4; routing latency breaks ties",
            "k": 4,
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument(
        "--verify-consumption", action="store_true",
        help="Run a one-token generation pass per native routing variant.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "pra_reachability_checkpoint.jsonl"
    rows = _load_checkpoint(checkpoint)
    if not args.postprocess_only:
        device = torch.device(args.device)
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
        complete = {str(row["case_id"]) for row in rows}
        cases = list(full_pra_context_cases())
        if args.case_limit is not None:
            cases = cases[: args.case_limit]
        for index, case in enumerate(cases, start=1):
            if case.case_id in complete:
                continue
            row = _run_case(
                pra, case, args.output_dir,
                verify_consumption=args.verify_consumption,
            )
            rows.append(row)
            _append_checkpoint(checkpoint, row)
            print(
                f"[{index}/{len(cases)}] {case.case_id} "
                f"semantic R4={row['PRA_NATIVE_SEMANTIC']['recall']['4']} "
                f"hybrid R4={row['PRA_NATIVE_HYBRID']['recall']['4']}",
                flush=True,
            )
    _summarize(rows, args.output_dir)


if __name__ == "__main__":
    main()
