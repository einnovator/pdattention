"""Run Paper 6.5 M7 native-Q/K controls on the frozen M6.5 test set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.agent_workflows import realistic_tool_catalog
from data.semantic_hard_tools import semantic_hardness_queries
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper6_5_tools.run_m6_native_discovery import (
    ROUTING_LAYER,
    NativeFeatureEncoder,
    _channel_scores,
    _load_selectors,
    _pad_keys,
    _structured_definition,
)
from experiments.paper6_5_tools.run_m6_5_external_semantics import _rank_row, _write_csv
from pra_hf.agent_resources import PersistentResourceIndex


NATIVE_MODE_MAP = {
    "native_mean_k": "native_mean_k",
    "native_token_qk": "native_token_qk",
    "paper2_8_rank16_zero_shot": "paper2_8_rank16_ensemble",
    "paper2_8_rank8_centroids_zero_shot": "paper2_8_rank8_centroids",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best_validation_mode(rows, candidates) -> str:
    values = []
    for mode in candidates:
        selected = [row for row in rows if row["split"] == "validation" and row["mode"] == mode]
        top1 = sum(row["top1_correct"] == "True" for row in selected) / max(len(selected), 1)
        mrr = sum(float(row["mrr"]) for row in selected) / max(len(selected), 1)
        values.append((top1, mrr, mode))
    return max(values)[2]


def _external_controls(m6_rows):
    dictionary = _best_validation_mode(m6_rows, ("P2_dictionary", "P3_tags"))
    embedding = _best_validation_mode(m6_rows, ("P5_english_embedding", "P6_multilingual_embedding"))
    return {
        "external_lexical_bm25": "P1_bm25",
        "external_dictionary": dictionary,
        "external_compact_embedding": embedding,
        "external_hybrid_p8": "P8_lexical_dictionary_embedding",
        "external_staged_p10": "P10_staged_external",
        "oracle_identity": "P9_oracle_identity",
    }


def run(args) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    m6_manifest = json.loads((args.m6_5_dir / "semantic_hardness_manifest.json").read_text(encoding="utf-8"))
    m6_rows = _read_csv(args.m6_5_dir / "semantic_hardness_rows.csv")
    test_queries = [row for row in semantic_hardness_queries() if row.split == "test"]
    if args.max_queries is not None:
        test_queries = test_queries[: args.max_queries]
    query_digest = hashlib.sha256(
        "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in semantic_hardness_queries()).encode("utf-8")
    ).hexdigest()
    if query_digest != m6_manifest["query_fingerprint"]:
        raise RuntimeError("M7 query fingerprint does not match frozen M6.5 benchmark.")

    external_modes = _external_controls(m6_rows)
    output_rows = []
    allowed_ids = {row.query_id for row in test_queries}
    for output_mode, source_mode in external_modes.items():
        for row in m6_rows:
            if row["query_id"] not in allowed_ids or row["mode"] != source_mode:
                continue
            copied = dict(row)
            copied["mode"] = output_mode
            copied["source_mode"] = source_mode
            copied["milestone"] = "M7"
            copied["query_encode_seconds"] = 0.0
            copied["score_seconds"] = copied["routing_seconds"]
            output_rows.append(copied)

    device = torch.device(args.device)
    encoder = NativeFeatureEncoder(args.model_id, args.revision, device, args.layer)
    resources = realistic_tool_catalog()
    resource_features = []
    for index, resource in enumerate(resources, start=1):
        resource_features.append(encoder.encode(_structured_definition(resource)))
        print(f"[M7 encode resource {index}/{len(resources)}] {resource.name}", flush=True)
    keys, mask = _pad_keys(resource_features)
    index = PersistentResourceIndex(resources)
    selectors16 = _load_selectors(args.checkpoint_dir, 16, device)
    selectors8 = _load_selectors(args.checkpoint_dir, 8, device)
    native_manifest = json.loads(
        (ROOT / "docs/papers/shared/results/paper6_5_tools/m6_native/manifest.json").read_text(encoding="utf-8")
    )
    by_external_query = {
        row["query_id"]: row
        for row in m6_rows
        if row["split"] == "test" and row["mode"] == "P1_bm25"
    }
    for query_index, query in enumerate(test_queries, start=1):
        full_query = f"Context: {query.context}\nQuery: {query.query}" if query.context else query.query
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        query_feature = encoder.encode(full_query)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        query_encode_seconds = time.perf_counter() - started
        scores_by_mode, timings = _channel_scores(
            full_query,
            query_feature,
            resources,
            resource_features,
            keys,
            mask,
            index,
            selectors16,
            selectors8,
            device,
        )
        external = by_external_query[query.query_id]
        for output_mode, internal_mode in NATIVE_MODE_MAP.items():
            score_seconds = timings[internal_mode]
            row = _rank_row(
                query,
                output_mode,
                scores_by_mode[internal_mode],
                resources,
                latency_seconds=query_encode_seconds + score_seconds,
                index_bytes=int(native_manifest["index_bytes"][internal_mode]),
                model_bytes=sum(parameter.numel() * parameter.element_size() for parameter in encoder.model.parameters()),
                token_overlap_value=float(external["token_overlap"]),
                bm25_score=float(external["bm25_required_score"]),
                bm25_rank=int(external["bm25_required_rank"]),
            )
            row["source_mode"] = internal_mode
            row["milestone"] = "M7"
            row["query_encode_seconds"] = query_encode_seconds
            row["score_seconds"] = score_seconds
            output_rows.append(row)
        print(f"[M7 query {query_index}/{len(test_queries)}] {query.query_id}", flush=True)

    _write_csv(args.output_dir / "m7_semantic_hard_rows.csv", output_rows)
    manifest = {
        "schema_version": "1.0",
        "milestone": "M7",
        "model_id": args.model_id,
        "model_revision": args.revision,
        "model_frozen": True,
        "routing_layer": args.layer,
        "query_fingerprint": query_digest,
        "m6_5_query_fingerprint": m6_manifest["query_fingerprint"],
        "test_query_count": len(test_queries),
        "external_controls_frozen_from_validation": external_modes,
        "native_modes": list(NATIVE_MODE_MAP),
        "tool_specific_training_used": False,
        "paper2_8_tool_supervision_used": False,
        "raw_native_state_persisted": False,
        "rows": len(output_rows),
        "runtime": runtime_metadata(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=int, default=ROUTING_LAYER)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_8_qk_compression/low_rank_frontier/checkpoints",
    )
    parser.add_argument(
        "--m6-5-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/m6_5_semantic_hard",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper6_5_tools/m7_semantic_hard_native",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
