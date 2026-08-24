"""Capture lean Paper 2.8 native-Q/K features for 2Wiki and MuSiQue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_5_iterative_pra.precompute_native_qk_features import (
    LOCAL_TOKENS,
    PARENT_TOKENS,
    _feature,
    _sha256,
)
from experiments.paper2_7_query_graph.helpers import write_csv
from experiments.paper2_8_qk_compression.run_gated_study import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_hf.multihop_routing_data import cohort_manifest, load_multihop_routing_examples
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


DATASET_DIR = {"2wikimultihopqa": "2wiki", "musique": "musique"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _initialize_model(args, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(args.routing_layer,),
            model_max_context_tokens=PARENT_TOKENS,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=PARENT_TOKENS,
            routing_chunk_tokens=PARENT_TOKENS,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="segment_mean",
            gists_per_chunk=8,
            max_materialized_memory_tokens=PARENT_TOKENS,
            top_k_references=1,
            top_k_chunks_per_reference=8,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            collect_detailed_timing=False,
            collect_routing_metrics=False,
        ),
    )
    return tokenizer, model, handle


def _capture_group(
    handle,
    tokenizer,
    examples,
    device,
    *,
    dataset: str,
    split: str,
    shard_dir: Path,
    resume: bool,
):
    shard_dir.mkdir(parents=True, exist_ok=True)
    features = []
    for index, example in enumerate(examples, start=1):
        shard_path = shard_dir / f"{example.example_id}.pt"
        if resume and shard_path.exists():
            feature = torch.load(shard_path, map_location="cpu", weights_only=False)
            if feature["example_id"] != example.example_id:
                raise ValueError(f"Resume shard identity mismatch: {shard_path}")
            action = "resume"
        else:
            feature = _feature(
                handle,
                tokenizer,
                example.as_feature_example(),
                device,
                include_local_pre_query=False,
            )
            feature["split"] = split
            feature["authored_evidence_items"] = len(example.evidence)
            feature["source_segments"] = len(example.source_segments)
            torch.save(feature, shard_path)
            action = "capture"
        features.append(feature)
        print(
            f"[multi-qk {action} {dataset} {split} {index}/{len(examples)}] "
            f"{example.example_id} tokens={feature['source_tokens']} "
            f"chunks={len(feature['local_positive_mask'])}",
            flush=True,
        )
    return features


def _audit_rows(examples, features):
    rows = []
    for example, feature in zip(examples, features):
        positive = torch.nonzero(feature["local_positive_mask"], as_tuple=False).flatten().tolist()
        rows.append(
            {
                "dataset": example.dataset,
                "split": example.split,
                "example_id": example.example_id,
                "authored_evidence_items": len(example.evidence),
                "positive_routing_chunks": len(positive),
                "four_chunk_authored_chain_feasible": len(example.evidence) <= 4,
                "four_chunk_full_span_feasible": len(positive) <= 4,
                "positive_chunk_ids": " ".join(map(str, positive)),
                "source_tokens": feature["source_tokens"],
                "candidate_chunks": len(feature["local_positive_mask"]),
                "source_segments": len(example.source_segments),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    args.output_root = args.output_root.resolve()
    examples = load_multihop_routing_examples(
        args.annotations, args.twowiki_dev, args.musique_dev
    )
    cohort = cohort_manifest(examples)
    requested = set(args.datasets)
    examples = [example for example in examples if example.dataset in requested]
    device = torch.device(args.device)
    tokenizer, model, handle = _initialize_model(args, device)
    adapter = next(iter(handle.adapters.values()))
    all_audits = []
    feature_artifacts = {}
    for dataset in sorted(requested):
        output_dir = args.output_root / DATASET_DIR[dataset]
        output_dir.mkdir(parents=True, exist_ok=True)
        feature_artifacts[dataset] = {}
        for split in args.splits:
            group = [
                example
                for example in examples
                if example.dataset == dataset and example.split == split
            ]
            if args.max_per_group is not None:
                group = group[: args.max_per_group]
            features = _capture_group(
                handle,
                tokenizer,
                group,
                device,
                dataset=dataset,
                split=split,
                shard_dir=args.output_root / ".feature_shards" / dataset / split,
                resume=args.resume,
            )
            path = output_dir / f"native_qk_features_{split}.pt"
            torch.save(features, path)
            all_audits.extend(_audit_rows(group, features))
            feature_artifacts[dataset][split] = {
                "path": str(path.relative_to(ROOT.resolve())),
                "examples": len(features),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "tracked": False,
            }
            del features
    write_csv(args.output_root / "alignment_audit.csv", all_audits)
    source_tokens = [int(row["source_tokens"]) for row in all_audits]
    positive_chunks = [int(row["positive_routing_chunks"]) for row in all_audits]
    config = model.config
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "cohort": cohort,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "routing_layer": adapter.layer_idx,
        "representation": f"layer_{adapter.layer_idx}_pre_rope_native_keys",
        "query_representation": f"layer_{adapter.layer_idx}_frozen_last_span_pre_query",
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(adapter.original_attention.head_dim),
        "contextual_encoding_tokens": PARENT_TOKENS,
        "routing_chunk_tokens": LOCAL_TOKENS,
        "materialization_budget_chunks": 4,
        "materialized_native_kv_unchanged": True,
        "source_tokens": {
            "min": min(source_tokens),
            "median": statistics.median(source_tokens),
            "max": max(source_tokens),
            "mean": statistics.fmean(source_tokens),
        },
        "positive_routing_chunks": {
            "min": min(positive_chunks),
            "median": statistics.median(positive_chunks),
            "max": max(positive_chunks),
            "mean": statistics.fmean(positive_chunks),
        },
        "four_chunk_authored_chain_feasible_fraction": statistics.fmean(
            float(row["four_chunk_authored_chain_feasible"]) for row in all_audits
        ),
        "four_chunk_full_span_feasible_fraction": statistics.fmean(
            float(row["four_chunk_full_span_feasible"]) for row in all_audits
        ),
        "source_artifacts": {
            "annotations": _file_sha256(args.annotations),
            "2wikimultihopqa": _file_sha256(args.twowiki_dev),
            "musique": _file_sha256(args.musique_dev),
        },
        "feature_artifacts": feature_artifacts,
    }
    (args.output_root / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    inherited = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
    default_output = (
        ROOT
        / "docs/papers/shared/results/paper2_8_qk_compression/multi_dataset"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--routing-layer", type=int, default=27)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_DIR),
        default=sorted(DATASET_DIR),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument("--max-per-group", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data/paper2_7_query_facets/annotations.jsonl",
    )
    parser.add_argument(
        "--twowiki-dev", type=Path, default=inherited / "2wiki/dev.json"
    )
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=inherited / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=default_output)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
