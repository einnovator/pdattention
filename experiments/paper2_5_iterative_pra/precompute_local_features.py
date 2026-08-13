"""Capture 256-token contextual parents and aligned 32-token local gists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_smoke import evidence_token_spans
from experiments.paper2_hf.routing.precompute_router_features import lexical_chunk_scores
from experiments.paper2_hf.routing.run_query_strategies import (
    REGISTRY,
    _capture_query_features,
    load_split_examples,
)
from experiments.paper2_hf.routing.run_representation import _overlaps
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, aggregate_query_states, inject_pra


@torch.no_grad()
def _feature(handle, tokenizer, example: dict, device: torch.device) -> dict:
    source_ids = tokenizer(
        example["source"], return_tensors="pt", add_special_tokens=False
    ).input_ids
    source_tokens = int(source_ids.shape[1])
    evidence_spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
    entry = handle.add_reference(
        f"benchmark://{example['dataset']}/{example['id']}",
        source_ids,
        text=example["source"],
    )
    hidden, _, prompt_tokens, question_span = _capture_query_features(
        handle, tokenizer, example, device
    )
    adapter = next(iter(handle.adapters.values()))
    chunks = entry.layer_memory[adapter.layer_idx].chunks
    parent_hidden, parent_spans, local_hidden, local_spans, local_parents = [], [], [], [], []
    for parent_index, chunk in enumerate(chunks):
        gists = chunk.routing_gist.k.float().cpu()
        spans = chunk.routing_gist.metadata["segment_token_spans"]
        occupancy = torch.tensor(
            [end - start for start, end in spans], dtype=torch.float32
        )
        parent_hidden.append((gists * occupancy[:, None]).sum(0) / occupancy.sum())
        parent_spans.append((int(chunk.logical_start), int(chunk.logical_end)))
        for gist, (start, end) in zip(gists, spans):
            local_hidden.append(gist)
            local_spans.append(
                (int(chunk.logical_start) + int(start), int(chunk.logical_start) + int(end))
            )
            local_parents.append(parent_index)
    parent_positive = torch.tensor(
        [_overlaps(span, evidence_spans) for span in parent_spans], dtype=torch.bool
    )
    local_positive = torch.tensor(
        [_overlaps(span, evidence_spans) for span in local_spans], dtype=torch.bool
    )
    if not bool(parent_positive.any()):
        raise RuntimeError(f"No evidence parent for {example['dataset']}:{example['id']}")
    query = aggregate_query_states(hidden, REGISTRY["last"].strategy)[0].float().cpu()
    handle.cache.clear()
    return {
        "dataset": example["dataset"],
        "example_id": example["id"],
        "query_hidden": query,
        "parent_hidden": torch.stack(parent_hidden),
        "parent_spans": parent_spans,
        "parent_positive_mask": parent_positive,
        "local_hidden": torch.stack(local_hidden),
        "local_spans": local_spans,
        "local_parent_indices": torch.tensor(local_parents, dtype=torch.long),
        "local_positive_mask": local_positive,
        "evidence_spans": evidence_spans,
        "source_tokens": source_tokens,
        "prompt_tokens": prompt_tokens,
        "question_tokens": question_span[1] - question_span[0],
        "parent_lexical_scores": lexical_chunk_scores(
            tokenizer, example["source"], example["question"], parent_spans
        ),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
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
            model_max_context_tokens=256,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=256,
            routing_chunk_tokens=256,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="segment_mean",
            gists_per_chunk=8,
            max_materialized_memory_tokens=256,
            top_k_references=1,
            top_k_chunks_per_reference=8,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            collect_detailed_timing=False,
            collect_routing_metrics=False,
        ),
    )
    examples = load_split_examples(args.cache_dir, args.examples, 8, args.seed)
    features = []
    for index, example in enumerate(examples, start=1):
        features.append(_feature(handle, tokenizer, example, device))
        print(f"[local {index}/{len(examples)}] {example['dataset']} {example['id']}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(features, args.output_dir / "local_router_features_test.pt")
    manifest = {
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "routing_layer": next(iter(handle.adapters)),
        "examples": len(features),
        "dataset_counts": {
            dataset: sum(row["dataset"] == dataset for row in features)
            for dataset in ("hotpotqa", "qasper")
        },
        "contextual_encoding_tokens": 256,
        "associative_local_tokens": 32,
        "local_gists_per_parent": 8,
        "materialization_parent_tokens": 256,
        "local_windows_reencoded": False,
        "native_limit_violations": handle.native_limit_violations,
    }
    (args.output_dir / "local_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--routing-layer", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/local_associative_closure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
