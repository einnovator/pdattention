"""Precompute frozen Qwen query/chunk features for tiny routing adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_smoke import evidence_token_spans
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import (
    REGISTRY,
    _capture_query_features,
    load_split_examples,
)
from experiments.paper2_hf.routing.run_representation import _configure, _overlaps
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    PRAHFConfig,
    aggregate_query_states,
    inject_pra,
)


SPLITS = {
    "validation": (0, 8),
    "test": (8, 16),
    "train": (24, 24),
}


def lexical_chunk_scores(tokenizer, source: str, question: str, spans) -> torch.Tensor:
    """Return token-set Jaccard overlap between the question and each source chunk."""
    source_ids = tokenizer(source, add_special_tokens=False).input_ids
    question_ids = set(tokenizer(question, add_special_tokens=False).input_ids)
    scores = []
    for start, end in spans:
        chunk_ids = set(source_ids[int(start) : int(end)])
        union = question_ids | chunk_ids
        scores.append(len(question_ids & chunk_ids) / len(union) if union else 0.0)
    return torch.tensor(scores, dtype=torch.float32)


def _features_for_example(handle, tokenizer, example: dict, device, query_specs) -> dict:
    _configure(handle, ATTENTION_INPUT_HIDDEN_STATE, 32, "mean", 1, "exact")
    source = tokenizer(
        example["source"], return_tensors="pt", add_special_tokens=False
    ).input_ids
    source_tokens = int(source.shape[1])
    evidence_spans = evidence_token_spans(tokenizer, example["source"], example["evidence"])
    handle.add_reference(
        f"benchmark://{example['dataset']}/{example['id']}",
        source,
        text=example["source"],
    )
    hidden, _, prompt_tokens, question_span = _capture_query_features(
        handle, tokenizer, example, device
    )
    adapter = next(iter(handle.adapters.values()))
    chunks = [
        chunk
        for entry in handle.cache.all_entries()
        for chunk in entry.layer_memory[adapter.layer_idx].chunks
    ]
    gists = torch.cat([chunk.routing_gist.k for chunk in chunks], dim=0).float().cpu()
    spans = [(int(chunk.logical_start), int(chunk.logical_end)) for chunk in chunks]
    positive = torch.tensor(
        [_overlaps(span, evidence_spans) for span in spans],
        dtype=torch.bool,
    )
    if not bool(positive.any()):
        raise RuntimeError(f"No evidence-positive chunk for {example['dataset']}:{example['id']}")
    queries = {}
    for spec in query_specs:
        queries[spec.name] = aggregate_query_states(
            hidden,
            spec.strategy,
            window=spec.window,
            half_life=spec.half_life,
            token_spans=[question_span] if spec.strategy.startswith("question_") else None,
        )[0].float().cpu()
    positions = torch.tensor(
        [((start + end) / 2) / max(source_tokens, 1) for start, end in spans],
        dtype=torch.float32,
    )
    return {
        "dataset": example["dataset"],
        "example_id": example["id"],
        "queries": queries,
        "memory_gists": gists,
        "positive_mask": positive,
        "normalized_positions": positions,
        "lexical_scores": lexical_chunk_scores(
            tokenizer, example["source"], example["question"], spans
        ),
        "chunk_spans": spans,
        "evidence_spans": evidence_spans,
        "source_tokens": source_tokens,
        "prompt_tokens": prompt_tokens,
        "question_tokens": question_span[1] - question_span[0],
    }


def run(args) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(-1,),
            model_max_context_tokens=256,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="mean",
            gists_per_chunk=1,
            max_materialized_memory_tokens=128,
            top_k_references=1,
            top_k_chunks_per_reference=16,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=False,
            collect_routing_metrics=True,
        ),
    )
    query_specs = [REGISTRY[name] for name in args.query_strategies]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {}
    identities = {}
    feature_width = None
    for split, (offset, default_count) in SPLITS.items():
        count = getattr(args, f"{split}_examples") or default_count
        examples = load_split_examples(args.cache_dir, count, offset, args.seed)
        features = []
        for index, example in enumerate(examples, start=1):
            feature = _features_for_example(
                handle, tokenizer, example, device, query_specs
            )
            feature["split"] = split
            features.append(feature)
            feature_width = int(feature["memory_gists"].shape[1])
            print(
                f"[{split} {index}/{len(examples)}] "
                f"{example['dataset']} {example['id']}",
                flush=True,
            )
        path = args.output_dir / f"router_features_{split}.pt"
        torch.save(features, path)
        identities[split] = {
            (feature["dataset"], feature["example_id"]) for feature in features
        }
        split_manifest[split] = {
            "path": path.name,
            "examples": len(features),
            "dataset_counts": {
                dataset: sum(feature["dataset"] == dataset for feature in features)
                for dataset in ("hotpotqa", "qasper")
            },
            "offset": offset,
            "positive_chunks": sum(
                int(feature["positive_mask"].sum().item()) for feature in features
            ),
            "candidate_chunks": sum(len(feature["positive_mask"]) for feature in features),
        }
    leakage = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(SPLITS)
        for right in list(SPLITS)[index + 1 :]
    }
    if any(leakage.values()):
        raise RuntimeError(f"Feature split identity leakage: {leakage}")
    manifest = {
        "runtime": runtime_metadata(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_layer": next(iter(handle.adapters)),
        "feature_source": ATTENTION_INPUT_HIDDEN_STATE,
        "feature_width": feature_width,
        "native_kv_heads": int(model.config.num_key_value_heads),
        "native_head_dim": int(model.config.head_dim),
        "native_kv_dtype_bytes": torch.tensor([], dtype=dtype).element_size(),
        "routing_chunk_tokens": 32,
        "gist_mode": "mean",
        "query_strategies": args.query_strategies,
        "seed": args.seed,
        "splits": split_manifest,
        "identity_leakage": leakage,
        "base_parameters_trainable": sum(parameter.requires_grad for parameter in model.parameters()),
        "base_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "native_limit_violations": handle.native_limit_violations,
    }
    (args.output_dir / "feature_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--validation-examples", type=int)
    parser.add_argument("--test-examples", type=int)
    parser.add_argument("--query-strategies", default="last,question_exp_h2.0")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "docs"
            / "papers"
            / "shared"
            / "results"
            / "paper2_hf"
            / "routing"
            / "learned_adapter"
        ),
    )
    args = parser.parse_args()
    args.query_strategies = tuple(
        value.strip() for value in args.query_strategies.split(",") if value.strip()
    )
    unknown = sorted(set(args.query_strategies) - set(REGISTRY))
    if unknown:
        parser.error(f"Unknown query strategies: {', '.join(unknown)}")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
