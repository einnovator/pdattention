"""Capture contextual layer-27 hidden states and tokenwise pre-RoPE Q/K."""

from __future__ import annotations

import argparse
import hashlib
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
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.precompute_router_features import lexical_chunk_scores
from experiments.paper2_hf.routing.run_query_strategies import (
    REGISTRY,
    _capture_query_features,
    load_split_examples,
)
from experiments.paper2_hf.routing.run_representation import _overlaps
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    PRAHFConfig,
    aggregate_query_states,
    inject_pra,
)


PARENT_TOKENS = 256
LOCAL_TOKENS = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _capture_source(handle, source_ids: torch.Tensor) -> dict:
    """Encode bounded parents once and retain aligned local routing tensors."""
    adapter = next(iter(handle.adapters.values()))
    device = handle.device
    parent_hidden, parent_spans = [], []
    local_hidden, local_spans, local_parents = [], [], []
    local_pre_query, local_pre_key, local_masks = [], [], []
    total = int(source_ids.shape[1])
    for block_start in range(0, total, PARENT_TOKENS):
        block_end = min(block_start + PARENT_TOKENS, total)
        block_ids = source_ids[:, block_start:block_end].to(device)
        positions = torch.arange(block_start, block_end, device=device).unsqueeze(0)
        adapter.begin_capture(positions)
        handle.model(
            input_ids=block_ids,
            attention_mask=torch.ones_like(block_ids),
            position_ids=positions,
            use_cache=False,
        )
        capture = adapter.consume_capture()
        hidden = capture.hidden_states[0].float()
        pre_query = capture.pre_query[0].permute(1, 0, 2)
        pre_key = capture.pre_key[0].permute(1, 0, 2)
        parent_index = len(parent_spans)
        parent_hidden.append(hidden.mean(dim=0).cpu())
        parent_spans.append((block_start, block_end))
        for local_start in range(0, block_end - block_start, LOCAL_TOKENS):
            local_end = min(local_start + LOCAL_TOKENS, block_end - block_start)
            length = local_end - local_start
            q = torch.zeros(
                (LOCAL_TOKENS, pre_query.shape[1], pre_query.shape[2]),
                dtype=torch.float16,
            )
            k = torch.zeros(
                (LOCAL_TOKENS, pre_key.shape[1], pre_key.shape[2]),
                dtype=torch.float16,
            )
            mask = torch.zeros(LOCAL_TOKENS, dtype=torch.bool)
            q[:length] = pre_query[local_start:local_end].to("cpu", torch.float16)
            k[:length] = pre_key[local_start:local_end].to("cpu", torch.float16)
            mask[:length] = True
            local_pre_query.append(q)
            local_pre_key.append(k)
            local_masks.append(mask)
            local_hidden.append(hidden[local_start:local_end].mean(dim=0).cpu())
            local_spans.append((block_start + local_start, block_start + local_end))
            local_parents.append(parent_index)
    return {
        "parent_hidden": torch.stack(parent_hidden),
        "parent_spans": parent_spans,
        "local_hidden": torch.stack(local_hidden),
        "local_spans": local_spans,
        "local_parent_indices": torch.tensor(local_parents, dtype=torch.long),
        "local_pre_query": torch.stack(local_pre_query),
        "local_pre_key": torch.stack(local_pre_key),
        "local_token_mask": torch.stack(local_masks),
    }


@torch.no_grad()
def _feature(handle, tokenizer, example: dict, device: torch.device) -> dict:
    source_ids = tokenizer(
        example["source"], return_tensors="pt", add_special_tokens=False
    ).input_ids
    captured = _capture_source(handle, source_ids)
    hidden, _, prompt_tokens, question_span = _capture_query_features(
        handle, tokenizer, example, device
    )
    evidence_spans = evidence_token_spans(
        tokenizer, example["source"], example["evidence"]
    )
    parent_positive = torch.tensor(
        [_overlaps(span, evidence_spans) for span in captured["parent_spans"]],
        dtype=torch.bool,
    )
    local_positive = torch.tensor(
        [_overlaps(span, evidence_spans) for span in captured["local_spans"]],
        dtype=torch.bool,
    )
    if not bool(parent_positive.any()):
        raise RuntimeError(f"No evidence parent for {example['dataset']}:{example['id']}")
    return {
        "dataset": example["dataset"],
        "example_id": example["id"],
        "query_hidden": aggregate_query_states(
            hidden, REGISTRY["last"].strategy
        )[0].float().cpu(),
        **captured,
        "parent_positive_mask": parent_positive,
        "local_positive_mask": local_positive,
        "evidence_spans": evidence_spans,
        "source_tokens": int(source_ids.shape[1]),
        "prompt_tokens": prompt_tokens,
        "question_tokens": question_span[1] - question_span[0],
        "parent_lexical_scores": lexical_chunk_scores(
            tokenizer,
            example["source"],
            example["question"],
            captured["parent_spans"],
        ),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision
    )
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
    offset = args.offset
    if offset is None:
        offset = 0 if args.split == "validation" else 8
    examples = load_split_examples(args.cache_dir, args.examples, offset, args.seed)
    features = []
    for index, example in enumerate(examples, start=1):
        features.append(_feature(handle, tokenizer, example, device))
        print(
            f"[native-qk {index}/{len(examples)}] "
            f"{example['dataset']} {example['id']}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / f"native_qk_features_{args.split}.pt"
    torch.save(features, feature_path)
    adapter = next(iter(handle.adapters.values()))
    config = model.config
    manifest = {
        "schema_version": "1.0",
        "split": args.split,
        "offset_per_dataset": offset,
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "routing_layer": adapter.layer_idx,
        "hidden_width": int(config.hidden_size),
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(adapter.original_attention.head_dim),
        "representation": "layer_27_pre_rope_native_qk",
        "contextual_encoding_tokens": PARENT_TOKENS,
        "associative_local_tokens": LOCAL_TOKENS,
        "local_windows_reencoded": False,
        "post_rope_payload_captured": False,
        "memory_use_adapter": None,
        "examples": len(features),
        "dataset_counts": {
            dataset: sum(row["dataset"] == dataset for row in features)
            for dataset in ("hotpotqa", "qasper")
        },
        "feature_artifact": {
            "path": feature_path.name,
            "bytes": feature_path.stat().st_size,
            "sha256": _sha256(feature_path),
            "tracked": False,
            "regenerate_with": (
                "python experiments/paper2_5_iterative_pra/"
                "precompute_native_qk_features.py --device cuda "
                f"--split {args.split} --offset {offset}"
            ),
        },
    }
    manifest_name = (
        "native_qk_feature_manifest.json"
        if args.split == "test"
        else f"native_qk_feature_manifest_{args.split}.json"
    )
    (args.output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--routing-layer", type=int, default=-1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--offset", type=int)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/native_qk_closure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
