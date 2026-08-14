"""Capture full-query contextual states for the Paper-2.5 entry study.

The existing 621 MB source-side feature cache remains frozen. This script adds
only query hidden states and native pre-RoPE query heads from one complete prompt
forward pass per example. Local query windows are derived offline afterward.
"""

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
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import (
    _prompt_with_question_span,
    load_split_examples,
)
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def capture_query_context(handle, tokenizer, example: dict, device: torch.device) -> dict:
    """Run one full prompt and retain contextual states without window replay."""
    encoded, question_span = _prompt_with_question_span(
        tokenizer, example["question"], 128
    )
    encoded = encoded.to(device)
    positions = torch.arange(encoded.input_ids.shape[1], device=device).unsqueeze(0)
    adapter = next(iter(handle.adapters.values()))
    handle.set_memory_enabled(False)
    adapter.begin_capture(positions)
    handle.model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        position_ids=positions,
        use_cache=False,
    )
    captured = adapter.consume_capture()
    hidden = captured.hidden_states[0].to("cpu", torch.float16)
    pre_query = captured.pre_query[0].permute(1, 0, 2).to("cpu", torch.float16)
    input_ids = encoded.input_ids[0].cpu()
    start, end = question_span
    return {
        "dataset": example["dataset"],
        "example_id": example["id"],
        "question": example["question"],
        "query_hidden_states": hidden,
        "query_pre_query": pre_query,
        "question_span": (int(start), int(end)),
        "prompt_input_ids": input_ids,
        "question_input_ids": input_ids[start:end].clone(),
        "prompt_tokens": int(input_ids.numel()),
        "question_tokens": int(end - start),
        "full_query_forward_count": 1,
        "independent_window_forward_count": 0,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    source_features = torch.load(
        args.source_feature_file, map_location="cpu", weights_only=False
    )
    source_by_id = {row["example_id"]: row for row in source_features}
    examples = load_split_examples(
        args.cache_dir, args.examples, args.example_offset, args.seed
    )
    expected_ids = [row["example_id"] for row in source_features]
    actual_ids = [row["id"] for row in examples]
    if actual_ids != expected_ids:
        raise ValueError("Query examples do not match the frozen source feature order.")

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
            collect_detailed_timing=False,
            collect_routing_metrics=False,
        ),
    )

    features = []
    maximum_baseline_error = 0.0
    for index, example in enumerate(examples, start=1):
        row = capture_query_context(handle, tokenizer, example, device)
        baseline = source_by_id[row["example_id"]]["query_hidden"].float()
        error = float((row["query_hidden_states"][-1].float() - baseline).abs().max())
        row["baseline_query_max_abs_error"] = error
        maximum_baseline_error = max(maximum_baseline_error, error)
        features.append(row)
        handle.cache.clear()
        print(
            f"[query-entry {index}/{len(examples)}] "
            f"{row['dataset']} {row['example_id']} baseline_error={error:.3g}",
            flush=True,
        )
    if maximum_baseline_error != 0.0:
        raise AssertionError(
            "New full-query capture did not reproduce the frozen final-token root."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "query_entry_features.pt"
    torch.save(features, feature_path)
    adapter = next(iter(handle.adapters.values()))
    config = model.config
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "routing_layer": adapter.layer_idx,
        "hidden_width": int(config.hidden_size),
        "query_heads": int(config.num_attention_heads),
        "kv_heads": int(config.num_key_value_heads),
        "head_dim": int(adapter.original_attention.head_dim),
        "examples": len(features),
        "dataset_counts": {
            dataset: sum(row["dataset"] == dataset for row in features)
            for dataset in ("hotpotqa", "qasper")
        },
        "query_encoding": "one_full_contextual_prompt_pass",
        "full_query_forward_count_per_example": 1,
        "independent_window_forward_count": 0,
        "global_query_definition": "final_prompt_token_attention_input_hidden_state",
        "native_head_space": "layer_27_pre_rope_query_heads",
        "maximum_baseline_query_error": maximum_baseline_error,
        "frozen_source_feature": str(
            args.source_feature_file.relative_to(ROOT)
        ).replace("\\", "/"),
        "feature_artifact": {
            "path": feature_path.name,
            "bytes": feature_path.stat().st_size,
            "sha256": _sha256(feature_path),
            "tracked": False,
            "regenerate_with": (
                "python experiments/paper2_5_iterative_pra/"
                "precompute_query_entry_features.py --device cuda"
            ),
        },
    }
    (args.output_dir / "query_entry_feature_manifest.json").write_text(
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
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/"
        "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/query_entry_facets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
