"""Capture Q1 states after a frozen model integrates query Q0 with active memory A."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from experiments.paper2_5_iterative_pra.run_oracle_convergence import (
    evidence_parent_groups,
    validation_partition,
)
from pra_hf.dynamic_query import RECONSTRUCTION_MODES, render_reconstructed_query
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _memory_text(tokenizer, source_ids: torch.Tensor, spans, source_group: set[int]):
    starts = [int(spans[index][0]) for index in source_group]
    ends = [int(spans[index][1]) for index in source_group]
    start, end = min(starts), max(ends)
    text = tokenizer.decode(source_ids[start:end], skip_special_tokens=True).strip()
    if not text:
        raise ValueError("Retrieved A decoded to empty text.")
    return text, (start, end)


@torch.no_grad()
def _capture(handle, reconstructed, device: torch.device, baseline_allocated: int) -> dict:
    input_ids = reconstructed.input_ids.to(device)
    attention_mask = reconstructed.attention_mask.to(device)
    positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    adapter = next(iter(handle.adapters.values()))
    handle.set_memory_enabled(False)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    adapter.begin_capture(positions)
    handle.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions,
        use_cache=False,
    )
    captured = adapter.consume_capture()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    return {
        "query_hidden_states": captured.hidden_states[0].to("cpu", torch.float16),
        "query_pre_query": captured.pre_query[0]
        .permute(1, 0, 2)
        .to("cpu", torch.float16),
        "prompt_input_ids": input_ids[0].cpu(),
        "prompt_tokens": int(input_ids.shape[1]),
        "reencoding_time": elapsed,
        "total_peak_gpu_allocated_bytes": peak_allocated,
        "total_peak_gpu_reserved_bytes": peak_reserved,
        "incremental_reencoding_peak_allocated_bytes": max(
            0, peak_allocated - baseline_allocated
        ),
        "h2d_prompt_bytes": (
            input_ids.numel() * input_ids.element_size()
            + attention_mask.numel() * attention_mask.element_size()
        ),
        "full_query_forward_count": 1,
        "independent_window_forward_count": 0,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    source_features = torch.load(
        args.source_feature_file, map_location="cpu", weights_only=False
    )
    examples = load_split_examples(
        args.cache_dir, args.examples, args.example_offset, args.seed
    )
    if [row["example_id"] for row in source_features] != [row["id"] for row in examples]:
        raise ValueError("Dynamic query examples do not match the frozen source cache.")

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
            model_max_context_tokens=args.max_tokens,
            max_prompt_direct_tokens=args.max_tokens,
            encoding_block_tokens=args.max_tokens,
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
    baseline_allocated = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    )
    baseline_reserved = (
        int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0
    )

    rows = []
    transition_count = 0
    for example, feature in zip(examples, source_features):
        if feature["dataset"] != "hotpotqa":
            continue
        source_ids = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        if int(source_ids.numel()) != int(feature["source_tokens"]):
            raise ValueError(f"Source token mismatch for {feature['example_id']}.")
        parent_texts = [
            tokenizer.decode(
                source_ids[int(start) : int(end)], skip_special_tokens=True
            ).strip()
            for start, end in feature["parent_spans"]
        ]
        groups = evidence_parent_groups(feature)
        for transition, (source_group, target_group) in enumerate(zip(groups, groups[1:])):
            transition_count += 1
            memory_text, memory_source_span = _memory_text(
                tokenizer, source_ids, feature["parent_spans"], source_group
            )
            for mode in args.modes:
                reconstructed = render_reconstructed_query(
                    tokenizer,
                    example["question"],
                    memory_text,
                    mode,
                    max_tokens=args.max_tokens,
                )
                captured = _capture(handle, reconstructed, device, baseline_allocated)
                row = {
                    "dataset": "hotpotqa",
                    "partition": validation_partition(feature["example_id"]),
                    "example_id": feature["example_id"],
                    "transition": transition,
                    "question": example["question"],
                    "query_state_id": f"{feature['example_id']}:t{transition + 1}:{mode}",
                    "query_reconstruction_mode": mode,
                    "source_parent_indices": tuple(sorted(source_group)),
                    "target_parent_indices": tuple(sorted(target_group)),
                    "memory_source_token_span": memory_source_span,
                    "memory_text": memory_text,
                    "parent_texts": parent_texts,
                    "question_span": reconstructed.question_span,
                    "memory_span": reconstructed.memory_span,
                    "logical_reference_tokens": int(feature["source_tokens"]),
                    "logical_parent_count": len(feature["parent_spans"]),
                    "conceptual_active_parent_count": len(source_group),
                    "materialized_parent_count": 0,
                    "materialized_native_kv_tokens": 0,
                    "native_kv_bytes": 0,
                    "active_native_kv_fraction": 0.0,
                    "peak_native_kv_tokens": 0,
                    "host_to_device_transfer_events": 2,
                    **captured,
                }
                row["query_feature_cache_bytes"] = (
                    row["query_hidden_states"].numel()
                    * row["query_hidden_states"].element_size()
                    + row["query_pre_query"].numel()
                    * row["query_pre_query"].element_size()
                )
                rows.append(row)
                handle.cache.clear()
            print(
                f"[dynamic-query] {feature['example_id']} transition={transition} "
                f"modes={len(args.modes)} A={sorted(source_group)} B={sorted(target_group)}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "dynamic_query_features.pt"
    torch.save(rows, feature_path)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "routing_layer": next(iter(handle.adapters.values())).layer_idx,
        "examples": len({row["example_id"] for row in rows}),
        "transitions": transition_count,
        "captures": len(rows),
        "reconstruction_modes": list(args.modes),
        "max_tokens": args.max_tokens,
        "model_baseline_gpu_allocated_bytes": baseline_allocated,
        "model_baseline_gpu_reserved_bytes": baseline_reserved,
        "source_feature_file_bytes": args.source_feature_file.stat().st_size,
        "source_feature_tensor_bytes": _tensor_bytes(source_features),
        "native_kv_materialization_performed": False,
        "generation_performed": False,
        "tpot_measured": False,
        "concurrency_measured": False,
        "dollar_cost_measured": False,
        "feature_artifact": {
            "path": feature_path.name,
            "bytes": feature_path.stat().st_size,
            "sha256": _sha256(feature_path),
            "tracked": False,
            "regenerate_with": (
                "python -m experiments.paper2_5_iterative_pra."
                "precompute_dynamic_query_features --device cuda"
            ),
        },
    }
    (args.output_dir / "dynamic_query_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--routing-layer", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--modes", default=",".join(RECONSTRUCTION_MODES))
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=result_root / "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=result_root / "dynamic_query_discovery"
    )
    args = parser.parse_args()
    args.modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not args.modes or any(mode not in RECONSTRUCTION_MODES for mode in args.modes):
        parser.error(f"--modes must be drawn from {RECONSTRUCTION_MODES}")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
