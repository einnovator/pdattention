"""Capture exact token hidden states needed by the Hotpot chunk-size ladder."""

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
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


ENCODING_BLOCK_TOKENS = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _capture_token_hidden(handle, source_ids: torch.Tensor) -> torch.Tensor:
    """Capture contextual token states under the frozen canonical block encoder."""
    adapter = next(iter(handle.adapters.values()))
    blocks = []
    total = int(source_ids.shape[1])
    for start in range(0, total, ENCODING_BLOCK_TOKENS):
        end = min(start + ENCODING_BLOCK_TOKENS, total)
        ids = source_ids[:, start:end].to(handle.device)
        positions = torch.arange(start, end, device=handle.device).unsqueeze(0)
        adapter.begin_capture(positions)
        handle.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions,
            use_cache=False,
        )
        capture = adapter.consume_capture()
        blocks.append(capture.hidden_states[0].to("cpu", torch.float16))
    return torch.cat(blocks, dim=0)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
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
            model_max_context_tokens=ENCODING_BLOCK_TOKENS,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=ENCODING_BLOCK_TOKENS,
            routing_chunk_tokens=ENCODING_BLOCK_TOKENS,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="segment_mean",
            gists_per_chunk=8,
            max_materialized_memory_tokens=ENCODING_BLOCK_TOKENS,
            top_k_references=1,
            top_k_chunks_per_reference=8,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            collect_detailed_timing=False,
            collect_routing_metrics=False,
        ),
    )
    frozen = torch.load(args.source_feature_file, map_location="cpu", weights_only=False)
    frozen = [row for row in frozen if row["dataset"] == "hotpotqa"]
    if len(frozen) != args.examples:
        raise ValueError(f"Expected {args.examples} frozen Hotpot rows, found {len(frozen)}.")
    examples = {
        row["id"]: row
        for row in load_split_examples(args.cache_dir, args.examples, 8, args.seed)
        if row["dataset"] == "hotpotqa"
    }
    rows = []
    started = time.perf_counter()
    for index, feature in enumerate(frozen, start=1):
        example_id = feature["example_id"]
        example = examples.get(example_id)
        if example is None:
            raise ValueError(f"Frozen example {example_id} is absent from the dataset split.")
        source_ids = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids
        token_count = int(source_ids.shape[1])
        if token_count != int(feature["source_tokens"]):
            raise ValueError(
                f"Token-count mismatch for {example_id}: {token_count} != "
                f"{feature['source_tokens']}"
            )
        token_started = time.perf_counter()
        hidden = _capture_token_hidden(handle, source_ids)
        rows.append(
            {
                "dataset": "hotpotqa",
                "example_id": example_id,
                "source_tokens": token_count,
                "token_hidden": hidden,
                "capture_seconds": time.perf_counter() - token_started,
            }
        )
        print(
            f"[chunk-hidden {index}/{len(frozen)}] {example_id} "
            f"tokens={token_count}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "chunk_granularity_token_hidden.pt"
    torch.save(rows, output)
    adapter = next(iter(handle.adapters.values()))
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "routing_layer": adapter.layer_idx,
        "backbone_frozen": True,
        "encoding_block_tokens": ENCODING_BLOCK_TOKENS,
        "position_policy": "absolute_logical_positions_within_256_token_encoding_blocks",
        "examples": len(rows),
        "tokens": sum(row["source_tokens"] for row in rows),
        "capture_seconds": time.perf_counter() - started,
        "artifact": {
            "path": output.name,
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "tracked": False,
            "regenerate_with": (
                "python experiments/paper2_5_iterative_pra/"
                "precompute_chunk_granularity_features.py --device cuda"
            ),
        },
    }
    (args.output_dir / "chunk_granularity_feature_manifest.json").write_text(
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
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=result_root / "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=result_root / "chunk_granularity"
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
