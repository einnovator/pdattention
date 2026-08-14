"""Capture frozen Qwen source/query features for MuSiQue and 2Wiki graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_5_iterative_pra.precompute_query_entry_features import (
    capture_query_context,
)
from pra_hf.natural_reasoning_graph import (
    char_spans_to_token_spans,
    load_2wiki,
    load_musique,
    stable_partition,
)
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


ENCODING_BLOCK_TOKENS = 256
LOCAL_TOKENS = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _capture_source(handle, source_ids: torch.Tensor) -> dict:
    """Capture exact token hidden and padded 32-token local native Q/K windows."""
    adapter = next(iter(handle.adapters.values()))
    device = handle.device
    token_hidden = []
    local_pre_query, local_pre_key, local_masks, local_spans = [], [], [], []
    total = int(source_ids.shape[1])
    for block_start in range(0, total, ENCODING_BLOCK_TOKENS):
        block_end = min(block_start + ENCODING_BLOCK_TOKENS, total)
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
        hidden = capture.hidden_states[0]
        pre_query = capture.pre_query[0].permute(1, 0, 2)
        pre_key = capture.pre_key[0].permute(1, 0, 2)
        token_hidden.append(hidden.to("cpu", torch.float16))
        for local_start in range(0, block_end - block_start, LOCAL_TOKENS):
            local_end = min(local_start + LOCAL_TOKENS, block_end - block_start)
            length = local_end - local_start
            query = torch.zeros(
                LOCAL_TOKENS, pre_query.shape[1], pre_query.shape[2], dtype=torch.float16
            )
            key = torch.zeros(
                LOCAL_TOKENS, pre_key.shape[1], pre_key.shape[2], dtype=torch.float16
            )
            mask = torch.zeros(LOCAL_TOKENS, dtype=torch.bool)
            query[:length] = pre_query[local_start:local_end].to("cpu", torch.float16)
            key[:length] = pre_key[local_start:local_end].to("cpu", torch.float16)
            mask[:length] = True
            local_pre_query.append(query)
            local_pre_key.append(key)
            local_masks.append(mask)
            local_spans.append((block_start + local_start, block_start + local_end))
    return {
        "token_hidden": torch.cat(token_hidden),
        "local_pre_query": torch.stack(local_pre_query),
        "local_pre_key": torch.stack(local_pre_key),
        "local_token_mask": torch.stack(local_masks),
        "local_spans": local_spans,
    }


def _load_selected(args: argparse.Namespace):
    selected_ids = {
        json.loads(line)["example_id"]
        for line in args.sample_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = load_musique(args.musique_dev) + load_2wiki(args.twowiki_dev)
    selected = sorted((row for row in rows if row.example_id in selected_ids), key=lambda row: row.example_id)
    if {row.example_id for row in selected} != selected_ids:
        raise ValueError("Selected manifest and local dataset identities differ.")
    return selected


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
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
    examples = _load_selected(args)
    features = []
    started = time.perf_counter()
    for index, example in enumerate(examples, start=1):
        encoded = tokenizer(
            example.source,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        source_ids = encoded.input_ids
        source_started = time.perf_counter()
        source = _capture_source(handle, source_ids)
        query = capture_query_context(
            handle,
            tokenizer,
            {"dataset": example.dataset, "id": example.example_id, "question": example.question},
            device,
        )
        features.append(
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "partition": stable_partition(example.example_id),
                "question": example.question,
                "question_type": example.question_type,
                "annotated_hops": example.annotated_hops,
                "graph_type": example.graph_type,
                "source_tokens": int(source_ids.shape[1]),
                "node_token_spans": char_spans_to_token_spans(offsets, example.nodes),
                "nodes": [asdict(node) for node in example.nodes],
                "annotated_edges": example.annotated_edges,
                "root_node_ids": example.root_node_ids,
                "capture_seconds": time.perf_counter() - source_started,
                **source,
                **{key: value for key, value in query.items() if key not in {"dataset", "example_id", "question"}},
            }
        )
        handle.cache.clear()
        print(
            f"[natural-graph {index}/{len(examples)}] {example.dataset} "
            f"{example.example_id} tokens={source_ids.shape[1]}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "natural_graph_features.pt"
    torch.save(features, artifact)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "routing_layer": next(iter(handle.adapters.values())).layer_idx,
        "backbone_frozen": True,
        "encoding_block_tokens": ENCODING_BLOCK_TOKENS,
        "local_tokens": LOCAL_TOKENS,
        "position_policy": "absolute logical positions within independent 256-token blocks",
        "examples": len(features),
        "dataset_counts": {
            dataset: sum(row["dataset"] == dataset for row in features)
            for dataset in ("musique", "2wikimultihopqa")
        },
        "source_tokens": sum(row["source_tokens"] for row in features),
        "capture_seconds": time.perf_counter() - started,
        "artifact": {
            "path": artifact.name,
            "bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
            "tracked": False,
            "regenerate_with": (
                "python experiments/paper2_5_iterative_pra/"
                "precompute_natural_graph_features.py --device cuda"
            ),
        },
    }
    (args.output_dir / "natural_graph_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--routing-layer", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    data = ROOT / "data/.paper2_5_datasets"
    parser.add_argument(
        "--musique-dev", type=Path, default=data / "musique/data/musique_ans_v1.0_dev.jsonl"
    )
    parser.add_argument("--twowiki-dev", type=Path, default=data / "2wiki/dev.json")
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth"
    parser.add_argument(
        "--sample-manifest", type=Path, default=output / "selected_raw_annotations.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
