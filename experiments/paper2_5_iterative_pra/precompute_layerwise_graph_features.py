"""Capture exact multi-layer Q/K and contextualization features for Paper 2.5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
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
from pra_hf.layerwise_context import (
    LayerContextCollector,
    attention_token_metrics,
    branch_token_metrics,
    causal_radius_mask,
    directional_rotation,
    normalized_displacement,
    summarize,
)
from pra_hf.natural_reasoning_graph import (
    char_spans_to_token_spans,
    load_2wiki,
    load_musique,
    stable_partition,
)
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


LAYERS = (0, 4, 8, 12, 16, 20, 24, 27)
ENCODING_BLOCK_TOKENS = 256
LOCAL_TOKENS = 32
SEARCH_CHUNK_TOKENS = 128
CONTEXT_RADII = (1, 16, 32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(dataset: str, example_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", example_id)
    suffix = hashlib.sha1(example_id.encode("utf-8")).hexdigest()[:10]
    return f"{dataset}__{clean[:80]}__{suffix}.pt"


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_selected(args: argparse.Namespace):
    selected_ids = {
        json.loads(line)["example_id"]
        for line in args.sample_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = load_musique(args.musique_dev) + load_2wiki(args.twowiki_dev)
    selected = sorted(
        (row for row in rows if row.example_id in selected_ids),
        key=lambda row: row.example_id,
    )
    if {row.example_id for row in selected} != selected_ids:
        raise ValueError("Selected manifest and local dataset identities differ.")
    return selected


def _token_classes(total: int, spans: dict[str, tuple[int, int]]) -> dict[str, torch.Tensor]:
    evidence = torch.zeros(total, dtype=torch.bool)
    for start, end in spans.values():
        evidence[int(start) : int(end)] = True
    oracle_parent = torch.zeros(total, dtype=torch.bool)
    for parent_start in range(0, total, SEARCH_CHUNK_TOKENS):
        parent_end = min(parent_start + SEARCH_CHUNK_TOKENS, total)
        if evidence[parent_start:parent_end].any():
            oracle_parent[parent_start:parent_end] = True
    return {
        "all": torch.ones(total, dtype=torch.bool),
        "evidence": evidence,
        "oracle_parent_non_evidence": oracle_parent & ~evidence,
        "distractor": ~oracle_parent,
    }


def _append_metric(
    accumulator,
    layer: int,
    token_classes: dict[str, torch.Tensor],
    block_start: int,
    block_end: int,
    metric: str,
    values: torch.Tensor,
    *,
    radius: str,
) -> None:
    token_values = values[0].detach().float().cpu()
    for token_class, global_mask in token_classes.items():
        mask = global_mask[block_start:block_end]
        if mask.any():
            accumulator[(layer, token_class, radius, metric)].append(token_values[mask])


@torch.no_grad()
def _capture_example(handle, collector, source_ids, token_classes, layers):
    layer_features = {
        layer: {
            "local_pre_query": [],
            "local_pre_key": [],
            "local_token_mask": [],
            "local_spans": [],
        }
        for layer in layers
    }
    metrics = defaultdict(list)
    total = int(source_ids.shape[1])
    full_forward_seconds = 0.0
    restricted_forward_seconds = {radius: 0.0 for radius in CONTEXT_RADII}
    full_reproduction_max_error = 0.0
    for block_start in range(0, total, ENCODING_BLOCK_TOKENS):
        block_end = min(block_start + ENCODING_BLOCK_TOKENS, total)
        block_ids = source_ids[:, block_start:block_end].to(handle.device)
        positions = torch.arange(block_start, block_end, device=handle.device).unsqueeze(0)
        for adapter in handle.adapters.values():
            adapter.begin_capture(positions)
        handle.set_attention_diagnostics(True)
        collector.clear()
        if handle.device.type == "cuda":
            torch.cuda.synchronize(handle.device)
        started = time.perf_counter()
        handle.model(
            input_ids=block_ids,
            attention_mask=torch.ones_like(block_ids),
            position_ids=positions,
            use_cache=False,
        )
        if handle.device.type == "cuda":
            torch.cuda.synchronize(handle.device)
        full_forward_seconds += time.perf_counter() - started
        collector.validate()
        captures = {
            layer: handle.adapters[layer].consume_capture() for layer in layers
        }
        full_states = {
            layer: collector.snapshots[layer].attention_input.detach().clone()
            for layer in layers
        }
        block_evidence = token_classes["evidence"][block_start:block_end].unsqueeze(0)
        for layer in layers:
            snapshot = collector.snapshots[layer]
            native = captures[layer]
            pre_query = native.pre_query[0].permute(1, 0, 2)
            pre_key = native.pre_key[0].permute(1, 0, 2)
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
                layer_features[layer]["local_pre_query"].append(query)
                layer_features[layer]["local_pre_key"].append(key)
                layer_features[layer]["local_token_mask"].append(mask)
                layer_features[layer]["local_spans"].append(
                    (block_start + local_start, block_start + local_end)
                )
            baseline = branch_token_metrics(snapshot)
            if snapshot.attention_weights is None:
                raise RuntimeError(f"Layer {layer} did not retain native attention weights.")
            baseline.update(
                attention_token_metrics(
                    snapshot.attention_weights,
                    local_window=LOCAL_TOKENS,
                    evidence_mask=block_evidence.to(snapshot.attention_weights.device),
                )
            )
            for metric, values in baseline.items():
                for radius in ("full", "1", "16", "32"):
                    _append_metric(
                        metrics,
                        layer,
                        token_classes,
                        block_start,
                        block_end,
                        metric,
                        values,
                        radius=radius,
                    )
            zeros = torch.zeros_like(next(iter(baseline.values())))
            for metric in ("intervention_displacement", "intervention_rotation"):
                _append_metric(
                    metrics,
                    layer,
                    token_classes,
                    block_start,
                    block_end,
                    metric,
                    zeros,
                    radius="full",
                )

        handle.set_attention_diagnostics(False)
        # A custom full mask validates that interventions alter visibility only.
        collector.clear()
        full_mask = causal_radius_mask(
            block_ids.shape[1], None, device=handle.device, dtype=torch.float32
        )
        handle.model(
            input_ids=block_ids,
            attention_mask={"full_attention": full_mask},
            position_ids=positions,
            use_cache=False,
        )
        for layer in layers:
            error = (
                collector.snapshots[layer].attention_input.float() - full_states[layer].float()
            ).abs().max()
            full_reproduction_max_error = max(full_reproduction_max_error, float(error))

        for radius in CONTEXT_RADII:
            collector.clear()
            mask = causal_radius_mask(
                block_ids.shape[1], radius, device=handle.device, dtype=torch.float32
            )
            if handle.device.type == "cuda":
                torch.cuda.synchronize(handle.device)
            started = time.perf_counter()
            handle.model(
                input_ids=block_ids,
                attention_mask={"full_attention": mask},
                position_ids=positions,
                use_cache=False,
            )
            if handle.device.type == "cuda":
                torch.cuda.synchronize(handle.device)
            restricted_forward_seconds[radius] += time.perf_counter() - started
            for layer in layers:
                restricted = collector.snapshots[layer].attention_input
                displacement = normalized_displacement(full_states[layer], restricted)
                rotation = directional_rotation(full_states[layer], restricted)
                _append_metric(
                    metrics,
                    layer,
                    token_classes,
                    block_start,
                    block_end,
                    "intervention_displacement",
                    displacement,
                    radius=str(radius),
                )
                _append_metric(
                    metrics,
                    layer,
                    token_classes,
                    block_start,
                    block_end,
                    "intervention_rotation",
                    rotation,
                    radius=str(radius),
                )
    for layer in layers:
        for key in ("local_pre_query", "local_pre_key", "local_token_mask"):
            layer_features[layer][key] = torch.stack(layer_features[layer][key])
    return (
        layer_features,
        metrics,
        full_forward_seconds,
        restricted_forward_seconds,
        full_reproduction_max_error,
    )


def _context_rows(example, token_classes, metrics, layers):
    rows = []
    metric_names = sorted({key[3] for key in metrics})
    for layer in layers:
        for token_class in token_classes:
            for radius in ("full", "1", "16", "32"):
                row = {
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "partition": stable_partition(example.example_id),
                    "layer": layer,
                    "token_class": token_class,
                    "context_radius": radius,
                    "token_count": int(token_classes[token_class].sum()),
                }
                for metric in metric_names:
                    values = metrics.get((layer, token_class, radius, metric), [])
                    if not values:
                        continue
                    summary = summarize(torch.cat(values))
                    row[metric] = summary["mean"]
                    row[f"{metric}_median"] = summary["median"]
                    row[f"{metric}_p90"] = summary["p90"]
                rows.append(row)
    return rows


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
    if int(model.config.num_hidden_layers) != 28:
        raise ValueError("Pinned Qwen layer count changed; layer indices require revalidation.")
    layers = tuple(args.layers)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=layers,
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
    handle.set_memory_enabled(False)
    examples = _load_selected(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "layerwise_feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    context_rows, cache_entries = [], []
    started = time.perf_counter()
    with LayerContextCollector(handle.model.model.layers, layers) as collector:
        for index, example in enumerate(examples, start=1):
            encoded = tokenizer(
                example.source,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=False,
            )
            offsets = encoded.pop("offset_mapping")[0].tolist()
            source_ids = encoded.input_ids.cpu()
            node_spans = char_spans_to_token_spans(offsets, example.nodes)
            token_classes = _token_classes(int(source_ids.shape[1]), node_spans)
            example_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            (
                layer_features,
                metrics,
                full_seconds,
                restricted_seconds,
                reproduction_error,
            ) = _capture_example(handle, collector, source_ids, token_classes, layers)
            payload = {
                "schema_version": "1.0",
                "dataset": example.dataset,
                "example_id": example.example_id,
                "partition": stable_partition(example.example_id),
                "question": example.question,
                "question_type": example.question_type,
                "annotated_hops": example.annotated_hops,
                "graph_type": example.graph_type,
                "source_tokens": int(source_ids.shape[1]),
                "source_token_ids": source_ids[0],
                "node_token_spans": node_spans,
                "nodes": [asdict(node) for node in example.nodes],
                "annotated_edges": example.annotated_edges,
                "root_node_ids": example.root_node_ids,
                "layers": layer_features,
            }
            artifact = cache_dir / _safe_name(example.dataset, example.example_id)
            torch.save(payload, artifact)
            layer_bytes = {
                str(layer): _tensor_bytes(layer_features[layer]) for layer in layers
            }
            cache_entries.append(
                {
                    "dataset": example.dataset,
                    "example_id": example.example_id,
                    "partition": stable_partition(example.example_id),
                    "source_tokens": int(source_ids.shape[1]),
                    "path": str(artifact.relative_to(args.output_dir)).replace("\\", "/"),
                    "bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                    "layer_tensor_bytes": layer_bytes,
                    "full_forward_seconds": full_seconds,
                    "restricted_forward_seconds": {
                        str(key): value for key, value in restricted_seconds.items()
                    },
                    "full_context_reproduction_max_abs_error": reproduction_error,
                    "example_seconds": time.perf_counter() - example_started,
                    "peak_gpu_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                }
            )
            context_rows.extend(_context_rows(example, token_classes, metrics, layers))
            print(
                f"[layerwise {index}/{len(examples)}] {example.dataset} "
                f"{example.example_id} tokens={source_ids.shape[1]} bytes={artifact.stat().st_size}",
                flush=True,
            )
    _write_csv(args.output_dir / "layerwise_context_rows.csv", context_rows)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "total_decoder_layers": int(model.config.num_hidden_layers),
        "layer_indexing": "zero-based decoder block; Q/K use that block's normalized attention input",
        "selected_layers": list(layers),
        "backbone_frozen": True,
        "training_performed": False,
        "encoding_block_tokens": ENCODING_BLOCK_TOKENS,
        "search_chunk_tokens": SEARCH_CHUNK_TOKENS,
        "local_native_tokens": LOCAL_TOKENS,
        "context_radii": ["full", *CONTEXT_RADII],
        "intervention_policy": (
            "causal K/V visibility mask; token IDs and absolute logical position IDs preserved"
        ),
        "attention_branch_semantics": (
            "native o_proj output immediately before residual addition; Qwen3 pre-norm"
        ),
        "ffn_branch_semantics": "native MLP output immediately before residual addition",
        "examples": len(examples),
        "source_tokens": sum(entry["source_tokens"] for entry in cache_entries),
        "capture_seconds": time.perf_counter() - started,
        "cache_bytes": sum(entry["bytes"] for entry in cache_entries),
        "cache_tracked": False,
        "regenerate_with": (
            "python -m experiments.paper2_5_iterative_pra."
            "precompute_layerwise_graph_features --device cuda"
        ),
        "entries": cache_entries,
    }
    (args.output_dir / "layerwise_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    data = ROOT / "data/.paper2_5_datasets"
    parser.add_argument(
        "--musique-dev", type=Path, default=data / "musique/data/musique_ans_v1.0_dev.jsonl"
    )
    parser.add_argument("--twowiki-dev", type=Path, default=data / "2wiki/dev.json")
    output = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/layerwise_graph"
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=ROOT
        / "docs/papers/shared/results/paper2_5_iterative_pra/natural_graph_depth/selected_raw_annotations.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=output)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
