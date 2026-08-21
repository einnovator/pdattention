"""Capture query-only and contextual Qwen representations for Paper 3.5.

The backbone is frozen.  Query-only rows use the exact chat-formatted prompt
already used by the frozen Paper-2.5 query cache and pool only the declared
question span.  Contextual rows prepend source text and are retained solely as
an expensive diagnostic upper bound.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata  # noqa: E402
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION  # noqa: E402
from experiments.paper2_hf.routing.run_query_strategies import (  # noqa: E402
    load_split_examples,
)
from pra_torch.hf import token_span_from_offsets  # noqa: E402
from pra_hf.self_router import (  # noqa: E402
    native_qk_representation,
    pool_query_tokens,
    query_span_mask,
)


HIDDEN_DEPTHS = (4, 7, 14, 18, 28)
NATIVE_LAYERS = (3, 13, 23, 27)
FIRST_MEMORY_LAYER = 24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _context_prompt(tokenizer, source: str, question: str, max_tokens: int):
    """Build a source-plus-question upper-bound prompt and recover the Q span."""

    question = question.strip()
    content = (
        "Use the context to answer briefly and directly.\n"
        f"Context:\n{source.strip()}\n\nQuestion: {question}"
    )
    rendered = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if tokenizer.chat_template
        else content
    )
    marker = f"Question: {question}"
    marker_start = rendered.rfind(marker)
    if marker_start < 0:
        raise ValueError("Contextual prompt lost the exact question marker.")
    char_start = marker_start + len("Question: ")
    char_end = char_start + len(question)
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_tokens,
    )
    tokenizer.truncation_side = previous
    offsets = encoded.pop("offset_mapping")[0].tolist()
    return encoded, token_span_from_offsets(offsets, char_start, char_end)


@torch.no_grad()
def _timed_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Capture all hidden states and cumulative layer latency in one forward."""

    layers = model.model.layers
    layer_times: list[float | None] = [None] * len(layers)
    handles = []
    embedding_time: float | None = None
    if input_ids.device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        embedding_event = torch.cuda.Event(enable_timing=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in layers]
        handles.append(
            model.model.embed_tokens.register_forward_hook(
                lambda _m, _a, _o: embedding_event.record()
            )
        )
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(lambda _m, _a, _o, i=index: events[i].record()))
        start.record()
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        torch.cuda.synchronize(input_ids.device)
        embedding_time = start.elapsed_time(embedding_event) / 1000.0
        for index, event in enumerate(events):
            layer_times[index] = start.elapsed_time(event) / 1000.0
    else:
        started = time.perf_counter()

        def record_embedding(_module, _args, _output):
            nonlocal embedding_time
            embedding_time = time.perf_counter() - started

        handles.append(model.model.embed_tokens.register_forward_hook(record_embedding))
        for index, layer in enumerate(layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _m, _a, _o, i=index: layer_times.__setitem__(
                        i, time.perf_counter() - started
                    )
                )
            )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
    for handle in handles:
        handle.remove()
    if embedding_time is None:
        raise RuntimeError("Embedding timing hook did not execute.")
    return output, tuple(float(value) for value in layer_times), float(embedding_time)


def _latency_at_depth(
    layer_times: tuple[float, ...], depth: int, embedding_time: float
) -> float:
    if depth <= 0:
        return embedding_time
    return layer_times[depth - 1]


@torch.no_grad()
def _representations(
    model,
    output,
    question_span: tuple[int, int],
    layer_times: tuple[float, ...],
    embedding_time: float,
    *,
    contextual: bool = False,
) -> dict[str, dict[str, Any]]:
    """Pool the staged representation set without retaining token-level states."""

    start, end = question_span
    mask = query_span_mask(output.hidden_states[0].shape[1], start, end, device=output.hidden_states[0].device)
    rows: dict[str, dict[str, Any]] = {}

    def add(name: str, vector: torch.Tensor, depth: int, family: str, pooling: str) -> None:
        rows[name] = {
            "vector": vector[0].detach().float().cpu(),
            "depth": int(depth),
            "family": family,
            "pooling": pooling,
            "prefill_latency_seconds": _latency_at_depth(
                layer_times, depth, embedding_time
            ),
        }

    for pooling in ("mean", "last"):
        add(
            f"S8_context_embed_{pooling}" if contextual else f"S2_embed_{pooling}",
            pool_query_tokens(output.hidden_states[0], mask, pooling),
            0,
            "embedding",
            pooling,
        )
    for depth in HIDDEN_DEPTHS:
        poolings = ("mean", "last") if depth in (14, 28) else ("mean",)
        stage = "S3" if depth < 14 else "S4" if depth < 28 else "S5"
        for pooling in poolings:
            add(
                f"S8_context_hidden_l{depth}_{pooling}"
                if contextual
                else f"{stage}_hidden_l{depth}_{pooling}",
                pool_query_tokens(output.hidden_states[depth], mask, pooling),
                depth,
                "hidden_state",
                pooling,
            )
    for layer_index in NATIVE_LAYERS:
        kinds = ("q", "k", "qk") if layer_index in (13, 23) else ("q",)
        for kind in kinds:
            stage = "S6" if kind == "q" else "S7"
            add(
                f"S8_context_native_{kind}_l{layer_index}_mean"
                if contextual
                else f"{stage}_native_{kind}_l{layer_index}_mean",
                native_qk_representation(
                    model,
                    output.hidden_states[layer_index],
                    layer_index,
                    mask,
                    kind=kind,
                    pooling="mean",
                ),
                # The projection consumes the input to zero-based layer i.
                # Charge through that layer (i + 1) so cumulative latency and
                # abstract work do not treat the native projection as free.
                layer_index + 1,
                f"native_{kind}",
                "mean",
            )
    return rows


def _config_rows(feature_rows: list[dict[str, Any]], total_layers: int) -> dict[str, Any]:
    sample = feature_rows[0]["representations"]
    return {
        "schema_version": "1.0",
        "backbone": MODEL_ID,
        "backbone_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "total_layers": total_layers,
        "first_memory_consumer_layer": FIRST_MEMORY_LAYER,
        "query_region": "explicit question span inside the normal question-only chat prompt",
        "representations": {
            name: {
                "family": value["family"],
                "pooling": value["pooling"],
                "depth": value["depth"],
                "width": int(value["vector"].numel()),
                "contextual_upper_bound": name.startswith("S8_context_"),
                "reuse_eligible_by_depth": value["depth"] <= FIRST_MEMORY_LAYER
                and not name.startswith("S8_context_"),
            }
            for name, value in sample.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Place checkpoint tensors directly on the target device before opening
    # dataset artifacts. This avoids a transient duplicate CPU/GPU copy on
    # small-pagefile Windows research machines.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gc.collect()
    query_cache = torch.load(
        args.query_feature_file, map_location="cpu", weights_only=False, mmap=True
    )
    examples = load_split_examples(
        args.cache_dir, args.examples, args.example_offset, args.seed
    )
    by_id = {row["id"]: row for row in examples}
    if [row["example_id"] for row in query_cache] != [row["id"] for row in examples]:
        raise ValueError("Dataset examples do not match the frozen query cache order.")

    # Warm up kernels once without retaining activations.
    first = query_cache[0]
    warm_ids = first["prompt_input_ids"].unsqueeze(0).to(device)
    model(input_ids=warm_ids, attention_mask=torch.ones_like(warm_ids), use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    rows = []
    for index, cached in enumerate(query_cache, start=1):
        example = by_id[cached["example_id"]]
        query_ids = cached["prompt_input_ids"].unsqueeze(0).to(device)
        query_mask = torch.ones_like(query_ids)
        query_output, query_times, query_embedding_time = _timed_forward(
            model, query_ids, query_mask
        )
        query_reps = _representations(
            model,
            query_output,
            tuple(cached["question_span"]),
            query_times,
            query_embedding_time,
        )

        contextual, contextual_span = _context_prompt(
            tokenizer, example["source"], example["question"], args.context_tokens
        )
        contextual = contextual.to(device)
        context_output, context_times, context_embedding_time = _timed_forward(
            model, contextual.input_ids, contextual.attention_mask
        )
        context_all = _representations(
            model,
            context_output,
            contextual_span,
            context_times,
            context_embedding_time,
            contextual=True,
        )
        # S8 is an upper bound, so retain only representative depths rather
        # than multiplying every source by every contextual variant.
        context_reps = {
            name: value
            for name, value in context_all.items()
            if name
            in {
                "S8_context_hidden_l14_mean",
                "S8_context_hidden_l28_mean",
                "S8_context_native_q_l13_mean",
                "S8_context_native_q_l23_mean",
            }
        }
        representations = {**query_reps, **context_reps}
        rows.append(
            {
                "dataset": cached["dataset"],
                "example_id": cached["example_id"],
                "question": cached["question"],
                "query_prompt_tokens": int(query_ids.shape[1]),
                "query_tokens": int(cached["question_tokens"]),
                "context_prompt_tokens": int(contextual.input_ids.shape[1]),
                "representations": representations,
            }
        )
        print(
            f"[self-router {index}/{len(query_cache)}] {cached['dataset']} "
            f"{cached['example_id']} q={query_ids.shape[1]} ctx={contextual.input_ids.shape[1]}",
            flush=True,
        )
        del query_output, context_output

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "self_router_representations.pt"
    torch.save(rows, feature_path)
    configs = _config_rows(rows, len(model.model.layers))
    (args.output_dir / "self_router_representation_configs.json").write_text(
        json.dumps(configs, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "examples": len(rows),
        "dataset_counts": {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in sorted({row["dataset"] for row in rows})
        },
        "feature_file": feature_path.name,
        "feature_bytes": feature_path.stat().st_size,
        "feature_sha256": _sha256(feature_path),
        "query_only": "normal chat prompt without source documents; pool explicit question span",
        "contextual_upper_bound": f"source-plus-query ordinary prompt truncated to {args.context_tokens} tokens",
        "latency": "single warmed forward; cumulative CUDA events recorded at decoder exits",
        "backbone_mutation": False,
    }
    (args.output_dir / "self_router_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _existing_or_sibling(relative: str) -> Path:
    local = ROOT / relative
    if local.exists():
        return local
    return ROOT.parent / "pdattention-iter-gist" / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--query-feature-file",
        type=Path,
        default=_existing_or_sibling(
            "docs/papers/shared/results/paper2_5_iterative_pra/"
            "query_entry_facets/query_entry_features.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
