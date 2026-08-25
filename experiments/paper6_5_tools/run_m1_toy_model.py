"""Train and evaluate the Paper 6.5 M1 opaque-tool causal model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for import_path in (ROOT, ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from data.agent_resources import generate_agent_catalog, synthetic_semantic_vector
from data.agent_tool_language import (
    ToolLanguageExample,
    compact_definition,
    compact_query,
    continuation_suffix,
    expected_answer,
    expected_call,
    formatted_argument,
    make_tool_examples,
    parse_call,
    render_definitions,
    render_supervised_trajectory,
    schema_code,
)
from data.tokenizer import PRATokenizer
from pra_hf.agent_resources import (
    DiscoveryHint,
    DiscoveryRequest,
    PersistentResourceIndex,
    ResourceDiscoveryEngine,
)
from pra_torch.config import PRAConfig
from pra_torch.model import TinyPRAModel


DEFAULT_SEEDS = (11, 23, 37, 53, 71)
DEFAULT_CATALOG_SIZES = (8, 32, 128)
CONDITIONS = (
    "oracle_memory",
    "discovered_memory",
    "shuffled_memory",
    "disabled_memory",
    "direct_selected",
    "eager_catalog",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _model_config(
    vocab_size: int,
    max_seq_len: int,
    *,
    d_model: int,
    n_layers: int,
    d_ff: int,
    pra_layer_ids: tuple[int, ...],
) -> PRAConfig:
    return PRAConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        model_max_context_tokens=max_seq_len,
        position_encoding="rope",
        model_variant="td_layered_pra",
        pra_layer_ids=pra_layer_ids,
        self_attention_window=None,
        top_k_references=1,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0,
        memory_transport="native_kv",
        max_materialized_memory_tokens=64,
        detail_materialization="selected_chunks",
        chunking_mode="none",
        cache_build_mode="trainable_gist",
        routing_backend="legacy",
        collect_routing_metrics=True,
    )


def build_model(
    tokenizer: PRATokenizer,
    *,
    seed: int,
    max_seq_len: int,
    device: torch.device,
    d_model: int = 96,
    n_layers: int = 4,
    d_ff: int = 384,
    pra_layer_ids: tuple[int, ...] = (1, 3),
):
    torch.manual_seed(seed)
    model = TinyPRAModel(
        _model_config(
            tokenizer.vocab_size,
            max_seq_len,
            d_model=d_model,
            n_layers=n_layers,
            d_ff=d_ff,
            pra_layer_ids=pra_layer_ids,
        )
    ).to(device)
    # Tied character embeddings materially improve held-out identity copying and
    # keep the comparison focused on memory presentation rather than output size.
    model.head.weight = model.token_emb.weight
    return model


def _publish_memory(model, tokenizer, resource, device) -> tuple[int, float]:
    model.clear_pra_cache()
    text = compact_definition(resource)
    started = time.perf_counter()
    entry = model.encode_reference_to_cache(
        resource.uri,
        text,
        tokenizer,
        device,
        metadata={"resource_kind": resource.kind, "version": resource.version},
    )
    model.pra_cache.put(entry)
    return len(tokenizer.encode(text)), (time.perf_counter() - started) * 1000.0


def _supervised_tensors(tokenizer, text, spans, device):
    ids = tokenizer.encode(text)
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    for start, stop in spans:
        labels[0, start - 1 : stop - 1] = torch.tensor(
            ids[start:stop], dtype=torch.long, device=device
        )
    return input_ids, labels


def _training_context(example, resources, mode, rng, eager_width):
    if mode == "memory":
        return (), example.resource
    if mode == "direct":
        return (example.resource,), None
    distractors = [resource for resource in resources if resource.uri != example.resource.uri]
    chosen = rng.sample(distractors, k=min(eager_width - 1, len(distractors)))
    definitions = [example.resource, *chosen]
    rng.shuffle(definitions)
    return tuple(definitions), None


def train_seed(
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    max_seq_len: int,
    eager_width: int,
    d_model: int,
    n_layers: int,
    d_ff: int,
    pra_layer_ids: tuple[int, ...],
    device: torch.device,
) -> tuple[TinyPRAModel, PRATokenizer, list[dict]]:
    tokenizer = PRATokenizer()
    model = build_model(
        tokenizer,
        seed=seed,
        max_seq_len=max_seq_len,
        device=device,
        d_model=d_model,
        n_layers=n_layers,
        d_ff=d_ff,
        pra_layer_ids=pra_layer_ids,
    )
    train_catalog = generate_agent_catalog(128, seed=seed + 10_000)
    resources = train_catalog.resources
    examples = make_tool_examples(
        resources,
        seed=seed + 20_000,
        count=max(steps, len(resources)),
        prefix=f"train-{seed}",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    rng = random.Random(seed + 30_000)
    rows = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        example = examples[step % len(examples)]
        draw = rng.random()
        mode = "memory" if draw < 0.50 else "direct" if draw < 0.80 else "eager"
        direct_definitions, memory_resource = _training_context(
            example, resources, mode, rng, eager_width
        )
        text, spans = render_supervised_trajectory(
            example, direct_definitions=direct_definitions
        )
        input_ids, labels = _supervised_tensors(tokenizer, text, spans, device)
        if input_ids.shape[1] > max_seq_len:
            raise RuntimeError(
                f"M1 training sequence {input_ids.shape[1]} exceeds {max_seq_len}."
            )
        model.clear_pra_cache()
        memory_tokens = 0
        if memory_resource is not None:
            memory_tokens, _ = _publish_memory(model, tokenizer, memory_resource, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, use_pra_memory=memory_resource is not None)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.clear_pra_cache()
        rows.append(
            {
                "seed": seed,
                "step": step + 1,
                "mode": mode,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
                "direct_tokens": int(input_ids.shape[1]),
                "memory_tokens": memory_tokens,
            }
        )
    elapsed = time.perf_counter() - started
    for row in rows:
        row["training_seconds"] = elapsed
        row["steps_per_second"] = steps / max(elapsed, 1e-9)
    return model, tokenizer, rows


def _greedy(model, tokenizer, prefix, *, use_memory, max_new_tokens, device):
    ids = tokenizer.encode(prefix)
    started = time.perf_counter()
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if len(ids) >= model.cfg.effective_model_max_context_tokens:
                break
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(tensor, use_pra_memory=use_memory)
            next_id = int(logits[0, -1].argmax().item())
            ids.append(next_id)
            generated.append(next_id)
            if tokenizer.decode([next_id]) == "\n":
                break
    return tokenizer.decode(generated), (time.perf_counter() - started) * 1000.0


def _teacher_nll(model, tokenizer, example, direct_definitions, use_memory, device):
    text, spans = render_supervised_trajectory(
        example, direct_definitions=direct_definitions
    )
    input_ids, labels = _supervised_tensors(tokenizer, text, spans, device)
    if input_ids.shape[1] > model.cfg.effective_model_max_context_tokens:
        return math.nan
    with torch.no_grad():
        logits = model(input_ids, use_pra_memory=use_memory)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
    return float(loss.cpu())


def _memory_diagnostics(model) -> tuple[int, int, str]:
    materialized = 0
    stored_bytes = 0
    selected = []
    for layer_id, diagnostics in model.pra_diagnostics_by_layer().items():
        materialized = max(
            materialized, int(diagnostics.get("memory_tokens_materialized", 0))
        )
        stored_bytes = max(
            stored_bytes, int(diagnostics.get("retrieved_kv_storage_bytes", 0))
        )
        for batch_hits in model.selected_chunks_by_layer().get(layer_id, []):
            selected.extend(hit.reference_uri for hit in batch_hits)
    return materialized, stored_bytes, "|".join(dict.fromkeys(selected))


def _discover(index, example):
    query = (
        f"Please {example.resource.metadata['action']} the "
        f"{example.resource.metadata['object']} in service family "
        f"{example.resource.metadata['family']}"
    )
    request = DiscoveryRequest(
        query=query,
        hint=DiscoveryHint("semantic", strict=True),
        namespace="synthetic",
        tenant_id="paper6_5",
        top_k=1,
    )
    engine = ResourceDiscoveryEngine(
        index,
        select_threshold=0.0,
        ask_threshold=0.0,
        margin_threshold=0.0,
    )
    trace = engine.discover(request)
    uri = trace.selected_uris[0] if trace.selected_uris else ""
    return uri, trace


def evaluate_condition(
    *,
    model,
    tokenizer,
    example,
    condition,
    catalog,
    index,
    shuffled_resource,
    device,
):
    logical_tokens = sum(len(tokenizer.encode(compact_definition(resource))) for resource in catalog.resources)
    selected_resource = None
    direct_definitions = ()
    discovery_uri = ""
    discovery_path = ""
    if condition == "oracle_memory":
        selected_resource = example.resource
    elif condition == "discovered_memory":
        discovery_uri, trace = _discover(index, example)
        discovery_path = ">".join(trace.executed_path)
        selected_resource = index.by_uri.get(discovery_uri)
    elif condition == "shuffled_memory":
        selected_resource = shuffled_resource
    elif condition == "direct_selected":
        direct_definitions = (example.resource,)
    elif condition == "eager_catalog":
        direct_definitions = catalog.resources

    model.clear_pra_cache()
    memory_tokens = 0
    cache_encode_ms = 0.0
    if selected_resource is not None:
        memory_tokens, cache_encode_ms = _publish_memory(
            model, tokenizer, selected_resource, device
        )
    use_memory = selected_resource is not None
    prefix = render_definitions(direct_definitions)
    call_prefix = prefix + compact_query(example)
    expected_slot_index = next(
        (
            index
            for index, resource in enumerate(direct_definitions)
            if resource.uri == example.resource.uri
        ),
        0,
    )
    expected_slot = f"@{expected_slot_index}"
    call_target = expected_call(example, slot=expected_slot)
    full_required = (
        call_prefix
        + call_target
        + continuation_suffix(example)
        + expected_answer(example)
    )
    fits = len(tokenizer.encode(full_required)) - 1 <= model.cfg.effective_model_max_context_tokens
    base = {
        "seed": catalog.seed,
        "catalog_size": len(catalog.resources),
        "example_id": example.example_id,
        "condition": condition,
        "target_uri": example.resource.uri,
        "discovered_uri": discovery_uri,
        "discovery_correct": int(discovery_uri == example.resource.uri) if discovery_uri else "",
        "discovery_path": discovery_path,
        "context_fit": int(fits),
        "logical_catalog_tokens": logical_tokens,
        "selected_definition_tokens": len(tokenizer.encode(compact_definition(example.resource))),
        "native_prompt_tokens": len(tokenizer.encode(call_prefix)),
        "cache_encode_ms": cache_encode_ms,
        "memory_source_tokens": memory_tokens,
        "side_effect_class": example.resource.side_effect_class.value,
        "host_authorized_fixture": 1,
    }
    if not fits:
        model.clear_pra_cache()
        return {
            **base,
            "teacher_nll": "",
            "generated_call": "",
            "call_syntax_valid": 0,
            "slot_exact": 0,
            "schema_exact": 0,
            "tool_exact": 0,
            "argument_exact": 0,
            "call_exact": 0,
            "execution_success": 0,
            "conditional_continuation_exact": 0,
            "end_to_end_success": 0,
            "call_generation_ms": "",
            "continuation_generation_ms": "",
            "materialized_kv_tokens": 0,
            "materialized_kv_bytes": 0,
            "model_selected_uris": "",
        }

    teacher_nll = _teacher_nll(
        model, tokenizer, example, direct_definitions, use_memory, device
    )
    generated_call, call_ms = _greedy(
        model,
        tokenizer,
        call_prefix,
        use_memory=use_memory,
        max_new_tokens=len(call_target) + 4,
        device=device,
    )
    materialized, kv_bytes, model_selected = _memory_diagnostics(model)
    parsed = parse_call(generated_call)
    slot_exact = int(parsed is not None and parsed[0] == expected_slot)
    schema_exact = int(
        parsed is not None and parsed[1] == schema_code(example.resource)
    )
    argument_exact = int(
        parsed is not None and parsed[2] == formatted_argument(example)
    )
    call_exact = int(generated_call == call_target)

    bound_resource = None
    if parsed is not None and parsed[0].startswith("@"):
        try:
            slot_index = int(parsed[0][1:])
        except ValueError:
            slot_index = -1
        if direct_definitions and 0 <= slot_index < len(direct_definitions):
            bound_resource = direct_definitions[slot_index]
        elif selected_resource is not None and slot_index == 0:
            bound_resource = selected_resource
    selected_identity_correct = int(
        bound_resource is not None and bound_resource.uri == example.resource.uri
    )
    tool_exact = selected_identity_correct

    oracle_continuation_prefix = (
        call_prefix + call_target + continuation_suffix(example)
    )
    generated_answer, continuation_ms = _greedy(
        model,
        tokenizer,
        oracle_continuation_prefix,
        use_memory=use_memory,
        max_new_tokens=len(expected_answer(example)) + 2,
        device=device,
    )
    continuation_exact = int(generated_answer == expected_answer(example))
    execution_success = int(call_exact and selected_identity_correct)
    model.clear_pra_cache()
    active_definition_tokens = (
        materialized if use_memory else sum(
            len(tokenizer.encode(compact_definition(resource)))
            for resource in direct_definitions
        )
    )
    return {
        **base,
        "teacher_nll": teacher_nll,
        "generated_call": generated_call.replace("\n", "\\n"),
        "call_syntax_valid": int(parsed is not None),
        "slot_exact": slot_exact,
        "schema_exact": schema_exact,
        "tool_exact": tool_exact,
        "selected_identity_correct": selected_identity_correct,
        "argument_exact": argument_exact,
        "call_exact": call_exact,
        "execution_success": execution_success,
        "conditional_continuation_exact": continuation_exact,
        "end_to_end_success": int(execution_success and continuation_exact),
        "call_generation_ms": call_ms,
        "continuation_generation_ms": continuation_ms,
        "materialized_kv_tokens": materialized,
        "materialized_kv_bytes": kv_bytes,
        "model_selected_uris": model_selected,
        "active_definition_tokens": active_definition_tokens,
        "active_fraction": active_definition_tokens / max(logical_tokens, 1),
    }


def evaluate_seed(
    *,
    model,
    tokenizer,
    seed,
    catalog_sizes,
    examples_per_size,
    device,
):
    rows = []
    model.eval()
    for size in catalog_sizes:
        catalog = generate_agent_catalog(size, seed=seed)
        index = PersistentResourceIndex(
            catalog.resources,
            semantic_encoder=lambda text: synthetic_semantic_vector(text, dimensions=96),
            fingerprint_metadata={"semantic_encoder": "synthetic-concept-hash-96-v1"},
        )
        examples = make_tool_examples(
            catalog.resources,
            seed=seed + size,
            count=min(examples_per_size, size),
            prefix=f"test-{seed}-{size}",
        )
        positions = {resource.uri: index for index, resource in enumerate(catalog.resources)}
        for example in examples:
            position = positions[example.resource.uri]
            shuffled = catalog.resources[(position + 1) % len(catalog.resources)]
            for condition in CONDITIONS:
                rows.append(
                    evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        example=example,
                        condition=condition,
                        catalog=catalog,
                        index=index,
                        shuffled_resource=shuffled,
                        device=device,
                    )
                )
    return rows


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/papers/shared/results/paper6_5_tools/m1"))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--catalog-sizes", default=",".join(map(str, DEFAULT_CATALOG_SIZES)))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--eager-width", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=384)
    parser.add_argument("--pra-layers", default="1,3")
    parser.add_argument("--examples-per-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    seeds = parse_ints(args.seeds)
    sizes = parse_ints(args.catalog_sizes)
    pra_layers = parse_ints(args.pra_layers)
    training_rows = []
    evaluation_rows = []
    run_started = time.perf_counter()
    for seed in seeds:
        model, tokenizer, seed_training = train_seed(
            seed=seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            max_seq_len=args.max_seq_len,
            eager_width=args.eager_width,
            d_model=args.d_model,
            n_layers=args.layers,
            d_ff=args.d_ff,
            pra_layer_ids=pra_layers,
            device=device,
        )
        training_rows.extend(seed_training)
        evaluation_rows.extend(
            evaluate_seed(
                model=model,
                tokenizer=tokenizer,
                seed=seed,
                catalog_sizes=sizes,
                examples_per_size=args.examples_per_size,
                device=device,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(args.output_dir / "m1_training_history.csv", training_rows)
    _write_csv(args.output_dir / "m1_example_results.csv", evaluation_rows)
    manifest = {
        "status": "m1_raw_complete",
        "seeds": list(seeds),
        "catalog_sizes": list(sizes),
        "conditions": list(CONDITIONS),
        "steps_per_seed": args.steps,
        "learning_rate": args.learning_rate,
        "max_seq_len": args.max_seq_len,
        "eager_training_width": args.eager_width,
        "architecture": {
            "d_model": args.d_model,
            "layers": args.layers,
            "d_ff": args.d_ff,
            "pra_layers": list(pra_layers),
        },
        "examples_per_size": args.examples_per_size,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameter_count": sum(
            parameter.numel()
            for parameter in build_model(
                PRATokenizer(),
                seed=0,
                max_seq_len=args.max_seq_len,
                device=torch.device("cpu"),
                d_model=args.d_model,
                n_layers=args.layers,
                d_ff=args.d_ff,
                pra_layer_ids=pra_layers,
            ).parameters()
        ),
        "elapsed_seconds": time.perf_counter() - run_started,
        "scope": "causal toy mechanism; no real side effects or pretrained-model claim",
    }
    (args.output_dir / "m1_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({**manifest, "rows": len(evaluation_rows), "final_train_loss": mean(
        row["loss"] for row in training_rows[-min(100, len(training_rows)):]
    )}, indent=2))


if __name__ == "__main__":
    main()
