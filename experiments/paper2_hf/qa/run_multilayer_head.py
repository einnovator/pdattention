"""Re-evaluate a bounded implicit prompt head through a late PRA layer band."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata, write_artifacts
from experiments.paper2_hf.qa.run_multilayer_pra import (
    _cache_bytes,
    _diagnostic_sums,
    _routing_ranking_trace,
    _selection_overlap,
    _sync,
    layer_schedules,
)
from experiments.paper2_hf.qa.run_oracle_memory_use import (
    _answer_ids,
    _memory_attention_trace,
)
from experiments.paper2_hf.qa.run_smoke import answer_metrics, evidence_token_spans
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_torch.hf import PRAHFConfig, inject_pra, load_hf_routing_projection


def _route_tail_once(handle, prepared, route_layer: int, device: torch.device) -> dict:
    """Capture one tail query and rank the implicit-head index once."""
    handle.configure_memory_layers(set())
    adapter = handle.adapters[route_layer]
    adapter.begin_capture(prepared.position_ids)
    _sync(device)
    query_started = time.perf_counter()
    with torch.no_grad():
        handle.model(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            position_ids=prepared.position_ids,
            use_cache=False,
        )
    _sync(device)
    query_seconds = time.perf_counter() - query_started
    captured = adapter.consume_capture()
    query = adapter._routing_query_states(
        captured.hidden_states, captured.pre_query, captured.post_query
    )
    _sync(device)
    routing_started = time.perf_counter()
    selected, rankings = adapter.pra_core.route_memory(query)
    _sync(device)
    return {
        "selected": selected,
        "rankings": rankings,
        "route_layer": route_layer,
        "query_encoding_seconds": query_seconds,
        "routing_seconds": time.perf_counter() - routing_started,
    }


def _score_condition(
    handle,
    tokenizer,
    prepared,
    answer_ids,
    evidence_spans,
    condition: str,
    layers: tuple[int, ...],
    fixed,
    new_tokens: int,
    device: torch.device,
) -> dict:
    handle.configure_memory_layers(set(layers), fixed_selections=fixed)
    prompt_tokens = int(prepared.input_ids.shape[1])
    full_ids = torch.cat((prepared.input_ids, answer_ids.to(device)), dim=1)
    full_mask = torch.ones_like(full_ids)
    full_positions = torch.arange(
        prepared.head_tokens,
        prepared.head_tokens + full_ids.shape[1],
        device=device,
    ).unsqueeze(0)
    prediction_positions = list(range(prompt_tokens - 1, full_ids.shape[1] - 1))
    handle.set_attention_diagnostics(True)
    _sync(device)
    score_started = time.perf_counter()
    with torch.no_grad():
        output = handle.model(
            input_ids=full_ids,
            attention_mask=full_mask,
            position_ids=full_positions,
            use_cache=False,
        )
    _sync(device)
    teacher_forced_seconds = time.perf_counter() - score_started
    logits = output.logits[:, prediction_positions, :].float()
    targets = answer_ids.to(device)
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)[0]
    first_logits = logits[0, 0]
    first_target = int(targets[0, 0])
    attention = {
        str(layer): _memory_attention_trace(
            handle.adapters[layer],
            handle.adapters[layer].last_attention_weights,
            prediction_positions,
            evidence_spans,
        )
        for layer in layers
    }
    handle.set_attention_diagnostics(False)

    _sync(device)
    generation_started = time.perf_counter()
    with torch.no_grad():
        generated = handle.model.generate(
            input_ids=prepared.input_ids,
            attention_mask=prepared.attention_mask,
            position_ids=prepared.position_ids,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
        )
    _sync(device)
    generation_seconds = time.perf_counter() - generation_started
    prediction = tokenizer.decode(
        generated[0, prompt_tokens:], skip_special_tokens=True
    ).strip()
    diagnostics = handle.diagnostics_by_layer()
    return {
        "condition": condition,
        "pra_layers": list(layers),
        "pra_layer_count": len(layers),
        "gold_sequence_logprob": float(token_log_probs.sum().item()),
        "gold_mean_token_logprob": float(token_log_probs.mean().item()),
        "gold_first_token_probability": float(
            first_logits.softmax(dim=-1)[first_target].item()
        ),
        "gold_first_token_rank": int(
            (first_logits > first_logits[first_target]).sum().item()
        ) + 1,
        "prediction": prediction,
        **answer_metrics(prediction, "cobalt"),
        "attention_by_layer": attention,
        "diagnostics_by_layer": diagnostics,
        "teacher_forced_seconds": teacher_forced_seconds,
        "generation_seconds": generation_seconds,
        **_diagnostic_sums(diagnostics, layers),
    }


def run(args) -> dict:
    torch.manual_seed(args.seed)
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
    projection = (
        load_hf_routing_projection(args.checkpoint, device=device)
        if args.router == "learned"
        else None
    )
    schedules = layer_schedules(int(model.config.num_hidden_layers))
    layers = schedules[args.schedule]
    all_layers = schedules["all"]
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=all_layers,
            model_max_context_tokens=args.native_tokens,
            max_prompt_direct_tokens=args.tail_tokens,
            encoding_block_tokens=args.encoding_block_tokens,
            routing_chunk_tokens=args.routing_chunk_tokens,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=args.top_k,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    fact = "The verification word is cobalt."
    filler = "The archive contains routine administrative notes with no verification word."
    text = " ".join(
        [fact, *([filler] * 36), "Question: What is the verification word? Answer with one word:"]
    )
    full_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    evidence_spans = evidence_token_spans(tokenizer, text, [fact])
    prepared = handle.prepare_long_prompt(full_ids)
    entry = handle.cache.get("#__head")
    if entry is None:
        raise RuntimeError("The long prompt did not publish an implicit head.")
    route = _route_tail_once(handle, prepared, max(all_layers), device)
    fixed = handle.map_chunk_identities_to_layers(route["selected"], layers)
    answer_ids = _answer_ids(tokenizer, "cobalt")
    baseline = _score_condition(
        handle, tokenizer, prepared, answer_ids, evidence_spans,
        "no_memory", (), None, args.new_tokens, device,
    )
    routed = _score_condition(
        handle, tokenizer, prepared, answer_ids, evidence_spans,
        f"{args.router}_{args.schedule}", layers, fixed, args.new_tokens, device,
    )
    routed["gold_logprob_delta_vs_none"] = (
        routed["gold_sequence_logprob"] - baseline["gold_sequence_logprob"]
    )
    routed["gold_mean_logprob_delta_vs_none"] = (
        routed["gold_mean_token_logprob"] - baseline["gold_mean_token_logprob"]
    )
    baseline["gold_logprob_delta_vs_none"] = 0.0
    baseline["gold_mean_logprob_delta_vs_none"] = 0.0
    cache_bytes = _cache_bytes(entry, all_layers)
    routing_trace = _routing_ranking_trace(
        route["rankings"], route["selected"], evidence_spans
    )
    return {
        "runtime": runtime_metadata(),
        "protocol": "bounded implicit head, route once at layer 27, layer-native K/V in a contiguous late band",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "router": args.router,
        "schedule": args.schedule,
        "resolved_layers": list(layers),
        "logical_prompt_tokens": int(full_ids.shape[1]),
        "head_tokens": prepared.head_tokens,
        "direct_tail_tokens": int(prepared.input_ids.shape[1]),
        "tail_position_start": int(prepared.position_ids[0, 0]),
        "tail_position_end": int(prepared.position_ids[0, -1]),
        "evidence_spans": evidence_spans,
        "selection": {
            **_selection_overlap(route["selected"], evidence_spans),
            **routing_trace,
            "route_layer": route["route_layer"],
            "query_encoding_seconds": route["query_encoding_seconds"],
            "routing_seconds": route["routing_seconds"],
        },
        "cache": cache_bytes,
        "rows": [baseline, routed],
        "max_native_operation_tokens": handle.max_native_operation_tokens,
        "native_limit_violations": handle.native_limit_violations,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }


def parse_args() -> argparse.Namespace:
    checkpoint = (
        ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
        / "routing" / "learned_adapter" / "checkpoints"
        / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--router", choices=("learned", "zero_shot"), default="learned")
    parser.add_argument("--schedule", choices=("last_8", "last_half"), default="last_8")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--tail-tokens", type=int, default=128)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--encoding-block-tokens", type=int, default=128)
    parser.add_argument("--routing-chunk-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path, default=checkpoint)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "docs" / "papers" / "shared" / "results"
        / "paper2_hf" / "multilayer_pra",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    artifact = run(arguments)
    stem = f"implicit_head_{arguments.router}_{arguments.schedule}"
    paths = write_artifacts(artifact, arguments.output_dir, stem)
    print(json.dumps({"artifacts": [str(path) for path in paths], "rows": artifact["rows"]}, indent=2))
