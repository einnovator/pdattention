"""Validate and run PRA-HF on the official Gemma 3 1B instruction checkpoint.

Gemma 3 alternates local sliding-window and global attention. This runner treats
that schedule as a host-model contract: local layers remain native and PRA is
injected only into a conservative late pair of global layers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from pra_torch.hf import (
    ATTENTION_INPUT_HIDDEN_STATE,
    PRAHFConfig,
    gemma3_global_layer_ids,
    inject_pra,
)
from pra_torch.memory import SelectedChunk


MODEL_ID = "google/gemma-3-1b-it"
MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
TOKENIZER_REVISION = MODEL_REVISION
SEEDS = (11, 23, 37, 53, 71)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.shape]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def access_preflight(model_id: str, revision: str, output: Path) -> dict:
    """Verify the exact official revision and gated config/tokenizer access."""
    result = {
        "runtime": runtime_metadata(),
        "model_id": model_id,
        "requested_revision": revision,
        "tokenizer_revision": revision,
        "access_requirement": (
            "Accept Google's Gemma usage terms for the official repository and "
            "authenticate the local Hugging Face client."
        ),
        "status": "checking",
    }
    try:
        metadata = model_info(model_id, revision=revision)
        result.update(
            {
                "resolved_revision": metadata.sha,
                "gated": metadata.gated,
                "license": next(
                    (
                        tag.split(":", 1)[1]
                        for tag in metadata.tags
                        if tag.startswith("license:")
                    ),
                    None,
                ),
                "architecture": (metadata.config or {}).get("architectures", [None])[0],
                "parameter_count_from_hub": (
                    int(metadata.safetensors.total) if metadata.safetensors else None
                ),
            }
        )
        config_path = hf_hub_download(model_id, "config.json", revision=revision)
        tokenizer_path = hf_hub_download(
            model_id, "tokenizer_config.json", revision=revision
        )
    except (GatedRepoError, HfHubHTTPError, OSError) as error:
        result.update(
            {
                "status": "blocked_external_access",
                "blocker": (
                    "The authenticated account has not been granted access to the "
                    "official Gemma repository. Accept the Gemma terms at the model "
                    "page, then rerun this command; no substitute checkpoint is used."
                ),
                "error_type": type(error).__name__,
                "http_status": getattr(
                    getattr(error, "response", None), "status_code", None
                ),
            }
        )
    else:
        result.update(
            {
                "status": "access_ready",
                "config_path": str(config_path),
                "tokenizer_config_path": str(tokenizer_path),
            }
        )
    _write(output, result)
    return result


def architecture_audit(model) -> dict:
    """Record Gemma's actual host attention, position, normalization, and cache contract."""
    config = model.config
    global_layers = gemma3_global_layer_ids(config)
    local_layers = tuple(
        layer for layer in range(int(config.num_hidden_layers)) if layer not in global_layers
    )
    physical = model.model.layers
    if any(bool(physical[layer].self_attn.is_sliding) for layer in global_layers):
        raise AssertionError("A configured Gemma global layer is physically sliding attention.")
    if any(not bool(physical[layer].self_attn.is_sliding) for layer in local_layers):
        raise AssertionError("A configured Gemma local layer is physically full attention.")
    return {
        "model_type": config.model_type,
        "layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "query_heads": int(config.num_attention_heads),
        "native_kv_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "layer_types": list(config.layer_types),
        "global_attention_layers": list(global_layers),
        "local_attention_layers": list(local_layers),
        "sliding_window_tokens": int(config.sliding_window),
        "sliding_window_pattern": int(config.sliding_window_pattern),
        "max_position_embeddings": int(config.max_position_embeddings),
        "global_rope_theta": float(config.rope_theta),
        "local_rope_theta": float(config.rope_local_base_freq),
        "qk_normalization": "per-head RMSNorm before RoPE",
        "normalization_order": "pre-attention and pre-MLP RMSNorm with post norms",
        "cache": "HybridCache: sliding storage for local layers, full storage for global layers",
        "mask_semantics": "causal; local layers additionally enforce the native sliding window",
    }


@torch.no_grad()
def validate_loaded_model(
    model,
    input_ids: torch.Tensor,
    reference_ids: torch.Tensor,
    long_prompt_ids: torch.Tensor,
    *,
    native_limit: int,
    direct_limit: int,
    encoding_block_tokens: int,
) -> tuple[object, dict]:
    """Gate exact disabled parity and native global-layer PRA behavior."""
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    mask = torch.ones_like(ids)
    audit = architecture_audit(model)
    global_layers = tuple(audit["global_attention_layers"])
    pra_layers = global_layers[-2:]
    local_types = {
        layer: type(model.model.layers[layer].self_attn)
        for layer in audit["local_attention_layers"]
    }

    _sync(device)
    started = time.perf_counter()
    baseline = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=True)
    baseline_generation = model.generate(
        ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=True
    )
    _sync(device)
    baseline_seconds = time.perf_counter() - started

    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=pra_layers,
            model_max_context_tokens=native_limit,
            max_prompt_direct_tokens=direct_limit,
            encoding_block_tokens=encoding_block_tokens,
            routing_chunk_tokens=32,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="mean",
            gists_per_chunk=1,
            max_materialized_memory_tokens=min(128, native_limit - direct_limit),
            context_safety_reserve_tokens=4,
            top_k_references=1,
            top_k_chunks_per_reference=4,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
    )
    handle.set_memory_enabled(False)
    _sync(device)
    started = time.perf_counter()
    wrapped = model(ids, attention_mask=mask, output_hidden_states=True, use_cache=True)
    wrapped_generation = model.generate(
        ids, attention_mask=mask, max_new_tokens=3, do_sample=False, use_cache=True
    )
    _sync(device)
    wrapped_seconds = time.perf_counter() - started

    baseline_cache = baseline.past_key_values.to_legacy_cache()
    wrapped_cache = wrapped.past_key_values.to_legacy_cache()
    parity = {
        "logits_exact": torch.equal(baseline.logits, wrapped.logits),
        "hidden_states_exact": all(
            torch.equal(left, right)
            for left, right in zip(baseline.hidden_states, wrapped.hidden_states)
        ),
        "greedy_generation_exact": torch.equal(
            baseline_generation, wrapped_generation
        ),
        "cache_length_exact": baseline.past_key_values.get_seq_length()
        == wrapped.past_key_values.get_seq_length(),
        "cache_tensors_exact": len(baseline_cache) == len(wrapped_cache)
        and all(
            torch.equal(left, right)
            for left_layer, right_layer in zip(baseline_cache, wrapped_cache)
            for left, right in zip(left_layer, right_layer)
        ),
        "local_layer_classes_exact": all(
            type(model.model.layers[layer].self_attn) is expected
            for layer, expected in local_types.items()
        ),
    }
    if not all(parity.values()):
        raise AssertionError(f"Gemma disabled-PRA parity failed: {parity}")

    entry = handle.add_reference(
        "validation://explicit-reference", reference_ids.cpu(), text="validation"
    )
    expected_heads = int(model.config.num_key_value_heads)
    expected_dim = int(model.config.head_dim)
    layer_chunks = {
        layer: entry.layer_memory[layer].chunks for layer in pra_layers
    }
    all_chunks = [chunk for chunks in layer_chunks.values() for chunk in chunks]
    native_kv = all(
        chunk.token_kv.k.ndim == 4
        and int(chunk.token_kv.k.shape[1]) == expected_heads
        and int(chunk.token_kv.k.shape[-1]) == expected_dim
        and chunk.token_kv.k.device.type == "cpu"
        and chunk.token_kv.position_state == "post_position"
        for chunk in all_chunks
    )
    positions_exact = all(
        torch.equal(
            chunk.token_kv.position_ids.cpu(),
            torch.arange(chunk.logical_start, chunk.logical_end).unsqueeze(0),
        )
        for chunk in all_chunks
    )
    if not native_kv or not positions_exact:
        raise AssertionError("Gemma reference K/V violated native MQA/RoPE/residency state.")

    routing_layer = pra_layers[-1]
    routing_chunk = layer_chunks[routing_layer][0]
    source = [[
        SelectedChunk(
            entry=entry,
            chunk=routing_chunk,
            reference_score=1.0,
            chunk_score=1.0,
            layer_id=routing_layer,
            reference_rank=1,
            rank_within_reference=1,
            metadata={"selection_source": "causal_validation"},
        )
    ]]
    fixed = handle.map_chunk_identities_to_layers(source, pra_layers)
    handle.configure_memory_layers(set(pra_layers), fixed_selections=fixed)
    causal_ids = ids[:, : min(int(ids.shape[1]), 8)].clone()
    changed_ids = causal_ids.clone()
    changed_ids[:, -1] = (changed_ids[:, -1] + 1) % int(model.config.vocab_size)
    causal_mask = torch.ones_like(causal_ids)
    positions = torch.arange(causal_ids.shape[1], device=device).unsqueeze(0)
    expected = model(
        causal_ids, attention_mask=causal_mask, position_ids=positions, use_cache=False
    ).logits
    changed = model(
        changed_ids, attention_mask=causal_mask, position_ids=positions, use_cache=False
    ).logits
    causal_prefix_exact = torch.equal(expected[:, :-1], changed[:, :-1])
    if not causal_prefix_exact:
        raise AssertionError("Gemma PRA violated the causal prefix invariant.")

    prepared = handle.prepare_long_prompt(long_prompt_ids.cpu())
    head_entries = [row for row in handle.cache.all_entries() if row.uri == "#__head"]
    handle.set_memory_enabled(True)
    active = model(
        prepared.input_ids.to(device),
        attention_mask=prepared.attention_mask.to(device),
        position_ids=prepared.position_ids.to(device),
        use_cache=False,
    )
    if not torch.isfinite(active.logits).all() or handle.native_limit_violations:
        raise AssertionError("Gemma enabled path is non-finite or exceeded its native limit.")

    first = routing_chunk.token_kv
    return handle, {
        "architecture_audit": audit,
        "disabled_parity": parity,
        "pra_placement": {
            "routing_layer": routing_layer,
            "consumption_layers": list(pra_layers),
            "local_layers_wrapped": [],
            "policy": "last two native global-attention layers",
        },
        "native_reference": {
            "detail_k_shape": _shape(first.k),
            "detail_v_shape": _shape(first.v),
            "position_state": first.position_state,
            "positions_exact": positions_exact,
            "residency": first.k.device.type,
            "permanently_expanded_to_query_heads": int(first.k.shape[1])
            == int(model.config.num_attention_heads),
        },
        "long_prompt": {
            "logical_tokens": int(long_prompt_ids.shape[1]),
            "direct_tokens": int(prepared.input_ids.shape[1]),
            "head_tokens": int(prepared.head_tokens),
            "head_reference_count": len(head_entries),
            "max_native_operation_tokens": int(handle.max_native_operation_tokens),
            "native_limit_violations": int(handle.native_limit_violations),
        },
        "enabled_path": {
            "finite_logits": True,
            "causal_prefix_exact": causal_prefix_exact,
        },
        "timing_seconds": {
            "baseline": baseline_seconds,
            "wrapped_disabled": wrapped_seconds,
        },
    }


def run_parity(args) -> dict:
    """Load the official checkpoint and execute all pre-router gates."""
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.tokenizer_revision
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
    input_ids = tokenizer(
        "PRA disabled parity must preserve this continuation.", return_tensors="pt"
    ).input_ids[:, :32]
    reference_ids = tokenizer(
        "Lisbon is the capital of Portugal. " * 24,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, :192]
    long_prompt_ids = tokenizer(
        "Earlier context becomes an implicit head reference. " * 80,
        return_tensors="pt",
    ).input_ids[:, :384]
    _, validation = validate_loaded_model(
        model,
        input_ids,
        reference_ids,
        long_prompt_ids,
        native_limit=args.native_limit,
        direct_limit=args.direct_limit,
        encoding_block_tokens=args.encoding_block_tokens,
    )
    result = {
        "runtime": runtime_metadata(),
        "status": "passed",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "attention_backend": "eager",
        "runtime_dtype": str(dtype),
        "device": str(device),
        "base_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        **validation,
    }
    _write(args.output_dir / "parity_native_kv.json", result)
    return result


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def run_suite(args) -> dict:
    """Run matched features, five router seeds, product demo, and causal probe."""
    access = access_preflight(
        args.model_id, args.model_revision, args.output_dir / "access_status.json"
    )
    if access["status"] != "access_ready":
        return access
    parity = run_parity(args)
    globals_ = parity["architecture_audit"]["global_attention_layers"]
    routing_layer = int(globals_[-1])
    consumption_layers = tuple(int(layer) for layer in globals_[-2:])
    feature_dir = args.output_dir / "features"
    router_dir = ROOT / "artifacts" / "pra_hf" / "routers" / "gemma3-1b-qasper-d128"
    router_json = args.output_dir / "router_five_seed.json"
    demo_json = args.output_dir / "product_demo.json"
    oracle_dir = args.output_dir / "oracle"
    python = sys.executable
    _run([
        python, "-m", "experiments.paper2_hf.routing.precompute_router_features",
        "--model-id", args.model_id,
        "--model-revision", args.model_revision,
        "--tokenizer-revision", args.tokenizer_revision,
        "--train-examples", str(args.train_examples),
        "--validation-examples", str(args.validation_examples),
        "--test-examples", str(args.test_examples),
        "--routing-layer", str(routing_layer),
        "--device", args.device,
        "--output-dir", str(feature_dir),
    ])
    _run([
        python, "-m", "experiments.paper2_hf.productize_router",
        "--feature-dir", str(feature_dir),
        "--output-router", str(router_dir),
        "--output-json", str(router_json),
        "--base-model", args.model_id,
        "--base-model-revision", args.model_revision,
        "--model-family", "gemma3",
        "--routing-layer", str(routing_layer),
        "--device", args.device,
    ])
    _run([
        python, "-m", "experiments.paper2_hf.run_product_demo",
        "--model", args.model_id,
        "--revision", args.model_revision,
        "--router", str(router_dir),
        "--output", str(demo_json),
        "--routing-layer", str(routing_layer),
        "--consumption-layers", ",".join(str(layer) for layer in consumption_layers),
        "--device", args.device,
    ])
    _run([
        python, "-m", "experiments.paper2_hf.qa.run_oracle_memory_use",
        "--model-id", args.model_id,
        "--model-revision", args.model_revision,
        "--checkpoint", str(router_dir),
        "--conditions", "no_memory,learned_router,oracle_last_1,direct_text_oracle",
        "--learned-depth", "2",
        "--output-dir", str(oracle_dir),
        "--device", args.device,
    ])
    result = {
        "runtime": runtime_metadata(),
        "status": "completed",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "seeds": list(SEEDS),
        "routing_layer": routing_layer,
        "consumption_layers": list(consumption_layers),
        "parity": str((args.output_dir / "parity_native_kv.json").relative_to(ROOT)),
        "features": str(feature_dir.relative_to(ROOT)),
        "router": str(router_dir.relative_to(ROOT)),
        "router_metrics": str(router_json.relative_to(ROOT)),
        "product_demo": str(demo_json.relative_to(ROOT)),
        "oracle": str(oracle_dir.relative_to(ROOT)),
    }
    _write(args.output_dir / "suite_status.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--native-limit", type=int, default=512)
    parser.add_argument("--direct-limit", type=int, default=128)
    parser.add_argument("--encoding-block-tokens", type=int, default=128)
    parser.add_argument("--train-examples", type=int, default=24)
    parser.add_argument("--validation-examples", type=int, default=8)
    parser.add_argument("--test-examples", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "gemma3_1b",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.direct_limit + 4 >= args.native_limit:
        parser.error("direct-limit plus the four-token reserve must fit native-limit")
    return args


if __name__ == "__main__":
    options = parse_args()
    if options.preflight_only:
        outcome = access_preflight(
            options.model_id,
            options.model_revision,
            options.output_dir / "access_status.json",
        )
    elif options.parity_only:
        access = access_preflight(
            options.model_id,
            options.model_revision,
            options.output_dir / "access_status.json",
        )
        outcome = run_parity(options) if access["status"] == "access_ready" else access
    else:
        outcome = run_suite(options)
    print(json.dumps(outcome, indent=2, sort_keys=True))
    raise SystemExit(
        0 if outcome["status"] in {"access_ready", "passed", "completed"} else 2
    )
