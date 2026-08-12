"""Validate and run the canonical PRA-HF protocol on Meta Llama 3.2-1B.

The official checkpoint is access-gated.  The preflight always writes a status
artifact, including enough environment and revision detail to distinguish an
external access block from a failed PRA integration gate.
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
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra
from pra_torch.memory import SelectedChunk


MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
TOKENIZER_REVISION = MODEL_REVISION
SEEDS = (11, 23, 37, 53, 71)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.shape]


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def access_preflight(model_id: str, revision: str, output: Path) -> dict:
    """Resolve the exact revision and verify authenticated access to its config."""
    metadata = model_info(model_id, revision=revision)
    result = {
        "runtime": runtime_metadata(),
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": metadata.sha,
        "tokenizer_revision": revision,
        "gated": metadata.gated,
        "license": next(
            (tag.split(":", 1)[1] for tag in metadata.tags if tag.startswith("license:")),
            None,
        ),
        "architecture": (metadata.config or {}).get("architectures", [None])[0],
        "parameter_count_from_hub": (
            int(metadata.safetensors.total) if metadata.safetensors else None
        ),
        "access_requirement": (
            "Accept the Meta Llama 3.2 community license for this repository and "
            "authenticate the local Hugging Face client."
        ),
        "status": "checking",
    }
    try:
        config_path = hf_hub_download(model_id, "config.json", revision=revision)
        tokenizer_path = hf_hub_download(
            model_id, "tokenizer_config.json", revision=revision
        )
    except (GatedRepoError, HfHubHTTPError) as error:
        result.update(
            {
                "status": "blocked_external_access",
                "blocker": (
                    "The official Meta repository requires accepted Llama 3.2 access "
                    "and an authenticated Hugging Face token. Run `hf auth login` after "
                    "the account has been granted access, then rerun this command."
                ),
                "error_type": type(error).__name__,
                "http_status": getattr(getattr(error, "response", None), "status_code", None),
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
    _write_status(output, result)
    return result


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
    """Prove disabled parity, native GQA/RoPE K/V, and bounded references.

    Baseline values are captured before injection on the same frozen model. The
    disabled adapter delegates to the original attention object, so exact tensor
    equality is the required gate rather than a tolerance-based comparison.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    _sync(device)
    baseline_started = time.perf_counter()
    baseline = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=True,
    )
    baseline_generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,
    )
    _sync(device)
    baseline_seconds = time.perf_counter() - baseline_started

    layer_count = int(model.config.num_hidden_layers)
    layers = tuple(range(max(0, layer_count - 8), layer_count))
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=layers,
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
    wrapped_started = time.perf_counter()
    wrapped = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=True,
    )
    wrapped_generation = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
        do_sample=False,
        use_cache=True,
    )
    _sync(device)
    wrapped_seconds = time.perf_counter() - wrapped_started

    logits_exact = torch.equal(baseline.logits, wrapped.logits)
    hidden_exact = all(
        torch.equal(left, right)
        for left, right in zip(baseline.hidden_states, wrapped.hidden_states)
    )
    generation_exact = torch.equal(baseline_generation, wrapped_generation)
    baseline_cache = baseline.past_key_values
    wrapped_cache = wrapped.past_key_values
    baseline_cache_rows = baseline_cache.to_legacy_cache()
    wrapped_cache_rows = wrapped_cache.to_legacy_cache()
    cache_length_exact = (
        baseline_cache.get_seq_length() == wrapped_cache.get_seq_length()
    )
    cache_tensors_exact = len(baseline_cache_rows) == len(wrapped_cache_rows) and all(
        torch.equal(expected, actual)
        for expected_layer, actual_layer in zip(baseline_cache_rows, wrapped_cache_rows)
        for expected, actual in zip(expected_layer, actual_layer)
    )
    if not all(
        (
            logits_exact,
            hidden_exact,
            generation_exact,
            cache_length_exact,
            cache_tensors_exact,
        )
    ):
        raise AssertionError("Meta Llama disabled-PRA exact parity gate failed.")

    entry = handle.add_reference(
        "validation://explicit-reference", reference_ids.cpu(), text="tokenized validation"
    )
    routing_layer = layers[-1]
    chunks = entry.layer_memory[routing_layer].chunks
    first_kv = chunks[0].token_kv
    expected_kv_heads = int(model.config.num_key_value_heads)
    expected_head_dim = int(
        getattr(
            model.config,
            "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        )
    )
    layer_chunks = [
        chunk
        for layer in layers
        for chunk in entry.layer_memory[layer].chunks
    ]
    native_kv_shape = all(
        chunk.token_kv.k.ndim == 4
        and int(chunk.token_kv.k.shape[1]) == expected_kv_heads
        and int(chunk.token_kv.k.shape[-1]) == expected_head_dim
        and chunk.token_kv.k.device.type == "cpu"
        and chunk.token_kv.position_state == "post_position"
        for chunk in layer_chunks
    )
    if not native_kv_shape:
        raise AssertionError("Reference K/V is not native-head, post-RoPE, CPU-resident state.")
    positions_exact = all(
        torch.equal(
            chunk.token_kv.position_ids.cpu(),
            torch.arange(chunk.logical_start, chunk.logical_end).unsqueeze(0),
        )
        for chunk in layer_chunks
    )
    if not positions_exact:
        raise AssertionError("Reference K/V positions do not match logical source positions.")

    fixed_source = [[
        SelectedChunk(
            entry=entry,
            chunk=chunks[0],
            reference_score=1.0,
            chunk_score=1.0,
            layer_id=routing_layer,
            reference_rank=1,
            rank_within_reference=1,
            metadata={"selection_source": "causal_mask_validation"},
        )
    ]]
    fixed = handle.map_chunk_identities_to_layers(fixed_source, layers)
    handle.configure_memory_layers(set(layers), fixed_selections=fixed)
    causal_ids = input_ids[:, : min(int(input_ids.shape[1]), 8)].clone()
    changed_ids = causal_ids.clone()
    changed_ids[:, -1] = (changed_ids[:, -1] + 1) % int(model.config.vocab_size)
    causal_mask = torch.ones_like(causal_ids)
    causal_positions = torch.arange(causal_ids.shape[1], device=device).unsqueeze(0)
    causal_expected = model(
        input_ids=causal_ids,
        attention_mask=causal_mask,
        position_ids=causal_positions,
        use_cache=False,
    ).logits
    causal_changed = model(
        input_ids=changed_ids,
        attention_mask=causal_mask,
        position_ids=causal_positions,
        use_cache=False,
    ).logits
    causal_prefix_exact = torch.equal(causal_expected[:, :-1], causal_changed[:, :-1])
    if not causal_prefix_exact:
        raise AssertionError("Fixed PRA memory violated the native causal-mask prefix invariant.")

    prepared = handle.prepare_long_prompt(long_prompt_ids.cpu())
    head_entries = [row for row in handle.cache.all_entries() if row.uri == "#__head"]
    if prepared.head_tokens <= 0 or len(head_entries) != 1:
        raise AssertionError("Long-prompt rollover did not create exactly one #__head reference.")

    handle.set_memory_enabled(True)
    active_ids = prepared.input_ids.to(device)
    active_mask = prepared.attention_mask.to(device)
    active_positions = prepared.position_ids.to(device)
    active = model(
        input_ids=active_ids,
        attention_mask=active_mask,
        position_ids=active_positions,
        use_cache=False,
    )
    if not torch.isfinite(active.logits).all():
        raise AssertionError("Enabled-PRA logits contain non-finite values.")
    if handle.native_limit_violations:
        raise AssertionError("A validation operation exceeded the declared native limit.")

    report = {
        "disabled_parity": {
            "logits_exact": logits_exact,
            "hidden_states_exact": hidden_exact,
            "greedy_generation_exact": generation_exact,
            "cache_length_exact": cache_length_exact,
            "cache_tensors_exact": cache_tensors_exact,
        },
        "model_contract": {
            "layers": layer_count,
            "hidden_size": int(model.config.hidden_size),
            "query_heads": int(model.config.num_attention_heads),
            "native_kv_heads": expected_kv_heads,
            "head_dim": expected_head_dim,
            "dtype": str(baseline.logits.dtype),
            "device": str(device),
            "cache_sequence_length": int(wrapped_cache.get_seq_length()),
            "cache_shapes": [
                {"key": _shape(key), "value": _shape(value)}
                for key, value in wrapped_cache_rows
            ],
        },
        "native_reference": {
            "detail_k_shape": _shape(first_kv.k),
            "detail_v_shape": _shape(first_kv.v),
            "position_state": first_kv.position_state,
            "positions_exact": positions_exact,
            "residency": first_kv.k.device.type,
            "dtype": str(first_kv.k.dtype),
            "permanently_expanded_to_query_heads": int(first_kv.k.shape[1])
            == int(model.config.num_attention_heads),
            "explicit_reference_chunks": len(chunks),
        },
        "long_prompt": {
            "input_tokens": int(long_prompt_ids.shape[1]),
            "direct_tokens": int(prepared.input_ids.shape[1]),
            "head_tokens": int(prepared.head_tokens),
            "head_reference_count": len(head_entries),
        },
        "enabled_path": {
            "finite_logits": True,
            "causal_prefix_exact": causal_prefix_exact,
            "active_layers": list(layers),
            "max_native_operation_tokens": int(handle.max_native_operation_tokens),
            "native_limit_violations": int(handle.native_limit_violations),
        },
        "timing_seconds": {
            "baseline": baseline_seconds,
            "wrapped_disabled": wrapped_seconds,
        },
    }
    return handle, report


def run_parity(args) -> dict:
    """Load the official checkpoint once and execute all pre-router correctness gates."""
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
        "PRA disabled parity must preserve this continuation.",
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids[:, :32]
    reference_ids = tokenizer(
        "Lisbon is the capital of Portugal. " * 24,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[:, :192]
    long_prompt_ids = tokenizer(
        "Earlier context becomes an implicit head reference. " * 80,
        return_tensors="pt",
        add_special_tokens=True,
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
        "dtype": str(dtype),
        "device": str(device),
        "base_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        **validation,
    }
    _write_status(args.output_dir / "parity_native_kv.json", result)
    return result


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def run_suite(args) -> dict:
    """Run the matched feature, five-seed, product, and causal-use protocol."""
    status_path = args.output_dir / "access_status.json"
    access = access_preflight(args.model_id, args.model_revision, status_path)
    if access["status"] != "access_ready":
        return access
    parity = run_parity(args)
    layer = int(parity["model_contract"]["layers"]) - 1
    feature_dir = args.output_dir / "features"
    router_dir = ROOT / "artifacts" / "pra_hf" / "routers" / "llama32-1b-qasper-d128"
    router_json = args.output_dir / "router_five_seed.json"
    demo_json = args.output_dir / "product_demo.json"
    oracle_dir = args.output_dir / "oracle"
    python = sys.executable
    _run(
        [
            python,
            "-m",
            "experiments.paper2_hf.routing.precompute_router_features",
            "--model-id",
            args.model_id,
            "--model-revision",
            args.model_revision,
            "--tokenizer-revision",
            args.tokenizer_revision,
            "--train-examples",
            str(args.train_examples),
            "--validation-examples",
            str(args.validation_examples),
            "--test-examples",
            str(args.test_examples),
            "--routing-layer",
            str(layer),
            "--device",
            args.device,
            "--output-dir",
            str(feature_dir),
        ]
    )
    _run(
        [
            python,
            "-m",
            "experiments.paper2_hf.productize_router",
            "--feature-dir",
            str(feature_dir),
            "--output-router",
            str(router_dir),
            "--output-json",
            str(router_json),
            "--base-model",
            args.model_id,
            "--base-model-revision",
            args.model_revision,
            "--model-family",
            "llama",
            "--routing-layer",
            str(layer),
            "--device",
            args.device,
        ]
    )
    _run(
        [
            python,
            "-m",
            "experiments.paper2_hf.run_product_demo",
            "--model",
            args.model_id,
            "--revision",
            args.model_revision,
            "--router",
            str(router_dir),
            "--output",
            str(demo_json),
            "--device",
            args.device,
        ]
    )
    _run(
        [
            python,
            "-m",
            "experiments.paper2_hf.qa.run_oracle_memory_use",
            "--model-id",
            args.model_id,
            "--model-revision",
            args.model_revision,
            "--checkpoint",
            str(router_dir),
            "--conditions",
            "no_memory,learned_router,oracle_last_8,direct_text_oracle",
            "--learned-depth",
            "8",
            "--output-dir",
            str(oracle_dir),
            "--device",
            args.device,
        ]
    )
    result = {
        "runtime": runtime_metadata(),
        "status": "completed",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "seeds": list(SEEDS),
        "parity": str((args.output_dir / "parity_native_kv.json").relative_to(ROOT)),
        "features": str(feature_dir.relative_to(ROOT)),
        "router": str(router_dir.relative_to(ROOT)),
        "router_metrics": str(router_json.relative_to(ROOT)),
        "product_demo": str(demo_json.relative_to(ROOT)),
        "oracle": str(oracle_dir.relative_to(ROOT)),
    }
    _write_status(args.output_dir / "suite_status.json", result)
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
        default=(
            ROOT
            / "docs"
            / "papers"
            / "shared"
            / "results"
            / "paper2_hf"
            / "llama32_1b"
        ),
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
