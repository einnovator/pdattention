"""Capture missing validation query-token states for the request/reply study.

Paper 2.7 already stores held-out HotpotQA/QASPER query states and both splits
for 2Wiki/MuSiQue.  This runner fills only the 16 validation HotpotQA/QASPER
states, using one complete contextual prompt pass and the frozen Qwen revision.
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

from experiments.paper2_7_query_graph.helpers import MODEL_ID, MODEL_REVISION  # noqa: E402
from experiments.paper2_hf.routing.run_query_strategies import (  # noqa: E402
    _prompt_with_question_span,
    load_split_examples,
)
from pra_torch.hf import (  # noqa: E402
    ATTENTION_INPUT_HIDDEN_STATE,
    PRAHFConfig,
    QUERY_QUESTION_EXPONENTIAL,
    aggregate_query_states,
    inject_pra,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_router_features(path: Path) -> Path:
    candidates = (
        path,
        ROOT.parent / "pdattention" / path.relative_to(ROOT),
        ROOT.parent / "pdattention-iter-gist" / path.relative_to(ROOT),
    )
    return next((candidate for candidate in candidates if candidate.exists()), path)


@torch.no_grad()
def _capture(handle, tokenizer, example: dict, device: torch.device) -> dict:
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
    start, end = map(int, question_span)
    return {
        "partition": "validation",
        "dataset": example["dataset"],
        "example_id": example["id"],
        "question": example["question"],
        "query_hidden_states": hidden,
        "query_pre_query": pre_query,
        "question_span": (start, end),
        "prompt_input_ids": input_ids,
        "question_input_ids": input_ids[start:end].clone(),
        "prompt_tokens": int(input_ids.numel()),
        "question_tokens": end - start,
        "full_query_forward_count": 1,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    source_path = _resolve_router_features(args.source_features)
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_by_id = {str(row["example_id"]): row for row in source}
    examples = load_split_examples(args.cache_dir, 8, 0, args.seed)
    if {str(row["id"]) for row in examples} != set(source_by_id):
        raise ValueError("Validation examples do not match the frozen router-feature cohort.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(27,),
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

    rows = []
    maximum_error = 0.0
    for index, example in enumerate(examples, start=1):
        row = _capture(handle, tokenizer, example, device)
        start, end = row["question_span"]
        pooled = aggregate_query_states(
            row["query_hidden_states"].float().unsqueeze(0),
            QUERY_QUESTION_EXPONENTIAL,
            half_life=2.0,
            token_spans=[(start, end)],
        )[0]
        expected = source_by_id[row["example_id"]]["queries"]["question_exp_h2.0"].float()
        error = float((pooled - expected).abs().max())
        row["pooled_query_max_abs_error"] = error
        maximum_error = max(maximum_error, error)
        rows.append(row)
        print(
            f"[request/reply query {index}/{len(examples)}] "
            f"{row['dataset']} {row['example_id']} error={error:.3g}",
            flush=True,
        )
    if maximum_error > args.maximum_error:
        raise AssertionError(
            f"Contextual-versus-frozen pooled query drift {maximum_error:.6g} exceeds "
            f"{args.maximum_error:.6g}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, args.output)
    manifest = {
        "schema_version": "1.0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backbone_frozen": True,
        "routing_layer": 27,
        "partition": "validation",
        "examples": len(rows),
        "dataset_counts": {
            dataset: sum(row["dataset"] == dataset for row in rows)
            for dataset in ("hotpotqa", "qasper")
        },
        "maximum_pooled_query_drift": maximum_error,
        "pooled_query_use": "diagnostic_only; frozen router vector remains the global facet",
        "feature_file": str(args.output),
        "feature_sha256": _sha256(args.output),
        "source_router_features": str(source_path),
        "device": str(device),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    result = ROOT / "docs/papers/shared/results/paper3_5_adaptive_pra/root_callback"
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--maximum-error", type=float, default=0.1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--source-features",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter/router_features_validation.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tmp/request_reply_callback/query_states_validation.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=result / "query_states_validation_manifest.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
