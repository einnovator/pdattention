"""Confirm hidden-state routing with native post-RoPE K/V end to end."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_hf.common.artifacts import runtime_metadata, write_artifacts
from experiments.paper2_hf.qa.run_smoke import (
    answer_metrics,
    evidence_token_spans,
    hotpot_example,
    qasper_example,
    run_condition,
)
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


def _selected_fraction(handle, uri: str, selected_count: int) -> float:
    entry = handle.cache.get(uri)
    if entry is None:
        return 0.0
    layer = next(iter(handle.adapters))
    candidate_count = len(entry.layer_memory[layer].chunks)
    return selected_count / max(candidate_count, 1)


def _last_layer_diagnostics(result: dict) -> dict:
    diagnostics = result.get("diagnostics", {})
    layer = max(diagnostics, key=int)
    return diagnostics[layer]


def _qa_confirmation(handle, tokenizer, example: dict, device, new_tokens: int) -> dict:
    result = run_condition(handle, tokenizer, example, "pra", device, new_tokens)
    uri = f"benchmark://{example['dataset']}/{example['id']}"
    diagnostics = _last_layer_diagnostics(result)
    result.update(
        {
            "dataset": example["dataset"],
            "id": example["id"],
            "answer": example["answer"],
            "routing_representation": ATTENTION_INPUT_HIDDEN_STATE,
            "attention_payload": "native_post_rope_k_and_native_v",
            "selected_fraction": _selected_fraction(
                handle, uri, len(result["selected_spans"])
            ),
            "materialized_tokens": diagnostics["memory_tokens_materialized"],
            "native_limit_violations": handle.native_limit_violations,
        }
    )
    return result


@torch.no_grad()
def _implicit_head_confirmation(handle, tokenizer, device, new_tokens: int) -> dict:
    handle.cache.clear()
    handle.set_memory_enabled(False)
    fact = "The verification word is cobalt."
    filler = "The archive contains routine administrative notes with no verification word."
    text = " ".join(
        [fact, *([filler] * 36), "Question: What is the verification word? Answer with one word:"]
    )
    full_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    evidence_spans = evidence_token_spans(tokenizer, text, [fact])
    prepared = handle.prepare_long_prompt(full_ids)
    handle.set_memory_enabled(True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    handle.model(
        input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        position_ids=prepared.position_ids,
        use_cache=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_seconds = time.perf_counter() - started

    adapter = next(iter(handle.adapters.values()))
    selected = [hit for row in adapter.last_selected_chunks for hit in row]
    selected_spans = [(hit.logical_start, hit.logical_end) for hit in selected]
    covered = sum(
        any(
            max(start, selected_start) < min(end, selected_end)
            for selected_start, selected_end in selected_spans
        )
        for start, end in evidence_spans
    )
    diagnostics = handle.diagnostics_by_layer()
    layer_diagnostics = diagnostics[max(diagnostics)]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = handle.model.generate(
        input_ids=prepared.input_ids,
        attention_mask=prepared.attention_mask,
        position_ids=prepared.position_ids,
        max_new_tokens=new_tokens,
        do_sample=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - started
    continuation = output[0, prepared.input_ids.shape[1] :]
    prediction = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    entry = handle.cache.get("#__head")
    layer = next(iter(handle.adapters))
    candidate_count = len(entry.layer_memory[layer].chunks) if entry is not None else 0

    return {
        "dataset": "implicit_head",
        "id": "synthetic-cobalt",
        "answer": "cobalt",
        "prediction": prediction,
        **answer_metrics(prediction, "cobalt"),
        "logical_prompt_tokens": int(full_ids.shape[1]),
        "head_tokens": prepared.head_tokens,
        "direct_tail_tokens": int(prepared.input_ids.shape[1]),
        "tail_position_start": int(prepared.position_ids[0, 0]),
        "tail_position_end": int(prepared.position_ids[0, -1]),
        "routing_representation": ATTENTION_INPUT_HIDDEN_STATE,
        "attention_payload": "native_post_rope_k_and_native_v",
        "selected_spans": selected_spans,
        "evidence_spans": evidence_spans,
        "routing_recall": covered / max(len(evidence_spans), 1),
        "selected_fraction": len(selected) / max(candidate_count, 1),
        "materialized_tokens": layer_diagnostics["memory_tokens_materialized"],
        "prefill_seconds": prefill_seconds,
        "generation_seconds": generation_seconds,
        "diagnostics": diagnostics,
        "native_limit_violations": handle.native_limit_violations,
    }


def run(args) -> dict:
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
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(-1,),
            model_max_context_tokens=256,
            max_prompt_direct_tokens=160,
            encoding_block_tokens=64,
            routing_chunk_tokens=32,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            max_materialized_memory_tokens=96,
            top_k_references=1,
            top_k_chunks_per_reference=3,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            kv_cache_non_blocking=device.type == "cuda",
        ),
    )
    examples = [
        hotpot_example(args.cache_dir),
        qasper_example(args.cache_dir / "qasper"),
    ]
    rows = [
        _qa_confirmation(handle, tokenizer, example, device, args.new_tokens)
        for example in examples
    ]
    rows.append(
        _implicit_head_confirmation(handle, tokenizer, device, args.new_tokens)
    )
    return {
        "runtime": runtime_metadata(),
        "protocol": (
            "canonical attention-input hidden-state routing with native post-RoPE K/V; "
            "compact end-to-end confirmation"
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "routing_representation": ATTENTION_INPUT_HIDDEN_STATE,
        "attention_payload": "native_post_rope_k_and_native_v",
        "rows": rows,
        "max_native_operation_tokens": handle.max_native_operation_tokens,
        "native_limit_violations": handle.native_limit_violations,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf" / "qa",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    artifact = run(arguments)
    json_path, csv_path = write_artifacts(
        artifact,
        arguments.output_dir,
        "qwen3_0_6b_hidden_postrope_confirmation",
    )
    print(json_path)
    print(csv_path)
