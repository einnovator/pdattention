"""Live HF-backed AirLLM correctness and native-PRA smoke experiment."""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from airllm import AutoModel

from pra_hf.airllm_adapter import wrap_airllm_hf_model
from pra_hf.config import PRAConfig


EVIDENCE = "The project access code is CYAN-ORBIT-47."
QUESTION = "Question: What is the project access code? Answer with only the code.\nAnswer:"


def _write(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _measure_generate(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids.to("cuda")
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        disable_compile=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = generated[:, input_ids.shape[1] :]
    return {
        "input_tokens": int(input_ids.shape[1]),
        "output_token_ids": new_ids[0].detach().cpu().tolist(),
        "output_text": tokenizer.decode(new_ids[0], skip_special_tokens=True),
        "completion_seconds": elapsed,
        "tokens_per_second": int(new_ids.shape[1]) / max(elapsed, 1e-9),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prefetching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    previous = None
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema_version": "paper6.6-airllm-cuda-native-v1",
        "evidence_tier": "LIVE_ENGINE_MECHANISM_SMOKE",
        "model": args.model,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "prefetching": args.prefetching,
        "status": "RUNNING",
        "rows": list(previous.get("rows", [])) if previous else [],
    }
    _write(args.output, report)
    try:
        initialized = time.perf_counter()
        air = AutoModel.from_pretrained(
            args.model,
            device="cuda:0",
            dtype=torch.float16,
            max_seq_len=2048,
            layer_shards_saving_path=str(args.shard_dir),
            prefetching=args.prefetching,
            profiling_mode=True,
        )
        report["initialization_seconds"] = time.perf_counter() - initialized
        tokenizer = air.tokenizer
        selected_prompt = f"Evidence: {EVIDENCE}\n{QUESTION}"
        full_prompt = ("Archive note without an access code. " * 48) + "\n" + selected_prompt
        prior_by_condition = {row["condition"]: row for row in report["rows"]}
        baseline_full = prior_by_condition.get("airllm_full_context_e0")
        if baseline_full is None:
            baseline_full = _measure_generate(
                air.model, tokenizer, full_prompt, args.max_new_tokens
            )
            baseline_full["condition"] = "airllm_full_context_e0"
            report["rows"].append(baseline_full)
            _write(args.output, report)
        baseline_selected = prior_by_condition.get("airllm_selected_text_e0")
        if baseline_selected is None:
            baseline_selected = _measure_generate(
                air.model, tokenizer, selected_prompt, args.max_new_tokens
            )
            baseline_selected["condition"] = "airllm_selected_text_e0"
            report["rows"].append(baseline_selected)
            _write(args.output, report)

        pra = wrap_airllm_hf_model(
            air,
            pra_config=PRAConfig(
                consumption_layers=(-4, -3, -2, -1),
                address_layers=(-4, -3, -2, -1),
                detail_kv_layers=(-4, -3, -2, -1),
                routing_layer=-1,
                selected_fraction=1.0,
                chunk_tokens=16,
                max_materialized_tokens=32,
                reference_device="cpu",
            ),
        )
        wrapped_disabled = prior_by_condition.get("airllm_pra_injected_disabled")
        if wrapped_disabled is None:
            wrapped_disabled = _measure_generate(
                pra.model, tokenizer, selected_prompt, args.max_new_tokens
            )
            wrapped_disabled["condition"] = "airllm_pra_injected_disabled"
            report["rows"].append(wrapped_disabled)
        report["disabled_exact_sequence"] = (
            baseline_selected["output_token_ids"] == wrapped_disabled["output_token_ids"]
        )
        _write(args.output, report)

        reference_started = time.perf_counter()
        handle = pra.add_reference("memory://paper6.6/access", text=EVIDENCE)
        report["reference_encode_seconds"] = time.perf_counter() - reference_started
        report["reference"] = {
            "tokens": handle.tokens,
            "chunks": handle.chunks,
        }
        torch.cuda.reset_peak_memory_stats()
        native = pra.generate(
            QUESTION,
            max_new_tokens=args.max_new_tokens,
            return_details=True,
            do_sample=False,
            use_cache=True,
        )
        native_row = {
            "condition": "airllm_native_pra",
            "input_tokens": native.prompt_tokens,
            "output_tokens": native.generated_tokens,
            "output_text": native.text,
            "completion_seconds": native.latency_seconds,
            "tokens_per_second": native.generated_tokens / max(native.latency_seconds, 1e-9),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "requested_kv_tokens": native.stats.get("requested_kv_tokens"),
            "materialized_kv_tokens": native.stats.get("materialized_kv_tokens"),
            "consumption_layers": native.stats.get("consumption_layers"),
        }
        report["rows"].append(native_row)
        report["status"] = "COMPLETE"
    except Exception as error:
        report["status"] = "BLOCKED"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    _write(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
