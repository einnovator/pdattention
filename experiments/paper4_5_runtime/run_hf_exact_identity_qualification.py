"""Run selector-frozen E0/E2 qualification on the eager HF PRA path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.engine_serving.matched_e0_e2_contract import (  # noqa: E402
    SCHEMA_VERSION,
    benchmark_metrics,
    benchmark_row,
    regime_schedule,
    validate_payload,
)
from experiments.engine_serving.matched_qa import load_matched_examples  # noqa: E402
from experiments.paper6_2_mlx.run_answer_quality_pressure import _metrics  # noqa: E402
from pra_hf import PRAConfig, PRAForCausalLM  # noqa: E402
from pra_hf.native_geometry import FrozenNativeAnchor, FrozenNativeSelection  # noqa: E402


def _bounded_text(tokenizer, text: str, limit: int) -> tuple[str, list[int]]:
    tokens = list(tokenizer.encode(text, add_special_tokens=False))[:limit]
    return tokenizer.decode(tokens, skip_special_tokens=True), tokens


def _generate(pra: PRAForCausalLM, prompt: str, *, plan, max_new_tokens: int):
    import torch

    if pra.device.type == "cuda":
        torch.cuda.synchronize(pra.device)
    started = time.perf_counter()
    if plan is None:
        pra.disable()
        encoded = pra.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded.input_ids.to(pra.device)
        attention_mask = encoded.attention_mask.to(pra.device)
        output = pra.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=torch.arange(
                input_ids.shape[1], device=pra.device
            ).unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated = output[:, input_ids.shape[1] :]
        result = SimpleNamespace(
            text=pra.tokenizer.decode(generated[0], skip_special_tokens=True),
            generated_tokens=int(generated.shape[1]),
        )
    else:
        pra.enable()
        result = pra.generate_with_native_plan(
            prompt,
            plan,
            max_new_tokens=max_new_tokens,
            return_details=True,
            do_sample=False,
        )
    if pra.device.type == "cuda":
        torch.cuda.synchronize(pra.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("qasper", "hotpotqa", "2wikimultihopqa"), required=True
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quantization", choices=("none", "bnb-8bit"), default="none")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--max-source-tokens", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers

    kwargs: dict[str, object] = {
        "revision": args.revision,
        "low_cpu_mem_usage": True,
    }
    if args.quantization == "bnb-8bit":
        kwargs.update(
            quantization_config=transformers.BitsAndBytesConfig(load_in_8bit=True),
            device_map="cuda:0",
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    config = PRAConfig(
        routing_layer=-1,
        consumption_layers=tuple(range(-args.layers, 0)),
        address_layers=(-1,),
        detail_kv_layers=tuple(range(-args.layers, 0)),
        chunk_tokens=32,
        selected_fraction=1.0,
        max_materialized_tokens=args.max_source_tokens,
        materialization_profile="paper8_full_record_diagnostic",
        reference_device="gpu" if torch.cuda.is_available() else "cpu",
    )
    pra = PRAForCausalLM.from_pretrained(args.model, pra_config=config, **kwargs)
    if args.quantization == "none":
        pra.model.to("cuda" if torch.cuda.is_available() else "cpu")
    pra.model.eval()

    manifest, examples = load_matched_examples(args.manifest, args.dataset, args.cache_dir)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    rows = []
    for example in examples:
        prepared_at = time.perf_counter()
        source_text, source_tokens = _bounded_text(
            pra.tokenizer, example.selected_source, args.max_source_tokens
        )
        # Keep the text/token boundary stable so E0 and E2 consume precisely
        # the same source token sequence rather than a tokenizer merge artifact.
        source_text = source_text.rstrip() + "\n\n"
        source_tokens = list(
            pra.tokenizer.encode(source_text, add_special_tokens=False)
        )
        candidate_tokens = list(
            pra.tokenizer.encode(example.candidate_source, add_special_tokens=False)
        )
        text_preparation_ms = (time.perf_counter() - prepared_at) * 1000.0

        pra.clear_references()
        encoded_at = time.perf_counter()
        handle = pra.add_reference("memory://frozen-selection", text=source_text)
        if pra.device.type == "cuda":
            torch.cuda.synchronize(pra.device)
        encode_ms = (time.perf_counter() - encoded_at) * 1000.0
        entry = pra._handle.cache.get(handle.uri)
        if entry is None:
            raise RuntimeError("Native reference encoding did not create a cache entry.")
        chunk = entry.layer_memory[pra.routing_layer].chunks[0]
        frozen = FrozenNativeSelection(
            (
                FrozenNativeAnchor(
                    handle.uri,
                    chunk.chunk_id,
                    0,
                    handle.tokens,
                ),
            )
        )
        plan = pra.plan_native_materialization(frozen, full_selected_record=True)
        native_bytes = sum(
            hit.chunk.token_kv.k.numel() * hit.chunk.token_kv.k.element_size()
            + hit.chunk.token_kv.v.numel() * hit.chunk.token_kv.v.element_size()
            for selected in plan.selections_by_layer.values()
            for hit in selected
            if hit.chunk.token_kv is not None
        )

        for request in regime_schedule(
            example.question,
            warm_repeats=args.warm_repeats,
            multi_query_count=args.multi_query_count,
            concurrency=args.concurrency,
        ):
            query = request.query.text
            query_tokens = list(pra.tokenizer.encode(query, add_special_tokens=False))
            combined_tokens = list(
                pra.tokenizer.encode(source_text + query, add_special_tokens=False)
            )
            if combined_tokens != source_tokens + query_tokens:
                raise RuntimeError("Tokenizer boundary changed the frozen E0/E2 source identity.")
            for condition in ("e0_selected_text", "e2_native_kv"):
                native = condition == "e2_native_kv"
                prompt = query if native else source_text + query
                result, elapsed_ms = _generate(
                    pra,
                    prompt,
                    plan=plan if native else None,
                    max_new_tokens=args.max_new_tokens,
                )
                exact, f1 = _metrics(result.text, example.answer)
                reused = request.regime != "cold_one_shot"
                rows.append(
                    benchmark_row(
                        condition=condition,
                        selection=example.selection,
                        request=request,
                        output=result.text.strip(),
                        metrics=benchmark_metrics(
                            exact_match=exact,
                            token_f1=f1,
                            gold_answer_logprob=None,
                            evidence_recall=example.evidence_recall,
                            candidate_tokens=len(candidate_tokens),
                            selected_source_tokens=len(source_tokens),
                            visible_prompt_tokens=(
                                len(query_tokens) if native else len(source_tokens) + len(query_tokens)
                            ),
                            selected_native_kv_tokens=len(source_tokens) if native else 0,
                            active_detail_bytes=native_bytes if native else 0,
                            retained_detail_bytes=native_bytes if native else 0,
                            text_preparation_ms=text_preparation_ms,
                            kv_encode_ms=encode_ms if native else 0.0,
                            index_construction_ms=0.0,
                            time_to_usable_context_ms=text_preparation_ms + (encode_ms if native else 0.0),
                            ttft_ms=None,
                            itl_ms=None,
                            total_latency_ms=elapsed_ms,
                            generated_tokens=result.generated_tokens,
                            ordinary_prefix_cache_hit_tokens=None,
                            pra_hot_hit=native and reused,
                            pra_warm_hit=False,
                            bytes_read=native_bytes if native else 0,
                            bytes_promoted=0,
                            bytes_avoided=native_bytes if native and reused else 0,
                            duplicate_physical_kv_avoided_bytes=native_bytes if native and reused else 0,
                        ),
                        extra={
                            "dataset": example.dataset,
                            "seed": example.seed,
                            "gold_answer": example.answer,
                            "execution_source_sha256": example.selected_source_sha256,
                            "concurrency_execution": "serialized_hf_reference",
                        },
                    )
                )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "paper4_5_exact_identity_hf_e0_e2_v1",
        "evidence_tier": "NATURAL_QA_MATCHED_SELECTION",
        "engine": "huggingface_eager",
        "engine_version": str(transformers.__version__),
        "model_id": args.model,
        "model_revision": args.revision,
        "quantization": args.quantization,
        "dataset": args.dataset,
        "cohort": manifest["cohort"],
        "selection_policy": manifest["selection_policy"],
        "max_source_tokens": args.max_source_tokens,
        "warm_repeats": args.warm_repeats,
        "multi_query_count": args.multi_query_count,
        "concurrency": args.concurrency,
        "rows": rows,
    }
    validate_payload(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
