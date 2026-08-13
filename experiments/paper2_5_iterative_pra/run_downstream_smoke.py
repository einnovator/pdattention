"""Small full-native-K/V validation for one-shot and iterative PRA."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_smoke import answer_metrics, hotpot_example, qasper_example
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from pra_hf import PRAConfig, PRAForCausalLM, PRARouter


def _prompt(question: str) -> str:
    return f"Answer briefly and directly.\nQuestion: {question}"


def run(args):
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
    router = PRARouter.from_experiment_checkpoint(args.checkpoint, device=device)
    config = PRAConfig(
        routing_layer=-1,
        consumption_layers=(-1,),
        chunk_tokens=32,
        selected_fraction=None,
        top_k=args.budget,
        max_direct_context=160,
        native_operation_limit=256,
        max_materialized_tokens=96,
        context_safety_reserve_tokens=0,
        encoding_block_tokens=64,
        reference_device="cpu",
        pin_reference_memory=device.type == "cuda",
        non_blocking_transfer=device.type == "cuda",
    )
    pra = PRAForCausalLM.from_model(model, tokenizer, pra_config=config, router=router)
    examples = [hotpot_example(args.cache_dir), qasper_example(args.cache_dir / "qasper")]
    rows = []
    for example in examples:
        pra.clear_references()
        pra.add_reference(example["source"], uri=f"benchmark://{example['dataset']}/{example['id']}")
        conditions = []
        for condition in ("none", "one_shot", "iterative"):
            if condition == "none":
                pra.disable()
            else:
                pra.enable()
                config.routing_mode = condition
                config.routing_depth = 2
                per_round = max(1, math.ceil(args.budget / config.routing_depth))
                config.branch_top_k = per_round
                config.beam_size = per_round
                config.max_unique_chunks = args.budget
                config.root_anchor_alpha = 0.25
                config.frontier_mode = "direct"
                config.path_score_mode = "product"
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = pra.generate(
                _prompt(example["question"]),
                max_new_tokens=args.new_tokens,
                return_details=True,
                do_sample=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = {
                "condition": condition,
                "prediction": result.text.strip(),
                "latency_seconds": time.perf_counter() - started,
                **answer_metrics(result.text, example["answer"]),
                **{
                    key: result.stats.get(key)
                    for key in (
                        "candidate_chunks", "requested_chunks", "requested_chunk_fraction",
                        "requested_kv_tokens", "requested_kv_token_fraction",
                        "materialized_kv_tokens", "materialized_kv_token_fraction",
                        "query_encoding_seconds", "routing_seconds", "generation_seconds",
                    )
                },
                "retrieval_graphs": result.stats.get("retrieval_graphs", []),
            }
            conditions.append(row)
        rows.append({
            "dataset": example["dataset"], "example_id": example["id"],
            "question": example["question"], "answer": example["answer"],
            "conditions": conditions,
        })
    return {
        "runtime": runtime_metadata(), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "protocol": "two-example downstream integration smoke; not an efficacy estimate",
        "budget_chunks": args.budget, "examples": rows,
        "max_native_operation_tokens": pra._handle.max_native_operation_tokens,
        "native_limit_violations": pra._handle.native_limit_violations,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=12)
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / "pdattention" / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_hf/routing/learned_adapter/checkpoints/asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/papers/shared/results/paper2_5_iterative_pra")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    artifact = run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "downstream_native_kv_smoke.json"
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(output)
