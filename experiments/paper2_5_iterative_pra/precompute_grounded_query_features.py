"""Capture controlled long-prompt states for query-support contamination tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra
from pra_torch.hf.query import token_span_from_offsets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_budget_text(tokenizer, text: str, target_tokens: int) -> str:
    """Repeat then truncate source text to an approximate tokenizer budget."""
    if target_tokens == 0:
        return ""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("Stale source text tokenized to an empty sequence.")
    repeated = (ids * math.ceil(target_tokens / len(ids)))[:target_tokens]
    return tokenizer.decode(repeated, skip_special_tokens=True).strip()


def _render_control_prompt(tokenizer, question: str, stale_text: str, max_tokens: int):
    """Render old history plus one explicit latest user message and locate spans."""
    old_content = (
        "Archived unrelated task context.\n"
        f"Stale memory phrase: {stale_text or '[none]'}\n"
        "Tool history: lookup completed for the archived task."
    )
    latest_content = f"Answer briefly and directly.\nQuestion: {question.strip()}"
    messages = [
        {"role": "user", "content": old_content},
        {"role": "assistant", "content": "The archived task is complete."},
        {"role": "user", "content": latest_content},
    ]
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        rendered = f"User: {old_content}\nAssistant: The archived task is complete.\nUser: {latest_content}\nAssistant:"
    latest_start = rendered.rfind(latest_content)
    question_start = rendered.rfind(question.strip())
    stale_start = rendered.find(stale_text) if stale_text else -1
    if latest_start < 0 or question_start < 0:
        raise ValueError("Rendered prompt lost the latest-message markers.")
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
    latest_span = token_span_from_offsets(
        offsets, latest_start, latest_start + len(latest_content)
    )
    question_span = token_span_from_offsets(
        offsets, question_start, question_start + len(question.strip())
    )
    stale_span = (
        token_span_from_offsets(offsets, stale_start, stale_start + len(stale_text))
        if stale_start >= 0
        else None
    )
    return encoded, latest_span, question_span, stale_span


@torch.no_grad()
def _capture(handle, encoded, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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
    return (
        captured.hidden_states[0].to("cpu", torch.float16),
        captured.pre_query[0].permute(1, 0, 2).to("cpu", torch.float16),
    )


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    source_features = torch.load(
        args.source_feature_file, map_location="cpu", weights_only=False
    )
    examples = load_split_examples(
        args.cache_dir, args.examples, args.example_offset, args.seed
    )
    if [row["example_id"] for row in source_features] != [row["id"] for row in examples]:
        raise ValueError("Controlled prompts do not match the frozen feature order.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=(args.routing_layer,),
            model_max_context_tokens=args.max_tokens,
            max_prompt_direct_tokens=args.max_tokens,
            encoding_block_tokens=args.max_tokens,
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
    for example_index, (example, feature) in enumerate(
        zip(examples, source_features), start=1
    ):
        source_ids = tokenizer(
            example["source"], return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        oracle = set(torch.nonzero(feature["parent_positive_mask"]).flatten().tolist())
        non_oracle = [
            index for index in range(len(feature["parent_spans"])) if index not in oracle
        ]
        stale_parent = max(
            non_oracle,
            key=lambda index: (float(feature["parent_lexical_scores"][index]), -index),
        )
        parent_start, parent_end = feature["parent_spans"][stale_parent]
        stale_source = tokenizer.decode(
            source_ids[parent_start:parent_end], skip_special_tokens=True
        ).strip()
        for target in args.junk_tokens:
            stale_text = _token_budget_text(tokenizer, stale_source, target)
            encoded, latest_span, question_span, stale_span = _render_control_prompt(
                tokenizer, example["question"], stale_text, args.max_tokens
            )
            hidden, pre_query = _capture(handle, encoded, device)
            ids = encoded.input_ids[0].cpu()
            rows.append(
                {
                    "dataset": example["dataset"],
                    "example_id": example["id"],
                    "question": example["question"],
                    "junk_target_tokens": target,
                    "junk_observed_tokens": (
                        stale_span[1] - stale_span[0] if stale_span is not None else 0
                    ),
                    "stale_parent": stale_parent,
                    "stale_parent_is_oracle": stale_parent in oracle,
                    "query_hidden_states": hidden,
                    "query_pre_query": pre_query,
                    "prompt_input_ids": ids,
                    "prompt_tokens": int(ids.numel()),
                    "latest_message_span": tuple(map(int, latest_span)),
                    "question_span": tuple(map(int, question_span)),
                    "stale_span": (
                        tuple(map(int, stale_span)) if stale_span is not None else None
                    ),
                    "full_query_forward_count": 1,
                    "independent_window_forward_count": 0,
                }
            )
        print(
            f"[grounded-query {example_index}/{len(examples)}] "
            f"{example['dataset']} {example['id']} stale_parent={stale_parent}",
            flush=True,
        )

    if any(row["stale_parent_is_oracle"] for row in rows):
        raise AssertionError("A stale contamination parent overlapped oracle evidence.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "grounded_query_features.pt"
    torch.save(rows, feature_path)
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "routing_layer": next(iter(handle.adapters.values())).layer_idx,
        "examples": len(examples),
        "captures": len(rows),
        "junk_target_tokens": list(args.junk_tokens),
        "prompt_form": "old_user_assistant_history_then_latest_user_question",
        "latest_message_boundary_recorded": True,
        "full_query_forward_count_per_capture": 1,
        "independent_window_forward_count": 0,
        "stale_parent_oracle_overlaps": 0,
        "feature_artifact": {
            "path": feature_path.name,
            "bytes": feature_path.stat().st_size,
            "sha256": _sha256(feature_path),
            "tracked": False,
            "regenerate_with": (
                "python -m experiments.paper2_5_iterative_pra."
                "precompute_grounded_query_features --device cuda"
            ),
        },
    }
    (args.output_dir / "grounded_query_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--routing-layer", type=int, default=-1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--example-offset", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--junk-tokens", default="0,16,32,64")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    result_root = ROOT / "docs/papers/shared/results/paper2_5_iterative_pra"
    parser.add_argument(
        "--source-feature-file",
        type=Path,
        default=result_root / "native_qk_closure/native_qk_features_test.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=result_root / "grounded_query_facets",
    )
    args = parser.parse_args()
    args.junk_tokens = tuple(map(int, args.junk_tokens.split(",")))
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
