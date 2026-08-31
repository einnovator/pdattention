"""Run live AirLLM/MLX selected-text E0 measurements on Apple silicon."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import psutil
from airllm import AutoModel


EVIDENCE = "The project access code is CYAN-ORBIT-47."
QUESTION = "Question: What is the project access code? Answer with only the code."


def _prompt(condition: str, distractor_tokens: int) -> str:
    selected = f"Evidence: {EVIDENCE}\n{QUESTION}\nAnswer:"
    if condition == "selected_text":
        return selected
    distractor = "Archive note: routine calibration completed without an access code. "
    return (distractor * max(1, distractor_tokens // 10)) + "\n" + selected


def _generate(model: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    encoded = model.tokenizer(
        [prompt], return_tensors="np", return_attention_mask=False, padding=False
    )["input_ids"]
    tokens = mx.array(encoded)
    started = time.perf_counter()
    first = None
    output_ids: list[int] = []
    for token in model.model_generate(tokens, temperature=0):
        if first is None:
            first = time.perf_counter()
        output_ids.append(int(token.item()))
        if len(output_ids) >= max_new_tokens:
            break
    ended = time.perf_counter()
    text = model.tokenizer.decode(output_ids)
    return {
        "input_tokens": int(tokens.shape[1]),
        "output_tokens": len(output_ids),
        "output_text": text,
        "contains_expected": "CYAN-ORBIT-47" in text.upper().replace(" ", ""),
        "ttft_seconds": None if first is None else first - started,
        "completion_seconds": ended - started,
        "tokens_per_second": len(output_ids) / max(ended - started, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--distractor-tokens", type=int, nargs="+", default=[256, 1024, 2048])
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--show-memory", action="store_true")
    args = parser.parse_args()

    process = psutil.Process()
    before_rss = process.memory_info().rss
    init_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model,
        max_seq_len=args.max_seq_len,
        layer_shards_saving_path=str(args.shard_dir) if args.shard_dir else None,
        show_memory_util=args.show_memory,
    )
    init_seconds = time.perf_counter() - init_started
    rows = []
    for distractor_tokens in args.distractor_tokens:
        for condition in ("full_context", "selected_text"):
            prompt = _prompt(condition, distractor_tokens)
            for repeat in range(args.repeats):
                row = _generate(model, prompt, args.max_new_tokens)
                row.update(
                    condition=condition,
                    distractor_tokens=distractor_tokens,
                    repeat=repeat,
                )
                rows.append(row)

    shard_root = Path(getattr(model, "checkpoint_path", ""))
    shard_files = [path for path in shard_root.rglob("*") if path.is_file()] if shard_root.exists() else []
    report = {
        "schema_version": "paper6.6-airllm-mlx-e0-v1",
        "evidence_tier": "LIVE_ENGINE_SELECTED_TEXT_E0",
        "claim_boundary": "AirLLM's separate MLX path; no native HF PRA injection.",
        "model": args.model,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "max_seq_len": args.max_seq_len,
        "initialization_seconds": init_seconds,
        "process_rss_before_bytes": before_rss,
        "process_rss_after_bytes": process.memory_info().rss,
        "airllm_max_consumed_host_mib": (
            None
            if getattr(model, "least_available", None) is None
            else model.initial_available - model.least_available
        ),
        "shard_file_count": len(shard_files),
        "shard_bytes": sum(path.stat().st_size for path in shard_files),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
