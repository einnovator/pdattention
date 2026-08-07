"""Compare legacy singleton and row-isolated batched PRA prompt forwards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data.collators import PRACollator
from data.schemas import QuestionSample, ReferenceSample
from data.tokenizer import PRATokenizer
from pra_torch.cache_services import build_cache_from_metadata
from pra_torch.config import PRAConfig
from pra_torch.memory import PRABatchedMemoryCache
from pra_torch.model import TinyPRAModel
from common.train import resolve_device


def _samples(batch_size: int) -> list[QuestionSample]:
    return [
        QuestionSample(
            id=f"benchmark-{row}",
            question=f"Find row {row} in <REF_1>",
            answer=f"value-{row}",
            references=[
                ReferenceSample(
                    id=1,
                    uri=f"bench://{row}",
                    summary=f"row {row} summary",
                    metadata={"text": (f"row {row} private memory ") * (row % 4 + 1)},
                )
            ],
            target_reference_ids=[1],
        )
        for row in range(batch_size)
    ]


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_prompt(call, *, repeats: int, device: str) -> tuple[float, int | None]:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(repeats):
        _synchronize(device)
        start = time.perf_counter()
        call()
        _synchronize(device)
        durations.append(time.perf_counter() - start)
    peak = torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None
    return mean(durations), peak


def benchmark(batch_size: int, *, repeats: int, device: str) -> dict:
    """Benchmark prompt execution after row-local reference caches are complete."""
    samples = _samples(batch_size)
    corpus = [
        text
        for sample in samples
        for text in (
            sample.question,
            sample.answer,
            sample.references[0].summary or "",
            str(sample.references[0].metadata["text"]),
        )
    ]
    tokenizer = PRATokenizer(corpus)
    batch = PRACollator(tokenizer, max_seq_len=64)(samples)
    input_ids = batch["input_ids"].to(device)
    cfg = PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.0,
        top_k_references=1,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0,
        memory_bucket_count=2,
        device=device,
    )
    model = TinyPRAModel(cfg).to(device).eval()
    row_caches = [
        build_cache_from_metadata(
            model,
            tokenizer,
            [metadata],
            device,
            attach_to_model=False,
        )
        for metadata in batch["metadata"]
    ]
    batch_cache = PRABatchedMemoryCache(row_caches)

    def singleton_prompt() -> None:
        for row_index, row_cache in enumerate(row_caches):
            model.set_pra_cache(row_cache)
            model(input_ids[row_index : row_index + 1])

    def batched_prompt() -> None:
        model.set_pra_cache(batch_cache)
        model(input_ids)

    with torch.no_grad():
        singleton_prompt()
        batched_prompt()
        singleton_seconds, singleton_peak = _time_prompt(
            singleton_prompt, repeats=repeats, device=device
        )
        batched_seconds, batched_peak = _time_prompt(
            batched_prompt, repeats=repeats, device=device
        )
        batched_prompt()

    diagnostics = model.pra_diagnostics_by_layer()
    padding = [
        values.get("memory_padding_fraction", 0.0)
        for values in diagnostics.values()
    ]
    tokens = int(batch["attention_mask"].sum().item())
    return {
        "batch_size": batch_size,
        "tokens": tokens,
        "legacy_prompt_forward_calls": batch_size,
        "batched_prompt_forward_calls": 1,
        "legacy_prompt_seconds": singleton_seconds,
        "batched_prompt_seconds": batched_seconds,
        "speedup": singleton_seconds / max(batched_seconds, 1e-12),
        "legacy_examples_per_second": batch_size / singleton_seconds,
        "batched_examples_per_second": batch_size / batched_seconds,
        "legacy_tokens_per_second": tokens / singleton_seconds,
        "batched_tokens_per_second": tokens / batched_seconds,
        "legacy_peak_cuda_bytes": singleton_peak,
        "batched_peak_cuda_bytes": batched_peak,
        "memory_padding_fraction": mean(padding) if padding else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    results = [
        benchmark(batch_size, repeats=max(args.repeats, 1), device=device)
        for batch_size in args.batch_sizes
    ]
    if args.as_json:
        print(json.dumps({"device": device, "results": results}, indent=2))
        return
    print(f"device={device}")
    print("B  calls(old/new)  prompt ms(old/new)  speedup  examples/s(new)  padding")
    for result in results:
        print(
            f"{result['batch_size']:>2}  "
            f"{result['legacy_prompt_forward_calls']}/{result['batched_prompt_forward_calls']:<11}  "
            f"{result['legacy_prompt_seconds'] * 1000:>7.2f}/"
            f"{result['batched_prompt_seconds'] * 1000:<7.2f}  "
            f"{result['speedup']:>6.2f}x  "
            f"{result['batched_examples_per_second']:>14.2f}  "
            f"{result['memory_padding_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
