"""Evaluate implicit historical prompt-head memory on trained answer-code probes."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pra_torch.memory import PRASimpleMemoryCache  # noqa: E402
from pra_torch.model import convert_sa_model_to_pra  # noqa: E402
from pra_torch.prompt import IMPLICIT_PROMPT_HEAD_URI  # noqa: E402
from run_native_kv_benchmark import (  # noqa: E402
    DATASET_DEFAULTS,
    SEEDS,
    _native_config,
    _prepare_synthetic,
    _set_seed,
    train_full_context_sa,
)


VERSION = "long_prompt_head_v1"
CONDITIONS = (
    "dense_full",
    "direct_truncation",
    "head_routed",
    "head_oracle",
    "head_shuffled",
    "head_independent",
)


def _json_safe(value):
    """Replace undefined floating-point metrics with strict-JSON null values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _repeat(values: list[int], count: int, offset: int = 0) -> list[int]:
    if count <= 0:
        return []
    if not values:
        raise ValueError("Long-prompt construction requires distractor tokens.")
    cycle = itertools.cycle(values[offset % len(values) :] + values[: offset % len(values)])
    return list(itertools.islice(cycle, count))


def _position_start(position: str, available: int, target_length: int, chunk_size: int) -> int:
    maximum = max(available - target_length, 0)
    if position == "early":
        return 0
    if position == "middle":
        return maximum // 2
    if position == "late":
        return maximum
    if position == "boundary":
        return min(max(chunk_size - 1, 0), maximum)
    raise ValueError(position)


def _make_prompt(sample, tokenizer, *, total_tokens: int, direct_budget: int, position: str, chunk_size: int):
    question_ids = list(tokenizer.encode(sample.question))
    if not question_ids or len(question_ids) > direct_budget:
        raise ValueError("Question must fit in the fixed direct budget.")
    target_uri = sample.target_reference_uris[0]
    target_reference = next(reference for reference in sample.references if reference.uri == target_uri)
    target_ids = list(tokenizer.encode(str(target_reference.metadata.get("text", ""))))
    distractors = [
        token
        for reference in sample.references
        if reference.uri != target_uri
        for token in tokenizer.encode(str(reference.metadata.get("text", "")))
    ]
    if not distractors:
        source_ids = list(tokenizer.encode(str(sample.metadata["row"]["source_text"])))
        distractors = source_ids or [1]

    head_length = max(total_tokens - direct_budget, 0)
    target_region_length = head_length or (direct_budget - len(question_ids))
    if len(target_ids) > target_region_length:
        target_ids = target_ids[:target_region_length]
    target_start = _position_start(position, target_region_length, len(target_ids), chunk_size)
    target_region = (
        _repeat(distractors, target_start, offset=17)
        + target_ids
        + _repeat(
            distractors,
            target_region_length - target_start - len(target_ids),
            offset=31,
        )
    )
    direct_filler_length = direct_budget - len(question_ids)
    if head_length:
        head = target_region
        direct = _repeat(distractors, direct_filler_length, offset=47) + question_ids
        span = (target_start, target_start + len(target_ids))
    else:
        head = []
        direct = target_region + question_ids
        span = (target_start, target_start + len(target_ids))
    ids = head + direct
    assert len(ids) == total_tokens
    assert len(head) + len(direct) == len(ids)
    return ids, head, direct, span


def _loss_row(logits: torch.Tensor, answer_id: int) -> tuple[float, float]:
    target = torch.tensor([answer_id], dtype=torch.long, device=logits.device)
    final = logits[:, -1, :]
    loss = float(F.cross_entropy(final, target).detach().cpu())
    accuracy = float(int(final.argmax(dim=-1).item() == answer_id))
    return loss, accuracy


def _filter_entry(entry, *, target_span: tuple[int, int], mode: str) -> None:
    start, end = target_span
    for memory in entry.layer_memory.values():
        target_chunks = [
            chunk
            for chunk in memory.chunks
            if max(chunk.token_start, start) < min(chunk.token_end, end)
        ]
        target_chunk_ids = {id(chunk) for chunk in target_chunks}
        wrong_chunks = [chunk for chunk in memory.chunks if id(chunk) not in target_chunk_ids]
        if mode == "oracle":
            memory.chunks[:] = target_chunks
        elif mode == "shuffled":
            memory.chunks[:] = wrong_chunks[: max(len(target_chunks), 1)]


def _evaluate_pra(
    model,
    tokenizer,
    *,
    head_ids: list[int],
    direct_ids: list[int],
    target_span: tuple[int, int],
    answer_id: int,
    condition: str,
    device: str,
) -> dict:
    model.clear_pra_cache()
    if head_ids:
        historical = condition != "head_independent"
        entry = model.encode_reference_tokens_to_cache(
            IMPLICIT_PROMPT_HEAD_URI,
            head_ids,
            tokenizer,
            device,
            metadata={"implicit": True, "display_name": "#__head", "source": "prompt"},
            max_chunks=None,
            use_configured_max_chunks=False,
            max_chunk_tokens=model.cfg.max_seq_len,
            historical_encoding=historical,
        )
        if condition == "head_oracle":
            _filter_entry(entry, target_span=target_span, mode="oracle")
        elif condition == "head_shuffled":
            _filter_entry(entry, target_span=target_span, mode="shuffled")
        cache = PRASimpleMemoryCache()
        cache.put(entry)
        model.set_pra_cache(cache)

    original_topk = model.cfg.top_k_chunks_per_reference
    if condition in {"head_oracle", "head_shuffled"}:
        model.cfg.top_k_chunks_per_reference = 1_000_000
    input_ids = torch.tensor([direct_ids], dtype=torch.long, device=device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(
            input_ids,
            position_offset=len(head_ids),
            use_pra_memory=bool(head_ids),
        )
    _sync(device)
    latency = time.perf_counter() - start
    model.cfg.top_k_chunks_per_reference = original_topk
    loss, accuracy = _loss_row(logits, answer_id)
    selections = model.selected_chunks_by_layer()
    selected = [hit for rows in selections.values() for row in rows for hit in row]
    target_recalled = not head_ids or any(
        max(hit.token_start, target_span[0]) < min(hit.token_end, target_span[1])
        for hit in selected
    )
    diagnostics = list(model.pra_diagnostics_by_layer().values())
    retrieved = statistics.fmean(
        float(row.get("retrieved_token_kv", 0.0)) for row in diagnostics
    ) if diagnostics else 0.0
    return {
        "condition": condition,
        "loss": loss,
        "accuracy": accuracy,
        "target_chunk_recall": float(target_recalled),
        "retrieved_kv_tokens": retrieved,
        "routing_ms": 1_000.0 * sum(
            float(row.get("routing_duration_seconds", 0.0)) for row in diagnostics
        ),
        "materialization_ms": 1_000.0 * sum(
            float(row.get("materialization_duration_seconds", 0.0)) for row in diagnostics
        ),
        "request_forward_ms": 1_000.0 * latency,
        "peak_cuda_allocated": (
            float(torch.cuda.max_memory_allocated(device))
            if str(device).startswith("cuda") else 0.0
        ),
        "selected_chunk_count": len({hit.chunk_id for hit in selected}),
    }


def _evaluate_case(
    source,
    model,
    tokenizer,
    sample,
    *,
    total_tokens,
    direct_budget,
    position,
    device,
    pra_conditions=CONDITIONS[2:],
):
    ids, head, direct, target_span = _make_prompt(
        sample,
        tokenizer,
        total_tokens=total_tokens,
        direct_budget=direct_budget,
        position=position,
        chunk_size=model.cfg.fixed_chunk_tokens,
    )
    answer_ids = tokenizer.encode(sample.answer.strip())
    answer_id = int(answer_ids[0])
    full = torch.tensor([ids], dtype=torch.long, device=device)
    tail = torch.tensor([direct], dtype=torch.long, device=device)
    with torch.no_grad():
        dense_logits = source(full, use_pra_memory=False)
        truncation_logits = source(tail, use_pra_memory=False)
    dense_loss, dense_accuracy = _loss_row(dense_logits, answer_id)
    truncation_loss, truncation_accuracy = _loss_row(truncation_logits, answer_id)
    rows = [
        {
            "condition": "dense_full",
            "loss": dense_loss,
            "accuracy": dense_accuracy,
            "target_chunk_recall": 1.0,
            "retrieved_kv_tokens": float(max(total_tokens - direct_budget, 0)),
            "routing_ms": 0.0,
            "materialization_ms": 0.0,
            "request_forward_ms": 0.0,
            "peak_cuda_allocated": 0.0,
            "selected_chunk_count": 0,
        },
        {
            "condition": "direct_truncation",
            "loss": truncation_loss,
            "accuracy": truncation_accuracy,
            "target_chunk_recall": float(not head),
            "retrieved_kv_tokens": 0.0,
            "routing_ms": 0.0,
            "materialization_ms": 0.0,
            "request_forward_ms": 0.0,
            "peak_cuda_allocated": 0.0,
            "selected_chunk_count": 0,
        },
    ]
    for condition in pra_conditions:
        rows.append(
            _evaluate_pra(
                model,
                tokenizer,
                head_ids=head,
                direct_ids=direct,
                target_span=target_span,
                answer_id=answer_id,
                condition=condition,
                device=device,
            )
        )
    denominator = truncation_loss - dense_loss
    for row in rows:
        row.update(
            {
                "total_prompt_tokens": total_tokens,
                "direct_tail_tokens": len(direct),
                "implicit_head_tokens": len(head),
                "target_position": position,
                "token_conservation": len(head) + len(direct) == len(ids),
                "active_kv_fraction": (
                    len(direct) + row["retrieved_kv_tokens"]
                ) / max(total_tokens, 1),
                "recovered_context_benefit": (
                    (truncation_loss - row["loss"]) / denominator
                    if abs(denominator) > 1e-9 else math.nan
                ),
            }
        )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    metrics = (
        "loss",
        "accuracy",
        "target_chunk_recall",
        "retrieved_kv_tokens",
        "active_kv_fraction",
        "recovered_context_benefit",
        "routing_ms",
        "materialization_ms",
        "request_forward_ms",
        "peak_cuda_allocated",
        "selected_chunk_count",
    )
    groups = {}
    for row in rows:
        key = (row["condition"], row["total_prompt_tokens"])
        groups.setdefault(key, []).append(row)
    output = []
    for (condition, total), members in sorted(groups.items()):
        item = {
            "condition": condition,
            "total_prompt_tokens": total,
            "seeds": len({row["seed"] for row in members}),
            "examples": len(members),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in members if math.isfinite(float(row[metric]))]
            item[f"{metric}_mean"] = statistics.fmean(values) if values else math.nan
            item[f"{metric}_stddev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    loss_by_condition_length = {
        (row["condition"], row["total_prompt_tokens"]): row["loss_mean"]
        for row in output
    }
    for row in output:
        total = row["total_prompt_tokens"]
        dense = loss_by_condition_length[("dense_full", total)]
        truncation = loss_by_condition_length[("direct_truncation", total)]
        denominator = truncation - dense
        row["recovered_context_benefit_from_mean_loss"] = (
            (truncation - row["loss_mean"]) / denominator
            if abs(denominator) > 1e-9
            else math.nan
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, aggregate: list[dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    styles = {
        "dense_full": ("#2f8567", "-"),
        "direct_truncation": ("#777777", "--"),
        "head_routed": ("#3366a8", "-"),
        "head_oracle": ("#d05a3a", "-"),
        "head_shuffled": ("#aa4d8f", ":"),
        "head_independent": ("#b4842f", "--"),
    }
    for condition in CONDITIONS:
        rows = sorted(
            (row for row in aggregate if row["condition"] == condition),
            key=lambda row: row["total_prompt_tokens"],
        )
        color, line = styles[condition]
        x = [row["total_prompt_tokens"] for row in rows]
        axes[0, 0].plot(x, [row["accuracy_mean"] for row in rows], marker="o", color=color, linestyle=line, label=condition)
        axes[0, 1].plot(x, [row["target_chunk_recall_mean"] for row in rows], marker="o", color=color, linestyle=line, label=condition)
        axes[1, 0].plot(x, [row["active_kv_fraction_mean"] for row in rows], marker="o", color=color, linestyle=line, label=condition)
        axes[1, 1].plot(x, [row["routing_ms_mean"] for row in rows], marker="o", color=color, linestyle=line, label=condition)
    axes[0, 0].set_ylabel("Answer-token accuracy")
    axes[0, 1].set_ylabel("Target-chunk recall")
    axes[1, 0].set_ylabel("Active K/V fraction")
    axes[1, 1].set_ylabel("Routing latency (ms/example)")
    for axis in axes.flat:
        axis.set_xlabel("Total prompt tokens")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure.suptitle("Implicit #__head on trained fixed-target probes (five seeds)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    settings = dict(DATASET_DEFAULTS["synthetic"])
    tokenizer, training_module, modules = _prepare_synthetic(settings)
    evaluation = modules[64].dataset
    result_root = REPO / "out" / "pra_long_prompt_head"
    published = REPO / "docs" / "papers" / "shared"
    result_root.mkdir(parents=True, exist_ok=True)
    (published / "results").mkdir(parents=True, exist_ok=True)
    (published / "figures").mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in args.seeds:
        output = result_root / f"seed-{seed}.json"
        if output.exists() and not args.force:
            payload = json.loads(output.read_text(encoding="utf-8"))
            if (
                payload.get("version") == VERSION
                and payload.get("lengths") == args.lengths
                and payload.get("examples") == args.examples
            ):
                all_rows.extend(payload["results"])
                print(f"reuse {output.relative_to(REPO)}", flush=True)
                continue
        _set_seed(seed)
        source, _ = train_full_context_sa(
            seed=seed,
            tokenizer=tokenizer,
            datamodule=training_module,
            settings=settings,
            run_dir=REPO / "out" / "native_kv_benchmarks" / "synthetic" / f"seed-{seed}",
            device=device,
            force=False,
        )
        cfg = _native_config(
            source,
            device,
            {
                "max_prompt_direct_tokens": args.direct_budget,
                "prompt_overflow_mode": "implicit_reference",
                "prompt_position_mode": "historical",
                "chunking_mode": "fixed",
                "fixed_chunk_tokens": args.chunk_size,
                "fixed_chunk_overlap_tokens": args.overlap,
                "max_prompt_gists": None,
                "top_k_references": 1,
                "top_k_chunks_per_reference": args.top_k,
                "collect_detailed_timing": True,
                "routing_backend": "tensorized",
            },
        )
        model = convert_sa_model_to_pra(source, cfg).to(device).eval()
        seed_rows = []
        for example_index in range(min(args.examples, len(evaluation))):
            sample = evaluation[example_index]
            for length in args.lengths:
                for position in args.positions:
                    rows = _evaluate_case(
                        source,
                        model,
                        tokenizer,
                        sample,
                        total_tokens=length,
                        direct_budget=args.direct_budget,
                        position=position,
                        device=device,
                    )
                    seed_rows.extend(
                        {
                            "seed": seed,
                            "example_id": str(sample.id),
                            "top_k": args.top_k,
                            "chunk_size": args.chunk_size,
                            "overlap": args.overlap,
                            **row,
                        }
                        for row in rows
                    )
            print(f"done seed-{seed}/example-{example_index}", flush=True)
        output.write_text(
            json.dumps(
                {
                    "version": VERSION,
                    "lengths": args.lengths,
                    "examples": args.examples,
                    "results": seed_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        all_rows.extend(seed_rows)
        del model, source
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not all(row["token_conservation"] for row in all_rows):
        raise AssertionError("Long-prompt token conservation failed.")
    aggregate = _aggregate(all_rows)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    payload = {
        "manifest": {
            "version": VERSION,
            "git_sha": git_sha,
            "device": device,
            "device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else "cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "dataset": "synthetic_native_kv_fixed_target",
            "seeds": args.seeds,
            "examples_per_seed": args.examples,
            "total_prompt_lengths": args.lengths,
            "direct_budget": args.direct_budget,
            "target_positions": args.positions,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "top_k_chunks": args.top_k,
            "head_encoding": "historical encode-once then native-KV slicing",
        },
        "aggregate": aggregate,
        "raw": all_rows,
    }
    path = published / "results" / "pra_long_prompt_head.json"
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_csv(published / "results" / "pra_long_prompt_head.csv", aggregate)
    _plot(published / "figures" / "pra_long_prompt_head.pdf", aggregate)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--lengths", nargs="+", type=int, default=[24, 48, 96, 192])
    parser.add_argument("--positions", nargs="+", choices=["early", "middle", "late", "boundary"], default=["early", "middle", "late", "boundary"])
    parser.add_argument("--direct-budget", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
