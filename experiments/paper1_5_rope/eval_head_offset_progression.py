"""Evaluate logical offsets, overlap, and oracle selection on continuous prompt history."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from data.datasets import SyntheticNativeKVFixedTargetDataset  # noqa: E402
from data.tokenizer import PRATokenizer  # noqa: E402
from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    SEEDS,
    TIERS,
    environment_metadata,
    refresh_manifest,
    set_seed,
    write_csv,
    write_json,
)
from pra_torch.chunking import ChunkingConfig  # noqa: E402
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402
from pra_torch.prompt import IMPLICIT_PROMPT_HEAD_URI, prepare_prompt_batch_for_pra  # noqa: E402


LOGICAL_TOKENS = 192
MODEL_OPERATION_TOKENS = 32
DIRECT_TOKENS = 8
MATERIALIZATION_BUDGET = 24
ENCODING_TOKENS = 16
ROUTING_TOKENS = 8


def _load(tier: str, mode: str, seed: int, device: str):
    path = REPO / "out" / "paper1_5_rope" / tier / mode / f"seed-{seed}" / "checkpoint.pt"
    checkpoint = torch.load(path, map_location=device)
    source_cfg = PRAConfig(**{**checkpoint["cfg"], "device": device})
    source = TinyPRAModel(source_cfg).to(device)
    source.load_state_dict(checkpoint["model"])
    target_cfg = PRAConfig(
        **{
            **source_cfg.__dict__,
            "model_variant": "td_pra",
            "memory_transport": "native_kv",
            "dropout": 0.0,
            "model_max_context_tokens": MODEL_OPERATION_TOKENS,
            "max_prompt_direct_tokens": DIRECT_TOKENS,
            "prompt_overflow_mode": "implicit_reference",
            "prompt_position_mode": "historical",
            "reference_position_mode": "global",
            "encoding_context_mode": "independent",
            "encoding_chunking": ChunkingConfig(
                mode="fixed",
                chunk_tokens=ENCODING_TOKENS,
            ),
            "routing_chunking": ChunkingConfig(
                mode="fixed",
                chunk_tokens=ROUTING_TOKENS,
            ),
            "max_prompt_gists": 128,
            "max_gists_per_reference": 128,
            "top_k_references": 1,
            "top_k_chunks_per_reference": 3,
            "trigger_threshold": float("-inf"),
            "max_materialized_memory_tokens": MATERIALIZATION_BUDGET,
            "context_safety_reserve_tokens": 0,
        }
    )
    converted = convert_sa_model_to_pra(source.eval(), target_cfg).to(device).eval()
    tokenizer = PRATokenizer.from_vocab(checkpoint["stoi"])
    return source.eval(), converted, tokenizer


def _examples(max_examples: int):
    root = REPO / "out" / "native_kv_data" / "synthetic" / "split-2"
    return list(SyntheticNativeKVFixedTargetDataset(root))[:max_examples]


def _prompt_ids(sample, tokenizer) -> list[int]:
    source = tokenizer.encode(str(sample.metadata["row"]["source_text"]))
    query = tokenizer.encode(sample.question)
    filler_pattern = tokenizer.encode(" neutral filler ") or [1]
    filler_count = LOGICAL_TOKENS - len(source) - len(query)
    if filler_count < 0:
        raise ValueError("Controlled head source exceeds the representable logical range.")
    filler = (filler_pattern * math.ceil(filler_count / len(filler_pattern)))[:filler_count]
    return [*source, *filler, *query]


def _configure_stage(model: TinyPRAModel, stage: str) -> None:
    overlap = 0.50 if "overlap" in stage else 0.0
    model.cfg.reference_position_mode = "global" if stage != "reset_routed" else "local"
    model.cfg.encoding_context_mode = "overlap" if overlap else "independent"
    model.cfg.encoding_chunking = ChunkingConfig(
        mode="fixed",
        chunk_tokens=ENCODING_TOKENS,
        overlap_fraction=overlap,
    )


def _prune_to_constructed_evidence(model, cache, tokenizer) -> None:
    """Keep the first source chunk, which contains the generated target marker."""
    entry = cache.get(IMPLICIT_PROMPT_HEAD_URI)
    if entry is None:
        raise RuntimeError("Oracle control requires an implicit prompt-head entry.")
    for memory in entry.layer_memory.values():
        evidence = [chunk for chunk in memory.chunks if chunk.logical_start == 0]
        if len(evidence) != 1:
            raise RuntimeError("Expected exactly one constructed evidence chunk per layer.")
        memory.chunks = evidence
    entry.reference_gists_by_layer.clear()
    cache.put(entry)
    model.rebuild_cache_routing_gists(cache, tokenizer=tokenizer)


def _selection_metrics(model: TinyPRAModel) -> tuple[float, float]:
    recalls = []
    selected_counts = []
    for rows in model.selected_chunks_by_layer().values():
        selected = rows[0] if rows else []
        selected_counts.append(len(selected))
        recalls.append(float(any(hit.logical_start == 0 for hit in selected)))
    return (
        statistics.fmean(recalls) if recalls else 0.0,
        statistics.fmean(selected_counts) if selected_counts else 0.0,
    )


@torch.no_grad()
def evaluate_seed(
    tier: str,
    mode: str,
    seed: int,
    device: str,
    max_examples: int,
    git_sha: str,
) -> list[dict]:
    set_seed(seed)
    source, model, tokenizer = _load(tier, mode, seed, device)
    rows = []
    for sample in _examples(max_examples):
        ids = _prompt_ids(sample, tokenizer)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        answer_id = tokenizer.encode(sample.answer.strip())[0]
        answer = torch.tensor([answer_id], device=device)
        dense_prediction = source(input_ids)[0, -1]
        dense_loss = float(F.cross_entropy(dense_prediction[None], answer))

        for stage in (
            "reset_routed",
            "offset_routed",
            "offset_overlap_routed",
            "offset_overlap_oracle",
        ):
            _configure_stage(model, stage)
            prepared = prepare_prompt_batch_for_pra(model, tokenizer, input_ids)
            cache = prepared.caches[0]
            if stage.endswith("oracle"):
                _prune_to_constructed_evidence(model, cache, tokenizer)
            model.set_pra_cache(cache)
            prediction = model(
                prepared.input_ids,
                attention_mask=prepared.attention_mask,
                position_offset=prepared.position_offsets,
            )[0, -1]
            recall, selected_chunks = _selection_metrics(model)
            diagnostics = model.pra_diagnostics_by_layer()
            disabled_prediction = model(
                prepared.input_ids,
                use_pra_memory=False,
                attention_mask=prepared.attention_mask,
                position_offset=prepared.position_offsets,
            )[0, -1]
            loss = float(F.cross_entropy(prediction[None], answer))
            disabled_loss = float(F.cross_entropy(disabled_prediction[None], answer))
            entry = cache.get(IMPLICIT_PROMPT_HEAD_URI)
            max_encoding = int(entry.metadata["max_encoding_input_tokens"])
            materialized = max(
                (row.get("memory_tokens_materialized", 0.0) for row in diagnostics.values()),
                default=0.0,
            )
            denominator = disabled_loss - dense_loss
            rows.append(
                {
                    "git_sha": git_sha,
                    "seed": seed,
                    "model_tier": tier,
                    "position_mode": mode,
                    "k_storage_mode": "post_position",
                    "logical_offset_policy": "reset" if stage == "reset_routed" else "source_relative",
                    "stage": stage,
                    "example_id": sample.id,
                    "native_context": MODEL_OPERATION_TOKENS,
                    "logical_context": len(ids),
                    "position_capacity": model.cfg.max_seq_len if mode == "absolute" else None,
                    "overlap_fraction": model.cfg.encoding_chunking_config.overlap_fraction,
                    "encoding_chunk_size": ENCODING_TOKENS,
                    "routing_chunk_size": ROUTING_TOKENS,
                    "materialization_budget": MATERIALIZATION_BUDGET,
                    "maximum_native_operation": max(
                        max_encoding,
                        int(prepared.input_ids.shape[1]),
                    ),
                    "native_limit_violations": int(
                        max(max_encoding, int(prepared.input_ids.shape[1]))
                        > MODEL_OPERATION_TOKENS
                    ),
                    "dense_loss": dense_loss,
                    "disabled_loss": disabled_loss,
                    "loss": loss,
                    "accuracy": int(prediction.argmax().item() == answer_id),
                    "recovered_context_benefit": (
                        (disabled_loss - loss) / denominator
                        if abs(denominator) > 1e-12
                        else float("nan")
                    ),
                    "evidence_chunk_recall": recall,
                    "selected_chunks_mean": selected_chunks,
                    "materialized_memory_tokens": materialized,
                    "encoding_cost_ratio": float(
                        entry.metadata["encoding_input_tokens_total"]
                    )
                    / max(int(entry.metadata["unique_source_tokens"]), 1),
                }
            )
    return rows


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_tier"], row["position_mode"], row["stage"], row["seed"])].append(row)
    seed_rows = []
    for (tier, mode, stage, seed), values in sorted(grouped.items()):
        seed_rows.append(
            {
                "model_tier": tier,
                "position_mode": mode,
                "stage": stage,
                "seed": seed,
                "example_count": len(values),
                "loss_mean": statistics.fmean(row["loss"] for row in values),
                "loss_median": statistics.median(row["loss"] for row in values),
                "accuracy_mean": statistics.fmean(row["accuracy"] for row in values),
                "evidence_chunk_recall_mean": statistics.fmean(
                    row["evidence_chunk_recall"] for row in values
                ),
                "encoding_cost_ratio_mean": statistics.fmean(
                    row["encoding_cost_ratio"] for row in values
                ),
                "dense_loss_mean": statistics.fmean(row["dense_loss"] for row in values),
                "disabled_loss_mean": statistics.fmean(row["disabled_loss"] for row in values),
            }
        )
    aggregate = []
    groups = defaultdict(list)
    for row in seed_rows:
        groups[(row["model_tier"], row["position_mode"], row["stage"])].append(row)
    for (tier, mode, stage), values in sorted(groups.items()):
        aggregate.append(
            {
                "model_tier": tier,
                "position_mode": mode,
                "stage": stage,
                "seed_count": len(values),
                "loss_mean": statistics.fmean(row["loss_mean"] for row in values),
                "loss_median": statistics.median(row["loss_mean"] for row in values),
                "loss_std": statistics.pstdev(row["loss_mean"] for row in values),
                "accuracy_mean": statistics.fmean(row["accuracy_mean"] for row in values),
                "evidence_chunk_recall_mean": statistics.fmean(
                    row["evidence_chunk_recall_mean"] for row in values
                ),
                "encoding_cost_ratio_mean": statistics.fmean(
                    row["encoding_cost_ratio_mean"] for row in values
                ),
            }
        )
    return seed_rows, aggregate


def _plot(seed_rows: list[dict], path: Path) -> None:
    stages = (
        "reset_routed",
        "offset_routed",
        "offset_overlap_routed",
        "offset_overlap_oracle",
    )
    labels = ("reset", "offset", "+overlap", "+oracle")
    present_tiers = [
        tier
        for tier in ("tiny", "small")
        if any(row["model_tier"] == tier for row in seed_rows)
    ]
    figure, axes = plt.subplots(
        len(present_tiers),
        2,
        figsize=(8.2, 3.2 * len(present_tiers)),
        sharex=True,
        squeeze=False,
    )
    colors = {"absolute": "#245A8D", "rope": "#A34832"}
    for row_index, tier in enumerate(present_tiers):
        for column, metric in enumerate(("loss_mean", "evidence_chunk_recall_mean")):
            axis = axes[row_index, column]
            present_modes = [
                mode
                for mode in ("absolute", "rope")
                if any(
                    row["model_tier"] == tier and row["position_mode"] == mode
                    for row in seed_rows
                )
            ]
            for mode in present_modes:
                present_seeds = sorted(
                    {
                        row["seed"]
                        for row in seed_rows
                        if row["model_tier"] == tier and row["position_mode"] == mode
                    }
                )
                values_by_seed = {
                    seed: [
                        next(
                            row[metric]
                            for row in seed_rows
                            if row["model_tier"] == tier
                            and row["position_mode"] == mode
                            and row["seed"] == seed
                            and row["stage"] == stage
                        )
                        for stage in stages
                    ]
                    for seed in present_seeds
                }
                for values in values_by_seed.values():
                    axis.plot(range(4), values, color=colors[mode], alpha=0.25, linewidth=0.8)
                axis.plot(
                    range(4),
                    [statistics.fmean(values[index] for values in values_by_seed.values()) for index in range(4)],
                    color=colors[mode],
                    marker="o",
                    linewidth=2,
                    label=mode,
                )
            axis.set_title(f"{tier}: {'loss' if column == 0 else 'evidence recall'}")
            axis.set_xticks(range(4), labels, rotation=20, ha="right")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False)
    for row_index in range(len(present_tiers)):
        axes[row_index, 0].set_ylabel("Answer-token loss")
        axes[row_index, 1].set_ylabel("Evidence-chunk recall")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args) -> Path:
    metadata = environment_metadata()
    rows = []
    for tier in args.tiers:
        for mode in ("absolute", "rope"):
            for seed in args.seeds:
                rows.extend(
                    evaluate_seed(
                        tier,
                        mode,
                        seed,
                        args.device,
                        args.max_examples,
                        metadata["git_sha"],
                    )
                )
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    seed_rows, aggregate = _aggregate(rows)
    expectations = [
        {
            "comparison": "reset -> source-relative offset",
            "expected": "offset lowers loss",
            "reason": "restores the continuous prompt coordinate",
        },
        {
            "comparison": "offset -> overlap",
            "expected": "overlap lowers loss",
            "reason": "restores part of the missing encoding history",
        },
        {
            "comparison": "routed -> constructed-evidence oracle",
            "expected": "oracle is no worse than routed",
            "reason": "removes selection failure for the known first-chunk marker",
        },
    ]
    write_json(
        RESULTS / "head_offset_progression.json",
        {
            "metadata": metadata,
            "expectations_recorded_before_analysis": expectations,
            "rows": rows,
            "seed_rows": seed_rows,
            "aggregate": aggregate,
        },
    )
    write_csv(RESULTS / "head_offset_progression.csv", rows)
    _plot(seed_rows, RESULTS / "head_offset_progression.png")
    return refresh_manifest(metadata=metadata)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-examples", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
