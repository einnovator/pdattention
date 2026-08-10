"""Evaluate source-relative continuity and the four-cell RoPE K-storage matrix."""

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
from experiments.paper1_5_rope.instrumented_model import materialize_raw_rope_key  # noqa: E402
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402


LOGICAL_TOKENS = 128
MODEL_OPERATION_TOKENS = 32
MATERIALIZATION_BUDGET = 160


def _load_converted(tier: str, mode: str, seed: int, device: str) -> TinyPRAModel:
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
        }
    )
    return convert_sa_model_to_pra(source.eval(), target_cfg).to(device).eval()


def _segmented_kv(
    model: TinyPRAModel,
    token_ids: list[int],
    *,
    overlap_fraction: float,
    logical_offsets: bool,
) -> tuple[dict[int, torch.Tensor], int]:
    output = {layer_id: [] for layer_id in range(model.cfg.n_layers)}
    processed_tokens = 0
    for logical_start in range(0, len(token_ids), MODEL_OPERATION_TOKENS):
        logical_end = min(logical_start + MODEL_OPERATION_TOKENS, len(token_ids))
        left_context = min(
            logical_start,
            round(MODEL_OPERATION_TOKENS * overlap_fraction),
        )
        encoding_start = logical_start - left_context
        encoded_ids = token_ids[encoding_start:logical_end]
        encoded = model._encode_reference_tokens(
            encoded_ids,
            next(model.parameters()).device,
            detach=True,
            use_pra_memory=False,
            position_offset=encoding_start if logical_offsets else 0,
        )
        processed_tokens += len(encoded_ids)
        core_tokens = logical_end - logical_start
        for layer_id, kv in encoded.items():
            output[layer_id].append(
                kv.k[:, :, left_context : left_context + core_tokens, :]
            )
    return {
        layer_id: torch.cat(parts, dim=2)
        for layer_id, parts in output.items()
    }, processed_tokens


@torch.no_grad()
def _representation_rows(
    model: TinyPRAModel,
    *,
    tier: str,
    mode: str,
    seed: int,
    git_sha: str,
) -> list[dict]:
    set_seed(20_000 + seed)
    token_ids = torch.randint(1, model.cfg.vocab_size, (LOGICAL_TOKENS,)).tolist()
    dense = model._encode_reference_tokens(
        token_ids,
        next(model.parameters()).device,
        detach=True,
        use_pra_memory=False,
        position_offset=0,
    )
    rows = []
    stages = (
        ("reset", False, 0.0),
        ("offset", True, 0.0),
        ("offset_overlap_25", True, 0.25),
        ("offset_overlap_50", True, 0.50),
    )
    for stage, logical_offsets, overlap in stages:
        segmented, processed = _segmented_kv(
            model,
            token_ids,
            overlap_fraction=overlap,
            logical_offsets=logical_offsets,
        )
        for layer_id, expected in dense.items():
            actual = segmented[layer_id]
            rows.append(
                {
                    "git_sha": git_sha,
                    "seed": seed,
                    "model_tier": tier,
                    "position_mode": mode,
                    "k_storage_mode": "post_position",
                    "logical_offset_policy": "source_relative" if logical_offsets else "reset",
                    "stage": stage,
                    "layer_id": layer_id,
                    "native_context": MODEL_OPERATION_TOKENS,
                    "logical_context": LOGICAL_TOKENS,
                    "position_capacity": model.cfg.max_seq_len if mode == "absolute" else None,
                    "overlap_fraction": overlap,
                    "routing_chunk_size": MODEL_OPERATION_TOKENS,
                    "materialization_budget": MATERIALIZATION_BUDGET,
                    "maximum_native_operation": MODEL_OPERATION_TOKENS,
                    "dense_counterfactual_tokens": LOGICAL_TOKENS,
                    "processed_tokens": processed,
                    "encoding_cost_ratio": processed / LOGICAL_TOKENS,
                    "native_k_rmse": float((expected.k - actual).square().mean().sqrt()),
                    "native_k_cosine": float(
                        F.cosine_similarity(
                            expected.k.flatten(0, 2),
                            actual.flatten(0, 2),
                            dim=-1,
                        ).mean()
                    ),
                }
            )
    return rows


@torch.no_grad()
def _capture_dense_raw(model: TinyPRAModel, token_ids: list[int]):
    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    positions = torch.arange(len(token_ids), device=device)
    hidden = model.position_encoding.apply_embeddings(
        model.token_emb(ids),
        positions,
        model.pos_emb,
    )
    captures = []
    for block in model.blocks:
        attention = block.attn
        normalized = block.ln1(hidden)
        captures.append(
            {
                "layer_id": block.layer_id,
                "raw_query": attention.split_heads(attention.q_proj(normalized)),
                "raw_key": attention.split_heads(attention.k_proj(normalized)),
                "value": attention.split_heads(attention.v_proj(normalized)),
                "rope": attention.position_encoding,
            }
        )
        hidden = block(
            hidden,
            use_pra_memory=False,
            position_ids=positions,
        )
    return captures


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    return scores, F.softmax(scores, dim=-1) @ value


@torch.no_grad()
def _storage_matrix_rows(
    model: TinyPRAModel,
    *,
    tier: str,
    seed: int,
    git_sha: str,
) -> list[dict]:
    set_seed(30_000 + seed)
    token_ids = torch.randint(1, model.cfg.vocab_size, (LOGICAL_TOKENS,)).tolist()
    rows = []
    source_start, source_end, query_index = 64, 96, 112
    source_positions = torch.arange(source_start, source_end, device=next(model.parameters()).device)
    local_positions = torch.arange(source_end - source_start, device=source_positions.device)
    for capture in _capture_dense_raw(model, token_ids):
        layer_id = capture["layer_id"]
        rope = capture["rope"]
        raw_key = capture["raw_key"][:, :, source_start:source_end, :]
        value = capture["value"][:, :, source_start:source_end, :]
        raw_query = capture["raw_query"][:, :, query_index : query_index + 1, :]
        query_position = torch.tensor([query_index], device=source_positions.device)
        query = rope.apply_rotary(raw_query, query_position)

        post_local = rope.apply_rotary(raw_key, local_positions)
        post_offset = rope.apply_rotary(raw_key, source_positions)
        pre_local, _ = materialize_raw_rope_key(
            raw_key,
            local_positions,
            query_index,
            policy="exact_logical",
            distance_limit=model.cfg.max_seq_len,
            rope=rope,
        )
        pre_offset, _ = materialize_raw_rope_key(
            raw_key,
            source_positions,
            query_index,
            policy="exact_logical",
            distance_limit=model.cfg.max_seq_len,
            rope=rope,
        )
        modes = {
            "A_post_reset": post_local,
            "B_post_offset": post_offset,
            "C_pre_reset": pre_local,
            "D_pre_offset": pre_offset,
        }
        reference_scores, reference_output = _attention(query, post_offset, value)
        for storage_mode, key in modes.items():
            scores, output = _attention(query, key, value)
            rows.append(
                {
                    "git_sha": git_sha,
                    "seed": seed,
                    "model_tier": tier,
                    "position_mode": "rope",
                    "k_storage_mode": storage_mode,
                    "logical_offset_policy": "source_relative" if "offset" in storage_mode else "reset",
                    "layer_id": layer_id,
                    "native_context": MODEL_OPERATION_TOKENS,
                    "logical_context": LOGICAL_TOKENS,
                    "overlap_fraction": 0.0,
                    "routing_chunk_size": source_end - source_start,
                    "materialization_budget": MATERIALIZATION_BUDGET,
                    "maximum_native_operation": MODEL_OPERATION_TOKENS,
                    "key_rmse_vs_post_offset": float(
                        (key - post_offset).square().mean().sqrt()
                    ),
                    "logit_rmse_vs_post_offset": float(
                        (scores - reference_scores).square().mean().sqrt()
                    ),
                    "output_rmse_vs_post_offset": float(
                        (output - reference_output).square().mean().sqrt()
                    ),
                    "top_token_agreement_vs_post_offset": float(
                        (scores.argmax(dim=-1) == reference_scores.argmax(dim=-1))
                        .float()
                        .mean()
                    ),
                }
            )

        rebound_positions = source_positions + 512
        rebound_key = rope.apply_rotary(raw_key, rebound_positions)
        rebound_scores, rebound_output = _attention(query, rebound_key, value)
        rows.append(
            {
                "git_sha": git_sha,
                "seed": seed,
                "model_tier": tier,
                "position_mode": "rope",
                "k_storage_mode": "pre_position_rebound",
                "logical_offset_policy": "source_relative_plus_512",
                "layer_id": layer_id,
                "native_context": MODEL_OPERATION_TOKENS,
                "logical_context": LOGICAL_TOKENS + 512,
                "overlap_fraction": 0.0,
                "routing_chunk_size": source_end - source_start,
                "materialization_budget": MATERIALIZATION_BUDGET,
                "maximum_native_operation": MODEL_OPERATION_TOKENS,
                "key_rmse_vs_post_offset": float(
                    (rebound_key - post_offset).square().mean().sqrt()
                ),
                "logit_rmse_vs_post_offset": float(
                    (rebound_scores - reference_scores).square().mean().sqrt()
                ),
                "output_rmse_vs_post_offset": float(
                    (rebound_output - reference_output).square().mean().sqrt()
                ),
                "top_token_agreement_vs_post_offset": float(
                    (rebound_scores.argmax(dim=-1) == reference_scores.argmax(dim=-1))
                    .float()
                    .mean()
                ),
            }
        )
    return rows


def _aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, values in sorted(grouped.items()):
        result = dict(zip(keys, identity))
        result["seed_count"] = len({row["seed"] for row in values})
        for metric in metrics:
            observed = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = statistics.fmean(observed)
            result[f"{metric}_median"] = statistics.median(observed)
            result[f"{metric}_std"] = statistics.pstdev(observed)
        output.append(result)
    return output


def _plot_progression(rows: list[dict], path: Path) -> None:
    present_tiers = [tier for tier in ("tiny", "small") if any(row["model_tier"] == tier for row in rows)]
    figure, axes = plt.subplots(
        1,
        len(present_tiers),
        figsize=(4.1 * len(present_tiers), 3.6),
        sharey=True,
        squeeze=False,
    )
    stages = ("reset", "offset", "offset_overlap_25", "offset_overlap_50")
    labels = ("reset", "offset", "+25% overlap", "+50% overlap")
    colors = {"absolute": "#245A8D", "rope": "#A34832"}
    for axis, tier in zip(axes[0], present_tiers):
        final_layer = TIERS[tier]["n_layers"] - 1
        present_modes = [
            mode
            for mode in ("absolute", "rope")
            if any(row["model_tier"] == tier and row["position_mode"] == mode for row in rows)
        ]
        for mode in present_modes:
            present_seeds = sorted(
                {
                    row["seed"]
                    for row in rows
                    if row["model_tier"] == tier and row["position_mode"] == mode
                }
            )
            for seed in present_seeds:
                values = [
                    next(
                        row["native_k_rmse"]
                        for row in rows
                        if row["model_tier"] == tier
                        and row["position_mode"] == mode
                        and row["seed"] == seed
                        and row["layer_id"] == final_layer
                        and row["stage"] == stage
                    )
                    for stage in stages
                ]
                axis.plot(
                    range(len(stages)),
                    values,
                    color=colors[mode],
                    alpha=0.28,
                    linewidth=0.8,
                )
            means = [
                statistics.fmean(
                    row["native_k_rmse"]
                    for row in rows
                    if row["model_tier"] == tier
                    and row["position_mode"] == mode
                    and row["layer_id"] == final_layer
                    and row["stage"] == stage
                )
                for stage in stages
            ]
            axis.plot(
                range(len(stages)),
                means,
                color=colors[mode],
                marker="o",
                linewidth=2,
                label=mode,
            )
        axis.set_title(f"{tier}, layer {final_layer}")
        axis.set_xticks(range(len(stages)), labels, rotation=20, ha="right")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0, 0].set_ylabel("Native-K RMSE vs dense source")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_storage(rows: list[dict], path: Path) -> None:
    present_tiers = [tier for tier in ("tiny", "small") if any(row["model_tier"] == tier for row in rows)]
    figure, axes = plt.subplots(
        1,
        len(present_tiers),
        figsize=(4.0 * len(present_tiers), 3.5),
        squeeze=False,
    )
    cells = ("A_post_reset", "C_pre_reset", "B_post_offset", "D_pre_offset")
    labels = ("A post/reset", "C pre/reset", "B post/offset", "D pre/offset")
    for axis, tier in zip(axes[0], present_tiers):
        final_layer = TIERS[tier]["n_layers"] - 1
        values = []
        for cell in cells:
            observed = [
                row["output_rmse_vs_post_offset"]
                for row in rows
                if row["model_tier"] == tier
                and row["layer_id"] == final_layer
                and row["k_storage_mode"] == cell
            ]
            values.append(statistics.fmean(observed))
            axis.scatter(
                [len(values) - 1] * len(observed),
                [max(value, 1e-12) for value in observed],
                color="#6B6B6B",
                alpha=0.55,
                s=18,
            )
        axis.bar(
            range(len(cells)),
            [max(value, 1e-12) for value in values],
            color=("#D08B32", "#D08B32", "#327A5A", "#327A5A"),
            alpha=0.75,
        )
        axis.set_yscale("log")
        axis.set_title(f"{tier}, layer {final_layer}")
        axis.set_xticks(range(len(cells)), labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Attention-output RMSE vs post/offset")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args) -> Path:
    metadata = environment_metadata()
    representation_rows = []
    storage_rows = []
    for tier in args.tiers:
        for mode in ("absolute", "rope"):
            for seed in args.seeds:
                model = _load_converted(tier, mode, seed, args.device)
                representation_rows.extend(
                    _representation_rows(
                        model,
                        tier=tier,
                        mode=mode,
                        seed=seed,
                        git_sha=metadata["git_sha"],
                    )
                )
                if mode == "rope":
                    storage_rows.extend(
                        _storage_matrix_rows(
                            model,
                            tier=tier,
                            seed=seed,
                            git_sha=metadata["git_sha"],
                        )
                    )
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    representation_aggregate = _aggregate(
        representation_rows,
        ("model_tier", "position_mode", "stage", "layer_id"),
        ("native_k_rmse", "native_k_cosine", "encoding_cost_ratio"),
    )
    storage_aggregate = _aggregate(
        storage_rows,
        ("model_tier", "k_storage_mode", "layer_id"),
        (
            "key_rmse_vs_post_offset",
            "logit_rmse_vs_post_offset",
            "output_rmse_vs_post_offset",
            "top_token_agreement_vs_post_offset",
        ),
    )
    expectations = [
        {
            "comparison": "absolute reset -> absolute offset",
            "expected": "strong layer-0 repair within the embedding-table range",
            "reason": "restores the source-relative learned embedding",
        },
        {
            "comparison": "RoPE reset -> RoPE offset",
            "expected": "strong layer-0 positional repair",
            "reason": "restores the source-relative rotary phase",
        },
        {
            "comparison": "offset -> overlap",
            "expected": "deeper-layer improvement for both mechanisms",
            "reason": "restores part of the missing left context",
        },
        {
            "comparison": "post-offset vs pre-offset",
            "expected": "near-exact semantic parity",
            "reason": "uses the same raw K and effective positions",
        },
        {
            "comparison": "intentional K-only rebinding",
            "expected": "attention changes",
            "reason": "the memory-query relative displacement changes",
        },
    ]
    write_json(
        RESULTS / "logical_offset_decomposition.json",
        {
            "metadata": metadata,
            "expectations_recorded_before_analysis": expectations,
            "representation_rows": representation_rows,
            "representation_aggregate": representation_aggregate,
            "storage_rows": storage_rows,
            "storage_aggregate": storage_aggregate,
        },
    )
    write_csv(RESULTS / "logical_offset_decomposition.csv", representation_rows)
    write_csv(RESULTS / "rope_storage_matrix.csv", storage_rows)
    _plot_progression(representation_rows, RESULTS / "logical_offset_progression.png")
    _plot_storage(storage_rows, RESULTS / "rope_storage_matrix.png")
    return refresh_manifest(metadata=metadata)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
