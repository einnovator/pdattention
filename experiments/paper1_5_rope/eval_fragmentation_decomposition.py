"""Decompose positional reset and lost-context effects in bounded block encoding."""

from __future__ import annotations

import argparse
import math
import sys
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
from pra_torch.config import PRAConfig  # noqa: E402
from pra_torch.model import TinyPRAModel, convert_sa_model_to_pra  # noqa: E402


def _load_converted(tier: str, mode: str, seed: int, device: str) -> TinyPRAModel:
    checkpoint_path = (
        REPO / "out" / "paper1_5_rope" / tier / mode / f"seed-{seed}" / "checkpoint.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
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


def _segmented_kv(model, token_ids, *, block_tokens: int, overlap: float, global_positions: bool):
    output = {layer_id: [] for layer_id in range(model.cfg.n_layers)}
    processed = 0
    for start in range(0, len(token_ids), block_tokens):
        end = min(start + block_tokens, len(token_ids))
        left = min(start, int(round(block_tokens * overlap)))
        encode_start = start - left
        encoded_ids = token_ids[encode_start:end]
        encoded = model._encode_reference_tokens(
            encoded_ids,
            next(model.parameters()).device,
            detach=True,
            use_pra_memory=False,
            position_offset=encode_start if global_positions else 0,
        )
        processed += len(encoded_ids)
        for layer_id, kv in encoded.items():
            output[layer_id].append(kv.k[:, :, left : left + end - start, :])
    return {layer_id: torch.cat(parts, dim=2) for layer_id, parts in output.items()}, processed


def evaluate_model(model, *, tier: str, mode: str, seed: int) -> list[dict]:
    set_seed(10_000 + seed)
    sequence_tokens = 128
    token_ids = torch.randint(
        1,
        model.cfg.vocab_size,
        (sequence_tokens,),
        device="cpu",
    ).tolist()
    dense = model._encode_reference_tokens(
        token_ids,
        next(model.parameters()).device,
        detach=True,
        use_pra_memory=False,
        position_offset=0,
    )
    rows = []
    strategies = [
        ("independent_local", 0.0, False),
        ("independent_global", 0.0, True),
        ("overlap_global", 0.05, True),
        ("overlap_global", 0.10, True),
        ("overlap_global", 0.25, True),
        ("overlap_global", 0.50, True),
    ]
    for strategy, overlap, global_positions in strategies:
        segmented, processed = _segmented_kv(
            model,
            token_ids,
            block_tokens=32,
            overlap=overlap,
            global_positions=global_positions,
        )
        for layer_id, expected in dense.items():
            actual = segmented[layer_id]
            cosine = F.cosine_similarity(
                expected.k.flatten(0, 2),
                actual.flatten(0, 2),
                dim=-1,
            ).mean()
            rows.append(
                {
                    "seed": seed,
                    "model_tier": tier,
                    "position_mode": mode,
                    "strategy": strategy,
                    "global_positions": global_positions,
                    "overlap_fraction": overlap,
                    "layer_id": layer_id,
                    "sequence_tokens": sequence_tokens,
                    "encoding_block_tokens": 32,
                    "processed_tokens": processed,
                    "encoding_cost_ratio": processed / sequence_tokens,
                    "native_k_rmse": float(
                        (expected.k - actual).square().mean().sqrt().cpu()
                    ),
                    "native_k_cosine": float(cosine.cpu()),
                }
            )
    return rows


def _plot(rows: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    colors = {"absolute": "#245A8D", "rope": "#A34832"}
    for mode in ("absolute", "rope"):
        for layer, linestyle in ((0, "--"), (1, "-")):
            points = [
                row for row in rows
                if row["position_mode"] == mode
                and row["model_tier"] == "tiny"
                and row["strategy"] == "overlap_global"
                and row["layer_id"] == layer
            ]
            grouped = {}
            for point in points:
                grouped.setdefault(point["overlap_fraction"], []).append(point["native_k_rmse"])
            x = sorted(grouped)
            y = [sum(grouped[value]) / len(grouped[value]) for value in x]
            axes[0].plot(
                x,
                y,
                color=colors[mode],
                linestyle=linestyle,
                marker="o",
                label=f"{mode}, layer {layer}",
            )
    for mode in ("absolute", "rope"):
        points = [
            row for row in rows
            if row["position_mode"] == mode
            and row["model_tier"] == "tiny"
            and row["overlap_fraction"] == 0
        ]
        labels = ("independent_local", "independent_global")
        values = [
            sum(row["native_k_rmse"] for row in points if row["strategy"] == label)
            / max(sum(row["strategy"] == label for row in points), 1)
            for label in labels
        ]
        offset = -0.18 if mode == "absolute" else 0.18
        axes[1].bar(
            [index + offset for index in range(2)],
            values,
            width=0.36,
            color=colors[mode],
            label=mode,
        )
    axes[0].set_xlabel("Left overlap fraction")
    axes[0].set_ylabel("Native-K RMSE vs dense encoding")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xticks((0, 1), ("local reset", "logical global"))
    axes[1].set_ylabel("Mean native-K RMSE")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args):
    rows = []
    for tier in args.tiers:
        for mode in ("absolute", "rope"):
            for seed in args.seeds:
                model = _load_converted(tier, mode, seed, args.device)
                rows.extend(evaluate_model(model, tier=tier, mode=mode, seed=seed))
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
    metadata = environment_metadata()
    write_json(
        RESULTS / "rope_overlap.json",
        {"metadata": metadata, "rows": rows},
    )
    write_csv(RESULTS / "rope_overlap.csv", rows)
    _plot(rows, RESULTS / "rope_overlap.png")
    refresh_manifest()
    return RESULTS / "rope_overlap.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--tiers", nargs="+", choices=tuple(TIERS), default=list(TIERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
