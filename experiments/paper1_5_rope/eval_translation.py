"""Pure position experiments that hold content and projections fixed."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pra_torch.positions import RotaryPositionEncoding


def _distribution_metrics(scores_a: torch.Tensor, scores_b: torch.Tensor, value: torch.Tensor):
    weights_a = F.softmax(scores_a, dim=-1)
    weights_b = F.softmax(scores_b, dim=-1)
    midpoint = 0.5 * (weights_a + weights_b)
    js = 0.5 * (
        F.kl_div(midpoint.log(), weights_a, reduction="batchmean")
        + F.kl_div(midpoint.log(), weights_b, reduction="batchmean")
    )
    output_a = weights_a @ value
    output_b = weights_b @ value
    return {
        "attention_logit_rmse": float((scores_a - scores_b).square().mean().sqrt()),
        "attention_js": float(js),
        "top_token_agreement": float(
            (scores_a.argmax(dim=-1) == scores_b.argmax(dim=-1)).float().mean()
        ),
        "attention_output_rmse": float((output_a - output_b).square().mean().sqrt()),
    }


def run_translation(seed: int = 1729) -> list[dict]:
    """Measure common-translation invariance from position 0 through 32 native lengths."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for head_dim in (8, 16, 32, 64):
        query = torch.randn(2, 4, 17, head_dim, generator=generator)
        key = torch.randn(2, 4, 17, head_dim, generator=generator)
        value = torch.randn(2, 4, 17, head_dim, generator=generator)
        rope = RotaryPositionEncoding(head_dim)
        positions = torch.arange(17)
        q0, k0 = rope.transform_qk(query, key, positions)
        base_scores = q0 @ k0.transpose(-2, -1) / math.sqrt(head_dim)
        for shift in (0, 1, 16, 128, 1_024, 8_192):
            q1, k1 = rope.transform_qk(query, key, positions + shift)
            shifted_scores = q1 @ k1.transpose(-2, -1) / math.sqrt(head_dim)
            rows.append(
                {
                    "position_mode": "rope",
                    "head_dim": head_dim,
                    "heads": 4,
                    "batch": 2,
                    "tokens": 17,
                    "translation": shift,
                    **_distribution_metrics(base_scores, shifted_scores, value),
                }
            )
    return rows


def run_pre_post(seed: int = 2718) -> list[dict]:
    """Compare published post-RoPE K with deferred rotation of pre-positional K."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for head_dim in (16, 32, 64):
        raw_query = torch.randn(1, 4, 11, head_dim, generator=generator)
        raw_key = torch.randn(1, 4, 11, head_dim, generator=generator)
        rope = RotaryPositionEncoding(head_dim)
        source_positions = torch.arange(11) + 300
        query_positions = torch.arange(11) + 320
        post_query = rope.apply_rotary(raw_query, query_positions)
        post_key = rope.apply_rotary(raw_key, source_positions)
        deferred_query, deferred_key = rope.transform_qk(
            raw_query, raw_key, source_positions
        )
        deferred_query = rope.apply_rotary(raw_query, query_positions)
        same_scores = post_query @ post_key.transpose(-2, -1)
        deferred_scores = deferred_query @ deferred_key.transpose(-2, -1)
        for relocation in (0, 128, 1_024, 8_192):
            relocated_positions = source_positions + relocation
            relocated_key = rope.apply_rotary(raw_key, relocated_positions)
            fixed_query_scores = post_query @ relocated_key.transpose(-2, -1)
            translated_query = rope.apply_rotary(raw_query, query_positions + relocation)
            common_translation_scores = translated_query @ relocated_key.transpose(-2, -1)
            rows.append(
                {
                    "head_dim": head_dim,
                    "tokens": 11,
                    "relocation": relocation,
                    "same_position_post_vs_deferred_rmse": float(
                        (same_scores - deferred_scores).square().mean().sqrt()
                    ),
                    "relocated_k_fixed_query_rmse": float(
                        (same_scores - fixed_query_scores).square().mean().sqrt()
                    ),
                    "common_translation_rmse": float(
                        (same_scores - common_translation_scores).square().mean().sqrt()
                    ),
                    "position_metadata_bytes": int(source_positions.numel() * 8),
                    "kv_bytes_fp32": int(2 * raw_key.numel() * 4),
                }
            )
    return rows


if __name__ == "__main__":
    from experiments.paper1_5_rope.common import (
        RESULTS,
        environment_metadata,
        refresh_manifest,
        write_csv,
        write_json,
    )
    from experiments.paper1_5_rope.reporting import plot_translation

    translation_rows = run_translation()
    pre_post_rows = run_pre_post()
    metadata = environment_metadata()
    write_json(
        RESULTS / "rope_translation.json",
        {"metadata": metadata, "rows": translation_rows},
    )
    write_csv(RESULTS / "rope_translation.csv", translation_rows)
    write_json(
        RESULTS / "rope_pre_post_k.json",
        {"metadata": metadata, "rows": pre_post_rows},
    )
    write_csv(RESULTS / "rope_pre_post_k.csv", pre_post_rows)
    plot_translation(translation_rows, RESULTS / "rope_translation.png")
    print(refresh_manifest(metadata=metadata))
