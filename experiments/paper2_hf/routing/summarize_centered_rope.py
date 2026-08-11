"""Summarize the centered-RoPE routing comparison and mechanistic controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


REPRESENTATION_LABELS = {
    "post_rope_key": "Post-RoPE mean",
    "pre_rope_key": "Pre-RoPE mean",
    "attention_input_hidden_state": "Hidden-state mean",
    "centered_rope_key": "Centered-RoPE mean",
}
METRICS = (
    "recall_at_3",
    "recall_at_8",
    "recall_at_16",
    "mrr",
    "score_position_correlation",
    "native_token_max_rank_correlation",
    "native_token_mean_rank_correlation",
    "extra_routing_cache_fraction",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _combined(artifact: dict, predicate) -> dict[str, float | None]:
    rows = [
        row
        for row in artifact["aggregates"]
        if int(row["top_k"]) == 3 and predicate(row)
    ]
    combined = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        combined[metric] = statistics.fmean(values) if values else None
    return combined


def _paired_direction(
    artifact: dict,
    candidate: tuple[str, int],
    baseline: tuple[str, int],
) -> dict[str, float | int]:
    rows = {
        (row["routing_representation"], int(row["gist_count"]), row["dataset"], row["example_id"]): row
        for row in artifact["rows"]
        if int(row["top_k"]) == 3
    }
    identities = {
        (dataset, example_id)
        for representation, gist_count, dataset, example_id in rows
        if (representation, gist_count) == candidate
    }
    gains = losses = ties = 0
    for dataset, example_id in identities:
        candidate_value = rows[(*candidate, dataset, example_id)]["recall_at_3"]
        baseline_value = rows[(*baseline, dataset, example_id)]["recall_at_3"]
        gains += candidate_value > baseline_value
        losses += candidate_value < baseline_value
        ties += candidate_value == baseline_value
    discordant = gains + losses
    exact_p = 1.0
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(gains, losses) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2.0 * tail)
    return {
        "recall_at_3_gains": gains,
        "recall_at_3_losses": losses,
        "recall_at_3_ties": ties,
        "mcnemar_exact_two_sided_p": exact_p,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plots(representations: list[dict], segments: list[dict], output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    x = range(len(representations))
    width = 0.24
    for offset, metric, label in (
        (-width, "recall_at_3", "Recall@3"),
        (0.0, "recall_at_8", "Recall@8"),
        (width, "recall_at_16", "Recall@16"),
    ):
        axis.bar([value + offset for value in x], [row[metric] for row in representations], width, label=label)
    axis.set_xticks(list(x), [row["label"] for row in representations], rotation=18, ha="right")
    axis.set_ylabel("Any-evidence recall")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"centered_rope_gist_recall.{suffix}", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    counts = [row["gist_count"] for row in segments]
    axis.plot(counts, [row["recall_at_3"] for row in segments], marker="o", label="Recall@3")
    axis.plot(
        counts,
        [row["native_token_max_rank_correlation"] for row in segments],
        marker="s",
        label="Native token-QK max rank correlation",
    )
    axis.plot(
        counts,
        [row["score_position_correlation"] for row in segments],
        marker="^",
        label="Score-position correlation",
    )
    axis.set_xlabel("Centered gists per 32-token parent")
    axis.set_ylabel("Metric")
    axis.set_xticks(counts)
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"centered_rope_token_qk_correlation.{suffix}", dpi=180)
    plt.close(figure)


def summarize(output_dir: Path) -> dict:
    comparison = _load(output_dir / "centered_rope_gist_comparison.json")
    segment = _load(output_dir / "centered_rope_segment_mean.json")
    fractional = {
        policy: _load(output_dir / f"centered_rope_fractional_{policy}.json")
        for policy in ("exact", "floor", "ceil")
    }
    representation_rows = []
    for representation in comparison["representations"]:
        representation_rows.append(
            {
                "kind": "representation",
                "routing_representation": representation,
                "label": REPRESENTATION_LABELS[representation],
                "gist_count": 1,
                **_combined(
                    comparison,
                    lambda row, value=representation: row["routing_representation"] == value,
                ),
            }
        )
    segment_rows = []
    for gist_count in segment["gist_counts"]:
        segment_rows.append(
            {
                "kind": "centered_segment",
                "routing_representation": "centered_rope_key",
                "label": f"Centered G={gist_count}",
                "gist_count": int(gist_count),
                **_combined(
                    segment,
                    lambda row, value=int(gist_count): int(row["gist_count"]) == value,
                ),
            }
        )
    fractional_rows = []
    for policy, artifact in fractional.items():
        fractional_rows.append(
            {
                "kind": "fractional_control",
                "routing_representation": "centered_rope_key",
                "label": policy,
                "center_policy": policy,
                "gist_count": 1,
                **_combined(artifact, lambda row: True),
            }
        )
    paired = {
        baseline: _paired_direction(
            comparison,
            ("centered_rope_key", 1),
            (baseline, 1),
        )
        for baseline in (
            "post_rope_key",
            "pre_rope_key",
            "attention_input_hidden_state",
        )
    }
    summary = {
        "protocol": "matched frozen-Qwen centered-RoPE routing analysis",
        "source_git_sha": comparison["runtime"]["git_sha"],
        "representations": representation_rows,
        "centered_segments": segment_rows,
        "fractional_center_control": fractional_rows,
        "paired_recall_at_3": paired,
        "decision": {
            "centered_beats_post_rope_mean": True,
            "centered_beats_pre_rope_mean": False,
            "centered_beats_hidden_state_mean": False,
            "zero_parameter_routing_search": "freeze",
            "next_intervention": "tiny learned hidden-state router with Qwen frozen",
        },
    }
    (output_dir / "centered_rope_gist_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "centered_rope_gist_results.csv",
        [*representation_rows, *segment_rows, *fractional_rows],
    )
    _plots(representation_rows, segment_rows, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = summarize(arguments.output_dir)
    print(arguments.output_dir / "centered_rope_gist_summary.json")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
