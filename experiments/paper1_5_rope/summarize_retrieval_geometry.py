"""Reduce the frozen RoPE retrieval-geometry runs into publication artifacts.

This script does not execute a model. It reads the per-example JSON emitted by
``run_retrieval_geometry_gate.py`` and regenerates seed-balanced tables and the
two manuscript figures, making post-run formatting changes reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from experiments.paper1_5_rope.common import write_csv, write_json  # noqa: E402
from experiments.paper1_5_rope.run_retrieval_geometry_gate import (
    CONTEXT_DIR,
    DISTANCE_DIR,
    _seed_balanced,
)  # noqa: E402


PUBLICATION_FIELDS = (
    "dataset",
    "model_tier",
    "stage",
    "setting",
    "split_count",
    "encoding_block_references",
    "position_policy",
    "requested_distance",
    "oracle_mode",
    "seed_count",
    "example_count",
    "loss_mean",
    "loss_std",
    "loss_median",
    "token_accuracy_mean",
    "rcb_mean",
    "retrieval_key_rmse_vs_exact_mean",
    "retrieval_logit_rmse_vs_exact_mean",
    "retrieval_attention_l1_vs_exact_mean",
    "retrieval_top_token_agreement_vs_exact_mean",
    "memory_attention_mass_mean",
    "memory_last_query_attention_mass_mean",
    "original_logical_distance_mean",
    "effective_distance_mean",
    "maximum_native_operation_mean",
    "native_limit_violations_mean",
)


def _publication_rows(rows):
    return [
        {field: row.get(field) for field in PUBLICATION_FIELDS if field in row}
        for row in rows
    ]


def _distance_plot(aggregate, selected_distance):
    colors = {"hotpotqa": "#245A8D", "qasper": "#A34832"}
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for dataset, color in colors.items():
        values = sorted(
            (
                row
                for row in aggregate
                if row["stage"] == "distance"
                and row["dataset"] == dataset
                and row["position_policy"].startswith("fixed_")
            ),
            key=lambda row: row["requested_distance"],
        )
        x = [row["requested_distance"] for row in values]
        loss = [row["loss_mean"] for row in values]
        loss_std = [row["loss_std"] for row in values]
        axes[0].plot(x, loss, marker="o", color=color, label=dataset)
        axes[0].fill_between(
            x,
            [mean - std for mean, std in zip(loss, loss_std)],
            [mean + std for mean, std in zip(loss, loss_std)],
            color=color,
            alpha=0.12,
        )
        exact = next(
            row
            for row in aggregate
            if row["stage"] == "distance"
            and row["dataset"] == dataset
            and row["position_policy"] == "exact"
        )
        axes[0].axhline(exact["loss_mean"], color=color, linestyle=":", linewidth=1.1)
        axes[1].plot(
            x,
            [row["retrieval_top_token_agreement_vs_exact_mean"] for row in values],
            marker="s",
            color=color,
            label=dataset,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.axvline(selected_distance, color="#555555", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        axis.set_xlabel("Nearest-token effective distance D")
    axes[0].set_ylabel("Oracle answer-token loss")
    axes[1].set_ylabel("Top-attended-token agreement with exact")
    figure.tight_layout()
    figure.savefig(DISTANCE_DIR / "rope_d_sweep.png", dpi=180)
    plt.close(figure)


def _context_plot(aggregate):
    colors = {
        ("hotpotqa", "tiny"): "#245A8D",
        ("hotpotqa", "small"): "#5C8FBA",
        ("qasper", "tiny"): "#A34832",
        ("qasper", "small"): "#D17A62",
    }
    markers = {"tiny": "o", "small": "s"}
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    panels = (
        (
            "encoding_context",
            "encoding_block_references",
            "native_all",
            "References per encoding block",
        ),
        ("routing_chunk", "split_count", "native_oracle", "Source split count"),
    )
    for axis, (stage, x_field, oracle_mode, xlabel) in zip(axes[:2], panels):
        for (dataset, tier), color in colors.items():
            values = sorted(
                (
                    row
                    for row in aggregate
                    if row["stage"] == stage
                    and row["dataset"] == dataset
                    and row["model_tier"] == tier
                    and row["position_policy"] == "exact"
                    and row["oracle_mode"] == oracle_mode
                ),
                key=lambda row: row[x_field],
            )
            axis.errorbar(
                [row[x_field] for row in values],
                [row["loss_mean"] for row in values],
                yerr=[row["loss_std"] for row in values],
                marker=markers[tier],
                color=color,
                capsize=2,
                label=f"{dataset}, {tier}",
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Answer-token loss")
        axis.grid(alpha=0.25)
    axes[0].set_title("Encoding context; all memory")
    axes[1].set_title("Materialized unit; oracle evidence")
    axes[0].legend(frameon=False, fontsize=7)

    selected = [
        row
        for row in aggregate
        if row["stage"] in {"distance", "encoding_context", "routing_chunk"}
        and row["oracle_mode"] == "native_oracle"
    ]
    for (dataset, tier), color in colors.items():
        values = [
            row
            for row in selected
            if row["dataset"] == dataset and row["model_tier"] == tier
        ]
        axes[2].scatter(
            [row["retrieval_key_rmse_vs_exact_mean"] for row in values],
            [row["loss_mean"] for row in values],
            s=20,
            alpha=0.65,
            marker=markers[tier],
            color=color,
            label=f"{dataset}, {tier}",
        )
    axes[2].set_xlabel("Retrieved-K RMSE vs exact phase")
    axes[2].set_ylabel("Oracle answer-token loss")
    axes[2].set_title("Geometry versus utility")
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(CONTEXT_DIR / "rope_context_gate.png", dpi=180)
    plt.close(figure)


def summarize():
    distance_payload = json.loads((DISTANCE_DIR / "rope_d_sweep.json").read_text())
    context_payload = json.loads((CONTEXT_DIR / "rope_context_gate.json").read_text())
    all_rows = distance_payload["rows"] + context_payload["rows"]
    # The distance rows are duplicated in the context JSON from the original run.
    unique = {}
    for row in all_rows:
        identity = tuple(sorted(row.items()))
        unique[identity] = row
    _, aggregate = _seed_balanced(list(unique.values()))
    selected_distance = int(context_payload["protocol"]["selected_fixed_distance"])

    distance_aggregate = [row for row in aggregate if row["stage"] == "distance"]
    context_aggregate = [row for row in aggregate if row["stage"] != "distance"]
    write_csv(
        DISTANCE_DIR / "rope_d_sweep_publication.csv",
        _publication_rows(distance_aggregate),
    )
    write_csv(
        CONTEXT_DIR / "rope_context_gate_publication.csv",
        _publication_rows(context_aggregate),
    )
    summary = {
        "metadata": context_payload["metadata"],
        "protocol": context_payload["protocol"],
        "expected_vs_observed": [
            {
                "hypothesis": "tiny_fragment_contextualization",
                "observed": (
                    "The causal evidence-only row is invariant to later encoding context. "
                    "With all neighboring K/V retained, larger encoding groups materially "
                    "reduce small-tier loss and modestly reduce tiny-QASPER loss."
                ),
                "verdict": "partly supported; composition and causal order confound it",
            },
            {
                "hypothesis": "larger_retrieved_unit",
                "observed": (
                    "Exact-position oracle loss generally falls from 16-way to 5-way and "
                    "2-way partitions, with one tiny-QASPER plateau."
                ),
                "verdict": "supported in this controlled probe",
            },
            {
                "hypothesis": "stable_distance_mismatch",
                "observed": (
                    "Fixed D=64 and exact split directions nearly evenly; clipped D=64 has "
                    "only a 9.3e-5 mean loss advantage per example. D=1024 and D=4096 can "
                    "fail sharply."
                ),
                "verdict": "placement matters, but no stable beneficial fixed D was found",
            },
        ],
        "paper1_5_implication": (
            "Exact source placement repairs reset geometry. Contextualization and memory "
            "composition dominate useful near-range placement differences in these models; "
            "remote rebinding can be destructive."
        ),
        "paper2_gate": (
            "Retain exact source coordinates for continuous history. Test exact versus a "
            "small clipped-distance oracle control before enabling general rebinding in a "
            "pretrained model."
        ),
        "adaptive_geometry_gate": (
            "closed: the fixed-distance study does not justify broad adaptive geometry"
        ),
        "native_limit_violations": int(
            sum(row.get("native_limit_violations", 0) for row in unique.values())
        ),
    }
    write_json(CONTEXT_DIR / "rope_context_gate_summary.json", summary)
    _distance_plot(distance_aggregate, selected_distance)
    _context_plot(aggregate)
    return CONTEXT_DIR / "rope_context_gate_summary.json"


def parse_args():
    return argparse.ArgumentParser(description=__doc__).parse_args()


if __name__ == "__main__":
    parse_args()
    print(summarize())
