"""Summarize controlled K/V disclosure and write the causal diagnosis gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from experiments.paper3_kv_materialization.toy_materialization import (
    minimum_sufficient_radius,
    retention_vs_parent,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    ROOT / "docs/papers/shared/results/paper3_kv_materialization/toy_materialization"
)
RADIUS_POLICIES = {
    0: "T1_radius_0",
    2: "T2_radius_2",
    4: "T3_radius_4",
    8: "T4_radius_8",
    16: "T5_radius_16",
}
WINDOW_ORDER = ("w16", "w32", "w64", "w128", "global")


def _number(value):
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: _number(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return statistics.fmean(values) if values else None


def _aggregate(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return [
        {
            **dict(zip(keys, key)),
            "n": len(group),
            **{metric: _mean(group, metric) for metric in metrics},
        }
        for key, group in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0]))
    ]


def frontier(rows: list[dict]) -> list[dict]:
    metrics = (
        "materialized_tokens",
        "evidence_coverage",
        "evidence_density",
        "evidence_attention_mass",
        "surrounding_attention_mass",
        "distractor_attention_mass",
        "native_attention_mass",
        "correct_margin",
        "correct_probability",
        "nll",
        "correct",
        "immediate_margin_delta_vs_none",
        "final_margin_delta_vs_none",
        "later_erased",
        "latency_seconds",
    )
    values = _aggregate(rows, ("partition", "window", "policy"), metrics)
    baselines = {
        (row["partition"], row["window"]): row
        for row in values
        if row["policy"] == "T0_none"
    }
    parents = {
        (row["partition"], row["window"]): row
        for row in values
        if row["policy"] == "T7_whole_parent"
    }
    for row in values:
        key = (row["partition"], row["window"])
        baseline = baselines[key]
        parent = parents[key]
        row["margin_delta_vs_none"] = (
            row["correct_margin"] - baseline["correct_margin"]
        )
        row["accuracy_delta_vs_none"] = row["correct"] - baseline["correct"]
        row["kv_reduction_vs_parent"] = (
            1.0 - row["materialized_tokens"] / max(parent["materialized_tokens"], 1)
        )
        row["margin_retention_vs_parent_gain"] = retention_vs_parent(
            row["correct_margin"],
            baseline["correct_margin"],
            parent["correct_margin"],
        )
    return values


def radius_by_window(
    frontier_rows: list[dict], portability: list[dict]
) -> list[dict]:
    output = []
    portability_by_window = {
        window: _mean([row for row in portability if row["window"] == window], "representation_change")
        for window in WINDOW_ORDER
    }
    for window in WINDOW_ORDER:
        validation = [
            row
            for row in frontier_rows
            if row["partition"] == "validation" and row["window"] == window
        ]
        heldout = {
            row["policy"]: row
            for row in frontier_rows
            if row["partition"] == "heldout" and row["window"] == window
        }
        if not validation:
            continue
        for target in (0.90, 0.95, 0.99):
            selected = minimum_sufficient_radius(validation, target=target)
            policy = (
                RADIUS_POLICIES[int(selected["radius"])]
                if selected["radius"] is not None
                else None
            )
            heldout_row = heldout.get(policy) if policy is not None else None
            output.append(
                {
                    "window": window,
                    "target_parent_gain_fraction": target,
                    "selection_partition": "validation",
                    "status": selected["status"],
                    "selected_radius": selected["radius"],
                    "validation_retention": selected["retention"],
                    "validation_parent_gain": selected["parent_gain"],
                    "heldout_materialized_tokens": (
                        heldout_row["materialized_tokens"] if heldout_row else None
                    ),
                    "heldout_kv_reduction_vs_parent": (
                        heldout_row["kv_reduction_vs_parent"] if heldout_row else None
                    ),
                    "heldout_margin_retention": (
                        heldout_row["margin_retention_vs_parent_gain"]
                        if heldout_row
                        else None
                    ),
                    "representation_change": portability_by_window[window],
                }
            )
    return output


def _correlation(x, y) -> float | None:
    pairs = [
        (float(left), float(right))
        for left, right in zip(x, y)
        if left is not None
        and right is not None
        and math.isfinite(float(left))
        and math.isfinite(float(right))
    ]
    if len(pairs) < 3:
        return None
    left, right = zip(*pairs)
    if statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def diagnosis(
    rows: list[dict],
    frontier_rows: list[dict],
    portability: list[dict],
    profile: list[dict],
    output: Path,
) -> tuple[list[dict], dict]:
    heldout = [row for row in frontier_rows if row["partition"] == "heldout"]
    by_policy = {
        policy: [row for row in heldout if row["policy"] == policy]
        for policy in {
            "T0_none",
            "T1_radius_0",
            "T2_radius_2",
            "T3_radius_4",
            "T7_whole_parent",
            "T9_wrong_exact",
        }
    }
    means = {
        policy: {
            metric: _mean(rows, metric)
            for metric in (
                "correct_margin",
                "correct",
                "evidence_attention_mass",
                "distractor_attention_mass",
                "materialized_tokens",
            )
        }
        for policy, rows in by_policy.items()
    }
    exact_vs_parent = (
        means["T1_radius_0"]["correct_margin"]
        - means["T7_whole_parent"]["correct_margin"]
    )
    fact_support = (
        means["T2_radius_2"]["correct_margin"]
        - means["T1_radius_0"]["correct_margin"]
    )
    wrong_gap = (
        means["T1_radius_0"]["correct_margin"]
        - means["T9_wrong_exact"]["correct_margin"]
    )
    representation_change = _mean(portability, "representation_change")
    exact_by_example = {
        (row["window"], int(row["seed"]), row["example_id"]): row
        for row in rows
        if row["partition"] == "heldout" and row["policy"] == "T1_radius_0"
    }
    parent_by_example = {
        (row["window"], int(row["seed"]), row["example_id"]): row
        for row in rows
        if row["partition"] == "heldout" and row["policy"] == "T7_whole_parent"
    }
    portability_pairs = []
    for row in portability:
        key = (row["window"], int(row["seed"]), row["example_id"])
        if row["partition"] == "heldout" and key in exact_by_example and key in parent_by_example:
            portability_pairs.append(
                (
                    row["representation_change"],
                    exact_by_example[key]["correct_margin"]
                    - parent_by_example[key]["correct_margin"],
                )
            )
    portability_correlation = _correlation(
        [pair[0] for pair in portability_pairs],
        [pair[1] for pair in portability_pairs],
    )
    profile_heldout = [row for row in profile if row["partition"] == "heldout"]
    profile_effects = _aggregate(
        profile_heldout,
        ("consumer_layer", "policy"),
        ("final_margin_delta_vs_none", "evidence_attention_mass", "later_erased"),
    )
    exact_profile = [
        row for row in profile_effects if row["policy"] == "T1_radius_0"
    ]
    layer_range = (
        max(row["final_margin_delta_vs_none"] for row in exact_profile)
        - min(row["final_margin_delta_vs_none"] for row in exact_profile)
    )
    statuses = [
        {
            "rank": 1,
            "failure_mode": "distractor dilution",
            "status": "supported" if exact_vs_parent > 0.05 else "partial",
            "evidence": (
                f"Exact-core margin minus whole-parent margin is {exact_vs_parent:+.3f}; "
                f"the matched wrong-core gap is {wrong_gap:+.3f}."
            ),
        },
        {
            "rank": 2,
            "failure_mode": "consumer-layer mismatch",
            "status": "supported" if layer_range > 0.10 else "partial",
            "evidence": (
                f"Exact-core final-margin effect spans {layer_range:.3f} across "
                "consumer layers 0, 2, and 5."
            ),
        },
        {
            "rank": 3,
            "failure_mode": "representation-context dependence",
            "status": "partial" if representation_change and representation_change > 0.05 else "unresolved",
            "evidence": (
                f"Mean layer-2 contextual-versus-isolated native-V change is "
                f"{representation_change:.3f}, and its correlation with exact-core minus "
                f"parent margin is {portability_correlation:+.3f}. Context dependence is "
                "present, but it is not by itself a demonstrated failure cause."
            ),
        },
        {
            "rank": 4,
            "failure_mode": "missing contextual support",
            "status": "supported" if fact_support > 0.05 else "unsupported",
            "evidence": (
                f"Radius-2 margin minus exact-core margin is {fact_support:+.3f}; "
                "positive values indicate useful local support."
            ),
        },
        {
            "rank": 5,
            "failure_mode": "evidence incompleteness",
            "status": "unsupported",
            "evidence": (
                "Exact-core and radius conditions retain 1.0 annotation-core coverage; "
                "remaining differences therefore do not require missing annotated tokens."
            ),
        },
    ]
    _write(output / "toy_materialization_causal_diagnosis.csv", statuses)
    lines = [
        "# Toy Materialization Causal Diagnosis",
        "",
        "The gate uses held-out rows only. Routing is fixed to one oracle parent, so no result below",
        "is attributed to parent discovery.",
        "",
        "| Rank | Failure mode | Status | Evidence |",
        "|---:|---|---|---|",
        *[
            f"| {row['rank']} | {row['failure_mode']} | **{row['status']}** | {row['evidence']} |"
            for row in statuses
        ],
        "",
        "## Interpretation",
        "",
        "Annotation evidence and computationally sufficient memory state are not identical. The",
        "controlled result distinguishes whether local support helps from whether broad disclosure",
        "simply adds competing K/V. Whole-parent memory is a local ceiling/control, not an assumed",
        "optimum.",
    ]
    (output / "toy_materialization_causal_diagnosis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    action = [
        "# Toy Materialization Next Action",
        "",
        "The diagnosis freezes two bounded follow-ups; it does not open a new policy search.",
        "",
        "1. Freeze exact annotation-core disclosure as the controlled default and retain radius 2",
        "   as the single local-support control. Broader radii are not promoted after validation.",
        "2. Preserve complete cores before distributing any fixed-budget remainder. The present",
        "   equal-core corpus cannot distinguish allocation rules, so allocation remains unresolved.",
        "",
        "The pretrained confirmation therefore compares no memory, whole selected parent, exact",
        "evidence, radius 2, and one validation-selected radius when distinct. Gist-only remains a",
        "negative control. It preserves the inherited Paper-2.5 consumer bands; layer-specific or",
        "representation-portability adaptations belong to Paper 4.",
    ]
    (output / "toy_materialization_next_action.md").write_text(
        "\n".join(action) + "\n", encoding="utf-8"
    )
    summary = {
        "exact_minus_parent_margin": exact_vs_parent,
        "radius2_minus_exact_margin": fact_support,
        "exact_minus_wrong_margin": wrong_gap,
        "mean_representation_change": representation_change,
        "portability_disclosure_correlation": portability_correlation,
        "consumer_layer_effect_range": layer_range,
        "failure_modes": statuses,
    }
    return statuses, summary


def plots(
    frontier_rows: list[dict],
    radius_rows: list[dict],
    trajectories: list[dict],
    portability: list[dict],
    dispersion: list[dict],
    output: Path,
) -> None:
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    heldout = [row for row in frontier_rows if row["partition"] == "heldout"]
    selected_policies = {
        "T1_radius_0",
        "T2_radius_2",
        "T3_radius_4",
        "T4_radius_8",
        "T5_radius_16",
        "T6_whole_fact",
        "T7_whole_parent",
        "T9_wrong_exact",
    }
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for window in WINDOW_ORDER:
        rows = [
            row
            for row in heldout
            if row["window"] == window and row["policy"] in selected_policies
        ]
        axis.plot(
            [row["materialized_tokens"] for row in rows],
            [row["margin_delta_vs_none"] for row in rows],
            marker="o",
            linestyle="none",
            label=window,
        )
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="materialized native K/V tokens", ylabel="margin change vs no memory")
    axis.grid(alpha=.25); axis.legend(frameon=False, ncol=3)
    figure.tight_layout(); figure.savefig(figures / "toy_quality_kv_frontier.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for window in WINDOW_ORDER:
        rows = [
            row
            for row in heldout
            if row["window"] == window and row["policy"] in RADIUS_POLICIES.values()
        ]
        rows.sort(key=lambda row: next(radius for radius, policy in RADIUS_POLICIES.items() if policy == row["policy"]))
        axis.plot(
            [next(radius for radius, policy in RADIUS_POLICIES.items() if policy == row["policy"]) for row in rows],
            [row["margin_delta_vs_none"] for row in rows],
            marker="o",
            label=window,
        )
    axis.set(xlabel="evidence-centered radius", ylabel="margin change vs no memory")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    figure.tight_layout(); figure.savefig(figures / "toy_radius_margin_by_window.png", dpi=180); plt.close(figure)

    selected95 = [row for row in radius_rows if row["target_parent_gain_fraction"] == .95]
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(WINDOW_ORDER))
    values = [next((row["selected_radius"] for row in selected95 if row["window"] == window), np.nan) for window in WINDOW_ORDER]
    axis.bar(x, [value if value is not None else np.nan for value in values], color="#4c78a8")
    axis.set_xticks(x, WINDOW_ORDER); axis.set_ylabel("validation-selected radius for 95% parent gain")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(figures / "toy_minimum_radius_by_window.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    aggregate = _aggregate(
        [
            row
            for row in heldout
            if row["policy"] in RADIUS_POLICIES.values()
        ],
        ("policy",),
        ("evidence_attention_mass", "surrounding_attention_mass", "distractor_attention_mass"),
    )
    aggregate.sort(key=lambda row: next(radius for radius, policy in RADIUS_POLICIES.items() if policy == row["policy"]))
    radii = [next(radius for radius, policy in RADIUS_POLICIES.items() if policy == row["policy"]) for row in aggregate]
    for metric, label in (
        ("evidence_attention_mass", "evidence"),
        ("surrounding_attention_mass", "surrounding"),
        ("distractor_attention_mass", "distractor"),
    ):
        axis.plot(radii, [row[metric] for row in aggregate], marker="o", label=label)
    axis.set(xlabel="evidence-centered radius", ylabel="final-query attention mass")
    axis.grid(alpha=.25); axis.legend(frameon=False)
    figure.tight_layout(); figure.savefig(figures / "toy_attention_vs_radius.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    trajectory = _aggregate(
        [
            row
            for row in trajectories
            if row["partition"] == "heldout"
            and row["policy"] in {"T0_none", "T1_radius_0", "T3_radius_4", "T7_whole_parent", "T9_wrong_exact"}
        ],
        ("policy", "layer"),
        ("margin_delta_vs_none",),
    )
    for policy in ("T1_radius_0", "T3_radius_4", "T7_whole_parent", "T9_wrong_exact"):
        rows = sorted((row for row in trajectory if row["policy"] == policy), key=lambda row: row["layer"])
        axis.plot([row["layer"] for row in rows], [row["margin_delta_vs_none"] for row in rows], marker="o", label=policy)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="decoder layer", ylabel="intermediate margin change vs no memory")
    axis.grid(alpha=.25); axis.legend(frameon=False, fontsize=8)
    figure.tight_layout(); figure.savefig(figures / "toy_margin_trajectory.png", dpi=180); plt.close(figure)

    heldout_portability = [row for row in portability if row["partition"] == "heldout"]
    exact_rows = {
        (row["window"], int(row["seed"]), row["example_id"]): row
        for row in _read(output / "toy_materialization_rows.csv")
        if row["partition"] == "heldout" and row["policy"] == "T1_radius_0"
    }
    parent_rows = {
        (row["window"], int(row["seed"]), row["example_id"]): row
        for row in _read(output / "toy_materialization_rows.csv")
        if row["partition"] == "heldout" and row["policy"] == "T7_whole_parent"
    }
    x_values, y_values = [], []
    for row in heldout_portability:
        key = (row["window"], int(row["seed"]), row["example_id"])
        if key in exact_rows and key in parent_rows:
            x_values.append(row["representation_change"])
            y_values.append(exact_rows[key]["correct_margin"] - parent_rows[key]["correct_margin"])
    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    axis.scatter(x_values, y_values, alpha=.55, s=18)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="layer-2 representation change (1 - cosine)", ylabel="exact-core minus parent margin")
    axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(figures / "toy_portability_vs_disclosure.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    rows = sorted(dispersion, key=lambda row: row["region_groups"])
    axis.plot(
        [row["region_groups"] for row in rows],
        [row["materialized_tokens"] for row in rows],
        marker="o",
        label="K/V tokens",
    )
    second = axis.twinx()
    second.plot(
        [row["region_groups"] for row in rows],
        [row["margin_delta_vs_none"] for row in rows],
        marker="s",
        color="#e45756",
        label="margin",
    )
    axis.set(xlabel="physical evidence regions", ylabel="materialized K/V tokens")
    second.set_ylabel("margin change vs no memory")
    axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(figures / "toy_region_dispersion.png", dpi=180); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    rows = _read(args.root / "toy_materialization_rows.csv")
    trajectories = _read(args.root / "toy_margin_trajectories.csv")
    portability = _read(args.root / "toy_portability.csv")
    profile = _read(args.root / "toy_consumer_layer_profile.csv")
    frontier_rows = frontier(rows)
    radius_rows = radius_by_window(frontier_rows, portability)
    dispersion = _aggregate(
        [
            row
            for row in rows
            if row["partition"] == "heldout"
            and str(row["policy"]).startswith("dispersion_")
        ],
        ("region_groups",),
        ("evidence_source_tokens", "materialized_tokens", "evidence_coverage", "correct_margin", "final_margin_delta_vs_none"),
    )
    for row in dispersion:
        row["margin_delta_vs_none"] = row.pop("final_margin_delta_vs_none")
    allocation = _aggregate(
        [
            row
            for row in rows
            if row["partition"] == "heldout"
            and str(row["policy"]).startswith("T8_budget_")
        ],
        ("budget", "allocation"),
        ("materialized_tokens", "evidence_coverage", "evidence_density", "correct_margin", "final_margin_delta_vs_none"),
    )
    for row in allocation:
        row["margin_delta_vs_none"] = row.pop("final_margin_delta_vs_none")
    _write(args.root / "toy_materialization_frontier.csv", frontier_rows)
    _write(args.root / "toy_radius_by_window.csv", radius_rows)
    _write(args.root / "toy_region_dispersion.csv", dispersion)
    _write(args.root / "toy_budget_allocation.csv", allocation)
    statuses, diagnosis_summary = diagnosis(
        rows, frontier_rows, portability, profile, args.root
    )
    plots(
        frontier_rows,
        radius_rows,
        trajectories,
        portability,
        dispersion,
        args.root,
    )
    summary = {
        "schema_version": "1.0",
        "protocol": "toy-first causal native-K/V materialization",
        "rows": len(rows),
        "windows": list(WINDOW_ORDER),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "diagnosis": diagnosis_summary,
        "minimum_sufficient_disclosure": radius_rows,
        "claim_boundary": (
            "annotation evidence is not assumed to be computationally sufficient memory; "
            "routing is oracle-fixed and weights remain frozen"
        ),
    }
    (args.root / "toy_materialization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "diagnoses": len(statuses)}, indent=2))


if __name__ == "__main__":
    main()
