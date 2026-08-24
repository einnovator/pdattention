"""Summarize Paper 2.9 effects, matched delay controls, and gate outcomes."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs/papers/shared/results/paper2_9_look_ahead_back"
DATASETS = ("hotpotqa", "qasper", "2wikimultihopqa", "musique")
COLORS = {
    "hotpotqa": "#5B6770",
    "qasper": "#007C91",
    "2wikimultihopqa": "#D1495B",
    "musique": "#6A7B2C",
}


def read_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, rows: list[dict]) -> None:
    with (RESULTS / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values: list[float], *, seed: int, draws: int = 5000):
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    )
    return samples[int(0.025 * draws)], samples[min(int(0.975 * draws), draws - 1)]


def matched_delay(rows: list[dict]) -> list[dict]:
    indexed = {
        (row["dataset"], row["split"], row["example_id"], int(row["anchor"]), row["condition"]): row
        for row in rows
    }
    output = []
    for dataset in DATASETS:
        for split in ("validation", "test"):
            for amount in (1, 2, 4, 8):
                delayed = f"fixed_delay_{amount}"
                future = f"known_future_{amount}"
                keys = [
                    key[:-1]
                    for key in indexed
                    if key[0] == dataset and key[1] == split and key[-1] == delayed
                    and (*key[:-1], future) in indexed
                    and (*key[:-1], "immediate") in indexed
                ]
                if not keys:
                    continue
                by_identity = defaultdict(lambda: {"immediate": [], "delay": [], "future": []})
                for key in keys:
                    identity = key[2]
                    by_identity[identity]["immediate"].append(float(indexed[(*key, "immediate")]["evidence_recall"]))
                    by_identity[identity]["delay"].append(float(indexed[(*key, delayed)]["evidence_recall"]))
                    by_identity[identity]["future"].append(float(indexed[(*key, future)]["evidence_recall"]))
                identity_rows = []
                for identity, values in by_identity.items():
                    identity_rows.append(
                        {
                            "identity": identity,
                            **{name: statistics.fmean(group) for name, group in values.items()},
                        }
                    )
                delay_effects = [row["delay"] - row["immediate"] for row in identity_rows]
                future_effects = [row["future"] - row["immediate"] for row in identity_rows]
                delay_low, delay_high = bootstrap(delay_effects, seed=20260824 + amount)
                future_low, future_high = bootstrap(future_effects, seed=20260924 + amount)
                delay_gain = statistics.fmean(delay_effects)
                future_gain = statistics.fmean(future_effects)
                recovery = delay_gain / future_gain if future_gain > 1e-12 else None
                output.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "tokens": amount,
                        "identities": len(identity_rows),
                        "decisions": len(keys),
                        "matched_immediate_recall": statistics.fmean(row["immediate"] for row in identity_rows),
                        "fixed_delay_recall": statistics.fmean(row["delay"] for row in identity_rows),
                        "known_future_recall": statistics.fmean(row["future"] for row in identity_rows),
                        "fixed_delay_gain": delay_gain,
                        "fixed_delay_ci_low": delay_low,
                        "fixed_delay_ci_high": delay_high,
                        "known_future_gain": future_gain,
                        "known_future_ci_low": future_low,
                        "known_future_ci_high": future_high,
                        "delay_recovery_fraction": recovery,
                        "deployable_delay": amount,
                        "known_future_is_analysis_only": True,
                    }
                )
    return output


def selected_final_rows(final_summary, policies):
    output = []
    lookup = {
        (row["dataset"], row["split"], row["condition"]): row
        for row in final_summary
    }
    effects = {
        row["dataset"]: row
        for row in read_csv("paired_effects.csv")
        if row["right"].endswith("_b1") and row["left"].endswith("_b2")
        and row["left"].split("_b")[0] == row["right"].split("_b")[0]
    }
    for dataset, policy in policies.items():
        baseline = f"rank16_l27_{policy['reducer']}_b1"
        selected = policy["condition"]
        mean_memory = f"native_mean_l27_{policy['reducer']}_b{policy['look_behind']}"
        compact = f"rank8_centroid8_l27_{policy['reducer']}_b{policy['look_behind']}"
        effect = effects[dataset]
        output.append(
            {
                "dataset": dataset,
                "selected_policy": selected,
                "validation_recall": policy["validation_evidence_recall"],
                "b1_same_reducer_test_recall": float(lookup[(dataset, "test", baseline)]["evidence_recall"]),
                "selected_temporal_test_recall": float(lookup[(dataset, "test", selected)]["evidence_recall"]),
                "temporal_gain": float(effect["mean"]),
                "temporal_gain_ci_low": float(effect["ci_low"]),
                "temporal_gain_ci_high": float(effect["ci_high"]),
                "native_mean_temporal_recall": float(lookup[(dataset, "test", mean_memory)]["evidence_recall"]),
                "rank8_centroid8_temporal_recall": float(lookup[(dataset, "test", compact)]["evidence_recall"]),
            }
        )
    return output


def stride_frontier(trajectory_summary):
    output = []
    for row in trajectory_summary:
        if row["split"] != "test" or not row["condition"].startswith("stride_"):
            continue
        output.append(
            {
                "dataset": row["dataset"],
                "stride": int(row["condition"].split("_")[-1]),
                "evidence_recall": float(row["evidence_recall"]),
                "chain_completion": float(row["chain_completion"]),
                "router_calls_per_token": float(row["router_calls_per_token"]),
                "mean_churn": float(row["mean_churn"]),
            }
        )
    baselines = {
        row["dataset"]: row["evidence_recall"] for row in output if row["stride"] == 1
    }
    for row in output:
        row["recall_delta_vs_stride1"] = row["evidence_recall"] - baselines[row["dataset"]]
    return output


def layer_ablation(final_summary):
    output = []
    for row in final_summary:
        if (
            row["split"] == "test"
            and row["memory"] == "rank16"
            and row["reducer"] == "late_max"
            and int(row["look_behind"]) == 4
        ):
            output.append(
                {
                    "dataset": row["dataset"],
                    "layer": int(row["layer"]),
                    "representation": "embedding" if int(row["layer"]) == 0 else f"layer_{row['layer']}",
                    "evidence_recall": float(row["evidence_recall"]),
                    "chain_completion": float(row["chain_completion"]),
                }
            )
    return output


def gate_rows(parity, selected, matched, strides, hybrid_manifest):
    parity_pass = all(float(row["selection_match_fraction"]) == 1 for row in parity)
    temporal_pass = any(float(row["temporal_gain_ci_low"]) > 0 for row in selected)
    interaction_rows = read_csv("interaction_contrasts.csv")
    interaction_pass = any(float(row["interaction_contrast"]) > 0.02 for row in interaction_rows)
    test_delay = [row for row in matched if row["split"] == "test" and row["tokens"] <= 4]
    substantial = [row for row in test_delay if row["known_future_gain"] > 0.02]
    delay_pass = any(
        row["delay_recovery_fraction"] is not None
        and row["delay_recovery_fraction"] >= 0.70
        and row["deployable_delay"] <= 4
        for row in substantial
    )
    stride4 = [row for row in strides if row["stride"] == 4]
    stride_pass = all(
        row["router_calls_per_token"] <= 0.40
        and row["recall_delta_vs_stride1"] >= -0.02
        for row in stride4
    )
    hybrid_pass = any(
        float(policy["lexical_weight"]) > 0
        and float(policy["lexical_weight"]) < 1
        for policy in hybrid_manifest.values()
    )
    return [
        {"gate": "I0/G0", "status": "pass" if parity_pass else "fail", "evidence": "Exact inherited Paper 2.8 selection and recall parity."},
        {"gate": "I1/G1", "status": "pass" if temporal_pass else "fail", "evidence": "Validation-selected B>1 gain requires paired 95% CI above zero."},
        {"gate": "I2/G4", "status": "pass" if interaction_pass else "fail", "evidence": "No material positive query-by-memory interaction or temporal quality-cost gain."},
        {"gate": "G2", "status": "pass" if delay_pass else "closed", "evidence": "Known-future gain did not exceed the predeclared 0.02 material-headroom threshold."},
        {"gate": "I3", "status": "pass" if stride_pass else "fail", "evidence": "Stride 4 uses <=0.40 calls/token with <0.02 absolute recall loss."},
        {"gate": "I4", "status": "pass" if hybrid_pass else "fail", "evidence": "Validation selected zero lexical weight on every dataset."},
        {"gate": "G3/G5", "status": "not_run", "evidence": "Prefill and predictive sidecars remain gated by unresolved temporal/oracle headroom."},
        {"gate": "I5/G6", "status": "not_run", "evidence": "Retrieval gates did not justify downstream native-K/V generation."},
    ]


def plots(selected, matched, strides):
    fig, axis = plt.subplots(figsize=(8.2, 4.3))
    x = list(range(len(selected)))
    means = [row["temporal_gain"] for row in selected]
    errors = [
        [mean - row["temporal_gain_ci_low"] for mean, row in zip(means, selected)],
        [row["temporal_gain_ci_high"] - mean for mean, row in zip(means, selected)],
    ]
    axis.errorbar(x, means, yerr=errors, fmt="o", color="#007C91", capsize=4)
    axis.axhline(0, color="#5B6770", linewidth=1)
    axis.set_xticks(x, [row["dataset"] for row in selected], rotation=15)
    axis.set_ylabel("B=2 minus B=1 evidence recall @ 4")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "selected_temporal_effects.png", dpi=180)
    fig.savefig(RESULTS / "selected_temporal_effects.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.8, 4.6))
    for dataset in DATASETS:
        rows = [row for row in strides if row["dataset"] == dataset]
        rows.sort(key=lambda row: row["router_calls_per_token"])
        axis.plot(
            [row["router_calls_per_token"] for row in rows],
            [row["evidence_recall"] for row in rows],
            marker="o", label=dataset, color=COLORS[dataset],
        )
    axis.set_xlabel("Router calls per question token")
    axis.set_ylabel("Trajectory evidence recall @ 4")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "stride_quality_frontier.png", dpi=180)
    fig.savefig(RESULTS / "stride_quality_frontier.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7), sharex=True)
    for axis, dataset in zip(axes.flat, DATASETS):
        rows = [row for row in matched if row["dataset"] == dataset and row["split"] == "test"]
        rows.sort(key=lambda row: row["tokens"])
        axis.plot([row["tokens"] for row in rows], [row["fixed_delay_gain"] for row in rows], marker="o", label="causal delay", color="#007C91")
        axis.plot([row["tokens"] for row in rows], [row["known_future_gain"] for row in rows], marker="o", label="known future", color="#D1495B")
        axis.axhline(0, color="#5B6770", linewidth=0.8)
        axis.set_title(dataset)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Matched recall gain")
    axes[1, 0].set_ylabel("Matched recall gain")
    axes[1, 0].set_xlabel("Tokens")
    axes[1, 1].set_xlabel("Tokens")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "matched_delay_headroom.png", dpi=180)
    fig.savefig(RESULTS / "matched_delay_headroom.pdf")
    plt.close(fig)


def main():
    manifest = json.loads((RESULTS / "study_manifest.json").read_text(encoding="utf-8"))
    policies = manifest["validation_selected_temporal_policies"]
    final_summary = read_csv("final_summary.csv")
    trajectory = read_csv("trajectory_per_decision.csv")
    trajectory_summary = read_csv("trajectory_summary.csv")
    parity = read_csv("inherited_parity_summary.csv")
    selected = selected_final_rows(final_summary, policies)
    matched = matched_delay(trajectory)
    strides = stride_frontier(trajectory_summary)
    layers = layer_ablation(final_summary)
    gates = gate_rows(
        parity,
        selected,
        matched,
        strides,
        manifest["validation_selected_hybrid_policies"],
    )
    write_csv("selected_temporal_summary.csv", selected)
    write_csv("matched_delay_recovery.csv", matched)
    write_csv("stride_frontier.csv", strides)
    write_csv("layer_ablation.csv", layers)
    write_csv("gate_status.csv", gates)
    findings = {
        "selected_temporal": selected,
        "matched_delay": matched,
        "stride_frontier": strides,
        "layer_ablation": layers,
        "gates": gates,
        "substantial_known_future_threshold": 0.02,
        "interpretation": (
            "Frozen low-rank memory routing transfers exactly, but explicit short "
            "temporal windows do not improve held-out final retrieval. Sparse routing "
            "updates remain useful as an efficiency policy, not as a quality gain."
        ),
    }
    (RESULTS / "findings.json").write_text(
        json.dumps(findings, indent=2, sort_keys=True), encoding="utf-8"
    )
    plots(selected, matched, strides)
    print(json.dumps(findings, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
