"""Render the Paper 8.5 own-trajectory versus paired-saving figure.

The plot reads only frozen checked-in evidence.  It deliberately keeps the
three-identity discovery cohorts separate from the failed same-prefix
diagnostic and emits a CSV plus source digests beside the figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/papers/shared/results/paper8_5_agent_memory"
OUTPUT = EVIDENCE / "v2_trajectory_economics"


def _load(relative: str) -> tuple[dict, Path]:
    path = EVIDENCE / relative
    return json.loads(path.read_text(encoding="utf-8")), path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows: list[dict[str, object]] = []
    sources: list[Path] = []

    frontier, path = _load("exact_prefix_m2_p1_strict_cohort_v1/evidence.json")
    sources.append(path)
    for task in frontier["tasks"]:
        rows.append({
            "policy": "Recent Frontier",
            "identity": task["instance_id"],
            "own_saving": task["candidate_own_saving_fraction"],
            "paired_saving": task["paired_saving_fraction"],
            "calls": task["candidate_calls"],
            "resolved": task["candidate_resolved"],
            "evidence_class": "qualified_discovery",
        })

    prompt_specs = [
        (
            "heldout_prefix5_cold_cache_v1/summary.json",
            "prompt_pinned_e2_f1c",
            "own_trajectory_materialized_saving_fraction",
            "paired_input_saving_fraction",
        ),
        (
            "heldout_prefix5_django15368_cold_v1/summary.json",
            "arms.prompt_pinned_e2_f1c",
            "own_trajectory_saving_fraction",
            "paired_saving_fraction",
        ),
        (
            "heldout_prefix5_scikit14496_cold_v1/summary.json",
            "arms.prompt_pinned_e2_f1c",
            "own_trajectory_saving_fraction",
            "paired_saving_fraction",
        ),
    ]
    for relative, key_path, own_key, paired_key in prompt_specs:
        payload, path = _load(relative)
        sources.append(path)
        arm: dict = payload
        for key in key_path.split("."):
            arm = arm[key]
        calls = arm["calls"][0] if isinstance(arm["calls"], list) else arm["calls"]
        resolved = (
            arm["official_resolved"][0]
            if isinstance(arm["official_resolved"], list)
            else arm["official_resolved"]
        )
        rows.append({
            "policy": "Prompt Pinned",
            "identity": payload["instance_id"],
            "own_saving": arm[own_key],
            "paired_saving": arm[paired_key],
            "calls": calls,
            "resolved": resolved,
            "evidence_class": "qualified_discovery",
        })

    failed, path = _load("heldout_prefix5_cold_cache_v1/summary.json")
    failed_arm = failed["m2_p1_negative"]
    rows.append({
        "policy": "Recent Frontier",
        "identity": failed["instance_id"],
        "own_saving": failed_arm["own_trajectory_materialized_saving_fraction"],
        "paired_saving": failed_arm["paired_input_saving_fraction"],
        "calls": failed_arm["calls"],
        "resolved": failed_arm["official_resolved"],
        "evidence_class": "failed_diagnostic",
    })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (OUTPUT / "trajectory_economics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "study": "paper8_5_v2_trajectory_economics_figure",
        "rows": rows,
        "sources": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": _digest(path)}
            for path in sorted(set(sources))
        ],
        "claim_boundary": (
            "Discovery and failed diagnostic points only; no held-out confirmation "
            "accuracy is inferred. Marker size is policy calls."
        ),
    }
    (OUTPUT / "evidence.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    colors = {"Recent Frontier": "#1f77b4", "Prompt Pinned": "#d95f02"}
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for row in rows:
        x = 100.0 * float(row["own_saving"])
        y = 100.0 * float(row["paired_saving"])
        size = 24.0 + 7.0 * int(row["calls"])
        if row["resolved"]:
            axis.scatter(x, y, s=size, marker="o", color=colors[str(row["policy"])],
                         alpha=0.80, edgecolor="black", linewidth=0.6)
        else:
            axis.scatter(x, y, s=size, marker="X", color="#b2182b",
                         edgecolor="black", linewidth=0.7)
        short = str(row["identity"]).split("__")[-1]
        axis.annotate(short, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)

    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.plot([-5, 90], [-5, 90], linestyle="--", color="#999999", linewidth=0.8,
              label="paired = own")
    axis.set_xlim(-3, 85)
    axis.set_ylim(-5, 90)
    axis.set_xlabel("Own-trajectory history saving (%)")
    axis.set_ylabel("Paired workload saving (%)")
    axis.grid(True, alpha=0.22)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors["Recent Frontier"],
                   markeredgecolor="black", label="Recent Frontier: resolved"),
        plt.Line2D([], [], marker="o", linestyle="", color=colors["Prompt Pinned"],
                   markeredgecolor="black", label="Prompt Pinned: resolved"),
        plt.Line2D([], [], marker="X", linestyle="", color="#b2182b",
                   markeredgecolor="black", label="failed diagnostic"),
    ]
    axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT / "trajectory_economics.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "trajectory_economics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
