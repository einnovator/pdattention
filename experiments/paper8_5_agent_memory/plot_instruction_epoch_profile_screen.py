"""Plot N-task logical saving for instruction-epoch progress-spine profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _label(configuration: dict[str, Any]) -> str:
    full = int(configuration["prior_full_epochs"])
    if full:
        return f"E{full}: keep {full} prior epoch{'s' if full != 1 else ''}"
    return (
        "E0"
        f" + R{configuration['prior_recent_turns']}"
        f"/M{configuration['prior_mutation_turns']}"
        f"/V{configuration['prior_verification_turns']}"
        f"/P{configuration['prior_protocol_turns']}"
    )


def profile_screen(paths: list[Path]) -> dict[str, Any]:
    rows = []
    locked_instances = None
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if artifact.get("study") != "paper8_5_instruction_epoch_fixed_trajectory_oracle":
            raise ValueError(f"{path}: unexpected study")
        instances = tuple(artifact["instance_ids"])
        if locked_instances is None:
            locked_instances = instances
        elif instances != locked_instances:
            raise ValueError("profile artifacts use different ordered task cohorts")
        config = dict(artifact["selector_configuration"])
        rows.append({
            "profile": path.stem,
            "label": _label(config),
            "configuration": config,
            "points": [
                {
                    "issue_count": int(point["issue_count"]),
                    "gross_saving_fraction": float(point["gross_saving_fraction"]),
                    "assistant_tool_saving_fraction": float(
                        point["assistant_tool_saving_fraction"]
                    ),
                }
                for point in artifact["points"]
            ],
        })
    return {
        "schema_version": 1,
        "study": "paper8_5_instruction_epoch_profile_opportunity_screen",
        "claim_scope": "fixed-trajectory logical opportunity; no quality claim",
        "instance_ids": list(locked_instances or ()),
        "profiles": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = profile_screen(args.profile)

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.axhspan(30, 50, color="#dff0d8", alpha=0.65, label="30--50% target")
    for row in summary["profiles"]:
        axis.plot(
            [point["issue_count"] for point in row["points"]],
            [100 * point["gross_saving_fraction"] for point in row["points"]],
            marker="o",
            linewidth=1.8,
            label=row["label"],
        )
    axis.set_xlabel("Issues accumulated in one session (N)")
    axis.set_ylabel("Fixed-trajectory gross input saving (%)")
    axis.set_ylim(0, 55)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
