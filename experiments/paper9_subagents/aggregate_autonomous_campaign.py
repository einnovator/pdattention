"""Aggregate independently checkpointed Paper 9 autonomous campaign seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


CAMPAIGN_SOURCE_FILES = (
    "experiments/paper9_subagents/run_autonomous_repository_campaign.py",
    "src/pra_hf/subagent_context.py",
    "src/pra_hf/subagent_harness.py",
    "src/pra_hf/subagent_routing.py",
    "src/pra_hf/subagent_mlx_native.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plot(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    order = (
        "sequential_isolated",
        "parallel_isolated",
        "sequential_completed",
        "parallel_completed",
    )
    labels = ("seq. isolated", "par. isolated", "seq. completed", "par. completed")
    accuracy = [summary["conditions"][name]["path_accuracy"]["mean"] for name in order]
    accuracy_error = [
        summary["conditions"][name]["path_accuracy"]["stdev"] for name in order
    ]
    wall = [summary["conditions"][name]["wall_seconds"]["mean"] for name in order]
    wall_error = [summary["conditions"][name]["wall_seconds"]["stdev"] for name in order]
    colors = ("#68747d", "#24796b", "#b8872d", "#255b96")
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    axes[0].bar(labels, accuracy, yerr=accuracy_error, capsize=3, color=colors)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact source-path accuracy")
    axes[1].bar(labels, wall, yerr=wall_error, capsize=3, color=colors)
    axes[1].set_ylabel("Campaign wall time (seconds)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output / "autonomous_campaign.pdf", bbox_inches="tight")
    figure.savefig(output / "autonomous_campaign.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--host-model", required=True)
    parser.add_argument("--ram-bytes", type=int, required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    summaries = [
        json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for path in args.inputs
    ]
    identity = {
        (summary["protocol"], summary["model"], tuple(summary["task_ids"]))
        for summary in summaries
    }
    if len(identity) != 1:
        raise ValueError("Campaign inputs do not share protocol, model, and task cohort.")
    condition_rows = [row for summary in summaries for row in summary["conditions"]]
    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_seed: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in condition_rows:
        by_condition[str(row["condition"])].append(row)
        by_seed[int(row["seed"])][str(row["condition"])] = row

    metrics = (
        "wall_seconds",
        "path_accuracy",
        "model_calls",
        "prompt_tokens",
        "generated_tokens",
        "logical_tool_calls",
        "physical_tool_calls",
        "reused_tool_calls",
        "routed_peer_records",
        "routed_peer_tokens",
    )
    aggregates = {}
    for condition, rows in by_condition.items():
        aggregates[condition] = {"seeds": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            aggregates[condition][metric] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

    paired = []
    for seed, rows in sorted(by_seed.items()):
        if set(rows) != set(by_condition):
            raise ValueError(f"Seed {seed} has an incomplete condition matrix.")
        paired.append(
            {
                "seed": seed,
                "isolated_parallel_speedup": rows["sequential_isolated"]["wall_seconds"]
                / rows["parallel_isolated"]["wall_seconds"],
                "completed_parallel_speedup": rows["sequential_completed"]["wall_seconds"]
                / rows["parallel_completed"]["wall_seconds"],
                "sequential_context_accuracy_delta": rows["sequential_completed"]["path_accuracy"]
                - rows["sequential_isolated"]["path_accuracy"],
                "sequential_context_physical_call_delta": rows["sequential_completed"][
                    "physical_tool_calls"
                ]
                - rows["sequential_isolated"]["physical_tool_calls"],
            }
        )
    paired_summary = {}
    for metric in (
        "isolated_parallel_speedup",
        "completed_parallel_speedup",
        "sequential_context_accuracy_delta",
        "sequential_context_physical_call_delta",
    ):
        values = [float(row[metric]) for row in paired]
        paired_summary[metric] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    first = summaries[0]
    final = {
        "protocol": "paper9-autonomous-repository-v1-aggregate",
        "evidence_scope": first["evidence_scope"],
        "agent_runs": sum(int(row["tasks"]) for row in condition_rows),
        "execution_failures": sum(int(row["failures"]) for row in condition_rows),
        "model": first["model"],
        "model_digest": args.model_digest,
        "engine": "ollama",
        "engine_version": args.engine_version,
        "host": args.host,
        "host_model": args.host_model,
        "ram_bytes": args.ram_bytes,
        "repository_revision": args.repository_revision,
        "source_hashes": {
            relative: _sha256(args.repo / relative)
            for relative in CAMPAIGN_SOURCE_FILES
        },
        "task_ids": first["task_ids"],
        "seeds": sorted(by_seed),
        "conditions": aggregates,
        "paired_seed_results": paired,
        "paired_summary": paired_summary,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(final, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "agent_rows.jsonl").open("w", encoding="utf-8") as target:
        for path in args.inputs:
            target.write((path / "agent_rows.jsonl").read_text(encoding="utf-8"))
    _plot(final, args.output)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
