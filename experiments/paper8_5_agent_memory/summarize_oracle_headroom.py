"""Reduce frozen oracle omission artifacts into auditable headroom tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .oracle_headroom_queue import _trial


def reduce_trials(
    artifacts: Iterable[tuple[Mapping[str, Any], str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trials = [_trial(artifact, source) for artifact, source in artifacts]
    rows: list[dict[str, Any]] = []
    for trial in trials:
        full = int(trial["full_history_tokens"])
        omitted = int(trial["omitted_tokens"])
        rows.append({
            **trial,
            "groups": ";".join(trial["groups"]),
            "omitted_fraction": omitted / full if full else 0.0,
        })
    rows.sort(key=lambda row: (
        row["instance_id"], row["decision"], row["depth"], row["groups"]
    ))
    summaries: list[dict[str, Any]] = []
    keys = sorted({
        (row["instance_id"], row["decision"], row["depth"])
        for row in rows
    })
    for instance_id, decision, depth in keys:
        cohort = [
            row for row in rows
            if (row["instance_id"], row["decision"], row["depth"])
            == (instance_id, decision, depth)
        ]
        valid = [row for row in cohort if row["action_valid"]]
        exact = [row for row in valid if row["exact_command"]]
        semantic = [
            row for row in valid if row["semantic_transition_equivalent"]
        ]
        summaries.append({
            "instance_id": instance_id,
            "decision": decision,
            "depth": depth,
            "trials": len(cohort),
            "valid_actions": len(valid),
            "exact_commands": len(exact),
            "semantic_transitions": len(semantic),
            "exact_command_rate": len(exact) / len(cohort) if cohort else 0.0,
            "semantic_transition_rate": (
                len(semantic) / len(cohort) if cohort else 0.0
            ),
            "max_exact_omitted_tokens": max(
                (row["omitted_tokens"] for row in exact), default=0
            ),
            "max_exact_omitted_fraction": max(
                (row["omitted_fraction"] for row in exact), default=0.0
            ),
            "max_semantic_omitted_tokens": max(
                (row["omitted_tokens"] for row in semantic), default=0
            ),
            "max_semantic_omitted_fraction": max(
                (row["omitted_fraction"] for row in semantic), default=0.0
            ),
        })
    return rows, summaries


def _load_artifacts(directory: Path) -> list[tuple[Mapping[str, Any], str]]:
    artifacts = []
    for path in sorted(directory.rglob("*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if artifact.get("study") != "paper8_5_agent_memory_frozen_replay":
            continue
        omission = artifact.get("run_configuration", {}).get(
            "oracle_omit_causal_group_ids"
        )
        if omission:
            artifacts.append((artifact, str(path.relative_to(directory))))
    return artifacts


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    rows, summaries = reduce_trials(_load_artifacts(args.input_directory))
    args.output_directory.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_directory / "oracle_trials.csv", rows)
    _write_csv(args.output_directory / "oracle_summary.csv", summaries)
    payload = {
        "schema_version": 1,
        "study": "paper8_5_oracle_headroom_summary",
        "evidence_class": "offline_oracle_not_deployable_policy_quality",
        "trial_count": len(rows),
        "cohorts": summaries,
    }
    (args.output_directory / "oracle_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"trial_count": len(rows), "cohorts": len(summaries)}))


if __name__ == "__main__":
    main()
