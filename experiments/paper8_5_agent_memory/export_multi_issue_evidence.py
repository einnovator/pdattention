"""Export compact, audited evidence from an autonomous multi-issue campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .adaptive_spine_bracket import paired_point
from .run_autonomous_multi_issue_campaign import _divergence_accounting


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace(output: Path) -> list[dict[str, Any]]:
    path = output / "request_selection.jsonl"
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode_summary(episode: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(str(episode["output"]))
    trace = _trace(output)
    indexes = [int(row["request_index"]) for row in trace]
    contiguous = indexes == list(range(1, len(indexes) + 1))
    upstream_errors = sum(
        int(row.get("upstream_status") or 0) < 100
        or int(row.get("upstream_status") or 0) >= 400
        for row in trace
    )
    manifest = _read(output / "run_manifest.json")
    official = _read(output / "official_result.json")
    full_tokens = int(
        episode.get("cumulative_full_tokens")
        or sum(int(row.get("full_tokens") or 0) for row in trace)
    )
    materialized_tokens = int(
        episode.get("cumulative_materialized_tokens")
        or sum(
            int(row.get("materialized_tokens", row.get("selected_tokens", 0)) or 0)
            for row in trace
        )
    )
    identities = {
        key: manifest.get(key) for key in (
            "model_revision", "tokenizer_revision", "harness_version_observed",
            "agent_behavior_sha256", "scaffold_identity_sha256",
            "workspace_source_identity_sha256", "temperature", "top_p", "seed",
            "max_completion_tokens",
        )
    }
    artifacts = {}
    for name in (
        "run_manifest.json", "request_selection.jsonl", "official_result.json",
        "persistent_episode_export.json",
    ):
        path = output / name
        if path.is_file():
            artifacts[name] = _sha256(path)
    return {
        "instance_id": episode["instance_id"],
        "output": str(output),
        "calls": len(trace),
        "cumulative_full_tokens": full_tokens,
        "cumulative_materialized_tokens": materialized_tokens,
        "candidate_trajectory_gross_saving_fraction": (
            1 - materialized_tokens / full_tokens if full_tokens else 0.0
        ),
        "official_resolved": official.get("resolved"),
        "official_error": official.get("error"),
        "trace_indices_contiguous": contiguous,
        "upstream_error_calls": upstream_errors,
        "evidence_admissible": (
            contiguous and upstream_errors == 0
            and official.get("official_grader") is True
            and isinstance(official.get("resolved"), bool)
        ),
        "identities": identities,
        "artifact_sha256": artifacts,
    }


def build(campaign_root: Path) -> dict[str, Any]:
    state_path = campaign_root / "campaign_state.json"
    state = _read(state_path)
    cells = state.get("cells") or {}
    controls: dict[tuple[str, int], tuple[str, Mapping[str, Any]]] = {}
    for control_id, control in cells.items():
        if (
            control.get("strategy_id") != "S01_persistent_full"
            or control.get("status") != "complete"
        ):
            continue
        key = (str(control["sequence_id"]), int(control["repeat"]))
        if key in controls:
            raise ValueError(
                "ambiguous persistent-FULL controls for "
                f"sequence={key[0]!r}, repeat={key[1]}"
            )
        controls[key] = (str(control_id), control)
    runs: list[dict[str, Any]] = []
    exportable_statuses = {
        "complete",
        "stopped_predeclared_quality_gate",
        "stopped_causal_attribution_gate",
    }
    for cell_id, cell in sorted(cells.items()):
        if cell.get("status") not in exportable_statuses:
            continue
        completed_episode_rows = (
            list(cell["episodes"].values())
            if cell.get("status") == "complete"
            else [
                row for row in cell["episodes"].values()
                if row.get("status") == "complete"
            ]
        )
        episodes = [_episode_summary(row) for row in completed_episode_rows]
        observed_issue_count = len(episodes)
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "sequence_id": cell["sequence_id"],
            "repeat": int(cell["repeat"]),
            "strategy_id": cell["strategy_id"],
            "strategy_config_id": cell.get("strategy_config_id", "default"),
            "official_resolved_count": int(cell["official_resolved_count"]),
            "issue_count": observed_issue_count,
            "planned_issue_count": int(cell["issue_count"]),
            "campaign_cell_status": cell.get("status"),
            "stop_gate": cell.get("stop_gate"),
            "calls": int(cell["calls"]),
            "cumulative_full_tokens": int(cell["cumulative_full_tokens"]),
            "cumulative_materialized_tokens": int(
                cell["cumulative_materialized_tokens"]
            ),
            "candidate_trajectory_gross_saving_fraction": (
                1 - int(cell["cumulative_materialized_tokens"])
                / int(cell["cumulative_full_tokens"])
            ),
            "episodes": episodes,
            "evidence_admissible": all(e["evidence_admissible"] for e in episodes),
        }
        if cell["strategy_id"] not in {"S00_fresh_full", "S01_persistent_full"}:
            control_entry = controls.get(
                (str(cell["sequence_id"]), int(cell["repeat"]))
            )
            if control_entry is not None:
                control_id, control = control_entry
                row["paired_persistent_full_cell_id"] = control_id
                divergences = []
                paired_issues = []
                control_summaries = [
                    _episode_summary(episode)
                    for episode in list(control["episodes"].values())[:observed_issue_count]
                ]
                if cell.get("status") == "complete":
                    row["paired"] = paired_point(cell, control)
                else:
                    baseline_tokens = sum(
                        int(item["cumulative_full_tokens"])
                        for item in control_summaries
                    )
                    lost = sum(
                        bool(base["official_resolved"])
                        and not bool(test["official_resolved"])
                        for test, base in zip(episodes, control_summaries)
                    )
                    raw = (
                        1 - int(cell["cumulative_materialized_tokens"])
                        / baseline_tokens
                        if baseline_tokens else 0.0
                    )
                    row["paired"] = {
                        "raw_saving_vs_persistent_full": raw,
                        "failure_aware_saving_vs_persistent_full": (
                            0.0 if lost else raw
                        ),
                        "lost_persistent_full_successes": lost,
                        "resolution_delta_vs_persistent_full": (
                            int(cell["official_resolved_count"])
                            - sum(bool(item["official_resolved"])
                                  for item in control_summaries)
                        ),
                        "partial_stopped_prefix": True,
                    }
                row["calls_delta_vs_persistent_full"] = (
                    int(cell["calls"])
                    - sum(int(item["calls"]) for item in control_summaries)
                )
                for candidate_episode, control_episode, candidate_summary, control_summary in zip(
                    completed_episode_rows,
                    list(control["episodes"].values())[:observed_issue_count],
                    episodes, control_summaries,
                ):
                    divergence = _divergence_accounting(
                        _trace(Path(str(candidate_episode["output"]))),
                        _trace(Path(str(control_episode["output"]))),
                    )
                    divergences.append(divergence)
                    baseline_tokens = int(control_summary["cumulative_full_tokens"])
                    candidate_tokens = int(
                        candidate_summary["cumulative_materialized_tokens"]
                    )
                    raw_saving = (
                        1 - candidate_tokens / baseline_tokens
                        if baseline_tokens else 0.0
                    )
                    lost_success = bool(
                        control_summary["official_resolved"]
                        and not candidate_summary["official_resolved"]
                    )
                    paired_issues.append({
                        "instance_id": candidate_summary["instance_id"],
                        "candidate_official_resolved": bool(
                            candidate_summary["official_resolved"]
                        ),
                        "control_official_resolved": bool(
                            control_summary["official_resolved"]
                        ),
                        "joint_success": bool(
                            candidate_summary["official_resolved"]
                            and control_summary["official_resolved"]
                        ),
                        "lost_persistent_full_success": lost_success,
                        "candidate_calls": int(candidate_summary["calls"]),
                        "control_calls": int(control_summary["calls"]),
                        "calls_delta_vs_persistent_full": (
                            int(candidate_summary["calls"])
                            - int(control_summary["calls"])
                        ),
                        "candidate_full_tokens": int(
                            candidate_summary["cumulative_full_tokens"]
                        ),
                        "candidate_materialized_tokens": candidate_tokens,
                        "control_full_tokens": baseline_tokens,
                        "raw_saving_vs_persistent_full": raw_saving,
                        "failure_aware_saving_vs_persistent_full": (
                            0.0 if lost_success else raw_saving
                        ),
                        "candidate_trajectory_gross_saving_fraction": (
                            candidate_summary[
                                "candidate_trajectory_gross_saving_fraction"
                            ]
                        ),
                        "evidence_admissible": bool(
                            candidate_summary["evidence_admissible"]
                            and control_summary["evidence_admissible"]
                        ),
                        **divergence,
                    })
                row["episode_divergence"] = divergences
                row["paired_issues"] = paired_issues
                first_candidate = Path(str(completed_episode_rows[0]["output"]))
                first_control = Path(str(next(iter(control["episodes"].values()))["output"]))
                row["shared_first_episode_exact"] = (
                    first_candidate.resolve() == first_control.resolve()
                    and _sha256(first_candidate / "request_selection.jsonl")
                    == _sha256(first_control / "request_selection.jsonl")
                )
        runs.append(row)
    return {
        "schema_version": 1,
        "study": "paper8_5_autonomous_multi_issue_evidence",
        "campaign_id": state["campaign_id"],
        "campaign_state_sha256": _sha256(state_path),
        "runs": runs,
    }


def write_bundle(evidence: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence.json").write_text(
        json.dumps(dict(evidence), indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "cell_id", "sequence_id", "repeat", "strategy_id", "strategy_config_id",
        "official_resolved_count", "issue_count", "calls",
        "cumulative_full_tokens", "cumulative_materialized_tokens",
        "candidate_trajectory_gross_saving_fraction",
        "failure_aware_saving_vs_persistent_full",
        "calls_delta_vs_persistent_full", "evidence_admissible",
    ]
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in evidence["runs"]:
            paired = run.get("paired") or {}
            writer.writerow({
                **{key: run.get(key) for key in fields},
                "failure_aware_saving_vs_persistent_full": paired.get(
                    "failure_aware_saving_vs_persistent_full"
                ),
            })
    issue_fields = [
        "sequence_id", "repeat", "strategy_id", "strategy_config_id",
        "instance_id", "candidate_official_resolved",
        "control_official_resolved", "joint_success",
        "lost_persistent_full_success", "candidate_calls", "control_calls",
        "calls_delta_vs_persistent_full", "candidate_full_tokens",
        "candidate_materialized_tokens", "control_full_tokens",
        "candidate_trajectory_gross_saving_fraction",
        "raw_saving_vs_persistent_full",
        "failure_aware_saving_vs_persistent_full",
        "first_action_diverged", "first_action_divergence_request",
        "identical_input_first_action_divergence",
        "selection_active_at_first_action_divergence",
        "selected_tokens_before_divergence_or_terminal",
        "full_tokens_before_divergence_or_terminal", "evidence_admissible",
    ]
    with (output / "issues.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=issue_fields)
        writer.writeheader()
        for run in evidence["runs"]:
            for issue in run.get("paired_issues") or ():
                writer.writerow({
                    "sequence_id": run["sequence_id"],
                    "repeat": run["repeat"],
                    "strategy_id": run["strategy_id"],
                    "strategy_config_id": run["strategy_config_id"],
                    **{key: issue.get(key) for key in issue_fields[4:]},
                })
    lines = [
        "# Autonomous persistent-session evidence", "",
        f"Campaign: `{evidence['campaign_id']}`", "",
        "| Sequence | Repeat | Strategy | Resolved | Calls | Candidate gross | Paired failure-aware | Call delta | Admissible |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for run in evidence["runs"]:
        paired = run.get("paired") or {}
        failure_aware = paired.get("failure_aware_saving_vs_persistent_full")
        lines.append(
            f"| {run['sequence_id']} | {run['repeat']} | "
            f"{run['strategy_id']}/{run['strategy_config_id']} | "
            f"{run['official_resolved_count']}/{run['issue_count']} | {run['calls']} | "
            f"{run['candidate_trajectory_gross_saving_fraction']:.2%} | "
            f"{failure_aware:.2%} | {run.get('calls_delta_vs_persistent_full')} | "
            f"{run['evidence_admissible']} |"
            if failure_aware is not None else
            f"| {run['sequence_id']} | {run['repeat']} | "
            f"{run['strategy_id']}/{run['strategy_config_id']} | "
            f"{run['official_resolved_count']}/{run['issue_count']} | {run['calls']} | "
            f"{run['candidate_trajectory_gross_saving_fraction']:.2%} | -- | -- | "
            f"{run['evidence_admissible']} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = build(args.campaign_root.resolve())
    write_bundle(evidence, args.output.resolve())
    print(json.dumps({"campaign_id": evidence["campaign_id"], "runs": len(evidence["runs"])}))


if __name__ == "__main__":
    main()
