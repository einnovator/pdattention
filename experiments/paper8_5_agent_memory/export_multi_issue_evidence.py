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
        "official_resolved": official.get("resolved"),
        "official_error": official.get("error"),
        "trace_indices_contiguous": contiguous,
        "upstream_error_calls": upstream_errors,
        "evidence_admissible": (
            contiguous and upstream_errors == 0
            and isinstance(official.get("resolved"), bool)
            and not bool(official.get("error"))
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
    for cell_id, cell in sorted(cells.items()):
        if cell.get("status") != "complete":
            continue
        episodes = [_episode_summary(row) for row in cell["episodes"].values()]
        row: dict[str, Any] = {
            "cell_id": cell_id,
            "sequence_id": cell["sequence_id"],
            "repeat": int(cell["repeat"]),
            "strategy_id": cell["strategy_id"],
            "strategy_config_id": cell.get("strategy_config_id", "default"),
            "official_resolved_count": int(cell["official_resolved_count"]),
            "issue_count": int(cell["issue_count"]),
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
                row["paired"] = paired_point(cell, control)
                row["calls_delta_vs_persistent_full"] = (
                    int(cell["calls"]) - int(control["calls"])
                )
                divergences = []
                for candidate_episode, control_episode in zip(
                    cell["episodes"].values(), control["episodes"].values()
                ):
                    divergences.append(_divergence_accounting(
                        _trace(Path(str(candidate_episode["output"]))),
                        _trace(Path(str(control_episode["output"]))),
                    ))
                row["episode_divergence"] = divergences
                first_candidate = Path(str(next(iter(cell["episodes"].values()))["output"]))
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
