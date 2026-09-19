"""Run a frozen M2/P1 diagnostic on unresolved exact-prefix FULL controls.

This is deliberately not a treatment qualification campaign.  It asks whether
retiring old completed-task history can recover a task that FULL failed from
the same persistent-session prefix.  Such a recovery is diagnostic evidence
for cross-task interference; it is never scored as preservation relative to a
successful FULL control.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .upstream_health import normalize_ollama_cold_start, probe_generation_health


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_action_sha256(output: Path) -> str | None:
    trace = output / "request_selection.jsonl"
    if not trace.is_file():
        return None
    first = next((line for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()), None)
    if first is None:
        return None
    value = json.loads(first).get("assistant_command_sha256")
    return str(value) if value else None


def _option(command: Sequence[str], name: str) -> str | None:
    try:
        index = list(command).index(name)
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise ValueError(f"{name}: missing value")
    return str(command[index + 1])


def _replace_option(command: list[str], name: str, value: str) -> None:
    try:
        index = command.index(name)
    except ValueError:
        command.extend((name, value))
        return
    if index + 1 >= len(command):
        raise ValueError(f"{name}: missing value")
    command[index + 1] = value


def validate_declaration(
    declaration: Mapping[str, Any], state: Mapping[str, Any], *, state_sha256: str,
) -> Mapping[str, Any]:
    if declaration.get("schema_version") != 1:
        raise ValueError("unsupported failed-FULL recovery diagnostic schema")
    if declaration.get("study") != "paper8_5_failed_full_recovery_diagnostic":
        raise ValueError("incorrect diagnostic study")
    source = declaration.get("source_campaign") or {}
    if source.get("campaign_id") != state.get("campaign_id"):
        raise ValueError("source campaign ID changed")
    if source.get("campaign_state_sha256") != state_sha256:
        raise ValueError("source campaign state digest changed")
    cell_id = str(source.get("cell_id") or "")
    cell = (state.get("cells") or {}).get(cell_id)
    if not isinstance(cell, Mapping):
        raise ValueError("declared source cell is missing")
    if cell.get("strategy_id") != "RECENT_FRONTIER_M2_P1_FROZEN":
        raise ValueError("source cell is not the frozen M2/P1 campaign")
    episodes = cell.get("episodes") or {}
    declared = list(declaration.get("episodes") or ())
    if not declared:
        raise ValueError("diagnostic declares no episodes")
    for row in declared:
        episode = episodes.get(row.get("episode_id"))
        if not isinstance(episode, Mapping):
            raise ValueError(f"missing source episode {row.get('episode_id')}")
        if episode.get("instance_id") != row.get("instance_id"):
            raise ValueError("declared instance identity changed")
        control = episode.get("same_prefix_full_control") or {}
        if control.get("status") != "full_control_unresolved":
            raise ValueError("diagnostic source is not an unresolved FULL control")
        if control.get("prefix_sha256") != row.get("prefix_sha256"):
            raise ValueError("declared prefix digest changed")
        command = control.get("command") or ()
        if _option(command, "--policy") != "full":
            raise ValueError("source qualification command was not FULL")
    return cell


def build_diagnostic_command(
    *, source_command: Sequence[str], declaration: Mapping[str, Any],
    output: Path, episode_number: int, instance_id: str,
) -> list[str]:
    command = [str(value) for value in source_command]
    policy = declaration["policy"]
    _replace_option(command, "--output", str(output))
    _replace_option(
        command, "--run-id",
        f"{declaration['diagnostic_id']}__e{episode_number:02d}",
    )
    _replace_option(command, "--pair-id", str(declaration["diagnostic_id"]))
    _replace_option(command, "--instance-id", instance_id)
    _replace_option(command, "--policy", str(policy["policy"]))
    _replace_option(
        command, "--frontier-recent-user-prompts",
        str(policy["frontier_recent_user_prompts"]),
    )
    _replace_option(
        command, "--frontier-protocol-exemplars",
        str(policy["frontier_protocol_exemplars"]),
    )
    _replace_option(
        command, "--frontier-workflow-exemplars",
        str(policy["frontier_workflow_exemplars"]),
    )
    _replace_option(command, "--boundary-mode", str(policy["boundary_mode"]))
    if policy.get("frontier_allow_heuristic") and "--frontier-allow-heuristic" not in command:
        command.append("--frontier-allow-heuristic")
    return command


def qualify_runtime(
    declaration: Mapping[str, Any], command: Sequence[str],
) -> dict[str, Any] | None:
    contract = declaration.get("runtime_qualification")
    if not isinstance(contract, Mapping):
        return None
    base_url = _option(command, "--upstream-base-url")
    model = _option(command, "--served-model")
    if not base_url or not model:
        raise ValueError("diagnostic command lacks upstream/model identity")
    timeout = float(contract.get("timeout_seconds", 600))
    attempts = int(contract.get("connect_attempts", 5))
    retry = float(contract.get("connect_retry_seconds", 5.0))
    curl = str(contract.get("curl_executable") or "") or None
    normalization = normalize_ollama_cold_start(
        base_url=base_url,
        model=model,
        timeout_seconds=timeout,
        connect_attempts=attempts,
        connect_retry_seconds=retry,
        curl_executable=curl,
    )
    if not normalization.get("healthy"):
        return {"healthy": False, "normalization": normalization, "health": None}
    health = probe_generation_health(
        base_url=base_url,
        model=model,
        count=int(contract.get("probe_count", 3)),
        latency_ceiling_seconds=float(contract.get("latency_ceiling_seconds", 60)),
        timeout_seconds=timeout,
        qualification_path=str(contract.get("qualification_path") or "/api/tags"),
        runtime_state_path=str(contract.get("runtime_state_path") or "/api/ps"),
        minimum_active_context_tokens=int(
            contract.get("minimum_active_context_tokens", 131072)
        ),
        connect_attempts=attempts,
        connect_retry_seconds=retry,
        curl_executable=curl,
    )
    return {
        "healthy": bool(normalization.get("healthy") and health.get("healthy")),
        "normalization": normalization,
        "health": health,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    declaration = _read(args.declaration.resolve())
    source_state_path = args.source_state.resolve()
    source_state = _read(source_state_path)
    source_cell = validate_declaration(
        declaration, source_state, state_sha256=_sha256(source_state_path)
    )
    output = args.output.resolve()
    ledger_path = output / "diagnostic_state.json"
    ledger = _read(ledger_path) if ledger_path.is_file() else {
        "schema_version": 1,
        "study": declaration["study"],
        "diagnostic_id": declaration["diagnostic_id"],
        "declaration_sha256": _sha256(args.declaration.resolve()),
        "source_campaign_state_sha256": _sha256(source_state_path),
        "claim_boundary": declaration["claim_boundary"],
        "episodes": {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    episodes = source_cell["episodes"]
    for declared in declaration["episodes"]:
        episode_id = str(declared["episode_id"])
        source_episode = episodes[episode_id]
        control = source_episode["same_prefix_full_control"]
        episode_number = int(episode_id.split("_", 2)[1])
        destination = output / f"episode_{episode_number:02d}_{declared['instance_id'].replace('__', '-')}"
        row = ledger["episodes"].get(episode_id) or {}
        official_path = destination / "official_result.json"
        metrics_path = destination / "autonomous_metrics.json"
        source_output = Path(str(control["output"]))
        source_full_calls = int(control.get("calls") or 0)
        source_full_tokens = int(control.get("cumulative_full_tokens") or 0)
        if official_path.is_file() and metrics_path.is_file():
            official = _read(official_path)
            metrics = _read(metrics_path)
            materialized = int(metrics.get("cumulative_materialized_tokens") or 0)
            full = int(metrics.get("cumulative_full_tokens") or 0)
            row.update({
                "status": "complete",
                "official_resolved": bool(official.get("resolved")),
                "calls": int(metrics.get("calls") or 0),
                "cumulative_full_tokens": full,
                "cumulative_materialized_tokens": materialized,
                "candidate_own_saving_fraction": 1 - materialized / full if full else 0.0,
                "source_full_calls": source_full_calls,
                "source_full_tokens": source_full_tokens,
                "call_delta_vs_failed_full": int(metrics.get("calls") or 0) - source_full_calls,
                "source_first_action_sha256": _first_action_sha256(source_output),
                "candidate_first_action_sha256": _first_action_sha256(destination),
            })
            ledger["episodes"][episode_id] = row
            continue
        command = build_diagnostic_command(
            source_command=control["command"], declaration=declaration,
            output=destination, episode_number=episode_number,
            instance_id=str(declared["instance_id"]),
        )
        row = {
            "instance_id": declared["instance_id"],
            "episode_index": episode_number,
            "diagnostic_role": declared["diagnostic_role"],
            "source_full_resolved": False,
            "source_full_calls": source_full_calls,
            "source_full_tokens": source_full_tokens,
            "prefix_sha256": declared["prefix_sha256"],
            "output": str(destination),
            "command": command,
            "status": "planned" if args.dry_run else "running",
        }
        ledger["episodes"][episode_id] = row
        _write(ledger_path, ledger)
        if args.dry_run:
            continue
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"incomplete diagnostic output is not empty: {destination}")
        qualification = qualify_runtime(declaration, command)
        row["runtime_qualification"] = qualification
        if qualification is not None and not qualification["healthy"]:
            row["status"] = "paused_upstream_unhealthy"
            _write(ledger_path, ledger)
            break
        row["started_at"] = datetime.now(timezone.utc).isoformat()
        _write(ledger_path, ledger)
        process = subprocess.run(command, check=False)
        row["returncode"] = process.returncode
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        if not official_path.is_file() or not metrics_path.is_file():
            row["status"] = "infrastructure_error"
            _write(ledger_path, ledger)
            break
        official = _read(official_path)
        metrics = _read(metrics_path)
        materialized = int(metrics.get("cumulative_materialized_tokens") or 0)
        full = int(metrics.get("cumulative_full_tokens") or 0)
        row.update({
            "status": "complete",
            "official_resolved": bool(official.get("resolved")),
            "calls": int(metrics.get("calls") or 0),
            "cumulative_full_tokens": full,
            "cumulative_materialized_tokens": materialized,
            "candidate_own_saving_fraction": 1 - materialized / full if full else 0.0,
            "call_delta_vs_failed_full": int(metrics.get("calls") or 0) - source_full_calls,
            "source_first_action_sha256": _first_action_sha256(source_output),
            "candidate_first_action_sha256": _first_action_sha256(destination),
        })
        _write(ledger_path, ledger)
    complete = [row for row in ledger["episodes"].values() if row.get("status") == "complete"]
    cross_task = [row for row in complete if row.get("diagnostic_role") == "cross_task_retirement_probe"]
    ledger["summary"] = {
        "completed": len(complete),
        "resolved": sum(bool(row.get("official_resolved")) for row in complete),
        "cross_task_completed": len(cross_task),
        "cross_task_recovered": sum(bool(row.get("official_resolved")) for row in cross_task),
        "empty_prefix_outcome_changed": any(
            row.get("diagnostic_role") == "empty_prefix_repeatability_control"
            and bool(row.get("official_resolved"))
            for row in complete
        ),
        "recovered_call_delta_vs_failed_full": sum(
            int(row.get("call_delta_vs_failed_full") or 0)
            for row in complete if row.get("official_resolved")
        ),
        "interpretation": (
            "Recovery is evidence consistent with removable cross-task interference; "
            "non-recovery is not a selector loss because FULL also failed."
        ),
    }
    ledger["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write(ledger_path, ledger)
    return ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result.get("summary") or {"status": "planned"}))


if __name__ == "__main__":
    main()
