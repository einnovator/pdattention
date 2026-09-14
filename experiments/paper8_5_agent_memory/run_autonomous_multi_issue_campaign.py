"""Run locked fresh-versus-persistent Paper 8.5 autonomous sequences.

Each issue still receives its clean SWE-bench workspace.  In persistent mode,
the completed model-visible trajectory is prepended to the next issue under one
stable logical session identity.  This separates conversation persistence from
workspace persistence and makes both semantics auditable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .run_autonomous_curve_campaign import _completed_result
from .run_autonomous_swebench import load_locked_task


TARGET_SAVING_MIN = 0.30
TARGET_SAVING_MAX = 0.50


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


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _resolve(spec_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (spec_path.parent / path).resolve()


def _slug(value: str) -> str:
    return value.replace("__", "-").replace("/", "-")


def validate_spec(spec: Mapping[str, Any], benchmark: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported autonomous multi-issue campaign schema")
    target = spec.get("primary_target") or {}
    if float(target.get("minimum_saving_fraction", -1)) != TARGET_SAVING_MIN:
        raise ValueError("primary target minimum must be frozen at 0.30")
    if float(target.get("maximum_saving_fraction", -1)) != TARGET_SAVING_MAX:
        raise ValueError("primary target maximum must be frozen at 0.50")
    if target.get("official_resolution_delta_minimum") != 0:
        raise ValueError("discovery target requires a nonnegative resolution delta")
    if target.get("lost_persistent_full_successes") != 0:
        raise ValueError("discovery target permits no lost persistent-FULL success")
    generation = spec.get("generation") or {}
    max_completion = generation.get("max_completion_tokens")
    if isinstance(max_completion, bool) or not isinstance(max_completion, int) or max_completion < 1:
        raise ValueError("max_completion_tokens must be a positive frozen integer")
    known = list(benchmark.get("instance_ids") or ())
    if not known or len(known) != len(set(known)):
        raise ValueError("benchmark card requires unique locked instance IDs")
    sequence_ids: set[str] = set()
    for sequence in spec.get("sequences") or ():
        sequence_id = str(sequence.get("sequence_id") or "")
        instance_ids = list(sequence.get("instance_ids") or ())
        if not sequence_id or sequence_id in sequence_ids:
            raise ValueError("sequence IDs must be distinct and non-empty")
        sequence_ids.add(sequence_id)
        if not instance_ids or len(instance_ids) > 5:
            raise ValueError(f"{sequence_id}: expected one to five issues")
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError(f"{sequence_id}: duplicate issue identity")
        if any(instance_id not in known for instance_id in instance_ids):
            raise ValueError(f"{sequence_id}: issue is outside the locked benchmark")
    strategy_cells: set[tuple[str, str]] = set()
    for strategy in spec.get("strategies") or ():
        strategy_id = str(strategy.get("strategy_id") or "")
        config_id = str(strategy.get("strategy_config_id") or "default")
        cell_identity = (strategy_id, config_id)
        if not strategy_id or cell_identity in strategy_cells:
            raise ValueError("strategy/config identities must be distinct and non-empty")
        strategy_cells.add(cell_identity)
        mode = strategy.get("session_mode")
        if mode not in {"fresh_per_issue", "persistent"}:
            raise ValueError(f"{strategy_id}: invalid session mode")
        if mode == "fresh_per_issue" and strategy.get("policy") != "full":
            raise ValueError("fresh-per-issue is a FULL control, not a selection arm")


def campaign_cells(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for sequence in spec.get("sequences") or ():
        for strategy in spec.get("strategies") or ():
            repeats = int(strategy.get("repeats", 1))
            if repeats < 1:
                raise ValueError("strategy repeats must be positive")
            config_id = str(strategy.get("strategy_config_id") or "default")
            strategy_cell = str(strategy["strategy_id"])
            if config_id != "default":
                strategy_cell += f"-{config_id}"
            for repeat in range(1, repeats + 1):
                cell_id = (
                    f"{sequence['sequence_id']}__{strategy_cell}"
                    f"__r{repeat:02d}"
                )
                cells.append({
                    "cell_id": cell_id,
                    "sequence_id": sequence["sequence_id"],
                    "sequence_stratum": sequence["sequence_stratum"],
                    "instance_ids": list(sequence["instance_ids"]),
                    "issue_count": len(sequence["instance_ids"]),
                    "strategy_id": strategy["strategy_id"],
                    "strategy_config_id": config_id,
                    "session_mode": strategy["session_mode"],
                    "policy": strategy["policy"],
                    "repeat": repeat,
                    "strategy": dict(strategy),
                })
    if len({row["cell_id"] for row in cells}) != len(cells):
        raise AssertionError("campaign cell identity collision")
    return cells


def _episode_command(
    *, spec: Mapping[str, Any], benchmark: Path, cell: Mapping[str, Any],
    instance_id: str, episode_number: int, output: Path, prefix: Path | None,
    session_id: str, args: argparse.Namespace,
) -> list[str]:
    generation = spec["generation"]
    history = spec.get("history") or {}
    strategy = cell["strategy"]
    command = [
        sys.executable, "-m", "experiments.paper8_5_agent_memory.run_autonomous_swebench",
        "--benchmark-card", str(benchmark),
        "--instance-id", instance_id,
        "--output", str(output),
        "--run-id", f"{cell['cell_id']}__e{episode_number:02d}",
        "--pair-id", f"{spec['campaign_id']}:{cell['sequence_id']}:r{cell['repeat']}",
        "--upstream-base-url", args.upstream_base_url,
        "--model", str(spec["model"]),
        "--served-model", str(spec["served_model"]),
        "--model-revision", str(spec["model_revision"]),
        "--tokenizer", args.tokenizer,
        "--tokenizer-revision", str(spec["tokenizer_revision"]),
        "--policy", str(cell["policy"]),
        "--budget-fraction", str(strategy.get("budget_fraction", 1.0)),
        "--negative-realization", str(strategy.get("negative_realization", "drop")),
        "--negative-fallback", str(strategy.get("negative_fallback", "none")),
        "--head", str(history.get("head_turns", 2)),
        "--tail", str(history.get("tail_turns", 4)),
        "--search-delay-turns", str(history.get("search_delay_turns", 8)),
        "--write-delay-turns", str(history.get("write_delay_turns", 1)),
        "--same-span-reads-to-keep", str(history.get("same_span_reads_to_keep", 2)),
        "--working-set-resources", str(history.get("working_set_resources", 4)),
        "--completed-recent-turns", str(strategy.get("completed_recent_turns", 1)),
        "--completed-mutation-turns", str(strategy.get("completed_mutation_turns", 1)),
        "--completed-verification-turns", str(strategy.get("completed_verification_turns", 1)),
        "--temperature", str(generation["temperature"]),
        "--top-p", str(generation["top_p"]),
        "--seed", str(generation["seed"]),
        "--max-calls", str(generation["max_calls"]),
        "--max-completion-tokens", str(generation["max_completion_tokens"]),
    ]
    if cell["session_mode"] == "persistent":
        command.extend((
            "--session-id", session_id,
            "--episode-index", str(episode_number),
        ))
        if prefix is not None:
            command.extend(("--persistent-prefix", str(prefix)))
    if not strategy.get("keep_completed_task_statements", True):
        command.append("--no-keep-completed-task-statements")
    option_map = {
        "materialization_mode": "--materialization-mode",
        "materialization_threshold_tokens": "--materialization-threshold-tokens",
        "materialization_head_lines": "--materialization-head-lines",
        "materialization_tail_lines": "--materialization-tail-lines",
        "materialization_match_context_lines": "--materialization-match-context-lines",
        "materialization_max_matched_lines": "--materialization-max-matched-lines",
    }
    for key, option in option_map.items():
        if key in strategy:
            command.extend((option, str(strategy[key])))
    if args.docker_executable:
        command.extend(("--docker-executable", args.docker_executable))
    if args.docker_platform:
        command.extend(("--docker-platform", args.docker_platform))
    for value in args.pythonpath:
        command.extend(("--pythonpath", value))
    if args.skip_grading:
        command.append("--skip-grading")
    if args.grade_auxiliary_workspace_state:
        command.append("--grade-auxiliary-workspace-state")
    return command


def _aggregate(cell: Mapping[str, Any], episode_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(episode_rows) != int(cell["issue_count"]):
        raise ValueError("cannot aggregate an incomplete ordered issue sequence")
    full = sum(int(row["cumulative_full_tokens"]) for row in episode_rows)
    materialized = sum(int(row["cumulative_materialized_tokens"]) for row in episode_rows)
    official = [bool(row["official_resolved"]) for row in episode_rows]
    saving = 1 - materialized / full if full else 0.0
    return {
        "status": "complete",
        "official_resolved_count": sum(official),
        "official_resolution_fraction": sum(official) / len(official),
        "all_issues_resolved": all(official),
        "calls": sum(int(row["calls"]) for row in episode_rows),
        "cumulative_full_tokens": full,
        "cumulative_materialized_tokens": materialized,
        "failure_aware_saving_fraction": saving,
        "in_primary_saving_target": TARGET_SAVING_MIN <= saving <= TARGET_SAVING_MAX,
        "candidate_all_issues_resolved": all(official),
        "discovery_quality_target_met": None,
        "primary_target_met": None,
        "pairing_status": (
            "requires contemporaneous persistent-FULL sequence comparison"
        ),
    }


def _trace(path: Path) -> list[dict[str, Any]]:
    trace_path = path / "request_selection.jsonl"
    if not trace_path.is_file():
        raise ValueError(f"missing autonomous request trace: {trace_path}")
    return [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _divergence_accounting(
    candidate: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_by_index = {int(row["request_index"]): row for row in candidate}
    baseline_by_index = {int(row["request_index"]): row for row in baseline}
    shared = sorted(set(candidate_by_index) & set(baseline_by_index))
    first = next((
        index for index in shared
        if candidate_by_index[index].get("assistant_command_sha256")
        != baseline_by_index[index].get("assistant_command_sha256")
    ), None)
    if first is None:
        one_sided = sorted(set(candidate_by_index) ^ set(baseline_by_index))
        first = one_sided[0] if one_sided else None
    through = first - 1 if first is not None else max(candidate_by_index, default=0)
    prefix = [
        row for index, row in candidate_by_index.items() if index <= through
    ]
    return {
        "first_action_diverged": first is not None,
        "first_action_divergence_request": first,
        "selected_tokens_before_divergence_or_terminal": sum(
            int(row.get("materialized_tokens", row.get("selected_tokens", 0)) or 0)
            for row in prefix
        ),
        "full_tokens_before_divergence_or_terminal": sum(
            int(row.get("full_tokens") or 0) for row in prefix
        ),
    }


def write_frontier_ledger(
    *, state: Mapping[str, Any], spec: Mapping[str, Any], output: Path,
) -> Path:
    """Emit only complete, contemporaneously paired sequence cells."""

    cells = [row for row in state["cells"].values() if row.get("status") == "complete"]
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in cells:
        grouped.setdefault((str(row["sequence_id"]), int(row["repeat"])), []).append(row)
    ledger: list[dict[str, Any]] = []
    for (sequence_id, repeat), rows in sorted(grouped.items()):
        fresh = next((row for row in rows if row["strategy_id"] == "S00_fresh_full"), None)
        persistent = next((row for row in rows if row["strategy_id"] == "S01_persistent_full"), None)
        if fresh is None or persistent is None:
            continue
        persistent_episodes = list(persistent["episodes"].values())
        manifests = [
            _read(Path(str(row["output"])) / "run_manifest.json")
            for row in persistent_episodes
        ]
        agent_revisions = {
            _digest({
                "agent_behavior": manifest.get("agent_behavior_sha256"),
                "scaffold": manifest.get("scaffold_identity_sha256"),
            })
            for manifest in manifests
        }
        harness_revisions = {
            str(manifest.get("harness_version_observed") or "") for manifest in manifests
        }
        if len(agent_revisions) != 1 or len(harness_revisions) != 1 or "" in harness_revisions:
            raise ValueError("paired sequence has inconsistent agent/harness identity")
        pair_id = f"{spec['campaign_id']}:{sequence_id}:r{repeat}"
        workspace_schedule_digest = _digest({
            "dataset_revision": manifests[0].get("dataset_revision"),
            "ordered_instance_ids": persistent["instance_ids"],
            "workspace_semantics": spec.get("workspace_semantics"),
        })
        decoding_digest = _digest(spec["generation"])
        for row in rows:
            issue_rows: list[dict[str, Any]] = []
            candidate_episodes = list(row["episodes"].values())
            if len(candidate_episodes) != len(persistent_episodes):
                raise ValueError("paired sequence issue count changed")
            for index, (candidate_episode, baseline_episode) in enumerate(
                zip(candidate_episodes, persistent_episodes), 1
            ):
                candidate_output = Path(str(candidate_episode["output"]))
                baseline_output = Path(str(baseline_episode["output"]))
                metrics = _read(candidate_output / "autonomous_metrics.json")
                repeated = metrics.get("repeated_same_operation_resource_counts") or {}
                divergence = _divergence_accounting(
                    _trace(candidate_output), _trace(baseline_output)
                )
                issue_rows.append({
                    "issue_index": index,
                    "instance_id": candidate_episode["instance_id"],
                    "resolved": bool(candidate_episode["official_resolved"]),
                    "selected_input_tokens": int(
                        candidate_episode["cumulative_materialized_tokens"]
                    ),
                    "calls": int(candidate_episode["calls"]),
                    "rediscovery_calls": sum(
                        int(repeated.get(kind) or 0) for kind in ("search", "read")
                    ),
                    **divergence,
                })
            strategy = dict(row["strategy"])
            for key in ("strategy_id", "strategy_config_id", "repeats"):
                strategy.pop(key, None)
            config_id = str(row.get("strategy_config_id") or "default")
            ledger.append({
                "schema_version": 1,
                "pair_id": pair_id,
                "sequence_family_id": sequence_id,
                "sequence_id": sequence_id,
                "sequence_digest": row["sequence_digest"],
                "sequence_stratum": row["sequence_stratum"],
                "agent_id": "mini-swe-agent",
                "agent_revision": next(iter(agent_revisions)),
                "ordered_instance_ids": list(row["instance_ids"]),
                "issue_count": int(row["issue_count"]),
                "session_mode": row["session_mode"],
                "strategy_id": row["strategy_id"],
                "strategy_config_id": config_id,
                "strategy_config": strategy,
                "strategy_config_digest": _digest(strategy),
                "model_revision": spec["model_revision"],
                "tokenizer_revision": spec["tokenizer_revision"],
                "harness_revision": next(iter(harness_revisions)),
                "decoding_digest": decoding_digest,
                "workspace_schedule_digest": workspace_schedule_digest,
                "issues": issue_rows,
            })
    path = output / "frontier_runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger),
        encoding="utf-8",
    )
    return path


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    spec_path = args.spec.resolve()
    spec = _read(spec_path)
    benchmark = _resolve(spec_path, str(spec["benchmark_card"]))
    benchmark_card = _read(benchmark)
    validate_spec(spec, benchmark_card)
    for instance_id in benchmark_card["instance_ids"]:
        load_locked_task(benchmark, task_index=None, instance_id=instance_id)

    cells = campaign_cells(spec)
    if args.sequence_id:
        cells = [row for row in cells if row["sequence_id"] in set(args.sequence_id)]
    if args.strategy_id:
        cells = [row for row in cells if row["strategy_id"] in set(args.strategy_id)]
    output = args.output.resolve()
    state_path = output / "campaign_state.json"
    spec_digest = _digest(spec)
    state = _read(state_path) if state_path.is_file() else {
        "schema_version": 1,
        "study": "paper8_5_autonomous_multi_issue_campaign",
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec_digest,
        "primary_target": {
            "minimum_saving_fraction": TARGET_SAVING_MIN,
            "maximum_saving_fraction": TARGET_SAVING_MAX,
            "official_resolution_delta_minimum": 0,
            "lost_persistent_full_successes": 0,
            "pairing_required": True,
        },
        "cells": {},
    }
    if state.get("campaign_spec_sha256") != spec_digest:
        raise ValueError("campaign state belongs to a different frozen specification")

    launched = 0
    for cell in cells:
        if args.max_cells is not None and launched >= args.max_cells:
            break
        cell_id = str(cell["cell_id"])
        cell_root = output / cell_id
        row = state["cells"].setdefault(cell_id, {
            **cell,
            "status": "planned",
            "episodes": {},
            "sequence_digest": _digest(cell["instance_ids"]),
        })
        if row.get("status") == "complete":
            continue
        session_id = f"{spec['campaign_id']}:{cell_id}"
        prefix_episodes: list[dict[str, Any]] = []
        episode_results: list[dict[str, Any]] = []
        infrastructure_error = False
        for episode_number, instance_id in enumerate(cell["instance_ids"], 1):
            episode_id = f"episode_{episode_number:02d}_{_slug(instance_id)}"
            episode_output = cell_root / episode_id
            prefix_path = cell_root / f"prefix_before_episode_{episode_number:02d}.json"
            use_prefix = (
                cell["session_mode"] == "persistent" and episode_number > 1
            )
            if use_prefix and not args.dry_run:
                if len(prefix_episodes) != episode_number - 1:
                    raise AssertionError("persistent episode prefix is incomplete")
                _write(prefix_path, {
                    "schema_version": 1,
                    "study": "paper8_5_persistent_autonomous_prefix",
                    "session_id": session_id,
                    "episodes": prefix_episodes,
                })
            completed = _completed_result(episode_output)
            completed_ok = bool(
                completed is not None and completed.get("status") == "complete"
            )
            export_path = episode_output / "persistent_episode_export.json"
            command = _episode_command(
                spec=spec,
                benchmark=benchmark,
                cell=cell,
                instance_id=instance_id,
                episode_number=(
                    episode_number if cell["session_mode"] == "persistent" else 1
                ),
                output=episode_output,
                prefix=prefix_path if use_prefix else None,
                session_id=(session_id if cell["session_mode"] == "persistent" else f"{session_id}:{episode_id}"),
                args=args,
            )
            row["episodes"][episode_id] = {
                "instance_id": instance_id,
                "status": "complete" if completed_ok else "planned",
                "output": str(episode_output),
                "prefix": str(prefix_path) if use_prefix else None,
                "command": command,
            }
            _write(state_path, state)
            if args.dry_run:
                continue
            if not completed_ok:
                if episode_output.exists() and any(episode_output.iterdir()):
                    row["episodes"][episode_id]["status"] = "infrastructure_error"
                    row["episodes"][episode_id]["reason"] = (
                        "nonempty incomplete or infrastructure-contaminated output; "
                        "preserve and inspect"
                    )
                    if completed is not None:
                        row["episodes"][episode_id]["quarantine"] = completed
                    infrastructure_error = True
                    break
                row["status"] = "running"
                row["episodes"][episode_id]["status"] = "running"
                row["episodes"][episode_id]["started_at"] = datetime.now(timezone.utc).isoformat()
                _write(state_path, state)
                process = subprocess.run(command, check=False)
                completed = _completed_result(episode_output)
                completed_ok = bool(
                    completed is not None and completed.get("status") == "complete"
                )
                row["episodes"][episode_id]["returncode"] = process.returncode
                row["episodes"][episode_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
            if not completed_ok or not export_path.is_file():
                row["episodes"][episode_id]["status"] = "infrastructure_error"
                row["episodes"][episode_id]["reason"] = "missing complete result or persistent episode export"
                infrastructure_error = True
                _write(state_path, state)
                break
            row["episodes"][episode_id].update(completed)
            episode_results.append(completed)
            exported = _read(export_path)
            prefix_episodes.append(exported)
            _write(state_path, state)
        if args.dry_run:
            row["status"] = "planned"
        elif infrastructure_error:
            row["status"] = "infrastructure_error"
        else:
            row.update(_aggregate(cell, episode_results))
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write(state_path, state)
        launched += 1
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    ledger = write_frontier_ledger(state=state, spec=spec, output=output)
    state["frontier_ledger"] = str(ledger)
    _write(state_path, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--docker-executable")
    parser.add_argument("--docker-platform")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--sequence-id", action="append")
    parser.add_argument("--strategy-id", action="append")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--skip-grading", action="store_true")
    parser.add_argument("--grade-auxiliary-workspace-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    state = run_campaign(build_parser().parse_args())
    counts: dict[str, int] = {}
    for row in state["cells"].values():
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"campaign_id": state["campaign_id"], "status_counts": counts}))


if __name__ == "__main__":
    main()
