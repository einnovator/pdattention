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

from .run_autonomous_curve_campaign import _completed_result, _retry_path
from .run_autonomous_swebench import load_locked_task
from .upstream_health import normalize_ollama_cold_start, probe_generation_health


TARGET_SAVING_MIN = 0.30
TARGET_SAVING_MAX = 0.50
MAX_SEQUENCE_ISSUES = 20


def _normalize_trial_start(
    *, spec: Mapping[str, Any], args: argparse.Namespace,
) -> dict[str, Any] | None:
    contract = dict(spec.get("runtime_qualification") or {})
    if not contract.get("cold_reset_between_trials"):
        return None
    mode = str(contract.get("cold_reset_mode") or "ollama_keep_alive_zero")
    if mode != "ollama_keep_alive_zero":
        raise ValueError(f"unsupported cold reset mode: {mode}")
    return normalize_ollama_cold_start(
        base_url=args.upstream_base_url,
        model=str(spec["served_model"]),
        timeout_seconds=args.health_timeout_seconds,
        connect_attempts=getattr(args, "upstream_connect_attempts", 1),
        connect_retry_seconds=getattr(args, "upstream_connect_retry_seconds", 1.0),
    )


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


def _control_cell_id(
    cell: Mapping[str, Any],
    state_cells: Mapping[str, Mapping[str, Any]],
) -> str:
    """Resolve the unique same-sequence persistent-FULL control identity.

    Cell IDs include ``strategy_config_id`` when it is non-default, so
    reconstructing the ID from only the strategy name silently fails for
    frozen configured controls.  Resolve from the campaign ledger instead.
    """

    candidates = [
        cell_id
        for cell_id, row in state_cells.items()
        if row.get("sequence_id") == cell.get("sequence_id")
        and int(row.get("repeat", -1)) == int(cell.get("repeat", -2))
        and row.get("strategy_id") == "S01_persistent_full"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "shared first episode requires exactly one same-sequence "
            f"persistent-FULL control; found {len(candidates)}"
        )
    return candidates[0]


def _shared_control_prefix_count(strategy: Mapping[str, Any]) -> int:
    """Return the number of leading episodes borrowed byte-for-byte from FULL."""

    explicit = strategy.get("share_full_prefix_episodes")
    if explicit is None:
        return 1 if bool(strategy.get("share_full_first_episode", False)) else 0
    if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 0:
        raise ValueError("share_full_prefix_episodes must be a nonnegative integer")
    if strategy.get("share_full_first_episode", False) and explicit < 1:
        raise ValueError(
            "share_full_first_episode conflicts with share_full_prefix_episodes"
        )
    return explicit


def _requires_same_prefix_full_control(cell: Mapping[str, Any]) -> bool:
    """Return whether an episode needs an in-place no-selection control.

    A separate persistent-FULL sequence is not enough after the treatment has
    changed an earlier episode: its later prefix is then a different prefix.
    Every persistent treatment episode is therefore qualified from the exact
    prefix that the treatment is about to consume.
    """

    return bool(
        cell.get("session_mode") == "persistent" and cell.get("policy") != "full"
    )


def _same_prefix_full_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Build a logical FULL arm without inheriting treatment materialization."""

    strategy = dict(cell["strategy"])
    strategy.update({
        "policy": "full",
        "budget_fraction": 1.0,
        "negative_realization": "drop",
        "negative_fallback": "none",
        "keep_completed_task_statements": True,
        "compact_completed_finalizations": False,
        "retire_closed_instructions": False,
        "materialization_mode": "whole_record",
    })
    return {
        **cell,
        "cell_id": f"{cell['cell_id']}__same_prefix_full",
        "strategy_id": "S01_same_prefix_full_qualification",
        "strategy_config_id": "same_prefix_full_v1",
        "policy": "full",
        "strategy": strategy,
    }


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
        if not instance_ids or len(instance_ids) > MAX_SEQUENCE_ISSUES:
            raise ValueError(
                f"{sequence_id}: expected one to {MAX_SEQUENCE_ISSUES} issues"
            )
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
        boundary_mode = str(strategy.get("boundary_mode", "explicit"))
        if boundary_mode not in {"explicit", "boundary_free"}:
            raise ValueError(f"{strategy_id}: invalid boundary mode")
        if (
            strategy.get("policy") in {
                "persistent_global_retirement",
                "persistent_instruction_epoch_retirement",
                "frontier_dag_retirement",
            }
            and boundary_mode != "boundary_free"
        ):
            raise ValueError(
                f"{strategy.get('policy')} requires boundary_free mode"
            )
        shared_prefix = _shared_control_prefix_count(strategy)
        if shared_prefix and mode != "persistent":
            raise ValueError("shared FULL prefixes require persistent session mode")
        if shared_prefix > min(
            (len(sequence.get("instance_ids") or ())
             for sequence in spec.get("sequences") or ()),
            default=0,
        ):
            raise ValueError("shared FULL prefix exceeds a registered sequence")


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
        "--completed-protocol-turns", str(strategy.get("completed_protocol_turns", 0)),
        "--completed-finalization-turns", str(
            strategy.get("completed_finalization_turns", 0)
        ),
        "--completed-instruction-epochs", str(
            strategy.get("completed_instruction_epochs", 0)
        ),
        "--frontier-recent-user-prompts", str(
            strategy.get("frontier_recent_user_prompts", 2)
        ),
        "--frontier-protocol-exemplars", str(
            strategy.get("frontier_protocol_exemplars", 0)
        ),
        "--frontier-workflow-exemplars", str(
            strategy.get("frontier_workflow_exemplars", 0)
        ),
        "--boundary-mode", str(strategy.get("boundary_mode", "explicit")),
        "--temperature", str(generation["temperature"]),
        "--top-p", str(generation["top_p"]),
        "--seed", str(generation["seed"]),
        "--max-calls", str(generation["max_calls"]),
        "--max-completion-tokens", str(generation["max_completion_tokens"]),
        "--upstream-timeout-seconds", str(args.upstream_request_timeout_seconds),
        "--docker-pull-timeout-seconds", str(getattr(
            args, "docker_pull_timeout_seconds", 900
        )),
    ]
    if strategy.get("frontier_allow_heuristic", False):
        command.append("--frontier-allow-heuristic")
    upstream_qualification_path = getattr(args, "upstream_qualification_path", None)
    if upstream_qualification_path:
        command.extend((
            "--upstream-qualification-path", upstream_qualification_path,
            "--upstream-connect-attempts",
            str(getattr(args, "upstream_connect_attempts", 1)),
            "--upstream-connect-retry-seconds",
            str(getattr(args, "upstream_connect_retry_seconds", 1.0)),
        ))
    upstream_curl_executable = getattr(args, "upstream_curl_executable", None)
    if upstream_curl_executable:
        command.extend(("--upstream-curl-executable", upstream_curl_executable))
    upstream_relay_target = getattr(args, "upstream_relay_target", None)
    if upstream_relay_target:
        command.extend(("--upstream-relay-target", upstream_relay_target))
    native_pra_builder = getattr(args, "native_pra_builder", None)
    if native_pra_builder:
        command.extend(("--native-pra-builder", str(native_pra_builder)))
    if cell["session_mode"] == "persistent":
        command.extend((
            "--session-id", session_id,
            "--episode-index", str(episode_number),
        ))
        if prefix is not None:
            command.extend(("--persistent-prefix", str(prefix)))
    if not strategy.get("keep_completed_task_statements", True):
        command.append("--no-keep-completed-task-statements")
    if strategy.get("compact_completed_finalizations", False):
        command.append("--compact-completed-finalizations")
    if strategy.get("retire_closed_instructions", False):
        command.append("--retire-closed-instructions")
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


def _run_same_prefix_full_control(
    *,
    spec: Mapping[str, Any],
    benchmark: Path,
    cell: Mapping[str, Any],
    instance_id: str,
    episode_number: int,
    output: Path,
    prefix: Path | None,
    session_id: str,
    args: argparse.Namespace,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run FULL from the treatment's exact prefix before the treatment.

    The returned record is deliberately self-contained so the campaign ledger
    can distinguish a task that FULL could not solve in this multi-task state
    from a task that a memory heuristic lost.
    """

    qualification_cell = _same_prefix_full_cell(cell)
    prior_attempts = list((previous or {}).get("attempts") or ())
    if previous and previous.get("output"):
        prior = {
            key: previous.get(key) for key in (
                "output", "status", "reason", "returncode", "quarantine",
                "started_at", "finished_at", "health_preflight",
            ) if previous.get(key) is not None
        }
        if prior and prior not in prior_attempts:
            prior_attempts.append(prior)
    recorded_output = Path(str((previous or {}).get("output", output)))
    completed = _completed_result(recorded_output) or _completed_result(output)
    qualification_output = recorded_output if completed is not None else output
    if completed is None and (qualification_output.exists() or prior_attempts):
        reserved = [
            Path(str(attempt["output"]))
            for attempt in prior_attempts
            if attempt.get("output")
        ]
        reserved.append(qualification_output)
        qualification_output = _retry_path(output, reserved=tuple(reserved))
    command = _episode_command(
        spec=spec,
        benchmark=benchmark,
        cell=qualification_cell,
        instance_id=instance_id,
        episode_number=episode_number,
        output=qualification_output,
        prefix=prefix,
        session_id=f"{session_id}:same-prefix-full",
        args=args,
    )
    record: dict[str, Any] = {
        "required": True,
        "qualification_kind": "exact_candidate_prefix_full_v1",
        "status": "complete" if completed is not None else "planned",
        "output": str(qualification_output),
        "prefix": str(prefix) if prefix is not None else None,
        "prefix_sha256": (
            hashlib.sha256(prefix.read_bytes()).hexdigest()
            if prefix is not None and prefix.is_file() else _digest([])
        ),
        "command": command,
        "attempts": prior_attempts,
    }
    if args.dry_run:
        return record
    if completed is None:
        runtime_qualification = dict(spec.get("runtime_qualification") or {})
        normalization = _normalize_trial_start(spec=spec, args=args)
        record["trial_start_normalization"] = normalization
        if normalization is not None and not normalization["healthy"]:
            record.update({
                "status": "paused_upstream_unhealthy",
                "reason": "cold trial-start normalization failed",
            })
            return record
        health = probe_generation_health(
            base_url=args.upstream_base_url,
            model=str(spec["served_model"]),
            count=args.health_probe_count,
            latency_ceiling_seconds=args.health_latency_ceiling_seconds,
            timeout_seconds=args.health_timeout_seconds,
            qualification_path=getattr(args, "upstream_qualification_path", None),
            runtime_state_path=(
                getattr(args, "upstream_runtime_state_path", None)
                or runtime_qualification.get("active_state_path")
            ),
            minimum_active_context_tokens=(
                getattr(args, "minimum_active_context_tokens", None)
                or runtime_qualification.get("minimum_active_context_tokens")
            ),
            connect_attempts=getattr(args, "upstream_connect_attempts", 1),
            connect_retry_seconds=getattr(
                args, "upstream_connect_retry_seconds", 1.0
            ),
            curl_executable=getattr(args, "upstream_curl_executable", None),
        )
        record["health_preflight"] = health
        if not health["healthy"]:
            record.update({
                "status": "paused_upstream_unhealthy",
                "reason": "generation-level upstream health gate failed",
            })
            return record
        record["status"] = "running"
        record["started_at"] = datetime.now(timezone.utc).isoformat()
        process = subprocess.run(command, check=False)
        record["returncode"] = process.returncode
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        completed = _completed_result(qualification_output)
    if completed is None or completed.get("status") != "complete":
        record.update({
            "status": "infrastructure_error",
            "reason": "same-prefix FULL control did not produce admissible results",
        })
        if completed is not None:
            record["quarantine"] = completed
        return record
    record.update(completed)
    record["status"] = (
        "qualified" if completed["official_resolved"]
        else "full_control_unresolved"
    )
    record["heuristic_attribution_admissible"] = bool(
        completed["official_resolved"]
    )
    return record


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
        "candidate_trajectory_gross_saving_fraction": saving,
        "unpaired_candidate_trajectory_target_region": (
            TARGET_SAVING_MIN <= saving <= TARGET_SAVING_MAX
        ),
        "failure_aware_saving_fraction": None,
        "in_primary_saving_target": None,
        "candidate_all_issues_resolved": all(official),
        "discovery_quality_target_met": None,
        "primary_target_met": None,
        "pairing_status": (
            "requires contemporaneous persistent-FULL sequence comparison"
        ),
    }


def _partial_aggregate(
    episode_rows: Sequence[Mapping[str, Any]], *, status: str,
) -> dict[str, Any]:
    """Summarize an intentionally stopped prefix without calling it a cohort."""

    if not episode_rows:
        raise ValueError("cannot summarize an empty stopped sequence")
    full = sum(int(row["cumulative_full_tokens"]) for row in episode_rows)
    materialized = sum(
        int(row["cumulative_materialized_tokens"]) for row in episode_rows
    )
    official = [bool(row["official_resolved"]) for row in episode_rows]
    return {
        "status": status,
        "observed_issue_count": len(episode_rows),
        "official_resolved_count": sum(official),
        "official_resolution_fraction": sum(official) / len(official),
        "all_observed_issues_resolved": all(official),
        "calls": sum(int(row["calls"]) for row in episode_rows),
        "cumulative_full_tokens": full,
        "cumulative_materialized_tokens": materialized,
        "candidate_trajectory_gross_saving_fraction": (
            1 - materialized / full if full else 0.0
        ),
        "failure_aware_saving_fraction": 0.0,
        "in_primary_saving_target": False,
        "primary_target_met": False,
        "pairing_status": "stopped after predeclared paired-quality gate",
    }


def _lost_paired_full_successes(
    *, cell: Mapping[str, Any], row: Mapping[str, Any],
    state_cells: Mapping[str, Mapping[str, Any]],
    observed_episode_ids: Sequence[str] | None = None,
) -> list[str]:
    """Return completed issue IDs lost relative to the paired FULL control."""

    control_id = _control_cell_id(cell, state_cells)
    control = state_cells.get(control_id) or {}
    losses: list[str] = []
    observed = set(observed_episode_ids) if observed_episode_ids is not None else None
    for episode_id, candidate in (row.get("episodes") or {}).items():
        if observed is not None and episode_id not in observed:
            continue
        baseline = (control.get("episodes") or {}).get(episode_id) or {}
        if (
            candidate.get("status") == "complete"
            and baseline.get("status") == "complete"
            and baseline.get("official_resolved") is True
            and candidate.get("official_resolved") is False
            and int(candidate.get("cumulative_materialized_tokens") or 0)
            < int(candidate.get("cumulative_full_tokens") or 0)
        ):
            losses.append(str(candidate.get("instance_id") or episode_id))
    return losses


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
    identical_input_divergence = False
    selection_active_at_divergence = False
    if first is not None and first in candidate_by_index and first in baseline_by_index:
        candidate_row = candidate_by_index[first]
        baseline_row = baseline_by_index[first]
        # ``request_input_sha256`` hashes the canonical pre-selection request.
        # Causal repeatability requires identity of what the model consumed.
        candidate_input = candidate_row.get("selected_messages_sha256")
        baseline_input = baseline_row.get("selected_messages_sha256")
        identical_input_divergence = bool(
            candidate_input
            and baseline_input
            and candidate_input == baseline_input
        )
        selection_active_at_divergence = int(
            candidate_row.get(
                "materialized_tokens", candidate_row.get("selected_tokens", 0)
            ) or 0
        ) < int(candidate_row.get("full_tokens") or 0)
    through = first - 1 if first is not None else max(candidate_by_index, default=0)
    prefix = [
        row for index, row in candidate_by_index.items() if index <= through
    ]
    return {
        "first_action_diverged": first is not None,
        "first_action_divergence_request": first,
        "identical_input_first_action_divergence": identical_input_divergence,
        "selection_active_at_first_action_divergence": selection_active_at_divergence,
        "selected_tokens_before_divergence_or_terminal": sum(
            int(row.get("materialized_tokens", row.get("selected_tokens", 0)) or 0)
            for row in prefix
        ),
        "full_tokens_before_divergence_or_terminal": sum(
            int(row.get("full_tokens") or 0) for row in prefix
        ),
    }


_PAIRING_MANIFEST_FIELDS = (
    "agent_behavior_sha256",
    "scaffold_identity_sha256",
    "harness_version_observed",
    "grader_version_observed",
    "model_revision",
    "tokenizer_revision",
    "dataset_revision",
    "benchmark_card_sha256",
    "temperature",
    "top_p",
    "seed",
    "max_calls",
    "max_completion_tokens",
)

_SAME_PREFIX_FULL_FIELDS = (
    "instance_id",
    "pair_id",
    "model_revision",
    "tokenizer_revision",
    "dataset_revision",
    "benchmark_card_sha256",
    "temperature",
    "top_p",
    "seed",
    "max_calls",
    "max_completion_tokens",
    "scaffold_identity_sha256",
    "workspace_source_identity_sha256",
)


def _validate_same_prefix_full_qualification(
    *, candidate_episode: Mapping[str, Any], candidate_manifest: Mapping[str, Any],
) -> None:
    qualification = candidate_episode.get("same_prefix_full_control") or {}
    if qualification.get("status") != "qualified":
        raise ValueError("treatment episode lacks a successful same-prefix FULL control")
    control_manifest = _read(
        Path(str(qualification["output"])) / "run_manifest.json"
    )
    for field in _SAME_PREFIX_FULL_FIELDS:
        if candidate_manifest.get(field) in (None, "") or (
            candidate_manifest.get(field) != control_manifest.get(field)
        ):
            raise ValueError(f"same-prefix FULL changed manifest identity {field}")
    candidate_session = candidate_manifest.get("persistent_session") or {}
    control_session = control_manifest.get("persistent_session") or {}
    if candidate_session.get("episode_index") != control_session.get("episode_index"):
        raise ValueError("same-prefix FULL changed episode index")
    candidate_prefix = candidate_session.get("prefix") or {}
    control_prefix = control_session.get("prefix") or {}
    if candidate_prefix.get("sha256") != control_prefix.get("sha256"):
        raise ValueError("same-prefix FULL did not consume the candidate prefix")
    if candidate_prefix.get("episode_count") != control_prefix.get("episode_count"):
        raise ValueError("same-prefix FULL changed prefix episode count")
    if (control_manifest.get("selection") or {}).get("policy") != "full":
        raise ValueError("same-prefix qualification control applied selection")
    control_created = str(control_manifest.get("created_at") or "")
    candidate_created = str(candidate_manifest.get("created_at") or "")
    if not control_created or not candidate_created or control_created >= candidate_created:
        raise ValueError("same-prefix FULL was not executed before the treatment")


def _validate_sequence_pairing_identity(
    *,
    candidate_manifests: Sequence[Mapping[str, Any]],
    baseline_manifests: Sequence[Mapping[str, Any]],
    expected_pair_id: str,
) -> str:
    """Validate identity per ordered task and return a sequence agent digest.

    ``agent_behavior_sha256`` contains task-specific command and environment
    identity, so it is expected to differ between episodes in a multi-issue
    sequence.  The scientific pairing invariant is that each candidate episode
    matches its corresponding persistent-FULL episode, while the scaffold and
    harness remain fixed over the ordered sequence.
    """

    if len(candidate_manifests) != len(baseline_manifests):
        raise ValueError("paired sequence issue count changed")
    scaffold_revisions: set[str] = set()
    harness_revisions: set[str] = set()
    ordered_agent_behaviors: list[str] = []
    for index, (candidate, baseline) in enumerate(
        zip(candidate_manifests, baseline_manifests), 1
    ):
        if candidate.get("instance_id") != baseline.get("instance_id"):
            raise ValueError(f"paired episode {index} instance identity changed")
        for manifest in (candidate, baseline):
            if manifest.get("pair_id") != expected_pair_id:
                raise ValueError(f"paired episode {index} pair_id changed")
        for field in _PAIRING_MANIFEST_FIELDS:
            baseline_value = baseline.get(field)
            candidate_value = candidate.get(field)
            if baseline_value in (None, "") or candidate_value in (None, ""):
                raise ValueError(
                    f"paired episode {index} is missing manifest identity {field}"
                )
            if candidate_value != baseline_value:
                raise ValueError(
                    f"paired episode {index} changed manifest identity {field}"
                )
        scaffold_revisions.add(str(baseline["scaffold_identity_sha256"]))
        harness_revisions.add(str(baseline["harness_version_observed"]))
        ordered_agent_behaviors.append(str(baseline["agent_behavior_sha256"]))
    if len(scaffold_revisions) != 1 or len(harness_revisions) != 1:
        raise ValueError("paired sequence has inconsistent scaffold/harness identity")
    return _digest({
        "ordered_agent_behavior_sha256": ordered_agent_behaviors,
        "scaffold_identity_sha256": next(iter(scaffold_revisions)),
        "harness_version_observed": next(iter(harness_revisions)),
    })


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
        if persistent is None:
            continue
        persistent_episodes = list(persistent["episodes"].values())
        manifests = [
            _read(Path(str(row["output"])) / "run_manifest.json")
            for row in persistent_episodes
        ]
        harness_revisions = {
            str(manifest.get("harness_version_observed") or "") for manifest in manifests
        }
        if len(harness_revisions) != 1 or "" in harness_revisions:
            raise ValueError("paired sequence has inconsistent harness identity")
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
            candidate_manifests = [
                _read(Path(str(episode["output"])) / "run_manifest.json")
                for episode in candidate_episodes
            ]
            if row["policy"] != "full":
                for candidate_episode, candidate_manifest in zip(
                    candidate_episodes, candidate_manifests
                ):
                    if candidate_episode.get("shared_control_episode"):
                        continue
                    _validate_same_prefix_full_qualification(
                        candidate_episode=candidate_episode,
                        candidate_manifest=candidate_manifest,
                    )
            agent_revision = _validate_sequence_pairing_identity(
                candidate_manifests=candidate_manifests,
                baseline_manifests=manifests,
                expected_pair_id=pair_id,
            )
            for index, (candidate_episode, baseline_episode) in enumerate(
                zip(candidate_episodes, persistent_episodes), 1
            ):
                candidate_output = Path(str(candidate_episode["output"]))
                baseline_output = Path(str(baseline_episode["output"]))
                metrics = _read(candidate_output / "autonomous_metrics.json")
                official = _read(candidate_output / "official_result.json")
                candidate_trace = _trace(candidate_output)
                trace_indexes = [int(item["request_index"]) for item in candidate_trace]
                trace_contiguous = trace_indexes == list(
                    range(1, len(trace_indexes) + 1)
                )
                upstream_error_calls = sum(
                    int(item.get("upstream_status") or 0) < 100
                    or int(item.get("upstream_status") or 0) >= 400
                    for item in candidate_trace
                )
                evidence_admissible = bool(
                    trace_contiguous
                    and upstream_error_calls == 0
                    and official.get("official_grader") is True
                    and isinstance(official.get("resolved"), bool)
                )
                repeated = metrics.get("repeated_same_operation_resource_counts") or {}
                divergence = _divergence_accounting(
                    candidate_trace, _trace(baseline_output)
                )
                issue_rows.append({
                    "issue_index": index,
                    "instance_id": candidate_episode["instance_id"],
                    "resolved": bool(candidate_episode["official_resolved"]),
                    "official_error": bool(official.get("error")),
                    "evidence_admissible": evidence_admissible,
                    "same_prefix_full_qualified": bool(
                        row["policy"] == "full"
                        or candidate_episode.get(
                            "heuristic_attribution_admissible", False
                        )
                    ),
                    "same_prefix_full_control_status": (
                        (
                            candidate_episode.get("same_prefix_full_control")
                            or {}
                        ).get("status")
                        if row["policy"] != "full" else "not_applicable_full"
                    ),
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
                "agent_revision": agent_revision,
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
    stop_after_episodes = getattr(args, "stop_after_completed_episodes", None)
    if stop_after_episodes is not None and int(stop_after_episodes) < 1:
        raise ValueError("stop_after_completed_episodes must be positive")
    stop_after_losses = getattr(
        args, "stop_after_lost_paired_full_successes", None
    )
    if stop_after_losses is not None and int(stop_after_losses) < 1:
        raise ValueError("stop_after_lost_paired_full_successes must be positive")
    spec_path = args.spec.resolve()
    spec = _read(spec_path)
    benchmark = _resolve(spec_path, str(spec["benchmark_card"]))
    benchmark_card = _read(benchmark)
    validate_spec(spec, benchmark_card)
    runtime_qualification = dict(spec.get("runtime_qualification") or {})
    runtime_state_path = (
        getattr(args, "upstream_runtime_state_path", None)
        or runtime_qualification.get("active_state_path")
    )
    minimum_active_context_tokens = (
        getattr(args, "minimum_active_context_tokens", None)
        or runtime_qualification.get("minimum_active_context_tokens")
    )
    for instance_id in benchmark_card["instance_ids"]:
        load_locked_task(benchmark, task_index=None, instance_id=instance_id)

    all_cells = campaign_cells(spec)
    cell_registry = {str(row["cell_id"]): row for row in all_cells}
    cells = list(all_cells)
    if args.sequence_id:
        cells = [row for row in cells if row["sequence_id"] in set(args.sequence_id)]
    if args.strategy_id:
        cells = [row for row in cells if row["strategy_id"] in set(args.strategy_id)]
    if args.strategy_config_id:
        cells = [
            row for row in cells
            if row["strategy_config_id"] in set(args.strategy_config_id)
        ]
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
    halt_campaign = False
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
        upstream_paused = False
        quality_gate_stopped = False
        causal_gate_stopped = False
        session_interference_stopped = False
        for episode_number, instance_id in enumerate(cell["instance_ids"], 1):
            stop_after_episodes = getattr(args, "stop_after_completed_episodes", None)
            if (
                stop_after_episodes is not None
                and episode_number > int(stop_after_episodes)
                and episode_results
            ):
                causal_gate_stopped = True
                row.update(_partial_aggregate(
                    episode_results,
                    status="stopped_causal_attribution_gate",
                ))
                row["stop_gate"] = {
                    "rule": "completed_episode_prefix",
                    "threshold": int(stop_after_episodes),
                    "observed": len(episode_results),
                    "reason": (
                        "stop before the first intervention because the "
                        "full-materialization repeat prefix diverged"
                    ),
                }
                for later in row["episodes"].values():
                    if later.get("status") != "complete":
                        later["status"] = "aborted_by_causal_attribution_gate"
                        later["reason"] = row["stop_gate"]["reason"]
                _write(state_path, state)
                break
            episode_id = f"episode_{episode_number:02d}_{_slug(instance_id)}"
            base_episode_output = cell_root / episode_id
            episode_output = base_episode_output
            shared_control_cell_id: str | None = None
            if episode_number <= _shared_control_prefix_count(cell["strategy"]):
                shared_control_cell_id = _control_cell_id(
                    cell,
                    {**cell_registry, **state["cells"]},
                )
                control = state["cells"].get(shared_control_cell_id)
                if not args.dry_run:
                    if not control or control.get("status") != "complete":
                        row["status"] = "waiting_for_persistent_full_control"
                        row["reason"] = (
                            f"shared first episode requires {shared_control_cell_id}"
                        )
                        _write(state_path, state)
                        infrastructure_error = True
                        break
                    control_episodes = list(control["episodes"].values())
                    if not control_episodes:
                        raise ValueError("persistent-FULL control has no episode ledger")
                    shared_episode = control_episodes[episode_number - 1]
                    if shared_episode.get("instance_id") != instance_id:
                        raise ValueError("shared FULL episode identity mismatch")
                    episode_output = Path(str(shared_episode["output"]))
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
            recorded_episode = row.get("episodes", {}).get(episode_id) or {}
            same_prefix_full_control: dict[str, Any] | None = None
            if (
                _requires_same_prefix_full_control(cell)
                and shared_control_cell_id is None
            ):
                same_prefix_full_control = _run_same_prefix_full_control(
                    spec=spec,
                    benchmark=benchmark,
                    cell=cell,
                    instance_id=instance_id,
                    episode_number=episode_number,
                    output=(
                        cell_root / "same_prefix_full_controls" / episode_id
                    ),
                    prefix=prefix_path if use_prefix else None,
                    session_id=session_id,
                    args=args,
                    previous=recorded_episode.get("same_prefix_full_control"),
                )
                if not args.dry_run and same_prefix_full_control["status"] != "qualified":
                    control_status = str(same_prefix_full_control["status"])
                    row["episodes"][episode_id] = {
                        "instance_id": instance_id,
                        "status": (
                            "unqualified_session_interference"
                            if control_status == "full_control_unresolved"
                            else control_status
                        ),
                        "output": str(base_episode_output),
                        "prefix": str(prefix_path) if use_prefix else None,
                        "command": None,
                        "same_prefix_full_control": same_prefix_full_control,
                        "heuristic_launched": False,
                        "heuristic_attribution_admissible": False,
                    }
                    if control_status == "paused_upstream_unhealthy":
                        row["status"] = "paused_upstream_unhealthy"
                        upstream_paused = True
                        halt_campaign = True
                    elif control_status == "infrastructure_error":
                        row["status"] = "infrastructure_error"
                        infrastructure_error = True
                        halt_campaign = True
                    else:
                        row["status"] = "stopped_unqualified_session_interference"
                        row["reason"] = (
                            "FULL did not resolve from the candidate's exact current "
                            "session prefix; no heuristic was launched and no failure "
                            "is attributed to selection"
                        )
                        session_interference_stopped = True
                    _write(state_path, state)
                    break
            recorded_output = Path(str(
                recorded_episode.get("output", base_episode_output)
            ))
            if shared_control_cell_id:
                completed = _completed_result(episode_output)
            else:
                recorded_result = _completed_result(recorded_output)
                base_result = _completed_result(base_episode_output)
                completed = recorded_result or base_result
                if completed is not None and completed.get("status") == "complete":
                    episode_output = (
                        recorded_output if recorded_result is not None
                        else base_episode_output
                    )
                else:
                    reserved_outputs = {
                        Path(str(attempt["output"]))
                        for attempt in recorded_episode.get("attempts", ())
                        if attempt.get("output")
                    }
                    if recorded_episode.get("output"):
                        reserved_outputs.add(recorded_output)
                    episode_output = _retry_path(
                        base_episode_output, reserved=tuple(reserved_outputs)
                    )
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
            prior_attempts = list(recorded_episode.get("attempts") or ())
            if recorded_episode and recorded_episode.get("output"):
                previous = {
                    key: recorded_episode.get(key) for key in (
                        "output", "status", "reason", "returncode", "quarantine",
                        "started_at", "finished_at", "health_preflight",
                    ) if recorded_episode.get(key) is not None
                }
                if previous and previous not in prior_attempts:
                    prior_attempts.append(previous)
            row["episodes"][episode_id] = {
                "instance_id": instance_id,
                "status": "complete" if completed_ok else "planned",
                "output": str(episode_output),
                "prefix": str(prefix_path) if use_prefix else None,
                "command": None if shared_control_cell_id else command,
                "shared_control_cell_id": shared_control_cell_id,
                "shared_control_episode": bool(shared_control_cell_id),
                "same_prefix_full_control": same_prefix_full_control,
                "heuristic_scheduled": bool(
                    cell["policy"] != "full" and not shared_control_cell_id
                ),
                "heuristic_launched": False,
                "heuristic_attribution_admissible": bool(
                    cell["policy"] == "full"
                    or shared_control_cell_id
                    or (
                        same_prefix_full_control
                        and same_prefix_full_control.get("status") == "qualified"
                    )
                ),
                "attempts": prior_attempts,
            }
            _write(state_path, state)
            if args.dry_run:
                continue
            if not completed_ok:
                if shared_control_cell_id:
                    row["episodes"][episode_id]["status"] = (
                        "invalid_shared_control_episode"
                    )
                    infrastructure_error = True
                    _write(state_path, state)
                    break
                normalization = _normalize_trial_start(spec=spec, args=args)
                row["episodes"][episode_id]["trial_start_normalization"] = normalization
                if normalization is not None and not normalization["healthy"]:
                    row["episodes"][episode_id]["status"] = "paused_upstream_unhealthy"
                    row["episodes"][episode_id]["reason"] = (
                        "cold trial-start normalization failed"
                    )
                    _write(state_path, state)
                    infrastructure_error = True
                    break
                health = probe_generation_health(
                    base_url=args.upstream_base_url,
                    model=str(spec["served_model"]),
                    count=args.health_probe_count,
                    latency_ceiling_seconds=args.health_latency_ceiling_seconds,
                    timeout_seconds=args.health_timeout_seconds,
                    qualification_path=getattr(
                        args, "upstream_qualification_path", None
                    ),
                    runtime_state_path=getattr(
                        args, "upstream_runtime_state_path", None
                    ) or runtime_state_path,
                    minimum_active_context_tokens=getattr(
                        args, "minimum_active_context_tokens", None
                    ) or minimum_active_context_tokens,
                    connect_attempts=getattr(args, "upstream_connect_attempts", 1),
                    connect_retry_seconds=getattr(
                        args, "upstream_connect_retry_seconds", 1.0
                    ),
                    curl_executable=getattr(
                        args, "upstream_curl_executable", None
                    ),
                )
                row["episodes"][episode_id]["health_preflight"] = health
                if not health["healthy"]:
                    row["episodes"][episode_id]["status"] = "paused_upstream_unhealthy"
                    row["episodes"][episode_id]["reason"] = (
                        "generation-level upstream health gate failed"
                    )
                    row["status"] = "paused_upstream_unhealthy"
                    upstream_paused = True
                    halt_campaign = True
                    _write(state_path, state)
                    break
                row["status"] = "running"
                row["episodes"][episode_id]["status"] = "running"
                row["episodes"][episode_id]["started_at"] = datetime.now(timezone.utc).isoformat()
                row["episodes"][episode_id]["heuristic_launched"] = bool(
                    cell["policy"] != "full" and not shared_control_cell_id
                )
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
                if completed is not None:
                    row["episodes"][episode_id]["quarantine"] = completed
                infrastructure_error = True
                halt_campaign = True
                _write(state_path, state)
                break
            row["episodes"][episode_id].update(completed)
            episode_results.append(completed)
            exported = _read(export_path)
            prefix_episodes.append(exported)
            _write(state_path, state)
            stop_after_losses = getattr(
                args, "stop_after_lost_paired_full_successes", None
            )
            if (
                stop_after_losses is not None
                and cell["strategy_id"] not in {"S00_fresh_full", "S01_persistent_full"}
            ):
                losses = _lost_paired_full_successes(
                    cell=cell,
                    row=row,
                    state_cells=state["cells"],
                    observed_episode_ids=(
                        list(row["episodes"])[:len(episode_results)]
                    ),
                )
                if len(losses) >= int(stop_after_losses):
                    quality_gate_stopped = True
                    row.update(_partial_aggregate(
                        episode_results,
                        status="stopped_predeclared_quality_gate",
                    ))
                    row["stop_gate"] = {
                        "rule": "lost_paired_full_successes",
                        "threshold": int(stop_after_losses),
                        "observed": len(losses),
                        "instance_ids": losses,
                    }
                    for later in row["episodes"].values():
                        if later.get("status") != "complete":
                            later["status"] = "aborted_by_quality_gate"
                            later["reason"] = (
                                "arm stopped after predeclared paired-quality gate"
                            )
                    _write(state_path, state)
                    break
        if args.dry_run:
            row["status"] = "planned"
        elif upstream_paused:
            row["status"] = "paused_upstream_unhealthy"
        elif infrastructure_error:
            row["status"] = "infrastructure_error"
        elif quality_gate_stopped:
            row["status"] = "stopped_predeclared_quality_gate"
        elif causal_gate_stopped:
            row["status"] = "stopped_causal_attribution_gate"
        elif session_interference_stopped:
            row["status"] = "stopped_unqualified_session_interference"
        else:
            row.update(_aggregate(cell, episode_results))
            row.pop("reason", None)
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write(state_path, state)
        launched += 1
        if halt_campaign:
            break
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
    parser.add_argument("--docker-pull-timeout-seconds", type=int, default=900)
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--sequence-id", action="append")
    parser.add_argument("--strategy-id", action="append")
    parser.add_argument(
        "--strategy-config-id",
        action="append",
        help="Run only explicitly named registered configurations.",
    )
    parser.add_argument("--max-cells", type=int)
    parser.add_argument(
        "--stop-after-lost-paired-full-successes",
        type=int,
        help=(
            "Stop a treatment cell after this many completed issues that its "
            "paired persistent-FULL control solved."
        ),
    )
    parser.add_argument(
        "--stop-after-completed-episodes",
        type=int,
        help=(
            "Record an intentional causal-attribution stop after this many "
            "completed episodes instead of launching the next episode."
        ),
    )
    parser.add_argument("--skip-grading", action="store_true")
    parser.add_argument("--grade-auxiliary-workspace-state", action="store_true")
    parser.add_argument("--health-probe-count", type=int, default=3)
    parser.add_argument("--health-latency-ceiling-seconds", type=float, default=60.0)
    parser.add_argument("--health-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--upstream-request-timeout-seconds", type=int, default=180)
    parser.add_argument("--upstream-qualification-path")
    parser.add_argument("--upstream-runtime-state-path")
    parser.add_argument("--minimum-active-context-tokens", type=int)
    parser.add_argument("--upstream-connect-attempts", type=int, default=1)
    parser.add_argument("--upstream-connect-retry-seconds", type=float, default=1.0)
    parser.add_argument("--upstream-curl-executable")
    parser.add_argument("--upstream-relay-target")
    parser.add_argument(
        "--native-pra-builder",
        type=Path,
        help=(
            "Forward every episode's exact frozen wire plan through the named "
            "Paper 4.5 native-PRA realization adapter."
        ),
    )
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
