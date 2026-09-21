"""Build auditable Paper 8.5 quality--saving trade-off curves.

The reducer deliberately keeps frozen next-action evidence separate from
autonomous task-quality evidence.  It also distinguishes within-run gross
history saving from paired end-to-end saving, which includes any extra calls
caused by the policy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


STRICT_PAIRING_KEYS = (
    "repository_revision", "benchmark_card_sha256", "benchmark_ids_sha256",
    "dataset", "dataset_revision", "split", "served_model", "model_revision",
    "tokenizer", "tokenizer_revision", "temperature", "top_p", "seed",
    "max_calls", "max_completion_tokens", "harness_version_requested",
    "harness_version_observed", "grader_version_requested",
    "grader_version_observed", "docker_platform", "environment_image",
    "environment_image_id", "workspace_source_identity_sha256",
    "instrument_observations", "agent_behavior_sha256",
    "scaffold_identity_sha256",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(spec_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (spec_path.parent / path).resolve()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (parent / ".git").exists():
            return resolved.relative_to(parent).as_posix()
    return str(resolved)


def _strategy_id(selection: Mapping[str, Any]) -> str:
    policy = str(selection.get("policy", "unknown"))
    materializer = str(selection.get("materialization_mode", "whole_record"))
    parts = [policy]
    if materializer != "whole_record":
        parts.append(materializer)
    parameters = {
        "Kf": selection.get("search_delay_turns"),
        "Kw": selection.get("write_delay_turns"),
        "Kr": selection.get("same_span_reads_to_keep"),
        "Kx": selection.get("working_set_resources"),
    }
    relevant = {
        "h1_search_consumed": ("Kf",),
        "h1_all_branches_consumed_strict": ("Kf",),
        "h2a_write_current_read": ("Kw",),
        "h2b_verified_write": ("Kw",),
        "h3_read_superseded": ("Kr",),
        "h4_working_set": ("Kx",),
        "safe2_h1_h3": ("Kf", "Kr"),
        "h2a_h3": ("Kw", "Kr"),
        "h2b_h3": ("Kw", "Kr"),
        "safe3_h1_h3_h2b": ("Kf", "Kw", "Kr"),
        "all_h1_h2a_h2b_h3_h4": ("Kf", "Kw", "Kr", "Kx"),
    }.get(policy, ())
    if relevant:
        parts.append(",".join(f"{key}={parameters[key]}" for key in relevant))
    if policy != "full":
        parts.append(
            f"H={selection.get('protected_head_turns')},"
            f"T={selection.get('protected_tail_turns')}"
        )
    if materializer != "whole_record":
        parts.append(f"threshold={selection.get('materialization_threshold_tokens')}")
        materialization_parameters = (
            ("head", "materialization_head_lines"),
            ("tail", "materialization_tail_lines"),
            ("match_ctx", "materialization_match_context_lines"),
            ("max_match", "materialization_max_matched_lines"),
        )
        parts.append(",".join(
            f"{short}={selection.get(key)}"
            for short, key in materialization_parameters
        ))
    budget = float(selection.get("budget_fraction", 1.0))
    if budget != 1.0:
        parts.append(f"B={budget:g}")
    # Human-readable coordinates above are followed by a canonical digest of
    # every policy/materialization switch. This prevents newly added options
    # from silently collapsing into an older curve point.
    if policy != "full":
        parameter_fields = {
            "Kf": "search_delay_turns",
            "Kw": "write_delay_turns",
            "Kr": "same_span_reads_to_keep",
            "Kx": "working_set_resources",
        }
        excluded = {
            "expected_model", "task_id", "tokenizer_identity",
            *parameter_fields.values(),
        }
        coordinate = {
            str(key): value for key, value in selection.items()
            if key not in excluded
        }
        # Heuristic thresholds are inert for policies that never consult the
        # corresponding heuristic.  Keep them in the digest only when they
        # are semantic coordinates of the selected policy; otherwise two
        # behaviorally identical recency arms would be split merely because
        # their harness defaults were declared at different times.
        coordinate.update({
            parameter_fields[key]: selection.get(parameter_fields[key])
            for key in relevant
        })
        encoded = json.dumps(
            coordinate, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        parts.append("cfg=" + hashlib.sha256(encoded).hexdigest()[:12])
    return ";".join(parts)


def _canonical_frozen_strategy(treatment: str) -> str:
    """Remove replicate identity while retaining every policy coordinate."""
    return re.sub(r";seed=[^;]+", "", treatment)


def load_frozen_comparison(
    path: Path, *, cohort: str, adaptive_eligible: bool = True
) -> list[dict[str, Any]]:
    artifact = _read_json(path)
    if artifact.get("study") != "paper8_5_agent_memory_frozen_replay_comparison":
        raise ValueError(f"{path}: not a frozen replay comparison")
    identity = artifact.get("comparison_identity") or {}
    if not identity.get("full_replay_digest"):
        raise ValueError(f"{path}: missing repeat-qualified FULL replay identity")
    rows: list[dict[str, Any]] = []
    for arm in artifact.get("arms") or ():
        if not isinstance(arm, Mapping):
            raise ValueError(f"{path}: invalid arm")
        attempted = int(arm.get("decisions_attempted") or 0)
        completed = int(arm.get("decisions_completed") or 0)
        if attempted <= 0 or completed != attempted:
            raise ValueError(f"{path}: incomplete frozen arm {arm.get('treatment')}")
        if int(arm.get("transport_failures") or 0):
            raise ValueError(f"{path}: transport failures in frozen arm")
        if "conservative_action_equivalent_rate_vs_full" not in arm:
            raise ValueError(f"{path}: missing frozen quality proxy")
        if (
            "materialized_token_saving_fraction" not in arm
            and "logical_token_saving_fraction" not in arm
        ):
            raise ValueError(f"{path}: missing frozen saving metric")
        first = arm.get("first_conservative_action_divergence")
        treatment = str(arm.get("treatment") or arm.get("policy"))
        family = str(arm.get("policy"))
        marker = "materialization="
        if marker in treatment:
            family = treatment.split(marker, 1)[1].split(";", 1)[0]
        rows.append({
            "evidence_class": "frozen_next_action",
            "cohort": cohort,
            "independence_key": hashlib.sha256(json.dumps({
                "instance_id": identity.get("instance_id"),
                "model": identity.get("model"),
                "trajectory_digest": identity.get("trajectory_digest"),
            }, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "task_id": identity.get("instance_id"),
            "model": identity.get("model"),
            "strategy": treatment,
            "strategy_coordinate": _canonical_frozen_strategy(treatment),
            "strategy_family": family,
            "saving_fraction": float(
                arm.get("materialized_token_saving_fraction",
                        arm.get("logical_token_saving_fraction", 0.0))
            ),
            "quality_proxy": float(
                arm.get("conservative_action_equivalent_rate_vs_full", 0.0)
            ),
            "exact_command_rate": float(arm.get("exact_command_rate_vs_full", 0.0)),
            "valid_action_rate": float(arm.get("valid_action_rate", 0.0)),
            "first_action_divergence": first,
            # Aggregate comparison files preserve global trajectory decision
            # IDs, not a local ordinal.  Normalizing those IDs by the suffix
            # cohort size would be invalid, so no fraction is synthesized.
            "first_action_divergence_fraction": None,
            "divergence_right_censored_at": attempted if first is None else None,
            "decisions": attempted,
            "official_resolved": None,
            "auxiliary_resolved": None,
            "calls": None,
            "tool_calls": None,
            "adaptive_eligible": adaptive_eligible,
            "source": _portable_path(path),
            "source_sha256": _sha256(path),
        })
    return rows


def _trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _agent_behavior_digest(manifest: Mapping[str, Any]) -> str:
    """Hash agent semantics while excluding per-run transport/output paths."""

    command = manifest.get("agent_command_template")
    if not isinstance(command, list) or not command:
        raise ValueError("run manifest lacks an agent command template")
    normalized: list[str] = ["<python>"]
    index = 1
    while index < len(command):
        argument = str(command[index])
        if argument == "-o":
            index += 2
            continue
        if argument == "-c" and index + 1 < len(command):
            value = str(command[index + 1])
            key = value.split("=", 1)[0]
            if key == "model.model_kwargs.api_base":
                value = key + "=<proxy>"
            elif key == "environment.executable":
                value = key + "=<docker-executable>"
            elif key == "environment.instrumentation_output_root":
                value = key + "=<instrumentation-output>"
            normalized.extend((argument, value))
            index += 2
            continue
        normalized.append(argument)
        index += 1
    return hashlib.sha256(json.dumps(
        normalized, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def _external_auxiliary_grade(
    path: Path,
    *,
    instance_id: str,
    auxiliary_patch_sha256: str | None,
) -> tuple[bool, dict[str, Any]]:
    """Validate a separately executed grade of the terminal workspace patch.

    Expensive SWE-bench grading may run on a different host from the agent.
    The receipt binds that grade to the exact auxiliary patch instead of
    mutating the primary run or treating the auxiliary outcome as submission
    success.
    """

    receipt = _read_json(path)
    if receipt.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported auxiliary-grade receipt schema")
    if receipt.get("evidence_role") != "auxiliary_workspace_grade":
        raise ValueError(f"{path}: not an auxiliary workspace grade receipt")
    if str(receipt.get("instance_id")) != instance_id:
        raise ValueError(f"{path}: auxiliary grade instance does not match run")
    declared_patch = receipt.get("auxiliary_patch_sha256")
    if not auxiliary_patch_sha256 or declared_patch != auxiliary_patch_sha256:
        raise ValueError(f"{path}: auxiliary grade patch digest does not match run")
    if not isinstance(receipt.get("resolved"), bool):
        raise ValueError(f"{path}: auxiliary grade lacks a Boolean resolved outcome")
    report_path = _resolve(path, str(receipt.get("grader_report")))
    if not report_path.is_file():
        raise ValueError(f"{path}: auxiliary grader report is missing")
    declared_report = receipt.get("grader_report_sha256")
    if not declared_report or _sha256(report_path) != declared_report:
        raise ValueError(f"{path}: auxiliary grader report digest does not match")
    report = _read_json(report_path)
    row = report.get(instance_id)
    if not isinstance(row, Mapping) or row.get("resolved") is not receipt["resolved"]:
        raise ValueError(f"{path}: auxiliary grader report outcome does not match receipt")
    return bool(receipt["resolved"]), {
        "receipt": _portable_path(path),
        "receipt_sha256": _sha256(path),
        "grader_report": _portable_path(report_path),
        "grader_report_sha256": declared_report,
    }


def load_autonomous_run(
    path: Path, *, auxiliary_grade_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = path / "run_manifest.json"
    metrics_path = path / "autonomous_metrics.json"
    manifest = _read_json(manifest_path)
    metrics = _read_json(metrics_path)
    if manifest.get("study") != "paper8_5_autonomous_agent_memory":
        raise ValueError(f"{manifest_path}: not a Paper 8.5 autonomous run")
    selection = manifest.get("selection") or {}
    official = metrics.get("official_result") or {}
    auxiliary = metrics.get("auxiliary_workspace_state") or {}
    auxiliary_summary = auxiliary.get("official_result_summary") or {}
    full_tokens = int(metrics.get("cumulative_full_tokens") or 0)
    materialized_tokens = int(
        metrics.get("cumulative_materialized_tokens",
        metrics.get("cumulative_selected_tokens", 0)) or 0
    )
    usage_coverage = int(metrics.get("reported_usage_coverage_calls") or 0)
    completion_tokens = int(metrics.get("cumulative_reported_completion_tokens") or 0)
    official_error = bool(official.get("error", False)) if official else False
    grader_log = path / "grader.log"
    grader_text = (
        grader_log.read_text(encoding="utf-8", errors="replace")
        if grader_log.is_file() else ""
    )
    failure_class = official.get("failure_class")
    if not failure_class and official_error:
        failure_class = (
            "patch_apply_failed" if "Patch Apply Failed" in grader_text
            else "grader_reported_error"
        )
    auxiliary_score = (
        bool(auxiliary_summary.get("resolved"))
        if isinstance(auxiliary_summary.get("resolved"), bool) else None
    )
    auxiliary_grade_provenance: dict[str, Any] | None = None
    if auxiliary_grade_path is not None:
        external_score, auxiliary_grade_provenance = _external_auxiliary_grade(
            auxiliary_grade_path,
            instance_id=str(manifest.get("instance_id")),
            auxiliary_patch_sha256=(
                str(auxiliary.get("patch_sha256"))
                if auxiliary.get("patch_sha256") else None
            ),
        )
        if auxiliary_score is not None and auxiliary_score is not external_score:
            raise ValueError(
                f"{auxiliary_grade_path}: external auxiliary grade conflicts with run"
            )
        auxiliary_score = external_score
    official_score = (
        bool(official.get("resolved"))
        if isinstance(official.get("resolved"), bool) else None
    )
    solution_state_score = (
        True if official_score is True
        else auxiliary_score if auxiliary_score is not None
        else None
    )
    failure_taxonomy = (
        None if official_score is True
        else "submission_protocol_failure_with_resolving_workspace"
        if official_score is False and auxiliary_score is True
        else "nonresolving_workspace"
        if official_score is False and auxiliary_score is False
        else "primary_failure_workspace_unavailable"
        if official_score is False
        else "no_definitive_primary_grade"
    )
    if not isinstance(official.get("resolved"), bool):
        raise ValueError(f"{metrics_path}: missing completed official result")
    if full_tokens <= 0 or materialized_tokens < 0:
        raise ValueError(f"{metrics_path}: invalid token totals")
    trace_path = path / "request_selection.jsonl"
    trace = _trace(trace_path)
    if not trace:
        raise ValueError(f"{trace_path}: empty or incomplete request trace")
    indexes = [int(row["request_index"]) for row in trace]
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"{trace_path}: duplicate request indexes")
    if sorted(indexes) != list(range(min(indexes), max(indexes) + 1)):
        raise ValueError(f"{trace_path}: sparse request indexes")
    if len(trace) != int(metrics.get("calls") or 0):
        raise ValueError(f"{trace_path}: request count does not match metrics")
    trace_full_tokens = sum(int(row.get("full_tokens") or 0) for row in trace)
    trace_materialized_tokens = sum(
        int(row.get("materialized_tokens", row.get("selected_tokens", 0)) or 0)
        for row in trace
    )
    if trace_full_tokens != full_tokens:
        raise ValueError(f"{trace_path}: full-token total does not match metrics")
    if trace_materialized_tokens != materialized_tokens:
        raise ValueError(
            f"{trace_path}: materialized-token total does not match metrics"
        )
    pairing_identity = {key: manifest.get(key) for key in STRICT_PAIRING_KEYS}
    # Early schema-2 manifests recorded Docker's immutable observed OS and
    # architecture but left ``docker_platform`` null when no platform was
    # explicitly requested.  Recover that same observed identity for pairing;
    # this is not a guessed legacy exception and does not mutate the evidence.
    if not pairing_identity.get("docker_platform"):
        image_os = manifest.get("environment_image_os")
        image_architecture = manifest.get("environment_image_architecture")
        if image_os and image_architecture:
            pairing_identity["docker_platform"] = (
                f"{image_os}/{image_architecture}"
            )
    pairing_identity["agent_behavior_sha256"] = (
        manifest.get("agent_behavior_sha256") or _agent_behavior_digest(manifest)
    )
    missing_pairing_identity = sorted(
        key for key, value in pairing_identity.items()
        if value is None or value == ""
    )
    return {
        "evidence_class": "autonomous_task_quality",
        "task_id": str(manifest.get("instance_id")),
        "model": str(manifest.get("model")),
        "seed": int(manifest.get("seed", 0)),
        "pair_id": manifest.get("pair_id"),
        "manifest_schema_version": int(manifest.get("schema_version") or 1),
        "missing_pairing_identity": missing_pairing_identity,
        "strategy": _strategy_id(selection),
        "strategy_family": str(selection.get("policy", "unknown")),
        "saving_fraction": 1.0 - materialized_tokens / full_tokens if full_tokens else 0.0,
        "cumulative_full_tokens": full_tokens,
        "cumulative_materialized_tokens": materialized_tokens,
        "reported_usage_coverage_calls": usage_coverage,
        "cumulative_reported_completion_tokens": completion_tokens,
        "cumulative_total_model_tokens": (
            materialized_tokens + completion_tokens
            if usage_coverage == int(metrics.get("calls") or 0) else None
        ),
        "official_resolved": bool(official.get("resolved")) if official else None,
        "official_error": official_error if official else None,
        "official_score": official_score,
        "official_failure_class": failure_class,
        "official_outcome": (
            "patch_apply_failed"
            if official and failure_class == "patch_apply_failed"
            else "empty_submission"
            if official and failure_class == "empty_submission"
            else "grader_error"
            if official and official_error
            else "resolved"
            if official and bool(official.get("resolved"))
            else "unresolved"
            if official
            else None
        ),
        "auxiliary_status": auxiliary.get("status"),
        "auxiliary_grade_available": auxiliary_score is not None,
        "auxiliary_resolved": auxiliary_score,
        "auxiliary_grade_provenance": auxiliary_grade_provenance,
        "solution_state_score": solution_state_score,
        "failure_taxonomy": failure_taxonomy,
        "calls": int(metrics.get("calls") or 0),
        "tool_calls": int(metrics.get("actions") or 0),
        "reacquisitions": int(metrics.get("reacquisition_events") or 0),
        "reacquired_observation_tokens": int(
            metrics.get("reacquired_observation_tokens") or 0
        ),
        "reacquisition_adjusted_gross_saving_fraction": (
            (
                full_tokens
                - materialized_tokens
                - int(metrics.get("reacquired_observation_tokens") or 0)
            ) / full_tokens
            if full_tokens else 0.0
        ),
        "trace": trace,
        "pairing_identity": pairing_identity,
        "source": _portable_path(path),
        "source_sha256": {
            "manifest": _sha256(manifest_path),
            "metrics": _sha256(metrics_path),
            "trace": _sha256(trace_path),
        },
    }


def pair_autonomous(
    candidate: dict[str, Any], baseline: dict[str, Any], *,
    allow_legacy_pair: bool = False, pairing_reason: str | None = None,
) -> None:
    if candidate["task_id"] != baseline["task_id"] or candidate["model"] != baseline["model"]:
        raise ValueError("paired autonomous runs must share task and model")
    legacy_contract = (
        candidate.get("manifest_schema_version", 1) < 2
        or baseline.get("manifest_schema_version", 1) < 2
    )
    missing_identity = {
        "candidate": candidate.get("missing_pairing_identity") or [],
        "baseline": baseline.get("missing_pairing_identity") or [],
    }
    if (legacy_contract or any(missing_identity.values())) and not allow_legacy_pair:
        raise ValueError(
            "paired autonomous runs require complete schema-2 execution identities; "
            "legacy evidence needs an explicit exception and reason"
        )
    pair_id_mismatch = (
        candidate.get("pair_id") is not None
        and baseline.get("pair_id") is not None
        and candidate["pair_id"] != baseline["pair_id"]
    )
    if pair_id_mismatch and not allow_legacy_pair:
        raise ValueError("paired autonomous runs declare different pair IDs")
    if not allow_legacy_pair and (
        candidate.get("pair_id") is None or baseline.get("pair_id") is None
    ):
        raise ValueError("paired autonomous runs require a shared non-null pair ID")
    identity_mismatches = {
        key: {"candidate": candidate.get("pairing_identity", {}).get(key),
              "baseline": baseline.get("pairing_identity", {}).get(key)}
        for key in set(candidate.get("pairing_identity", {}))
        | set(baseline.get("pairing_identity", {}))
        if candidate.get("pairing_identity", {}).get(key)
        != baseline.get("pairing_identity", {}).get(key)
    }
    disallowed_legacy_mismatches = set(identity_mismatches) - {"repository_revision"}
    if identity_mismatches and (
        not allow_legacy_pair or disallowed_legacy_mismatches
    ):
        raise ValueError("paired autonomous runs differ in frozen execution identity")
    if allow_legacy_pair and not pairing_reason:
        raise ValueError("legacy autonomous pairing requires an explicit reason")
    if baseline.get("strategy") != "full" or baseline.get("saving_fraction") != 0.0:
        raise ValueError("paired autonomous baseline must be an uncompressed FULL run")
    candidate["baseline_strategy"] = baseline["strategy"]
    candidate["pairing_role"] = (
        "full_repeat_control"
        if candidate.get("strategy_family") == "full"
        else "policy_candidate"
    )
    candidate["pairing_reason"] = pairing_reason
    candidate["legacy_pairing_identity_mismatches"] = identity_mismatches
    candidate["legacy_pairing_missing_fields"] = (
        missing_identity if legacy_contract or any(missing_identity.values()) else None
    )
    candidate["legacy_pair_id_mismatch"] = (
        {
            "candidate": candidate.get("pair_id"),
            "baseline": baseline.get("pair_id"),
        }
        if pair_id_mismatch else None
    )
    candidate["baseline_official_resolved"] = baseline["official_resolved"]
    candidate["baseline_official_error"] = baseline.get("official_error")
    candidate["baseline_official_score"] = baseline.get("official_score")
    candidate["baseline_official_outcome"] = baseline.get("official_outcome")
    candidate["baseline_auxiliary_resolved"] = baseline["auxiliary_resolved"]
    candidate["call_delta"] = candidate["calls"] - baseline["calls"]
    candidate["tool_call_delta"] = candidate["tool_calls"] - baseline["tool_calls"]
    candidate["efficiency_qualified"] = bool(
        candidate.get("official_score") is True
        and baseline.get("official_score") is True
    )
    candidate["trajectory_outcome"] = (
        "both_resolved" if candidate["efficiency_qualified"]
        else "candidate_grader_error" if candidate.get("official_error")
        else "baseline_grader_error" if baseline.get("official_error")
        else "candidate_failed" if not candidate["official_resolved"]
        else "baseline_failed"
    )
    baseline_tokens = baseline["cumulative_materialized_tokens"]
    candidate["paired_net_saving_fraction"] = (
        1.0 - candidate["cumulative_materialized_tokens"] / baseline_tokens
        if baseline_tokens else None
    )
    # A failed/errored candidate has no tokens-to-solution. Prevent early
    # termination from being reported as an efficiency gain by charging at
    # least the paired FULL workload in the failure-aware coordinate.
    failure_aware_tokens = (
        candidate["cumulative_materialized_tokens"]
        if candidate["efficiency_qualified"]
        else max(candidate["cumulative_materialized_tokens"], baseline_tokens)
    )
    candidate["failure_aware_paired_net_saving_fraction"] = (
        1.0 - failure_aware_tokens / baseline_tokens if baseline_tokens else None
    )
    baseline_total = baseline.get("cumulative_total_model_tokens")
    candidate_total = candidate.get("cumulative_total_model_tokens")
    candidate["paired_total_model_token_saving_fraction"] = (
        1.0 - float(candidate_total) / float(baseline_total)
        if baseline_total and candidate_total is not None else None
    )
    candidate_by_index = {int(row["request_index"]): row for row in candidate["trace"]}
    baseline_by_index = {int(row["request_index"]): row for row in baseline["trace"]}
    shared = sorted(set(candidate_by_index) & set(baseline_by_index))
    first = next((
        index for index in shared
        if candidate_by_index[index].get("assistant_command_sha256")
        != baseline_by_index[index].get("assistant_command_sha256")
    ), None)
    divergence_kind = "command_mismatch" if first is not None else None
    if first is None:
        candidate_only = set(candidate_by_index) - set(baseline_by_index)
        baseline_only = set(baseline_by_index) - set(candidate_by_index)
        if candidate_only or baseline_only:
            first = min(candidate_only | baseline_only)
            divergence_kind = (
                "baseline_terminated" if first in candidate_only
                else "candidate_terminated"
            )
    candidate["first_action_divergence"] = first
    candidate["first_action_divergence_kind"] = divergence_kind
    candidate["divergence_right_censored_at"] = min(
        len(candidate["trace"]), len(baseline["trace"])
    ) if first is None else None
    selection_change = next((
        index for index in shared
        if candidate_by_index[index].get("selected_messages_sha256")
        != baseline_by_index[index].get("selected_messages_sha256")
    ), None)
    candidate["first_selection_change"] = selection_change
    candidate["selection_to_action_divergence_lag"] = (
        first - selection_change
        if first is not None and selection_change is not None
        and first >= selection_change else None
    )
    # The divergent request is downstream of the changed action. Measure only
    # requests preceding it so post-divergence compression cannot be presented
    # as a possible cause of the divergence.
    through = (
        first - 1 if first is not None
        else candidate["divergence_right_censored_at"]
    )
    causal_rows = [
        row for row in candidate["trace"]
        if through is not None and int(row["request_index"]) <= int(through)
    ]
    causal_full = sum(int(row.get("full_tokens") or 0) for row in causal_rows)
    causal_materialized = sum(
        int(row.get("materialized_tokens", row.get("selected_tokens", 0)) or 0)
        for row in causal_rows
    )
    candidate["saving_before_first_action_divergence_fraction"] = (
        1.0 - causal_materialized / causal_full if causal_full else 0.0
    )
    candidate["saving_through_first_divergence_fraction"] = candidate[
        "saving_before_first_action_divergence_fraction"
    ]


def pareto_frontier(
    rows: Iterable[Mapping[str, Any]], *, quality_key: str
) -> list[dict[str, Any]]:
    eligible = [
        dict(row) for row in rows
        if row.get(quality_key) is not None and row.get("saving_fraction") is not None
    ]
    frontier = []
    for row in eligible:
        dominated = any(
            other is not row
            and float(other["saving_fraction"]) >= float(row["saving_fraction"])
            and float(other[quality_key]) >= float(row[quality_key])
            and (
                float(other["saving_fraction"]) > float(row["saving_fraction"])
                or float(other[quality_key]) > float(row[quality_key])
            )
            for other in eligible
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: float(row["saving_fraction"]))


def adaptive_decisions(
    frozen_rows: Sequence[Mapping[str, Any]],
    *,
    low_yield_saving: float,
    low_yield_quality: float,
    minimum_quality_for_promotion: float = .85,
    exceptional_saving: float = .20,
    minimum_independent_cohorts_for_promotion: int = 2,
    control_families: Iterable[str] = (),
    mechanism_only_families: Iterable[str] = (),
    bounded_probe_families: Iterable[str] = (),
) -> list[dict[str, Any]]:
    control_families = set(control_families)
    mechanism_only_families = set(mechanism_only_families)
    bounded_probe_families = set(bounded_probe_families)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        if row.get("strategy_family") != "full" and row.get("adaptive_eligible", True):
            groups[str(row.get("strategy_coordinate", row.get("strategy")))].append(row)
    by_cohort: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        if row.get("adaptive_eligible", True):
            by_cohort[str(row.get("cohort"))].append(row)
    frontier_by_cohort = {
        cohort: {
            str(row.get("strategy_coordinate", row["strategy"]))
            for row in pareto_frontier(rows, quality_key="quality_proxy")
        }
        for cohort, rows in by_cohort.items()
    }
    decisions = []
    for strategy, rows in sorted(groups.items()):
        independent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            independent[str(row.get("independence_key", row.get("cohort")))].append(row)
        # A suffix and its enclosing full-trajectory cohort are overlapping
        # evidence. Within each task/trajectory cluster retain the broadest
        # decision coverage, averaging only exact-coverage repeats.
        representatives = []
        collapsed = 0
        for cluster in independent.values():
            broadest = max(int(row.get("decisions") or 0) for row in cluster)
            selected = [
                row for row in cluster
                if int(row.get("decisions") or 0) == broadest
            ]
            representatives.append(selected)
            collapsed += len(cluster) - len(selected)
        saving = mean(
            mean(float(row["saving_fraction"]) for row in cluster)
            for cluster in representatives
        )
        quality = mean(
            mean(float(row["quality_proxy"]) for row in cluster)
            for cluster in representatives
        )
        families = {str(row.get("strategy_family")) for row in rows}
        if not saving:
            action, reason = "abstention_control", "exact fail-closed arm with no observed saving"
        elif families & control_families:
            action, reason = "control_only", "retain as matched baseline; do not build combinations from it"
        elif families & mechanism_only_families:
            action, reason = "mechanism_only", "retain to test a declared mechanism, not as a profile candidate"
        elif families & bounded_probe_families:
            action, reason = "bounded_autonomous_probe", "plausible information-preserving design; do not expand tasks until task quality passes"
        elif quality < low_yield_quality:
            action, reason = "stop", "proxy quality is below the absolute screening floor"
        elif saving < low_yield_saving:
            action, reason = "stop", "saving is below the minimum campaign yield"
        elif quality < minimum_quality_for_promotion and saving < exceptional_saving:
            action, reason = "stop", "proxy quality falls too quickly for the observed saving"
        elif any(
            strategy not in frontier_by_cohort.get(str(row.get("cohort")), set())
            for row in rows
        ):
            action, reason = "do_not_combine", "dominated in at least one observed matched cohort"
        elif len(independent) < minimum_independent_cohorts_for_promotion:
            action, reason = "replicate_frozen", "nondominated but observed in too few independent cohorts"
        elif quality < minimum_quality_for_promotion:
            action, reason = "bounded_autonomous_probe", "exceptional saving clears the absolute floor but not the promotion-quality gate"
        else:
            action, reason = "promote", "currently nondominated; requires autonomous qualification"
        decisions.append({
            "strategy": strategy,
            "frozen_observations": len(rows),
            "independent_cohorts": len(independent),
            "overlapping_observations_collapsed": collapsed,
            "mean_saving_fraction": saving,
            "mean_quality_proxy": quality,
            "decision": action,
            "reason": reason,
        })
    return decisions


def _strip_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "trace"}


def _official_quality(row: Mapping[str, Any]) -> bool | None:
    """Return a grade only when the official grader produced a valid outcome."""

    if "official_score" in row:
        value = row.get("official_score")
        return bool(value) if value is not None else None
    value = row.get("official_resolved")
    return bool(value) if value is not None else None


def summarize_autonomous(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate runs without treating repeats of one task as new tasks."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["strategy"])].append(row)
    summaries = []
    for strategy, strategy_rows in sorted(grouped.items()):
        by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in strategy_rows:
            by_task[str(row["task_id"])].append(row)
        task_rates = []
        solution_state_rates = []
        task_savings = []
        task_reacquisition_adjusted_savings = []
        for task_rows in by_task.values():
            valid = [
                value for value in (_official_quality(row) for row in task_rows)
                if value is not None
            ]
            if valid:
                task_rates.append(mean(valid))
            solution_valid = [
                bool(row["solution_state_score"])
                for row in task_rows
                if row.get("solution_state_score") is not None
            ]
            if solution_valid:
                solution_state_rates.append(mean(solution_valid))
            task_full = sum(int(row["cumulative_full_tokens"]) for row in task_rows)
            task_materialized = sum(
                int(row["cumulative_materialized_tokens"]) for row in task_rows
            )
            if task_full:
                task_savings.append(1 - task_materialized / task_full)
                task_reacquired = sum(
                    int(row.get("reacquired_observation_tokens") or 0)
                    for row in task_rows
                )
                task_reacquisition_adjusted_savings.append(
                    (task_full - task_materialized - task_reacquired) / task_full
                )
        paired = [
            row for row in strategy_rows
            if row.get("paired_net_saving_fraction") is not None
        ]
        qualified = [row for row in paired if row.get("efficiency_qualified")]
        baseline_success = [
            row for row in paired if row.get("baseline_official_score") is True
        ]
        full = sum(int(row["cumulative_full_tokens"]) for row in strategy_rows)
        materialized = sum(
            int(row["cumulative_materialized_tokens"]) for row in strategy_rows
        )
        reacquired_observation_tokens = sum(
            int(row.get("reacquired_observation_tokens") or 0)
            for row in strategy_rows
        )
        resolution_ci = (
            _cluster_bootstrap_mean_ci(task_rates) if task_rates else (None, None)
        )
        solution_state_ci = (
            _cluster_bootstrap_mean_ci(solution_state_rates)
            if solution_state_rates else (None, None)
        )
        saving_ci = (
            _cluster_bootstrap_mean_ci(task_savings) if task_savings else (None, None)
        )
        summaries.append({
            "strategy": strategy,
            "runs": len(strategy_rows),
            "task_count": len(by_task),
            "graded_task_count": len(task_rates),
            "task_ids": ";".join(sorted(by_task)),
            "official_success_runs": sum(
                _official_quality(row) is True for row in strategy_rows
            ),
            "official_grader_error_runs": sum(
                bool(row.get("official_error")) for row in strategy_rows
            ),
            "macro_task_resolution": mean(task_rates) if task_rates else None,
            "macro_task_resolution_ci95_lower": resolution_ci[0],
            "macro_task_resolution_ci95_upper": resolution_ci[1],
            "auxiliary_grade_available_runs": sum(
                row.get("auxiliary_resolved") is not None for row in strategy_rows
            ),
            "auxiliary_success_runs": sum(
                row.get("auxiliary_resolved") is True for row in strategy_rows
            ),
            "solution_state_known_runs": sum(
                row.get("solution_state_score") is not None for row in strategy_rows
            ),
            "macro_task_solution_state_resolution": (
                mean(solution_state_rates) if solution_state_rates else None
            ),
            "macro_task_solution_state_ci95_lower": solution_state_ci[0],
            "macro_task_solution_state_ci95_upper": solution_state_ci[1],
            "workload_gross_saving_ratio_of_sums": 1 - materialized / full,
            "macro_task_gross_saving": mean(task_savings),
            "macro_task_gross_saving_ci95_lower": saving_ci[0],
            "macro_task_gross_saving_ci95_upper": saving_ci[1],
            "mean_run_gross_saving": mean(
                float(row["saving_fraction"]) for row in strategy_rows
            ),
            "reacquired_observation_tokens": reacquired_observation_tokens,
            "workload_reacquisition_adjusted_gross_saving_ratio_of_sums": (
                (full - materialized - reacquired_observation_tokens) / full
            ),
            "macro_task_reacquisition_adjusted_gross_saving": mean(
                task_reacquisition_adjusted_savings
            ),
            "paired_runs": len(paired),
            "efficiency_qualified_pairs": len(qualified),
            "paired_baseline_successes": len(baseline_success),
            "paired_preserved_successes": sum(
                _official_quality(row) is True for row in baseline_success
            ),
            "paired_success_preservation": (
                mean(_official_quality(row) is True for row in baseline_success)
                if baseline_success else None
            ),
            "mean_paired_net_saving_all": (
                mean(float(row["paired_net_saving_fraction"]) for row in paired)
                if paired else None
            ),
            "mean_failure_aware_paired_net_saving": (
                mean(float(row["failure_aware_paired_net_saving_fraction"])
                     for row in paired)
                if paired else None
            ),
            "mean_paired_total_model_token_saving": (
                mean(float(row["paired_total_model_token_saving_fraction"])
                     for row in paired
                     if row.get("paired_total_model_token_saving_fraction") is not None)
                if any(row.get("paired_total_model_token_saving_fraction") is not None
                       for row in paired) else None
            ),
            "mean_tool_call_delta_all": (
                mean(float(row["tool_call_delta"]) for row in paired)
                if paired else None
            ),
            "mean_paired_net_saving_successful_pairs": (
                mean(float(row["paired_net_saving_fraction"]) for row in qualified)
                if qualified else None
            ),
            "mean_tool_call_delta_successful_pairs": (
                mean(float(row["tool_call_delta"]) for row in qualified)
                if qualified else None
            ),
        })
    return summaries


def summarize_full_repeat_controls(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expose FULL-to-FULL instability separately from policy effects."""

    repeats = [
        row for row in rows if row.get("pairing_role") == "full_repeat_control"
    ]
    return [{
        "task_id": row["task_id"],
        "model": row["model"],
        "pair_id": row.get("pair_id"),
        "baseline_official_outcome": row.get("baseline_official_outcome"),
        "repeat_official_outcome": row.get("official_outcome"),
        "endpoint_agreement": (
            row.get("baseline_official_outcome") == row.get("official_outcome")
        ),
        "exact_action_trajectory": (
            row.get("first_action_divergence") is None
            and row.get("call_delta") == 0
        ),
        "first_action_divergence": row.get("first_action_divergence"),
        "first_action_divergence_kind": row.get("first_action_divergence_kind"),
        "baseline_calls": row.get("calls") - row.get("call_delta", 0),
        "repeat_calls": row.get("calls"),
        "source": row.get("source"),
    } for row in repeats]


def _cluster_bootstrap_mean_ci(
    values: Sequence[float], *, samples: int = 5000, seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval over task-level means, not pseudo-Bernoulli runs."""

    if not values:
        raise ValueError("bootstrap interval requires at least one task")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    draws = sorted(
        mean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return (
        float(draws[int(.025 * (samples - 1))]),
        float(draws[int(.975 * (samples - 1))]),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row if key not in {"trace", "source_sha256"}})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)


def _plot(output: Path, frozen: Sequence[Mapping[str, Any]], autonomous: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    def short(row: Mapping[str, Any]) -> str:
        family = str(row.get("strategy_family"))
        names = {
            "full": "FULL",
            "dag_certified_exclusion": "DAG-cert",
            "matched_token_tail": "token tail",
            "h1_search_consumed": "H1",
            "h2a_write_current_read": "H2a",
            "h2b_verified_write": "H2b",
            "h3_read_superseded": "H3",
            "h4_working_set": "H4",
            "safe2_h1_h3": "H1+H3",
            "h2a_h3": "H2a+H3",
            "all_h1_h2a_h2b_h3_h4": "all guarded",
            "tool_structured_evidence": "structured",
            "tool_matched_span": "matched span",
            "tool_head_tail": "head/tail",
        }
        label = names.get(family, family)
        treatment = str(row.get("strategy"))
        for parameter in ("Kf", "Kw", "Kr", "Kx"):
            marker = f"{parameter}="
            if marker in treatment:
                label += f" {marker}{treatment.split(marker, 1)[1].split(';', 1)[0].split(',', 1)[0]}"
        if "threshold=" in treatment:
            threshold = treatment.split("threshold=", 1)[1].split(";", 1)[0]
            label += f" t={threshold}"
        return label

    by_cohort: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frozen:
        by_cohort[str(row["cohort"])].append(row)
    if by_cohort:
        columns = 2
        dense_count = sum(len(rows) > 6 for rows in by_cohort.values())
        figure_rows = math.ceil((len(by_cohort) + dense_count) / columns)
        fig, axes = plt.subplots(
            figure_rows, columns, figsize=(12, 4 * figure_rows), squeeze=False,
        )
        dense_legends: list[tuple[str, list[str]]] = []
        for axis, (cohort, rows) in zip(axes.flat, sorted(by_cohort.items())):
            dense = len(rows) > 6
            if dense:
                dense_legends.append((cohort, [short(row) for row in rows]))
            for index, row in enumerate(rows):
                x = 100 * float(row["saving_fraction"])
                y = 100 * float(row["quality_proxy"])
                axis.scatter(x, y, s=38)
                label = str(index + 1) if dense else short(row)
                y_offset = 5 if dense else (5, -10, 12, -17)[index % 4]
                treatment = str(row.get("strategy"))
                if not dense and "threshold=256" in treatment:
                    y_offset = 8
                elif not dense and "threshold=512" in treatment:
                    y_offset = -12
                axis.annotate(label, (x, y), xytext=(4, y_offset),
                              textcoords="offset points", fontsize=6.5)
            by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in rows:
                by_family[str(row.get("strategy_family"))].append(row)
            for family_rows in by_family.values():
                if len(family_rows) > 1:
                    ordered = sorted(
                        family_rows,
                        key=lambda row: float(row["saving_fraction"]),
                    )
                    axis.plot(
                        [100 * float(row["saving_fraction"]) for row in ordered],
                        [100 * float(row["quality_proxy"]) for row in ordered],
                        linewidth=.8, alpha=.45,
                    )
            axis.set_title(cohort.replace("_", " "), fontsize=10)
            axis.set_xlabel("Materialized-token saving (%)")
            axis.set_ylabel("Conservative action agreement (%)")
            axis.set_ylim(0, 107)
            axis.grid(alpha=.25)
        unused = list(axes.flat[len(by_cohort):])
        for axis, (cohort, labels) in zip(unused, dense_legends):
            axis.axis("off")
            axis.text(.02, .98, cohort.replace("_", " ") + " labels\n\n" +
                      "\n".join(f"{index}. {label}" for index, label in enumerate(labels, 1)),
                      va="top", fontsize=8, transform=axis.transAxes)
        for axis in unused[len(dense_legends):]:
            axis.axis("off")
        fig.suptitle(
            "Frozen screening by matched cohort (proxy, not task accuracy)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, .97))
        save(fig, "frozen_saving_vs_action_agreement")

    def accuracy_plot(
        rows: Sequence[Mapping[str, Any]], metric: str, filename: str, xlabel: str,
        *, quality_key: str = "official_score", ylabel: str = "Official task resolution (%)",
        title: str = "Autonomous quality frontier",
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            quality = (
                _official_quality(row) if quality_key == "official_score"
                else row.get(quality_key)
            )
            if row.get(metric) is not None and quality is not None:
                grouped[str(row["strategy"])].append(row)
        for strategy, strategy_rows in grouped.items():
            if metric == "saving_fraction":
                numerator = sum(int(row["cumulative_materialized_tokens"])
                                for row in strategy_rows)
                denominator = sum(int(row["cumulative_full_tokens"])
                                  for row in strategy_rows)
                x = 100 * (1 - numerator / denominator)
            else:
                x = 100 * mean(float(row[metric]) for row in strategy_rows)
            task_rates: dict[str, list[bool]] = defaultdict(list)
            for row in strategy_rows:
                quality = (
                    _official_quality(row) if quality_key == "official_score"
                    else row.get(quality_key)
                )
                if quality is not None:
                    task_rates[str(row["task_id"])].append(bool(quality))
            task_means = [mean(values) for values in task_rates.values()]
            n = len(task_means)
            rate = mean(task_means)
            lower, upper = _cluster_bootstrap_mean_ci(task_means)
            ax.errorbar(
                x, 100 * rate,
                yerr=[[100 * (rate - lower)], [100 * (upper - rate)]],
                fmt="o", capsize=3,
            )
            ax.annotate(
                f"{short(strategy_rows[0])} (tasks={n}, runs={len(strategy_rows)})",
                (x, 100 * rate),
                xytext=(4, 4), textcoords="offset points", fontsize=7,
            )
        if not grouped:
            ax.text(.5, .5, "No qualifying observations", ha="center", va="center",
                    transform=ax.transAxes, color="0.4")
        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(-5, 105), title=title)
        ax.grid(alpha=.25)
        save(fig, filename)

    accuracy_plot(
        autonomous, "saving_fraction", "autonomous_saving_vs_accuracy",
        "Within-run history-token saving (%)",
    )
    accuracy_plot(
        autonomous, "reacquisition_adjusted_gross_saving_fraction",
        "autonomous_reacquisition_adjusted_saving_vs_accuracy",
        "Reacquisition-adjusted history-token saving (%)",
    )
    accuracy_plot(
        autonomous, "saving_fraction", "autonomous_saving_vs_workspace_capability",
        "Within-run history-token saving (%)",
        quality_key="solution_state_score",
        ylabel="Resolving solution-state evidence (%)",
        title="Autonomous workspace-capability frontier",
    )
    accuracy_plot(
        autonomous, "saving_fraction", "autonomous_saving_vs_auxiliary_workspace",
        "Within-run history-token saving (%)",
        quality_key="auxiliary_resolved",
        ylabel="Auxiliary workspace resolution (%)",
        title="Auxiliary workspace outcome (available grades only)",
    )

    paired = [row for row in autonomous if row.get("tool_call_delta") is not None]
    accuracy_plot(
        paired, "paired_net_saving_fraction", "autonomous_net_saving_vs_accuracy",
        "Paired end-to-end input-token saving (%)",
    )
    accuracy_plot(
        paired, "failure_aware_paired_net_saving_fraction",
        "autonomous_failure_aware_saving_vs_accuracy",
        "Failure-aware paired input-token saving (%)",
    )

    def tool_delta_plot(metric: str, filename: str, xlabel: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
        for axis, successful_only in zip(axes, (False, True)):
            rows = [
                row for row in paired
                if row.get(metric) is not None
                and (not successful_only or row.get("efficiency_qualified"))
            ]
            for row in rows:
                marker = "o" if _official_quality(row) is True else "x"
                axis.scatter(100 * float(row[metric]), float(row["tool_call_delta"]),
                             s=48, marker=marker)
                axis.annotate(str(row["strategy"]),
                              (100 * float(row[metric]), float(row["tool_call_delta"])),
                              xytext=(4, 4), textcoords="offset points", fontsize=7)
            if not successful_only and rows:
                axis.text(
                    .02, .02, "o resolved candidate; x failed candidate\n"
                    "failed points are diagnostic, not efficiency claims",
                    transform=axis.transAxes, fontsize=7, va="bottom",
                )
            if not rows:
                axis.text(.5, .5, "No qualifying observations", ha="center", va="center",
                          transform=axis.transAxes, color="0.4")
            axis.axhline(0, color="black", linewidth=.8)
            axis.set_xlabel(xlabel)
            axis.set_title("Successful runs" if successful_only else "All paired runs")
            axis.grid(alpha=.25)
        axes[0].set_ylabel("Tool-call delta vs paired FULL")
        save(fig, filename)

    tool_delta_plot(
        "saving_fraction", "autonomous_gross_saving_vs_tool_call_delta",
        "Within-run history-token saving (%)",
    )
    tool_delta_plot(
        "reacquisition_adjusted_gross_saving_fraction",
        "autonomous_reacquisition_adjusted_saving_vs_tool_call_delta",
        "Reacquisition-adjusted history-token saving (%)",
    )
    tool_delta_plot(
        "paired_net_saving_fraction", "autonomous_net_saving_vs_tool_call_delta",
        "Paired end-to-end input-token saving (%)",
    )

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for row in paired:
        first = row.get("first_action_divergence")
        censored = row.get("divergence_right_censored_at")
        y = first if first is not None else censored
        if y is None:
            continue
        marker = (
            "x" if _official_quality(row) is not True
            else "o" if first is not None else "^"
        )
        causal_saving = row.get("saving_before_first_action_divergence_fraction")
        if causal_saving is None:
            continue
        ax.scatter(100 * float(causal_saving), int(y), s=48, marker=marker)
        prefix = "" if first is not None else ">"
        kind = row.get("first_action_divergence_kind") or "right-censored"
        task = str(row["task_id"]).replace("__", "-").split("-")[-1]
        ax.annotate(f"{task} | {short(row)} | {prefix}{y}, {kind}",
                    (100 * float(causal_saving), int(y)),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Cumulative saving through divergence/censoring (%)",
           ylabel="First divergent tool action (call)",
           title="First action divergence by task")
    ax.text(
        .02, .02,
        "o divergence in resolved run; ^ resolved/right-censored; x failed run",
        transform=ax.transAxes, fontsize=7, va="bottom",
    )
    ax.grid(alpha=.25)
    save(fig, "autonomous_saving_vs_first_action_divergence")


def build(spec_path: Path, output: Path) -> dict[str, Any]:
    spec = _read_json(spec_path)
    frozen: list[dict[str, Any]] = []
    for item in spec.get("frozen_comparisons") or ():
        frozen.extend(load_frozen_comparison(
            _resolve(spec_path, str(item["path"])), cohort=str(item["cohort"]),
            adaptive_eligible=bool(item.get("adaptive_eligible", True)),
        ))
    autonomous_items = list(spec.get("autonomous_runs") or ())
    autonomous_names = [str(item["name"]) for item in autonomous_items]
    if len(autonomous_names) != len(set(autonomous_names)):
        raise ValueError("duplicate autonomous run names in curve specification")
    autonomous_by_name = {
        str(item["name"]): load_autonomous_run(
            _resolve(spec_path, str(item["path"])),
            auxiliary_grade_path=(
                _resolve(spec_path, str(item["auxiliary_grade"]))
                if item.get("auxiliary_grade") else None
            ),
        )
        for item in autonomous_items
    }
    paired_candidates: set[str] = set()
    for pair in spec.get("autonomous_pairs") or ():
        candidate_name = str(pair["candidate"])
        if candidate_name in paired_candidates:
            raise ValueError(f"autonomous candidate paired more than once: {candidate_name}")
        paired_candidates.add(candidate_name)
        pair_autonomous(
            autonomous_by_name[candidate_name],
            autonomous_by_name[str(pair["baseline"])],
            allow_legacy_pair=bool(pair.get("allow_legacy_pair", False)),
            pairing_reason=pair.get("pairing_reason"),
        )
    autonomous = list(autonomous_by_name.values())
    unpaired_policies = [
        name for name, row in autonomous_by_name.items()
        if row.get("strategy_family") != "full" and name not in paired_candidates
    ]
    if unpaired_policies:
        raise ValueError(
            "non-FULL autonomous runs require paired FULL controls: "
            + ", ".join(sorted(unpaired_policies))
        )
    for row in autonomous:
        if row.get("strategy_family") == "full":
            continue
        controls = [
            control for control in autonomous
            if control.get("strategy_family") == "full"
            and control.get("task_id") == row.get("task_id")
            and control.get("model") == row.get("model")
        ]
        if len(controls) < 2:
            raise ValueError(
                "autonomous policy arms require two contemporaneous FULL controls "
                f"for task/model: {row.get('task_id')}/{row.get('model')}"
            )
    thresholds = spec.get("adaptive_pruning") or {}
    decisions = adaptive_decisions(
        frozen,
        low_yield_saving=float(thresholds.get("low_yield_saving_fraction", .02)),
        low_yield_quality=float(thresholds.get("low_yield_quality_proxy", .80)),
        minimum_quality_for_promotion=float(
            thresholds.get("minimum_quality_proxy_for_promotion", .85)
        ),
        exceptional_saving=float(thresholds.get("exceptional_saving_fraction", .20)),
        minimum_independent_cohorts_for_promotion=int(
            thresholds.get("minimum_independent_cohorts_for_promotion", 2)
        ),
        control_families=thresholds.get("control_families") or (),
        mechanism_only_families=thresholds.get("mechanism_only_families") or (),
        bounded_probe_families=thresholds.get("bounded_probe_families") or (),
    )
    autonomous_summary = summarize_autonomous(autonomous)
    full_repeat_summary = summarize_full_repeat_controls(autonomous)
    result = {
        "schema_version": 1,
        "study": "paper8_5_agent_memory_tradeoff_curves",
        "metric_contract": {
            "primary_quality": "autonomous official task resolution",
            "frozen_quality_proxy": "conservative next-action agreement versus repeat-qualified FULL",
            "gross_saving": "one minus materialized/full tokens along the candidate trajectory",
            "reacquisition_adjusted_gross_saving": "gross saved history tokens minus the directly observed token size of tool observations caused by actions that reacquire excluded resources, divided by full-history tokens; a diagnostic lower bound that is not added to paired net saving",
            "paired_net_saving": "one minus candidate materialized message-content input tokens / paired FULL materialized message-content input tokens; includes call-count divergence but excludes completion and chat-template tokens",
            "failure_aware_paired_net_saving": "paired net saving when both runs resolve; otherwise the failed candidate is charged at least the paired FULL input workload so early termination cannot appear efficient",
            "paired_total_model_token_saving": "one minus candidate / paired FULL for materialized input plus endpoint-reported completion tokens; emitted only with complete usage coverage",
            "successful_tool_delta": "tool-call delta is included in the successful-only curve only when both candidate and paired FULL resolve officially",
            "accuracy_aggregation": "resolution is macro-averaged by task; repeated runs do not increase the task denominator",
            "grader_errors": "a definitive Boolean unresolved grade remains a task failure, including malformed-patch/apply errors; only runs lacking a definitive Boolean grade are excluded",
            "auxiliary_quality": "separately labelled official grade of a provenance-checked terminal workspace patch; never replaces primary submission quality",
            "solution_state_quality": "resolved primary submission, otherwise an available auxiliary workspace grade; unavailable failed-primary workspaces remain unknown",
            "uncertainty": "task-clustered percentile bootstrap over per-task resolution means",
            "frozen_independence": "unique task/model/trajectory identity, not analyst cohort label; overlapping suffixes are collapsed to broadest coverage",
            "pre_divergence_saving": "cumulative saving over requests strictly before the first divergent action; the divergent request itself is excluded",
        },
        "frozen_observations": [_strip_trace(row) for row in frozen],
        "autonomous_observations": [_strip_trace(row) for row in autonomous],
        "autonomous_strategy_summary": autonomous_summary,
        "full_repeat_control_summary": full_repeat_summary,
        "frozen_pareto_frontier_by_cohort": {
            cohort: [_strip_trace(row) for row in pareto_frontier(
                cohort_rows, quality_key="quality_proxy"
            )]
            for cohort, cohort_rows in sorted({
                str(row["cohort"]): [
                    member for member in frozen
                    if str(member["cohort"]) == str(row["cohort"])
                ]
                for row in frozen
            }.items())
        },
        "adaptive_decisions": decisions,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "tradeoff_curves.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "frozen_observations.csv", frozen)
    _write_csv(output / "autonomous_observations.csv", autonomous)
    _write_csv(output / "autonomous_strategy_summary.csv", autonomous_summary)
    _write_csv(output / "full_repeat_control_summary.csv", full_repeat_summary)
    _write_csv(output / "adaptive_decisions.csv", decisions)
    _plot(output, frozen, autonomous)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.spec.resolve(), args.output.resolve())
    print(json.dumps({
        "frozen_observations": len(result["frozen_observations"]),
        "autonomous_observations": len(result["autonomous_observations"]),
        "output": str(args.output.resolve()),
    }))


if __name__ == "__main__":
    main()
