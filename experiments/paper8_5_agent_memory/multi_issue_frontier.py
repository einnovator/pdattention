"""Strict reduction for Paper 8.5 multi-issue quality--saving frontiers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PAIRING_FIELDS = (
    "pair_id",
    "sequence_family_id",
    "sequence_digest",
    "agent_id",
    "agent_revision",
    "model_revision",
    "tokenizer_revision",
    "harness_revision",
    "decoding_digest",
    "workspace_schedule_digest",
)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_strategy_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("multi-issue strategy registry requires schema_version=1")
    strategies = registry.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("strategy registry has no strategies")
    ids = [str(row.get("id") or "") for row in strategies]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("strategy IDs must be unique and non-empty")
    known = set(ids)
    for row in strategies:
        for parent in row.get("parents") or ():
            if parent not in known:
                raise ValueError(f"unknown parent strategy {parent!r}")
        required = {"family", "rationale", "primary_comparison", "status"}
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(f"{row['id']}: missing {', '.join(missing)}")
    if registry.get("issue_counts") != list(range(1, 11)):
        raise ValueError("registry must preserve the locked 1--10 issue axis")
    target = registry.get("primary_target") or {}
    saving = target.get("failure_aware_saving_fraction") or {}
    if saving.get("minimum") != 0.30 or saving.get("maximum") != 0.50:
        raise ValueError("registry must freeze the primary saving region at 30--50%")
    return registry


def _validate_run(run: Mapping[str, Any], known_strategies: set[str]) -> None:
    if run.get("schema_version") != 1:
        raise ValueError("multi-issue run requires schema_version=1")
    missing = [field for field in PAIRING_FIELDS if not run.get(field)]
    if missing:
        raise ValueError("run lacks strict pairing identity: " + ", ".join(missing))
    strategy = str(run.get("strategy_id") or "")
    if strategy not in known_strategies:
        raise ValueError(f"unregistered strategy {strategy!r}")
    issue_count = int(run.get("issue_count") or 0)
    if issue_count not in set(range(1, 11)):
        raise ValueError("issue_count must be in 1..10")
    ordered = run.get("ordered_instance_ids")
    issues = run.get("issues")
    if not isinstance(ordered, list) or len(ordered) != issue_count:
        raise ValueError("ordered_instance_ids do not match issue_count")
    if not isinstance(issues, list) or len(issues) != issue_count:
        raise ValueError("issue rows do not match issue_count")
    config = run.get("strategy_config")
    if not isinstance(config, Mapping):
        raise ValueError("strategy_config must be an explicit mapping")
    config_id = str(run.get("strategy_config_id") or "")
    if not config_id:
        raise ValueError("strategy_config_id must be explicit")
    expected_config_digest = _digest(config)
    if run.get("strategy_config_digest") != expected_config_digest:
        raise ValueError("strategy_config_digest does not match strategy_config")
    if [row.get("issue_index") for row in issues] != list(range(1, issue_count + 1)):
        raise ValueError("issue rows must be a complete ordered 1..N sequence")
    for row in issues:
        if not isinstance(row.get("resolved"), bool):
            raise ValueError("every issue requires a boolean official resolved outcome")
        if "official_error" in row and not isinstance(row["official_error"], bool):
            raise ValueError("official_error must be boolean when present")
        if "evidence_admissible" in row and not isinstance(
            row["evidence_admissible"], bool
        ):
            raise ValueError("evidence_admissible must be boolean when present")
        if "same_prefix_full_qualified" in row and not isinstance(
            row["same_prefix_full_qualified"], bool
        ):
            raise ValueError("same_prefix_full_qualified must be boolean when present")
        for field in ("selected_input_tokens", "calls", "rediscovery_calls"):
            value = row.get(field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"every issue requires non-negative integer {field}")
        if not isinstance(row.get("first_action_diverged"), bool):
            raise ValueError("every issue requires a boolean first_action_diverged")
        for field in (
            "identical_input_first_action_divergence",
            "selection_active_at_first_action_divergence",
        ):
            if field in row and not isinstance(row[field], bool):
                raise ValueError(f"{field} must be boolean when present")
        for field in (
            "selected_tokens_before_divergence_or_terminal",
            "full_tokens_before_divergence_or_terminal",
        ):
            value = row.get(field)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"every issue requires non-negative integer {field}")


def _totals(run: Mapping[str, Any]) -> dict[str, Any]:
    issues = run["issues"]
    return {
        "resolved": sum(int(row["resolved"]) for row in issues),
        "all_resolved": all(row["resolved"] for row in issues),
        "tokens": sum(int(row["selected_input_tokens"]) for row in issues),
        "calls": sum(int(row["calls"]) for row in issues),
        "resolved_calls": sum(int(row["calls"]) for row in issues if row["resolved"]),
        "rediscovery_calls": sum(int(row["rediscovery_calls"]) for row in issues),
        # Historical frozen ledgers predate this field and are treated as
        # admissible.  New autonomous ledgers always emit it explicitly.
        "evidence_admissible": all(
            bool(row.get("evidence_admissible", True)) for row in issues
        ),
        "official_error_count": sum(
            int(bool(row.get("official_error", False))) for row in issues
        ),
    }


def _same_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in PAIRING_FIELDS)


def _saving(candidate_tokens: int, baseline_tokens: int) -> float:
    if baseline_tokens <= 0:
        raise ValueError("paired baseline must consume positive input tokens")
    return 1.0 - candidate_tokens / baseline_tokens


def _failure_aware_saving(
    candidate_run: Mapping[str, Any], baseline_run: Mapping[str, Any]
) -> float:
    charged = sum(
        int(row["selected_input_tokens"]) for row in candidate_run["issues"]
    )
    baseline_tokens = sum(
        int(row["selected_input_tokens"]) for row in baseline_run["issues"]
    )
    lost_baseline_success = any(
        baseline["resolved"] and not candidate["resolved"]
        for candidate, baseline in zip(candidate_run["issues"], baseline_run["issues"])
    )
    if lost_baseline_success:
        charged = max(charged, baseline_tokens)
    return _saving(charged, baseline_tokens)


def _pre_divergence_saving(run: Mapping[str, Any]) -> float | None:
    selected = sum(
        int(row["selected_tokens_before_divergence_or_terminal"])
        for row in run["issues"]
    )
    full = sum(
        int(row["full_tokens_before_divergence_or_terminal"])
        for row in run["issues"]
    )
    return _saving(selected, full) if full else None


def _successful_call_delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[int | None, int]:
    deltas = [
        int(candidate_row["calls"]) - int(baseline_row["calls"])
        for candidate_row, baseline_row in zip(candidate["issues"], baseline["issues"])
        if candidate_row["resolved"] and baseline_row["resolved"]
    ]
    return (sum(deltas), len(deltas)) if deltas else (None, 0)


def reduce_multi_issue_runs(
    runs: Sequence[Mapping[str, Any]], registry: Mapping[str, Any]
) -> dict[str, Any]:
    """Pair every treatment with persistent FULL and optional fresh FULL."""

    known = {str(row["id"]) for row in registry["strategies"]}
    materialized = [dict(run) for run in runs]
    for run in materialized:
        _validate_run(run, known)
    cell_keys = [
        (
            run["pair_id"],
            run["session_mode"],
            run["strategy_id"],
            run["strategy_config_id"],
        )
        for run in materialized
    ]
    if len(cell_keys) != len(set(cell_keys)):
        raise ValueError("duplicate strategy/config cell within a paired sequence")

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in materialized:
        by_pair[str(run["pair_id"])].append(run)

    rows: list[dict[str, Any]] = []
    for pair_id, paired in by_pair.items():
        fresh = next(
            (
                row for row in paired
                if row["strategy_id"] == "S00_fresh_full"
                and row["session_mode"] == "fresh_per_issue"
            ),
            None,
        )
        persistent = next(
            (
                row for row in paired
                if row["strategy_id"] == "S01_persistent_full"
                and row["session_mode"] == "persistent"
            ),
            None,
        )
        if persistent is None:
            raise ValueError(f"{pair_id}: persistent FULL control is required")
        for run in paired:
            if (
                not _same_pair(run, persistent)
                or (fresh is not None and not _same_pair(run, fresh))
            ):
                raise ValueError(f"{pair_id}: strict pairing identity mismatch")
        fresh_totals = _totals(fresh) if fresh is not None else None
        persistent_totals = _totals(persistent)
        for run in paired:
            totals = _totals(run)
            is_treatment = run["strategy_id"] not in {
                "S00_fresh_full", "S01_persistent_full"
            }
            qualified_issue_count = sum(
                int(bool(issue.get("same_prefix_full_qualified", False)))
                for issue in run["issues"]
            )
            heuristic_attribution_admissible = bool(
                not is_treatment or qualified_issue_count == run["issue_count"]
            )
            if fresh is not None:
                fresh_success_calls, fresh_joint_successes = _successful_call_delta(
                    run, fresh
                )
            else:
                fresh_success_calls, fresh_joint_successes = None, 0
            persistent_success_calls, persistent_joint_successes = (
                _successful_call_delta(run, persistent)
            )
            failure_aware_vs_persistent = (
                _failure_aware_saving(run, persistent)
                if heuristic_attribution_admissible else None
            )
            lost_persistent_successes = (
                sum(
                    int(baseline["resolved"] and not candidate["resolved"])
                    for candidate, baseline in zip(
                        run["issues"], persistent["issues"]
                    )
                )
                if heuristic_attribution_admissible else None
            )
            gained_over_persistent = sum(
                int(candidate["resolved"] and not baseline["resolved"])
                for candidate, baseline in zip(run["issues"], persistent["issues"])
            )
            resolution_delta_vs_persistent = (
                totals["resolved"] - persistent_totals["resolved"]
            ) / run["issue_count"]
            target_region = bool(
                failure_aware_vs_persistent is not None
                and 0.30 <= failure_aware_vs_persistent <= 0.50
            )
            comparison_admissible = bool(
                totals["evidence_admissible"]
                and persistent_totals["evidence_admissible"]
                and (fresh_totals is None or fresh_totals["evidence_admissible"])
                and heuristic_attribution_admissible
            )
            rows.append({
                "pair_id": pair_id,
                "sequence_family_id": run["sequence_family_id"],
                "sequence_id": run.get("sequence_id"),
                "sequence_digest": run["sequence_digest"],
                "sequence_stratum": run.get("sequence_stratum"),
                "agent_id": run["agent_id"],
                "agent_revision": run["agent_revision"],
                "model_revision": run["model_revision"],
                "tokenizer_revision": run["tokenizer_revision"],
                "harness_revision": run["harness_revision"],
                "issue_count": run["issue_count"],
                "session_mode": run["session_mode"],
                "strategy_id": run["strategy_id"],
                "strategy_config_id": run["strategy_config_id"],
                "strategy_config": dict(run["strategy_config"]),
                "strategy_config_digest": run["strategy_config_digest"],
                "official_resolution": totals["resolved"] / run["issue_count"],
                "all_issues_resolved": totals["all_resolved"],
                "resolved_issues": totals["resolved"],
                "selected_input_tokens": totals["tokens"],
                "calls": totals["calls"],
                "resolved_calls": totals["resolved_calls"],
                "rediscovery_calls": totals["rediscovery_calls"],
                "official_error_count": totals["official_error_count"],
                "evidence_admissible": totals["evidence_admissible"],
                "persistent_full_evidence_admissible": persistent_totals[
                    "evidence_admissible"
                ],
                "comparison_evidence_admissible": comparison_admissible,
                "same_prefix_full_qualified_issues": qualified_issue_count,
                "heuristic_attribution_admissible": (
                    heuristic_attribution_admissible
                ),
                "saving_vs_persistent_full": _saving(
                    totals["tokens"], persistent_totals["tokens"]
                ),
                "saving_vs_fresh_full": (
                    _saving(totals["tokens"], fresh_totals["tokens"])
                    if fresh_totals is not None else None
                ),
                "failure_aware_saving_vs_persistent_full": failure_aware_vs_persistent,
                "failure_aware_saving_vs_fresh_full": (
                    _failure_aware_saving(run, fresh)
                    if fresh is not None and heuristic_attribution_admissible
                    else None
                ),
                "resolution_delta_vs_persistent_full": resolution_delta_vs_persistent,
                "lost_persistent_full_successes": lost_persistent_successes,
                "gained_over_persistent_full": gained_over_persistent,
                "resolution_delta_vs_fresh_full": (
                    (totals["resolved"] - fresh_totals["resolved"])
                    / run["issue_count"] if fresh_totals is not None else None
                ),
                "calls_delta_vs_persistent_full": (
                    totals["calls"] - persistent_totals["calls"]
                ),
                "calls_delta_vs_fresh_full": (
                    totals["calls"] - fresh_totals["calls"]
                    if fresh_totals is not None else None
                ),
                "successful_calls_delta_vs_persistent_full": (
                    persistent_success_calls
                ),
                "successful_calls_delta_vs_fresh_full": fresh_success_calls,
                "jointly_resolved_issues_vs_persistent_full": (
                    persistent_joint_successes
                ),
                "jointly_resolved_issues_vs_fresh_full": fresh_joint_successes,
                "rediscovery_delta_vs_fresh_full": (
                    totals["rediscovery_calls"] - fresh_totals["rediscovery_calls"]
                    if fresh_totals is not None else None
                ),
                "cost_per_resolved_issue": (
                    totals["tokens"] / totals["resolved"]
                    if totals["resolved"] else None
                ),
                "first_divergence_preceding_saving": _pre_divergence_saving(run),
                "first_action_divergence_rate": sum(
                    int(row["first_action_diverged"]) for row in run["issues"]
                ) / run["issue_count"],
                "identical_input_first_action_divergence_rate": sum(
                    int(bool(row.get("identical_input_first_action_divergence", False)))
                    for row in run["issues"]
                ) / run["issue_count"],
                "selection_active_first_action_divergence_rate": sum(
                    int(bool(row.get("selection_active_at_first_action_divergence", False)))
                    for row in run["issues"]
                ) / run["issue_count"],
                "in_primary_saving_target": (
                    target_region if heuristic_attribution_admissible else None
                ),
                "discovery_primary_target_met": bool(
                    run["strategy_id"] not in {"S00_fresh_full", "S01_persistent_full"}
                    and comparison_admissible
                    and target_region
                    and lost_persistent_successes == 0
                    and resolution_delta_vs_persistent >= 0
                ),
                "confirmation_primary_target_met": None,
                "confirmation_note": (
                    "requires held-out ordered-sequence-clustered 95% interval"
                ),
                "run_digest": _digest(run),
            })
    return {
        "schema_version": 1,
        "study": "paper8_5_multi_issue_quality_saving_frontier",
        "uncertainty_unit": "ordered_issue_sequence",
        "pair_count": len(by_pair),
        "row_count": len(rows),
        "rows": rows,
    }


def _read_runs(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, Mapping):
        payload = payload.get("runs", [payload])
    if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
        raise ValueError(f"{path}: expected a run, a run list, or JSONL runs")
    return [dict(row) for row in payload]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce strictly paired Paper 8.5 multi-issue runs."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = [run for path in args.input for run in _read_runs(path)]
    result = reduce_multi_issue_runs(runs, load_strategy_registry(args.registry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
