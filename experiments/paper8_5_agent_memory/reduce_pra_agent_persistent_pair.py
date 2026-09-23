"""Reduce one paired PRA Agent persistent-session cohort into N-prefix rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolved_ids(report: Mapping[str, Any]) -> set[str]:
    values = report.get("resolved_ids", ())
    if not isinstance(values, list) or not all(isinstance(row, str) for row in values):
        raise ValueError("official report resolved_ids must be a string array")
    return set(values)


def _saving(denominator: int, numerator: int) -> float:
    if denominator < 0 or numerator < 0:
        raise ValueError("token counts cannot be negative")
    return 0.0 if denominator == 0 else 1.0 - numerator / denominator


def _assert_pair(full: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    fields = (
        "agent",
        "boundary_mode",
        "task_order",
        "model",
        "served_model",
        "model_revision",
        "tokenizer",
        "tokenizer_revision",
        "generation",
    )
    mismatched = [name for name in fields if full.get(name) != policy.get(name)]
    if mismatched:
        raise ValueError("paired manifests differ on: " + ", ".join(mismatched))
    if full.get("history_policy") != "full":
        raise ValueError("FULL comparator does not declare history_policy=full")
    if len(full.get("episodes", ())) != len(policy.get("episodes", ())):
        raise ValueError("paired manifests have different episode counts")


def reduce_pair(
    full_manifest: Mapping[str, Any],
    policy_manifest: Mapping[str, Any],
    full_report: Mapping[str, Any],
    policy_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return cumulative N-prefix accuracy, token, and call-cost curves."""

    _assert_pair(full_manifest, policy_manifest)
    full_resolved = _resolved_ids(full_report)
    policy_resolved = _resolved_ids(policy_report)
    full_episodes = list(full_manifest["episodes"])
    policy_episodes = list(policy_manifest["episodes"])
    rows: list[dict[str, Any]] = []
    full_calls = 0
    policy_calls = 0
    full_tokens = 0
    policy_full_tokens = 0
    policy_materialized_tokens = 0
    full_successes = 0
    policy_successes = 0
    preserved_successes = 0

    for index, (full_episode, policy_episode) in enumerate(
        zip(full_episodes, policy_episodes), start=1
    ):
        instance_id = str(full_episode["instance_id"])
        if instance_id != str(policy_episode["instance_id"]):
            raise ValueError(f"episode {index} identity mismatch")
        full_ok = instance_id in full_resolved
        policy_ok = instance_id in policy_resolved
        full_successes += int(full_ok)
        policy_successes += int(policy_ok)
        preserved_successes += int(full_ok and policy_ok)
        full_calls += int(full_episode["request_count"])
        policy_calls += int(policy_episode["request_count"])
        full_tokens += int(full_episode["cumulative_materialized_history_tokens"])
        policy_full_tokens += int(policy_episode["cumulative_full_history_tokens"])
        policy_materialized_tokens += int(
            policy_episode["cumulative_materialized_history_tokens"]
        )
        rows.append({
            "n": index,
            "instance_id": instance_id,
            "full_resolved": full_ok,
            "policy_resolved": policy_ok,
            "full_successes": full_successes,
            "policy_successes": policy_successes,
            "conditional_preserved_successes": preserved_successes,
            "conditional_preservation_fraction": (
                None if full_successes == 0
                else preserved_successes / full_successes
            ),
            "full_resolution_fraction": full_successes / index,
            "policy_resolution_fraction": policy_successes / index,
            "full_cumulative_materialized_tokens": full_tokens,
            "policy_cumulative_full_history_tokens": policy_full_tokens,
            "policy_cumulative_materialized_tokens": policy_materialized_tokens,
            "own_logical_saving_fraction": _saving(
                policy_full_tokens, policy_materialized_tokens
            ),
            "paired_workload_saving_fraction": _saving(
                full_tokens, policy_materialized_tokens
            ),
            "full_cumulative_requests": full_calls,
            "policy_cumulative_requests": policy_calls,
            "cumulative_request_delta": policy_calls - full_calls,
        })

    return {
        "schema_version": 1,
        "study": "paper8_5_pra_agent_persistent_pair",
        "agent": full_manifest["agent"],
        "model": full_manifest["model"],
        "model_revision": full_manifest.get("model_revision"),
        "boundary_mode": full_manifest["boundary_mode"],
        "full_session_id": full_manifest.get("session_id"),
        "policy_session_id": policy_manifest.get("session_id"),
        "policy": policy_manifest["history_policy"],
        "task_order": list(full_manifest["task_order"]),
        "prefix_rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--full-grade", required=True, type=Path)
    parser.add_argument("--policy-grade", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = reduce_pair(
        _load(args.full),
        _load(args.policy),
        _load(args.full_grade),
        _load(args.policy_grade),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
