"""Build one-causal-group-at-a-time frozen oracle add-back trials.

The queue diagnoses a policy divergence; it is not autonomous task-quality
evidence and the oracle interventions are never candidate deployment policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_queue(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("study") != "paper8_5_agent_memory_frozen_replay":
        raise ValueError("candidate is not a Paper 8.5 frozen replay")
    rows = candidate.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate has no replay rows")
    divergence = candidate.get("first_command_divergence")
    if divergence is None:
        divergent = next(
            (row for row in rows if not bool(row.get("action_valid", True))),
            None,
        )
        divergence = divergent.get("decision") if divergent else None
    if divergence is None:
        raise ValueError("candidate has no command or validity divergence to diagnose")
    row = next(
        (item for item in rows if int(item.get("decision", -1)) == int(divergence)),
        None,
    )
    if row is None:
        raise ValueError("divergence decision is absent from replay rows")
    exclusions = row.get("exclusions") or []
    trials = []
    seen = set()
    for exclusion in sorted(
        exclusions,
        key=lambda item: (
            int(item.get("excluded_tokens", 0)),
            str(item.get("causal_group_id", "")),
        ),
    ):
        group_id = str(exclusion.get("causal_group_id", ""))
        if not group_id or group_id in seen:
            continue
        seen.add(group_id)
        trials.append({
            "trial_id": f"addback-{len(trials) + 1:03d}",
            "decision": int(divergence),
            "oracle_addback_causal_group_ids": [group_id],
            "excluded_tokens": int(exclusion.get("excluded_tokens", 0)),
            "rule_id": exclusion.get("rule_id"),
            "classification": exclusion.get("classification"),
            "resource_ids": exclusion.get("resource_ids") or [],
            "witness_record_ids": exclusion.get("witness_record_ids") or [],
            "cli_arguments": [
                "--min-decision", str(divergence),
                "--max-decisions", "1",
                "--oracle-addback-group", group_id,
            ],
        })
    return {
        "schema_version": 1,
        "study": "paper8_5_frozen_oracle_addback_queue",
        "evidence_class": "diagnostic_oracle_not_autonomous_task_quality",
        "instance_id": candidate.get("instance_id"),
        "candidate_run_configuration_digest": candidate.get(
            "run_configuration_digest"
        ),
        "candidate_artifact_sha256": _digest(candidate),
        "divergence_decision": int(divergence),
        "ranking": "fewest_restored_tokens_first_then_causal_group_id",
        "intervention": "restore_exactly_one_complete_causal_group",
        "trial_count": len(trials),
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = build_queue(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "divergence_decision": result["divergence_decision"],
        "trial_count": result["trial_count"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
