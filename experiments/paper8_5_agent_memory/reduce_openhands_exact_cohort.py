"""Aggregate strict OpenHands pair reductions by unique task identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def reduce_cohort(paths: list[Path]) -> dict[str, Any]:
    pairs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    task_ids = [str(pair["instance_id"]) for pair in pairs]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("cohort contains overlapping task identities")
    if not all(pair.get("schema_version") == 3 for pair in pairs):
        raise ValueError("cohort requires schema-version-3 strict pair reductions")

    full_tokens = sum(int(pair["full"]["materialized_tokens"]) for pair in pairs)
    candidate_full = sum(int(pair["candidate"]["full_tokens"]) for pair in pairs)
    candidate_materialized = sum(
        int(pair["candidate"]["materialized_tokens"]) for pair in pairs
    )
    full_prompt = sum(int(pair["full"]["provider_prompt_tokens"]) for pair in pairs)
    candidate_prompt = sum(
        int(pair["candidate"]["provider_prompt_tokens"]) for pair in pairs
    )
    full_actions = sum(int(pair["full"]["actions"]) for pair in pairs)
    candidate_actions = sum(int(pair["candidate"]["actions"]) for pair in pairs)
    return {
        "schema_version": 1,
        "study": "paper8_5_cross_agent_exact_tokenizer_transfer_cohort",
        "agent": "openhands-sdk",
        "task_count": len(pairs),
        "task_ids": task_ids,
        "full_official_resolved": sum(
            bool(pair["full"]["official_resolved"]) for pair in pairs
        ),
        "candidate_official_resolved": sum(
            bool(pair["candidate"]["official_resolved"]) for pair in pairs
        ),
        "full_actions": full_actions,
        "candidate_actions": candidate_actions,
        "action_delta": candidate_actions - full_actions,
        "full_materialized_tokens": full_tokens,
        "candidate_own_full_tokens": candidate_full,
        "candidate_materialized_tokens": candidate_materialized,
        "own_logical_saving": (
            1 - candidate_materialized / candidate_full if candidate_full else 0.0
        ),
        "paired_input_saving": (
            1 - candidate_materialized / full_tokens if full_tokens else 0.0
        ),
        "paired_provider_prompt_saving": (
            1 - candidate_prompt / full_prompt if full_prompt else 0.0
        ),
        "mean_task_paired_input_saving": (
            sum(float(pair["paired_input_saving"]) for pair in pairs) / len(pairs)
            if pairs else 0.0
        ),
        "pair_qualifications": {
            str(pair["instance_id"]): str(pair["qualification"]) for pair in pairs
        },
        "qualification": (
            "pass" if pairs and all(pair.get("qualification") == "pass" for pair in pairs)
            else "fail"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce_cohort([path.resolve() for path in args.qualification])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
