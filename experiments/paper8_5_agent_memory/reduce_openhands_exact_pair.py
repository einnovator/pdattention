"""Reduce one exact-token OpenHands FULL/selective pair with strict gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"trace contains a non-object row: {path}")
    observed = [int(row.get("request_index") or 0) for row in rows]
    if observed != list(range(1, len(rows) + 1)):
        raise ValueError(f"noncontiguous request indices in {path}: {observed}")
    return rows


def _arm(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(path / "run_manifest.json")
    summary = _json(path / "event_summary.json")
    report = _json(path / "official_report.json")
    rows = _rows(path / "proxy_trace.jsonl")
    instance_id = str(manifest["instance_id"])
    resolved = instance_id in set(report.get("resolved_ids") or ())
    tokenizers = {str(row.get("tokenizer") or "") for row in rows}
    if len(tokenizers) != 1 or any("whitespace" in value for value in tokenizers):
        raise ValueError(f"arm lacks one exact tokenizer identity: {sorted(tokenizers)}")
    full_tokens = sum(int(row.get("full_tokens") or 0) for row in rows)
    materialized = sum(int(row.get("materialized_tokens") or 0) for row in rows)
    prompt_tokens = sum(int(row.get("reported_prompt_tokens") or 0) for row in rows)
    completion_tokens = sum(
        int(row.get("reported_completion_tokens") or 0) for row in rows
    )
    retentions = [float(row.get("logical_retention_fraction") or 0) for row in rows]
    violations = []
    for row in rows:
        generation = row.get("generation")
        generation = generation if isinstance(generation, Mapping) else {}
        limit = generation.get("max_completion_tokens")
        observed = row.get("reported_completion_tokens")
        if isinstance(limit, int) and not isinstance(limit, bool) and observed is not None:
            if int(observed) > limit:
                violations.append(int(row["request_index"]))
    return ({
        "official_resolved": resolved,
        "actions": int(summary.get("action_count") or 0),
        "calls": len(rows),
        "full_tokens": full_tokens,
        "materialized_tokens": materialized,
        "provider_prompt_tokens": prompt_tokens,
        "provider_completion_tokens": completion_tokens,
        "maximum_reported_completion_tokens": max(
            (int(row.get("reported_completion_tokens") or 0) for row in rows),
            default=0,
        ),
        "completion_limit_violation_request_indices": violations,
        "minimum_logical_retention": min(retentions, default=0.0),
        "last_request_logical_retention": retentions[-1] if retentions else 0.0,
        "underfilled_requests": sum(
            int(int(row.get("materialized_tokens") or 0)
                < int(row.get("requested_budget_tokens") or 0))
            for row in rows
        ),
        "unused_budget_tokens": sum(
            int(row.get("materialized_budget_unused_tokens") or 0) for row in rows
        ),
        "patch_bytes": int(manifest.get("patch_bytes") or 0),
        "patch_sha256": manifest.get("patch_sha256"),
        "tokenizer": next(iter(tokenizers)),
        "instance_id": instance_id,
        "model": manifest.get("served_model"),
        "model_revision": manifest.get("model_revision"),
        "tokenizer_revision": manifest.get("tokenizer_revision"),
        "reference_trajectory_sha256": manifest.get("reference_trajectory_sha256"),
        "source_image": manifest.get("source_image"),
    }, rows)


def reduce_pair(
    full_path: Path,
    candidate_path: Path,
    *,
    policy: str,
    budget_fraction: float,
    protected_head_turns: int,
    protected_tail_turns: int,
) -> dict[str, Any]:
    full, _ = _arm(full_path)
    candidate, _ = _arm(candidate_path)
    identity_keys = (
        "instance_id", "model", "model_revision", "tokenizer_revision",
        "reference_trajectory_sha256", "source_image",
    )
    mismatches = {
        key: {"full": full.get(key), "candidate": candidate.get(key)}
        for key in identity_keys if full.get(key) != candidate.get(key)
    }
    candidate_full = int(candidate["full_tokens"])
    candidate_materialized = int(candidate["materialized_tokens"])
    full_materialized = int(full["materialized_tokens"])
    candidate["own_logical_saving"] = (
        1 - candidate_materialized / candidate_full if candidate_full else 0.0
    )
    qualification = (
        "pass" if not mismatches
        and bool(full["official_resolved"])
        and bool(candidate["official_resolved"])
        and not full["completion_limit_violation_request_indices"]
        and not candidate["completion_limit_violation_request_indices"]
        else "fail"
    )
    return {
        "schema_version": 3,
        "study": "paper8_5_cross_agent_exact_tokenizer_transfer",
        "agent": "openhands-sdk",
        "instance_id": full["instance_id"],
        "policy": policy,
        "nominal_budget_fraction": budget_fraction,
        "protected_head_turns": protected_head_turns,
        "protected_tail_turns": protected_tail_turns,
        "identity_mismatches": mismatches,
        "full": full,
        "candidate": candidate,
        "paired_input_saving": (
            1 - candidate_materialized / full_materialized
            if full_materialized else 0.0
        ),
        "paired_provider_prompt_saving": (
            1 - int(candidate["provider_prompt_tokens"])
            / int(full["provider_prompt_tokens"])
            if int(full["provider_prompt_tokens"]) else 0.0
        ),
        "action_delta": int(candidate["actions"]) - int(full["actions"]),
        "qualification": qualification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", default="matched_token_tail")
    parser.add_argument("--budget-fraction", type=float, default=0.9)
    parser.add_argument("--protected-head-turns", type=int, default=2)
    parser.add_argument("--protected-tail-turns", type=int, default=4)
    args = parser.parse_args()
    result = reduce_pair(
        args.full.resolve(), args.candidate.resolve(), policy=args.policy,
        budget_fraction=args.budget_fraction,
        protected_head_turns=args.protected_head_turns,
        protected_tail_turns=args.protected_tail_turns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
