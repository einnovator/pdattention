"""Build bounded pair/beam queues from completed oracle omission trials."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .miniswe_semantics import classify_bash_operation, extract_resource_ids


_SYNTAX_VERIFY = re.compile(r"(?:ast\.parse|py_compile|compile\s*\()")
_RESOURCE_SUFFIX = re.compile(
    r"\.(?:py|pyi|js|jsx|ts|tsx|java|c|cc|cpp|h|hpp|rs|go|rb|php|"
    r"json|ya?ml|toml|ini|cfg|txt|md|rst|html|css|xml|sql|patch|diff)$",
    re.IGNORECASE,
)


def _action_signature(command: Any) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(command, str) or not command.strip():
        return None
    kind = (
        "finalization"
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command
        else "syntax_verify" if _SYNTAX_VERIFY.search(command)
        else classify_bash_operation(command).value
    )
    resources = tuple(sorted(
        resource for resource in extract_resource_ids(command, "")
        if "/" in resource or _RESOURCE_SUFFIX.search(resource)
    ))
    return kind, resources


def _trial(artifact: Mapping[str, Any], source: str) -> dict[str, Any]:
    if artifact.get("study") != "paper8_5_agent_memory_frozen_replay":
        raise ValueError(f"{source}: not a Paper 8.5 frozen replay")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"{source}: oracle subset trials require exactly one decision")
    row = rows[0]
    omission = row.get("oracle_omission") or {}
    groups = tuple(sorted(str(value) for value in omission.get(
        "omitted_causal_group_ids", ()
    )))
    if not groups:
        raise ValueError(f"{source}: no realized oracle omission")
    if groups != tuple(sorted(str(value) for value in (
        artifact.get("run_configuration", {}).get(
            "oracle_omit_causal_group_ids", ()
        )
    ))):
        raise ValueError(f"{source}: omission result/configuration mismatch")
    return {
        "source": source,
        "instance_id": str(artifact.get("instance_id")),
        "decision": int(row["decision"]),
        "reference_replay_digest": artifact.get("reference_replay_digest"),
        "groups": groups,
        "depth": len(groups),
        "omitted_tokens": int(omission.get("omitted_tokens") or 0),
        "full_history_tokens": int(row.get("full_history_tokens") or 0),
        "action_valid": bool(row.get("action_valid")),
        "exact_command": bool(row.get("exact_command")),
        "exact_content": bool(row.get("exact_content")),
        "semantic_transition_equivalent": (
            _action_signature(row.get("generated_command")) is not None
            and _action_signature(row.get("generated_command"))
            == _action_signature(row.get("reference_command"))
        ),
    }


def build_queue(
    artifacts: Iterable[tuple[Mapping[str, Any], str]],
    *,
    target_depth: int,
    beam_width: int,
    qualification: str = "exact_command",
) -> dict[str, Any]:
    if target_depth < 2:
        raise ValueError("target_depth must be at least two")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if qualification not in {"exact_command", "semantic_transition"}:
        raise ValueError("qualification must be exact_command or semantic_transition")
    trials = [_trial(artifact, source) for artifact, source in artifacts]
    if not trials:
        raise ValueError("no oracle trials supplied")
    identities = {
        (row["instance_id"], row["decision"], row["reference_replay_digest"])
        for row in trials
    }
    if len(identities) != 1:
        raise ValueError("oracle trials do not share task, decision, and FULL reference")
    singleton_tokens = {
        row["groups"][0]: row["omitted_tokens"]
        for row in trials if row["depth"] == 1
    }
    quality_key = (
        "exact_command"
        if qualification == "exact_command" else "semantic_transition_equivalent"
    )
    safe_singletons = {
        row["groups"][0] for row in trials
        if row["depth"] == 1 and row["action_valid"] and row[quality_key]
    }
    if len(safe_singletons) < target_depth:
        candidates: set[tuple[str, ...]] = set()
    elif target_depth == 2:
        candidates = set(combinations(sorted(safe_singletons), 2))
    else:
        parents = [
            row for row in trials
            if row["depth"] == target_depth - 1
            and row["action_valid"] and row[quality_key]
        ]
        candidates = {
            tuple(sorted((*parent["groups"], group)))
            for parent in parents
            for group in safe_singletons
            if group not in parent["groups"]
        }
    observed = {row["groups"] for row in trials}
    candidates.difference_update(observed)
    ranked = sorted(
        candidates,
        key=lambda groups: (
            -sum(singleton_tokens.get(group, 0) for group in groups), groups
        ),
    )[:beam_width]
    instance_id, decision, reference_digest = next(iter(identities))
    queue = []
    for index, groups in enumerate(ranked, 1):
        arguments = ["--min-decision", str(decision), "--max-decisions", "1"]
        for group in groups:
            arguments.extend(("--oracle-omit-group", group))
        queue.append({
            "trial_id": f"depth{target_depth}-{index:03d}",
            "omitted_causal_group_ids": groups,
            "estimated_omitted_tokens_from_singletons": sum(
                singleton_tokens.get(group, 0) for group in groups
            ),
            "cli_arguments": arguments,
        })
    return {
        "schema_version": 1,
        "study": "paper8_5_oracle_headroom_subset_queue",
        "evidence_class": "offline_oracle_not_deployable_policy_quality",
        "instance_id": instance_id,
        "decision": decision,
        "reference_replay_digest": reference_digest,
        "target_depth": target_depth,
        "beam_width": beam_width,
        "qualification": qualification,
        "completed_trials": len(trials),
        "safe_singletons": sorted(safe_singletons),
        "safe_singleton_count": len(safe_singletons),
        "trial_count": len(queue),
        "ranking": "largest_additive_singleton_token_saving_then_group_identity",
        "trials": queue,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument(
        "--qualification",
        choices=("exact_command", "semantic_transition"),
        default="exact_command",
    )
    args = parser.parse_args()
    paths = sorted(args.input_directory.glob("*.json"))
    artifacts = []
    for path in paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if artifact.get("study") == "paper8_5_agent_memory_frozen_replay" and (
            artifact.get("run_configuration", {}).get(
                "oracle_omit_causal_group_ids"
            )
        ):
            artifacts.append((artifact, path.name))
    result = build_queue(
        artifacts, target_depth=args.target_depth, beam_width=args.beam_width,
        qualification=args.qualification,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "safe_singleton_count": result["safe_singleton_count"],
        "trial_count": result["trial_count"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
