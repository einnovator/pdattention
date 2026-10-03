"""Phase-level diagnostics for mini-swe-agent action trajectories.

The accuracy--saving reducer intentionally treats tool-call count as an
outcome.  This module explains *where* a call delta came from without calling
every different shell spelling a behavioral failure.  It is evaluation code;
selection policies do not consume these labels.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .miniswe_semantics import classify_bash_operation, extract_resource_ids
from pra_hf.tool_semantics import OperationKind


_COMMAND = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
_RETURNCODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>")
_SUBMISSION = re.compile(r"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|patch\.txt", re.I)


def _command(content: str) -> str | None:
    matches = _COMMAND.findall(content)
    return matches[0].strip() if len(matches) == 1 else None


def _returncode(content: str) -> int | None:
    match = _RETURNCODE.search(content)
    return int(match.group(1)) if match else None


def extract_miniswe_steps(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return complete action--observation steps from one episode."""

    steps: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if str(message.get("role")) != "assistant":
            continue
        command = _command(str(message.get("content") or ""))
        if command is None:
            continue
        observation = messages[index + 1] if index + 1 < len(messages) else {}
        observation_content = (
            str(observation.get("content") or "")
            if str(observation.get("role")) in {"user", "tool"} else ""
        )
        operation = classify_bash_operation(command)
        resources = extract_resource_ids(command, observation_content)
        steps.append({
            "call": len(steps) + 1,
            "command": command,
            "operation": operation.value,
            "resources": list(resources),
            "returncode": _returncode(observation_content),
            "failed": (_returncode(observation_content) not in {None, 0}),
            "submission": bool(_SUBMISSION.search(command)),
        })
    return steps


def _first_call(steps: Iterable[Mapping[str, Any]], *operations: str) -> int | None:
    wanted = set(operations)
    for step in steps:
        if str(step.get("operation")) in wanted:
            return int(step["call"])
    return None


def summarize_miniswe_steps(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize workflow phase and avoidable-call proxies."""

    operations = Counter(str(step.get("operation")) for step in steps)
    signatures: Counter[tuple[str, tuple[str, ...]]] = Counter()
    for step in steps:
        signatures[
            (
                str(step.get("operation")),
                tuple(sorted(str(row) for row in step.get("resources") or ())),
            )
        ] += 1
    repeated = sum(max(0, count - 1) for count in signatures.values())
    first_mutation = _first_call(steps, OperationKind.WRITE.value)
    first_verification = _first_call(steps, OperationKind.VERIFY.value)
    first_submission = next(
        (int(step["call"]) for step in steps if step.get("submission")), None
    )
    post_mutation = (
        len(steps) - first_mutation if first_mutation is not None else None
    )
    return {
        "calls": len(steps),
        "operation_counts": dict(operations),
        "failed_tool_calls": sum(bool(step.get("failed")) for step in steps),
        "repeated_operation_resource_calls": repeated,
        "calls_to_first_mutation": first_mutation,
        "calls_to_first_verification": first_verification,
        "calls_to_first_submission": first_submission,
        "calls_after_first_mutation": post_mutation,
        "submission_calls": sum(bool(step.get("submission")) for step in steps),
    }


def compare_miniswe_trajectories(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare exact and coarse-semantic divergence plus phase cost."""

    exact_prefix = 0
    semantic_prefix = 0
    for left, right in zip(reference, candidate):
        if left.get("command") == right.get("command"):
            exact_prefix += 1
        else:
            break
    for left, right in zip(reference, candidate):
        if (
            left.get("operation") == right.get("operation")
            and set(left.get("resources") or ()) == set(right.get("resources") or ())
        ):
            semantic_prefix += 1
        else:
            break
    ref_summary = summarize_miniswe_steps(reference)
    candidate_summary = summarize_miniswe_steps(candidate)
    return {
        "reference": ref_summary,
        "candidate": candidate_summary,
        "call_delta": candidate_summary["calls"] - ref_summary["calls"],
        "exact_common_prefix_calls": exact_prefix,
        "semantic_common_prefix_calls": semantic_prefix,
        "first_exact_divergence_call": (
            exact_prefix + 1 if exact_prefix < min(len(reference), len(candidate)) else None
        ),
        "first_semantic_divergence_call": (
            semantic_prefix + 1
            if semantic_prefix < min(len(reference), len(candidate)) else None
        ),
        "operation_count_delta": {
            key: candidate_summary["operation_counts"].get(key, 0)
            - ref_summary["operation_counts"].get(key, 0)
            for key in sorted(
                set(ref_summary["operation_counts"])
                | set(candidate_summary["operation_counts"])
            )
        },
    }


__all__ = [
    "compare_miniswe_trajectories",
    "extract_miniswe_steps",
    "summarize_miniswe_steps",
]


def _load_steps(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = payload.get("trajectory", payload)
    messages = trajectory.get("messages") if isinstance(trajectory, Mapping) else None
    if not isinstance(messages, list):
        raise ValueError(f"{path} has no trajectory messages")
    return extract_miniswe_steps(messages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = _load_steps(args.reference)
    result = {
        "schema_version": 1,
        "study": "paper8_5_miniswe_trajectory_divergence",
        "reference": str(args.reference),
        "reference_summary": summarize_miniswe_steps(reference),
        "comparisons": [
            {
                "candidate": str(path),
                **compare_miniswe_trajectories(reference, _load_steps(path)),
            }
            for path in args.candidate
        ],
        "guardrail": (
            "Phase labels explain observed call deltas; they do not establish "
            "that a selection policy caused the divergence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
