"""Replay a captured request after changing only its immutable workflow anchor.

This is a frozen mechanism test, not an autonomous task score.  It keeps the
selected messages and generation coordinates byte-for-byte stable except for
one declared system-prefix anchor substitution.  Candidates must pass both a
first-decision request and a pre-mutation request before autonomous admission.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.request import Request, urlopen

from .autonomous_proxy import STATIC_WORKFLOW_ANCHORS
from .miniswe_semantics import bash_action_contract
from .recordizer import extract_miniswe_command


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def replace_static_anchor(
    payload: Mapping[str, Any],
    *,
    source_anchor: str,
    candidate_anchor: str,
) -> dict[str, Any]:
    """Return a copy with exactly one declared system-anchor substitution."""

    if source_anchor == candidate_anchor:
        raise ValueError("source and candidate anchors must differ")
    try:
        source = STATIC_WORKFLOW_ANCHORS[source_anchor]
        candidate = STATIC_WORKFLOW_ANCHORS[candidate_anchor]
    except KeyError as error:
        raise ValueError(f"unknown static workflow anchor: {error.args[0]}") from error
    if not source or not candidate:
        raise ValueError("counterfactual anchors must both be non-empty")

    transformed = copy.deepcopy(dict(payload))
    messages = transformed.get("messages")
    if not isinstance(messages, list):
        raise ValueError("captured request must contain a message list")
    replacements = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        occurrences = content.count(source)
        if occurrences:
            if occurrences != 1:
                raise ValueError("source anchor occurs more than once in one message")
            message["content"] = content.replace(source, candidate)
            replacements += 1
    if replacements != 1:
        raise ValueError(
            f"expected exactly one source-anchor occurrence, found {replacements}"
        )
    return transformed


def _post_json(url: str, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def replay_counterfactual(
    payload: Mapping[str, Any],
    *,
    endpoint: str,
    source_anchor: str,
    candidate_anchor: str,
    repeats: int,
    timeout: float,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    transformed = replace_static_anchor(
        payload,
        source_anchor=source_anchor,
        candidate_anchor=candidate_anchor,
    )
    rows = []
    for repeat in range(1, repeats + 1):
        response = _post_json(
            endpoint.rstrip("/") + "/chat/completions", transformed, timeout
        )
        content = response["choices"][0]["message"]["content"]
        command = extract_miniswe_command(content)
        rows.append({
            "repeat": repeat,
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "valid_single_action": command is not None,
            "command": command,
            "action_contract": bash_action_contract(command) if command else None,
            "finish_reason": response["choices"][0].get("finish_reason"),
            "usage": response.get("usage"),
            "content": content,
        })
    valid = all(row["valid_single_action"] for row in rows)
    commands = {row["command"] for row in rows}
    return {
        "schema_version": 1,
        "study": "paper8_5_static_anchor_frozen_counterfactual",
        "source_request_sha256": _digest(payload),
        "counterfactual_request_sha256": _digest(transformed),
        "source_anchor": source_anchor,
        "candidate_anchor": candidate_anchor,
        "generation": {
            key: transformed.get(key)
            for key in ("model", "temperature", "top_p", "top_k", "seed", "max_tokens")
        },
        "repeat_count": repeats,
        "all_valid_single_action": valid,
        "exact_command_repeatability": valid and len(commands) == 1,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--source-anchor", choices=tuple(STATIC_WORKFLOW_ANCHORS), required=True)
    parser.add_argument("--candidate-anchor", choices=tuple(STATIC_WORKFLOW_ANCHORS), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.request.read_text(encoding="utf-8"))
    receipt = replay_counterfactual(
        payload,
        endpoint=args.endpoint,
        source_anchor=args.source_anchor,
        candidate_anchor=args.candidate_anchor,
        repeats=args.repeats,
        timeout=args.timeout,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "all_valid_single_action": receipt["all_valid_single_action"],
        "exact_command_repeatability": receipt["exact_command_repeatability"],
        "commands": [row["command"] for row in receipt["rows"]],
    }, indent=2))


if __name__ == "__main__":
    main()
