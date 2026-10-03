"""Reduce frozen-request cold/warm decoding controls.

The reducer is intentionally separate from policy comparisons.  A request is
admitted only when every source contains the same selected-message digest.
It reports response/command clusters and whether a chosen token differs from
the highest *reported* logprob token; the latter is descriptive because some
backends expose pre-sampler rather than sampler probabilities.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = payload.get("rows")
    if not isinstance(value, list):
        raise ValueError("decode-state input has no rows")
    return [row for row in value if isinstance(row, Mapping)]


def _reported_top1_mismatches(row: Mapping[str, Any]) -> int:
    logprobs = row.get("choice_logprobs")
    content = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    if not isinstance(content, list):
        return 0
    mismatches = 0
    for item in content:
        if not isinstance(item, Mapping):
            continue
        candidates = [
            candidate for candidate in item.get("top_logprobs") or ()
            if isinstance(candidate, Mapping)
            and isinstance(candidate.get("logprob"), (int, float))
        ]
        if not candidates:
            continue
        top = max(candidates, key=lambda candidate: float(candidate["logprob"]))
        mismatches += str(item.get("token")) != str(top.get("token"))
    return mismatches


def reduce_decode_state(
    cohorts: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    digests = {
        str(row.get("selected_messages_sha256"))
        for _, payload in cohorts for row in _rows(payload)
        if row.get("selected_messages_sha256")
    }
    exact_request_identity = len(digests) == 1
    rows_by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for label, payload in cohorts:
        rows_by_label[label].extend(_rows(payload))
    summaries: dict[str, Any] = {}
    for label, rows in rows_by_label.items():
        clean = [row for row in rows if not row.get("transport_error")]
        summaries[label] = {
            "attempts": len(rows),
            "clean_attempts": len(clean),
            "transport_errors": dict(Counter(
                str(row.get("transport_error")) for row in rows
                if row.get("transport_error")
            )),
            "response_clusters": dict(Counter(
                str(row.get("response_content_sha256")) for row in clean
                if row.get("response_content_sha256")
            )),
            "command_clusters": dict(Counter(
                str(row.get("command")) for row in clean if row.get("command")
            )),
            "reported_top1_mismatch_tokens": sum(
                _reported_top1_mismatches(row) for row in clean
            ),
            "elapsed_seconds": [row.get("elapsed_seconds") for row in clean],
        }
    return {
        "schema_version": 1,
        "study": "paper8_5_frozen_decode_state_control",
        "selected_messages_sha256": next(iter(digests), None),
        "exact_request_identity": exact_request_identity,
        "causal_admission": exact_request_identity and all(
            summary["clean_attempts"] == summary["attempts"]
            for summary in summaries.values()
        ),
        "cohorts": summaries,
        "interpretation_guardrail": (
            "Reported logprobs may describe pre-sampler logits. A chosen token "
            "that is not reported top-1 is not, by itself, proof of request "
            "temperature handling. Cold/warm clustering is a backend-state "
            "control and is never attributed to history selection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohorts = [
        (label, json.loads(Path(path).read_text(encoding="utf-8")))
        for label, path in args.cohort
    ]
    result = reduce_decode_state(cohorts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
