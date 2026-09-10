"""Audit direct-versus-gateway agent exchanges at every model request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ENGINE_FIELDS = (
    "prefix_cache_hit",
    "prefix_cached_tokens",
    "engine_cached_tokens_total",
    "native_tokens",
    "wire_tokens",
    "physical_kv_copy",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_request(rows: list[Mapping[str, Any]]) -> dict[int, dict[str, Mapping[str, Any]]]:
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        index = row.get("request_index")
        event = str(row.get("event", ""))
        if not isinstance(index, int) or event not in {"request", "response"}:
            continue
        grouped.setdefault(index, {})[event] = row
    return grouped


def _resources(request: Mapping[str, Any]) -> list[tuple[str, str]]:
    physical = request.get("physical_payload") or {}
    pra = physical.get("pra") or {}
    return [
        (str(row.get("resource_id")), str(row.get("text", "")))
        for row in pra.get("resources", ())
    ]


def _response_text(response: Mapping[str, Any]) -> str:
    payload = response.get("payload") or {}
    choices = payload.get("choices") or [{}]
    return str((choices[0].get("message") or {}).get("content", ""))


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = _by_request(_rows(reference_path))
    candidate = _by_request(_rows(candidate_path))
    indices = sorted(set(reference) | set(candidate))
    comparisons = []
    for index in indices:
        left = reference.get(index, {})
        right = candidate.get(index, {})
        left_request = left.get("request")
        right_request = right.get("request")
        left_response = left.get("response")
        right_response = right.get("response")
        request_complete = left_request is not None and right_request is not None
        response_complete = left_response is not None and right_response is not None
        logical_equal = bool(request_complete and (
            left_request.get("request_input_sha256")
            == right_request.get("request_input_sha256")
        ))
        resources_equal = bool(request_complete and (
            _resources(left_request) == _resources(right_request)
        ))
        response_equal = bool(response_complete and (
            _response_text(left_response) == _response_text(right_response)
        ))
        left_pra = ((left_response or {}).get("payload") or {}).get("pra") or {}
        right_pra = ((right_response or {}).get("payload") or {}).get("pra") or {}
        engine_equal = bool(response_complete and all(
            left_pra.get(field) == right_pra.get(field) for field in ENGINE_FIELDS
        ))
        comparisons.append({
            "request_index": index,
            "request_complete": request_complete,
            "response_complete": response_complete,
            "logical_request_equal": logical_equal,
            "selected_resources_equal": resources_equal,
            "response_text_equal": response_equal,
            "engine_execution_equal": engine_equal,
            "reference_response_sha256": (
                _digest(_response_text(left_response)) if left_response else None
            ),
            "candidate_response_sha256": (
                _digest(_response_text(right_response)) if right_response else None
            ),
            "reference_engine": {
                field: left_pra.get(field) for field in ENGINE_FIELDS
            },
            "candidate_engine": {
                field: right_pra.get(field) for field in ENGINE_FIELDS
            },
        })
    complete = [row for row in comparisons if row["response_complete"]]
    exact = [
        row for row in complete
        if row["logical_request_equal"]
        and row["selected_resources_equal"]
        and row["response_text_equal"]
        and row["engine_execution_equal"]
    ]
    first_difference = next(
        (row["request_index"] for row in complete if row not in exact), None,
    )
    return {
        "schema_version": 1,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "reference_requests": len(reference),
        "candidate_requests": len(candidate),
        "complete_paired_responses": len(complete),
        "exact_paired_responses": len(exact),
        "first_difference": first_difference,
        "transport_equivalent": (
            len(reference) == len(candidate) == len(complete) == len(exact)
        ),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in (
            "complete_paired_responses", "exact_paired_responses",
            "first_difference", "transport_equivalent",
        )
    }))


if __name__ == "__main__":
    main()
