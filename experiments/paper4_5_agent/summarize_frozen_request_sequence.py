"""Reduce one hash-bound frozen request sequence across a native engine.

This reducer intentionally stops at request-sequence qualification.  It does
not reinterpret a set of continuation checks as an autonomous agent solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retention(payload: dict[str, Any]) -> float:
    value = payload.get(
        "realized_total_retention_fraction",
        payload.get("realized_retention_fraction"),
    )
    if value is None:
        raise ValueError("Artifact does not report realized retention.")
    return float(value)


def _attachment_copy(payload: dict[str, Any]) -> bool:
    if "initial_attachment_physical_kv_copy" in payload:
        return bool(payload["initial_attachment_physical_kv_copy"])
    return bool(payload.get("physical_kv_copy", True))


def _oracle_exact(payload: dict[str, Any]) -> bool | None:
    checks = dict(payload.get("checks") or {})
    names = (
        "ordinary_full_engine_token_exact",
        "ordinary_full_engine_oracle_token_exact",
        "dense_engine_oracle_token_exact",
    )
    values = [bool(checks[name]) for name in names if name in checks]
    return all(values) if values else None


def summarize(
    paths: Iterable[Path],
    expected_requests: tuple[int, ...],
    expect_full_retention: bool,
) -> dict[str, Any]:
    files = tuple(paths)
    if not files:
        raise ValueError("At least one request artifact is required.")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    engines = {str(payload["engine"]) for payload in payloads}
    models = {str(payload["model"]) for payload in payloads}
    if len(engines) != 1 or len(models) != 1:
        raise ValueError("Artifacts do not describe one frozen engine/model cell.")

    rows: list[dict[str, Any]] = []
    for path, payload in zip(files, payloads):
        checks = dict(payload.get("checks") or {})
        row = {
            "request_index": int(payload["request_index"]),
            "request_input_sha256": str(payload["request_input_sha256"]),
            "source_policy": str(payload["source_policy"]),
            "source_tokens": int(payload["source_tokens"]),
            "wire_suffix_tokens": int(payload["wire_suffix_tokens"]),
            "selected_kv_tokens": int(payload["selected_kv_tokens"]),
            "realized_retention_fraction": _retention(payload),
            "page_rounding_added_tokens": int(
                payload.get("page_rounding_added_tokens", 0)
            ),
            "same_subset_token_exact": bool(
                checks.get("same_subset_token_exact", False)
            ),
            "ordinary_full_oracle_token_exact": _oracle_exact(payload),
            "selected_history_reencoded_tokens": int(
                payload["selected_text_reencoded_tokens"]
            ),
            "selection_pack_bytes": int(payload.get("selection_pack_bytes", 0)),
            "attachment_physical_kv_copy": _attachment_copy(payload),
            "engine_lifecycle_qualified": bool(
                payload["engine_lifecycle_qualified"]
            ),
            "qualification_blockers": list(
                payload.get("qualification_blockers") or ()
            ),
            "artifact": str(path),
            "artifact_sha256": _sha256(path),
        }
        rows.append(row)

    rows.sort(key=lambda row: row["request_index"])
    observed = tuple(row["request_index"] for row in rows)
    if observed != expected_requests:
        raise ValueError(
            f"Expected request identities {expected_requests}, observed {observed}."
        )
    digests = [row["request_input_sha256"] for row in rows]
    if len(set(digests)) != len(digests):
        raise ValueError("Frozen request digests are not unique.")

    source_policies = {row["source_policy"] for row in rows}
    if len(source_policies) != 1:
        raise ValueError("Artifacts mix source-policy identities.")

    request_qualified = [
        row["engine_lifecycle_qualified"]
        and not row["qualification_blockers"]
        and row["same_subset_token_exact"]
        and row["selected_history_reencoded_tokens"] == 0
        and row["selection_pack_bytes"] == 0
        and not row["attachment_physical_kv_copy"]
        and (
            not expect_full_retention
            or (
                abs(row["realized_retention_fraction"] - 1.0) <= 1e-12
                and row["ordinary_full_oracle_token_exact"] is True
            )
        )
        for row in rows
    ]
    total_visible = sum(
        row["source_tokens"] + row["wire_suffix_tokens"] for row in rows
    )
    selected_visible = sum(
        row["selected_kv_tokens"] + row["wire_suffix_tokens"] for row in rows
    )
    return {
        "schema_version": "paper4.5.frozen-request-sequence.v1",
        "claim_boundary": "request_sequence_not_autonomous_task",
        "engine": next(iter(engines)),
        "model": next(iter(models)),
        "source_policy": next(iter(source_policies)),
        "expect_full_retention": expect_full_retention,
        "expected_requests": list(expected_requests),
        "completed_requests": len(rows),
        "qualified_requests": sum(int(value) for value in request_qualified),
        "sequence_qualified": all(request_qualified),
        "weighted_realized_retention_fraction": selected_visible
        / max(total_visible, 1),
        "realized_retention_fraction": {
            "minimum": min(row["realized_retention_fraction"] for row in rows),
            "maximum": max(row["realized_retention_fraction"] for row in rows),
        },
        "selected_history_reencoded_tokens": sum(
            row["selected_history_reencoded_tokens"] for row in rows
        ),
        "selection_pack_bytes": sum(row["selection_pack_bytes"] for row in rows),
        "attachment_physical_kv_copy_requests": sum(
            int(row["attachment_physical_kv_copy"]) for row in rows
        ),
        "page_rounding_added_tokens": sum(
            row["page_rounding_added_tokens"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--expected-requests", required=True)
    parser.add_argument(
        "--expect-full-retention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected = tuple(int(value) for value in args.expected_requests.split(","))
    result = summarize(args.inputs, expected, args.expect_full_retention)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "engine": result["engine"],
                "completed_requests": result["completed_requests"],
                "qualified_requests": result["qualified_requests"],
                "sequence_qualified": result["sequence_qualified"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if result["sequence_qualified"] else 1)


if __name__ == "__main__":
    main()
