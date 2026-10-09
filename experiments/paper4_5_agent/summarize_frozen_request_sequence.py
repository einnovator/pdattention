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


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _frozen_ledger(
    selection_fixture: Path | None,
    request_replay: Path | None,
    expected_requests: tuple[int, ...],
) -> tuple[dict[int, dict[str, str]], dict[str, str] | None]:
    if (selection_fixture is None) != (request_replay is None):
        raise ValueError(
            "selection fixture and request replay must be supplied together"
        )
    if selection_fixture is None or request_replay is None:
        return {}, None
    fixture_rows = _jsonl(selection_fixture)
    replay_rows = _jsonl(request_replay)
    if len(fixture_rows) != len(replay_rows):
        raise ValueError("Frozen fixture and replay lengths differ.")
    complete_ledger: dict[int, dict[str, str]] = {}
    for expected, (fixture, replay) in enumerate(
        zip(fixture_rows, replay_rows), 1
    ):
        fixture_index = int(fixture.get("request_index", -1))
        replay_index = int(replay.get("request_index", -1))
        if fixture_index != expected or replay_index != expected:
            raise ValueError("Frozen ledger request identities are not contiguous.")
        fixture_request = str(fixture.get("request_input_sha256", ""))
        replay_request = str(replay.get("request_input_sha256", ""))
        if not fixture_request or fixture_request != replay_request:
            raise ValueError("Frozen fixture and replay request identities differ.")
        required = {
            "request_input_sha256": fixture_request,
            "source_plan_digest": str(fixture.get("source_plan_digest", "")),
            "selected_resource_digest": str(
                fixture.get("selected_resource_digest", "")
            ),
            "source_wire_plan_digest": str(
                fixture.get("source_wire_plan_digest", "")
            ),
        }
        if any(not value for value in required.values()):
            raise ValueError("Frozen selection row lacks a required identity digest.")
        complete_ledger[expected] = required
    if any(index not in complete_ledger for index in expected_requests):
        raise ValueError("Frozen ledger lacks an expected request identity.")
    ledger = {index: complete_ledger[index] for index in expected_requests}
    return ledger, {
        "selection_fixture": str(selection_fixture),
        "selection_fixture_sha256": _sha256(selection_fixture),
        "request_replay": str(request_replay),
        "request_replay_sha256": _sha256(request_replay),
    }


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


def _consumer_temporary_bytes(payload: dict[str, Any]) -> int | None:
    if payload.get("transient_attention_bytes") is not None:
        return int(payload["transient_attention_bytes"])
    allocation = payload.get("disjoint_attention_allocation")
    if isinstance(allocation, dict) and allocation.get("peak_delta_bytes") is not None:
        return int(allocation["peak_delta_bytes"])
    return None


def summarize(
    paths: Iterable[Path],
    expected_requests: tuple[int, ...],
    expect_full_retention: bool,
    *,
    selection_fixture: Path | None = None,
    request_replay: Path | None = None,
) -> dict[str, Any]:
    files = tuple(paths)
    if not files:
        raise ValueError("At least one request artifact is required.")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    engines = {str(payload["engine"]) for payload in payloads}
    models = {str(payload["model"]) for payload in payloads}
    if len(engines) != 1 or len(models) != 1:
        raise ValueError("Artifacts do not describe one frozen engine/model cell.")

    ledger, ledger_files = _frozen_ledger(
        selection_fixture, request_replay, expected_requests
    )
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
            "selection_pack_bytes": int(
                payload.get(
                    "selection_pack_bytes",
                    payload.get("interval_pack_bytes", 0),
                )
            ),
            "attachment_physical_kv_copy": _attachment_copy(payload),
            "consumer_peak_temporary_bytes": _consumer_temporary_bytes(payload),
            "offloaded_payload_bytes": (
                int(payload["offloaded_payload_bytes"])
                if payload.get("offloaded_payload_bytes") is not None
                else None
            ),
            "engine_lifecycle_qualified": bool(
                payload["engine_lifecycle_qualified"]
            ),
            "qualification_blockers": list(
                payload.get("qualification_blockers") or ()
            ),
            "artifact": str(path),
            "artifact_sha256": _sha256(path),
        }
        frozen_identity = ledger.get(row["request_index"])
        if frozen_identity is not None:
            if row["request_input_sha256"] != frozen_identity["request_input_sha256"]:
                raise ValueError(
                    f"Artifact request {row['request_index']} does not match frozen ledger."
                )
            row.update(frozen_identity)
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
    source_kv_tokens = sum(row["source_tokens"] for row in rows)
    selected_kv_tokens = sum(row["selected_kv_tokens"] for row in rows)
    wire_suffix_tokens = sum(row["wire_suffix_tokens"] for row in rows)
    consumer_temporaries = [
        row["consumer_peak_temporary_bytes"]
        for row in rows
        if row["consumer_peak_temporary_bytes"] is not None
    ]
    offload_bytes = [
        row["offloaded_payload_bytes"]
        for row in rows
        if row["offloaded_payload_bytes"] is not None
    ]
    return {
        "schema_version": "paper4.5.frozen-request-sequence.v1",
        "claim_boundary": "request_sequence_not_autonomous_task",
        "engine": next(iter(engines)),
        "model": next(iter(models)),
        "source_policy": next(iter(source_policies)),
        "frozen_ledger_bound": bool(ledger),
        "frozen_ledger_files": ledger_files,
        "expect_full_retention": expect_full_retention,
        "expected_requests": list(expected_requests),
        "completed_requests": len(rows),
        "qualified_requests": sum(int(value) for value in request_qualified),
        "sequence_qualified": all(request_qualified),
        "weighted_realized_retention_fraction": selected_visible
        / max(total_visible, 1),
        "visible_context_omission_fraction": 1.0
        - selected_visible / max(total_visible, 1),
        "cumulative_source_kv_tokens": source_kv_tokens,
        "cumulative_selected_kv_tokens": selected_kv_tokens,
        "cumulative_wire_suffix_tokens": wire_suffix_tokens,
        "weighted_historical_kv_retention_fraction": selected_kv_tokens
        / max(source_kv_tokens, 1),
        "historical_kv_omission_fraction": 1.0
        - selected_kv_tokens / max(source_kv_tokens, 1),
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
        "consumer_peak_temporary_bytes_max": (
            max(consumer_temporaries) if consumer_temporaries else None
        ),
        "offloaded_payload_bytes_cumulative": (
            sum(offload_bytes) if len(offload_bytes) == len(rows) else None
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
    parser.add_argument("--selection-fixture", type=Path)
    parser.add_argument("--request-replay", type=Path)
    args = parser.parse_args()
    expected = tuple(int(value) for value in args.expected_requests.split(","))
    result = summarize(
        args.inputs,
        expected,
        args.expect_full_retention,
        selection_fixture=args.selection_fixture,
        request_replay=args.request_replay,
    )
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
