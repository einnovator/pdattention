"""Gate identical frozen logical ledgers across native K/V engine sequences.

This reducer does not compare latency and does not infer autonomous quality.
It proves that independently qualified engines consumed the same hash-bound
Paper 8.5 request/selection ledger, then reports logical, resident-K/V and
visible-context omission as separate quantities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


IDENTITY_FIELDS = (
    "request_input_sha256",
    "source_plan_digest",
    "selected_resource_digest",
    "source_wire_plan_digest",
)


def _engine_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or ())
    if not rows:
        raise ValueError("Engine sequence contains no request rows.")

    source = payload.get("cumulative_source_kv_tokens")
    selected = payload.get("cumulative_selected_kv_tokens")
    if source is None or selected is None:
        source = sum(
            int(row.get("source_tokens", row.get("logical_source_tokens", 0)))
            for row in rows
        )
        selected = sum(
            int(row.get("selected_kv_tokens", row.get("selected_logical_kv_tokens", 0)))
            for row in rows
        )
    source = int(source)
    selected = int(selected)

    logical_source = int(payload.get("cumulative_logical_source_tokens", source))
    logical_selected = int(
        payload.get("cumulative_selected_logical_kv_tokens", selected)
    )
    resident_source = int(payload.get("cumulative_resident_source_tokens", source))
    resident_selected = int(
        payload.get("cumulative_selected_original_page_tokens", selected)
    )
    visible_omission = payload.get("visible_context_omission_fraction")
    if visible_omission is None:
        full_visible = sum(int(row.get("full_visible_tokens", 0)) for row in rows)
        visible = sum(int(row.get("total_visible_tokens", 0)) for row in rows)
        visible_omission = 1.0 - visible / max(full_visible, 1)

    reencoded = int(payload.get("selected_history_reencoded_tokens", -1))
    if "selected_history_kv_copy_bytes" in payload:
        attachment_copy = int(payload["selected_history_kv_copy_bytes"])
    else:
        attachment_copy = int(payload.get("attachment_physical_kv_copy_requests", -1))
    pack_bytes = int(payload.get("selection_pack_bytes", 0))
    return {
        "engine_reported": str(payload.get("engine")),
        "model": str(payload.get("model")),
        "claim_boundary": str(payload.get("claim_boundary")),
        "requests": len(rows),
        "logical_source_tokens": logical_source,
        "logical_selected_tokens": logical_selected,
        "logical_history_omission_fraction": 1.0
        - logical_selected / max(logical_source, 1),
        "resident_source_tokens": resident_source,
        "resident_selected_tokens": resident_selected,
        "resident_kv_omission_fraction": 1.0
        - resident_selected / max(resident_source, 1),
        "visible_context_omission_fraction": float(visible_omission),
        "selected_history_reencoded_tokens": reencoded,
        "selection_pack_bytes": pack_bytes,
        "attachment_copy_indicator": attachment_copy,
        "sequence_qualified": bool(payload.get("sequence_qualified")),
        "frozen_ledger_bound": bool(payload.get("frozen_ledger_bound")),
    }


def summarize(
    summaries: Mapping[str, Path],
    *,
    required_engines: tuple[str, ...] = (),
    paper8_logical_omission_fraction: float | None = None,
) -> dict[str, Any]:
    missing = sorted(set(required_engines) - set(summaries))
    if missing:
        raise ValueError(f"Missing required engine summaries: {', '.join(missing)}")
    if not summaries:
        raise ValueError("At least one engine summary is required.")

    payloads = {
        engine: json.loads(path.read_text(encoding="utf-8"))
        for engine, path in summaries.items()
    }
    reference_engine = next(iter(payloads))
    reference = payloads[reference_engine]
    reference_files = dict(reference.get("frozen_ledger_files") or {})
    ledger_identity = {
        "selection_fixture_sha256": reference_files.get("selection_fixture_sha256"),
        "request_replay_sha256": reference_files.get("request_replay_sha256"),
    }
    if not all(ledger_identity.values()):
        raise ValueError(f"{reference_engine} does not bind both frozen ledger files.")
    reference_requests = tuple(reference.get("expected_requests") or ())
    reference_rows = list(reference.get("rows") or ())
    reference_identities = tuple(
        tuple(row.get(field) for field in IDENTITY_FIELDS) for row in reference_rows
    )

    engines: dict[str, Any] = {}
    blockers: list[str] = []
    for engine, payload in payloads.items():
        files = dict(payload.get("frozen_ledger_files") or {})
        observed_identity = {
            "selection_fixture_sha256": files.get("selection_fixture_sha256"),
            "request_replay_sha256": files.get("request_replay_sha256"),
        }
        rows = list(payload.get("rows") or ())
        row_identities = tuple(
            tuple(row.get(field) for field in IDENTITY_FIELDS) for row in rows
        )
        checks = {
            "sequence_qualified": payload.get("sequence_qualified") is True,
            "frozen_ledger_bound": payload.get("frozen_ledger_bound") is True,
            "ledger_hashes_identical": observed_identity == ledger_identity,
            "request_indices_identical": tuple(payload.get("expected_requests") or ())
            == reference_requests,
            "request_and_plan_identities_identical": row_identities
            == reference_identities,
        }
        metrics = _engine_metrics(payload)
        checks.update(
            {
                "selected_history_not_reencoded": metrics[
                    "selected_history_reencoded_tokens"
                ]
                == 0,
                "selection_not_packed": metrics["selection_pack_bytes"] == 0,
                "selected_kv_attachment_not_copied": metrics[
                    "attachment_copy_indicator"
                ]
                == 0,
            }
        )
        if paper8_logical_omission_fraction is not None:
            metrics["paper8_logical_omission_fraction"] = (
                paper8_logical_omission_fraction
            )
            metrics["resident_minus_paper8_percentage_points"] = 100.0 * (
                metrics["resident_kv_omission_fraction"]
                - paper8_logical_omission_fraction
            )
            metrics["visible_minus_paper8_percentage_points"] = 100.0 * (
                metrics["visible_context_omission_fraction"]
                - paper8_logical_omission_fraction
            )
        engines[engine] = {
            "artifact": summaries[engine].as_posix(),
            "checks": checks,
            **metrics,
        }
        blockers.extend(
            f"{engine}:{name}" for name, passed in checks.items() if not passed
        )

    return {
        "schema_version": "paper4.5.cross-engine-frozen-sequence-gate.v1",
        "claim_boundary": (
            "identical_frozen_logical_ledger_and_native_kv_realization_"
            "not_autonomous_quality_or_latency"
        ),
        "required_engines": list(required_engines),
        "ledger_identity": ledger_identity,
        "expected_requests": list(reference_requests),
        "engines": engines,
        "qualification_blockers": blockers,
        "cross_engine_sequence_qualified": not blockers,
    }


def _engine_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ENGINE=PATH.")
    engine, path = value.split("=", 1)
    if not engine or not path:
        raise argparse.ArgumentTypeError("Expected non-empty ENGINE=PATH.")
    return engine, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-summary", action="append", type=_engine_summary, required=True
    )
    parser.add_argument("--require-engine", action="append", default=[])
    parser.add_argument("--paper8-logical-omission-fraction", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = dict(args.engine_summary)
    if len(summaries) != len(args.engine_summary):
        raise ValueError("Engine labels must be unique.")
    result = summarize(
        summaries,
        required_engines=tuple(args.require_engine),
        paper8_logical_omission_fraction=args.paper8_logical_omission_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["cross_engine_sequence_qualified"] else 1)


if __name__ == "__main__":
    main()
