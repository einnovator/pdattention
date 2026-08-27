"""Validation/test fixtures for full-backing PRA and controller calibration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from pra_hf.adaptive_context_runtime import CursorOperation
from pra_hf.context_records import RecordType
from pra_hf.progressive_context import ContextAction, RecordCapabilities


class OmissionStratum(str, Enum):
    """Relationship between the visible compact view and required evidence."""

    COMPACT_EXPLICIT = "compact_explicit"
    SEMANTIC_OMISSION = "semantic_omission"
    OPAQUE_HIDDEN = "opaque_hidden"
    ABSENT_FROM_BACKING = "absent_from_backing"


@dataclass(frozen=True)
class FullPRAContextCase:
    """One typed-result task with a known evidence location and control action."""

    case_id: str
    partition: str
    case_class: str
    omission_stratum: OmissionStratum
    record_type: RecordType
    payload: object
    query: str
    expected_answer: str
    evidence_marker: str
    expected_action: ContextAction
    capabilities: RecordCapabilities
    selector: Mapping[str, object] | None = None
    search_query: str | None = None
    cursor_collection: str | None = None
    cursor_query: Mapping[str, object] | None = None
    tool_name: str | None = None
    tool_payload: object | None = None
    semantic_cue: str | None = None


_DOMAINS = (
    ("billing", "BILLING", "payment authentication failures", "merchant token expired"),
    ("deployment", "DEPLOY", "service rollout failures", "dependency health check blocked restart"),
    ("inventory", "STOCK", "inventory count anomalies", "warehouse scanner duplicate update"),
    ("identity", "IDENTITY", "sign in authorization failures", "service account certificate expired"),
    ("support", "SUPPORT", "customer ticket escalation", "attachment parser rejected archive"),
)


def _answer(prefix: str, index: int) -> str:
    return f"{prefix}-{index:02d}-OK"


def _lookup(domain: str, index: int) -> str:
    return f"ZX-{domain[:3].upper()}-{index:02d}-91Q"


def _query(domain: str, specific: str, lookup: str, natural: bool) -> str:
    if natural:
        return f"Return the ANSWER_CODE for the {specific} in the {domain} result."
    return f"Return the ANSWER_CODE associated with opaque lookup key {lookup}."


def _db_payload(
    domain: str,
    broad: str,
    specific: str,
    lookup: str,
    answer: str,
    *,
    target_index: int,
    natural: bool,
) -> dict[str, object]:
    rows = [
        {
            "row": row,
            "domain": domain,
            "status": "ordinary",
            "observation": f"routine item {row:02d}",
        }
        for row in range(20)
    ]
    rows[0]["observation"] = broad if natural else "routine processing summary"
    rows[target_index] = {
        "row": target_index,
        "domain": domain,
        "status": "target",
        "observation": specific if natural else "opaque keyed result",
        "lookup_key": lookup,
        "answer_code": answer,
    }
    return {
        "columns": [
            "row", "domain", "status", "observation", "lookup_key", "answer_code"
        ],
        "rows": rows,
    }


def _log_payload(
    domain: str,
    broad: str,
    specific: str,
    lookup: str,
    answer: str,
    *,
    natural: bool,
) -> str:
    lines = [f"line_{row:02d} domain={domain} status=ordinary" for row in range(24)]
    lines[0] = (
        f"line_00 domain={domain} warning={broad}"
        if natural else f"line_00 domain={domain} status=routine"
    )
    lines[7] = (
        f"line_07 diagnosis={specific} ANSWER_CODE={answer}"
        if natural else f"line_07 lookup={lookup} ANSWER_CODE={answer}"
    )
    return "\n".join(lines)


def full_pra_context_cases() -> tuple[FullPRAContextCase, ...]:
    """Return balanced validation/test cases without changing frozen v1 fixtures."""

    cases = []
    for index, (domain, prefix, broad, specific) in enumerate(_DOMAINS):
        partition = "validation" if index < 2 else "test"
        natural = index % 2 == 0
        hidden_stratum = (
            OmissionStratum.SEMANTIC_OMISSION
            if natural else OmissionStratum.OPAQUE_HIDDEN
        )
        answer = _answer(prefix, index)
        lookup = _lookup(domain, index)
        query = _query(domain, specific, lookup, natural)

        explicit_payload = _db_payload(
            domain, broad, specific, lookup, answer,
            target_index=0, natural=natural,
        )
        cases.append(FullPRAContextCase(
            f"c0-{domain}", partition, "C0_CONTINUE",
            OmissionStratum.COMPACT_EXPLICIT, RecordType.DB_RESULT,
            explicit_payload, query, answer, answer, ContextAction.CONTINUE,
            RecordCapabilities(full_available=True, full_bounded=True),
            semantic_cue=broad if natural else lookup,
        ))

        api_payload = {
            "request": domain,
            "status": "stored_result_complete",
            "issue_summary": broad if natural else "routine processing summary",
            "owner": "paper7",
            "diagnosis": specific if natural else "opaque keyed result",
            "lookup_key": lookup,
            "answer_code": answer,
        }
        cases.append(FullPRAContextCase(
            f"c1-{domain}", partition, "C1_FULL", hidden_stratum,
            RecordType.API_RESULT, api_payload, query, answer, answer,
            ContextAction.MATERIALIZE_FULL,
            RecordCapabilities(full_available=True, full_bounded=True),
            semantic_cue=broad if natural else None,
        ))

        db_payload = _db_payload(
            domain, broad, specific, lookup, answer,
            target_index=7, natural=natural,
        )
        cases.append(FullPRAContextCase(
            f"c2-{domain}", partition, "C2_MORE", hidden_stratum,
            RecordType.DB_RESULT, db_payload, query, answer, answer,
            ContextAction.MATERIALIZE_MORE,
            RecordCapabilities(
                full_available=True, full_bounded=False,
                partial_selectors=("rows",),
            ),
            selector={"rows": [7, 8]}, semantic_cue=broad if natural else None,
        ))

        cursor_target = 5 if natural else 7
        cursor_payload = _db_payload(
            domain, broad, specific, lookup, answer,
            target_index=cursor_target, natural=natural,
        )
        cursor_query = None if natural else {
            "operation": CursorOperation.SEARCH.value,
            "query": lookup,
            "limit": 2,
        }
        cases.append(FullPRAContextCase(
            f"c3-{domain}", partition, "C3_CURSOR", hidden_stratum,
            RecordType.DB_RESULT, cursor_payload, query, answer, answer,
            ContextAction.CURSOR_NEXT if natural else ContextAction.CURSOR_QUERY,
            RecordCapabilities(full_available=True, full_bounded=False),
            cursor_collection="rows", cursor_query=cursor_query,
            semantic_cue=broad if natural else None,
        ))

        cases.append(FullPRAContextCase(
            f"c4-{domain}", partition, "C4_SEARCH", hidden_stratum,
            RecordType.LOG_BLOCK,
            _log_payload(domain, broad, specific, lookup, answer, natural=natural),
            query, answer, answer, ContextAction.SEARCH_RECORD,
            RecordCapabilities(full_available=True, full_bounded=False, searchable=True),
            search_query=specific if natural else lookup,
            semantic_cue=broad if natural else None,
        ))

        cases.append(FullPRAContextCase(
            f"c5-{domain}", partition, "C5_TOOL",
            OmissionStratum.ABSENT_FROM_BACKING, RecordType.API_RESULT,
            {
                "request": domain,
                "status": "requires_external_lookup",
                "issue_summary": broad if natural else "opaque external request",
            },
            query, answer, answer, ContextAction.CALL_TOOL,
            RecordCapabilities(full_available=True, full_bounded=True),
            tool_name=f"lookup_{domain}",
            tool_payload={"domain": domain, "answer_code": answer},
            semantic_cue=broad if natural else None,
        ))
    return tuple(cases)
