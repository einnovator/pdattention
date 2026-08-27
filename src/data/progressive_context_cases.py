"""Balanced progressive-context fixtures for Paper 7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from pra_hf.adaptive_context_runtime import CursorOperation
from pra_hf.context_records import RecordType
from pra_hf.progressive_context import ContextAction, RecordCapabilities


class ContextCaseClass(str, Enum):
    """Preregistered sufficiency/escalation class for one typed-result task."""

    C0_CONTINUE = "C0_CONTINUE"
    C1_FULL = "C1_FULL"
    C2_MORE = "C2_MORE"
    C3_CURSOR = "C3_CURSOR"
    C4_SEARCH = "C4_SEARCH"
    C5_TOOL = "C5_TOOL"


@dataclass(frozen=True)
class ProgressiveContextCase:
    """One task whose evidence location determines the correct context action."""

    case_id: str
    case_class: ContextCaseClass
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


_DOMAINS = (
    ("billing", "BILLING", "DB"),
    ("deployment", "DEPLOY", "LOG"),
    ("inventory", "STOCK", "API"),
    ("identity", "IDENTITY", "GRAPH"),
    ("support", "SUPPORT", "RAG"),
)


def _answer(prefix: str, index: int) -> str:
    return f"{prefix}-{index:02d}-OK"


def _db_rows(domain: str, answer: str | None, *, answer_index: int = 7) -> dict[str, object]:
    rows = [
        {"row": index, "domain": domain, "state": "normal", "value": f"item-{index:02d}"}
        for index in range(20)
    ]
    if answer is not None:
        rows[answer_index] = {
            "row": answer_index,
            "domain": domain,
            "state": "target",
            "answer_code": answer,
            "lookup_key": f"needle-{domain}",
        }
    return {"columns": ["row", "domain", "state", "value", "answer_code", "lookup_key"], "rows": rows}


def _text_lines(domain: str, answer: str | None, *, answer_index: int = 7) -> str:
    lines = [f"line_{index:02d} domain={domain} status=normal" for index in range(24)]
    if answer is not None:
        lines[answer_index] = (
            f"line_{answer_index:02d} lookup=needle-{domain} ANSWER_CODE={answer}"
        )
    return "\n".join(lines)


def progressive_context_cases() -> tuple[ProgressiveContextCase, ...]:
    """Return five balanced identities for each of the six control classes."""

    cases = []
    for index, (domain, prefix, _kind) in enumerate(_DOMAINS):
        answer = _answer(prefix, index)
        query = (
            f"Return the ANSWER_CODE for lookup key needle-{domain} in the {domain} request."
        )

        # C0 places the answer in a representative row retained by compaction.
        cases.append(ProgressiveContextCase(
            f"c0-{domain}", ContextCaseClass.C0_CONTINUE, RecordType.DB_RESULT,
            _db_rows(domain, answer, answer_index=0), query, answer, answer,
            ContextAction.CONTINUE,
            RecordCapabilities(full_available=True, full_bounded=True),
        ))

        # C1 exposes no narrower interface; the bounded exact record is appropriate.
        full_payload = {
            "request": domain,
            "status": "complete",
            "owner": "paper7",
            "details": "ordinary",
            "notes": "ordinary",
            "audit": "ordinary",
            "lookup_key": f"needle-{domain}",
            "answer_code": answer,
        }
        cases.append(ProgressiveContextCase(
            f"c1-{domain}", ContextCaseClass.C1_FULL, RecordType.API_RESULT,
            full_payload, query, answer, answer, ContextAction.MATERIALIZE_FULL,
            RecordCapabilities(full_available=True, full_bounded=True),
        ))

        # C2 advertises the exact typed range that can expose a localized detail.
        cases.append(ProgressiveContextCase(
            f"c2-{domain}", ContextCaseClass.C2_MORE, RecordType.DB_RESULT,
            _db_rows(domain, answer), query, answer, answer,
            ContextAction.MATERIALIZE_MORE,
            RecordCapabilities(
                full_available=True,
                full_bounded=False,
                partial_selectors=("rows",),
            ),
            selector={"rows": [7, 8]},
        ))

        # C3 alternates sequential continuation and queryable-cursor decisions.
        cursor_query = None if index % 2 == 0 else {
            "operation": CursorOperation.SEARCH.value,
            "query": f"needle-{domain}",
            "limit": 2,
        }
        cases.append(ProgressiveContextCase(
            f"c3-{domain}", ContextCaseClass.C3_CURSOR, RecordType.DB_RESULT,
            _db_rows(domain, answer), query, answer, answer,
            ContextAction.CURSOR_NEXT if cursor_query is None else ContextAction.CURSOR_QUERY,
            RecordCapabilities(full_available=True, full_bounded=False),
            cursor_collection="rows",
            cursor_query=cursor_query,
        ))

        # C4 is a large known object with an exact in-record search surface.
        cases.append(ProgressiveContextCase(
            f"c4-{domain}", ContextCaseClass.C4_SEARCH, RecordType.LOG_BLOCK,
            _text_lines(domain, answer), query, answer, answer,
            ContextAction.SEARCH_RECORD,
            RecordCapabilities(full_available=True, full_bounded=False, searchable=True),
            search_query=f"needle-{domain}",
        ))

        # C5 deliberately omits the answer from retained backing state.
        cases.append(ProgressiveContextCase(
            f"c5-{domain}", ContextCaseClass.C5_TOOL, RecordType.API_RESULT,
            {
                "request": domain,
                "status": "requires_external_lookup",
                "available_fields": ["request", "status"],
            },
            query, answer, answer, ContextAction.CALL_TOOL,
            RecordCapabilities(full_available=True, full_bounded=True),
            tool_name=f"lookup_{domain}",
            tool_payload={"domain": domain, "answer_code": answer},
        ))
    return tuple(cases)
