"""Realistic nested callable schemas for Paper 6.5 disclosure stress tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class DeliveryChannel(str, Enum):
    """Supported report-delivery transports."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry behavior for one remote operation."""

    attempts: int
    backoff_seconds: float
    retry_on: list[int]


@dataclass(frozen=True)
class DeliveryTarget:
    """One typed destination and its delivery constraints."""

    channel: DeliveryChannel
    address: str
    locale: str
    retry: RetryPolicy | None = None


@dataclass(frozen=True)
class ReportSection:
    """A requested report section with structured evidence inputs."""

    title: str
    source_ids: list[str]
    rendering: Literal["table", "chart", "narrative"]
    required: bool = True


@dataclass(frozen=True)
class AuditPolicy:
    """Retention and approval policy attached to a generated artifact."""

    retention_days: int
    reviewers: list[str]
    classification: Literal["public", "internal", "confidential"]
    legal_hold: bool = False


def publish_governed_report(
    report_id: str,
    title: str,
    sections: list[ReportSection],
    targets: list[DeliveryTarget],
    audit: AuditPolicy,
    output_format: Literal["pdf", "html", "docx"] = "pdf",
    labels: list[str] | None = None,
) -> dict[str, list[str] | str]:
    """Render and deliver a governed report to typed destinations.

    Args:
        report_id: Stable report identifier.
        title: Human-readable publication title.
        sections: Ordered report sections and their evidence sources.
        targets: Typed delivery destinations with retry behavior.
        audit: Retention, classification, and reviewer requirements.
        output_format: Rendered artifact format.
        labels: Optional catalog labels attached to the publication.

    Returns:
        Publication identifier and accepted delivery destinations.
    """

    return {"report_id": report_id, "deliveries": [row.address for row in targets]}


def schedule_governed_report(
    report_id: str,
    schedule: Literal["daily", "weekly", "monthly"],
    timezone: str,
    sections: list[ReportSection],
    targets: list[DeliveryTarget],
    audit: AuditPolicy,
    start_at: str | None = None,
) -> dict[str, str]:
    """Schedule recurring production of a governed report.

    Args:
        report_id: Stable report identifier.
        schedule: Recurrence frequency.
        timezone: IANA timezone used for recurrence boundaries.
        sections: Ordered report sections and evidence sources.
        targets: Typed delivery destinations.
        audit: Retention and approval requirements.
        start_at: Optional ISO-8601 first-run timestamp.

    Returns:
        Stable schedule identifier and activation state.
    """

    return {"schedule_id": f"schedule-{report_id}", "state": "active"}


def validate_governed_report(
    report_id: str,
    sections: list[ReportSection],
    audit: AuditPolicy,
    checks: list[Literal["sources", "permissions", "retention", "rendering"]],
    fail_on_warning: bool = False,
) -> dict[str, object]:
    """Validate evidence and governance requirements before publication.

    Args:
        report_id: Stable report identifier.
        sections: Sections whose source links must be validated.
        audit: Governance policy to enforce.
        checks: Validation categories to execute.
        fail_on_warning: Whether warnings should reject validation.

    Returns:
        Validation status and completed check names.
    """

    return {"report_id": report_id, "valid": True, "checks": checks}


LARGE_SCHEMA_CALLABLES = (
    publish_governed_report,
    schedule_governed_report,
    validate_governed_report,
)
