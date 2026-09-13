"""Portable tool-effect declarations used by agent-memory mediation.

PRA understands generic effects and versioned resources.  It does not know
application tool names, shell syntax, browser actions, database commands, or
agent control loops.  A harness can attach a complete declaration; otherwise
the operation remains an exclusion barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

from .agent_history import AgentRecord


class EffectKind(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    VERIFY = "verify"
    PURE = "pure"
    UNKNOWN = "unknown"


class OperationKind(str, Enum):
    """Portable operation class used by history policies.

    Tool adapters map their native calls to this deliberately small vocabulary.
    The policy layer must never recover this value by parsing tool syntax.
    """

    SEARCH_DISCOVERY = "search_discovery"
    READ = "read"
    WRITE = "write"
    DIFF = "diff"
    VERIFY = "verify"
    OTHER = "other"
    UNKNOWN = "unknown"


class EffectProvenance(str, Enum):
    DECLARED_TOOL_SEMANTICS = "declared_tool_semantics"
    RUNTIME_TRACED = "runtime_traced"
    STATIC_HEURISTIC = "static_heuristic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResourceEffect:
    record_id: str
    resource_id: str
    kind: EffectKind
    resource_version_fingerprint: str | None = None
    provenance: EffectProvenance = EffectProvenance.STATIC_HEURISTIC
    span: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EffectKind(self.kind))
        object.__setattr__(self, "provenance", EffectProvenance(self.provenance))
        if self.span is not None:
            object.__setattr__(self, "span", dict(self.span))


@dataclass(frozen=True)
class ResourceAccess:
    """Versioned resource evidence independent of the originating tool."""

    resource_id: str
    version: str
    span_kind: str = "unknown"
    span_start: int | None = None
    span_end: int | None = None
    signature: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResourceAccess":
        return cls(
            resource_id=str(value["resource_id"]),
            version=str(value.get("version") or "unknown"),
            span_kind=str(value.get("span_kind") or "unknown"),
            span_start=(int(value["span_start"]) if value.get("span_start") is not None else None),
            span_end=(int(value["span_end"]) if value.get("span_end") is not None else None),
            signature=(str(value["signature"]) if value.get("signature") is not None else None),
        )


@dataclass(frozen=True)
class ToolEffectAnalysis:
    tool_category: str
    effects: tuple[ResourceEffect, ...]
    provenance: EffectProvenance
    complete: bool
    unknown_barrier: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "provenance", EffectProvenance(self.provenance))
        if self.complete and self.unknown_barrier:
            raise ValueError("a complete tool-effect analysis cannot be an unknown barrier")


class ToolSemanticsProvider(Protocol):
    def analyze(
        self,
        action: AgentRecord,
        observations: tuple[AgentRecord, ...],
    ) -> ToolEffectAnalysis: ...


class DeclaredToolSemanticsProvider:
    """Validate generic effect metadata supplied by a tool or agent harness."""

    trusted_provenance = {
        EffectProvenance.DECLARED_TOOL_SEMANTICS,
        EffectProvenance.RUNTIME_TRACED,
    }

    def analyze(
        self,
        action: AgentRecord,
        observations: tuple[AgentRecord, ...],
    ) -> ToolEffectAnalysis:
        declaration = next((
            row.metadata.get("tool_semantics")
            for row in reversed(observations)
            if isinstance(row.metadata.get("tool_semantics"), Mapping)
        ), None)
        if declaration is None:
            return ToolEffectAnalysis(
                "unknown", (), EffectProvenance.UNKNOWN, False, True,
            )
        try:
            provenance = EffectProvenance(str(declaration["provenance"]))
            complete = bool(declaration["complete"])
            effects = tuple(
                ResourceEffect(
                    record_id=action.record_id,
                    resource_id=str(row["resource_id"]),
                    kind=EffectKind(str(row["kind"])),
                    resource_version_fingerprint=(
                        str(row["resource_version_fingerprint"])
                        if row.get("resource_version_fingerprint") is not None
                        else None
                    ),
                    provenance=provenance,
                    span=(dict(row["span"]) if isinstance(row.get("span"), Mapping) else None),
                )
                for row in declaration.get("effects", ())
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid tool_semantics declaration") from error
        trusted = provenance in self.trusted_provenance
        authoritative = bool(trusted and complete and effects)
        return ToolEffectAnalysis(
            str(declaration.get("category", "generic")),
            effects,
            provenance,
            complete and trusted,
            not authoritative,
        )


class UnknownToolSemanticsProvider:
    """Fail-closed provider used when no application declaration is available."""

    def analyze(
        self,
        action: AgentRecord,
        observations: tuple[AgentRecord, ...],
    ) -> ToolEffectAnalysis:
        del action, observations
        return ToolEffectAnalysis(
            "unknown", (), EffectProvenance.UNKNOWN, False, True,
        )


__all__ = [
    "DeclaredToolSemanticsProvider",
    "EffectKind",
    "EffectProvenance",
    "OperationKind",
    "ResourceAccess",
    "ResourceEffect",
    "ToolEffectAnalysis",
    "ToolSemanticsProvider",
    "UnknownToolSemanticsProvider",
]
