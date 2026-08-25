"""Typed declarative skill records for Paper 6.5 capability disclosure."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .agent_resources import AgentResource, SideEffectClass, normalize_text, resource_uri, terms
from .context_records import ContextRecord, RecordType, RecordView, RecordViewName


@dataclass(frozen=True)
class SkillRecord:
    """Versioned text-only procedural capability with selection/full views."""

    name: str
    description: str
    when_to_use: str
    instructions: str
    namespace: str = "default"
    tenant_id: str = "default"
    version: str = "v1"
    aliases: tuple[str, ...] = ()
    manual_tags: frozenset[str] = frozenset()
    auto_tags: frozenset[str] = frozenset()
    keywords: frozenset[str] = frozenset()
    constraints: tuple[str, ...] = ()
    ordered_steps: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("name", "description", "when_to_use", "instructions", "namespace", "version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"SkillRecord {field_name} is required.")
        if self.metadata.get("script") or self.metadata.get("executable"):
            raise ValueError("Paper 6.5 skills are declarative text only.")
        object.__setattr__(self, "aliases", tuple(dict.fromkeys(self.aliases)))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "ordered_steps", tuple(self.ordered_steps))
        object.__setattr__(self, "examples", tuple(self.examples))
        object.__setattr__(self, "dependencies", tuple(dict.fromkeys(self.dependencies)))
        object.__setattr__(self, "references", tuple(dict.fromkeys(self.references)))
        object.__setattr__(self, "manual_tags", frozenset(normalize_text(value) for value in self.manual_tags if value))
        inferred = {
            token for token in terms(" ".join((self.name, self.description, self.when_to_use)))
            if len(token) > 2
        }
        object.__setattr__(self, "auto_tags", frozenset((*self.auto_tags, *inferred)))
        object.__setattr__(self, "keywords", frozenset((*self.keywords, *inferred)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def uri(self) -> str:
        return resource_uri("skill", self.namespace, self.name, self.version)

    @property
    def selection_payload(self) -> str:
        return "\n".join((
            self.name,
            self.description,
            f"Use when: {self.when_to_use}",
        ))

    @property
    def full_payload(self) -> dict[str, object]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "instructions": self.instructions,
            "constraints": list(self.constraints),
            "ordered_steps": list(self.ordered_steps),
            "examples": list(self.examples),
            "dependencies": list(self.dependencies),
            "references": list(self.references),
            "version": self.version,
            "namespace": self.namespace,
        }

    def to_context_record(
        self,
        *,
        parent_id: str | None = None,
        selection_provenance: Mapping[str, object] | None = None,
    ) -> ContextRecord:
        """Return an atomic context record with deterministic named views."""

        full = self.full_payload
        fingerprint = hashlib.sha256(
            json.dumps(full, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ContextRecord(
            record_id=self.uri,
            record_type=RecordType.SKILL,
            payload=full,
            parent_id=parent_id,
            selection_provenance=dict(selection_provenance or {}),
            version=self.version,
            source_fingerprint=fingerprint,
            views={
                RecordViewName.SELECTION: RecordView(
                    RecordViewName.SELECTION,
                    self.selection_payload,
                    ("name", "description", "when_to_use"),
                ),
                RecordViewName.FULL: RecordView(
                    RecordViewName.FULL,
                    full,
                    tuple(full),
                ),
            },
        )

    def to_agent_resource(self) -> AgentResource:
        """Expose the same typed skill through the shared discovery substrate."""

        semantic_terms = (
            *self.aliases,
            *sorted(self.manual_tags),
            *sorted(self.auto_tags),
            self.when_to_use,
        )
        return AgentResource(
            uri=self.uri,
            kind="skill",
            namespace=self.namespace,
            name=self.name,
            version=self.version,
            description=self.description,
            content=json.dumps(self.full_payload, sort_keys=True),
            aliases=self.aliases,
            side_effect_class=SideEffectClass.NONE,
            tenant_id=self.tenant_id,
            metadata={
                **dict(self.metadata),
                "tags": tuple(sorted(self.manual_tags)),
                "auto_tags": tuple(sorted(self.auto_tags)),
                "keywords": tuple(sorted(self.keywords)),
                "semantic_terms": semantic_terms,
                "record_type": RecordType.SKILL.value,
                "declarative_only": True,
            },
        )


def skill_records_to_resources(records: Sequence[SkillRecord]) -> tuple[AgentResource, ...]:
    """Convert a stable skill registry without creating a separate index type."""

    return tuple(record.to_agent_resource() for record in records)
