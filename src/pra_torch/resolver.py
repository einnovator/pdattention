"""Explicit URI resolver contract used by recursive PRA cache construction."""

from __future__ import annotations

from dataclasses import dataclass, field

from pra_core.references import ReferenceHandle, ResolvedReference as LegacyResolvedReference
from .refs import split_uri_anchor


@dataclass
class ResolvedReference:
    uri: str
    text: str
    reference_table: dict[str, str] = field(default_factory=dict)
    anchors: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    version: str | None = None
    summary: str | None = None


class InMemoryResolver:
    """Dictionary-backed resolver with explicit local child-reference tables."""

    def __init__(
        self,
        documents: dict,
        summaries: dict[str, str] | None = None,
        reference_tables: dict[str, dict[str, str]] | None = None,
        metadata: dict[str, dict] | None = None,
        versions: dict[str, str] | None = None,
    ):
        self.documents = dict(documents)
        self.summaries = summaries or {}
        self.reference_tables = reference_tables or {}
        self.metadata = metadata or {}
        self.versions = versions or {}

    def resolve(self, uri: str, summary_only: bool = False) -> ResolvedReference:
        value = self.documents.get(uri)
        if value is None:
            base, _anchor = split_uri_anchor(uri)
            value = self.documents.get(base)
        if value is None:
            raise KeyError(f"Reference URI is not available: {uri}")
        if isinstance(value, dict):
            text = str(value.get("text", ""))
            table = dict(value.get("reference_table") or self.reference_tables.get(uri) or {})
            anchors = dict(value.get("anchors") or {})
            item_metadata = {**self.metadata.get(uri, {}), **dict(value.get("metadata") or {})}
            version = value.get("version", self.versions.get(uri))
            summary = value.get("summary", self.summaries.get(uri))
        else:
            text = str(value)
            table = dict(self.reference_tables.get(uri) or {})
            anchors = {}
            item_metadata = dict(self.metadata.get(uri, {}))
            version = self.versions.get(uri)
            summary = self.summaries.get(uri)
        return ResolvedReference(
            uri=uri,
            text=str(summary or "") if summary_only else text,
            reference_table={str(token): str(child_uri) for token, child_uri in table.items()},
            anchors=anchors,
            metadata=item_metadata,
            version=str(version) if version is not None else None,
            summary=str(summary) if summary is not None else None,
        )

    def resolve_handle(self, handle: ReferenceHandle, summary_only: bool = False) -> LegacyResolvedReference:
        resolved = self.resolve(handle.uri, summary_only=summary_only)
        children = [
            ReferenceHandle(id=-1, token=token, uri=uri)
            for token, uri in resolved.reference_table.items()
        ]
        return LegacyResolvedReference(
            handle=handle,
            text=resolved.text,
            children=children,
            metadata={**resolved.metadata, "version": resolved.version},
        )
