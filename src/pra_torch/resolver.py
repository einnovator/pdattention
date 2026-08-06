"""Explicit URI resolver contract used by recursive PRA cache construction."""

from __future__ import annotations

from dataclasses import dataclass, field

from pra_core.references import ReferenceHandle, ResolvedReference as LegacyResolvedReference
from .refs import split_uri_anchor


@dataclass
class ResolvedReference:
    """Backend-neutral source object consumed by recursive cache construction."""

    uri: str  # Canonical identity requested by the caller.
    text: str  # Full source content encoded into chunk/layer K/V.
    reference_table: dict[str, str] = field(default_factory=dict)  # Local <REF_n> -> URI map.
    anchors: dict[str, str] = field(default_factory=dict)  # Optional named source fragments.
    metadata: dict = field(default_factory=dict)  # Dataset/backend provenance.
    version: str | None = None  # Source version included in cache identity.
    summary: str | None = None  # Optional short text encoded for summary routing.


class InMemoryResolver:
    """Dictionary-backed implementation of PRA's URI resolution boundary.

    PRA routes stable identities, not embedded document text. The resolver owns
    the mapping from a URI to current content, version, summary, and the local
    reference table used for recursive expansion.
    """

    def __init__(
        self,
        documents: dict,
        summaries: dict[str, str] | None = None,
        reference_tables: dict[str, dict[str, str]] | None = None,
        metadata: dict[str, dict] | None = None,
        versions: dict[str, str] | None = None,
    ):
        """Store document payloads and optional side tables by canonical URI."""
        self.documents = dict(documents)  # URI -> string or structured source object.
        self.summaries = summaries or {}  # URI -> routing summary text.
        self.reference_tables = reference_tables or {}  # URI -> local token/child URI map.
        self.metadata = metadata or {}  # URI -> backend/dataset metadata.
        self.versions = versions or {}  # URI -> source version for invalidation.

    def resolve(self, uri: str, summary_only: bool = False) -> ResolvedReference:
        """Resolve a URI (or its base anchor) into a normalized source object.

        ``summary_only`` changes returned text for compatibility; the summary is
        still retained separately so normal cache construction can encode both.
        """
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
        """Adapt the URI resolver result to the older handle-oriented API."""
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
