from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReferenceHandle:
    """Lightweight prompt token bound to a stable external reference identity."""

    id: int  # Dataset/runtime-local numeric identifier.
    token: str  # Prompt-visible handle such as ``<REF_1>``.
    uri: str  # Stable source identity resolved outside the prompt.
    summary: str | None = None  # Optional routing/preview text, not source identity.
    metadata: dict = field(default_factory=dict)  # Dataset and resolver payload.


@dataclass
class ResolvedReference:
    """Legacy handle-oriented view of resolved source text and child handles."""

    handle: ReferenceHandle  # Handle whose URI was resolved.
    text: str  # Resolved full or summary content.
    children: list[ReferenceHandle] = field(default_factory=list)  # Locally named children.
    metadata: dict = field(default_factory=dict)  # Version/backend provenance.


class ReferenceTable:
    """Runtime mapping from lightweight reference tokens to source URIs."""

    def __init__(self):
        """Create independent ID and token indexes for prompt-time lookup."""
        self._by_id: dict[int, ReferenceHandle] = {}
        self._by_token: dict[str, ReferenceHandle] = {}
        self._next_id = 1

    def register(
        self,
        uri: str,
        summary: str | None = None,
        metadata: dict | None = None,
        id: int | None = None,
        token: str | None = None,
    ) -> ReferenceHandle:
        """Bind a token/ID to a URI and advance automatic ID allocation."""
        ref_id = self._next_id if id is None else int(id)
        self._next_id = max(self._next_id, ref_id + 1)
        handle = ReferenceHandle(
            id=ref_id,
            token=token or f"<REF_{ref_id}>",
            uri=uri,
            summary=summary,
            metadata=dict(metadata or {}),
        )
        self._by_id[ref_id] = handle
        self._by_token[handle.token] = handle
        return handle

    def get(self, id: int) -> ReferenceHandle:
        """Return a handle by numeric runtime ID."""
        return self._by_id[id]

    def find_by_token(self, token: str) -> ReferenceHandle | None:
        """Resolve a prompt-visible token to its URI-bearing handle."""
        return self._by_token.get(token)

    def all(self) -> list[ReferenceHandle]:
        """Return handles in registration/insertion order."""
        return list(self._by_id.values())
