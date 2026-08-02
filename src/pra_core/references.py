from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReferenceHandle:
    id: int
    token: str
    uri: str
    summary: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ResolvedReference:
    handle: ReferenceHandle
    text: str
    children: list[ReferenceHandle] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class ReferenceTable:
    """Runtime mapping from lightweight reference tokens to source URIs."""

    def __init__(self):
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
        return self._by_id[id]

    def find_by_token(self, token: str) -> ReferenceHandle | None:
        return self._by_token.get(token)

    def all(self) -> list[ReferenceHandle]:
        return list(self._by_id.values())
