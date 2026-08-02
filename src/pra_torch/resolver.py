from dataclasses import dataclass, field
from pra_core.references import ReferenceHandle, ResolvedReference
from .refs import split_uri_anchor


@dataclass
class ResolvedRef:
    uri: str
    text: str
    summary: str
    children: dict[str, str] = field(default_factory=dict)  # child_anchor -> summary
    metadata: dict = field(default_factory=dict)


class InMemoryResolver:
    """Simple URI resolver for synthetic experiments.

    Store keys like:
      mem://doc1
      mem://doc1#section.a
    """

    def __init__(self, documents: dict[str, str], summaries: dict[str, str] | None = None):
        self.documents = documents
        self.summaries = summaries or {}

    def resolve(self, uri: str, summary_only: bool = False) -> ResolvedRef:
        if uri not in self.documents and uri not in self.summaries:
            base, anchor = split_uri_anchor(uri)
            if base in self.documents:
                text = self.documents[base]
            else:
                text = f"[missing resource: {uri}]"
        else:
            text = self.documents.get(uri, self.summaries.get(uri, ""))

        summary = self.summaries.get(uri)
        if summary is None:
            summary = self._simple_summary(text)

        if summary_only:
            text = summary

        children = {
            k: self.summaries.get(k, self._simple_summary(v))
            for k, v in self.documents.items()
            if k.startswith(uri + ".") or k.startswith(uri + "/")
        }
        return ResolvedRef(uri=uri, text=text, summary=summary, children=children)

    def resolve_handle(self, handle: ReferenceHandle, summary_only: bool = False) -> ResolvedReference:
        resolved = self.resolve(handle.uri, summary_only=summary_only)
        children = [
            ReferenceHandle(
                id=-1,
                token=f"{handle.token}.{i + 1}",
                uri=uri,
                summary=summary,
            )
            for i, (uri, summary) in enumerate(resolved.children.items())
        ]
        return ResolvedReference(
            handle=handle,
            text=resolved.text,
            children=children,
            metadata=dict(resolved.metadata),
        )

    @staticmethod
    def _simple_summary(text: str, max_chars: int = 160) -> str:
        text = " ".join(text.split())
        return text[:max_chars]
