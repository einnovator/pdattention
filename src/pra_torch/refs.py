import re
import warnings
from dataclasses import dataclass

from pra_core.references import ReferenceHandle, ReferenceTable

LEGACY_REF_PATTERN = re.compile(r"!!ref:(?P<uri>[^!]+)!!")
REF_TOKEN_PATTERN = re.compile(r"<REF_(?P<id>\d+)>")


@dataclass(frozen=True)
class RefHandle:
    raw: str
    uri: str
    start: int
    end: int


@dataclass(frozen=True)
class ReferenceTokenOccurrence:
    token: str
    id: int
    start: int
    end: int
    handle: ReferenceHandle | None = None


def parse_refs(text: str) -> list[RefHandle]:
    """Deprecated parser for legacy !!ref:uri!! handles."""
    warnings.warn(
        "parse_refs() supports deprecated !!ref:uri!! syntax; use parse_ref_tokens() with a ReferenceTable.",
        DeprecationWarning,
        stacklevel=2,
    )
    refs: list[RefHandle] = []
    for m in LEGACY_REF_PATTERN.finditer(text):
        refs.append(RefHandle(raw=m.group(0), uri=m.group("uri"), start=m.start(), end=m.end()))
    return refs


def parse_ref_tokens(text: str, reference_table: ReferenceTable | None = None) -> list[ReferenceTokenOccurrence]:
    """Parse lightweight <REF_n> tokens and optionally attach table handles."""
    refs: list[ReferenceTokenOccurrence] = []
    for m in REF_TOKEN_PATTERN.finditer(text):
        token = m.group(0)
        refs.append(
            ReferenceTokenOccurrence(
                token=token,
                id=int(m.group("id")),
                start=m.start(),
                end=m.end(),
                handle=reference_table.find_by_token(token) if reference_table else None,
            )
        )
    return refs


def normalize_ref_tokens(text: str, ref_token: str | None = None) -> str:
    """Normalize legacy refs while preserving new <REF_n> tokens.

    New prompts should already contain lightweight reference tokens such as
    <REF_1>. If old !!ref:uri!! text is encountered, it can still be collapsed
    for compatibility.
    """
    if ref_token is None:
        ref_token = "<REF>"
    return LEGACY_REF_PATTERN.sub(ref_token, text)


def split_uri_anchor(uri: str) -> tuple[str, str | None]:
    if "#" not in uri:
        return uri, None
    base, anchor = uri.split("#", 1)
    return base, anchor or None


def parent_anchor(anchor: str | None) -> str | None:
    if not anchor or "." not in anchor:
        return None
    return ".".join(anchor.split(".")[:-1])


def child_anchor(anchor: str | None, child: str) -> str:
    if not anchor:
        return child
    return f"{anchor}.{child}"
