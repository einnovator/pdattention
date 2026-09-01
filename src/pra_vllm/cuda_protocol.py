"""Stable control messages for the experimental vLLM CUDA KV connector."""

from __future__ import annotations

import re
from dataclasses import dataclass


_PREFIX = "pra-cuda-v1"
_SCOPED_PREFIX = "pra-cuda-v2"
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


@dataclass(frozen=True)
class CudaConnectorCommand:
    """Describe one semantic-resource store or load request."""

    mode: str
    logical_key: str
    source_tokens: int
    residency: str = "warm"
    request_scope: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"store", "load"}:
            raise ValueError(f"Unknown CUDA connector mode: {self.mode}")
        if self.residency not in {"warm", "hot"}:
            raise ValueError(f"Unknown CUDA connector residency: {self.residency}")
        if self.mode == "store" and self.residency != "warm":
            raise ValueError("CUDA connector store commands do not select residency.")
        if self.request_scope is not None and not _KEY.fullmatch(self.request_scope):
            raise ValueError("CUDA connector request scopes must be URL-safe identifiers.")
        if not _KEY.fullmatch(self.logical_key):
            raise ValueError("CUDA connector logical keys must be URL-safe identifiers.")
        if self.source_tokens <= 0:
            raise ValueError("CUDA connector source_tokens must be positive.")

    def cache_salt(self) -> str:
        """Encode the command in vLLM's request-scoped cache-salt field."""

        if self.residency == "warm" and self.request_scope is None:
            return f"{_PREFIX}:{self.mode}:{self.source_tokens}:{self.logical_key}"
        return (
            f"{_SCOPED_PREFIX}:{self.mode}:{self.residency}:"
            f"{self.source_tokens}:{self.logical_key}:{self.request_scope or '-'}"
        )

    @classmethod
    def parse(cls, value: str | None) -> "CudaConnectorCommand | None":
        """Decode a PRA command, returning ``None`` for ordinary requests."""

        if not value or not value.startswith((f"{_PREFIX}:", f"{_SCOPED_PREFIX}:")):
            return None
        parts = value.split(":")
        request_scope: str | None = None
        if parts[0] == _PREFIX and len(parts) == 4:
            _, mode, source_tokens, logical_key = parts
            residency = "warm"
        elif parts[0] == _SCOPED_PREFIX and len(parts) == 6:
            _, mode, residency, source_tokens, logical_key, scope = parts
            request_scope = None if scope == "-" else scope
        else:
            raise ValueError("Malformed PRA CUDA connector cache salt.")
        try:
            count = int(source_tokens)
        except ValueError as error:
            raise ValueError("Malformed PRA CUDA source-token count.") from error
        return cls(
            mode=mode,
            logical_key=logical_key,
            source_tokens=count,
            residency=residency,
            request_scope=request_scope,
        )
