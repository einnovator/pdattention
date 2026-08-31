"""Stable control messages for the experimental vLLM CUDA KV connector."""

from __future__ import annotations

import re
from dataclasses import dataclass


_PREFIX = "pra-cuda-v1"
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


@dataclass(frozen=True)
class CudaConnectorCommand:
    """Describe one semantic-resource store or load request."""

    mode: str
    logical_key: str
    source_tokens: int

    def __post_init__(self) -> None:
        if self.mode not in {"store", "load"}:
            raise ValueError(f"Unknown CUDA connector mode: {self.mode}")
        if not _KEY.fullmatch(self.logical_key):
            raise ValueError("CUDA connector logical keys must be URL-safe identifiers.")
        if self.source_tokens <= 0:
            raise ValueError("CUDA connector source_tokens must be positive.")

    def cache_salt(self) -> str:
        """Encode the command in vLLM's request-scoped cache-salt field."""

        return f"{_PREFIX}:{self.mode}:{self.source_tokens}:{self.logical_key}"

    @classmethod
    def parse(cls, value: str | None) -> "CudaConnectorCommand | None":
        """Decode a PRA command, returning ``None`` for ordinary requests."""

        if not value or not value.startswith(f"{_PREFIX}:"):
            return None
        parts = value.split(":", 3)
        if len(parts) != 4:
            raise ValueError("Malformed PRA CUDA connector cache salt.")
        _, mode, source_tokens, logical_key = parts
        try:
            count = int(source_tokens)
        except ValueError as error:
            raise ValueError("Malformed PRA CUDA source-token count.") from error
        return cls(mode=mode, logical_key=logical_key, source_tokens=count)

