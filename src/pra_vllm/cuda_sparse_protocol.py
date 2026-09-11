"""Sparse-position control messages for the experimental vLLM CUDA connector."""

from __future__ import annotations

from dataclasses import dataclass

from pra_vllm.cuda_protocol import CudaConnectorCommand


_PREFIX = "pra-cuda-sparse-v2"
_LEGACY_PREFIX = "pra-cuda-sparse-v1"


@dataclass(frozen=True)
class SparseCudaConnectorCommand:
    """Attach selected K/V while positioning the suffix after the full source."""

    mode: str
    logical_key: str
    source_tokens: int
    source_position_base: int
    source_generation: int = 1
    residency: str = "hot"
    request_scope: str | None = None

    def __post_init__(self) -> None:
        # Reuse the stable key and mode validation from the contiguous protocol.
        CudaConnectorCommand(
            self.mode,
            self.logical_key,
            self.source_tokens,
            self.residency,
            self.request_scope,
        )
        if self.source_position_base < self.source_tokens:
            raise ValueError(
                "Sparse CUDA source_position_base must cover selected source tokens."
            )
        if self.source_generation <= 0:
            raise ValueError("Sparse CUDA source_generation must be positive.")

    def cache_salt(self) -> str:
        return (
            f"{_PREFIX}:{self.mode}:{self.residency}:{self.source_tokens}:"
            f"{self.source_position_base}:{self.source_generation}:"
            f"{self.logical_key}:{self.request_scope or '-'}"
        )

    @classmethod
    def parse(
        cls, value: str | None
    ) -> "SparseCudaConnectorCommand | CudaConnectorCommand | None":
        if not value or not value.startswith((f"{_PREFIX}:", f"{_LEGACY_PREFIX}:")):
            return CudaConnectorCommand.parse(value)
        parts = value.split(":")
        if parts[0] == _PREFIX and len(parts) == 8:
            (
                _, mode, residency, selected, position_base, generation,
                logical_key, scope,
            ) = parts
        elif parts[0] == _LEGACY_PREFIX and len(parts) == 7:
            _, mode, residency, selected, position_base, logical_key, scope = parts
            generation = "1"
        else:
            raise ValueError("Malformed sparse PRA CUDA connector cache salt.")
        try:
            selected_tokens = int(selected)
            source_position_base = int(position_base)
            source_generation = int(generation)
        except ValueError as error:
            raise ValueError("Malformed sparse PRA CUDA token geometry.") from error
        return cls(
            mode=mode,
            logical_key=logical_key,
            source_tokens=selected_tokens,
            source_position_base=source_position_base,
            source_generation=source_generation,
            residency=residency,
            request_scope=None if scope == "-" else scope,
        )
