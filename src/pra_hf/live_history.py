"""Engine-neutral contracts for selecting records from resident agent K/V.

The selection plan describes logical token coordinates in the canonical live
agent trajectory.  Engines may expose those cells as views, sequence
memberships, or page references, but must not reconstruct them from text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class LiveKVInterval:
    """Half-open token interval in the canonical source-position frame."""

    start: int
    end: int
    record_id: str = ""
    causal_group_id: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Live K/V intervals must be non-empty and non-negative.")

    @property
    def tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LiveKVSelectionPlan:
    """Validated resident-K/V selection for one request.

    ``source_position_base`` is the absolute position of the request suffix.
    It is intentionally independent of ``selected_tokens``: sparse selections
    can contain holes while their already-RoPE-encoded K/V retain original
    coordinates.
    """

    source_tokens: int
    source_position_base: int
    intervals: tuple[LiveKVInterval, ...]

    def __post_init__(self) -> None:
        if self.source_tokens < 0:
            raise ValueError("source_tokens cannot be negative.")
        if self.source_position_base < self.source_tokens:
            raise ValueError("source_position_base cannot precede the live source.")
        prior_end = 0
        identities: set[str] = set()
        for interval in self.intervals:
            if interval.end > self.source_tokens:
                raise ValueError("A live K/V interval extends past the source cache.")
            if interval.start < prior_end:
                raise ValueError("Live K/V intervals must be ordered and disjoint.")
            prior_end = interval.end
            if interval.record_id:
                if interval.record_id in identities:
                    raise ValueError("A stable record may occur only once in a plan.")
                identities.add(interval.record_id)

    @classmethod
    def full(cls, source_tokens: int) -> "LiveKVSelectionPlan":
        source_tokens = int(source_tokens)
        intervals = (
            (LiveKVInterval(0, source_tokens, "full-history", "full-history"),)
            if source_tokens
            else ()
        )
        return cls(source_tokens, source_tokens, intervals)

    @classmethod
    def create(
        cls,
        source_tokens: int,
        intervals: Iterable[LiveKVInterval | tuple[int, int]],
        *,
        source_position_base: int | None = None,
    ) -> "LiveKVSelectionPlan":
        normalized = tuple(
            value
            if isinstance(value, LiveKVInterval)
            else LiveKVInterval(int(value[0]), int(value[1]))
            for value in intervals
        )
        return cls(
            int(source_tokens),
            int(source_tokens if source_position_base is None else source_position_base),
            normalized,
        )

    @property
    def selected_tokens(self) -> int:
        return sum(interval.tokens for interval in self.intervals)

    @property
    def full_retention(self) -> bool:
        return (
            self.source_tokens == self.source_position_base
            and self.intervals
            == (LiveKVInterval(0, self.source_tokens, "full-history", "full-history"),)
        ) or self.source_tokens == 0

    @property
    def has_holes(self) -> bool:
        return self.selected_tokens != self.source_tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "source_tokens": self.source_tokens,
            "source_position_base": self.source_position_base,
            "selected_tokens": self.selected_tokens,
            "full_retention": self.full_retention,
            "has_holes": self.has_holes,
            "intervals": [
                {
                    "start": row.start,
                    "end": row.end,
                    "record_id": row.record_id,
                    "causal_group_id": row.causal_group_id,
                }
                for row in self.intervals
            ],
        }

