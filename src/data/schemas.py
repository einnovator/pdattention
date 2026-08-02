from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReferenceSample:
    """Reference document metadata exposed to datasets and collators."""

    id: int
    uri: str
    summary: str | None = None
    anchor: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionSample:
    """A non-tensor QA training example with reference supervision."""

    id: str
    question: str
    answer: str
    references: list[ReferenceSample]
    target_reference_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata identifying the dataset source and stage."""

    dataset_name: str
    stage: str
    version: str = "0.1.0"
