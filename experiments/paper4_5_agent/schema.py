"""Strict campaign contracts for reproduction-first agent evaluation."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .benchmark import load_benchmark_card


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CampaignMode(str, Enum):
    """Execution modes ordered from ordinary inference to native PRA."""

    NATIVE = "native"
    TRUNCATION = "truncation"
    GATEWAY_PASSTHROUGH = "gateway_passthrough"
    GATEWAY_PRA = "gateway_pra"
    NATIVE_PRA = "native_pra"


class ReproductionStatus(str, Enum):
    PLANNED = "PLANNED"
    BASELINE_ATTEMPTED = "BASELINE_ATTEMPTED"
    BASELINE_REPRODUCED = "BASELINE_REPRODUCED"
    BASELINE_FAILED = "BASELINE_FAILED"
    BLOCKED = "BLOCKED"


class PublishedBaseline(StrictModel):
    """Externally reported result and all identity fields needed to reproduce it."""

    baseline_id: str
    source_url: str
    source_revision: str
    benchmark_revision: str = "NOT_REPORTED_BY_SOURCE"
    task_ids_sha256: str | None = None
    published_score: float = Field(ge=0, le=1)
    published_resolved: int | None = Field(default=None, ge=0)
    published_total: int = Field(ge=1)
    benchmark: str
    dataset: str
    split: str
    task_ids: tuple[str, ...] = ()
    task_ids_path: str | None = None
    harness: str
    harness_version: str
    model: str
    model_revision: str
    tokenizer_revision: str
    engine: str
    engine_version: str
    dtype: str
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    scaffold: str
    context_limit: int = Field(ge=1)
    max_steps: int = Field(ge=1)
    max_steps_absolute: int = Field(ge=1)
    temperature: float = Field(ge=0)
    function_calling: bool
    prefix_caching: bool
    tool_configuration: str
    grading: str
    notes: tuple[str, ...] = ()


class ResultContract(StrictModel):
    """Location and tolerance for a normalized official-grader result."""

    path: str
    absolute_tolerance: float = Field(default=0.05, ge=0, le=1)
    require_exact_cohort: bool = True


class CampaignCell(StrictModel):
    """One executable cell; non-baselines must name an admitted baseline cell."""

    cell_id: str
    stage: Literal["A", "B", "C", "D"]
    mode: CampaignMode
    baseline_id: str
    baseline_cell: str | None = None
    command: tuple[str, ...]
    working_directory: str = "."
    environment: Mapping[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=3600, ge=1)
    minimum_baseline_score: float = Field(default=0.0, ge=0, le=1)
    result: ResultContract
    enabled: bool = True
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def baseline_dependency_matches_mode(self) -> "CampaignCell":
        if self.mode == CampaignMode.NATIVE and self.baseline_cell is not None:
            raise ValueError("native no-PRA cells cannot depend on another baseline")
        if self.mode != CampaignMode.NATIVE and not self.baseline_cell:
            raise ValueError("gateway and PRA cells require baseline_cell")
        return self


class CampaignConfig(StrictModel):
    schema_version: Literal[1] = 1
    campaign_id: str
    output_directory: str
    baselines: tuple[PublishedBaseline, ...]
    cells: tuple[CampaignCell, ...]

    @model_validator(mode="after")
    def identities_and_dependencies_exist(self) -> "CampaignConfig":
        baseline_ids = [row.baseline_id for row in self.baselines]
        cell_ids = [row.cell_id for row in self.cells]
        if len(baseline_ids) != len(set(baseline_ids)):
            raise ValueError("baseline_id values must be unique")
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("cell_id values must be unique")
        known_baselines = set(baseline_ids)
        known_cells = {row.cell_id: row for row in self.cells}
        for cell in self.cells:
            if cell.baseline_id not in known_baselines:
                raise ValueError(f"{cell.cell_id} references unknown baseline {cell.baseline_id}")
            if cell.baseline_cell:
                dependency = known_cells.get(cell.baseline_cell)
                if dependency is None:
                    raise ValueError(f"{cell.cell_id} references unknown cell {cell.baseline_cell}")
                if dependency.mode != CampaignMode.NATIVE:
                    raise ValueError(f"{cell.cell_id} baseline_cell must be a native no-PRA cell")
                if dependency.baseline_id != cell.baseline_id:
                    raise ValueError(f"{cell.cell_id} and its baseline use different published targets")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "CampaignConfig":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for baseline in payload.get("baselines", []):
            task_ids_path = baseline.get("task_ids_path")
            if task_ids_path and not baseline.get("task_ids"):
                root = _find_repository_root(config_path.resolve())
                card = load_benchmark_card(root / task_ids_path)
                baseline["task_ids"] = card["instance_ids"]
        return cls.model_validate(payload)

    def baseline(self, baseline_id: str) -> PublishedBaseline:
        return next(row for row in self.baselines if row.baseline_id == baseline_id)


def _find_repository_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "experiments").is_dir():
            return parent
    raise ValueError(f"cannot resolve repository-relative campaign artifacts from {path}")
