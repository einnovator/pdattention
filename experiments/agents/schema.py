"""Validated manifests and normalized results for coding-agent experiments."""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields so stale experiment configuration fails loudly."""

    model_config = ConfigDict(extra="forbid")


class AgentType(str, Enum):
    COMMERCIAL = "commercial"
    OPEN_SOURCE = "open_source"


class Connectivity(str, Enum):
    DIRECT = "DIRECT"
    GATEWAY = "GATEWAY"
    COMMERCIAL_NATIVE = "COMMERCIAL_NATIVE"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Qualification(str, Enum):
    DIRECT = "DIRECT"
    GATEWAY = "GATEWAY"
    COMMERCIAL_NATIVE = "COMMERCIAL_NATIVE"
    TESTED = "TESTED"
    QUALIFICATION_PENDING = "QUALIFICATION_PENDING"
    BLOCKED = "BLOCKED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PRAMode(str, Enum):
    NONE = "none"
    SELECTED_CONTEXT = "selected-context"
    NATIVE_MEMORY = "native-memory"


class PRAProfile(str, Enum):
    NONE = "none"
    QUALITY = "quality"
    BALANCED = "balanced"
    ECONOMY = "economy"


class CacheState(str, Enum):
    COLD = "cold"
    WARM = "warm"
    HOT = "hot"
    DECLARED_NATIVE = "agent-native"


class AgentCatalogEntry(StrictModel):
    name: str
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: str
    version_audited_at: date
    type: AgentType
    source: str
    license: str
    executable: str
    version_command: tuple[str, ...]
    automation: bool
    custom_endpoint: bool
    protocols: tuple[str, ...]
    connectivity: tuple[Connectivity, ...]
    gateway_support: str
    direct_engine_support: str
    tool_support: str
    token_accounting: str
    cost_accounting: str
    session_persistence: str
    context_compaction: str
    limitations: tuple[str, ...] = ()


class AgentCatalog(StrictModel):
    schema_version: Literal[1] = 1
    audited_at: date
    agents: tuple[AgentCatalogEntry, ...]

    @model_validator(mode="after")
    def unique_agents(self) -> "AgentCatalog":
        slugs = [agent.slug for agent in self.agents]
        if len(slugs) != len(set(slugs)):
            raise ValueError("agent slugs must be unique")
        return self


class AgentCommandSpec(StrictModel):
    slug: str
    executable: str
    version_args: tuple[str, ...] = ("--version",)
    run_args: tuple[str, ...]
    prompt_arg: str | None = None
    model_arg: str | None = None
    output_format: str
    verified: bool
    external_sandbox_required: bool = True
    notes: tuple[str, ...] = ()


class AgentCommandManifest(StrictModel):
    schema_version: Literal[1] = 1
    audited_at: date
    commands: tuple[AgentCommandSpec, ...]

    @model_validator(mode="after")
    def unique_commands(self) -> "AgentCommandManifest":
        slugs = [row.slug for row in self.commands]
        if len(slugs) != len(set(slugs)):
            raise ValueError("agent command slugs must be unique")
        return self


class HardwareTarget(StrictModel):
    host: str
    role: str
    os: str
    architecture: str
    cpu: str
    ram_bytes: int | None = Field(default=None, ge=0)
    accelerator: str | None = None
    accelerator_memory_bytes: int | None = Field(default=None, ge=0)
    status: str = "available"
    limitations: tuple[str, ...] = ()


class HardwareManifest(StrictModel):
    schema_version: Literal[1] = 1
    captured_at: datetime
    hosts: tuple[HardwareTarget, ...]


class ProtocolStatus(StrictModel):
    protocol: str
    status: Literal["TESTED", "QUALIFICATION_PENDING", "BLOCKED", "NOT_APPLICABLE"]
    streaming: str
    tool_calls: str
    cancellation: str
    usage: str
    session_identity: str
    evidence: str


class ProtocolManifest(StrictModel):
    schema_version: Literal[1] = 1
    audited_at: date
    protocols: tuple[ProtocolStatus, ...]


class AgentEngineMatrix(StrictModel):
    schema_version: Literal[1] = 1
    audited_at: date
    legend: tuple[Qualification, ...]
    engines: tuple[str, ...]
    agents: Mapping[str, Mapping[str, Qualification]]

    @model_validator(mode="after")
    def complete_rectangular_matrix(self) -> "AgentEngineMatrix":
        expected = set(self.engines)
        for agent, values in self.agents.items():
            if set(values) != expected:
                missing = sorted(expected - set(values))
                extra = sorted(set(values) - expected)
                raise ValueError(f"{agent} engine cells differ: missing={missing}, extra={extra}")
        return self


class BenchmarkManifest(StrictModel):
    schema_version: Literal[1] = 1
    name: str
    benchmark: Literal["fixture", "terminal-bench", "swe-bench"]
    dataset: str
    revision: str
    cohort: Literal["smoke", "pilot", "main"]
    task_ids: tuple[str, ...]
    repeats: int = Field(default=1, ge=1, le=20)
    timeout_seconds: int = Field(default=1800, ge=1)
    environment: str = "container"
    official_metric: str = "task_success"
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def frozen_nonempty_tasks(self) -> "BenchmarkManifest":
        if not self.task_ids:
            raise ValueError("a frozen manifest requires task_ids")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task_ids must be unique")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkManifest":
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


class RunIdentity(StrictModel):
    run_id: str
    agent: str
    agent_version: str
    benchmark: str
    benchmark_revision: str
    task_id: str
    repeat: int = Field(ge=0)
    engine: str
    engine_version: str | None = None
    host: str
    hardware: Mapping[str, Any] = Field(default_factory=dict)
    model: str
    model_revision: str | None = None
    quantization: str | None = None
    pra_version: str
    pra_bundle: str | None = None
    pra_bundle_revision: str | None = None
    pra_mode: PRAMode
    pra_profile: PRAProfile
    connection: Literal["gateway", "direct", "commercial-native", "fixture"]
    engine_pra_enabled: bool | None = None
    gateway_pra_enabled: bool | None = None
    gateway_mode: str | None = None
    protocol: str
    cache_state: CacheState
    session_state: Literal["reset", "continued"] = "reset"
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def mode_and_profile_agree(self) -> "RunIdentity":
        if self.pra_mode == PRAMode.NONE and self.pra_profile != PRAProfile.NONE:
            raise ValueError("No-PRA runs must use the none profile")
        if self.pra_mode != PRAMode.NONE and self.pra_profile == PRAProfile.NONE:
            raise ValueError("PRA runs require quality, balanced, or economy")
        if self.gateway_pra_enabled and self.connection != "gateway":
            raise ValueError("gateway PRA can only be enabled on a gateway connection")
        if self.gateway_mode is not None and self.connection != "gateway":
            raise ValueError("gateway_mode is only valid on a gateway connection")
        if self.pra_mode == PRAMode.NATIVE_MEMORY and self.engine_pra_enabled is False:
            raise ValueError("native-memory runs require a PRA-enabled engine")
        return self


class OutcomeMetrics(StrictModel):
    success: bool
    official_score: float | None = None
    tests_passed: int | None = Field(default=None, ge=0)
    tests_total: int | None = Field(default=None, ge=0)
    patch_correct: bool | None = None
    failure_kind: str | None = None


class AgentBehaviorMetrics(StrictModel):
    turns: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    shell_calls: int = Field(default=0, ge=0)
    file_reads: int = Field(default=0, ge=0)
    file_writes: int = Field(default=0, ge=0)
    tests: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    context_compactions: int = Field(default=0, ge=0)


class TokenMetrics(StrictModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    logical_source_tokens: int = Field(default=0, ge=0)
    selected_tokens: int = Field(default=0, ge=0)
    newly_materialized_tokens: int = Field(default=0, ge=0)
    visible_reuse_tokens: int = Field(default=0, ge=0)
    prefix_cache_reused_tokens: int = Field(default=0, ge=0)
    cumulative_context_sent: int = Field(default=0, ge=0)
    max_context_tokens: int = Field(default=0, ge=0)


class TimingMetrics(StrictModel):
    task_wall_ms: float = Field(ge=0)
    inference_ms: float = Field(default=0, ge=0)
    queue_ms: float = Field(default=0, ge=0)
    completion_ms: float = Field(default=0, ge=0)
    tool_ms: float = Field(default=0, ge=0)
    ttft_samples_ms: tuple[float, ...] = ()
    itl_samples_ms: tuple[float, ...] = ()
    turn_samples_ms: tuple[float, ...] = ()


class ResourceMetrics(StrictModel):
    native_resources: int = Field(default=0, ge=0)
    native_active_bytes: int = Field(default=0, ge=0)
    native_retained_bytes: int = Field(default=0, ge=0)
    hot_bytes: int = Field(default=0, ge=0)
    warm_bytes: int = Field(default=0, ge=0)
    promotions: int = Field(default=0, ge=0)
    reloads: int = Field(default=0, ge=0)
    disk_read_bytes: int = Field(default=0, ge=0)
    network_bytes: int = Field(default=0, ge=0)
    peak_ram_bytes: int | None = Field(default=None, ge=0)
    peak_accelerator_bytes: int | None = Field(default=None, ge=0)


class CostMetrics(StrictModel):
    currency: str = "USD"
    provider_input: float = Field(default=0, ge=0)
    provider_cached_input: float = Field(default=0, ge=0)
    provider_output: float = Field(default=0, ge=0)
    tool: float = Field(default=0, ge=0)
    total: float | None = Field(default=None, ge=0)
    estimated: bool = False
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def calculate_total(self) -> "CostMetrics":
        if self.total is None:
            self.total = self.provider_input + self.provider_cached_input + self.provider_output + self.tool
        return self


class ToolResultLifecycle(StrictModel):
    record_id: str
    record_type: str
    retained: bool
    dropped: bool
    reselected: int = Field(default=0, ge=0)
    rematerialized: int = Field(default=0, ge=0)
    bytes: int = Field(default=0, ge=0)


class CodingAgentRun(StrictModel):
    schema_version: Literal[1] = 1
    identity: RunIdentity
    outcome: OutcomeMetrics
    behavior: AgentBehaviorMetrics
    tokens: TokenMetrics
    timings: TimingMetrics
    resources: ResourceMetrics = Field(default_factory=ResourceMetrics)
    cost: CostMetrics = Field(default_factory=CostMetrics)
    tool_results: tuple[ToolResultLifecycle, ...] = ()
    trace_id: str | None = None
    artifacts: Mapping[str, str] = Field(default_factory=dict)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_native_semantics(self) -> "CodingAgentRun":
        if self.identity.pra_mode != PRAMode.NATIVE_MEMORY:
            if self.resources.native_resources or self.resources.native_active_bytes:
                raise ValueError("non-native runs cannot claim native resources or bytes")
        return self

    def json_line(self) -> str:
        return self.model_dump_json(exclude_none=True)
