"""Public SDK facade for typed tools, skills, and lazy capability records."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence

from .capability_runtime import (
    CapabilityActivation,
    CapabilityEncodingPolicy,
    CapabilityPaletteActivation,
    LazyCapabilityRuntime,
)
from .context_records import (
    ContextRecord,
    RecordViewName,
    capability_slice_records,
    serialize_record,
    tool_definition_record,
)
from .skill_records import (
    Skill,
    SkillFolderCache,
    SkillFolderFormat,
    load_skill_directory,
    merge_skills,
)
from .tool_records import ToolRecord, tool_record_from_callable


@dataclass(frozen=True)
class AgentConfig:
    """Provider-neutral capability registration and lazy-disclosure settings."""

    tools: tuple[ToolRecord | Callable[..., object], ...] = ()
    skills: tuple[Skill, ...] = ()
    skills_path: str | Path | None = None
    skill_format: SkillFolderFormat | str = SkillFolderFormat.AUTO
    namespace: str = "default"
    tenant_id: str = "default"
    max_candidates: int = 24
    selection_view_token_budget: int | None = 2048
    allowed_capability_uris: frozenset[str] | None = None
    encoding: CapabilityEncodingPolicy = field(default_factory=CapabilityEncodingPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "skills", tuple(self.skills))
        object.__setattr__(self, "skill_format", SkillFolderFormat(self.skill_format))
        if self.allowed_capability_uris is not None:
            object.__setattr__(
                self, "allowed_capability_uris", frozenset(self.allowed_capability_uris)
            )
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        if self.selection_view_token_budget is not None and self.selection_view_token_budget <= 0:
            raise ValueError("selection_view_token_budget must be positive when provided.")


class CapabilitySDK:
    """Normalize capabilities and expose local model-side view activation."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        token_counter: Callable[[str], int] | None = None,
        encoder: Callable[[str], object] | None = None,
        native_kv_bytes_per_token: int = 0,
        skill_cache: SkillFolderCache | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        folder_skills = ()
        if self.config.skills_path is not None:
            folder_skills = load_skill_directory(
                self.config.skills_path,
                format=self.config.skill_format,
                namespace=self.config.namespace,
                tenant_id=self.config.tenant_id,
                cache=skill_cache,
            )
        self.skills = merge_skills(self.config.skills, folder_skills)
        tool_records = []
        for tool in self.config.tools:
            if isinstance(tool, ToolRecord):
                tool_records.append(tool)
            elif callable(tool):
                tool_records.append(
                    tool_record_from_callable(
                        tool,
                        namespace=self.config.namespace,
                        tenant_id=self.config.tenant_id,
                    )
                )
            else:
                raise TypeError("AgentConfig.tools accepts ToolRecord objects or Python callables.")
        self.tools = tuple(tool_records)
        self.records: tuple[ContextRecord, ...] = (
            *(tool_definition_record(tool.to_agent_resource()) for tool in self.tools),
            *(skill.to_context_record() for skill in self.skills),
        )
        self._by_id = {record.record_id: record for record in self.records}
        if len(self._by_id) != len(self.records):
            raise ValueError("Tool and skill capability URIs must be unique.")
        self.runtime = LazyCapabilityRuntime(
            self.records,
            policy=self.config.encoding,
            token_counter=token_counter,
            encoder=encoder,
            native_kv_bytes_per_token=native_kv_bytes_per_token,
        )

    def resources(self, *, kinds: Sequence[str] = ("tool", "skill")):
        """Return authorized discovery resources using the configured initial view."""

        selected = set(kinds)
        values = []
        if "tool" in selected:
            values.extend(tool.to_agent_resource() for tool in self.tools)
        if "skill" in selected:
            values.extend(skill.to_agent_resource() for skill in self.skills)
        allowed = self.config.allowed_capability_uris
        if self.config.encoding.initial_view == RecordViewName.SELECTION:
            values = [
                replace(
                    resource,
                    description=serialize_record(
                        self._by_id[resource.uri], view=RecordViewName.SELECTION
                    ),
                    content="",
                    metadata={**resource.metadata, "discovery_view": "selection"},
                )
                for resource in values
            ]
        return tuple(sorted(
            (resource for resource in values if allowed is None or resource.uri in allowed),
            key=lambda resource: resource.uri,
        ))

    def activate_candidates(self, record_ids: Sequence[str]) -> CapabilityPaletteActivation:
        """Authorize, then activate bounded compact views for a discovered palette."""

        allowed = self.config.allowed_capability_uris
        visible = tuple(
            record_id for record_id in record_ids
            if allowed is None or record_id in allowed
        )

        return self.runtime.activate_selection_palette(
            visible,
            max_candidates=self.config.max_candidates,
            selection_view_token_budget=self.config.selection_view_token_budget,
        )

    def activate_selected(self, record_id: str) -> CapabilityActivation:
        """Activate one exact backing full view without semantic rediscovery."""

        allowed = self.config.allowed_capability_uris
        if allowed is not None and record_id not in allowed:
            raise PermissionError(f"Capability is not authorized for disclosure: {record_id}")

        return self.runtime.activate_selected(record_id)

    def capability_slice(self, record_ids: Sequence[str], *, slice_id: str = "capabilities"):
        """Build the typed token-stream parent and children for one palette."""

        allowed = self.config.allowed_capability_uris
        visible = tuple(
            record_id for record_id in record_ids
            if allowed is None or record_id in allowed
        )
        return capability_slice_records(
            self.records,
            slice_id=slice_id,
            selected_record_ids=visible,
        )
