"""Typed declarative skill records for Paper 6.5 capability disclosure."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from .agent_resources import AgentResource, SideEffectClass, normalize_text, resource_uri, terms
from .context_records import ContextRecord, RecordType, RecordView, RecordViewName


class SkillFolderFormat(str, Enum):
    """Supported filesystem skill conventions."""

    AUTO = "auto"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    PORTABLE = "portable"


class SkillFolderError(ValueError):
    """Raised when a skill directory is malformed or format-ambiguous."""


@dataclass(frozen=True)
class Skill:
    """Versioned declarative skill with compact selection and full views."""

    name: str
    description: str
    when_to_use: str
    instructions: str
    namespace: str = "default"
    tenant_id: str = "default"
    version: str = "v1"
    aliases: tuple[str, ...] = ()
    manual_tags: frozenset[str] = frozenset()
    auto_tags: frozenset[str] = frozenset()
    keywords: frozenset[str] = frozenset()
    constraints: tuple[str, ...] = ()
    ordered_steps: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    references: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("name", "description", "when_to_use", "instructions", "namespace", "version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"SkillRecord {field_name} is required.")
        if self.metadata.get("script") or self.metadata.get("executable"):
            raise ValueError("Paper 6.5 skills are declarative text only.")
        object.__setattr__(self, "aliases", tuple(dict.fromkeys(self.aliases)))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "ordered_steps", tuple(self.ordered_steps))
        object.__setattr__(self, "examples", tuple(self.examples))
        object.__setattr__(self, "dependencies", tuple(dict.fromkeys(self.dependencies)))
        object.__setattr__(self, "references", tuple(dict.fromkeys(self.references)))
        object.__setattr__(self, "manual_tags", frozenset(normalize_text(value) for value in self.manual_tags if value))
        inferred = {
            token for token in terms(" ".join((self.name, self.description, self.when_to_use)))
            if len(token) > 2
        }
        object.__setattr__(self, "auto_tags", frozenset((*self.auto_tags, *inferred)))
        object.__setattr__(self, "keywords", frozenset((*self.keywords, *inferred)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def uri(self) -> str:
        return resource_uri("skill", self.namespace, self.name, self.version)

    @property
    def selection_payload(self) -> str:
        return "\n".join((
            self.name,
            self.description,
            f"Use when: {self.when_to_use}",
        ))

    @property
    def full_payload(self) -> dict[str, object]:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "when_to_use": self.when_to_use,
            "instructions": self.instructions,
            "constraints": list(self.constraints),
            "ordered_steps": list(self.ordered_steps),
            "examples": list(self.examples),
            "dependencies": list(self.dependencies),
            "references": list(self.references),
            "version": self.version,
            "namespace": self.namespace,
        }

    def to_context_record(
        self,
        *,
        parent_id: str | None = None,
        selection_provenance: Mapping[str, object] | None = None,
    ) -> ContextRecord:
        """Return an atomic context record with deterministic named views."""

        full = self.full_payload
        fingerprint = hashlib.sha256(
            json.dumps(full, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ContextRecord(
            record_id=self.uri,
            record_type=RecordType.SKILL,
            payload=full,
            parent_id=parent_id,
            selection_provenance=dict(selection_provenance or {}),
            version=self.version,
            source_fingerprint=fingerprint,
            views={
                RecordViewName.SELECTION: RecordView(
                    RecordViewName.SELECTION,
                    self.selection_payload,
                    ("name", "description", "when_to_use"),
                ),
                RecordViewName.FULL: RecordView(
                    RecordViewName.FULL,
                    full,
                    tuple(full),
                ),
            },
        )

    def to_agent_resource(self) -> AgentResource:
        """Expose the same typed skill through the shared discovery substrate."""

        semantic_terms = (
            *self.aliases,
            *sorted(self.manual_tags),
            *sorted(self.auto_tags),
            self.when_to_use,
        )
        return AgentResource(
            uri=self.uri,
            kind="skill",
            namespace=self.namespace,
            name=self.name,
            version=self.version,
            description=self.description,
            content=json.dumps(self.full_payload, sort_keys=True),
            aliases=self.aliases,
            side_effect_class=SideEffectClass.NONE,
            tenant_id=self.tenant_id,
            metadata={
                **dict(self.metadata),
                "tags": tuple(sorted(self.manual_tags)),
                "auto_tags": tuple(sorted(self.auto_tags)),
                "keywords": tuple(sorted(self.keywords)),
                "semantic_terms": semantic_terms,
                "record_type": RecordType.SKILL.value,
                "declarative_only": True,
            },
        )


SkillRecord = Skill
"""Backward-compatible name for the public :class:`Skill` object."""


_FRONTMATTER = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)\Z", re.DOTALL)
_ANTHROPIC_ONLY_KEYS = frozenset({"allowed-tools", "argument-hint", "model", "compatibility"})
_IGNORED_ASSET_DIRS = ("scripts", "assets")


def _read_skill_source(folder: Path) -> tuple[dict[str, object], str, bytes]:
    skill_file = folder / "SKILL.md"
    if not skill_file.is_file():
        raise SkillFolderError(f"Skill folder {folder} is missing SKILL.md.")
    raw = skill_file.read_bytes()
    text = raw.decode("utf-8")
    match = _FRONTMATTER.match(text)
    if match is None:
        raise SkillFolderError(f"Skill file {skill_file} requires YAML frontmatter.")
    parsed = yaml.safe_load(match.group(1)) or {}
    if not isinstance(parsed, Mapping):
        raise SkillFolderError(f"Skill frontmatter in {skill_file} must be a mapping.")
    body = match.group(2).strip()
    if not body:
        raise SkillFolderError(f"Skill file {skill_file} requires an instruction body.")
    return dict(parsed), body, raw


def _detect_skill_format(folder: Path, frontmatter: Mapping[str, object]) -> SkillFolderFormat:
    openai = (folder / "agents" / "openai.yaml").is_file()
    anthropic = bool(_ANTHROPIC_ONLY_KEYS & set(frontmatter))
    if openai and anthropic:
        raise SkillFolderError(
            f"Skill folder {folder} contains both OpenAI- and Anthropic-specific markers; "
            "set format explicitly."
        )
    if openai:
        return SkillFolderFormat.OPENAI
    if anthropic:
        return SkillFolderFormat.ANTHROPIC
    # The common Agent Skills subset is provider-portable and needs no guess.
    return SkillFolderFormat.PORTABLE


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item).strip())
    raise SkillFolderError("Expected a string or sequence of strings in skill frontmatter.")


def _reference_paths(folder: Path, declared: object) -> tuple[str, ...]:
    references = list(_string_tuple(declared))
    reference_root = folder / "references"
    if reference_root.is_dir():
        references.extend(
            path.relative_to(folder).as_posix()
            for path in sorted(reference_root.rglob("*"))
            if path.is_file()
        )
    return tuple(dict.fromkeys(references))


def load_skill_folder(
    folder: str | Path,
    *,
    format: SkillFolderFormat | str = SkillFolderFormat.AUTO,
    namespace: str = "skills",
    tenant_id: str = "default",
) -> Skill:
    """Normalize one recognized OpenAI/Anthropic-style skill directory.

    The loader reads declarative text and provenance only. Script and asset
    directories are recorded as unsupported metadata and are never executed.
    """

    source = Path(folder).expanduser().resolve()
    if not source.is_dir():
        raise SkillFolderError(f"Skill folder does not exist: {source}")
    frontmatter, instructions, raw = _read_skill_source(source)
    requested = SkillFolderFormat(format)
    if requested == SkillFolderFormat.AUTO:
        detected = resolved = _detect_skill_format(source, frontmatter)
    else:
        resolved = requested
        try:
            detected = _detect_skill_format(source, frontmatter)
        except SkillFolderError:
            detected = requested
    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    if not name:
        raise SkillFolderError(f"Skill folder {source} is missing frontmatter.name.")
    if not description:
        raise SkillFolderError(f"Skill folder {source} is missing frontmatter.description.")
    source_hash = hashlib.sha256(raw).hexdigest()
    unsupported = tuple(
        entry.relative_to(source).as_posix()
        for directory in _IGNORED_ASSET_DIRS
        if (source / directory).is_dir()
        for entry in sorted((source / directory).rglob("*"))
        if entry.is_file()
    )
    metadata = dict(frontmatter.get("metadata") or {})
    metadata.update({
        "source_format": resolved.value,
        "detected_source_format": detected.value,
        "source_path": source.as_posix(),
        "source_hash": source_hash,
        "unsupported_assets": unsupported,
        "declarative_only": True,
    })
    return Skill(
        name=name,
        description=description,
        when_to_use=str(
            frontmatter.get("when_to_use")
            or frontmatter.get("applicability")
            or frontmatter.get("trigger")
            or description
        ),
        instructions=instructions,
        namespace=str(frontmatter.get("namespace") or namespace),
        tenant_id=tenant_id,
        version=str(frontmatter.get("version") or f"sha256-{source_hash[:12]}"),
        aliases=_string_tuple(frontmatter.get("aliases")),
        manual_tags=frozenset(_string_tuple(frontmatter.get("tags"))),
        constraints=_string_tuple(frontmatter.get("constraints")),
        examples=_string_tuple(frontmatter.get("examples")),
        dependencies=_string_tuple(frontmatter.get("dependencies")),
        references=_reference_paths(source, frontmatter.get("references")),
        metadata=metadata,
    )


@dataclass
class SkillFolderCache:
    """Incremental cache keyed by source content, format, and normalization policy."""

    _entries: dict[tuple[str, str, str, str, str], Skill] = field(default_factory=dict)

    def load(
        self,
        folder: str | Path,
        *,
        format: SkillFolderFormat | str = SkillFolderFormat.AUTO,
        namespace: str = "skills",
        tenant_id: str = "default",
    ) -> Skill:
        source = Path(folder).expanduser().resolve()
        _frontmatter, _body, raw = _read_skill_source(source)
        source_hash = hashlib.sha256(raw).hexdigest()
        key = (
            source.as_posix(), SkillFolderFormat(format).value, namespace, tenant_id, source_hash
        )
        if key not in self._entries:
            self._entries[key] = load_skill_folder(
                source, format=format, namespace=namespace, tenant_id=tenant_id
            )
        return self._entries[key]


def load_skill_directory(
    parent: str | Path,
    *,
    format: SkillFolderFormat | str = SkillFolderFormat.AUTO,
    namespace: str = "skills",
    tenant_id: str = "default",
    cache: SkillFolderCache | None = None,
) -> tuple[Skill, ...]:
    """Load immediate child folders that contain a recognized ``SKILL.md``."""

    root = Path(parent).expanduser().resolve()
    if not root.is_dir():
        raise SkillFolderError(f"Skills parent directory does not exist: {root}")
    loader = cache or SkillFolderCache()
    records = [
        loader.load(child, format=format, namespace=namespace, tenant_id=tenant_id)
        for child in sorted(root.iterdir(), key=lambda path: path.name.casefold())
        if child.is_dir() and (child / "SKILL.md").is_file()
    ]
    if not records:
        raise SkillFolderError(f"No child skill directories found under {root}.")
    return tuple(records)


def merge_skills(*groups: Iterable[Skill]) -> tuple[Skill, ...]:
    """Merge object and folder skills by stable URI, rejecting collisions."""

    by_uri: dict[str, Skill] = {}
    for skill in (item for group in groups for item in group):
        previous = by_uri.get(skill.uri)
        if previous is not None and previous.full_payload != skill.full_payload:
            raise SkillFolderError(f"Conflicting skill definitions share URI {skill.uri}.")
        by_uri[skill.uri] = skill
    return tuple(by_uri[uri] for uri in sorted(by_uri))


def skill_records_to_resources(records: Sequence[Skill]) -> tuple[AgentResource, ...]:
    """Convert a stable skill registry without creating a separate index type."""

    return tuple(record.to_agent_resource() for record in records)
