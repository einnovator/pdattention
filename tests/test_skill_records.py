"""Contracts for declarative skill records and named views."""

from __future__ import annotations

import pytest

from data.declarative_skills import declarative_skill_catalog, skill_semantic_hard_queries
from pra_hf.context_records import RecordAtomicity, RecordType
from pra_hf.skill_records import (
    Skill,
    SkillFolderCache,
    SkillFolderError,
    SkillRecord,
    load_skill_directory,
    load_skill_folder,
)


def _skill(version: str = "v1") -> SkillRecord:
    return SkillRecord(
        name="github_issue_triage",
        description="Prioritize incoming repository issues using evidence and impact.",
        when_to_use="Use when an issue queue needs engineering triage.",
        instructions="Read every issue, verify reproducibility, assign impact, and state the next owner.",
        constraints=("Do not close an issue without evidence.",),
        ordered_steps=("Collect evidence.", "Assign impact.", "Recommend ownership."),
        examples=("Decision: investigate; Evidence: stack trace; Next action: reproduce.",),
        dependencies=("repository_issue_list",),
        references=("triage_policy_v2",),
        aliases=("issue intake review",),
        manual_tags=frozenset(("github", "triage")),
        namespace="paper6_5",
        tenant_id="paper6_5",
        version=version,
    )


def test_skill_selection_view_is_compact_and_full_view_retains_instructions() -> None:
    record = _skill().to_context_record()
    selection = record.materialize("selection")
    full = record.materialize("full")

    assert record.record_type == RecordType.SKILL
    assert record.policy.atomicity == RecordAtomicity.RECORD
    assert selection.fields == ("name", "description", "when_to_use")
    assert "instructions" not in selection.payload
    assert full.payload["instructions"].startswith("Read every issue")
    assert full.payload["ordered_steps"]
    assert full.payload["constraints"]


def test_skill_identity_is_versioned_and_shared_with_discovery_resource() -> None:
    first = _skill("v1")
    second = _skill("v2")

    assert first.uri != second.uri
    assert first.to_context_record().record_id == first.to_agent_resource().uri
    assert first.to_agent_resource().kind == "skill"
    assert first.to_agent_resource().metadata["declarative_only"]


def test_scripted_skill_payload_is_rejected() -> None:
    with pytest.raises(ValueError):
        SkillRecord(
            name="unsafe_script",
            description="Run a script.",
            when_to_use="Never in this paper.",
            instructions="Execute code.",
            metadata={"script": "rm -rf /"},
        )


def test_skill_benchmark_is_complete_confusable_and_name_blind() -> None:
    skills = declarative_skill_catalog()
    queries = skill_semantic_hard_queries()

    assert 20 <= len(skills) <= 50
    assert len(queries) == 2 * len(skills)
    assert len({skill.name for skill in skills}) == len(skills)
    assert {query.target_skill for query in queries} == {skill.name for skill in skills}
    assert all(query.target_skill.replace("_", " ") not in query.query.casefold() for query in queries)
    assert all(len(skill.instructions.split()) >= 70 for skill in skills)


def _write_skill(folder, frontmatter: str, body: str = "Follow the procedure exactly.") -> None:
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n" + (f"\n# Instructions\n\n{body}\n" if body else ""),
        encoding="utf-8",
    )


def test_openai_and_anthropic_skill_folders_normalize_without_scripts(tmp_path) -> None:
    openai = tmp_path / "openai-skill"
    _write_skill(
        openai,
        "name: openai-review\ndescription: Review changes when a patch needs analysis.\n"
        "metadata:\n  short-description: Patch review\nversion: v2",
    )
    (openai / "agents").mkdir()
    (openai / "agents" / "openai.yaml").write_text("interface: {}\n", encoding="utf-8")
    anthropic = tmp_path / "anthropic-skill"
    _write_skill(
        anthropic,
        "name: anthropic-review\ndescription: Review incidents when service health degrades.\n"
        "allowed-tools: Read Glob",
    )
    (anthropic / "scripts").mkdir()
    (anthropic / "scripts" / "ignored.py").write_text("raise RuntimeError()\n", encoding="utf-8")

    loaded = load_skill_directory(tmp_path)

    assert [skill.metadata["source_format"] for skill in loaded] == ["anthropic", "openai"]
    assert loaded[0].metadata["unsupported_assets"] == ("scripts/ignored.py",)
    assert all(isinstance(skill, Skill) for skill in loaded)


def test_skill_folder_validation_and_ambiguous_auto_detection(tmp_path) -> None:
    missing = tmp_path / "missing"
    _write_skill(missing, "name: no-description")
    with pytest.raises(SkillFolderError, match="description"):
        load_skill_folder(missing)

    empty = tmp_path / "empty"
    _write_skill(empty, "name: empty\ndescription: Empty body", body="")
    with pytest.raises(SkillFolderError, match="instruction body"):
        load_skill_folder(empty)

    ambiguous = tmp_path / "ambiguous"
    _write_skill(
        ambiguous,
        "name: ambiguous\ndescription: Has conflicting provider markers.\n"
        "metadata: {}\nallowed-tools: Read",
    )
    (ambiguous / "agents").mkdir()
    (ambiguous / "agents" / "openai.yaml").write_text("interface: {}\n", encoding="utf-8")
    with pytest.raises(SkillFolderError, match="both OpenAI- and Anthropic"):
        load_skill_folder(ambiguous)
    assert load_skill_folder(ambiguous, format="openai").metadata["source_format"] == "openai"


def test_skill_folder_cache_invalidates_only_changed_source(tmp_path) -> None:
    folder = tmp_path / "cached"
    frontmatter = "name: cached\ndescription: Cache this procedure."
    _write_skill(folder, frontmatter, "First instruction body.")
    cache = SkillFolderCache()

    first = cache.load(folder)
    repeated = cache.load(folder)
    (folder / "SKILL.md").write_text(
        f"---\n{frontmatter}\n---\n\nSecond instruction body.\n", encoding="utf-8"
    )
    changed = cache.load(folder)

    assert repeated is first
    assert changed is not first
    assert changed.version != first.version
    assert changed.metadata["source_hash"] != first.metadata["source_hash"]


def test_skill_folder_cache_does_not_cross_tenant_boundaries(tmp_path) -> None:
    folder = tmp_path / "tenant-scoped"
    _write_skill(folder, "name: scoped\ndescription: Tenant-scoped procedure.")
    cache = SkillFolderCache()

    first = cache.load(folder, tenant_id="tenant-a")
    second = cache.load(folder, tenant_id="tenant-b")

    assert first.tenant_id == "tenant-a"
    assert second.tenant_id == "tenant-b"
    assert first is not second
