"""Contracts for declarative skill records and named views."""

from __future__ import annotations

import pytest

from data.declarative_skills import declarative_skill_catalog, skill_semantic_hard_queries
from pra_hf.context_records import RecordAtomicity, RecordType
from pra_hf.skill_records import SkillRecord


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
