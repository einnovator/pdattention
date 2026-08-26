"""Public SDK and model-side lazy capability record contracts."""

from __future__ import annotations

import pytest

from pra_hf.capability_runtime import CapabilityEncodingPolicy, CapabilityEncodingState
from pra_hf.capability_sdk import AgentConfig, CapabilitySDK
from pra_hf.context_records import RecordType, RecordViewName
from pra_hf.skill_records import Skill, SkillFolderError


def inspect_release(release_id: str) -> dict[str, str]:
    """Inspect one release record without modifying it."""

    return {"release_id": release_id}


def _skill(name: str = "release_review", version: str = "v1") -> Skill:
    return Skill(
        name=name,
        description="Review release evidence and identify blockers.",
        when_to_use="Use before publishing a release.",
        instructions="Check tests, artifacts, migration, monitoring, and rollback evidence.",
        namespace="sdk-test",
        version=version,
    )


def _write_folder_skill(root, *, name="folder-review", version="v1", body="Apply every check."):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Review a folder-backed release procedure.\n"
        f"version: {version}\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return folder


def test_sdk_accepts_skill_objects_paths_and_stable_merge(tmp_path) -> None:
    _write_folder_skill(tmp_path)
    sdk = CapabilitySDK(
        AgentConfig(
            tools=(inspect_release,),
            skills=(_skill(),),
            skills_path=tmp_path,
            namespace="sdk-test",
            selection_view_token_budget=None,
        )
    )

    assert len(sdk.tools) == 1
    assert {skill.name for skill in sdk.skills} == {"release_review", "folder-review"}
    resources = sdk.resources()
    assert {resource.kind for resource in resources} == {"tool", "skill"}
    assert all(not resource.content for resource in resources)
    assert all(resource.metadata["discovery_view"] == "selection" for resource in resources)
    visible_text = "\n".join(resource.description for resource in resources)
    assert "Check tests, artifacts, migration" not in visible_text
    assert "Apply every check" not in visible_text
    parent, children = sdk.capability_slice(
        [record.record_id for record in sdk.records], slice_id="mixed-palette"
    )
    assert parent.record_type == RecordType.CAPABILITY_SLICE
    assert {record.record_type for record in children} == {
        RecordType.TOOL_RECORD,
        RecordType.SKILL_RECORD,
    }


def test_sdk_rejects_conflicting_skill_uri_from_list_and_path(tmp_path) -> None:
    _write_folder_skill(tmp_path, name="release_review", version="v1", body="Different body.")
    with pytest.raises(SkillFolderError, match="Conflicting skill definitions"):
        CapabilitySDK(
            AgentConfig(
                skills=(_skill(),),
                skills_path=tmp_path,
                namespace="sdk-test",
            )
        )


def test_lazy_runtime_encodes_selection_then_exact_full_record_and_reuses_cache() -> None:
    encoded = []
    sdk = CapabilitySDK(
        AgentConfig(
            tools=(inspect_release,),
            skills=(_skill(),),
            namespace="sdk-test",
            selection_view_token_budget=None,
        ),
        token_counter=lambda text: len(text.split()),
        encoder=lambda text: encoded.append(text) or text.encode("utf-8"),
        native_kv_bytes_per_token=128,
    )
    record_ids = [record.record_id for record in sdk.records]
    skill_id = next(record.record_id for record in sdk.records if record.record_type == RecordType.SKILL_RECORD)
    tool_id = next(record.record_id for record in sdk.records if record.record_type == RecordType.TOOL_RECORD)

    assert all(sdk.runtime.state(record_id) == CapabilityEncodingState.UNENCODED for record_id in record_ids)
    palette = sdk.activate_candidates(record_ids)
    assert palette.cold_encodes == 2
    assert all(sdk.runtime.cached_views(record_id) == {RecordViewName.SELECTION} for record_id in record_ids)

    cold = sdk.activate_selected(skill_id)
    assert cold.record_id == skill_id
    assert not cold.cache_hit
    assert cold.semantic_rediscovery_calls == 0
    assert sdk.runtime.active_view(tool_id) == RecordViewName.SELECTION
    assert RecordViewName.FULL not in sdk.runtime.cached_views(tool_id)

    sdk.runtime.deactivate()
    warm_palette = sdk.activate_candidates(record_ids)
    warm = sdk.activate_selected(skill_id)
    assert warm_palette.cache_hits == 2
    assert warm.cache_hit
    assert warm.encode_seconds == 0


def test_lazy_palette_enforces_candidate_and_selection_token_budgets() -> None:
    sdk = CapabilitySDK(
        AgentConfig(
            skills=(_skill("one"), _skill("two"), _skill("three")),
            max_candidates=2,
            selection_view_token_budget=80,
        ),
        token_counter=lambda text: len(text.split()),
    )

    activation = sdk.activate_candidates([record.record_id for record in sdk.records])

    assert len(activation.admitted_record_ids) <= 2
    assert activation.selection_tokens <= 80
    assert activation.dropped_record_ids
    assert all(
        RecordViewName.FULL not in sdk.runtime.cached_views(record_id)
        for record_id in sdk.runtime.record_ids
    )


def test_capability_allowlist_filters_before_visibility_and_full_activation() -> None:
    skills = (_skill("allowed"), _skill("blocked"))
    allowed_id = skills[0].to_context_record().record_id
    blocked_id = skills[1].to_context_record().record_id
    sdk = CapabilitySDK(
        AgentConfig(
            skills=skills,
            allowed_capability_uris=frozenset({allowed_id}),
            selection_view_token_budget=None,
        )
    )

    palette = sdk.activate_candidates((allowed_id, blocked_id))

    assert {resource.uri for resource in sdk.resources()} == {allowed_id}
    assert palette.admitted_record_ids == (allowed_id,)
    assert sdk.runtime.active_view(blocked_id) is None
    _parent, children = sdk.capability_slice((allowed_id, blocked_id))
    assert tuple(record.record_id for record in children) == (allowed_id,)
    with pytest.raises(PermissionError):
        sdk.activate_selected(blocked_id)


def test_eager_override_preencodes_selection_and_full_views() -> None:
    sdk = CapabilitySDK(
        AgentConfig(
            skills=(_skill(),),
            encoding=CapabilityEncodingPolicy(lazy_selection=False, lazy_full=False),
        )
    )
    record_id = sdk.records[0].record_id

    assert sdk.runtime.cached_views(record_id) == {
        RecordViewName.SELECTION,
        RecordViewName.FULL,
    }
    assert sdk.runtime.state(record_id) == CapabilityEncodingState.FULL_ENCODED


def test_explicit_full_initial_view_exposes_full_discovery_payload() -> None:
    skill = _skill()
    sdk = CapabilitySDK(
        AgentConfig(
            skills=(skill,),
            encoding=CapabilityEncodingPolicy(initial_view=RecordViewName.FULL),
        )
    )

    resource = sdk.resources(kinds=("skill",))[0]

    assert skill.instructions in resource.content
    assert resource.metadata.get("discovery_view") is None
