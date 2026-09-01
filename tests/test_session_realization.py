from __future__ import annotations

from pra_hf.deployment import PRAWireResource
from pra_hf.execution_modes import ExecutionModeResolver
from pra_hf.product_qualification import EngineProductRegistry
from pra_hf.session_realization import (
    MaterializationIdentity,
    PrefixReuseObservation,
    PrefixReuseStatus,
    RealizationDecision,
    RealizationPlanner,
    VisibilityState,
    VisibleMaterialization,
    VisibleMaterializationLedger,
)


def _resource(
    *, version: str = "v1", interval=(0, 3), profile_text: str = "alpha beta gamma"
) -> PRAWireResource:
    return PRAWireResource(
        "facts",
        "pra://tenant-a/facts",
        text=profile_text,
        version=version,
        authorization_scope="tenant-a",
        metadata={
            "tenant_id": "tenant-a",
            "version": version,
            "selected_interval": interval,
        },
    )


def _visible(resource: PRAWireResource, profile: str = "before_current_user/BALANCED"):
    message = {
        "role": "user",
        "content": f"[PRA resource {resource.uri}]\n{resource.text}\n\nquestion",
    }
    entries = VisibleMaterializationLedger.add_materializations(
        (),
        (resource,),
        (message,),
        tenant_id="tenant-a",
        session_id="session-a",
        rendering_profile=profile,
        turn=1,
    )
    return entries, message


def test_materialization_identity_requires_version_span_profile_and_content() -> None:
    base = MaterializationIdentity.from_resource(_resource(), "profile-a")

    assert MaterializationIdentity.from_resource(_resource(), "profile-a") == base
    assert MaterializationIdentity.from_resource(_resource(version="v2"), "profile-a") != base
    assert MaterializationIdentity.from_resource(_resource(interval=(1, 3)), "profile-a") != base
    assert MaterializationIdentity.from_resource(_resource(), "profile-b") != base
    assert MaterializationIdentity.from_resource(
        _resource(profile_text="different content"), "profile-a"
    ) != base


def test_ledger_reconciles_against_actual_serialized_context() -> None:
    resource = _resource()
    entries, message = _visible(resource)

    retained = VisibleMaterializationLedger.reconcile(entries, (message,), turn=2)
    dropped = VisibleMaterializationLedger.reconcile(entries, (), turn=2)

    assert retained[0].state == VisibilityState.VISIBLE
    assert retained[0].last_visible_turn == 2
    assert dropped[0].state == VisibilityState.DROPPED_FROM_ACTIVE_CONTEXT


def test_planner_keeps_logical_visibility_and_native_residency_disjoint() -> None:
    resource = _resource()
    entries, _ = _visible(resource)
    planner = RealizationPlanner()

    visible = planner.plan(
        (resource,),
        entries,
        requested_mode="selected-context",
        resolved_mode="selected-context",
        rendering_profile="before_current_user/BALANCED",
        tenant_id="tenant-a",
    )
    hot = _resource()
    hot = PRAWireResource(**{
        **hot.to_dict(),
        "metadata": {**hot.metadata, "native_residency": "HOT", "native_bytes": 128},
    })
    native = planner.plan(
        (hot,),
        (),
        requested_mode="native-memory",
        resolved_mode="native-memory",
        rendering_profile="before_current_user/BALANCED",
        tenant_id="tenant-a",
        native_capable=True,
    )

    assert visible.items[0].decision == RealizationDecision.ALREADY_VISIBLE
    assert visible.diagnostics["visible_reuse_tokens"] == 3
    assert native.items[0].decision == RealizationDecision.NATIVE_AVAILABLE
    assert native.diagnostics["native_reuse_tokens"] == 3
    assert native.diagnostics["native_attach_bytes"] == 128


def test_prefix_observation_never_guesses_a_confirmed_hit() -> None:
    confirmed = PrefixReuseObservation.from_result(
        {"cached_tokens": 64, "worker_identity": "worker-1"},
        prefix_cache_mode="automatic_prefix_cache",
        prior_engine_session=True,
        prefix_digest="digest",
        prefix_token_count=96,
        model_fingerprint="model-a",
    )
    likely = PrefixReuseObservation.from_result(
        {},
        prefix_cache_mode="automatic_prefix_cache",
        prior_engine_session=True,
        prefix_digest="digest",
        prefix_token_count=96,
        model_fingerprint="model-a",
    )
    absent = PrefixReuseObservation.from_result(
        {},
        prefix_cache_mode="stateless",
        prior_engine_session=False,
        prefix_digest=None,
        prefix_token_count=None,
        model_fingerprint="model-a",
    )

    assert confirmed.status == PrefixReuseStatus.CONFIRMED
    assert likely.status == PrefixReuseStatus.LIKELY
    assert absent.status == PrefixReuseStatus.ABSENT


def test_prefix_observation_invalidates_unproven_physical_continuity() -> None:
    common = {
        "prefix_cache_mode": "automatic_prefix_cache",
        "prior_engine_session": True,
        "prefix_digest": "digest",
        "prefix_token_count": 96,
        "model_fingerprint": "model-b",
        "prior_worker_identity": "worker-1",
        "prior_model_fingerprint": "model-a",
    }
    worker_switch = PrefixReuseObservation.from_result(
        {"worker_identity": "worker-2", "model_fingerprint": "model-a"},
        **common,
    )
    model_switch = PrefixReuseObservation.from_result(
        {"worker_identity": "worker-1", "model_fingerprint": "model-b"},
        **common,
    )
    restart = PrefixReuseObservation.from_result(
        {}, **common, engine_restarted=True
    )

    assert worker_switch.status == PrefixReuseStatus.UNKNOWN
    assert worker_switch.continuity_reason == "worker_changed"
    assert model_switch.status == PrefixReuseStatus.UNKNOWN
    assert model_switch.continuity_reason == "model_fingerprint_changed"
    assert restart.status == PrefixReuseStatus.UNKNOWN
    assert restart.continuity_reason == "engine_restart"


def test_closed_task_resource_does_not_satisfy_visible_reuse() -> None:
    active = _resource()
    entries, _ = _visible(active)
    closed = PRAWireResource(**{
        **active.to_dict(),
        "metadata": {**active.metadata, "task_status": "closed"},
    })
    plan = RealizationPlanner().plan(
        (closed,),
        entries,
        requested_mode="selected-context",
        resolved_mode="selected-context",
        rendering_profile="before_current_user/BALANCED",
        tenant_id="tenant-a",
    )

    assert plan.items[0].decision == RealizationDecision.MUST_MATERIALIZE


def test_mode_resolver_is_conservative_for_every_registered_engine() -> None:
    registry = EngineProductRegistry.default()
    resolver = ExecutionModeResolver()

    for engine in registry.engines:
        resolution = resolver.resolve("auto", engine)
        assert resolution.resolved_mode.value in {
            "selected-context", "native-memory", "native-serving"
        }
        if resolution.resolved_mode.value != "selected-context":
            candidate = next(
                row for row in resolution.candidates
                if row.mode == resolution.resolved_mode
            )
            assert candidate.qualifies_for_auto
