"""Contract tests for the Paper 6.5 agent-resource discovery layer."""

from __future__ import annotations

from pra_hf.agent_resources import (
    AgentResource,
    DiscoveryDecision,
    DiscoveryHint,
    DiscoveryMode,
    DiscoveryPolicyHints,
    DiscoveryRequest,
    PersistentResourceIndex,
    ReliabilityCalibrator,
    ResourceDiscoveryEngine,
    SideEffectClass,
    resource_uri,
)


def _resource(
    name: str,
    description: str,
    *,
    namespace: str = "demo",
    aliases=(),
    version: str = "v1",
    tenant_id: str = "tenant-a",
    **kwargs,
) -> AgentResource:
    return AgentResource(
        uri=resource_uri("tool", namespace, name, version),
        kind="tool",
        namespace=namespace,
        name=name,
        version=version,
        description=description,
        aliases=tuple(aliases),
        tenant_id=tenant_id,
        **kwargs,
    )


def _engine(*resources: AgentResource, **kwargs) -> ResourceDiscoveryEngine:
    return ResourceDiscoveryEngine(
        PersistentResourceIndex(resources),
        select_threshold=0.65,
        ask_threshold=0.30,
        margin_threshold=0.05,
        **kwargs,
    )


def test_explicit_uri_resolves_deterministically_without_semantic_help():
    weather = _resource("weather", "retrieve atmospheric conditions")
    calendar = _resource("calendar", "list scheduled meetings")
    result = _engine(weather, calendar).discover(
        DiscoveryRequest(
            query=f"Use {calendar.uri}",
            hint=DiscoveryHint("explicit", strict=True),
            tenant_id="tenant-a",
        )
    )
    assert result.selected_uris[0] == calendar.uri
    assert result.executed_path == ("explicit",)
    assert result.decision == DiscoveryDecision.SELECT


def test_strict_hint_disables_fallback_but_non_strict_records_escalation():
    weather = _resource("weather", "retrieve atmospheric conditions")
    calendar = _resource("calendar", "list scheduled meetings")
    engine = _engine(weather, calendar)
    strict = engine.discover(
        DiscoveryRequest(
            query="atmospheric forecast",
            hint=DiscoveryHint("explicit", strict=True),
            tenant_id="tenant-a",
        )
    )
    fallback = engine.discover(
        DiscoveryRequest(
            query="atmospheric forecast",
            hint=DiscoveryHint("explicit", strict=False),
            tenant_id="tenant-a",
        )
    )
    assert strict.executed_path == ("explicit",)
    assert strict.decision == DiscoveryDecision.ABSTAIN
    assert fallback.executed_path[0] == "explicit"
    assert fallback.fallback_count > 0
    assert fallback.selected_uris[0] == weather.uri


def test_hint_precedence_is_request_reference_namespace_collection():
    tool = _resource(
        "weather",
        "retrieve atmospheric conditions",
        discovery_hint=DiscoveryHint("semantic", strict=True),
    )
    hints = DiscoveryPolicyHints(
        collection=DiscoveryHint("index"),
        namespaces={"demo": DiscoveryHint("token")},
        references={tool.uri: DiscoveryHint("hybrid")},
    )
    engine = _engine(tool, hints=hints)
    reference = engine.discover(
        DiscoveryRequest(
            query=tool.uri,
            explicit_reference_uris=(tool.uri,),
            tenant_id="tenant-a",
        )
    )
    request = engine.discover(
        DiscoveryRequest(
            query="weather",
            namespace="demo",
            hint=DiscoveryHint("explicit", strict=True),
            explicit_reference_uris=(tool.uri,),
            tenant_id="tenant-a",
        )
    )
    assert reference.requested_hint == "hybrid"
    assert request.requested_hint == "explicit"


def test_resource_level_hint_applies_without_external_reference_override():
    tool = _resource(
        "weather",
        "retrieve atmospheric conditions",
        discovery_hint=DiscoveryHint("semantic", strict=True),
    )
    result = _engine(tool).discover(
        DiscoveryRequest(
            query="atmospheric conditions",
            explicit_reference_uris=(tool.uri,),
            tenant_id="tenant-a",
        )
    )
    assert result.requested_hint == "semantic"
    assert result.executed_path == ("semantic",)


def test_tenant_allowed_uri_revocation_and_namespace_filters_are_enforced():
    allowed = _resource("weather", "weather data")
    other_tenant = _resource("secret", "weather data", tenant_id="tenant-b")
    revoked = _resource("legacy", "weather data", revoked=True)
    other_namespace = _resource("calendar", "weather data", namespace="office")
    result = _engine(allowed, other_tenant, revoked, other_namespace).discover(
        DiscoveryRequest(
            query="weather",
            namespace="demo",
            allowed_uris=frozenset((allowed.uri, revoked.uri, other_tenant.uri)),
            tenant_id="tenant-a",
        )
    )
    assert {row.uri for row in result.candidates} == {allowed.uri}


def test_semantic_only_and_nonindexable_metadata_change_available_channels():
    semantic_only = _resource(
        "atmosphere",
        "weather climate forecast",
        semantic_only=True,
    )
    dynamic = _resource(
        "calendar",
        "meeting agenda schedule",
        indexable=False,
    )
    rows = {row.uri: row for row in PersistentResourceIndex((semantic_only, dynamic)).score(
        DiscoveryRequest(query="weather forecast", tenant_id="tenant-a"),
        channels=("token", "index", "semantic"),
    )}
    assert rows[semantic_only.uri].token == 0.0
    assert rows[semantic_only.uri].index == 0.0
    assert rows[semantic_only.uri].semantic > 0.0
    assert rows[dynamic.uri].index == 0.0


def test_index_fingerprint_changes_with_version_content_and_configuration():
    first = _resource("weather", "weather data", version="v1")
    second = _resource("weather", "weather data changed", version="v2")
    a = PersistentResourceIndex((first,))
    b = PersistentResourceIndex((second,))
    c = PersistentResourceIndex((first,), fingerprint_metadata={"model": "qwen3"})
    assert a.fingerprint.digest != b.fingerprint.digest
    assert a.fingerprint.digest != c.fingerprint.digest
    assert a.estimated_bytes > 0


def test_calibrator_is_validation_fitted_and_bounded():
    calibrator = ReliabilityCalibrator.fit(
        ((0.1, False), (0.2, False), (0.8, True), (0.9, True)),
        bins=2,
    )
    assert 0.0 <= calibrator(0.15) <= 1.0
    assert calibrator(0.85) > calibrator(0.15)


def test_side_effecting_selection_requires_host_confirmation():
    delete = _resource(
        "delete_project",
        "delete a project permanently",
        side_effect_class=SideEffectClass.DESTRUCTIVE,
    )
    result = _engine(delete).discover(
        DiscoveryRequest(
            query="delete project",
            hint=DiscoveryHint("token", strict=True),
            tenant_id="tenant-a",
            side_effecting=True,
        )
    )
    assert result.selected_uris == (delete.uri,)
    assert result.decision == DiscoveryDecision.ASK
    assert not result.execution_authorized


def test_auto_prefers_exact_token_and_adaptive_logs_actual_path():
    weather = _resource("get_weather", "retrieve atmospheric conditions")
    calendar = _resource("calendar", "list scheduled meetings")
    engine = _engine(weather, calendar)
    auto = engine.discover(
        DiscoveryRequest(query="get weather", tenant_id="tenant-a")
    )
    adaptive = engine.discover(
        DiscoveryRequest(
            query="retrieve atmospheric conditions",
            hint=DiscoveryHint(DiscoveryMode.ADAPTIVE),
            tenant_id="tenant-a",
        )
    )
    assert auto.resolved_hint == "token"
    assert auto.executed_path[0] == "token"
    assert adaptive.executed_path
    assert adaptive.requested_hint == "adaptive"
    assert adaptive.selected_uris[0] == weather.uri


def test_precomputed_channel_scores_preserve_policy_result():
    weather = _resource("weather", "retrieve atmospheric conditions")
    calendar = _resource("calendar", "list scheduled meetings")
    engine = _engine(weather, calendar)
    request = DiscoveryRequest(
        query="weather",
        hint=DiscoveryHint("token", strict=True),
        tenant_id="tenant-a",
    )
    rows = engine.index.score(request, channels=("token", "index", "semantic"))
    direct = engine.discover(request)
    replay = engine.discover(request, scored_candidates=rows)
    assert replay.selected_uris == direct.selected_uris
    assert replay.executed_path == direct.executed_path
    assert replay.confidence == direct.confidence
