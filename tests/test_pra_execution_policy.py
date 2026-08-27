from __future__ import annotations

import pytest

from pra_torch.execution import (
    PRAExecutionCapabilities,
    PRAExecutionPolicy,
    PRALayerPayloadResolver,
    PRAMaterializationManager,
    PRAMaterializationScope,
    PRARequestExecutionContext,
    PRAResidencyPolicy,
    PRARoutingLayerPolicy,
    PRASelectedIdentity,
    PRASelectionController,
    PRASelectionLayerScope,
    PRASelectionPlan,
    PRASelectionStage,
    resolve_execution_policy,
    resolve_routing_layer,
    analytical_routing_operations,
)


CAPABILITIES = PRAExecutionCapabilities(
    engine="test",
    phase_selection=True,
    token_selection=True,
    per_layer_selection=True,
    phase_materialization=True,
    layer_materialization=True,
    token_materialization=True,
    layer_lifetime_residency=True,
)


def _identity(name: str) -> PRASelectedIdentity:
    return PRASelectedIdentity("mem://facts", name, 0, 4)


def _rows(name: str = "c0"):
    return ((_identity(name),),)


def _context(policy: PRAExecutionPolicy) -> PRARequestExecutionContext:
    resolved = resolve_execution_policy(
        global_policy=policy,
        capabilities=CAPABILITIES,
        active_layers=(2, 4, 6),
        configured_routing_layer=4,
    )
    return PRARequestExecutionContext(resolved, request_id="request-1")


def test_policy_precedence_is_field_level_and_inspectable():
    resolved = resolve_execution_policy(
        global_policy={"routing_layer_policy": "first_pra_layer"},
        model_policy={"selection_layer_scope": "per_layer"},
        request_policy={
            "selection_stage": "token",
            "materialization_scope": "token",
        },
        capabilities=CAPABILITIES,
        active_layers=(2, 4),
        configured_routing_layer=2,
    )

    assert resolved.policy.selection_stage == PRASelectionStage.TOKEN
    assert resolved.policy.selection_layer_scope == PRASelectionLayerScope.PER_LAYER
    assert resolved.field_sources["routing_layer_policy"] == "global_default"
    assert resolved.field_sources["selection_layer_scope"] == "model_default"
    assert resolved.field_sources["selection_stage"] == "request_override"
    assert resolved.downgrades == ()


def test_unsupported_modes_fail_instead_of_silently_downgrading():
    with pytest.raises(ValueError, match="does not support token selection"):
        resolve_execution_policy(
            request_policy={
                "selection_stage": "token",
                "materialization_scope": "token",
            },
            capabilities=PRAExecutionCapabilities(engine="request-only"),
        )
    with pytest.raises(ValueError, match="TOKEN selection cannot"):
        resolve_execution_policy(
            request_policy={"selection_stage": "token"},
            capabilities=CAPABILITIES,
        )


def test_named_routing_layer_policies_are_deterministic():
    layers = (3, 7, 11, 15)
    assert resolve_routing_layer(
        PRAExecutionPolicy(routing_layer_policy=PRARoutingLayerPolicy.FIRST_PRA_LAYER),
        layers,
        15,
    ) == 3
    assert resolve_routing_layer(
        PRAExecutionPolicy(routing_layer_policy=PRARoutingLayerPolicy.MIDDLE_PRA_LAYER),
        layers,
        15,
    ) == 7
    assert resolve_routing_layer(
        PRAExecutionPolicy(routing_layer_policy=PRARoutingLayerPolicy.LAST_PRA_LAYER),
        layers,
        3,
    ) == 15


def test_shared_plan_resolves_layer_native_payload_without_embedding_tensors():
    plan = PRASelectionPlan(
        PRASelectionStage.REQUEST,
        PRASelectionLayerScope.SHARED,
        source_layer=4,
        epoch_id=1,
        shared_rows=_rows(),
    )
    resolver = PRALayerPayloadResolver(
        lambda rows, layer_id: (layer_id, tuple(item.chunk_id for row in rows for item in row))
    )

    assert plan.rows_for(2) == plan.rows_for(6)
    assert resolver.resolve(plan, 6) == (6, ("c0",))
    assert not hasattr(plan.shared_rows[0][0], "key") or isinstance(
        plan.shared_rows[0][0].key, tuple
    )


def test_request_selection_reuses_one_epoch_and_token_selection_tracks_churn():
    controller = PRASelectionController()
    request_context = _context(PRAExecutionPolicy())
    calls = []

    def route(layer_id):
        calls.append(layer_id)
        return _rows()

    first = controller.selection_for(
        context=request_context,
        layer_id=4,
        phase="request",
        token_index=0,
        route=route,
    )
    second = controller.selection_for(
        context=request_context,
        layer_id=4,
        phase="decode",
        token_index=1,
        route=route,
    )
    assert first is second
    assert calls == [4]

    token_context = _context(
        PRAExecutionPolicy(
            selection_stage="token",
            materialization_scope="token",
        )
    )
    names = iter(("c0", "c1"))
    for token in range(2):
        controller.selection_for(
            context=token_context,
            layer_id=4,
            phase="decode",
            token_index=token,
            route=lambda _layer: _rows(next(names)),
        )
    summary = token_context.summary()
    assert summary["selection_epochs"] == 2
    assert summary["temporal_selection_jaccard_mean"] == 0.0
    assert summary["trace"][-1]["selection_additions"] == 1
    assert summary["trace"][-1]["selection_removals"] == 1


def test_token_shared_reuses_current_epoch_across_later_layers():
    controller = PRASelectionController()
    context = _context(
        PRAExecutionPolicy(
            selection_stage="token",
            materialization_scope="token",
        )
    )
    calls = []
    first = controller.selection_for(
        context=context,
        layer_id=2,
        phase="decode",
        token_index=3,
        route=lambda layer: calls.append(layer) or _rows(),
    )
    later = controller.selection_for(
        context=context,
        layer_id=6,
        phase="decode",
        token_index=3,
        route=lambda layer: calls.append(layer) or _rows("unexpected"),
    )
    assert later is first
    assert calls == [2]


def test_per_layer_plan_preserves_row_and_layer_isolation():
    controller = PRASelectionController()
    context = _context(PRAExecutionPolicy(selection_layer_scope="per_layer"))
    for layer in (2, 4):
        controller.selection_for(
            context=context,
            layer_id=layer,
            phase="request",
            token_index=0,
            route=lambda current: _rows(f"c{current}"),
        )
    assert context.selection_plan is not None
    assert context.selection_plan.rows_for(2)[0][0].chunk_id == "c2"
    assert context.selection_plan.rows_for(4)[0][0].chunk_id == "c4"


def test_materialization_cache_obeys_request_and_layer_lifetimes():
    calls = []

    def materialize(plan, layer_id, metadata):
        calls.append((plan.epoch_id, layer_id, dict(metadata)))
        return object()

    plan = PRASelectionPlan(
        "request",
        "shared",
        source_layer=4,
        epoch_id=1,
        shared_rows=_rows(),
    )
    request_context = _context(PRAExecutionPolicy())
    manager = PRAMaterializationManager(materialize)
    manager.begin_request(request_context)
    first = manager.get_layer_memory(
        context=request_context, plan=plan, layer_id=4, direct_tokens=5
    )
    second = manager.get_layer_memory(
        context=request_context, plan=plan, layer_id=4, direct_tokens=5
    )
    assert first is second
    assert len(calls) == 1

    layer_context = _context(
        PRAExecutionPolicy(
            materialization_scope=PRAMaterializationScope.LAYER,
            residency_policy=PRAResidencyPolicy.LAYER_LIFETIME,
        )
    )
    manager.begin_request(layer_context)
    manager.get_layer_memory(context=layer_context, plan=plan, layer_id=4)
    manager.end_layer(layer_context, 4)
    manager.get_layer_memory(context=layer_context, plan=plan, layer_id=4)
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("stage", "scope", "expected"),
    (
        ("request", "shared", 1),
        ("request", "per_layer", 4),
        ("phase", "shared", 2),
        ("phase", "per_layer", 8),
        ("token", "shared", 16),
        ("token", "per_layer", 64),
    ),
)
def test_analytical_routing_operation_model(stage, scope, expected):
    policy = PRAExecutionPolicy(
        selection_stage=stage,
        selection_layer_scope=scope,
        materialization_scope=(
            "phase" if stage == "phase" else "token" if stage == "token" else "request"
        ),
    )
    assert analytical_routing_operations(
        policy, active_layers=4, phases=2, generated_tokens=16
    ) == expected
