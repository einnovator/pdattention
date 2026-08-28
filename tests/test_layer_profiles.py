"""Contract tests for independent PRA layer roles and profile calibration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pra_hf import (
    LayerProfileRegistry,
    LayerSelection,
    PRAConfig,
    ResolvedLayerRoles,
    common_calibration_candidates,
    native_index_lifecycle,
    resolve_detail_availability,
)


def _qwen(layers: int = 28):
    return SimpleNamespace(num_hidden_layers=layers, model_type="qwen3")


def _gemma():
    return SimpleNamespace(
        num_hidden_layers=6,
        model_type="gemma3_text",
        layer_types=(
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ),
    )


def test_normalized_selection_modes_are_deterministic() -> None:
    assert LayerSelection.from_value("all_layers").resolve(8) == tuple(range(8))
    assert LayerSelection.from_value("last_n:4").resolve(8) == (4, 5, 6, 7)
    assert LayerSelection.from_value("last_fraction:0.25").resolve(8) == (6, 7)
    assert LayerSelection.from_value("evenly_spaced_n:4").resolve(8) == (0, 2, 5, 7)
    assert LayerSelection.from_value("explicit:-1,0").resolve(8) == (0, 7)


def test_four_roles_keep_minimal_detail_storage_separate_from_addresses() -> None:
    config = PRAConfig(
        routing_layers=(3, 11),
        consumption_profile="last_n:4",
        address_layers=(3, 11),
        detail_kv_encoding_policy="minimal",
    )
    roles = config.resolved_layer_roles(_qwen())

    assert roles.routing_layers == (3, 11)
    assert roles.address_layers == (3, 11)
    assert roles.consumption_layers == (24, 25, 26, 27)
    assert roles.detail_kv_layers == roles.consumption_layers
    assert roles.injected_layers == (3, 11, 24, 25, 26, 27)


def test_incoherent_role_sets_fail_before_reference_encoding() -> None:
    with pytest.raises(ValueError, match="consumption layer"):
        PRAConfig(
            consumption_layers=(6, 7),
            detail_kv_layers=(7,),
        ).resolved_layer_roles(8)
    with pytest.raises(ValueError, match="routing layer"):
        PRAConfig(
            routing_layers=(7,),
            address_layers=(6,),
        ).resolved_layer_roles(8)


def test_missing_detail_policy_and_lifecycle_are_explicit() -> None:
    roles = PRAConfig(
        routing_layer=7,
        consumption_layers=(4, 5, 6, 7),
        missing_detail_kv_policy="fail",
    ).resolved_layer_roles(8)
    states = native_index_lifecycle(
        roles,
        built_address_layers=(7,),
        built_detail_layers=(6, 7),
    )

    assert roles.missing_detail_kv_policy == "fail"
    assert states == {"address_state": "BUILT", "detail_kv_state": "PARTIAL"}
    with pytest.raises(RuntimeError, match="reencode_missing"):
        resolve_detail_availability((4, 5, 6, 7), (6, 7), policy="reencode_missing")
    with pytest.raises(ValueError, match="Missing detail K/V"):
        resolve_detail_availability((4, 5, 6, 7), (6, 7), policy="fail")
    assert resolve_detail_availability(
        (4, 5, 6, 7), (6, 7), policy="downgrade_profile"
    ) == (6, 7)


def test_external_addresses_are_separate_from_native_detail_storage() -> None:
    roles = PRAConfig(
        address_encoding_policy="external_only",
        consumption_layers=(-2, -1),
    ).resolved_layer_roles(8)

    assert roles.address_mode == "external"
    assert roles.address_layers == ()
    assert roles.detail_kv_layers == (6, 7)
    assert native_index_lifecycle(
        roles, built_address_layers=(), built_detail_layers=(6, 7)
    )["address_state"] == "SKIPPED"


def test_gemma_profiles_normalize_over_global_layers_only() -> None:
    roles = PRAConfig(
        routing_layer=-1,
        consumption_profile="all_layers",
    ).resolved_layer_roles(_gemma())

    assert roles.routing_layers == (5,)
    assert roles.consumption_layers == (2, 5)
    with pytest.raises(ValueError, match="not eligible"):
        PRAConfig(routing_layer=0, consumption_layers=(2,)).resolved_layer_roles(
            _gemma()
        )


def test_registry_resolution_and_internal_provenance() -> None:
    registry = LayerProfileRegistry.default()
    row, source = registry.resolve(
        family="qwen",
        model_id=None,
        workload=None,
        materialization=None,
        objective="balanced",
    )
    assert row["name"] == "qwen_balanced_all"
    assert source == "family_balanced_fallback"

    config = PRAConfig(layer_profile_name="default", model_id="offline/qwen")
    roles = config.resolved_layer_roles(_qwen())
    internal = config.to_internal(_qwen())
    assert roles.consumption_layers == tuple(range(28))
    assert internal.address_layer_ids == (27,)
    assert internal.detail_kv_layer_ids == tuple(range(28))
    assert internal.consumption_layer_ids == tuple(range(28))


def test_common_calibration_space_contains_contiguous_and_sparse_controls() -> None:
    candidates = common_calibration_candidates(28)
    assert candidates["last_14"].resolve(28) == tuple(range(14, 28))
    assert len(candidates["even_8"].resolve(28)) == 8
    assert candidates["all_layers"].resolve(28) == tuple(range(28))


def test_resolved_role_provenance_serializes_all_four_axes() -> None:
    roles = ResolvedLayerRoles(
        address_layers=(7,),
        detail_kv_layers=(4, 5, 6, 7),
        routing_layers=(7,),
        consumption_layers=(4, 5, 6, 7),
        detail_kv_encoding_policy="minimal",
        address_encoding_policy="routing_only",
        missing_detail_kv_policy="reencode_missing",
        address_mode="native",
        profile_name="economy",
    )
    payload = roles.to_dict()
    assert payload["address_layers"] == (7,)
    assert payload["detail_kv_layers"] == (4, 5, 6, 7)
    assert payload["routing_layers"] == (7,)
    assert payload["consumption_layers"] == (4, 5, 6, 7)
