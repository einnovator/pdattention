from __future__ import annotations

from types import SimpleNamespace

import pytest

from pra_hf.hf_live_kv import select_dynamic_cache
from pra_hf.live_history import LiveKVInterval, LiveKVSelectionPlan


def test_live_plan_keeps_selected_width_and_position_extent_separate() -> None:
    plan = LiveKVSelectionPlan.create(
        100,
        (
            LiveKVInterval(0, 12, "system", "system"),
            LiveKVInterval(40, 55, "turn-4", "turn-4"),
            LiveKVInterval(90, 100, "active", "active"),
        ),
    )
    assert plan.selected_tokens == 37
    assert plan.source_position_base == 100
    assert plan.has_holes
    assert not plan.full_retention


def test_live_plan_rejects_overlap_and_duplicate_record_identity() -> None:
    with pytest.raises(ValueError, match="ordered and disjoint"):
        LiveKVSelectionPlan.create(20, ((0, 10), (9, 15)))
    with pytest.raises(ValueError, match="stable record"):
        LiveKVSelectionPlan.create(
            20,
            (
                LiveKVInterval(0, 5, "same", "g"),
                LiveKVInterval(8, 12, "same", "g"),
            ),
        )


def test_hf_full_selection_is_a_view_and_sparse_selection_reports_pack_copy() -> None:
    torch = pytest.importorskip("torch")
    keys = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)
    values = keys + 100
    source = SimpleNamespace(layers=[SimpleNamespace(keys=keys, values=values)])

    full = select_dynamic_cache(source, LiveKVSelectionPlan.full(6))
    assert not full.physical_kv_copy
    assert full.selected_text_reencoded_tokens == 0
    assert full.cache.layers[0].keys.data_ptr() == keys.data_ptr()

    sparse = select_dynamic_cache(
        source,
        LiveKVSelectionPlan.create(6, ((0, 2), (4, 6))),
    )
    assert sparse.physical_kv_copy
    assert sparse.cache.layers[0].keys.shape[-2] == 4
    assert sparse.cache.layers[0].keys[0, 0, :, 0].tolist() == [0, 4, 16, 20]

