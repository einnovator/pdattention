"""Offline protocol checks for the Paper 2 multi-layer consumption sweep."""

from experiments.paper2_hf.qa.run_multilayer_pra import layer_schedules


def test_qwen_28_layer_schedules_resolve_declared_depth_bands():
    schedules = layer_schedules(28)

    assert schedules["last_1"] == (27,)
    assert schedules["last_4"] == (24, 25, 26, 27)
    assert schedules["last_8"] == tuple(range(20, 28))
    assert schedules["last_12"] == tuple(range(16, 28))
    assert schedules["last_14"] == tuple(range(14, 28))
    assert schedules["last_16"] == tuple(range(12, 28))
    assert schedules["last_20"] == tuple(range(8, 28))
    assert schedules["last_24"] == tuple(range(4, 28))
    assert schedules["last_quarter"] == tuple(range(21, 28))
    assert schedules["last_half"] == tuple(range(14, 28))
    assert schedules["all"] == tuple(range(28))


def test_placement_schedules_hold_count_or_declared_stride():
    schedules = layer_schedules(28)

    assert schedules["early_4"] == (0, 1, 2, 3)
    assert schedules["middle_4"] == (12, 13, 14, 15)
    assert schedules["even_4"] == (0, 9, 18, 27)
    assert schedules["even_8"] == (0, 4, 8, 12, 15, 19, 23, 27)
    assert schedules["every_4"] == (0, 4, 8, 12, 16, 20, 24)
