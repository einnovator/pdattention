from __future__ import annotations

from experiments.paper6_3_openvino.run_physical_telemetry import _numeric_total


def test_numeric_total_preserves_nested_plugin_memory_counters() -> None:
    assert _numeric_total({"usm_device": 10, "usm_host": 20}) == 30
    assert _numeric_total({"unsupported": "n/a"}) is None
    assert _numeric_total(True) is None
