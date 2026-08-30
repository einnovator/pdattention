from __future__ import annotations

from experiments.paper6_3_openvino.run_natural_e0 import _mean, _metric


class _Metric:
    mean = 12.5


class _Perf:
    def get_ttft(self):
        return _Metric()


def test_metric_accepts_openvino_mean_wrapper():
    assert _metric(_Perf(), "get_ttft") == 12.5
    assert _metric(_Perf(), "missing") is None


def test_mean_ignores_unavailable_measurements():
    rows = [{"value": 1.0}, {"value": None}, {"value": 3.0}]
    assert _mean(rows, "value") == 2.0
    assert _mean([{"value": None}], "value") is None
