import sys
import types

import pytest

from experiments.paper4_5_agent.run_mlx_live_agent_kv_lifecycle import (
    _chunked_source_prefill,
)


class _FakeMX(types.ModuleType):
    int32 = "int32"

    def __init__(self):
        super().__init__("mlx.core")
        self.evaluated = []

    @staticmethod
    def array(values, dtype=None):
        return {"values": values, "dtype": dtype}

    def eval(self, value):
        self.evaluated.append(value)


def test_source_prefill_is_bounded_and_preserves_order(monkeypatch):
    fake_mx = _FakeMX()
    fake_package = types.ModuleType("mlx")
    fake_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    observed = []

    def model(values, *, cache):
        observed.extend(values["values"][0])
        return tuple(values["values"][0])

    _chunked_source_prefill(model, object(), list(range(10)), step_size=4)
    assert observed == list(range(10))
    assert fake_mx.evaluated == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]


@pytest.mark.parametrize("source_ids,step_size", [([], 4), ([1], 0)])
def test_source_prefill_rejects_invalid_geometry(monkeypatch, source_ids, step_size):
    fake_mx = _FakeMX()
    fake_package = types.ModuleType("mlx")
    fake_package.core = fake_mx
    monkeypatch.setitem(sys.modules, "mlx", fake_package)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    with pytest.raises(ValueError):
        _chunked_source_prefill(lambda *_args, **_kwargs: None, object(), source_ids, step_size=step_size)
