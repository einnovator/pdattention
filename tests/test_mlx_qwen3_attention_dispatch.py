from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from pra_mlx.qwen3_segmented import (
    install_qwen3_segmented_attention,
    set_qwen3_segmented_attention_active,
)


class _Module:
    def __init__(self) -> None:
        pass


class _Attention:
    pass


class _Layer:
    def __init__(self) -> None:
        self.self_attn = _Attention()


class _Model:
    def __init__(self) -> None:
        self.args = SimpleNamespace(model_type="qwen3")
        self.layers = (_Layer(), _Layer())

    @staticmethod
    def parameters():
        return ()


def test_qwen3_patch_can_be_prepared_inactive_and_dispatched(monkeypatch) -> None:
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.eval = lambda *_: None
    nn = ModuleType("mlx.nn")
    nn.Module = _Module
    mlx.core = core
    mlx.nn = nn
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.nn", nn)

    model = _Model()
    native = tuple(layer.self_attn for layer in model.layers)

    assert install_qwen3_segmented_attention(
        model, compiled=False, active=False
    ) == 2
    assert tuple(layer.self_attn for layer in model.layers) == native
    assert model._pra_segmented_attention_active is False

    assert set_qwen3_segmented_attention_active(model, True) == 2
    segmented = tuple(layer.self_attn for layer in model.layers)
    assert all(after is not before for after, before in zip(segmented, native))
    assert model._pra_segmented_attention_active is True

    assert set_qwen3_segmented_attention_active(model, False) == 2
    assert tuple(layer.self_attn for layer in model.layers) == native
    assert model._pra_segmented_attention_active is False

    # Reinstallation is idempotent but still honors the requested dispatch.
    assert install_qwen3_segmented_attention(model, active=True) == 0
    assert tuple(layer.self_attn for layer in model.layers) == segmented
