"""Import guards for SGLang's MLX backend on hosts without CUDA/Triton.

SGLang's MLX model runner currently imports shared CUDA-oriented modules while
building its Python module graph.  Those imports require Triton and inspect a
device ``Stream`` even though the MLX request path executes neither.  This
guard keeps those definitions importable, fails closed if a Triton symbol is
actually executed, and restores global PyTorch/Transformers hooks immediately
after the MLX runner has imported.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Iterator


class TritonExecutionForbidden(RuntimeError):
    """Raised if the MLX path tries to execute an import-only Triton symbol."""


class _Config:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _JITFunction:
    pass


class _Constexpr:
    def __call__(self, value=None):
        return value


class _LanguageSymbol:
    def __init__(self, name: str, attempts: list[str]):
        self.name = name
        self.attempts = attempts

    def __call__(self, *args, **kwargs):
        self.attempts.append(self.name)
        raise TritonExecutionForbidden(
            f"Triton language symbol {self.name!r} executed on the MLX path"
        )

    def __getattr__(self, name: str):
        return _LanguageSymbol(f"{self.name}.{name}", self.attempts)


@dataclass(frozen=True)
class SGLangMLXImportGuardReport:
    triton_import_stub_installed: bool
    torch_compile_guarded: bool
    torch_device_annotation_guarded: bool
    duplicate_config_registration_guarded: bool
    forbidden_execution_attempts: list[str]


def _decorator(function=None, *args, **kwargs):
    if callable(function):
        return function

    def decorate(candidate):
        return candidate

    return decorate


def _install_import_only_triton() -> tuple[bool, list[str]]:
    existing = sys.modules.get("triton")
    if existing is not None:
        attempts = getattr(existing, "__pra_forbidden_execution_attempts__", [])
        return False, attempts
    if importlib.util.find_spec("triton") is not None:
        return False, []

    attempts: list[str] = []
    root = ModuleType("triton")
    root.__path__ = []
    root.__pra_import_only__ = True
    root.__pra_forbidden_execution_attempts__ = attempts
    root.Config = _Config
    root.jit = _decorator
    root.autotune = _decorator
    root.heuristics = _decorator
    root.cdiv = lambda left, right: (left + right - 1) // right
    root.next_power_of_2 = lambda value: (
        1 if value <= 1 else 1 << (int(value) - 1).bit_length()
    )
    root.runtime = SimpleNamespace(jit=SimpleNamespace(JITFunction=_JITFunction))

    language = ModuleType("triton.language")
    language.__path__ = []
    language.constexpr = _Constexpr()
    language.__getattr__ = lambda name: _LanguageSymbol(name, attempts)
    root.language = language

    extra = ModuleType("triton.language.extra")
    extra.__path__ = []
    libdevice = ModuleType("triton.language.extra.libdevice")

    def libdevice_symbol(name: str):
        return _LanguageSymbol(f"libdevice.{name}", attempts)

    libdevice.__getattr__ = libdevice_symbol
    extra.libdevice = libdevice

    testing = ModuleType("triton.testing")

    def do_bench(*args, **kwargs):
        attempts.append("testing.do_bench")
        raise TritonExecutionForbidden(
            "Triton benchmark executed on the MLX path"
        )

    testing.do_bench = do_bench
    root.testing = testing

    sys.modules.update(
        {
            "triton": root,
            "triton.language": language,
            "triton.language.extra": extra,
            "triton.language.extra.libdevice": libdevice,
            "triton.testing": testing,
        }
    )
    return True, attempts


@contextmanager
def sglang_mlx_import_guard() -> Iterator[SGLangMLXImportGuardReport]:
    """Guard only the import of SGLang's MLX runner.

    ``mlx_lm`` and PyTorch load before the Triton stub becomes importable so
    PyTorch records the truthful no-Triton platform state.  Import-only global
    hooks are restored in ``finally``; the stub remains to fail closed if a
    CUDA-only function is accidentally called later.
    """

    import mlx_lm  # noqa: F401
    import torch
    from transformers import AutoConfig

    installed, attempts = _install_import_only_triton()
    original_compile = torch.compile
    original_get_device_module = torch.get_device_module
    original_register = AutoConfig.register
    original_register_descriptor = inspect.getattr_static(AutoConfig, "register")

    class _ImportOnlyStream:
        pass

    torch.compile = _decorator
    torch.get_device_module = lambda *args, **kwargs: SimpleNamespace(
        Stream=_ImportOnlyStream
    )

    def register_idempotently(model_type, config, **kwargs):
        return original_register(model_type, config, exist_ok=True)

    AutoConfig.register = staticmethod(register_idempotently)
    report = SGLangMLXImportGuardReport(
        triton_import_stub_installed=installed,
        torch_compile_guarded=True,
        torch_device_annotation_guarded=True,
        duplicate_config_registration_guarded=True,
        forbidden_execution_attempts=attempts,
    )
    try:
        yield report
    finally:
        torch.compile = original_compile
        torch.get_device_module = original_get_device_module
        AutoConfig.register = original_register_descriptor
