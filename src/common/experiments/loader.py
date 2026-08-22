"""Dynamic experiment entrypoint loading without package-install requirements."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import runpy
import hashlib
from pathlib import Path

from .models import ExperimentContext, ExperimentEntrypoint


def load_callable(entrypoint: ExperimentEntrypoint):
    if not entrypoint.function:
        raise ValueError("Script-only entrypoints do not expose a callable.")
    if entrypoint.module:
        module = importlib.import_module(entrypoint.module)
    else:
        path = Path(str(entrypoint.file)).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]
        name = f"common_experiment_{path.stem}_{digest}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load experiment file {path}.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    function = getattr(module, entrypoint.function, None)
    if not callable(function):
        raise AttributeError(f"Entrypoint function {entrypoint.function!r} is not callable.")
    return function


def invoke_callable(entrypoint: ExperimentEntrypoint, params: dict, context: ExperimentContext):
    """Invoke the stable contract while tolerating one-argument legacy callables."""

    function = load_callable(entrypoint)
    signature = inspect.signature(function)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(positional) >= 2 or any(
        parameter.kind == parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return function(params, context)
    if len(positional) == 1:
        return function(params)
    return function()


def run_script(entrypoint: ExperimentEntrypoint) -> None:
    if not entrypoint.script_only:
        raise ValueError("run_script is only valid for file-only entrypoints.")
    runpy.run_path(str(Path(str(entrypoint.file)).resolve()), run_name="__main__")
