"""Reusable arbitrary-Python experiment and sweep infrastructure."""

from .aggregate import aggregate_metrics
from .loader import invoke_callable, load_callable
from .models import (
    ExperimentContext,
    ExperimentDefinition,
    ExperimentEntrypoint,
    RetryConfig,
    Trial,
    TrialState,
)
from .runner import ExperimentRunResult, run_experiment
from .sweep import expand_trials, parse_seed_spec, set_dotted, stable_fingerprint

__all__ = [
    "ExperimentContext",
    "ExperimentDefinition",
    "ExperimentEntrypoint",
    "ExperimentRunResult",
    "RetryConfig",
    "Trial",
    "TrialState",
    "aggregate_metrics",
    "expand_trials",
    "invoke_callable",
    "load_callable",
    "parse_seed_spec",
    "run_experiment",
    "set_dotted",
    "stable_fingerprint",
]
