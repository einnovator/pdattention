"""Compatibility re-exports for :mod:`common.plots`."""

from common.plots import (
    MetricsHistory,
    save_metric_plots,
    save_metrics_json,
    save_metrics_markdown,
)

__all__ = [
    "MetricsHistory",
    "save_metric_plots",
    "save_metrics_json",
    "save_metrics_markdown",
]
