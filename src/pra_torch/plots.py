"""Metric history persistence and matplotlib plots for PRA training runs."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _plain_config(config) -> dict | str:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    if isinstance(config, dict):
        return dict(config)
    return str(config)


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


_RECORD_DIMENSIONS = {"epoch", "batch_in_epoch", "batch_step", "optimizer_step", "optimizer_updated", "batches"}


def save_metrics_json(path: str | Path, records: list[dict], config=None) -> Path:
    """Save scalar metric history as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": _plain_config(config), "records": records}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def save_metrics_markdown(path: str | Path, records: list[dict], plot_paths: list[Path]) -> Path:
    """Write a compact markdown report for the metric history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latest: dict[str, dict] = {}
    for record in records:
        latest[record["split"]] = record

    lines = ["# Training Metrics", ""]
    if latest:
        lines.extend(["## Latest Metrics", ""])
        for split, record in sorted(latest.items()):
            lines.append(f"### {split}")
            lines.append("")
            lines.append(f"- step: {record['step']}")
            for key, value in sorted(record["metrics"].items()):
                lines.append(f"- {key}: {value:.6g}")
            lines.append("")
    if plot_paths:
        lines.extend(["## Plots", ""])
        for plot in plot_paths:
            rel = plot.relative_to(path.parent)
            lines.append(f"- [{plot.stem}]({rel.as_posix()})")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path


def save_metric_plots(records: list[dict], plot_dir: str | Path) -> list[Path]:
    """Save one PNG per scalar metric using matplotlib."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    series: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        split = record["split"]
        step = int(record["step"])
        for key, value in record["metrics"].items():
            series[key][split].append((step, float(value)))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        note = plot_dir / "plotting_unavailable.txt"
        note.write_text(f"matplotlib unavailable: {exc}\n", encoding="utf-8")
        return []

    plot_paths: list[Path] = []
    for metric, split_values in sorted(series.items()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for split, values in sorted(split_values.items()):
            values = sorted(values)
            ax.plot([step for step, _ in values], [value for _, value in values], marker="o", label=split)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plot_dir / f"{metric}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        plot_paths.append(path)
    return plot_paths


class MetricsHistory:
    """Collect metrics and write JSON, markdown, and PNG plot artifacts."""

    def __init__(self, run_dir: str | Path, save_plots: bool = True):
        self.run_dir = Path(run_dir)
        self.save_plots = save_plots
        self.records: list[dict] = []
        self.config = None

    def log_config(self, config) -> None:
        self.config = config

    def log_metrics(self, metrics: dict, step: int, split: str) -> None:
        scalar_metrics = _scalar_metrics(metrics)
        dimensions = {key: scalar_metrics.pop(key) for key in list(scalar_metrics) if key in _RECORD_DIMENSIONS}
        if scalar_metrics:
            self.records.append(
                {"step": int(step), "split": str(split), **dimensions, "metrics": scalar_metrics}
            )

    def close(self) -> None:
        metrics_path = save_metrics_json(self.run_dir / "metrics.json", self.records, self.config)
        plot_paths = save_metric_plots(self.records, self.run_dir / "plots") if self.save_plots else []
        save_metrics_markdown(metrics_path.with_suffix(".md"), self.records, plot_paths)
