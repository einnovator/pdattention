"""Structured and reproducible artifact helpers for Paper 2."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers


def git_value(*args: str) -> str:
    """Return one Git value without making experiment logging a hard dependency."""
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def runtime_metadata() -> dict:
    """Capture the software, repository, and accelerator state for one run."""
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": device,
    }


def write_artifacts(result: dict, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write a complete JSON record and one-row scalar CSV summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(child) for child in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    json_path.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    flat: dict[str, str | int | float | bool | None] = {}

    def visit(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else key, child)
        elif isinstance(value, Path):
            flat[prefix] = str(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[prefix] = value

    visit("", result)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(flat))
        writer.writeheader()
        writer.writerow(flat)
    return json_path, csv_path
