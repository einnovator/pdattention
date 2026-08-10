"""Shared configuration and artifact helpers for Paper 1.5 experiments."""

from __future__ import annotations

import csv
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs" / "papers" / "shared" / "results" / "paper1_5_rope"
SEEDS = (1, 7, 21, 42, 87)
SPLIT_COUNTS = (2, 5, 16, 32, 64)
TIERS = {
    # Controlled tiers are deliberately smaller than the product-profile names in config.yml.
    "tiny": {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 128, "steps": 100},
    "small": {"d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 256, "steps": 150},
}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch without changing deterministic algorithm policy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_metadata() -> dict:
    """Capture the code/device identity attached to every canonical result file."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    return {
        "git_sha": sha,
        "branch": branch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write flat experimental rows with a stable union of observed columns."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refresh_manifest(*, metadata: dict | None = None, **fields) -> Path:
    """Refresh the canonical artifact inventory after any independent experiment."""
    path = RESULTS / "manifest.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = {
        **existing,
        **fields,
        "metadata": metadata or existing.get("metadata") or environment_metadata(),
        "artifacts": sorted(item.name for item in RESULTS.iterdir() if item.name != path.name),
    }
    write_json(path, payload)
    return path
