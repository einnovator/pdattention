"""Attach the current committed code identity to completed Paper 1.5 artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from experiments.paper1_5_rope.common import (  # noqa: E402
    RESULTS,
    environment_metadata,
    refresh_manifest,
    write_json,
)


def main() -> Path:
    """Restamp result envelopes without changing measured rows or aggregates."""
    metadata = environment_metadata()
    for path in RESULTS.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "metadata" in payload:
            payload["metadata"] = metadata
            write_json(path, payload)
    return refresh_manifest(metadata=metadata)


if __name__ == "__main__":
    print(main())
