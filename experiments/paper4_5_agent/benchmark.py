"""Immutable benchmark-card validation for controlled agent experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_benchmark_card(path: str | Path) -> dict[str, Any]:
    """Load a benchmark card and verify its ordered source-list digest."""

    card = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = card.get("instance_ids") or []
    expected_count = int(card.get("expected_count", 50))
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError(
            f"fixed SWE-bench cohort must contain exactly {expected_count} unique IDs"
        )
    source_bytes = ("\n".join(ids) + "\n").encode("utf-8")
    observed = hashlib.sha256(source_bytes).hexdigest()
    expected = str(card.get("canonical_ids_sha256") or "").lower()
    if observed != expected:
        raise ValueError(f"benchmark cohort digest mismatch: {observed} != {expected}")
    return card


def precision_diagnostic_ids(card: dict[str, Any]) -> tuple[str, ...]:
    """Return the preregistered first-ten diagnostic without resampling."""

    return tuple(card["instance_ids"][:10])
