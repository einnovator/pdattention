"""Tiny dependency-free entrypoints used in documentation and smoke tests."""

from __future__ import annotations

import json

from .models import ExperimentContext


def scalar(params: dict, context: ExperimentContext) -> dict:
    """Return a deterministic scalar and demonstrate artifact creation."""

    seed = int(params.get("seed", 0))
    offset = float(params.get("offset", 0.0))
    suffix = "identity.json" if context.rank == 0 else f"identity-rank-{context.rank}.json"
    artifact = context.output_dir / "artifacts" / suffix
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps({"seed": seed, "worker": context.worker_name}, indent=2),
        encoding="utf-8",
    )
    return {"score": seed + offset}
