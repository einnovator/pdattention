"""Deterministic selected-memory residency and eviction pressure sweep."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pra_hf.engine_memory import LogicalPRABlock, LogicalPRABlockId, LogicalPRABlockStore
from pra_hf.engine_residency import EnginePRAResidencyManager, PRAEvictionPolicy


SEEDS = (11, 23, 37, 53, 71)
POLICIES = tuple(PRAEvictionPolicy)
MIB = 1024 * 1024


def _register(store: LogicalPRABlockStore, index: int) -> str:
    identity = LogicalPRABlockId(
        tenant_id="benchmark",
        session_id=None,
        resource_id=f"resource-{index}",
        resource_version="v1",
        record_type="document",
        token_start=0,
        token_end=(index + 1) * 64,
        layer=0,
        model_revision="controlled",
        dtype="fp16",
        layout="opaque-engine-native",
        materialization_profile="controlled",
        position_policy="source_local",
    )
    return store.register(LogicalPRABlock(identity, address_bytes=64, detail_bytes=0))


def _trace(seed: int, length: int = 120) -> list[int]:
    rng = random.Random(seed)
    weights = [12, 10, 8, 6, 4, 3, 2, 1]
    trace = rng.choices(range(len(weights)), weights=weights, k=length)
    # Force periodic scans so the budget is genuinely pressured.
    for start in range(0, length, 24):
        trace[start : start + 8] = range(8)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sizes = tuple((index + 1) * MIB for index in range(8))
    budget = 14 * MIB
    rows = []
    for seed in SEEDS:
        trace = _trace(seed)
        for policy in POLICIES:
            store = LogicalPRABlockStore()
            keys = tuple(_register(store, index) for index in range(8))
            manager = EnginePRAResidencyManager(
                store, max_resident_bytes=budget, policy=policy
            )
            for request_index, resource_index in enumerate(trace):
                key = keys[resource_index]
                store.select((key,), tenant_id="benchmark")
                manager.resolve(
                    key,
                    lambda index=resource_index: (
                        {"resource": index},
                        sizes[index],
                    ),
                    request_id=f"seed-{seed}-request-{request_index}",
                )
                with manager.pin_request(f"seed-{seed}-request-{request_index}", (key,)):
                    pass
            metrics = manager.metrics().to_dict()
            manager.close()
            rows.append(
                {
                    "seed": seed,
                    "policy": policy.value,
                    "requests": len(trace),
                    "working_set_bytes": sum(sizes),
                    "residency_budget_bytes": budget,
                    "reload_amplification": metrics["loads"] / len(trace),
                    **metrics,
                }
            )

    payload = {
        "schema_version": "1.0",
        "experiment": "pra_engine_residency_pressure_v1",
        "evidence_tier": "CONTROLLED_SIMULATION",
        "seeds": list(SEEDS),
        "policies": [policy.value for policy in POLICIES],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
