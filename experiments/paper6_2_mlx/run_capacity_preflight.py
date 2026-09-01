"""Record whether an MLX model fits a benchmark-valid unified-memory budget."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def required_bytes(
    *,
    checkpoint_bytes: int,
    physical_memory_bytes: int,
    selected_kv_bytes: int,
    reserve_fraction: float,
    minimum_reserve_bytes: int,
) -> dict[str, int | float | str]:
    """Return a conservative capacity decision before model download/load."""

    reserve = max(
        int(physical_memory_bytes * reserve_fraction), int(minimum_reserve_bytes)
    )
    required = int(checkpoint_bytes) + int(selected_kv_bytes) + reserve
    return {
        "checkpoint_bytes": int(checkpoint_bytes),
        "selected_kv_allowance_bytes": int(selected_kv_bytes),
        "workspace_and_os_reserve_bytes": reserve,
        "physical_memory_bytes": int(physical_memory_bytes),
        "required_bytes": required,
        "required_over_physical": required / max(int(physical_memory_bytes), 1),
        "status": "ELIGIBLE" if required <= physical_memory_bytes else "NOT_RUN_CAPACITY_GATE",
    }


def _physical_memory() -> int:
    return int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--selected-kv-bytes", type=int, default=128 * 2**20)
    parser.add_argument("--reserve-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-reserve-bytes", type=int, default=4 * 2**30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import model_info

    info = model_info(args.model, revision=args.revision, files_metadata=True)
    checkpoint_bytes = sum(
        int(sibling.size or 0)
        for sibling in info.siblings
        if sibling.rfilename.endswith((".safetensors", ".npz"))
    )
    decision = required_bytes(
        checkpoint_bytes=checkpoint_bytes,
        physical_memory_bytes=_physical_memory(),
        selected_kv_bytes=args.selected_kv_bytes,
        reserve_fraction=args.reserve_fraction,
        minimum_reserve_bytes=args.minimum_reserve_bytes,
    )
    payload = {
        "schema_version": "paper6.2-mlx-capacity-preflight-v1",
        "experiment": "benchmark_valid_unified_memory_preflight",
        "evidence_tier": "PHYSICAL_CHECKPOINT_AND_HARDWARE_METADATA",
        "model_id": args.model,
        "model_revision": info.sha,
        "decision": decision,
        "claim_boundary": (
            "The gate prevents swap-dominated timing; it is not a claim that "
            "MLX cannot execute the model with virtual memory."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
