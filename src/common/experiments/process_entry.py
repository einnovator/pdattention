"""Private subprocess entrypoint for isolated experiment trials."""

from __future__ import annotations

import sys
from pathlib import Path

from common.distributed.models import DistributionMode, ResourceRequirements, WorkerConfig

from .models import ExperimentEntrypoint, Trial
from .state import read_json
from .trial import execute_trial_local


def main() -> int:
    invocation = read_json(sys.argv[1])
    manifest = invocation["trial"]
    trial = Trial(
        experiment_name=manifest["experiment"],
        trial_id=manifest["trial_id"],
        parameters=manifest["parameters"],
        entrypoint=ExperimentEntrypoint.from_mapping(manifest["entrypoint"]),
        distribution=DistributionMode.from_value(manifest["distribution"]),
        cluster_name=manifest["cluster"],
        storage_name=manifest.get("storage"),
        resources=ResourceRequirements.from_mapping(manifest.get("resources")),
        fingerprint=manifest["fingerprint"],
        assigned_workers=tuple(manifest.get("workers") or ()),
        attempt=int(invocation.get("attempt", 1)),
    )
    worker = WorkerConfig.from_mapping(invocation["worker"], invocation["worker_config"])
    execute_trial_local(
        trial,
        run_id=invocation["run_id"],
        run_dir=Path(invocation["run_dir"]),
        worker=worker,
        resumed=bool(invocation.get("resumed")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
