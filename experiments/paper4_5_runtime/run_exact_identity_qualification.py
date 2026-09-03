"""Run and canonicalize exact-identity bundle qualification cohorts.

The runner deliberately keeps model revision and quantization in the bundle
specification. It executes one frozen-selection E0/E2 artifact per dataset and
only emits the combined canonical summary after all requested datasets exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.paper4_5_runtime.build_hf_catalog_bundles import (
    EXACT_QUALIFICATION_RESULTS,
    RESULTS,
    ROOT,
    SPECS,
    _load_paired_evidence,
)
from pra_hf.bundle_evidence import canonicalize_paired_transport_evidence
from pra_hf.canonical_evidence import MeasurementState


DATASETS = ("qasper", "hotpotqa", "2wikimultihopqa")


def build_command(
    slug: str,
    dataset: str,
    *,
    max_examples: int,
    concurrency: int,
) -> tuple[list[str], Path]:
    """Build one reproducible MLX qualification invocation."""

    spec = SPECS[slug]
    if spec.get("engine", "mlx") != "mlx" or not spec.get("matched_evidence"):
        raise ValueError(f"{slug} is not configured for MLX matched qualification")
    output = EXACT_QUALIFICATION_RESULTS / slug / f"matched_e0_e2_{dataset}.json"
    command = [
        sys.executable,
        str(ROOT / "experiments/paper6_2_mlx/run_matched_e0_e2.py"),
        "--dataset",
        dataset,
        "--model",
        spec["base_model"],
        "--revision",
        spec["revision"],
        "--max-examples",
        str(max_examples),
        "--warm-repeats",
        "2",
        "--multi-query-count",
        "3",
        "--concurrency",
        str(concurrency),
        "--output",
        str(output),
    ]
    return command, output


def write_summary(slug: str) -> Path:
    """Validate all dataset artifacts and write the canonical three-arm record."""

    rows, artifacts = _load_paired_evidence(slug, SPECS[slug])
    if not rows:
        raise RuntimeError(f"No complete exact-identity evidence exists for {slug}")
    has_router = (RESULTS / slug / "comparison.json").is_file()
    state = (
        MeasurementState.NEEDS_RUN
        if has_router
        else MeasurementState.NO_QUALIFIED_ADAPTER
    )
    combined = next(row for row in rows if row["dataset"] == "combined")
    canonical = canonicalize_paired_transport_evidence(
        combined,
        adaptor_state=state,
        adaptor_note=(
            "The exact learned router is bundled but has not completed the matched end-task arm."
            if has_router
            else "No learned adaptor qualified for this exact model revision and quantization."
        ),
    )
    output = EXACT_QUALIFICATION_RESULTS / slug / "qualification_summary.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundle_slug": slug,
                "artifacts": [str(path.relative_to(ROOT)).replace("\\", "/") for path in artifacts],
                "paired_evidence": rows,
                "canonical_evidence": canonical.serialize_for_control_plane(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=tuple(
            slug
            for slug, spec in SPECS.items()
            if spec.get("matched_evidence") and spec.get("engine", "mlx") == "mlx"
        ),
    )
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    environment = os.environ.copy()
    source_paths = (str(ROOT), str(ROOT / "src"))
    environment["PYTHONPATH"] = os.pathsep.join(
        (*source_paths, environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    for dataset in datasets:
        command, output = build_command(
            args.model,
            dataset,
            max_examples=args.max_examples,
            concurrency=args.concurrency,
        )
        print(" ".join(command))
        if args.dry_run or (output.is_file() and not args.force):
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    if not args.dry_run and args.dataset == "all":
        print(write_summary(args.model))


if __name__ == "__main__":
    main()
