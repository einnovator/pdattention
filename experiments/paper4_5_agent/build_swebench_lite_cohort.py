"""Build the deterministic SWE-bench Lite-50 follow-up cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .build_easy_cohorts import ids_digest


DATASET = "SWE-bench/SWE-bench_Lite"
DATASET_REVISION = "b0dde1093fe417d83b7184254edf8199c1f0dff5"
SELECTION_SALT = "paper4.5-swebench-lite50-v1"


def select_rows(
    rows: Iterable[Mapping[str, object]], *, limit: int = 50,
) -> list[Mapping[str, object]]:
    """Select a stable cohort without consulting patches, tests, or outcomes."""

    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SALT}\0{row['instance_id']}".encode("utf-8")
        ).digest(),
    )
    if len(ordered) < limit:
        raise ValueError(f"need {limit} Lite rows, found {len(ordered)}")
    return ordered[:limit]


def build_card(rows: Iterable[Mapping[str, object]], *, count: int = 50) -> dict[str, object]:
    """Create a public identity card that excludes hidden grading material."""

    selected = select_rows(rows, limit=count)
    instance_ids = [str(row["instance_id"]) for row in selected]
    return {
        "schema_version": 2,
        "benchmark": "SWE-bench Lite-50",
        "dataset": DATASET,
        "split": "test",
        "expected_count": count,
        "execution_dataset_revision": DATASET_REVISION,
        "source_revision": DATASET_REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "MIT",
        "selection_seed": 0,
        "selection_salt": SELECTION_SALT,
        "selection_rule": (
            f"sort test rows by SHA256({SELECTION_SALT!r} + NUL + instance_id); "
            f"take first {count}"
        ),
        "canonical_ids_sha256": ids_digest(instance_ids),
        "instance_ids": instance_ids,
        "task_metadata": [
            {
                "instance_id": str(row["instance_id"]),
                "repo": str(row["repo"]),
                "base_commit": str(row["base_commit"]),
                "version": str(row["version"]),
            }
            for row in selected
        ],
    }


def generate(output: Path) -> Path:
    """Download the pinned source revision and write the Lite-50 card."""

    from datasets import load_dataset

    rows = list(load_dataset(DATASET, split="test", revision=DATASET_REVISION))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_card(rows), indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/paper4_5_agent/benchmarks/swebench_lite50.json"),
    )
    print(generate(parser.parse_args().output))


if __name__ == "__main__":
    main()
