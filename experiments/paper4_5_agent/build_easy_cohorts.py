"""Build deterministic SWE-bench Verified Easy-20 and Easy-50 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


DATASET = "princeton-nlp/SWE-bench_Verified"
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DIFFICULTY = "<15 min fix"
SELECTION_SALT = "paper4.5-swebench-verified-easy-v1"


def ids_digest(instance_ids: Iterable[str]) -> str:
    """Hash an ordered cohort using the same newline contract as the runner."""

    payload = "\n".join(instance_ids) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_easy_rows(
    rows: Iterable[Mapping[str, object]], *, limit: int
) -> list[Mapping[str, object]]:
    """Select a stable, repository-mixed prefix without using model outcomes."""

    eligible = [row for row in rows if row.get("difficulty") == DIFFICULTY]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SALT}\0{row['instance_id']}".encode("utf-8")
        ).digest()
    )
    if len(eligible) < limit:
        raise ValueError(f"need {limit} easy rows, found {len(eligible)}")
    return eligible[:limit]


def build_card(
    rows: Iterable[Mapping[str, object]], *, count: int, eligible_count: int
) -> dict[str, object]:
    """Create a public card containing identities but no hidden gold patches."""

    selected = select_easy_rows(rows, limit=count)
    instance_ids = [str(row["instance_id"]) for row in selected]
    metadata = [
        {
            "instance_id": str(row["instance_id"]),
            "repo": str(row["repo"]),
            "base_commit": str(row["base_commit"]),
            "version": str(row["version"]),
            "difficulty": str(row["difficulty"]),
        }
        for row in selected
    ]
    return {
        "schema_version": 2,
        "benchmark": "SWE-bench Verified Easy",
        "dataset": DATASET,
        "split": "test",
        "expected_count": count,
        "execution_dataset_revision": DATASET_REVISION,
        "source_revision": DATASET_REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "MIT",
        "difficulty_field": "difficulty",
        "difficulty_value": DIFFICULTY,
        "eligible_count": eligible_count,
        "selection_seed": 0,
        "selection_salt": SELECTION_SALT,
        "selection_rule": (
            f"filter difficulty == {DIFFICULTY!r}; sort by SHA256("
            f"{SELECTION_SALT!r} + NUL + instance_id); take first {count}"
        ),
        "canonical_ids_sha256": ids_digest(instance_ids),
        "instance_ids": instance_ids,
        "task_metadata": metadata,
    }


def generate(output: Path) -> tuple[Path, Path]:
    """Download the pinned source and write nested Easy-20/Easy-50 cards."""

    from datasets import load_dataset

    rows = list(load_dataset(DATASET, split="test", revision=DATASET_REVISION))
    eligible_count = sum(row.get("difficulty") == DIFFICULTY for row in rows)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for count in (20, 50):
        path = output / f"swebench_verified_easy{count}.json"
        card = build_card(rows, count=count, eligible_count=eligible_count)
        path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/paper4_5_agent/benchmarks"),
    )
    args = parser.parse_args()
    for path in generate(args.output):
        print(path)


if __name__ == "__main__":
    main()
