"""Reuse an exact nested calibration prefix without rerunning model inference."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import load_benchmark_card


def promote(
    source_card_path: Path,
    destination_card_path: Path,
    source_output: Path,
    destination_output: Path,
) -> Path:
    """Copy only complete per-task chunks after proving exact prefix identity."""

    source_card = load_benchmark_card(source_card_path)
    destination_card = load_benchmark_card(destination_card_path)
    source_ids = source_card["instance_ids"]
    destination_ids = destination_card["instance_ids"]
    if destination_ids[:len(source_ids)] != source_ids:
        raise ValueError("source cohort is not an ordered prefix of destination cohort")
    destination_output.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, instance_id in enumerate(source_ids):
        source_chunk = source_output / f"chunk_{index:02d}"
        receipt_path = source_chunk / "official_chunk_result.json"
        if not receipt_path.is_file():
            raise ValueError(f"source chunk {index} has no official receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("submitted_ids") != [instance_id]:
            raise ValueError(f"source chunk {index} does not match {instance_id}")
        destination_chunk = destination_output / source_chunk.name
        if destination_chunk.exists():
            existing = json.loads(
                (destination_chunk / "official_chunk_result.json").read_text(encoding="utf-8")
            )
            if existing.get("submitted_ids") != [instance_id]:
                raise ValueError(f"destination chunk {index} contains a different task")
        else:
            shutil.copytree(source_chunk, destination_chunk)
        for suffix in ("agent.log", "grader.log"):
            source_log = source_output / f"chunk_{index:02d}.{suffix}"
            destination_log = destination_output / source_log.name
            if source_log.is_file() and not destination_log.exists():
                shutil.copy2(source_log, destination_log)
        copied.append(instance_id)
    manifest = destination_output / "nested_promotion.json"
    manifest.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_card_sha256": source_card["canonical_ids_sha256"],
        "destination_card_sha256": destination_card["canonical_ids_sha256"],
        "source_output": str(source_output.resolve()),
        "destination_output": str(destination_output.resolve()),
        "copied_task_ids": copied,
        "reuse_semantics": "exact ordered prefix; inference and official grading are not replayed",
    }, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument("--destination-card", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--destination-output", type=Path, required=True)
    args = parser.parse_args()
    print(promote(
        args.source_card, args.destination_card,
        args.source_output, args.destination_output,
    ))


if __name__ == "__main__":
    main()
