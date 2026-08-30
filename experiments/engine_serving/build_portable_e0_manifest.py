"""Build a self-contained selected-text benchmark for remote engine hosts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.engine_serving.matched_qa import DATASETS, load_matched_examples
from experiments.paper6_2_mlx.run_answer_quality_pressure import _examples


def build_portable_manifest(
    manifest_path: Path,
    cache_dir: Path,
) -> dict[str, object]:
    """Embed frozen questions, selections, and non-selected distractor text."""

    entries: list[dict[str, object]] = []
    source_manifest: dict[str, object] | None = None
    for dataset in DATASETS:
        manifest, matched = load_matched_examples(manifest_path, dataset, cache_dir)
        source_manifest = manifest
        raw_by_id = {
            example.example_id: example for example in _examples(dataset, cache_dir)
        }
        for example in matched:
            raw = raw_by_id[example.example_id]
            selected_ids = set(example.selected_document_ids)
            distractors = [
                f"Document: {document.title}\n{document.text}"
                for document in raw.documents
                if document.document_id not in selected_ids
            ]
            entries.append(
                {
                    "dataset": dataset,
                    "seed": example.seed,
                    "example_id": example.example_id,
                    "question": example.question,
                    "answer": example.answer,
                    "selection_id": example.selection.selection_id,
                    "selected_document_ids": list(example.selected_document_ids),
                    "selected_source_sha256": example.selected_source_sha256,
                    "selected_source": example.selected_source,
                    "distractor_source": "\n\n".join(distractors),
                    "evidence_recall_at_4": example.evidence_recall,
                }
            )
    if source_manifest is None:
        raise RuntimeError("No datasets were loaded for the portable manifest.")
    return {
        "schema_version": "1.0",
        "cohort": "paper6_cross_engine_portable_e0_v1",
        "source_cohort": source_manifest["cohort"],
        "selection_policy": source_manifest["selection_policy"],
        "full_context_policy": "selected_documents_first_then_nonselected_documents",
        "entry_count": len(entries),
        "datasets": list(DATASETS),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/papers/shared/results/matched_e0_e2_qa_manifest.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/papers/shared/results/portable_e0_qa_manifest.json"),
    )
    args = parser.parse_args()
    payload = build_portable_manifest(args.manifest, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "entry_count": payload["entry_count"]}))


if __name__ == "__main__":
    main()
