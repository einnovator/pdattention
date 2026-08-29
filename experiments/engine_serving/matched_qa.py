"""Frozen-evidence utilities for the cross-engine E0/E2 QA benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from experiments.paper6_2_mlx.run_answer_quality_pressure import QAExample, _examples


DATASETS = ("qasper", "hotpotqa", "2wikimultihopqa")


@dataclass(frozen=True)
class MatchedQAExample:
    """One question with a frozen ordered set of selected source documents."""

    dataset: str
    example_id: str
    question: str
    answer: str
    selected_document_ids: tuple[str, ...]
    selected_source: str
    selected_source_sha256: str


def selected_source(example: QAExample, document_ids: Iterable[str]) -> str:
    """Materialize selected documents in their frozen routing order."""

    documents = {document.document_id: document for document in example.documents}
    selected = []
    for document_id in document_ids:
        try:
            document = documents[str(document_id)]
        except KeyError as error:
            raise ValueError(
                f"Unknown selected document {document_id!r} for {example.example_id}"
            ) from error
        selected.append(f"Document: {document.title}\n{document.text}")
    return "\n\n".join(selected)


def source_digest(source: str) -> str:
    """Return the stable identity checked by every engine adapter."""

    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def manifest_entries_from_rows(
    examples: Iterable[QAExample], rows: Iterable[Mapping[str, object]]
) -> list[dict[str, object]]:
    """Build manifest rows from one routed result without rerunning routing."""

    by_id = {example.example_id: example for example in examples}
    entries = []
    seen = set()
    for row in rows:
        if row.get("condition") != "routed_native":
            continue
        example_id = str(row["example_id"])
        if example_id in seen:
            continue
        seen.add(example_id)
        example = by_id[example_id]
        document_ids = tuple(str(value) for value in row["selected_document_ids"])
        source = selected_source(example, document_ids)
        entries.append(
            {
                "dataset": example.dataset,
                "example_id": example.example_id,
                "question": example.question,
                "answer": example.answer,
                "selected_document_ids": list(document_ids),
                "selected_source_sha256": source_digest(source),
                "selected_source_characters": len(source),
                "evidence_recall_at_4": float(row["evidence_recall_at_4"]),
            }
        )
    return sorted(entries, key=lambda row: str(row["example_id"]))


def build_manifest(
    result_dir: Path, cache_dir: Path, *, datasets: Iterable[str] = DATASETS
) -> dict[str, object]:
    """Freeze routed identities and source hashes for all requested datasets."""

    entries = []
    for dataset in datasets:
        payload = json.loads(
            (result_dir / f"routed_answer_quality_{dataset}.json").read_text(
                encoding="utf-8"
            )
        )
        examples = _examples(dataset, cache_dir)
        entries.extend(manifest_entries_from_rows(examples, payload["rows"]))
    return {
        "schema_version": "1.0",
        "cohort": "paper6_cross_engine_matched_e0_e2_v1",
        "selection_policy": "frozen Paper 6.2 SDK hybrid top-4 document routing",
        "datasets": list(datasets),
        "entry_count": len(entries),
        "entries": entries,
    }


def load_matched_examples(
    manifest_path: Path, dataset: str, cache_dir: Path
) -> tuple[dict[str, object], list[MatchedQAExample]]:
    """Reconstruct and hash-check one dataset's frozen selected sources."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = {example.example_id: example for example in _examples(dataset, cache_dir)}
    matched = []
    for entry in manifest["entries"]:
        if entry["dataset"] != dataset:
            continue
        example = examples[str(entry["example_id"])]
        document_ids = tuple(str(value) for value in entry["selected_document_ids"])
        source = selected_source(example, document_ids)
        digest = source_digest(source)
        if digest != entry["selected_source_sha256"]:
            raise RuntimeError(
                f"Frozen source mismatch for {dataset}/{example.example_id}: {digest}"
            )
        if example.question != entry["question"] or example.answer != entry["answer"]:
            raise RuntimeError(f"Dataset content changed for {dataset}/{example.example_id}")
        matched.append(
            MatchedQAExample(
                dataset=dataset,
                example_id=example.example_id,
                question=example.question,
                answer=example.answer,
                selected_document_ids=document_ids,
                selected_source=source,
                selected_source_sha256=digest,
            )
        )
    return manifest, matched
