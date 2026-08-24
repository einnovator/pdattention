"""Auditable 2WikiMultiHopQA and MuSiQue inputs for PRA routing studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MultiHopRoutingExample:
    """One fixed routing identity with exact source/evidence provenance.

    ``evidence`` contains source substrings with unique document/sentence markers.
    This lets the tokenizer map authored evidence to candidate chunks without a
    fuzzy text match or a dataset-specific post-processing heuristic.
    """

    dataset: str
    example_id: str
    split: str
    question: str
    answer: str
    source: str
    evidence: tuple[str, ...]
    source_segments: tuple[str, ...]
    evidence_segment_indices: tuple[int, ...]

    def as_feature_example(self) -> dict:
        """Return the mapping consumed by the frozen native-Q/K capture path."""

        return {
            "dataset": self.dataset,
            "id": self.example_id,
            "question": self.question,
            "answer": self.answer,
            "source": self.source,
            "evidence": list(self.evidence),
        }

    def identity_record(self) -> dict[str, str]:
        """Return the stable fields used in cohort hash manifests."""

        return {
            "dataset": self.dataset,
            "example_id": self.example_id,
            "split": self.split,
        }


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load_annotations(path: Path) -> list[dict]:
    return list(_read_jsonl(path))


def _twowiki_example(annotation: dict, row: dict) -> MultiHopRoutingExample:
    supporting = {
        (str(title), int(sentence_index))
        for title, sentence_index in row["supporting_facts"]
    }
    segments: list[str] = []
    evidence_indices: list[int] = []
    for document_index, (title, sentences) in enumerate(row["context"]):
        for sentence_index, sentence in enumerate(sentences):
            segment = (
                f"[doc={document_index:03d} sent={sentence_index:03d}] "
                f"{str(title)}: {str(sentence).strip()}"
            )
            if (str(title), sentence_index) in supporting:
                evidence_indices.append(len(segments))
            segments.append(segment)
    if not evidence_indices:
        raise ValueError(f"No 2Wiki evidence for {row['_id']}")
    evidence = tuple(segments[index] for index in evidence_indices)
    return MultiHopRoutingExample(
        dataset="2wikimultihopqa",
        example_id=str(row["_id"]),
        split=str(annotation["split"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        source="\n".join(segments),
        evidence=evidence,
        source_segments=tuple(segments),
        evidence_segment_indices=tuple(evidence_indices),
    )


def _musique_example(annotation: dict, row: dict) -> MultiHopRoutingExample:
    segments: list[str] = []
    evidence_indices: list[int] = []
    for document_index, paragraph in enumerate(row["paragraphs"]):
        segment = (
            f"[doc={document_index:03d}] {str(paragraph['title'])}: "
            f"{str(paragraph['paragraph_text']).strip()}"
        )
        if bool(paragraph["is_supporting"]):
            evidence_indices.append(len(segments))
        segments.append(segment)
    if not evidence_indices:
        raise ValueError(f"No MuSiQue evidence for {row['id']}")
    evidence = tuple(segments[index] for index in evidence_indices)
    return MultiHopRoutingExample(
        dataset="musique",
        example_id=str(row["id"]),
        split=str(annotation["split"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        source="\n".join(segments),
        evidence=evidence,
        source_segments=tuple(segments),
        evidence_segment_indices=tuple(evidence_indices),
    )


def load_multihop_routing_examples(
    annotations_path: Path,
    twowiki_path: Path,
    musique_path: Path,
) -> list[MultiHopRoutingExample]:
    """Load the frozen Paper 2.7 identities with their original evidence.

    The annotation artifact chooses validation/test identities. Raw dataset rows
    supply source documents and authored evidence labels; no labels are inferred
    from answers or model outputs.
    """

    annotations = _load_annotations(annotations_path)
    twowiki_rows = {
        str(row["_id"]): row
        for row in json.loads(twowiki_path.read_text(encoding="utf-8"))
    }
    musique_rows = {str(row["id"]): row for row in _read_jsonl(musique_path)}
    examples = []
    for annotation in annotations:
        dataset = str(annotation["dataset"])
        example_id = str(annotation["example_id"])
        if dataset == "2wikimultihopqa":
            example = _twowiki_example(annotation, twowiki_rows[example_id])
        elif dataset == "musique":
            example = _musique_example(annotation, musique_rows[example_id])
        else:
            raise ValueError(f"Unsupported routing dataset: {dataset}")
        if example.question != str(annotation["question"]):
            raise ValueError(f"Question mismatch for {dataset}/{example_id}")
        if any(example.source.count(text) != 1 for text in example.evidence):
            raise ValueError(f"Evidence is not uniquely aligned for {dataset}/{example_id}")
        examples.append(example)
    identities = [example.identity_record() for example in examples]
    if len({(row["dataset"], row["example_id"]) for row in identities}) != len(identities):
        raise ValueError("Duplicate identities in the multihop routing cohort")
    return examples


def cohort_manifest(examples: Iterable[MultiHopRoutingExample]) -> dict:
    """Summarize split provenance and produce a stable identity hash."""

    examples = list(examples)
    records = sorted(
        (example.identity_record() for example in examples),
        key=lambda row: (row["split"], row["dataset"], row["example_id"]),
    )
    validation = {
        (row["dataset"], row["example_id"])
        for row in records
        if row["split"] == "validation"
    }
    test = {
        (row["dataset"], row["example_id"])
        for row in records
        if row["split"] == "test"
    }
    if validation & test:
        raise ValueError("Validation and test identities overlap")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "examples": len(records),
        "identity_sha256": hashlib.sha256(payload).hexdigest(),
        "validation_test_identity_disjoint": True,
        "dataset_split_counts": {
            dataset: {
                split: sum(
                    row["dataset"] == dataset and row["split"] == split
                    for row in records
                )
                for split in ("validation", "test")
            }
            for dataset in sorted({row["dataset"] for row in records})
        },
        "evidence_item_distribution": {
            dataset: sorted(
                len(example.evidence)
                for example in examples
                if example.dataset == dataset
            )
            for dataset in sorted({example.dataset for example in examples})
        },
    }
