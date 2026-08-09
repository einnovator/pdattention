"""Fixed-source datasets for isolating native-KV transport and routing.

Every split-count variant preserves the complete source text, local question,
answer, and example order.  Only the boundaries that turn source units into PRA
references change, so loss differences can be attributed to memory partitioning.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


NATIVE_KV_SPLIT_COUNTS = (2, 3, 5, 8, 16, 32, 64)
NATIVE_KV_SCALE_SPLIT_COUNTS = (*NATIVE_KV_SPLIT_COUNTS, 128, 256)


@dataclass(frozen=True)
class EvidenceUnit:
    """One indivisible source unit and whether it supports the answer."""

    text: str
    is_evidence: bool = False


@dataclass(frozen=True)
class NativeKVBenchmarkExample:
    """Dataset-neutral source, tail query, answer, and evidence annotations."""

    id: str
    source_units: tuple[EvidenceUnit, ...]
    question: str
    answer: str
    metadata: dict


def _stable_digest(*values: object) -> str:
    material = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _partition_units(
    units: Sequence[EvidenceUnit], reference_count: int
) -> list[list[EvidenceUnit]]:
    """Partition ordered units into exactly ``reference_count`` nonempty groups."""

    if reference_count < 1:
        raise ValueError("reference_count must be positive")
    if len(units) < reference_count:
        raise ValueError(
            f"Need at least {reference_count} source units, received {len(units)}"
        )
    boundaries = [
        index * len(units) // reference_count for index in range(reference_count + 1)
    ]
    return [list(units[boundaries[index] : boundaries[index + 1]]) for index in range(reference_count)]


def write_native_kv_benchmark(
    data_dir: str | Path,
    *,
    stage: str,
    dataset_name: str,
    split_count: int,
    examples: Iterable[NativeKVBenchmarkExample],
    generation_version: str,
) -> Path:
    """Write one invariant-target partition as project-native JSONL files."""

    if split_count not in NATIVE_KV_SCALE_SPLIT_COUNTS:
        allowed = ", ".join(str(value) for value in NATIVE_KV_SCALE_SPLIT_COUNTS)
        raise ValueError(f"split_count must be one of {allowed}")
    stage_dir = Path(data_dir) / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    reference_count = split_count - 1
    documents: list[dict] = []
    references: list[dict] = []
    questions: list[dict] = []

    for example_index, example in enumerate(examples):
        groups = _partition_units(example.source_units, reference_count)
        source_text = " ".join(unit.text for unit in example.source_units)
        reference_uris: list[str] = []
        target_reference_ids: list[int] = []
        target_reference_uris: list[str] = []
        target_chunk_spans: list[dict] = []
        cursor = 0
        for reference_id, group in enumerate(groups, start=1):
            text = " ".join(unit.text for unit in group)
            uri = f"benchmark://{dataset_name}/{example.id}/part-{reference_id}"
            reference_uris.append(uri)
            is_evidence = any(unit.is_evidence for unit in group)
            if is_evidence:
                target_reference_ids.append(reference_id)
                target_reference_uris.append(uri)
                target_chunk_spans.append(
                    {
                        "reference_uri": uri,
                        "unit_start": cursor,
                        "unit_end": cursor + len(group),
                    }
                )
            documents.append(
                {
                    "id": f"{example.id}-part-{reference_id}",
                    "uri": uri,
                    "title": f"{dataset_name} {example.id} part {reference_id}",
                    "text": text,
                }
            )
            references.append(
                {
                    "id": reference_id,
                    "token": f"<REF_{reference_id}>",
                    "uri": uri,
                    "metadata": {
                        "dataset": dataset_name,
                        "example_id": example.id,
                        "unit_span": [cursor, cursor + len(group)],
                        "is_evidence": is_evidence,
                        "split_count": split_count,
                    },
                }
            )
            cursor += len(group)

        fixed_target_id = _stable_digest(example.id, source_text, example.question, example.answer)
        questions.append(
            {
                "id": example.id,
                "prompt": example.question,
                "answer": example.answer,
                "reference_uris": reference_uris,
                "expected_ref_ids": target_reference_ids,
                "expected_ref_uris": target_reference_uris,
                "expected_chunk_spans": target_chunk_spans,
                "source_text": source_text,
                "source_unit_count": len(example.source_units),
                "split_count": split_count,
                "reference_count": reference_count,
                "fixed_target_id": fixed_target_id,
                "generation_version": generation_version,
                **example.metadata,
            }
        )

    for filename, rows in (
        ("documents.jsonl", documents),
        ("references.jsonl", references),
        ("questions.jsonl", questions),
    ):
        with (stage_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    (stage_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset_name,
                "split_count": split_count,
                "example_count": len(questions),
                "generation_version": generation_version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return stage_dir


def synthetic_native_kv_examples(
    *, max_examples: int = 320, seed: int = 1729
) -> list[NativeKVBenchmarkExample]:
    """Create random key/value lookup examples with 63 fixed source facts.

    Keys and assignments are regenerated per example.  The queried binary value
    is random and its marked fact occupies a fixed historical position.  A
    tail-only model can therefore do no better than answer-prior guessing, while
    every example has exact reference supervision.
    """

    all_keys = list(string.ascii_lowercase + string.ascii_uppercase + string.digits + "@")
    examples = []
    for example_index in range(max_examples):
        rng = random.Random(seed + example_index * 104_729)
        distractor_keys = [key for key in all_keys if key != "@"]
        rng.shuffle(distractor_keys)
        keys = ["@", *distractor_keys]
        answer = str(rng.randrange(2))
        target_position = keys.index("@")
        values = [str(rng.randrange(2)) for _ in keys]
        values[target_position] = answer
        units = tuple(
            EvidenceUnit(
                text=f"{key}{value}",
                is_evidence=position == target_position,
            )
            for position, (key, value) in enumerate(zip(keys, values))
        )
        examples.append(
            NativeKVBenchmarkExample(
                id=f"synthetic-native-{example_index:05d}",
                source_units=units,
                question="?",
                answer=answer,
                metadata={
                    "task_type": "fixed_position_key_value_retrieval",
                    "target_source_unit": target_position,
                    "local_context_sufficient": False,
                    "answer_space_size": 2,
                },
            )
        )
    return examples


def generate_synthetic_native_kv_dataset(
    data_dir: str | Path,
    *,
    split_count: int,
    max_examples: int = 320,
    seed: int = 1729,
) -> Path:
    """Generate the controlled fixed-source native-KV lookup benchmark."""

    return write_native_kv_benchmark(
        data_dir,
        stage="synthetic_nativekv_fixed_target",
        dataset_name="synthetic_native_kv",
        split_count=split_count,
        examples=synthetic_native_kv_examples(max_examples=max_examples, seed=seed),
        generation_version="synthetic_nativekv_fixed_target_v5",
    )


def hotpotqa_native_kv_examples(
    rows: Iterable[dict[str, Any]],
    *,
    max_examples: int,
    seed: int,
    source_unit_count: int = 63,
    max_evidence_units: int = 59,
) -> list[NativeKVBenchmarkExample]:
    """Convert balanced HotpotQA yes/no rows into fixed-source examples.

    Supporting sentences are retained in full and followed by an explicit
    ``AnswerCode yes|no`` evidence anchor. Distractor words fill the remaining
    source budget. This is a transport probe over natural HotpotQA context, not a
    claim that the small from-scratch model solves unrestricted multi-hop QA.
    """

    candidates: dict[str, list[NativeKVBenchmarkExample]] = {"yes": [], "no": []}
    for row in rows:
        answer = str(row.get("answer", "")).strip().lower()
        if answer not in candidates:
            continue
        supporting = {
            (str(title), int(sentence_id))
            for title, sentence_id in zip(
                row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
            )
        }
        evidence_words: list[str] = []
        distractor_words: list[str] = []
        supporting_sentences: list[str] = []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            for sentence_id, sentence in enumerate(sentences):
                words = str(sentence).strip().split()
                if (str(title), sentence_id) in supporting:
                    evidence_words.extend(words)
                    supporting_sentences.append(str(sentence).strip())
                else:
                    distractor_words.extend(words)
        if not evidence_words or len(evidence_words) > max_evidence_units:
            continue
        needed = source_unit_count - len(evidence_words) - 2
        if needed < 0 or len(distractor_words) < needed:
            continue
        row_seed = int(_stable_digest(seed, row.get("id", ""))[:16], 16)
        rng = random.Random(row_seed)
        # Sample distractors deterministically, then restore their document order.
        distractor_indices = sorted(rng.sample(range(len(distractor_words)), needed))
        selected_distractors = [distractor_words[index] for index in distractor_indices]
        units = tuple(
            [EvidenceUnit("AnswerCode", True), EvidenceUnit(answer, True)]
            + [EvidenceUnit(word, False) for word in evidence_words]
            + [EvidenceUnit(word, False) for word in selected_distractors]
        )
        candidates[answer].append(
            NativeKVBenchmarkExample(
                id=f"hotpotqa-{row['id']}",
                source_units=units,
                question=" Question: Return the AnswerCode. Answer:",
                answer=answer,
                metadata={
                    "task_type": "hotpotqa_answer_code_transport_probe",
                    "original_question": str(row["question"]).strip(),
                    "source_dataset_id": str(row["id"]),
                    "hotpot_type": str(row.get("type", "")),
                    "hotpot_level": str(row.get("level", "")),
                    "supporting_sentences": supporting_sentences,
                    "evidence_unit_count": len(evidence_words),
                    "local_context_sufficient": False,
                    "answer_space_size": 2,
                },
            )
        )

    per_label = min(len(candidates["yes"]), len(candidates["no"]), max_examples // 2)
    selected: list[NativeKVBenchmarkExample] = []
    for label_index, label in enumerate(("yes", "no")):
        values = candidates[label]
        random.Random(seed + label_index * 1_000_003).shuffle(values)
        selected.extend(values[:per_label])
    random.Random(seed + 2_000_003).shuffle(selected)
    return selected[:max_examples]


def generate_hotpotqa_native_kv_dataset(
    data_dir: str | Path,
    *,
    split_count: int,
    dataset_split: str,
    max_examples: int,
    seed: int,
    source_unit_count: int = 63,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download and write the balanced HotpotQA native-KV benchmark slice."""

    from datasets import load_dataset

    rows = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=dataset_split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    examples = hotpotqa_native_kv_examples(
        rows,
        max_examples=max_examples,
        seed=seed,
        source_unit_count=source_unit_count,
    )
    if len(examples) < max_examples:
        raise ValueError(
            f"HotpotQA {dataset_split} yielded {len(examples)} usable balanced examples; "
            f"requested {max_examples}"
        )
    return write_native_kv_benchmark(
        data_dir,
        stage="hotpotqa_nativekv_fixed_target",
        dataset_name="hotpotqa",
        split_count=split_count,
        examples=examples,
        generation_version="hotpotqa_nativekv_answer_code_v3",
    )


QASPER_ARCHIVE_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
QASPER_SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
}


def load_qasper_papers(
    dataset_split: str, *, cache_dir: str | Path | None = None
) -> dict[str, dict[str, Any]]:
    """Load official QASPER v0.3 JSON without its retired HF dataset script."""

    if dataset_split not in QASPER_SPLIT_FILES:
        raise ValueError(f"Unsupported QASPER split: {dataset_split}")
    root = Path(cache_dir or Path.home() / ".cache" / "qasper")
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "qasper-train-dev-v0.3.tgz"
    if not archive_path.exists():
        urllib.request.urlretrieve(QASPER_ARCHIVE_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.extractfile(QASPER_SPLIT_FILES[dataset_split])
        if member is None:
            raise FileNotFoundError(QASPER_SPLIT_FILES[dataset_split])
        return json.load(member)


def qasper_native_kv_examples(
    papers: dict[str, dict[str, Any]],
    *,
    max_examples: int,
    seed: int,
    source_unit_count: int = 63,
    max_evidence_units: int = 59,
) -> list[NativeKVBenchmarkExample]:
    """Convert QASPER yes/no annotations into natural-text transport probes."""

    candidates: dict[str, list[NativeKVBenchmarkExample]] = {"yes": [], "no": []}
    for paper_id, paper in papers.items():
        paper_words = str(paper.get("abstract", "")).split()
        for section in paper.get("full_text", []):
            for paragraph in section.get("paragraphs", []):
                paper_words.extend(str(paragraph).split())
        for qa in paper.get("qas", []):
            annotated = []
            for annotation in qa.get("answers", []):
                answer = annotation.get("answer", {})
                if answer.get("yes_no") is not None:
                    annotated.append((bool(answer["yes_no"]), list(answer.get("evidence") or [])))
            if not annotated:
                continue
            positive = sum(int(value) for value, _ in annotated)
            answer = "yes" if positive * 2 >= len(annotated) else "no"
            matching_evidence = [
                text
                for value, evidence in annotated
                if ("yes" if value else "no") == answer
                for text in evidence
                if str(text).strip()
            ]
            if not matching_evidence:
                continue
            evidence_words = min(matching_evidence, key=lambda value: len(value.split())).split()[
                :max_evidence_units
            ]
            if not evidence_words:
                continue
            needed = source_unit_count - len(evidence_words) - 2
            if needed < 0 or len(paper_words) < needed:
                continue
            question_id = str(qa.get("question_id", len(candidates[answer])))
            row_seed = int(_stable_digest(seed, paper_id, question_id)[:16], 16)
            rng = random.Random(row_seed)
            indices = sorted(rng.sample(range(len(paper_words)), needed))
            distractors = [paper_words[index] for index in indices]
            units = tuple(
                [EvidenceUnit("AnswerCode", True), EvidenceUnit(answer, True)]
                + [EvidenceUnit(word, False) for word in evidence_words]
                + [EvidenceUnit(word, False) for word in distractors]
            )
            candidates[answer].append(
                NativeKVBenchmarkExample(
                    id=f"qasper-{paper_id}-{question_id}",
                    source_units=units,
                    question=" Question: Return the AnswerCode. Answer:",
                    answer=answer,
                    metadata={
                        "task_type": "qasper_answer_code_transport_probe",
                        "paper_id": str(paper_id),
                        "paper_title": str(paper.get("title", "")),
                        "question_id": question_id,
                        "original_question": str(qa.get("question", "")),
                        "supporting_evidence": matching_evidence,
                        "evidence_unit_count": len(evidence_words),
                        "local_context_sufficient": False,
                        "answer_space_size": 2,
                    },
                )
            )
    per_label = min(len(candidates["yes"]), len(candidates["no"]), max_examples // 2)
    selected: list[NativeKVBenchmarkExample] = []
    for label_index, label in enumerate(("yes", "no")):
        values = candidates[label]
        random.Random(seed + label_index * 1_000_003).shuffle(values)
        selected.extend(values[:per_label])
    random.Random(seed + 2_000_003).shuffle(selected)
    return selected[:max_examples]


def generate_qasper_native_kv_dataset(
    data_dir: str | Path,
    *,
    split_count: int,
    dataset_split: str,
    max_examples: int,
    seed: int,
    source_unit_count: int = 63,
    cache_dir: str | Path | None = None,
) -> Path:
    """Write one fixed-target QASPER native-KV partition."""

    examples = qasper_native_kv_examples(
        load_qasper_papers(dataset_split, cache_dir=cache_dir),
        max_examples=max_examples,
        seed=seed,
        source_unit_count=source_unit_count,
    )
    if len(examples) < max_examples:
        raise ValueError(
            f"QASPER {dataset_split} yielded {len(examples)} usable balanced examples; "
            f"requested {max_examples}"
        )
    return write_native_kv_benchmark(
        data_dir,
        stage="qasper_nativekv_fixed_target",
        dataset_name="qasper",
        split_count=split_count,
        examples=examples,
        generation_version="qasper_nativekv_answer_code_v2",
    )
