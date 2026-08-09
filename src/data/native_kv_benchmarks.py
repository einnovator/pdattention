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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


NATIVE_KV_SPLIT_COUNTS = (2, 3, 5, 8, 16, 32, 64)


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

    if split_count not in NATIVE_KV_SPLIT_COUNTS:
        allowed = ", ".join(str(value) for value in NATIVE_KV_SPLIT_COUNTS)
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
