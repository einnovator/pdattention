import hashlib
import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from pra_core.references import ReferenceTable
from .schemas import DatasetMetadata, QuestionSample, ReferenceSample


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows, returning an empty list when the file is absent."""
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class PRADataset(Dataset, ABC):
    """Base class for PRA datasets that return ``QuestionSample`` objects."""

    dataset_name = "pra"
    stage = "unknown"

    def __init__(self, data_dir: str | Path = "data", max_examples: int | None = None):
        self.data_dir = Path(data_dir)
        self.stage_path = self.data_dir / self.stage
        self.max_examples = max_examples
        self.metadata = DatasetMetadata(dataset_name=self.dataset_name, stage=self.stage)
        self.documents: list[dict[str, Any]] = []
        self.reference_rows: list[dict[str, Any]] = []
        self.question_rows: list[dict[str, Any]] = []
        self.samples: list[QuestionSample] = []
        self.load()

    def __len__(self) -> int:
        """Return the number of prepared question samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> QuestionSample:
        """Return one non-tensor sample for collation."""
        return self.samples[idx]

    def load(self) -> None:
        """Load stage JSONL files and materialize question samples."""
        self.documents = read_jsonl(self.stage_path / "documents.jsonl")
        self.reference_rows = read_jsonl(self.stage_path / "references.jsonl")
        self.question_rows = read_jsonl(self.stage_path / "questions.jsonl")
        if self.max_examples is not None:
            self.question_rows = self.question_rows[: self.max_examples]
        self.samples = self._build_samples()

    def build_reference_table(self, sample: QuestionSample | None = None) -> ReferenceTable:
        """Build a reference table for one sample or the full dataset."""
        table = ReferenceTable()
        refs = sample.references if sample is not None else self._reference_samples()
        for ref in refs:
            table.register(
                uri=ref.uri,
                summary=ref.summary,
                metadata=ref.metadata,
                id=ref.id,
                token=f"<REF_{ref.id}>",
            )
        return table

    def evaluate(self, predictions: list[str]) -> dict[str, float]:
        """Compute simple answer containment accuracy for predictions."""
        total = max(len(self.samples), 1)
        exact = sum(sample.answer.strip() in pred for sample, pred in zip(self.samples, predictions))
        return {"answer_accuracy": exact / total}

    @abstractmethod
    def generate(self, *args, **kwargs) -> None:
        """Generate source JSONL files for this dataset stage."""
        raise NotImplementedError

    def _reference_samples(self) -> list[ReferenceSample]:
        docs_by_uri = {row["uri"]: row for row in self.documents}
        refs = []
        for row in self.reference_rows:
            doc = docs_by_uri.get(row["uri"], {})
            metadata = dict(row.get("metadata") or {})
            metadata.setdefault("text", doc.get("text", ""))
            metadata.setdefault("title", doc.get("title"))
            metadata.setdefault("anchors", doc.get("anchors", []))
            metadata.setdefault("reference_table", dict(doc.get("reference_table") or {}))
            if metadata["reference_table"]:
                metadata.setdefault(
                    "documents",
                    {
                        uri: {
                            "text": value.get("text", ""),
                            "summary": value.get("summary"),
                            "reference_table": value.get("reference_table", {}),
                            "metadata": value.get("metadata", {}),
                            "version": value.get("version"),
                        }
                        for uri, value in docs_by_uri.items()
                    },
                )
            refs.append(
                ReferenceSample(
                    id=int(row["id"]),
                    uri=row["uri"],
                    summary=row.get("summary") or doc.get("summary"),
                    anchor=row.get("anchor"),
                    metadata=metadata,
                )
            )
        return refs

    def _build_samples(self) -> list[QuestionSample]:
        refs = self._reference_samples()
        refs_by_uri = {ref.uri: ref for ref in refs}
        samples = []
        for row in self.question_rows:
            row_ref_uris = [str(value) for value in row.get("reference_uris") or []]
            row_ref_ids = {int(value) for value in row.get("reference_ids") or []}
            sample_refs = refs
            if row_ref_uris:
                sample_refs = [refs_by_uri[uri] for uri in row_ref_uris if uri in refs_by_uri]
            elif row_ref_ids:
                sample_refs = [ref for ref in refs if ref.id in row_ref_ids]
            samples.append(
                QuestionSample(
                    id=str(row.get("id", len(samples))),
                    question=row["prompt"],
                    answer=str(row["answer"]),
                    references=sample_refs,
                    target_reference_ids=[int(v) for v in row.get("expected_ref_ids", [])],
                    target_reference_uris=[str(v) for v in row.get("expected_ref_uris", [])],
                    target_chunk_ids=[str(v) for v in row.get("expected_chunk_ids", [])],
                    target_chunk_spans=list(row.get("expected_chunk_spans", [])),
                    metadata={
                        "expected_anchors": list(row.get("expected_anchors", [])),
                        "row": row,
                        "dataset": self.metadata,
                    },
                )
            )
        return samples


class SyntheticMemoryQADataset(PRADataset):
    dataset_name = "synthetic_memory_qa"
    stage = "stage0_synthetic_memory"

    def generate(self, *args, **kwargs) -> None:
        from .generators.synthetic import generate

        generate(*args, **kwargs)


class HierarchicalReferenceDataset(PRADataset):
    dataset_name = "hierarchical_reference"
    stage = "stage1_hierarchical_synthetic"

    def generate(self, *args, **kwargs) -> None:
        from .generators.hierarchy import generate

        generate(*args, **kwargs)


class CodeRepositoryDataset(PRADataset):
    dataset_name = "code_repository"
    stage = "stage2_code_repos"

    def generate(self, *args, **kwargs) -> None:
        from .generators.code_repo import generate

        generate(*args, **kwargs)


class DocumentationDataset(PRADataset):
    dataset_name = "documentation"
    stage = "stage5_technical_docs"

    def generate(self, *args, **kwargs) -> None:
        from .generators.docs import generate

        generate(*args, **kwargs)


class WikipediaDataset(PRADataset):
    dataset_name = "wikipedia"
    stage = "stage3_wikipedia"

    def generate(self, *args, **kwargs) -> None:
        from .generators.wikipedia import generate

        generate(*args, **kwargs)


class BooksDataset(PRADataset):
    dataset_name = "books"
    stage = "stage4_books"

    def generate(self, *args, **kwargs) -> None:
        from .generators.books import generate

        generate(*args, **kwargs)


class GitHubRepositoryDataset(PRADataset):
    dataset_name = "github_repository"
    stage = "stage6_github_repos"

    def generate(self, *args, **kwargs) -> None:
        from .generators.code_repo import generate

        generate(*args, **kwargs)


class WikiTextReferenceDataset(PRADataset):
    dataset_name = "wikitext_reference_memory"
    stage = "wikitext2_references"

    def generate(self, *args, **kwargs) -> None:
        generate_wikitext_reference_dataset(*args, **kwargs)


class WikiTextReferenceV2Dataset(PRADataset):
    dataset_name = "wikitext_reference_memory_v2"
    stage = "wikitext2_references_v2"

    def generate(self, *args, **kwargs) -> None:
        generate_wikitext_reference_dataset_v2(*args, **kwargs)


class WikiTextNativeKVFixedTargetDataset(PRADataset):
    """WikiText history partitions with an invariant direct tail and target."""

    dataset_name = "wikitext_native_kv_fixed_target"
    stage = "wikitext2_nativekv_fixed_target"

    def generate(self, *args, **kwargs) -> None:
        generate_wikitext_nativekv_fixed_target_dataset(*args, **kwargs)


class SyntheticNativeKVFixedTargetDataset(PRADataset):
    """Controlled key/value retrieval with invariant source, tail, and target."""

    dataset_name = "synthetic_native_kv"
    stage = "synthetic_nativekv_fixed_target"

    def generate(self, *args, **kwargs) -> None:
        from .native_kv_benchmarks import generate_synthetic_native_kv_dataset

        generate_synthetic_native_kv_dataset(*args, **kwargs)


class HotpotQANativeKVFixedTargetDataset(PRADataset):
    """Balanced HotpotQA yes/no evidence with invariant source and target."""

    dataset_name = "hotpotqa_native_kv"
    stage = "hotpotqa_nativekv_fixed_target"

    def generate(self, *args, **kwargs) -> None:
        from .native_kv_benchmarks import generate_hotpotqa_native_kv_dataset

        generate_hotpotqa_native_kv_dataset(*args, **kwargs)


class QASPERNativeKVFixedTargetDataset(PRADataset):
    """QASPER evidence spans with an invariant answer-code transport target."""

    dataset_name = "qasper_native_kv"
    stage = "qasper_nativekv_fixed_target"

    def generate(self, *args, **kwargs) -> None:
        from .native_kv_benchmarks import generate_qasper_native_kv_dataset

        generate_qasper_native_kv_dataset(*args, **kwargs)


DATASET_REGISTRY = {
    SyntheticMemoryQADataset.stage: SyntheticMemoryQADataset,
    HierarchicalReferenceDataset.stage: HierarchicalReferenceDataset,
    CodeRepositoryDataset.stage: CodeRepositoryDataset,
    DocumentationDataset.stage: DocumentationDataset,
    WikipediaDataset.stage: WikipediaDataset,
    BooksDataset.stage: BooksDataset,
    GitHubRepositoryDataset.stage: GitHubRepositoryDataset,
    WikiTextReferenceDataset.stage: WikiTextReferenceDataset,
    WikiTextReferenceV2Dataset.stage: WikiTextReferenceV2Dataset,
    WikiTextNativeKVFixedTargetDataset.stage: WikiTextNativeKVFixedTargetDataset,
    SyntheticNativeKVFixedTargetDataset.stage: SyntheticNativeKVFixedTargetDataset,
    HotpotQANativeKVFixedTargetDataset.stage: HotpotQANativeKVFixedTargetDataset,
    QASPERNativeKVFixedTargetDataset.stage: QASPERNativeKVFixedTargetDataset,
}


def dataset_class_for_stage(stage: str):
    """Resolve a dataset stage name to its dataset class."""
    if stage not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset stage: {stage}")
    return DATASET_REGISTRY[stage]


def load_wikitext_splits(
    dataset_name: str = "wikitext-2-raw-v1",
    *,
    cache_dir: str | Path | None = None,
):
    """Load WikiText-2 or WikiText-103 from the Salesforce Hugging Face dataset."""
    if dataset_name not in {"wikitext-2-raw-v1", "wikitext-103-raw-v1"}:
        raise ValueError(f"Unsupported WikiText dataset: {dataset_name}")
    from datasets import load_dataset

    return load_dataset("Salesforce/wikitext", dataset_name, cache_dir=str(cache_dir) if cache_dir else None)


def wikitext_documents(split, *, max_documents: int | None = None) -> list[str]:
    """Return non-empty WikiText rows as normalized document strings."""
    documents = [str(row["text"]).strip() for row in split if str(row["text"]).strip()]
    return documents[:max_documents] if max_documents is not None else documents


def generate_wikitext_reference_dataset(
    data_dir: str | Path,
    *,
    dataset_name: str = "wikitext-2-raw-v1",
    max_examples: int = 128,
    max_reference_parts: int = 5,
    split_count: int | None = None,
    seed: int = 7,
    min_words: int = 80,
    cache_dir: str | Path | None = None,
) -> Path:
    """Create PRA examples whose final split references the preceding text splits.

    ``split_count`` fixes the total number of parts, including the final tail. When
    omitted, the legacy mixed dataset cycles through one to ``max_reference_parts``
    references per example.
    """
    if split_count is not None and split_count < 2:
        raise ValueError("split_count must include at least one reference and one tail")
    if max_reference_parts < 1:
        raise ValueError("max_reference_parts must be at least 1")
    data_dir = Path(data_dir)
    stage_dir = data_dir / WikiTextReferenceDataset.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    splits = load_wikitext_splits(
        dataset_name, cache_dir=cache_dir or data_dir / ".hf_cache"
    )
    candidates = [
        text for text in wikitext_documents(splits["train"]) if len(text.split()) >= min_words
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    documents: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []

    for example_index, text in enumerate(candidates[:max_examples]):
        words = text.split()
        part_count = split_count or (2 + example_index % max_reference_parts)
        reference_count = part_count - 1
        part_size = max(1, len(words) // part_count)
        parts = [
            " ".join(words[index * part_size : (index + 1) * part_size])
            for index in range(reference_count)
        ]
        tail = " ".join(words[reference_count * part_size :])
        tail_words = tail.split()
        answer_size = max(8, min(16, len(tail_words) // 3))
        local_context_size = 0
        prompt_tail = "Continue the referenced WikiText passage:"
        answer = " ".join(tail_words[-answer_size:])
        ref_uris = []
        for local_id, part in enumerate(parts, start=1):
            uri = f"wikitext://{dataset_name}/{example_index}/part-{local_id}"
            ref_uris.append(uri)
            documents.append(
                {"id": f"wiki-{example_index}-{local_id}", "uri": uri, "title": f"WikiText {example_index} part {local_id}", "text": part}
            )
            references.append(
                {"id": local_id, "token": f"<REF_{local_id}>", "uri": uri, "metadata": {"dataset": dataset_name}}
            )
        ref_tokens = " ".join(f"<REF_{index}>" for index in range(1, reference_count + 1))
        questions.append(
            {
                "id": f"wikitext-ref-{example_index}",
                "prompt": f"{ref_tokens} {prompt_tail}",
                "answer": answer,
                "reference_uris": ref_uris,
                "expected_ref_ids": list(range(1, reference_count + 1)),
                "expected_anchors": [],
                "source_wikitext_entry": example_index,
                "part_boundaries": [index * part_size for index in range(part_count + 1)],
                "part_identifiers": [f"part-{index}" for index in range(1, reference_count + 1)],
                "candidate_reference_parts": list(range(1, reference_count + 1)),
                "intended_relevant_parts": list(range(1, reference_count + 1)),
                "tail_span": [len(words) - answer_size, len(words)],
                "token_distance": max(len(words) - reference_count * part_size, 0),
                "number_of_parts": part_count,
                "split_count": part_count,
                "reference_count": reference_count,
                "local_context_sufficient": False,
                "reference_relation": "indirect_natural_continuation",
            }
        )

    for filename, rows in (
        ("documents.jsonl", documents),
        ("references.jsonl", references),
        ("questions.jsonl", questions),
    ):
        with (stage_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    return stage_dir


def generate_wikitext_reference_dataset_v2(
    data_dir: str | Path,
    *,
    dataset_name: str = "wikitext-2-raw-v1",
    max_examples: int = 512,
    max_reference_parts: int = 5,
    split_count: int | None = None,
    seed: int = 1729,
    min_words: int = 80,
    cache_dir: str | Path | None = None,
) -> Path:
    """Create a selection-labelled reference-conditioned continuation set.

    ``split_count`` fixes the total number of parts, including the held-out tail.
    Omitting it retains the legacy mixed-reference-count generation policy.
    """
    if split_count is not None and split_count < 2:
        raise ValueError("split_count must include at least one reference and one tail")
    if max_reference_parts < 1:
        raise ValueError("max_reference_parts must be at least 1")
    data_dir = Path(data_dir)
    stage_dir = data_dir / WikiTextReferenceV2Dataset.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    splits = load_wikitext_splits(
        dataset_name, cache_dir=cache_dir or data_dir / ".hf_cache"
    )
    candidates = [
        (source_index, text)
        for source_index, text in enumerate(wikitext_documents(splits["train"]))
        if len(text.split()) >= min_words
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    documents: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []

    for example_index, (source_index, text) in enumerate(candidates[:max_examples]):
        words = text.split()
        part_count = split_count or (2 + example_index % max_reference_parts)
        reference_count = part_count - 1
        part_size = max(1, len(words) // part_count)
        parts = [
            " ".join(words[index * part_size : (index + 1) * part_size])
            for index in range(reference_count)
        ]
        tail_words = words[reference_count * part_size :]
        answer_size = min(12, max(4, len(tail_words) // 3))
        local_context_size = min(16, max(1, len(tail_words) - answer_size))
        local_context = " ".join(tail_words[:local_context_size])
        answer = " ".join(tail_words[local_context_size : local_context_size + answer_size])

        assigned_ids = list(range(1, reference_count + 1))
        rng.shuffle(assigned_ids)
        candidates_with_ids = list(enumerate(zip(parts, assigned_ids), start=1))
        relevant_id = candidates_with_ids[-1][1][1]
        rng.shuffle(candidates_with_ids)
        ref_uris = []
        candidate_ids = []
        for chronological_part, (part, reference_id) in candidates_with_ids:
            uri = f"wikitext://{dataset_name}/{source_index}/part-{chronological_part}"
            ref_uris.append(uri)
            candidate_ids.append(reference_id)
            documents.append(
                {
                    "id": f"wiki-{source_index}-{chronological_part}",
                    "uri": uri,
                    "title": f"WikiText source {source_index} part {chronological_part}",
                    "text": part,
                }
            )
            references.append(
                {
                    "id": reference_id,
                    "token": f"<REF_{reference_id}>",
                    "uri": uri,
                    "metadata": {
                        "dataset": dataset_name,
                        "source_entry_id": source_index,
                        "chronological_part": chronological_part,
                        "generation_version": "wikitext_refs_v2",
                    },
                }
            )

        ref_tokens = " ".join(f"<REF_{reference_id}>" for reference_id in candidate_ids)
        questions.append(
            {
                "id": f"wikitext-ref-v2-{source_index}",
                "prompt": (
                    f"{ref_tokens} Continue the referenced WikiText passage after this lead-in: "
                    f"{local_context}"
                ),
                "answer": answer,
                "reference_uris": ref_uris,
                "expected_ref_ids": [relevant_id],
                "expected_anchors": [],
                "source_entry_id": source_index,
                "part_ids": candidate_ids,
                "relevant_part_ids": [relevant_id],
                "target_part_id": f"tail-{source_index}",
                "num_parts": part_count,
                "split_count": part_count,
                "reference_count": reference_count,
                "reference_distance_tokens": local_context_size,
                "local_context_sufficient": False,
                "dependency_type": "indirect_continuation",
                "generation_version": "wikitext_refs_v2",
                "candidate_order_randomized": True,
            }
        )

    for filename, rows in (
        ("documents.jsonl", documents),
        ("references.jsonl", references),
        ("questions.jsonl", questions),
    ):
        with (stage_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    return stage_dir


def generate_wikitext_nativekv_fixed_target_dataset(
    data_dir: str | Path,
    *,
    split_count: int,
    dataset_name: str = "wikitext-2-raw-v1",
    max_examples: int = 128,
    seed: int = 1729,
    min_words: int = 160,
    local_tail_words: int = 16,
    target_words: int = 12,
    cache_dir: str | Path | None = None,
) -> Path:
    """Partition only displaced history while holding prompt and target fixed.

    ``split_count`` includes the direct tail, so it creates ``split_count - 1``
    metadata-only implicit history references. Across calls with the same source,
    seed, and token budgets, the local prompt, answer, source rows, and evaluation
    mask are identical. Only boundaries inside the historical prefix change.
    """
    if split_count not in {2, 3, 5, 8, 16, 32, 64}:
        raise ValueError("split_count must be one of 2, 3, 5, 8, 16, 32, or 64")
    if local_tail_words <= 0 or target_words <= 0:
        raise ValueError("local_tail_words and target_words must be positive")
    data_dir = Path(data_dir)
    stage_dir = data_dir / WikiTextNativeKVFixedTargetDataset.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    splits = load_wikitext_splits(
        dataset_name, cache_dir=cache_dir or data_dir / ".hf_cache"
    )
    candidates = [
        (source_index, text)
        for source_index, text in enumerate(wikitext_documents(splits["train"]))
        if len(text.split()) >= min_words
    ]
    random.Random(seed).shuffle(candidates)
    documents: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    reference_count = split_count - 1

    for source_index, text in candidates[:max_examples]:
        words = text.split()
        target = words[-target_words:]
        local_tail = words[-(target_words + local_tail_words) : -target_words]
        history = words[: -(target_words + local_tail_words)]
        if len(history) < reference_count:
            continue
        boundaries = [index * len(history) // reference_count for index in range(reference_count + 1)]
        reference_ids = list(range(1, reference_count + 1))
        reference_uris = []
        for part_index, reference_id in enumerate(reference_ids):
            start, end = boundaries[part_index], boundaries[part_index + 1]
            uri = (
                f"wikitext://{dataset_name}/{source_index}/"
                f"native-history-{part_index + 1}-of-{reference_count}"
            )
            reference_uris.append(uri)
            documents.append(
                {
                    "id": f"native-{source_index}-{part_index + 1}",
                    "uri": uri,
                    "title": f"WikiText source {source_index} history {part_index + 1}",
                    "text": " ".join(history[start:end]),
                }
            )
            references.append(
                {
                    "id": reference_id,
                    "token": f"<REF_{reference_id}>",
                    "uri": uri,
                    "metadata": {
                        "dataset": dataset_name,
                        "source_entry_id": source_index,
                        "implicit_reference": "#__head",
                        "history_word_span": [start, end],
                    },
                }
            )

        prompt = " ".join(local_tail)
        answer = " ".join(target)
        fixed_target_id = hashlib.sha256(
            f"{source_index}\n{prompt}\n{answer}".encode("utf-8")
        ).hexdigest()
        questions.append(
            {
                "id": f"wikitext-native-{source_index}",
                "prompt": prompt,
                "answer": answer,
                "reference_uris": reference_uris,
                "expected_ref_ids": reference_ids,
                "source_entry_id": source_index,
                "split_count": split_count,
                "reference_count": reference_count,
                "history_word_count": len(history),
                "history_boundaries": boundaries,
                "local_tail_word_count": len(local_tail),
                "target_word_count": len(target),
                "fixed_target_id": fixed_target_id,
                "implicit_reference_uri": "#__head",
                "generation_version": "wikitext_nativekv_fixed_target_v1",
            }
        )

    for filename, rows in (
        ("documents.jsonl", documents),
        ("references.jsonl", references),
        ("questions.jsonl", questions),
    ):
        with (stage_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    return stage_dir
