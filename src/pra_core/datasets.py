import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .references import ReferenceTable


@dataclass
class DatasetExample:
    id: str
    prompt: str
    target: str
    refs: dict[str, str]
    summaries: dict[str, str]
    reference_table: ReferenceTable
    expected_ref_ids: list[int] = field(default_factory=list)
    expected_anchors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_documents(stage_path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(Path(stage_path) / "documents.jsonl")


def load_references(stage_path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(Path(stage_path) / "references.jsonl")


def load_questions(stage_path: str | Path) -> list[dict[str, Any]]:
    return load_jsonl(Path(stage_path) / "questions.jsonl")


def reference_table_from_rows(reference_rows: list[dict[str, Any]]) -> ReferenceTable:
    table = ReferenceTable()
    for row in reference_rows:
        table.register(
            uri=row["uri"],
            summary=row.get("summary"),
            metadata=row.get("metadata") or {},
            id=row.get("id"),
            token=row.get("token"),
        )
    return table


def build_examples(
    documents: list[dict[str, Any]],
    references: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    max_examples: int | None = None,
) -> list[DatasetExample]:
    docs_by_uri = {row["uri"]: row for row in documents}
    global_table = reference_table_from_rows(references)
    examples = []
    for row in questions[:max_examples]:
        table = reference_table_from_rows(references)
        refs: dict[str, str] = {}
        summaries: dict[str, str] = {}
        for handle in table.all():
            doc = docs_by_uri.get(handle.uri)
            refs[handle.uri] = doc.get("text", "") if doc else ""
            summaries[handle.uri] = handle.summary or (doc.get("summary", "") if doc else "")

        examples.append(
            DatasetExample(
                id=str(row.get("id", len(examples))),
                prompt=row["prompt"],
                target=f" {row['answer']}\n",
                refs=refs,
                summaries=summaries,
                reference_table=table,
                expected_ref_ids=[int(v) for v in row.get("expected_ref_ids", [])],
                expected_anchors=list(row.get("expected_anchors", [])),
                metadata={"question": row, "global_reference_table": global_table},
            )
        )
    return examples


def load_dataset(stage: str, data_dir: str | Path = "data", max_examples: int | None = None) -> list[DatasetExample]:
    stage_path = Path(data_dir) / stage
    if not stage_path.exists():
        raise FileNotFoundError(f"Dataset stage not found: {stage_path}")
    return build_examples(
        documents=load_documents(stage_path),
        references=load_references(stage_path),
        questions=load_questions(stage_path),
        max_examples=max_examples,
    )
