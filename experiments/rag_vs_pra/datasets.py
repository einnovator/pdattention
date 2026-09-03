"""Dataset adapters for the RAG/PRA evaluation ladder."""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence

from pra_hf.rag_evaluation import RAGDocument, RAGQuestion


MULTIHOP_RAG_REVISION = "cde8e844af14b3012f20158abc2854fe8458212a"
MULTIHOP_RAG_DATASET_BLOB = "fcb9efe65c7730dd4126a42afb6c2c7e45721ebb"
MULTIHOP_RAG_CORPUS_BLOB = "bb98345ef3921312aad05fd117b5a3e39888de2c"
MULTIHOP_RAG_BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_multihop_rag(cache_dir: Path) -> tuple[Path, Path]:
    """Download the two official ODC-BY release files when absent."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    questions = cache_dir / "MultiHopRAG.json"
    corpus = cache_dir / "corpus.json"
    for path in (questions, corpus):
        if not path.exists():
            urllib.request.urlretrieve(f"{MULTIHOP_RAG_BASE}/{path.name}", path)
    return questions, corpus


def _document_id(url: str) -> str:
    return "multihoprag:" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]


def load_multihop_rag(
    cache_dir: Path,
) -> tuple[tuple[RAGDocument, ...], tuple[RAGQuestion, ...], Mapping[str, str]]:
    """Load the official corpus and resolve every evidence URL to a stable ID."""

    question_path, corpus_path = ensure_multihop_rag(cache_dir)
    corpus_rows = json.loads(corpus_path.read_text(encoding="utf-8"))
    question_rows = json.loads(question_path.read_text(encoding="utf-8"))
    documents = tuple(
        RAGDocument(
            document_id=_document_id(str(row["url"])),
            title=str(row["title"]),
            text=str(row["body"]),
            source=str(row.get("source", "")),
            uri=str(row["url"]),
            version=MULTIHOP_RAG_CORPUS_BLOB,
            mime="text/html",
            metadata={
                "author": row.get("author"),
                "category": row.get("category"),
                "published_at": row.get("published_at"),
            },
        )
        for row in corpus_rows
    )
    by_uri = {document.uri: document for document in documents}
    questions: list[RAGQuestion] = []
    for index, row in enumerate(question_rows):
        spans: dict[str, list[tuple[int, int]]] = {}
        gold_ids = []
        for evidence in row["evidence_list"]:
            document = by_uri[str(evidence["url"])]
            gold_ids.append(document.document_id)
            fact = str(evidence.get("fact", "")).strip()
            if fact:
                start = document.text.casefold().find(fact.casefold())
                if start >= 0:
                    spans.setdefault(document.document_id, []).append(
                        (start, start + len(fact))
                    )
        questions.append(
            RAGQuestion(
                example_id=f"multihoprag:{index:04d}",
                question=str(row["query"]),
                answers=(str(row["answer"]),),
                gold_document_ids=frozenset(gold_ids),
                gold_spans={key: tuple(value) for key, value in spans.items()},
                question_type=str(row.get("question_type", "")),
            )
        )
    metadata = {
        "dataset_revision": MULTIHOP_RAG_DATASET_BLOB,
        "corpus_revision": MULTIHOP_RAG_CORPUS_BLOB,
        "corpus_sha256": file_sha256(corpus_path),
        "questions_sha256": file_sha256(question_path),
        "upstream_revision": MULTIHOP_RAG_REVISION,
        "license": "ODC-BY-1.0",
    }
    return documents, tuple(questions), metadata


def controlled_fixture(
    *, seed: int = 11, document_count: int = 60
) -> tuple[tuple[RAGDocument, ...], tuple[RAGQuestion, ...], Mapping[str, str]]:
    """Create deterministic lookup, bridge, synthesis, and distractor cases."""

    if document_count < 20:
        raise ValueError("the controlled fixture requires at least 20 documents")
    facts = (
        ("Ada Rivera", "cobalt engine", "Lisbon", "2024"),
        ("Mina Chen", "amber telescope", "Oslo", "2018"),
        ("Jon Bell", "silver compiler", "Dublin", "2021"),
        ("Ravi Shah", "violet sensor", "Kyoto", "2016"),
        ("Lena Ortiz", "green battery", "Tallinn", "2023"),
    )
    documents: list[RAGDocument] = []
    questions: list[RAGQuestion] = []
    for index, (person, project, city, year) in enumerate(facts):
        person_id = f"fixture:person:{index}"
        launch_id = f"fixture:launch:{index}"
        outcome_id = f"fixture:outcome:{index}"
        documents.extend(
            (
                RAGDocument(
                    person_id,
                    f"Profile of {person}",
                    f"{person} designed the {project}. The work began in {city}.",
                    source="controlled_fixture",
                    uri=f"fixture://person/{index}",
                ),
                RAGDocument(
                    launch_id,
                    f"Launch record for {project}",
                    f"The {project} launched in {year} after trials in {city}.",
                    source="controlled_fixture",
                    uri=f"fixture://launch/{index}",
                ),
                RAGDocument(
                    outcome_id,
                    f"Outcome of {project}",
                    f"Reviewers called the {project} reliable and awarded it prize {index + 7}.",
                    source="controlled_fixture",
                    uri=f"fixture://outcome/{index}",
                ),
            )
        )
        questions.extend(
            (
                RAGQuestion(
                    f"fixture:lookup:{index}",
                    f"Where did work on the {project} begin?",
                    (city,),
                    frozenset({person_id}),
                    question_type="single_document_lookup",
                ),
                RAGQuestion(
                    f"fixture:bridge:{index}",
                    f"When did the project designed by {person} launch?",
                    (year,),
                    frozenset({person_id, launch_id}),
                    question_type="two_document_bridge",
                ),
                RAGQuestion(
                    f"fixture:synthesis:{index}",
                    f"What prize number was awarded to the project designed by {person}?",
                    (str(index + 7),),
                    frozenset({person_id, outcome_id}),
                    question_type="three_document_synthesis",
                ),
            )
        )
    rng = random.Random(seed)
    colors = ("cobalt", "amber", "silver", "violet", "green")
    cities = ("Lisbon", "Oslo", "Dublin", "Kyoto", "Tallinn")
    while len(documents) < document_count:
        index = len(documents)
        color = rng.choice(colors)
        city = rng.choice(cities)
        documents.append(
            RAGDocument(
                f"fixture:distractor:{index}",
                f"Archive note {index} about {color}",
                (
                    f"This long archival note discusses {color} materials in {city}. "
                    f"It does not describe an engine launch or project award. " * 8
                ),
                source="controlled_fixture",
                uri=f"fixture://distractor/{index}",
            )
        )
    corpus_sha = hashlib.sha256(
        "".join(document.fingerprint for document in documents).encode("ascii")
    ).hexdigest()
    metadata = {
        "dataset_revision": "controlled_fixture_v1",
        "corpus_revision": "controlled_fixture_v1",
        "corpus_sha256": corpus_sha,
        "questions_sha256": hashlib.sha256(
            "".join(question.example_id for question in questions).encode("ascii")
        ).hexdigest(),
        "upstream_revision": "local",
        "license": "project_fixture",
    }
    return tuple(documents), tuple(questions), metadata


def select_cohort(
    questions: Sequence[RAGQuestion], *, max_examples: int, seed: int
) -> tuple[RAGQuestion, ...]:
    """Choose a stable held-out cohort without changing dataset order globally."""

    if max_examples <= 0 or max_examples >= len(questions):
        return tuple(questions)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(questions)), max_examples))
    return tuple(questions[index] for index in indices)
