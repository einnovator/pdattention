import json
from pathlib import Path


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def generate(out_dir: str | Path, n: int = 4) -> None:
    out_dir = Path(out_dir)
    animals = ["cat", "dog", "owl", "bee"][:n]
    documents, references, questions = [], [], []
    for i, animal in enumerate(animals, start=1):
        uri = f"memory://animal/{animal}"
        documents.append(
            {
                "id": f"doc{i}",
                "uri": uri,
                "title": animal,
                "text": f"The animal is {animal}. Secret code: CODE{i * 17}.",
                "summary": f"Facts about {animal}.",
                "anchors": [],
            }
        )
        references.append({"id": i, "token": f"<REF_{i}>", "uri": uri, "summary": f"Facts about {animal}.", "metadata": {"stage": 0}})
    questions.append({"id": "q0", "prompt": "Which animal hunts mice? <REF_1> <REF_2>", "answer": "cat", "expected_ref_ids": [1], "expected_anchors": []})
    save_jsonl(out_dir / "documents.jsonl", documents)
    save_jsonl(out_dir / "references.jsonl", references)
    save_jsonl(out_dir / "questions.jsonl", questions)
