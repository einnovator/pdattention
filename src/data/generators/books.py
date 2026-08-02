from .synthetic import save_jsonl


def generate(out_dir, n: int = 1) -> None:
    documents = [{"id": "book", "uri": "book://demo/chapter1", "title": "Chapter 1", "text": "The hidden word is lantern.", "summary": "Chapter one.", "anchors": ["chapter1.word"]}]
    references = [{"id": 1, "token": "<REF_1>", "uri": "book://demo/chapter1", "summary": "Chapter one.", "metadata": {"stage": 4}}]
    questions = [{"id": "q4", "prompt": "What is the hidden word? <REF_1>", "answer": "lantern", "expected_ref_ids": [1], "expected_anchors": ["chapter1.word"]}]
    save_jsonl(f"{out_dir}/documents.jsonl", documents)
    save_jsonl(f"{out_dir}/references.jsonl", references)
    save_jsonl(f"{out_dir}/questions.jsonl", questions)
