from .synthetic import save_jsonl


def generate(out_dir, n: int = 1) -> None:
    documents = [{"id": "doc", "uri": "docs://tool/install", "title": "Install", "text": "Set PRA_CACHE=1 to enable cache.", "summary": "Install docs.", "anchors": ["install.cache"]}]
    references = [{"id": 1, "token": "<REF_1>", "uri": "docs://tool/install", "summary": "Install docs.", "metadata": {"stage": 5}}]
    questions = [{"id": "q5", "prompt": "Which env var enables cache? <REF_1>", "answer": "PRA_CACHE", "expected_ref_ids": [1], "expected_anchors": ["install.cache"]}]
    save_jsonl(f"{out_dir}/documents.jsonl", documents)
    save_jsonl(f"{out_dir}/references.jsonl", references)
    save_jsonl(f"{out_dir}/questions.jsonl", questions)
