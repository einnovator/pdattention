from .synthetic import save_jsonl


def generate(out_dir, n: int = 1) -> None:
    documents = [{"id": "repo", "uri": "repo://tinyapp", "title": "tinyapp", "text": "def authenticate_user(): pass", "anchors": ["auth.py.authenticate_user"]}]
    references = [{"id": 1, "token": "<REF_1>", "uri": "repo://tinyapp", "metadata": {"stage": 2}}]
    questions = [{"id": "q2", "prompt": "Which function authenticates users? <REF_1>", "answer": "authenticate_user", "expected_ref_ids": [1], "expected_anchors": ["auth.py.authenticate_user"]}]
    save_jsonl(f"{out_dir}/documents.jsonl", documents)
    save_jsonl(f"{out_dir}/references.jsonl", references)
    save_jsonl(f"{out_dir}/questions.jsonl", questions)
