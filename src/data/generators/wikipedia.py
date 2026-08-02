from .synthetic import save_jsonl


def generate(out_dir, n: int = 1) -> None:
    documents = [{"id": "article", "uri": "wiki://Ada_Lovelace", "title": "Ada Lovelace", "text": "Ada Lovelace worked on the Analytical Engine.", "summary": "Ada Lovelace article.", "anchors": ["career"]}]
    references = [{"id": 1, "token": "<REF_1>", "uri": "wiki://Ada_Lovelace", "summary": "Ada Lovelace article.", "metadata": {"stage": 3}}]
    questions = [{"id": "q3", "prompt": "What engine did Ada work on? <REF_1>", "answer": "Analytical Engine", "expected_ref_ids": [1], "expected_anchors": ["career"]}]
    save_jsonl(f"{out_dir}/documents.jsonl", documents)
    save_jsonl(f"{out_dir}/references.jsonl", references)
    save_jsonl(f"{out_dir}/questions.jsonl", questions)
