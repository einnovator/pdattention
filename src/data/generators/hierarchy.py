from .synthetic import save_jsonl


def generate(out_dir, n: int = 1) -> None:
    documents = [
        {
            "id": "root",
            "uri": "memory://project/demo",
            "title": "demo",
            "text": "The authentication policy is defined in <REF_1>.",
            "anchors": ["Authentication"],
            "reference_table": {"<REF_1>": "memory://project/demo#Authentication"},
        },
        {"id": "auth", "uri": "memory://project/demo#Authentication", "title": "Authentication", "text": "JWT expiration is 37 minutes.", "anchors": ["Authentication.details"]},
    ]
    references = [{"id": 1, "token": "<REF_1>", "uri": "memory://project/demo", "metadata": {"stage": 1}}]
    questions = [{"id": "q1", "prompt": "What is the JWT expiration? <REF_1>", "answer": "37 minutes", "expected_ref_ids": [1], "expected_anchors": ["Authentication"]}]
    save_jsonl(f"{out_dir}/documents.jsonl", documents)
    save_jsonl(f"{out_dir}/references.jsonl", references)
    save_jsonl(f"{out_dir}/questions.jsonl", questions)
