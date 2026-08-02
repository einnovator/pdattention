import json

from data.datasets import WikiTextReferenceDataset
from data.language_modeling import TokenBlockDataset
from data.tokenizer import BPETokenizer


def test_bpe_tokenizer_uses_atomic_reference_special_tokens():
    tokenizer = BPETokenizer.train(
        ["alpha beta <REF_1>", "beta gamma <REF_2>"],
        vocab_size=32,
        reference_tokens=["<REF_1>", "<REF_2>"],
        min_frequency=1,
    )

    ids = tokenizer.encode("alpha <REF_1> beta")

    assert tokenizer.stoi["<REF_1>"] in ids
    assert "alpha" in tokenizer.decode(ids)
    assert "<REF_1>" in tokenizer.decode(ids)
    assert "beta" in tokenizer.decode(ids)
    assert BPETokenizer.from_json(tokenizer.to_json()).stoi == tokenizer.stoi


def test_token_block_dataset_builds_shifted_language_targets():
    dataset = TokenBlockDataset(list(range(10)), seq_len=4)

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert dataset[0]["labels"].tolist() == [1, 2, 3, 4]


def test_wikitext_reference_rows_select_sample_local_documents(tmp_path):
    stage = tmp_path / "wikitext2_references"
    stage.mkdir()
    documents = [
        {"uri": "wiki://a/1", "text": "alpha"},
        {"uri": "wiki://b/1", "text": "beta"},
    ]
    references = [
        {"id": 1, "uri": "wiki://a/1", "summary": "alpha"},
        {"id": 1, "uri": "wiki://b/1", "summary": "beta"},
    ]
    questions = [
        {"id": "a", "prompt": "<REF_1> continue", "answer": "A", "reference_uris": ["wiki://a/1"], "expected_ref_ids": [1]},
        {"id": "b", "prompt": "<REF_1> continue", "answer": "B", "reference_uris": ["wiki://b/1"], "expected_ref_ids": [1]},
    ]
    for name, rows in (("documents", documents), ("references", references), ("questions", questions)):
        content = "".join(json.dumps(row) + "\n" for row in rows)
        (stage / f"{name}.jsonl").write_text(content, encoding="utf-8")

    dataset = WikiTextReferenceDataset(tmp_path)

    assert [ref.uri for ref in dataset[0].references] == ["wiki://a/1"]
    assert [ref.uri for ref in dataset[1].references] == ["wiki://b/1"]
