import torch

import pytest

from data.collators import PRACollator
from data.datamodules import PRADataModule
from data.datasets import (
    BooksDataset,
    CodeRepositoryDataset,
    DocumentationDataset,
    HierarchicalReferenceDataset,
    PRADataset,
    SyntheticMemoryQADataset,
    WikipediaDataset,
    WikiTextReferenceV2Dataset,
    WikiTextNativeKVFixedTargetDataset,
    generate_wikitext_nativekv_fixed_target_dataset,
    generate_wikitext_reference_dataset_v2,
)
from data.schemas import QuestionSample
from data.tokenizer import PRATokenizer
from pra_torch.config import PRAConfig, TrainConfig
from pra_torch.pra_train import train_pra_model


def test_dataset_classes_return_question_samples():
    dataset_classes = [
        SyntheticMemoryQADataset,
        HierarchicalReferenceDataset,
        CodeRepositoryDataset,
        DocumentationDataset,
        WikipediaDataset,
        BooksDataset,
    ]
    for dataset_cls in dataset_classes:
        dataset = dataset_cls("data", max_examples=1)
        assert isinstance(dataset, PRADataset)
        assert len(dataset) == 1
        assert isinstance(dataset[0], QuestionSample)
        assert dataset[0].references


def test_tokenizer_preserves_reference_tokens_and_dynamic_registration():
    tok = PRATokenizer(["hello <REF_1>"])
    assert tok.encode("<REF_1>") == [tok.stoi["<REF_1>"]]
    new_id = tok.register_reference_token("<REF_99>")
    assert tok.encode("<REF_99>") == [new_id]


def test_collator_builds_tensors_and_reference_tables():
    dataset = SyntheticMemoryQADataset("data", max_examples=2)
    tok = PRATokenizer([sample.question + sample.answer for sample in dataset])
    batch = PRACollator(tok, max_seq_len=64)([dataset[0], dataset[1]])
    assert set(batch) == {"input_ids", "labels", "attention_mask", "reference_tables", "metadata"}
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    assert batch["reference_tables"][0].find_by_token("<REF_1>") is not None
    prompt_length = len(tok.encode(dataset[0].question))
    assert batch["labels"][0, : prompt_length - 1].eq(0).all()
    assert batch["labels"][0, prompt_length - 1 :].ne(0).any()


def test_datamodule_returns_dataloader_batches():
    dm = PRADataModule("stage0_synthetic_memory", "data", max_examples=2, batch_size=2, max_seq_len=64).load()
    batch = next(iter(dm.train_loader()))
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["metadata"][0]["references"]


def test_datamodule_split_is_independent_of_global_training_seed():
    torch.manual_seed(1)
    first = PRADataModule(
        "stage0_synthetic_memory", "data", split_seed=1729
    ).load()
    torch.manual_seed(21)
    second = PRADataModule(
        "stage0_synthetic_memory", "data", split_seed=1729
    ).load()

    assert first.train_dataset.indices == second.train_dataset.indices
    assert first.val_dataset.indices == second.val_dataset.indices
    assert first.test_dataset.indices == second.test_dataset.indices


def test_wikitext_reference_v2_has_one_randomized_relevant_candidate(tmp_path, monkeypatch):
    documents = [
        {"text": " ".join(f"source{source}_token{index}" for index in range(120))}
        for source in range(5)
    ]
    monkeypatch.setattr(
        "data.datasets.load_wikitext_splits",
        lambda *args, **kwargs: {"train": documents},
    )

    generate_wikitext_reference_dataset_v2(
        tmp_path, max_examples=5, max_reference_parts=5, seed=1729
    )
    dataset = WikiTextReferenceV2Dataset(tmp_path)

    assert len(dataset) == 5
    assert all(len(sample.target_reference_ids) == 1 for sample in dataset)
    assert all(
        sample.target_reference_ids[0] in {reference.id for reference in sample.references}
        for sample in dataset
    )
    assert all(
        sample.metadata["row"]["generation_version"] == "wikitext_refs_v2"
        for sample in dataset
    )


@pytest.mark.parametrize("split_count", [2, 5])
def test_wikitext_reference_v2_supports_fixed_split_counts(
    tmp_path, monkeypatch, split_count
):
    documents = [
        {"text": " ".join(f"token{index}" for index in range(120))}
        for _ in range(3)
    ]
    monkeypatch.setattr(
        "data.datasets.load_wikitext_splits",
        lambda *args, **kwargs: {"train": documents},
    )

    generate_wikitext_reference_dataset_v2(
        tmp_path, max_examples=3, split_count=split_count, seed=1729
    )
    dataset = WikiTextReferenceV2Dataset(tmp_path)

    assert all(len(sample.references) == split_count - 1 for sample in dataset)
    assert all(sample.metadata["row"]["split_count"] == split_count for sample in dataset)
    assert all(
        sample.metadata["row"]["reference_count"] == split_count - 1
        for sample in dataset
    )


def test_native_kv_split_counts_preserve_the_exact_local_target(tmp_path, monkeypatch):
    documents = [
        {"text": " ".join(f"source{source}_word{index}" for index in range(192))}
        for source in range(3)
    ]
    monkeypatch.setattr(
        "data.datasets.load_wikitext_splits",
        lambda *args, **kwargs: {"train": documents},
    )

    snapshots = []
    for split_count in (2, 3, 5, 8, 16, 32, 64):
        generate_wikitext_nativekv_fixed_target_dataset(
            tmp_path, max_examples=3, split_count=split_count, seed=1729
        )
        dataset = WikiTextNativeKVFixedTargetDataset(tmp_path)
        sample = dataset[0]
        snapshots.append(
            (
                sample.question,
                sample.answer,
                sample.metadata["row"]["fixed_target_id"],
            )
        )
        assert len(sample.references) == split_count - 1
        assert "<REF_" not in sample.question
        assert " ".join(ref.metadata["text"] for ref in sample.references) == " ".join(
            documents[sample.metadata["row"]["source_entry_id"]]["text"].split()[:-28]
        )

    assert len(set(snapshots)) == 1


def test_synthetic_native_kv_partitions_preserve_source_tail_target_and_evidence(tmp_path):
    from data.datasets import SyntheticNativeKVFixedTargetDataset
    from data.native_kv_benchmarks import generate_synthetic_native_kv_dataset

    snapshots = []
    for split_count in (2, 3, 5, 8, 16, 32, 64):
        root = tmp_path / f"split-{split_count}"
        generate_synthetic_native_kv_dataset(
            root, split_count=split_count, max_examples=4, seed=1729
        )
        dataset = SyntheticNativeKVFixedTargetDataset(root)
        sample = dataset[0]
        row = sample.metadata["row"]
        snapshots.append(
            (row["source_text"], sample.question, sample.answer, row["fixed_target_id"])
        )
        assert len(sample.references) == split_count - 1
        assert sample.target_reference_ids
        assert all(
            reference.metadata["is_evidence"]
            for reference in sample.references
            if reference.id in sample.target_reference_ids
        )
        assert " ".join(reference.metadata["text"] for reference in sample.references) == row[
            "source_text"
        ]

    assert len(set(snapshots)) == 1


def test_hotpotqa_adapter_balances_answers_and_preserves_evidence_partitions(tmp_path):
    from data.datasets import HotpotQANativeKVFixedTargetDataset
    from data.native_kv_benchmarks import (
        hotpotqa_native_kv_examples,
        write_native_kv_benchmark,
    )

    distractor = " ".join(f"distractor{index}" for index in range(150))
    rows = []
    for index, answer in enumerate(("yes", "yes", "no", "no")):
        rows.append(
            {
                "id": str(index),
                "question": f"Is statement {index} supported?",
                "answer": answer,
                "type": "comparison",
                "level": "easy",
                "supporting_facts": {"title": ["Evidence"], "sent_id": [0]},
                "context": {
                    "title": ["Evidence", "Distractor"],
                    "sentences": [
                        ["The record directly supports the claim."],
                        [distractor],
                    ],
                },
            }
        )
    examples = hotpotqa_native_kv_examples(rows, max_examples=4, seed=7)
    assert len(examples) == 4
    assert [example.answer for example in examples].count("yes") == 2
    assert [example.answer for example in examples].count("no") == 2

    snapshots = []
    for split_count in (2, 3, 5, 8, 16, 32, 64):
        root = tmp_path / f"hotpot-{split_count}"
        write_native_kv_benchmark(
            root,
            stage=HotpotQANativeKVFixedTargetDataset.stage,
            dataset_name="hotpotqa",
            split_count=split_count,
            examples=examples,
            generation_version="test",
        )
        sample = HotpotQANativeKVFixedTargetDataset(root)[0]
        row = sample.metadata["row"]
        snapshots.append(
            (row["source_text"], sample.question, sample.answer, row["fixed_target_id"])
        )
        assert len(sample.references) == split_count - 1
        assert sample.target_reference_ids
        assert " ".join(ref.metadata["text"] for ref in sample.references) == row["source_text"]

    assert len(set(snapshots)) == 1


def test_native_kv_scale_partitions_preserve_all_255_source_units(tmp_path):
    from data.datasets import HotpotQANativeKVFixedTargetDataset
    from data.native_kv_benchmarks import (
        EvidenceUnit,
        NativeKVBenchmarkExample,
        write_native_kv_benchmark,
    )

    example = NativeKVBenchmarkExample(
        id="scale-example",
        source_units=tuple(
            EvidenceUnit(f"unit-{index}", is_evidence=index in {0, 127})
            for index in range(255)
        ),
        question=" Question: Return the code. Answer:",
        answer="yes",
        metadata={"task_type": "scale_invariant"},
    )

    for split_count in (128, 256):
        root = tmp_path / f"scale-{split_count}"
        write_native_kv_benchmark(
            root,
            stage=HotpotQANativeKVFixedTargetDataset.stage,
            dataset_name="scale",
            split_count=split_count,
            examples=[example],
            generation_version="test-scale",
        )
        sample = HotpotQANativeKVFixedTargetDataset(root)[0]

        assert len(sample.references) == split_count - 1
        assert sum(len(ref.metadata["text"].split()) for ref in sample.references) == 255
        assert " ".join(ref.metadata["text"] for ref in sample.references) == sample.metadata[
            "row"
        ]["source_text"]
        assert sample.target_reference_ids


def test_qasper_adapter_uses_yes_no_evidence_and_balances_answers():
    from data.native_kv_benchmarks import qasper_native_kv_examples

    papers = {}
    for index, value in enumerate((True, True, False, False)):
        papers[str(index)] = {
            "title": f"Paper {index}",
            "abstract": " ".join(f"abstract{word}" for word in range(90)),
            "full_text": [],
            "qas": [
                {
                    "question": f"Question {index}?",
                    "question_id": f"q{index}",
                    "answers": [
                        {
                            "answer": {
                                "yes_no": value,
                                "evidence": ["The annotated evidence supports this response."],
                            }
                        }
                    ],
                }
            ],
        }
    examples = qasper_native_kv_examples(papers, max_examples=4, seed=11)
    assert len(examples) == 4
    assert [example.answer for example in examples].count("yes") == 2
    assert [example.answer for example in examples].count("no") == 2
    assert all(len(example.source_units) == 63 for example in examples)
    assert all(sum(unit.is_evidence for unit in example.source_units) == 2 for example in examples)
    assert all(example.metadata["original_question"].startswith("Question") for example in examples)


def test_training_loop_uses_dataloader(tmp_path):
    dm = PRADataModule("stage0_synthetic_memory", "data", max_examples=2, batch_size=1, max_seq_len=64).load()
    cfg = PRAConfig(
        vocab_size=dm.tokenizer.vocab_size,
        max_seq_len=64,
        batch_size=1,
        d_model=32,
        n_heads=4,
        n_layers=2,
        steps=1,
        device="cpu",
    )
    train_cfg = TrainConfig(
        experiment_name="smoke",
        output_dir=str(tmp_path),
        device="cpu",
        epochs=1,
        batch_size=1,
        max_seq_len=64,
        max_examples=2,
        use_tensorboard=False,
        eval_every_steps=99,
        save_every_steps=99,
        log_every_steps=99,
    )
    result = train_pra_model(
        cfg=cfg,
        train_config=train_cfg,
        datamodule=dm,
    )
    assert result["checkpoint_dir"].exists()
    assert result["model"] is not None
