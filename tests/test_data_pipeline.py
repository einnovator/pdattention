import torch

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
