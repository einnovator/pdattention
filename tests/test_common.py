import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from common.config import TrainConfig, deep_update, load_yaml_config
from common.train import default_batch_step, move_batch, train_model


class TinyLanguageModel(torch.nn.Module):
    """Minimal non-PRA model used to exercise the reusable training engine."""

    def __init__(self, vocab_size: int = 11, width: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.output = torch.nn.Linear(width, vocab_size)

    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))


def _batch():
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    return {
        "input_ids": input_ids,
        "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
        "attention_mask": torch.ones_like(input_ids),
        "metadata": [{"source": "first"}, {"source": "second"}],
    }


def test_common_import_does_not_load_pra_package():
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import common, sys; "
            "assert not any(name == 'pra_torch' or name.startswith('pra_torch.') "
            "for name in sys.modules)",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_move_batch_recurses_without_changing_metadata():
    batch = _batch()
    moved = move_batch(batch, "cpu")

    assert moved["input_ids"].device.type == "cpu"
    assert moved["metadata"] == batch["metadata"]


def test_default_language_model_step_uses_standard_ignore_index_without_mask():
    batch = _batch()
    batch.pop("attention_mask")
    batch["labels"][0, 0] = -100

    loss, metrics = default_batch_step(TinyLanguageModel(), batch, "cpu")

    assert torch.isfinite(loss)
    assert metrics == {"tokens": 5, "examples": 2}


def test_yaml_configs_merge_without_experiment_specific_code(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text("model:\n  width: 32\ntrain:\n  epochs: 2\n", encoding="utf-8")
    second.write_text("model:\n  depth: 4\ntrain:\n  epochs: 3\n", encoding="utf-8")

    config = load_yaml_config(first, second, base={"model": {"width": 16}})

    assert config == {"model": {"width": 32, "depth": 4}, "train": {"epochs": 3}}
    assert deep_update({"a": {"b": 1}}, {"a": {"c": 2}}) == {
        "a": {"b": 1, "c": 2}
    }


def test_common_training_loop_accepts_non_pra_model_and_custom_metrics(tmp_path):
    model = TinyLanguageModel()
    config = TrainConfig(
        experiment_name="common_test",
        output_dir=str(tmp_path),
        device="cpu",
        epochs=1,
        max_steps=1,
        use_tensorboard=False,
        save_metric_plots=False,
        log_every_steps=1,
    )

    def batch_step(current_model, batch, device):
        batch = move_batch(batch, device)
        logits = current_model(batch["input_ids"])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            batch["labels"].reshape(-1),
        )
        return loss, {
            "examples": batch["input_ids"].shape[0],
            "tokens": batch["attention_mask"].sum().item(),
            "metrics": {"experiment_specific_score": torch.tensor(0.75)},
        }

    result = train_model(
        model=model,
        train_config=config,
        train_loader=[_batch()],
        batch_step=batch_step,
    )

    assert result["global_step"] == 1
    records = json.loads(
        (tmp_path / "common_test" / "metrics.json").read_text(encoding="utf-8")
    )["records"]
    batch_record = next(record for record in records if record["split"] == "train_batch")
    assert batch_record["metrics"]["experiment_specific_score"] == 0.75


def test_pra_training_config_extends_common_config():
    from pra_torch.config import TrainConfig as PRATrainConfig

    config = PRATrainConfig(use_tensorboard=False)

    assert isinstance(config, TrainConfig)
    assert config.resolver_config.type == "in_memory"
    assert config.cache_config.type == "simple"
