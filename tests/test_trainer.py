import subprocess
import sys
import json
from pathlib import Path

from data.datamodules import PRADataModule
from pra_torch.checkpointing import load_checkpoint
from pra_torch.config import PRAConfig, TrainConfig
from pra_torch.logging import ClearMLLogger, TensorBoardLogger, WandBLogger
from pra_torch.trainer import PRAStandaloneTrainer


def tiny_trainer(tmp_path, resume_from=None, save_metric_plots=True, epochs=1):
    dm = PRADataModule(
        "stage0_synthetic_memory",
        "data",
        max_examples=3,
        batch_size=2,
        max_seq_len=64,
    ).load()
    model_cfg = PRAConfig(
        vocab_size=dm.tokenizer.vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        batch_size=2,
        device="cpu",
    )
    train_cfg = TrainConfig(
        experiment_name="test_run",
        output_dir=str(tmp_path),
        device="cpu",
        epochs=epochs,
        batch_size=2,
        max_seq_len=64,
        max_examples=3,
        eval_every_steps=100,
        save_every_steps=100,
        log_every_steps=100,
        resume_from=resume_from,
        use_tensorboard=True,
        save_metric_plots=save_metric_plots,
    )
    return PRAStandaloneTrainer(model_cfg, train_cfg, dm)


def test_trainer_initializes(tmp_path):
    trainer = tiny_trainer(tmp_path)
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.datamodule.train_loader() is not None


def test_one_training_epoch_and_validation_run(tmp_path):
    trainer = tiny_trainer(tmp_path)
    metrics = trainer.train()
    assert "test_loss" in metrics
    assert "answer_accuracy" in metrics
    assert trainer.checkpoint.latest_path.exists()
    assert trainer.checkpoint.best_path.exists()
    timing = metrics["timing_metrics"]
    assert timing["train_duration_seconds"] > 0
    assert timing["validation_duration_seconds"] > 0
    assert timing["processed_tokens"] > 0
    assert timing["optimizer_steps"] == trainer.global_step


def test_metric_history_tracks_every_batch_and_epoch(tmp_path):
    trainer = tiny_trainer(tmp_path, save_metric_plots=False, epochs=3)

    trainer.train()

    records = json.loads((trainer.run_dir / "metrics.json").read_text(encoding="utf-8"))["records"]
    batch_records = [record for record in records if record["split"] == "train_batch"]
    epoch_records = [record for record in records if record["split"] == "train_epoch"]
    val_epoch_records = [record for record in records if record["split"] == "val_epoch"]

    assert [record["step"] for record in batch_records] == [1, 2, 3]
    assert [record["epoch"] for record in batch_records] == [1.0, 2.0, 3.0]
    assert [record["step"] for record in epoch_records] == [1, 2, 3]
    assert [record["step"] for record in val_epoch_records] == [1, 2, 3]
    assert trainer.batch_step == 3


def test_partial_gradient_accumulation_applies_final_step(tmp_path):
    trainer = tiny_trainer(tmp_path)
    trainer.config.grad_accum_steps = 100

    trainer.train()

    assert trainer.global_step == 1


def test_checkpoint_save_load_and_resume(tmp_path):
    trainer = tiny_trainer(tmp_path)
    trainer.train()
    ckpt = load_checkpoint(trainer.checkpoint.best_path, trainer.model, trainer.optimizer, trainer.scheduler, map_location="cpu")
    assert "model" in ckpt
    resumed = tiny_trainer(tmp_path / "resume", resume_from=str(trainer.checkpoint.best_path))
    assert resumed.global_step >= 0
    assert resumed.batch_step == trainer.batch_step


def test_tensorboard_logger_creates_event_file(tmp_path):
    logger = TensorBoardLogger(tmp_path)
    logger.log_metrics({"loss": 1.0}, step=1, split="train")
    logger.close()
    assert any(path.name.startswith("events.out.tfevents") for path in Path(tmp_path).iterdir())


def test_metric_plot_persistence_can_be_disabled(tmp_path):
    trainer = tiny_trainer(tmp_path, save_metric_plots=False)

    trainer.train()

    assert (trainer.run_dir / "metrics.json").exists()
    assert (trainer.run_dir / "metrics.md").exists()
    assert not (trainer.run_dir / "plots").exists()


def test_optional_loggers_do_not_crash_when_missing(tmp_path):
    cfg = TrainConfig(output_dir=str(tmp_path), use_wandb=True, use_clearml=True)
    WandBLogger(cfg, enabled=True).log_metrics({"x": 1}, 1, "train")
    ClearMLLogger(cfg, enabled=True).log_metrics({"x": 1}, 1, "train")


def test_metrics_have_expected_keys(tmp_path):
    trainer = tiny_trainer(tmp_path)
    metrics = trainer.validate()
    expected = {
        "loss",
        "perplexity",
        "answer_accuracy",
        "reference_retrieval_accuracy",
        "expected_anchor_hit",
        "expansion_depth",
        "expanded_ref_count",
        "average_retrieved_tokens",
        "cache_hit_ratio",
        "latency",
        "examples_per_second",
        "tokens_per_second",
        "gpu_memory_allocated",
    }
    assert expected.issubset(metrics)


def test_test_writes_predictions_and_traces(tmp_path):
    trainer = tiny_trainer(tmp_path)
    predictions = tmp_path / "artifacts" / "predictions.jsonl"
    traces = tmp_path / "artifacts" / "traces.jsonl"

    metrics = trainer.test(str(predictions), str(traces))

    assert "test_loss" in metrics
    assert predictions.read_text(encoding="utf-8").strip()
    assert traces.read_text(encoding="utf-8").strip()


def test_eval_script_can_load_checkpoint(tmp_path):
    trainer = tiny_trainer(tmp_path)
    trainer.train()
    predictions = tmp_path / "predictions.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_standalone.py",
            "--config",
            "config/config.yml",
            "--model",
            "standalone_tiny",
            "--checkpoint",
            str(trainer.checkpoint.best_path),
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--predictions-jsonl",
            str(predictions),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert predictions.exists()
