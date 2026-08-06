"""Compatibility shell for the canonical functional engine in ``train.py``."""

from __future__ import annotations

from pathlib import Path

from data.datamodules import PRADataModule
from .config import CacheServiceConfig, PRAConfig, ResolverServiceConfig, TrainConfig
from .pra_train import (
    create_pra_training_state,
    evaluate_pra_model,
    train_pra_model,
)
from .train import (
    resume_training_state,
    save_training_state,
)


class PRAStandaloneTrainer:
    """Thin object API over functional training and PRA cache-aware adapters.

    This class owns no training algorithm. ``train.py`` manages generic state and
    optimization, while ``pra_train.py`` builds reference caches and metrics.
    The properties below expose that shared ``TrainingState`` for older callers.
    """

    def __init__(self, model_config: PRAConfig, train_config: TrainConfig, datamodule: PRADataModule):
        """Create reusable state without starting optimization."""
        self.model_config = model_config  # Decoder, routing, and cache policy.
        self.config = train_config  # Loop, data, logging, and checkpoint policy.
        self.datamodule = datamodule  # Tokenizer and train/validation/test loaders.
        self.resolver_config = ResolverServiceConfig.from_value(train_config.resolver_config)
        self.cache_config = CacheServiceConfig.from_value(train_config.cache_config)
        self._state = create_pra_training_state(model_config, train_config, datamodule)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        """Return the ``TinyPRAModel`` held by the functional training state."""
        return self._state.model

    @property
    def optimizer(self):
        """Return the optimizer owned by the functional engine."""
        return self._state.optimizer

    @property
    def scheduler(self):
        """Return the learning-rate scheduler owned by the functional engine."""
        return self._state.scheduler

    @property
    def device(self) -> str:
        """Return the resolved PyTorch device string."""
        return self._state.device

    @property
    def run_dir(self) -> Path:
        """Return the directory containing this experiment's artifacts."""
        return self._state.run_dir

    @property
    def trace_dir(self) -> Path:
        """Return the conventional location for per-example PRA traces."""
        return self.run_dir / "traces"

    @property
    def checkpoint(self):
        """Return the checkpoint-path helper owned by the training state."""
        return self._state.checkpoint

    @property
    def global_step(self) -> int:
        """Return completed optimizer updates."""
        return self._state.global_step

    @property
    def batch_step(self) -> int:
        """Return consumed dataloader batches, including accumulation steps."""
        return self._state.batch_step

    @property
    def start_epoch(self) -> int:
        """Return the epoch index restored from a checkpoint."""
        return self._state.start_epoch

    @property
    def best_val_loss(self) -> float:
        """Return the best validation loss observed by checkpoint selection."""
        return self._state.best_val_loss

    def train(self) -> dict:
        """Delegate full training/testing to ``train_pra_model``."""
        result = train_pra_model(
            cfg=self.model_config,
            train_config=self.config,
            datamodule=self.datamodule,
            resolver_config=self.resolver_config,
            cache_config=self.cache_config,
            state=self._state,
        )
        return {
            **result["test_metrics"],
            "timing_metrics": result["timing_metrics"],
        }

    def validate(self) -> dict:
        """Evaluate the validation loader with PRA traces/metrics in memory."""
        return self._evaluate(self.datamodule.val_loader(), "val")

    def test(self, save_predictions: str | None = None, save_traces: str | None = None) -> dict:
        """Evaluate the test loader and optionally persist predictions/traces."""
        return self._evaluate(
            self.datamodule.test_loader(),
            "test",
            save_predictions=save_predictions,
            save_traces=save_traces,
        )

    def resume(self, path: str | Path) -> None:
        """Restore model, optimizer, scheduler, and progress into shared state."""
        resume_training_state(self._state, path)

    def save(self, path: str | Path, epoch: int) -> Path:
        """Persist the shared functional state and PRA reproducibility metadata."""
        return save_training_state(self._state, path, epoch)

    def _evaluate(
        self,
        loader,
        split: str,
        save_predictions: str | None = None,
        save_traces: str | None = None,
    ) -> dict:
        """Run the common PRA evaluator and log metrics when the logger is open."""
        metrics = evaluate_pra_model(
            model=self.model,
            loader=loader,
            tokenizer=self.datamodule.tokenizer,
            train_config=self.config,
            device=self.device,
            split=split,
            resolver_config=self.resolver_config,
            cache_config=self.cache_config,
            save_predictions=save_predictions,
            save_traces=save_traces,
        )
        if not self._state.logger_closed:
            self._state.logger.log_metrics(metrics, self.global_step, split)
        return metrics
