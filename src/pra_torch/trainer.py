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
    """Object API delegating to the generic engine and PRA-specific adapters."""

    def __init__(self, model_config: PRAConfig, train_config: TrainConfig, datamodule: PRADataModule):
        self.model_config = model_config
        self.config = train_config
        self.datamodule = datamodule
        self.resolver_config = ResolverServiceConfig.from_value(train_config.resolver_config)
        self.cache_config = CacheServiceConfig.from_value(train_config.cache_config)
        self._state = create_pra_training_state(model_config, train_config, datamodule)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self):
        return self._state.model

    @property
    def optimizer(self):
        return self._state.optimizer

    @property
    def scheduler(self):
        return self._state.scheduler

    @property
    def device(self) -> str:
        return self._state.device

    @property
    def run_dir(self) -> Path:
        return self._state.run_dir

    @property
    def trace_dir(self) -> Path:
        return self.run_dir / "traces"

    @property
    def checkpoint(self):
        return self._state.checkpoint

    @property
    def global_step(self) -> int:
        return self._state.global_step

    @property
    def batch_step(self) -> int:
        return self._state.batch_step

    @property
    def start_epoch(self) -> int:
        return self._state.start_epoch

    @property
    def best_val_loss(self) -> float:
        return self._state.best_val_loss

    def train(self) -> dict:
        result = train_pra_model(
            cfg=self.model_config,
            train_config=self.config,
            datamodule=self.datamodule,
            resolver_config=self.resolver_config,
            cache_config=self.cache_config,
            state=self._state,
        )
        return result["test_metrics"]

    def validate(self) -> dict:
        return self._evaluate(self.datamodule.val_loader(), "val")

    def test(self, save_predictions: str | None = None, save_traces: str | None = None) -> dict:
        return self._evaluate(
            self.datamodule.test_loader(),
            "test",
            save_predictions=save_predictions,
            save_traces=save_traces,
        )

    def resume(self, path: str | Path) -> None:
        resume_training_state(self._state, path)

    def save(self, path: str | Path, epoch: int) -> Path:
        return save_training_state(self._state, path, epoch)

    def _evaluate(
        self,
        loader,
        split: str,
        save_predictions: str | None = None,
        save_traces: str | None = None,
    ) -> dict:
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
