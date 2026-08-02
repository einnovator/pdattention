from pathlib import Path

from .plots import MetricsHistory


class ExperimentLogger:
    """Small logging interface shared by console and experiment trackers."""

    def log_metrics(self, metrics: dict, step: int, split: str) -> None:
        """Record scalar metrics for a split at a training step."""
        pass

    def log_text(self, name: str, text: str, step: int) -> None:
        """Record text artifacts such as configs or examples."""
        pass

    def log_config(self, config) -> None:
        """Record the experiment configuration."""
        pass

    def close(self) -> None:
        """Flush and release logger resources."""
        pass


class ConsoleLogger(ExperimentLogger):
    """Human-readable logger for local runs and tests."""

    def log_metrics(self, metrics: dict, step: int, split: str) -> None:
        formatted = " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))
        print(f"[{split} step={step}] {formatted}")

    def log_text(self, name: str, text: str, step: int) -> None:
        print(f"[text step={step}] {name}: {text}")

    def log_config(self, config) -> None:
        print(f"[config] {config}")


class TensorBoardLogger(ExperimentLogger):
    """TensorBoard logger with a file fallback when TensorBoard is unavailable."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        self.fallback_path = self.log_dir / "events.out.tfevents.fallback"
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.log_dir))
        except Exception:
            self.fallback_path.touch()

    def log_metrics(self, metrics: dict, step: int, split: str) -> None:
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{key}", value, step)
        else:
            with self.fallback_path.open("a", encoding="utf-8") as f:
                f.write(f"{step}\t{split}\t{metrics}\n")

    def log_text(self, name: str, text: str, step: int) -> None:
        if self.writer is not None:
            self.writer.add_text(name, text, step)

    def log_config(self, config) -> None:
        self.log_text("config", str(config), 0)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class WandBLogger(ExperimentLogger):
    """Optional Weights & Biases adapter that silently disables if unavailable."""

    def __init__(self, config, enabled: bool):
        self.run = None
        if enabled:
            try:
                import wandb

                self.run = wandb.init(project=getattr(config, "experiment_name", "pra"), config=config.__dict__)
            except Exception:
                self.run = None

    def log_metrics(self, metrics: dict, step: int, split: str) -> None:
        if self.run is not None:
            self.run.log({f"{split}/{k}": v for k, v in metrics.items()}, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


class ClearMLLogger(ExperimentLogger):
    """Optional ClearML adapter that silently disables if unavailable."""

    def __init__(self, config, enabled: bool):
        self.task = None
        if enabled:
            try:
                from clearml import Task

                self.task = Task.init(project_name="PRA", task_name=getattr(config, "experiment_name", "standalone"))
                self.task.connect(config.__dict__)
            except Exception:
                self.task = None

    def log_metrics(self, metrics: dict, step: int, split: str) -> None:
        if self.task is not None:
            logger = self.task.get_logger()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.report_scalar(split, key, value, iteration=step)


class MultiLogger(ExperimentLogger):
    """Fan-out logger that writes to multiple logger implementations."""

    def __init__(self, loggers: list[ExperimentLogger]):
        self.loggers = loggers

    def log_metrics(self, metrics: dict, step: int, split: str) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step, split)

    def log_text(self, name: str, text: str, step: int) -> None:
        for logger in self.loggers:
            logger.log_text(name, text, step)

    def log_config(self, config) -> None:
        for logger in self.loggers:
            logger.log_config(config)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


def build_logger(config, run_dir: str | Path) -> ExperimentLogger:
    """Create the configured experiment logger stack."""
    loggers: list[ExperimentLogger] = [
        ConsoleLogger(),
        MetricsHistory(run_dir, save_plots=getattr(config, "save_metric_plots", True)),
    ]
    if config.use_tensorboard:
        loggers.append(TensorBoardLogger(Path(run_dir) / "tensorboard"))
    loggers.append(WandBLogger(config, config.use_wandb))
    loggers.append(ClearMLLogger(config, config.use_clearml))
    return MultiLogger(loggers)
