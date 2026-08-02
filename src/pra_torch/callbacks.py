from dataclasses import dataclass
from pathlib import Path


@dataclass
class EarlyStopping:
    """Track validation improvements and request stop after patience expires."""

    patience: int | None
    best: float = float("inf")
    bad_steps: int = 0

    def update(self, value: float) -> bool:
        if self.patience is None:
            return False
        if value < self.best:
            self.best = value
            self.bad_steps = 0
            return False
        self.bad_steps += 1
        return self.bad_steps >= self.patience


class LearningRateMonitor:
    """Read the current learning rate from an optimizer."""

    def current_lr(self, optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])


class ProgressPrinter:
    """Format scalar metrics for compact progress output."""

    def format(self, metrics: dict) -> str:
        return " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (int, float)))


class ModelCheckpoint:
    """Manage conventional latest/best checkpoint paths."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_dir / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / "best.pt"
