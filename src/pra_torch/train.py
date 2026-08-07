"""Compatibility re-exports for the model-agnostic engine in :mod:`common.train`.

New experiment code should import from ``common.train`` directly.
"""

from common.train import (
    TrainingState,
    close_training_state,
    create_training_state,
    default_batch_step,
    move_batch,
    resolve_device,
    resume_training_state,
    save_training_state,
    seed_everything,
    train_model,
    validated_extra_metrics,
)

__all__ = [
    "TrainingState",
    "close_training_state",
    "create_training_state",
    "default_batch_step",
    "move_batch",
    "resolve_device",
    "resume_training_state",
    "save_training_state",
    "seed_everything",
    "train_model",
    "validated_extra_metrics",
]
