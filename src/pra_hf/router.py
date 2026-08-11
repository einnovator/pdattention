"""Portable learned-router artifacts for PRA-HF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from pra_torch.hf import HFRoutingProjection


class PRARouter(HFRoutingProjection):
    """A frozen-backbone routing projection plus reproducibility metadata."""

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "adapter_model.pt"

    def __init__(
        self,
        input_width: int,
        routing_width: int = 128,
        architecture: str = "asymmetric_linear",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(input_width, routing_width, architecture)
        self.metadata = dict(metadata or {})

    def artifact_config(self) -> dict[str, Any]:
        """Return architecture and provenance without embedding model weights."""
        return {
            "format_version": 1,
            "adapter_type": self.architecture,
            "input_dim": self.input_width,
            "routing_dim": self.routing_width,
            **self.metadata,
        }

    def freeze(self) -> "PRARouter":
        """Switch to inference mode and disable gradients in place."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def save_pretrained(self, directory: str | Path) -> Path:
        """Save weights, machine-readable metadata, and a concise model card."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / self.WEIGHTS_NAME)
        config = self.artifact_config()
        (directory / self.CONFIG_NAME).write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        datasets = ", ".join(config.get("training_datasets", [])) or "not recorded"
        card = (
            "# PRA routing adapter\n\n"
            f"- Base model: `{config.get('base_model', 'not recorded')}`\n"
            f"- Family: `{config.get('model_family', 'not recorded')}`\n"
            f"- Routing representation: `{config.get('routing_representation', 'not recorded')}`\n"
            f"- Architecture: `{self.architecture}` ({self.input_width} -> {self.routing_width})\n"
            f"- Training data: {datasets}\n"
            f"- Parameters: {self.parameter_count:,}\n\n"
            "This artifact contains routing weights only, not base-model weights. "
            "See `config.json` for metrics, revisions, and reproducibility metadata.\n"
        )
        (directory / "README.md").write_text(card, encoding="utf-8")
        return directory

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "PRARouter":
        """Load a stable router artifact and freeze it for inference."""
        directory = Path(directory)
        config = json.loads((directory / cls.CONFIG_NAME).read_text(encoding="utf-8"))
        reserved = {"format_version", "adapter_type", "input_dim", "routing_dim"}
        router = cls(
            int(config["input_dim"]),
            int(config["routing_dim"]),
            str(config["adapter_type"]),
            metadata={key: value for key, value in config.items() if key not in reserved},
        ).to(device)
        state = torch.load(directory / cls.WEIGHTS_NAME, map_location=device, weights_only=True)
        router.load_state_dict(state)
        return router.freeze()

    @classmethod
    def from_experiment_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        device: torch.device | str = "cpu",
    ) -> "PRARouter":
        """Convert a legacy research checkpoint into the stable router object."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        router = cls(
            int(checkpoint["input_width"]),
            int(checkpoint["routing_width"]),
            str(checkpoint["architecture"]),
            metadata=metadata,
        ).to(device)
        router.load_state_dict(checkpoint["state_dict"])
        return router.freeze()
