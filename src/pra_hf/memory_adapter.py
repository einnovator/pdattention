"""Portable PRA-conditional memory-use adapter artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class PRAMemoryAdapter:
    """Conditional output-LoRA weights plus their inference contract.

    Unlike a routing adapter, this artifact does not decide which chunks are
    selected. It calibrates how already-selected native K/V affects the frozen
    decoder and is executed only on PRA-active attention calls.
    """

    CONFIG_NAME = "memory_adapter_config.json"
    WEIGHTS_NAME = "memory_adapter_model.pt"
    ADAPTER_TYPE = "conditional_output_lora"
    STATE_PREFIX = "pra_late_band_lora."

    def __init__(
        self,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        layer_ids: tuple[int, ...] | list[int],
        state_dict: dict[str, torch.Tensor],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.layer_ids = tuple(sorted({int(layer_id) for layer_id in layer_ids}))
        self.state_dict = {
            str(name): value.detach().cpu().clone() for name, value in state_dict.items()
        }
        self.metadata = dict(metadata or {})
        self._validate()

    def _validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("Conditional memory-adapter rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("Conditional memory-adapter alpha must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Conditional memory-adapter dropout must be in [0, 1).")
        if not self.layer_ids:
            raise ValueError("Conditional memory adapter requires at least one layer.")
        if not self.state_dict:
            raise ValueError("Conditional memory adapter has no weights.")
        unexpected = [
            name for name in self.state_dict if not name.startswith(self.STATE_PREFIX)
        ]
        if unexpected:
            raise ValueError(f"Memory adapter contains non-LoRA tensors: {unexpected[:3]}")
        if any(not isinstance(value, torch.Tensor) for value in self.state_dict.values()):
            raise TypeError("Memory-adapter state values must be tensors.")

    @property
    def parameter_count(self) -> int:
        """Return the number of stored conditional LoRA scalars."""

        return sum(int(value.numel()) for value in self.state_dict.values())

    def artifact_config(self) -> dict[str, Any]:
        """Return JSON-compatible architecture and provenance metadata."""

        return {
            "format_version": 1,
            "adapter_type": self.ADAPTER_TYPE,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "layer_ids": list(self.layer_ids),
            "parameter_count": self.parameter_count,
            **self.metadata,
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        """Write a small Hugging Face-style conditional adapter directory."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict, directory / self.WEIGHTS_NAME)
        config = self.artifact_config()
        (directory / self.CONFIG_NAME).write_text(
            json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
        )
        card = (
            "# PRA conditional memory-use adapter\n\n"
            f"- Base model: `{config.get('base_model', 'not recorded')}`\n"
            f"- Base revision: `{config.get('base_model_revision', 'not recorded')}`\n"
            f"- Type: `{self.ADAPTER_TYPE}`\n"
            f"- Rank / alpha: {self.rank} / {self.alpha:g}\n"
            f"- PRA layers: {', '.join(str(value) for value in self.layer_ids)}\n"
            f"- Parameters: {self.parameter_count:,}\n\n"
            "These weights calibrate selected native K/V only. They contain neither "
            "base-model nor routing weights and are structurally bypassed when PRA is off.\n"
        )
        (directory / "README.md").write_text(card, encoding="utf-8")
        return directory

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "PRAMemoryAdapter":
        """Load one packaged conditional adapter without applying it yet."""

        directory = Path(directory)
        config = json.loads((directory / cls.CONFIG_NAME).read_text(encoding="utf-8"))
        if config.get("adapter_type") != cls.ADAPTER_TYPE:
            raise ValueError(f"Unsupported memory adapter type: {config.get('adapter_type')}")
        reserved = {
            "format_version",
            "adapter_type",
            "rank",
            "alpha",
            "dropout",
            "layer_ids",
            "parameter_count",
        }
        state = torch.load(
            directory / cls.WEIGHTS_NAME,
            map_location=device,
            weights_only=True,
        )
        adapter = cls(
            rank=int(config["rank"]),
            alpha=float(config["alpha"]),
            dropout=float(config["dropout"]),
            layer_ids=config["layer_ids"],
            state_dict=state,
            metadata={key: value for key, value in config.items() if key not in reserved},
        )
        recorded = config.get("parameter_count")
        if recorded is not None and int(recorded) != adapter.parameter_count:
            raise ValueError("Memory-adapter parameter count does not match its weights.")
        return adapter

    @classmethod
    def from_experiment_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        layer_ids: tuple[int, ...] | list[int],
        alpha: float | None = None,
        dropout: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> "PRAMemoryAdapter":
        """Convert a last-14 experiment checkpoint into the public artifact."""

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        variant = payload.get("variant", {})
        rank = int(variant.get("lora_rank", 0))
        if rank <= 0 or int(variant.get("residual_width", 0)):
            raise ValueError("Checkpoint must contain a LoRA-only PRA variant.")
        state = {
            name: value
            for name, value in payload.get("state_dict", {}).items()
            if name.startswith(cls.STATE_PREFIX)
        }
        provenance = dict(metadata or {})
        provenance.setdefault("training_seed", payload.get("seed"))
        provenance.setdefault("training_report", payload.get("training", {}))
        return cls(
            rank=rank,
            alpha=float(rank if alpha is None else alpha),
            dropout=dropout,
            layer_ids=layer_ids,
            state_dict=state,
            metadata=provenance,
        )

    def apply(self, handle) -> None:
        """Configure and load the matching lazy conditional-LoRA bank."""

        available_layers = tuple(handle.late_band_lora.layer_ids)
        if self.layer_ids != available_layers:
            raise ValueError(
                f"Memory adapter expects layers {self.layer_ids}, received {available_layers}."
            )
        handle.configure_late_band_lora(
            self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            reset=True,
        )
        named = dict(handle.model.named_parameters())
        active = {
            name
            for name, parameter in named.items()
            if name.startswith(self.STATE_PREFIX) and parameter.requires_grad
        }
        stored = set(self.state_dict)
        if active != stored:
            missing = sorted(active - stored)
            unexpected = sorted(stored - active)
            raise ValueError(
                f"Memory-adapter tensor contract mismatch; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}."
            )
        for name, value in self.state_dict.items():
            target = named[name]
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(
                    f"Memory-adapter shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.data.copy_(value.to(device=target.device, dtype=target.dtype))
            target.requires_grad_(False)
