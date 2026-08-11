"""Tiny projection metrics for frozen-backbone HF PRA routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HFRoutingProjection(nn.Module):
    """Project query and memory hidden states into a compact cosine-routing space."""

    def __init__(
        self,
        input_width: int,
        routing_width: int,
        architecture: str = "asymmetric_linear",
    ) -> None:
        super().__init__()
        if input_width <= 0 or routing_width <= 0:
            raise ValueError("Routing projection widths must be positive.")
        if architecture not in {"shared_linear", "asymmetric_linear", "asymmetric_mlp"}:
            raise ValueError(f"Unsupported routing projection architecture: {architecture}")
        self.input_width = int(input_width)
        self.routing_width = int(routing_width)
        self.architecture = architecture
        if architecture == "shared_linear":
            projection = nn.Linear(input_width, routing_width, bias=False)
            self.query_projection = projection
            self.memory_projection = projection
        elif architecture == "asymmetric_linear":
            self.query_projection = nn.Linear(input_width, routing_width, bias=False)
            self.memory_projection = nn.Linear(input_width, routing_width, bias=False)
        else:
            self.query_projection = self._mlp(input_width, routing_width)
            self.memory_projection = self._mlp(input_width, routing_width)

    @staticmethod
    def _mlp(input_width: int, routing_width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_width, routing_width, bias=False),
            nn.GELU(),
            nn.Linear(routing_width, routing_width, bias=False),
        )

    def _validate(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[-1] != self.input_width:
            raise ValueError(
                f"Routing features must have shape [items,{self.input_width}]."
            )

    def project_query(self, query: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """Project query rows ``[B,D_model]`` into ``[B,D_route]``."""
        self._validate(query)
        projected = self.query_projection(query)
        return F.normalize(projected, dim=-1, eps=1e-12) if normalize else projected

    def project_memory(self, memory: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        """Project memory rows ``[C,D_model]`` into ``[C,D_route]``."""
        self._validate(memory)
        projected = self.memory_projection(memory)
        return F.normalize(projected, dim=-1, eps=1e-12) if normalize else projected

    def scores(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Return normalized query-memory scores with shape ``[B,C]``."""
        return self.project_query(query) @ self.project_memory(memory).transpose(0, 1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
