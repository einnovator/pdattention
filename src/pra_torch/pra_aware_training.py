"""Differentiable native-KV training utilities for Paper 4 experiments.

The production PRA path normally builds a detached, reusable reference cache.
This module provides the complementary training path: known reference tokens
are encoded inside the current autograd graph and their native K/V are consumed
by selected PRA layers.  Keeping oracle selection fixed isolates the question
of whether a transformer can learn to *use* sparse memory before router or
materializer learning is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch
import torch.nn as nn

from .controlled_local_sa import ControlledExample
from .model import PRATransformerBlock, TinyPRAModel


MemoryCondition = Literal[
    "none",
    "evidence_only",
    "whole_parent",
    "matched_distractor",
]
AdaptationRegime = Literal[
    "frozen",
    "consumer_lora",
    "interface_lora",
    "broad_lora",
    "full_weight",
    "native_scratch",
]


@dataclass(frozen=True)
class ControlledMemoryBatch:
    """Padded oracle memory and token roles for one controlled-model batch.

    ``input_ids`` and masks have shape ``[B,M]``. ``evidence_mask`` marks the
    physical memory positions belonging to gold path facts; it is diagnostic
    metadata and is never exposed to the model.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    evidence_mask: torch.Tensor
    condition: MemoryCondition

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the valid native-KV token count for each batch row."""
        return tuple(int(value) for value in self.attention_mask.sum(dim=1).tolist())


@dataclass(frozen=True)
class NativeMemoryForward:
    """Model output plus layer-local memory-consumption diagnostics."""

    logits: torch.Tensor  # [B,T,V]
    hidden_states: torch.Tensor  # [B,T,D] after final decoder normalization
    layer_metrics: dict[int, dict[str, float]]


class LoRALinear(nn.Module):
    """Low-rank residual update around a frozen linear projection.

    The wrapped ``base`` projection is retained byte-for-byte. Trainable
    matrices implement ``base(x) + scale * B(A(x))``; zero-initialized ``B``
    makes installation function-preserving.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen base projection and its trainable low-rank delta."""
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


def build_controlled_memory_batch(
    examples: Sequence[ControlledExample],
    *,
    condition: MemoryCondition,
    pad_token_id: int,
    device: str | torch.device | None = None,
) -> ControlledMemoryBatch:
    """Materialize a blinded oracle-memory condition from controlled references.

    ``evidence_only`` includes only path facts, ``whole_parent`` includes every
    fact, and ``matched_distractor`` includes as many distractor facts as there
    are evidence facts. Reference order remains the generated parent order.
    """
    if condition not in {
        "none",
        "evidence_only",
        "whole_parent",
        "matched_distractor",
    }:
        raise ValueError(f"Unsupported memory condition: {condition}")

    token_rows: list[list[int]] = []
    evidence_rows: list[list[bool]] = []
    for example in examples:
        if condition == "none":
            selected = []
        elif condition == "evidence_only":
            selected = [reference for reference in example.references if reference.is_evidence]
        elif condition == "whole_parent":
            selected = list(example.references)
        else:
            evidence_count = sum(reference.is_evidence for reference in example.references)
            selected = [
                reference for reference in example.references if not reference.is_evidence
            ][:evidence_count]

        tokens: list[int] = []
        roles: list[bool] = []
        for reference in selected:
            tokens.extend(reference.token_ids)
            roles.extend([reference.is_evidence] * len(reference.token_ids))
        token_rows.append(tokens)
        evidence_rows.append(roles)

    width = max(max((len(row) for row in token_rows), default=0), 1)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    evidence_mask = torch.zeros((len(examples), width), dtype=torch.bool)
    for index, (tokens, roles) in enumerate(zip(token_rows, evidence_rows)):
        if not tokens:
            continue
        input_ids[index, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[index, : len(tokens)] = 1
        evidence_mask[index, : len(tokens)] = torch.tensor(roles, dtype=torch.bool)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        evidence_mask = evidence_mask.to(device)
    return ControlledMemoryBatch(input_ids, attention_mask, evidence_mask, condition)


def _empty_layer_memory(
    model: TinyPRAModel,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Create zero-length ``[1,H,0,Dh]`` rows for every PRA consumer layer."""
    result = {}
    for block in model.blocks:
        if not isinstance(block, PRATransformerBlock):
            continue
        empty = torch.empty(
            (1, model.cfg.n_heads, 0, model.cfg.d_model // model.cfg.n_heads),
            device=device,
            dtype=dtype,
        )
        result[block.layer_id] = (
            [empty for _ in range(batch_size)],
            [empty for _ in range(batch_size)],
        )
    return result


def encode_differentiable_memory_kv(
    model: TinyPRAModel,
    memory: ControlledMemoryBatch,
) -> dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Encode ``[B,M]`` references and retain per-layer native K/V autograd.

    Each output row is ``[1,H,M_i,Dh]`` and is captured immediately before the
    corresponding PRA sublayer, matching detached cache construction semantics.
    Padding is removed before native attention.
    """
    batch_size, token_count = memory.input_ids.shape
    if not bool(memory.attention_mask.any()):
        return _empty_layer_memory(
            model,
            batch_size,
            device=memory.input_ids.device,
            dtype=model.token_emb.weight.dtype,
        )
    if token_count > model.cfg.effective_model_max_context_tokens:
        raise ValueError("Differentiable memory encoding exceeds model context capacity.")

    positions = torch.arange(token_count, device=memory.input_ids.device)
    hidden = model.position_encoding.apply_embeddings(
        model.token_emb(memory.input_ids),
        positions,
        model.pos_emb,
    )
    lengths = memory.lengths
    encoded = {}
    for block in model.blocks:
        if isinstance(block, PRATransformerBlock):
            layer_kv = block.project_reference_kv(
                hidden,
                detach=False,
                position_ids=positions,
            )
            encoded[block.layer_id] = (
                [layer_kv.k[row : row + 1, :, :length, :] for row, length in enumerate(lengths)],
                [layer_kv.v[row : row + 1, :, :length, :] for row, length in enumerate(lengths)],
            )
        hidden = block(
            hidden,
            use_pra_memory=False,
            attention_mask=memory.attention_mask,
            position_ids=positions,
        )
    return encoded


def _memory_role_mass(
    final_weights: tuple[tuple[tuple[float, ...], ...], ...],
    memory: ControlledMemoryBatch,
) -> tuple[float, float]:
    """Reduce per-head final-token weights into evidence and distractor mass."""
    evidence_total = distractor_total = 0.0
    count = 0
    for row, heads in enumerate(final_weights):
        length = int(memory.attention_mask[row].sum())
        role = memory.evidence_mask[row, :length].detach().cpu()
        for head in heads:
            weights = torch.tensor(head[:length], dtype=torch.float32)
            evidence_total += float(weights[role].sum())
            distractor_total += float(weights[~role].sum())
            count += 1
    return evidence_total / max(count, 1), distractor_total / max(count, 1)


def forward_with_differentiable_memory(
    model: TinyPRAModel,
    query_input_ids: torch.Tensor,
    memory: ControlledMemoryBatch,
    *,
    attention_mask: torch.Tensor | None = None,
) -> NativeMemoryForward:
    """Run query tokens against oracle-selected native K/V with full autograd.

    Router and cache identity are intentionally absent. This is the Paper 4
    consumer-learning intervention: the model sees only the selected physical
    token K/V and must learn whether and how strongly to use them.
    """
    if query_input_ids.ndim != 2:
        raise ValueError("query_input_ids must have shape [B,T].")
    if memory.input_ids.shape[0] != query_input_ids.shape[0]:
        raise ValueError("Memory and query batch sizes must match.")
    batch_size, token_count = query_input_ids.shape
    positions = torch.arange(token_count, device=query_input_ids.device)
    hidden = model.position_encoding.apply_embeddings(
        model.token_emb(query_input_ids),
        positions,
        model.pos_emb,
    )
    memory_by_layer = encode_differentiable_memory_kv(model, memory)
    layer_metrics: dict[int, dict[str, float]] = {}
    for block in model.blocks:
        if not isinstance(block, PRATransformerBlock):
            hidden = block(
                hidden,
                use_pra_memory=False,
                attention_mask=attention_mask,
                position_ids=positions,
            )
            continue
        keys, values = memory_by_layer[block.layer_id]
        normalized = block.ln1(hidden)
        update = block.attn.forward_native_kv(
            normalized,
            keys,
            values,
            attention_mask=attention_mask,
            position_ids=positions,
        )
        hidden = hidden + update
        hidden = hidden + block.ff(block.ln2(hidden))
        stats = block.attn.last_memory_batching_stats
        evidence_mass = distractor_mass = 0.0
        if stats is not None and stats.final_token_memory_weights:
            evidence_mass, distractor_mass = _memory_role_mass(
                stats.final_token_memory_weights,
                memory,
            )
        layer_metrics[block.layer_id] = {
            **block.attn.last_diagnostics,
            "evidence_attention_mass": evidence_mass,
            "distractor_attention_mass": distractor_mass,
            "memory_update_norm": float(update.detach().norm().cpu()),
            "memory_update_per_token": float(
                update.detach().norm(dim=-1).mean().cpu()
            ),
        }
    normalized = model.ln(hidden)
    return NativeMemoryForward(model.head(normalized), normalized, layer_metrics)


def _replace_linear(model: nn.Module, path: str, *, rank: int, alpha: float, dropout: float) -> None:
    """Replace one dotted-path linear module with a function-preserving LoRA wrapper."""
    parent = model
    pieces = path.split(".")
    for piece in pieces[:-1]:
        parent = parent[int(piece)] if piece.isdigit() else getattr(parent, piece)
    name = pieces[-1]
    layer = parent[int(name)] if name.isdigit() else getattr(parent, name)
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"LoRA target {path} is not nn.Linear.")
    replacement = LoRALinear(
        layer,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
    ).to(device=layer.weight.device, dtype=layer.weight.dtype)
    if name.isdigit():
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)


def install_adaptation_regime(
    model: TinyPRAModel,
    regime: AdaptationRegime,
    *,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
) -> tuple[str, ...]:
    """Freeze or expose the parameter groups defined by the Paper 4 ladder.

    Consumer LoRA adapts PRA Q/O and the immediately following FFN. Interface
    LoRA additionally adapts memory-producing K/V. Broad LoRA targets every
    decoder attention/FFN linear while leaving embeddings and LM head frozen.
    """
    if regime not in {
        "frozen",
        "consumer_lora",
        "interface_lora",
        "broad_lora",
        "full_weight",
        "native_scratch",
    }:
        raise ValueError(f"Unsupported adaptation regime: {regime}")
    for parameter in model.parameters():
        parameter.requires_grad = regime in {"full_weight", "native_scratch"}
    if regime in {"frozen", "full_weight", "native_scratch"}:
        return tuple()

    pra_layers = {
        block.layer_id for block in model.blocks if isinstance(block, PRATransformerBlock)
    }
    targets: list[str] = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not path.startswith("blocks."):
            continue
        pieces = path.split(".")
        layer_id = int(pieces[1])
        is_pra = layer_id in pra_layers
        if regime == "consumer_lora":
            selected = is_pra and (
                path.endswith("attn.q_proj")
                or path.endswith("attn.o_proj")
                or ".ff." in path
            )
        elif regime == "interface_lora":
            selected = is_pra and (
                any(path.endswith(f"attn.{name}_proj") for name in ("q", "k", "v", "o"))
                or ".ff." in path
            )
        else:
            selected = True
        if selected:
            targets.append(path)
    # Replace deepest paths after discovery so named-module iteration is stable.
    for path in targets:
        _replace_linear(
            model,
            path,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
    return tuple(targets)


def parameter_summary(model: nn.Module) -> dict[str, float | int]:
    """Report total/trainable parameters and adaptation fraction."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": float(trainable / max(total, 1)),
    }


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield the parameters exposed by the selected adaptation regime."""
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
