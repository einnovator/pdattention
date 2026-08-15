"""Exact layerwise residual capture and contextualization diagnostics.

The utilities in this module are intentionally independent of PRA routing.  They
observe a pre-norm Hugging Face decoder while memory is disabled, preserving the
native residual, attention, MLP, head, and positional semantics used to build a
PRA graph later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class LayerContextSnapshot:
    """One decoder layer's exact states for a single forward pass.

    Every state is ``[B,T,D]``. ``attention_input`` is the normalized state that
    feeds native Q/K projection; ``pre_attention_residual`` is the residual before
    that normalization. ``attention_contribution`` and ``ffn_contribution`` are
    the branch vectors immediately before their respective residual additions.
    Attention probabilities, when retained, are ``[B,H_q,T,T]``.
    """

    pre_attention_residual: torch.Tensor | None = None
    attention_input: torch.Tensor | None = None
    attention_contribution: torch.Tensor | None = None
    post_attention_residual: torch.Tensor | None = None
    ffn_contribution: torch.Tensor | None = None
    layer_output: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None


class LayerContextCollector:
    """Capture exact pre-norm decoder branch states without changing execution.

    The collector expects decoder blocks with ``self_attn``, ``mlp``, and
    ``post_attention_layernorm`` attributes, as used by Qwen 3. Hooks only retain
    detached views from the latest forward. Call :meth:`close` after use.
    """

    def __init__(self, layers, layer_ids: Iterable[int]) -> None:
        self.layer_ids = tuple(sorted({int(layer) for layer in layer_ids}))
        if not self.layer_ids or self.layer_ids[0] < 0 or self.layer_ids[-1] >= len(layers):
            raise ValueError("Layer selection is empty or outside the decoder stack.")
        self.snapshots = {layer: LayerContextSnapshot() for layer in self.layer_ids}
        self._handles = []
        for layer_id in self.layer_ids:
            layer = layers[layer_id]
            for name in ("self_attn", "mlp", "post_attention_layernorm"):
                if not hasattr(layer, name):
                    raise TypeError(f"Layer {layer_id} does not expose {name}.")
            self._handles.extend(
                [
                    layer.register_forward_pre_hook(
                        self._capture_pre_attention(layer_id), with_kwargs=True
                    ),
                    layer.self_attn.register_forward_pre_hook(
                        self._capture_attention_input(layer_id), with_kwargs=True
                    ),
                    layer.self_attn.register_forward_hook(
                        self._capture_attention_output(layer_id), with_kwargs=True
                    ),
                    layer.post_attention_layernorm.register_forward_pre_hook(
                        self._capture_post_attention(layer_id), with_kwargs=True
                    ),
                    layer.mlp.register_forward_hook(
                        self._capture_ffn_output(layer_id), with_kwargs=True
                    ),
                    layer.register_forward_hook(
                        self._capture_layer_output(layer_id), with_kwargs=True
                    ),
                ]
            )

    @staticmethod
    def _hidden(args, kwargs) -> torch.Tensor:
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Could not identify decoder hidden states.")
        return hidden.detach()

    @staticmethod
    def _tensor_output(output) -> torch.Tensor:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Expected a tensor decoder output.")
        return value.detach()

    def _capture_pre_attention(self, layer_id):
        def hook(_module, args, kwargs):
            self.snapshots[layer_id].pre_attention_residual = self._hidden(args, kwargs)

        return hook

    def _capture_attention_input(self, layer_id):
        def hook(_module, args, kwargs):
            self.snapshots[layer_id].attention_input = self._hidden(args, kwargs)

        return hook

    def _capture_attention_output(self, layer_id):
        def hook(module, _args, _kwargs, output):
            snapshot = self.snapshots[layer_id]
            snapshot.attention_contribution = self._tensor_output(output)
            weights = getattr(module, "last_attention_weights", None)
            snapshot.attention_weights = None if weights is None else weights.detach()

        return hook

    def _capture_post_attention(self, layer_id):
        def hook(_module, args, kwargs):
            self.snapshots[layer_id].post_attention_residual = self._hidden(args, kwargs)

        return hook

    def _capture_ffn_output(self, layer_id):
        def hook(_module, _args, _kwargs, output):
            self.snapshots[layer_id].ffn_contribution = self._tensor_output(output)

        return hook

    def _capture_layer_output(self, layer_id):
        def hook(_module, _args, _kwargs, output):
            self.snapshots[layer_id].layer_output = self._tensor_output(output)

        return hook

    def clear(self) -> None:
        """Discard tensors from the preceding forward while retaining hooks."""
        self.snapshots = {layer: LayerContextSnapshot() for layer in self.layer_ids}

    def validate(self, *, atol: float = 2e-3, rtol: float = 2e-3) -> None:
        """Verify both captured residual additions against actual block states."""
        for layer_id, snapshot in self.snapshots.items():
            values = (
                snapshot.pre_attention_residual,
                snapshot.attention_input,
                snapshot.attention_contribution,
                snapshot.post_attention_residual,
                snapshot.ffn_contribution,
                snapshot.layer_output,
            )
            if any(value is None for value in values):
                raise RuntimeError(f"Layer {layer_id} capture is incomplete.")
            expected_post = (
                snapshot.pre_attention_residual.float()
                + snapshot.attention_contribution.float()
            )
            if not torch.allclose(
                expected_post, snapshot.post_attention_residual.float(), atol=atol, rtol=rtol
            ):
                raise AssertionError(f"Layer {layer_id} attention residual identity failed.")
            expected_output = (
                snapshot.post_attention_residual.float()
                + snapshot.ffn_contribution.float()
            )
            if not torch.allclose(
                expected_output, snapshot.layer_output.float(), atol=atol, rtol=rtol
            ):
                raise AssertionError(f"Layer {layer_id} FFN residual identity failed.")

    def close(self) -> None:
        """Remove every observation hook from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


def causal_radius_mask(
    tokens: int,
    radius: int | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``[1,1,T,T]`` causal visibility with fixed token coordinates.

    ``radius=1`` is self-only, ``radius=16`` exposes the current and previous
    fifteen tokens, and ``None`` reproduces full causal visibility. Tokens are
    masked in place; no token is removed or position ID rebound.
    """
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if radius is not None and radius <= 0:
        raise ValueError("radius must be positive or None")
    query = torch.arange(tokens, device=device).unsqueeze(1)
    key = torch.arange(tokens, device=device).unsqueeze(0)
    visible = key <= query
    if radius is not None:
        visible &= key > (query - int(radius))
    mask = torch.zeros((tokens, tokens), device=device, dtype=dtype)
    mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


def normalized_displacement(
    left: torch.Tensor, right: torch.Tensor, *, eps: float = 1e-8
) -> torch.Tensor:
    """Return tokenwise ``||right-left|| / (||left||+eps)`` for ``[...,D]``."""
    return (right.float() - left.float()).norm(dim=-1) / (
        left.float().norm(dim=-1) + eps
    )


def directional_rotation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return tokenwise cosine distance between two ``[...,D]`` state tensors."""
    return 1.0 - torch.nn.functional.cosine_similarity(
        left.float(), right.float(), dim=-1, eps=1e-8
    )


def branch_token_metrics(snapshot: LayerContextSnapshot) -> dict[str, torch.Tensor]:
    """Compute C1--C3 and the FFN control as tokenwise ``[B,T]`` tensors."""
    pre = snapshot.pre_attention_residual
    alpha = snapshot.attention_contribution
    post = snapshot.post_attention_residual
    beta = snapshot.ffn_contribution
    if any(value is None for value in (pre, alpha, post, beta)):
        raise RuntimeError("Branch metrics require a complete layer snapshot.")
    pre_norm = pre.float().norm(dim=-1)
    alpha_norm = alpha.float().norm(dim=-1)
    post_norm = post.float().norm(dim=-1)
    beta_norm = beta.float().norm(dim=-1)
    return {
        "pre_attention_residual_norm": pre_norm,
        "attention_output_norm": alpha_norm,
        "attention_contribution_ratio": alpha_norm / (pre_norm + 1e-8),
        "post_attention_displacement": normalized_displacement(pre, post),
        "post_attention_rotation": directional_rotation(pre, post),
        "ffn_output_norm": beta_norm,
        "ffn_contribution_ratio": beta_norm / (post_norm + 1e-8),
    }


def attention_token_metrics(
    weights: torch.Tensor,
    *,
    local_window: int = 32,
    evidence_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate native query-head attention diagnostics to tokenwise ``[B,T]``.

    Qwen expands native GQA K/V heads inside its eager kernel; consequently the
    retained probabilities already have query-head layout and are averaged over
    those heads here.
    """
    if weights.ndim != 4 or weights.shape[-1] != weights.shape[-2]:
        raise ValueError("attention weights must have shape [B,H,T,T]")
    probabilities = weights.float().clamp_min(0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    tokens = probabilities.shape[-1]
    positions = torch.arange(tokens, device=probabilities.device)
    self_fraction = probabilities.diagonal(dim1=-2, dim2=-1)
    query = positions.unsqueeze(1)
    key = positions.unsqueeze(0)
    local = (key <= query) & (key > query - int(local_window))
    distant = key <= query - int(local_window)
    local_fraction = (probabilities * local).sum(dim=-1)
    distant_fraction = (probabilities * distant).sum(dim=-1)
    result = {
        "attention_entropy": entropy.mean(dim=1),
        "effective_attention_support": entropy.exp().mean(dim=1),
        "self_attention_fraction": self_fraction.mean(dim=1),
        "local_attention_fraction": local_fraction.mean(dim=1),
        "distant_attention_fraction": distant_fraction.mean(dim=1),
    }
    if evidence_mask is not None:
        if evidence_mask.shape != (probabilities.shape[0], tokens):
            raise ValueError("evidence_mask must have shape [B,T]")
        evidence = evidence_mask[:, None, None, :].to(probabilities.dtype)
        result["evidence_attention_fraction"] = (
            probabilities * evidence
        ).sum(dim=-1).mean(dim=1)
    return result


def summarize(values: torch.Tensor) -> dict[str, float]:
    """Return stable descriptive statistics without treating tokens as replicates."""
    flat = values.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "p90": float(torch.quantile(flat, 0.9)),
    }
