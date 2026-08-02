"""Utilities for estimating PRA model parameter and memory requirements."""

from __future__ import annotations


DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
}


def bytes_to_mib(value: float) -> float:
    """Convert bytes to MiB."""
    return float(value) / (1024**2)


def _linear_params(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def _layer_norm_params(d_model: int) -> int:
    return 2 * d_model


def vanilla_block_params(d_model: int) -> int:
    """Estimate parameters in ``VanillaTransformerBlock``."""
    attention = 3 * d_model * d_model + 3 * d_model
    attention += _linear_params(d_model, d_model)
    feed_forward = _linear_params(d_model, 4 * d_model) + _linear_params(4 * d_model, d_model)
    norms = 2 * _layer_norm_params(d_model)
    return attention + feed_forward + norms


def pra_block_params(d_model: int) -> int:
    """Estimate parameters in ``PRATransformerBlock``."""
    attention = 5 * _linear_params(d_model, d_model)
    feed_forward = _linear_params(d_model, 4 * d_model) + _linear_params(4 * d_model, d_model)
    norms = 2 * _layer_norm_params(d_model)
    return attention + feed_forward + norms


def mixed_block_params(d_model: int) -> int:
    """Estimate parameters in ``PRASATransformerBlock``."""
    self_attention = 3 * d_model * d_model + 3 * d_model
    self_attention += _linear_params(d_model, d_model)
    pra_attention = 5 * _linear_params(d_model, d_model)
    feed_forward = _linear_params(d_model, 4 * d_model) + _linear_params(4 * d_model, d_model)
    norms = 3 * _layer_norm_params(d_model)
    return self_attention + pra_attention + feed_forward + norms


def estimate_model_size(
    cfg: dict,
    *,
    vocab_size: int = 128,
    batch_size: int | None = None,
    dtype: str = "float32",
    optimizer: str = "adamw",
) -> dict:
    """Estimate parameter count and memory needs for a resolved config dict."""
    model = cfg["model"]
    train = cfg.get("train", {})
    d_model = int(model["d_model"])
    n_layers = int(model["n_layers"])
    n_vanilla = int(model.get("n_vanilla_layers", 0))
    n_mixed = int(model.get("n_mixed_layers", 0))
    n_pra = max(n_layers - n_vanilla - n_mixed, 0)
    max_seq_len = int(model["max_seq_len"])
    batch_size = int(batch_size if batch_size is not None else train.get("batch_size", 1))
    bytes_per_value = DTYPE_BYTES.get(dtype.lower())
    if bytes_per_value is None:
        raise ValueError(f"Unsupported dtype for size estimate: {dtype}")

    embedding_params = vocab_size * d_model + max_seq_len * d_model
    block_params = (
        n_vanilla * vanilla_block_params(d_model)
        + n_mixed * mixed_block_params(d_model)
        + n_pra * pra_block_params(d_model)
    )
    final_norm_params = _layer_norm_params(d_model)
    output_head_params = d_model * vocab_size
    total_params = embedding_params + block_params + final_norm_params + output_head_params

    parameter_bytes = total_params * bytes_per_value
    gradient_bytes = total_params * bytes_per_value
    optimizer_multiplier = 2 if optimizer.lower() in {"adam", "adamw"} else 0
    optimizer_state_bytes = total_params * bytes_per_value * optimizer_multiplier

    hidden_values = batch_size * max_seq_len * d_model
    activation_bytes = hidden_values * bytes_per_value * max(n_layers, 1) * 2
    inference_bytes = parameter_bytes + activation_bytes
    training_bytes = parameter_bytes + gradient_bytes + optimizer_state_bytes + activation_bytes

    return {
        "selected_model": cfg.get("selected_model", "default"),
        "d_model": d_model,
        "n_layers": n_layers,
        "n_vanilla_layers": n_vanilla,
        "n_mixed_layers": n_mixed,
        "n_pra_layers": n_pra,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "vocab_size": int(vocab_size),
        "dtype": dtype,
        "optimizer": optimizer,
        "total_params": int(total_params),
        "parameter_mib": bytes_to_mib(parameter_bytes),
        "gradient_mib": bytes_to_mib(gradient_bytes),
        "optimizer_state_mib": bytes_to_mib(optimizer_state_bytes),
        "activation_mib": bytes_to_mib(activation_bytes),
        "inference_mib": bytes_to_mib(inference_bytes),
        "training_mib": bytes_to_mib(training_bytes),
    }
