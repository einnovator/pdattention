"""Pin and validate the official Qwen3-30B-A3B HF MoE attention contract."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import transformers
from huggingface_hub import HfApi
from transformers import AutoConfig


MODEL_ID = "Qwen/Qwen3-30B-A3B"
DEFAULT_OUTPUT = Path(
    "docs/papers/shared/results/paper2_hf/qwen3_moe/qwen3_30b_a3b_contract.json"
)


def validate_contract(model_id: str = MODEL_ID) -> dict[str, object]:
    """Return a weight-free receipt for the exact public model architecture."""

    info = HfApi().model_info(model_id)
    config = AutoConfig.from_pretrained(model_id, revision=info.sha, trust_remote_code=False)
    from transformers.models.qwen3_moe import modeling_qwen3_moe as implementation

    observed = {
        "architecture": tuple(config.architectures or ()),
        "model_type": config.model_type,
        "layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "query_heads": int(config.num_attention_heads),
        "native_kv_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "experts": int(config.num_experts),
        "experts_per_token": int(config.num_experts_per_tok),
        "decoder_sparse_step": int(config.decoder_sparse_step),
        "mlp_only_layers": list(config.mlp_only_layers),
        "max_position_embeddings": int(config.max_position_embeddings),
    }
    expected = {
        "architecture": ("Qwen3MoeForCausalLM",),
        "model_type": "qwen3_moe",
        "layers": 48,
        "hidden_size": 2048,
        "query_heads": 32,
        "native_kv_heads": 4,
        "head_dim": 128,
        "experts": 128,
        "experts_per_token": 8,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "max_position_embeddings": 40960,
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    attention = implementation.Qwen3MoeAttention
    return {
        "schema_version": "pra-hf-moe-contract-v1",
        "model_id": model_id,
        "model_revision": info.sha,
        "transformers_version": transformers.__version__,
        "observed": observed,
        "attention_contract": {
            "module": f"{attention.__module__}.{attention.__qualname__}",
            "forward_signature": str(inspect.signature(attention.forward)),
            "eager_signature": str(inspect.signature(implementation.eager_attention_forward)),
            "rope_signature": str(inspect.signature(implementation.apply_rotary_pos_emb)),
            "pra_scope": "attention_only",
            "router_ownership": "host_model",
            "expert_execution_ownership": "host_model",
        },
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
        "evidence_boundary": (
            "Pinned public configuration and installed-code structural validation only; "
            "not a loaded-checkpoint quality, parity, latency, or memory result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate_contract(args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
