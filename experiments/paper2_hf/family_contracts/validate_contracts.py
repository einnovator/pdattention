"""Pin weight-free architecture receipts for later PRA-HF family ports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import transformers
from huggingface_hub import HfApi, hf_hub_download

from pra_torch.hf import describe_qwen3_hybrid_plan


MODELS = {
    "mistral3": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "gpt_oss": "openai/gpt-oss-20b",
    "qwen3_8": "Qwen/Qwen3.8-Flash-Next-FP8",
}
DEFAULT_OUTPUT = Path(
    "docs/papers/shared/results/paper2_hf/family_contracts/extended_family_contracts.json"
)


def _download_config(model_id: str, revision: str) -> dict:
    path = hf_hub_download(model_id, "config.json", revision=revision)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_contracts() -> dict[str, object]:
    """Validate published configurations without downloading model weights."""

    api = HfApi()
    receipts = {}
    for name, model_id in MODELS.items():
        revision = api.model_info(model_id).sha
        config = _download_config(model_id, revision)
        receipts[name] = {
            "model_id": model_id,
            "revision": revision,
            "model_type": config["model_type"],
            "architecture": config.get("architectures", []),
        }
        if name == "mistral3":
            text = config["text_config"]
            receipts[name]["text_decoder"] = {
                key: text[key]
                for key in (
                    "model_type",
                    "num_hidden_layers",
                    "hidden_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "sliding_window",
                )
            }
            receipts[name]["pra_scope"] = "nested_text_decoder_attention"
        elif name == "gpt_oss":
            layer_types = tuple(config["layer_types"])
            receipts[name].update(
                {
                    "layers": len(layer_types),
                    "full_attention_layers": [
                        i for i, kind in enumerate(layer_types) if kind == "full_attention"
                    ],
                    "sliding_attention_layers": [
                        i for i, kind in enumerate(layer_types) if kind == "sliding_attention"
                    ],
                    "experts": config["num_local_experts"],
                    "experts_per_token": config["num_experts_per_tok"],
                    "pra_scope": "full_attention_only",
                }
            )
        else:
            text = config["text_config"]
            plan = describe_qwen3_hybrid_plan(SimpleNamespace(**text))
            receipts[name].update(
                {
                    "text_model_type": text["model_type"],
                    "total_parameters": "125B",
                    "active_parameters": "6B",
                    "full_attention_interval": text["full_attention_interval"],
                    "indexer_budget": text["indexer_budget"],
                    "indexer_compress_ratio": text["indexer_compress_ratio"],
                    "plan": plan.to_dict(),
                    "evaluation_blocker": (
                        "Installed Transformers lacks qwen4_exp; full 125B/FP8 evaluation "
                        "also exceeds the available local memory and disk budget."
                    ),
                }
            )

    checks = {
        "mistral3_nested_text_decoder": receipts["mistral3"]["model_type"] == "mistral3",
        "gpt_oss_alternating_attention": (
            len(receipts["gpt_oss"]["full_attention_layers"]) == 12
            and len(receipts["gpt_oss"]["sliding_attention_layers"]) == 12
        ),
        "qwen3_8_hybrid_36_gdn_12_qsa": (
            len(receipts["qwen3_8"]["plan"]["gdn_layer_ids"]) == 36
            and len(receipts["qwen3_8"]["plan"]["qsa_layer_ids"]) == 12
        ),
    }
    return {
        "schema_version": "pra-hf-extended-family-contract-v1",
        "transformers_version": transformers.__version__,
        "receipts": receipts,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "evidence_boundary": (
            "Pinned public configuration plus tiny synthetic-model adapter tests; "
            "not loaded-checkpoint quality, latency, or memory evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = validate_contracts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
