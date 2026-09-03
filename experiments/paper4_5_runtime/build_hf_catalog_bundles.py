"""Build the cross-family PRA bundle catalog from qualified routing artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from pra_hf.bundle import BundleBuilder
from pra_hf.bundle_evidence import (
    EvidenceIdentity,
    canonicalize_paired_transport_evidence,
    import_mlx_paired_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
COLLECTION = "EInnovator/pra-bundles-6a971e52093232f858e660f6"
RESULTS = ROOT / "docs/papers/shared/results/paper4_5_runtime/hf_catalog_adapters"
ROUTERS = ROOT / "artifacts/pra_hf/routers"
BUNDLES = ROOT / "artifacts/pra_hf/bundles"
MLX_RESULTS = ROOT / "docs/papers/shared/results/mac_scaling"
QUANTIZED_RESULTS = (
    ROOT / "docs/papers/shared/results/paper4_5_runtime/quantized_bundles"
)

SPECS = {
    "qwen2.5-1.5b-instruct": {
        "label": "Qwen2.5-1.5B-Instruct",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "architecture": "Qwen2ForCausalLM",
        "family": "qwen",
        "model_type": "qwen2",
        "layers": 28,
        "hidden_size": 1536,
        "heads": {"query": 12, "kv": 2, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "1.5B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen2-5-1-5b-instruct",
        "engine": "hf",
        "engine_version": "transformers 5.16.1",
        "hardware": "NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB",
        "quantization": "bfloat16",
        "post_training": "general instruction tuning",
        "qualification_date": "2026-09-03",
    },
    "qwen2.5-coder-1.5b-instruct": {
        "label": "Qwen2.5-Coder-1.5B-Instruct",
        "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a",
        "architecture": "Qwen2ForCausalLM",
        "family": "qwen",
        "model_type": "qwen2",
        "layers": 28,
        "hidden_size": 1536,
        "heads": {"query": 12, "kv": 2, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "1.5B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen2-5-coder-1-5b-instruct",
        "engine": "hf",
        "engine_version": "transformers 5.16.1",
        "hardware": "NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB",
        "quantization": "bfloat16",
        "post_training": "code and instruction tuning",
        "qualification_date": "2026-09-03",
        "extra_limitations": [
            "This qualification uses natural-document routing; code retrieval and coding-agent task success remain separately unmeasured."
        ],
    },
    "qwen2.5-1.5b-instruct-bnb-8bit": {
        "label": "Qwen2.5-1.5B-Instruct (bitsandbytes 8-bit)",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "architecture": "Qwen2ForCausalLM",
        "family": "qwen",
        "model_type": "qwen2",
        "layers": 28,
        "hidden_size": 1536,
        "heads": {"query": 12, "kv": 2, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "1.5B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen2-5-1-5b-instruct-bnb-8bit",
        "engine": "hf",
        "engine_version": "transformers 5.16.1 / bitsandbytes",
        "hardware": "NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB",
        "quantization": "bnb-8bit",
        "quantization_config": {
            "bits": 8,
            "scheme": "LLM.int8",
            "runtime": "bitsandbytes/PyTorch",
        },
        "post_training": "general instruction tuning",
        "qualification_date": "2026-09-03",
        "routing_artifact": False,
    },
    "qwen3-4b": {
        "label": "Qwen3-4B",
        "base_model": "mlx-community/Qwen3-4B-4bit",
        "revision": "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 36,
        "hidden_size": 2560,
        "heads": {"query": 32, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "4B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-4b-mlx-4bit",
    },
    "qwen3-4b-mlx-8bit": {
        "label": "Qwen3-4B (MLX 8-bit)",
        "base_model": "mlx-community/Qwen3-4B-8bit",
        "revision": "0348ad770d2ae658ca47b0579b2d2c37b20bbcac",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 36,
        "hidden_size": 2560,
        "heads": {"query": 32, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "4B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-4b-mlx-8bit",
        "quantization": "8bit",
    },
    "qwen3-14b": {
        "label": "Qwen3-14B",
        "base_model": "mlx-community/Qwen3-14B-4bit",
        "revision": "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 40,
        "hidden_size": 5120,
        "heads": {"query": 40, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "14B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-14b-mlx-4bit",
        "paired_evidence": "qwen3_14b_mlx_profiles.json",
    },
    "qwen3-14b-mlx-8bit": {
        "label": "Qwen3-14B (MLX 8-bit)",
        "base_model": "mlx-community/Qwen3-14B-8bit",
        "revision": "da33cf28f06636847fd9e93e0a03d819b84cb55e",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 40,
        "hidden_size": 5120,
        "heads": {"query": 40, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "14B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-14b-mlx-8bit",
        "quantization": "8bit",
        "routing_artifact": False,
    },
    "qwen3-8b": {
        "label": "Qwen3-8B",
        "base_model": "mlx-community/Qwen3-8B-4bit",
        "revision": "545dc4251c05440727734bcd94334791f6ab0192",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 36,
        "hidden_size": 4096,
        "heads": {"query": 32, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "8B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-8b-mlx-4bit",
        "paired_evidence": "qwen3_8b_mlx_profiles.json",
        "routing_artifact": False,
    },
    "qwen3-8b-mlx-8bit": {
        "label": "Qwen3-8B (MLX 8-bit)",
        "base_model": "mlx-community/Qwen3-8B-8bit",
        "revision": "48a0b75b1ae72503e21e1558d040bc227510ff06",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 36,
        "hidden_size": 4096,
        "heads": {"query": 32, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "8B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-8b-mlx-8bit",
        "quantization": "8bit",
        "routing_artifact": False,
    },
    "qwen3-8b-mlx-6bit": {
        "label": "Qwen3-8B (MLX 6-bit)",
        "base_model": "mlx-community/Qwen3-8B-6bit",
        "revision": "35a99712f90d6c2c9a2407a3857e104a46edd9e6",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 36,
        "hidden_size": 4096,
        "heads": {"query": 32, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "8B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-8b-mlx-6bit",
        "quantization": "6bit",
        "routing_artifact": False,
    },
    "qwen3-32b": {
        "label": "Qwen3-32B",
        "base_model": "mlx-community/Qwen3-32B-4bit",
        "revision": "bcaaf7f538adf166c1080a2befdb4f6019f66639",
        "architecture": "Qwen3ForCausalLM",
        "family": "qwen",
        "layers": 64,
        "hidden_size": 5120,
        "heads": {"query": 64, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "32B",
        "license": "apache-2.0",
        "repo": "EInnovator/pra-qwen3-32b-mlx-4bit",
        "paired_evidence": "qwen3_32b_mlx_profiles.json",
        "routing_artifact": False,
    },
    "llama3-8b": {
        "label": "Llama-3.1-8B",
        "base_model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "revision": "90215b22ec18e72f623dde2ea7af4097025160e2",
        "architecture": "LlamaForCausalLM",
        "family": "llama",
        "layers": 32,
        "hidden_size": 4096,
        "heads": {"query": 32, "kv": 8, "head_dim": 128},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "8B",
        "license": "llama3.1",
        "repo": "EInnovator/pra-llama3-1-8b-mlx-4bit",
    },
    "llama3.2-1b-mlx-8bit": {
        "label": "Llama-3.2-1B-Instruct (MLX 8-bit)",
        "base_model": "mlx-community/Llama-3.2-1B-Instruct-8bit",
        "revision": "d48cdf0a4ea22d893b7c63a99d6a693e24822795",
        "architecture": "LlamaForCausalLM",
        "family": "llama",
        "layers": 16,
        "hidden_size": 2048,
        "heads": {"query": 32, "kv": 8, "head_dim": 64},
        "topology": {"type": "homogeneous_global", "gqa": True},
        "consumer_layers": "all eligible",
        "parameters": "1B",
        "license": "llama3.2",
        "repo": "EInnovator/pra-llama3-2-1b-mlx-8bit",
        "quantization": "8bit",
        "post_training": "instruction tuning",
        "routing_artifact": False,
    },
    "gemma3-1b": {
        "label": "Gemma-3-1B",
        "base_model": "mlx-community/gemma-3-1b-it-4bit",
        "revision": "2d44e83dc9e80843d22fb941d3d699a0b1351aa6",
        "architecture": "Gemma3ForCausalLM",
        "family": "gemma3",
        "layers": 26,
        "hidden_size": 1152,
        "heads": {"query": 4, "kv": 1, "head_dim": 256},
        "topology": {
            "type": "mixed_sliding_global",
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "native_memory_eligible_layers": [5, 11, 17, 23],
            "gqa": True,
        },
        "consumer_layers": [5, 11, 17, 23],
        "parameters": "1B",
        "license": "gemma",
        "repo": "EInnovator/pra-gemma3-1b-mlx-4bit",
    },
    "gemma3-1b-mlx-8bit": {
        "label": "Gemma-3-1B (MLX 8-bit)",
        "base_model": "mlx-community/gemma-3-1b-it-8bit",
        "revision": "7b963136f21d05ca8b367c93a9c47a7944ad281a",
        "architecture": "Gemma3ForCausalLM",
        "family": "gemma3",
        "layers": 26,
        "hidden_size": 1152,
        "heads": {"query": 4, "kv": 1, "head_dim": 256},
        "topology": {
            "type": "mixed_sliding_global",
            "sliding_window": 512,
            "sliding_window_pattern": 6,
            "native_memory_eligible_layers": [5, 11, 17, 23],
            "gqa": True,
        },
        "consumer_layers": [5, 11, 17, 23],
        "parameters": "1B",
        "license": "gemma",
        "repo": "EInnovator/pra-gemma3-1b-mlx-8bit",
        "quantization": "8bit",
        "routing_artifact": False,
    },
}


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _structural_adapter(spec: dict) -> dict:
    return {
        "schema_version": 1,
        "base_model": {"id": spec["base_model"], "revision": spec["revision"]},
        "architecture": {
            "model_type": spec.get("model_type", spec["family"]),
            "architecture": spec["architecture"],
            "layers": spec["layers"],
            "hidden_size": spec["hidden_size"],
        },
        "mapping": {
            "layers": "model.layers",
            "attention": "self_attn",
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "o_proj": "o_proj",
        },
        "heads": spec["heads"],
        "position": {"type": "rope", "implementation": "native"},
        "topology": spec["topology"],
    }


def _quantization_manifest(spec: dict) -> dict:
    """Describe the exact weight representation consumed by the runtime."""

    if "quantization_config" in spec:
        return dict(spec["quantization_config"])
    quantization = str(spec.get("quantization", "4bit")).lower()
    if quantization in {"4bit", "6bit", "8bit"}:
        return {
            "bits": int(quantization.removesuffix("bit")),
            "group_size": 64,
            "runtime": "MLX",
        }
    return {"name": quantization, "runtime": "PyTorch"}


def _metric_rows(comparison: dict, spec: dict) -> list[dict]:
    rows = []
    for dataset in ("qasper", "hotpotqa", "combined"):
        for mode, key, profile in (
            ("Generic cosine routing", "generic_baseline", "balanced"),
            ("Learned asymmetric routing", "selected_learned", "qasper-learned"),
        ):
            summary = comparison[key][dataset]["summary"]
            count = int(comparison[key][dataset]["examples"])
            rows.append(
                {
                    "metric_class": "ROUTING_DIAGNOSTIC",
                    "model_id": spec["base_model"],
                    "model_revision": spec["revision"],
                    "quantization": spec.get("quantization", "4bit"),
                    "engine": spec.get("engine_version", "mlx-lm 0.31.3"),
                    "hardware": spec.get("hardware", "Apple M4 Pro, 48 GB"),
                    "dataset": dataset,
                    "workload": "held-out frozen-feature routing comparison",
                    "mode": mode,
                    "quality_metric": "R@20%",
                    "quality": summary["R@20%"],
                    "secondary_metrics": {
                        "MRR": summary["MRR"],
                        "AUC0-30": summary["AUC0-30"],
                    },
                    "visible_tokens": None,
                    "ttft_ms": None,
                    "throughput": None,
                    "sample_count": count,
                    "profile": profile,
                    "evidence_tier": "CONTROLLED",
                    "date": spec.get("qualification_date", "2026-09-01"),
                    "provenance": "qualification/comparison.json",
                }
            )
    return rows


def _manifest(
    slug: str,
    spec: dict,
    comparison: dict | None,
    router_config: dict | None,
    paired_evidence: list[dict],
) -> dict:
    commit = _git_commit()
    runtime_smoke = QUANTIZED_RESULTS / slug / "runtime_smoke.json"
    learned_adapters = {}
    if comparison is not None and router_config is not None:
        learned_adapters["combined-router-d128"] = {
            "path": "learned_adapters/combined-router-d128",
            "type": "routing",
            "status": "controlled-artifact",
            "default": False,
            "training": {
                "datasets": ["QASPER", "HotpotQA"],
                "seed": comparison["selected_seed"],
                "seeds_evaluated": comparison["seeds"],
                "steps": comparison["steps"],
                "parameters": comparison["adapter_parameters"],
                "objective": router_config["training_objective"],
            },
            "validation": router_config["metrics"],
        }
    engine = spec.get("engine", "mlx")
    quantization = spec.get("quantization", "4bit")
    profiles = {
        "quality": {
            "purpose": "Candidate maximum-quality profile; held-out calibration is incomplete",
            "routing_adapter": None,
            "consumer_layers": spec["consumer_layers"],
            "status": "CALIBRATION_PENDING",
            "recommended": False,
            "engine": engine,
            "mode": "Native Memory",
        },
        "balanced": {
            "purpose": "Qualified default preserving the all-eligible consumer geometry",
            "routing_adapter": None,
            "consumer_layers": spec["consumer_layers"],
            "status": "QUALIFIED",
            "recommended": True,
            "engine": engine,
            "mode": "Native Memory" if paired_evidence else "Selected Context",
        },
        "economy": {
            "purpose": "Reduced-consumer candidate; the held-out quality gate has not passed",
            "routing_adapter": None,
            "consumer_layers": "CALIBRATION_PENDING",
            "status": "CALIBRATION_PENDING",
            "recommended": False,
            "engine": engine,
            "mode": "Native Memory",
        },
    }
    if learned_adapters:
        profiles["qasper-learned"] = {
            "purpose": "Research-only learned routing profile qualified only on matched QASPER routing diagnostics",
            "routing_adapter": "combined-router-d128",
            "consumer_layers": spec["consumer_layers"],
            "status": "RESEARCH",
            "recommended": False,
            "engine": engine,
            "mode": "Native Memory",
        }
    diagnostics = _metric_rows(comparison, spec) if comparison is not None else []
    return {
        "schema_version": 2,
        "base_model": {
            "id": spec["base_model"],
            "revision": spec["revision"],
            "tokenizer_revision": spec["revision"],
            "architecture": spec["architecture"],
            "family": spec["family"],
            "parameter_count_approx": spec["parameters"],
            "quantization": _quantization_manifest(spec),
            "post_training": spec.get("post_training", "pretrained and post-trained"),
        },
        "structural_adapter": {"path": "structural_adapter", "status": "validated"},
        "learned_adapters": learned_adapters,
        "profiles": profiles,
        "runtime_compatibility": (
            {
                "hf": {
                    "selected_context": "validated" if comparison is not None else "SMOKE" if runtime_smoke.is_file() else "AVAILABLE",
                    "native_memory": "AVAILABLE",
                    "native_serving": "NOT_MEASURED",
                    "recommended": "Selected Context with BALANCED",
                },
                "mlx": {
                    "selected_context": "portable",
                    "native_memory": "NOT_MEASURED for a quantized MLX derivative",
                    "native_serving": "NOT_MEASURED",
                    "recommended": "Selected Context; exact HF artifact only",
                },
            }
            if engine == "hf"
            else {
                "mlx": {
                    "selected_context": "validated" if (paired_evidence or comparison is not None) else "SMOKE" if runtime_smoke.is_file() else "AVAILABLE",
                    "native_memory": "QUALIFIED" if paired_evidence else "AVAILABLE",
                    "native_serving": "NOT_MEASURED",
                    "recommended": "Native Memory with BALANCED" if paired_evidence else "Selected Context with BALANCED",
                },
                "hf": {
                    "selected_context": "portable",
                    "native_memory": "NOT_MEASURED for the full-precision HF counterpart",
                    "native_serving": "NOT_MEASURED",
                    "recommended": "Selected Context; exact MLX artifact only",
                },
            }
        ),
        "engine_realizations": {
            engine: {"bundle_consumed_by": "pra_runtime"},
            "remote_engines": {"bundle_consumed_by": "pra_gateway"},
        },
        "qualification": {
            "contract_version": 1,
            "status": (
                "ENGINE_QUALIFIED"
                if paired_evidence
                else "CONTROLLED"
                if comparison is not None
                else "SMOKE"
                if runtime_smoke.is_file()
                else "NOT_MEASURED"
            ),
            "headline": [row for row in paired_evidence if row["dataset"] == "combined"],
            "canonical_evidence": [
                canonicalize_paired_transport_evidence(row).model_dump(mode="json")
                for row in paired_evidence
                if row["dataset"] == "combined"
            ],
            "metrics": paired_evidence + diagnostics,
            "routing_diagnostics": diagnostics,
            "runtime_smoke": (
                json.loads(runtime_smoke.read_text(encoding="utf-8"))
                if runtime_smoke.is_file()
                else None
            ),
            "training": ({
                "datasets": "QASPER and HotpotQA",
                "train_examples": comparison["training_examples"],
                "validation_examples": comparison["validation_examples"],
                "held_out_test_examples": comparison["test_examples"],
                "seeds": comparison["seeds"],
                "selection": "maximum combined validation AUC0-30",
                "method": router_config["training_objective"],
                "parameter_count": comparison["adapter_parameters"],
                "base_revision": spec["revision"],
            } if comparison is not None and router_config is not None else {}),
            "limitations": [
                *(
                    ["The learned router improves QASPER but is not uniformly positive on HotpotQA; it is opt-in rather than the bundle default."]
                    if comparison is not None
                    else ["No learned router is bundled for this exact quantized identity; routing-adapter transfer from another quantization is intentionally disallowed."]
                ),
                *(
                    ["Paired natural-QA evidence contains five examples per dataset and supports engine qualification, not production qualification."]
                    if paired_evidence
                    else (
                        [f"The held-out routing diagnostic contains {comparison['test_examples'] // 2} examples per dataset and supports controlled routing claims only."]
                        if comparison is not None
                        else ["Only immutable-config structural validation is available for this exact quantized identity."]
                    )
                ),
                *(
                    ["Reduced consumer-layer configurations failed the held-out quality gate; BALANCED therefore retains all eligible layers."]
                    if paired_evidence
                    else ["Native consumer-layer profiles and end-task generation remain uncalibrated for this exact identity."]
                ),
                f"The qualification identity is the exact {quantization} {engine.upper()} model and revision; it does not transfer automatically to another checkpoint, engine, or quantization.",
                *(
                    ["Routing evidence compares a frozen generic router with a small learned router; it does not establish end-task generation quality."]
                    if comparison is not None
                    else [
                        "The runtime smoke loads the exact quantized checkpoint and generates a fixed prompt; it is not an end-task quality or serving benchmark."
                        if runtime_smoke.is_file()
                        else "End-task quality, serving latency, and native-memory parity remain NOT_MEASURED for this exact identity."
                    ]
                ),
                *spec.get("extra_limitations", []),
                "Base-model and dataset licenses apply separately to the router artifact.",
            ],
            "artifacts": [
                *(["qualification/structural_validation.json"] if (QUANTIZED_RESULTS / slug / "structural_validation.json").is_file() else []),
                *(["qualification/runtime_smoke.json"] if runtime_smoke.is_file() else []),
                *(["qualification/comparison.json", "qualification/feature_dataset_manifest.json", "qualification/catalog_summary.json"] if comparison is not None else []),
                *(
                    [f"qualification/{spec['paired_evidence']}", "qualification/canonical_evidence.json"]
                    if paired_evidence else []
                ),
            ],
        },
        "provenance": {
            "pra_version": "0.2.0rc1",
            "pra_commit": commit,
            "bundle_build_commit": commit,
            "hf_repo": spec["repo"],
            "hf_collection": COLLECTION,
            "license": spec["license"],
            "license_note": "Router and project terms do not replace the base-model or dataset licenses.",
            "source_artifact": spec.get(
                "paired_evidence",
                f"docs/papers/shared/results/paper4_5_runtime/quantized_bundles/{slug}",
            ),
            "selection_reason": (
                "Exact-identity paired natural-QA qualification, with routing research kept separate."
                if paired_evidence
                else "Fleet-compatible exact quantization with evidence isolated from other weight representations."
            ),
        },
        "trust": {
            "status": "eInnovator-qualified" if (paired_evidence or comparison is not None) else "eInnovator-maintained",
            "publisher": "EInnovator",
            "scope": f"exact {quantization} {engine.upper()} model identity; evidence does not transfer across revisions, engines, or quantizations",
        },
    }


def build_one(slug: str, *, force: bool = False) -> Path:
    spec = SPECS[slug]
    result_dir = RESULTS / slug
    has_router = spec.get("routing_artifact", True) and (result_dir / "comparison.json").is_file()
    comparison = json.loads((result_dir / "comparison.json").read_text(encoding="utf-8")) if has_router else None
    router_dir = ROUTERS / f"{slug}-combined-d128"
    router_config = json.loads((router_dir / "config.json").read_text(encoding="utf-8")) if has_router else None
    evidence_path = MLX_RESULTS / spec["paired_evidence"] if spec.get("paired_evidence") else None
    paired_evidence = import_mlx_paired_evidence(
        evidence_path,
        EvidenceIdentity(
            model_id=spec["base_model"], model_revision=spec["revision"],
            quantization=spec.get("quantization", "4bit"), engine="mlx-lm", engine_version="0.31.3",
            profile="balanced", execution_mode="Native Memory",
        ),
        hardware="Apple M4 Pro (Mac16,7), 48 GB",
        artifact_reference=f"qualification/{spec['paired_evidence']}",
    ) if evidence_path and evidence_path.is_file() else []
    structural_validation = QUANTIZED_RESULTS / slug / "structural_validation.json"
    runtime_smoke = QUANTIZED_RESULTS / slug / "runtime_smoke.json"
    output = BUNDLES / spec["repo"].split("/", 1)[1]
    with tempfile.TemporaryDirectory(prefix=f"pra-{slug}-bundle-") as temporary:
        run = Path(temporary)
        structural = run / "structural_adapter"
        structural.mkdir()
        (structural / "pra_adapter.yaml").write_text(
            yaml.safe_dump(_structural_adapter(spec), sort_keys=False), encoding="utf-8"
        )
        if has_router:
            shutil.copytree(router_dir, run / "learned_adapters/combined-router-d128")
        qualification = run / "qualification"
        qualification.mkdir()
        if has_router:
            shutil.copy2(result_dir / "comparison.json", qualification / "comparison.json")
            shutil.copy2(result_dir / "feature_dataset_manifest.json", qualification / "feature_dataset_manifest.json")
            shutil.copy2(RESULTS / "summary.json", qualification / "catalog_summary.json")
        if paired_evidence:
            shutil.copy2(evidence_path, qualification / spec["paired_evidence"])
            canonical = [
                canonicalize_paired_transport_evidence(row).serialize_for_control_plane()
                for row in paired_evidence
                if row["dataset"] == "combined"
            ]
            (qualification / "canonical_evidence.json").write_text(
                json.dumps(canonical, indent=2) + "\n", encoding="utf-8"
            )
        if structural_validation.is_file():
            shutil.copy2(structural_validation, qualification / "structural_validation.json")
        if runtime_smoke.is_file():
            shutil.copy2(runtime_smoke, qualification / "runtime_smoke.json")
        (run / "pra.yaml").write_text(
            yaml.safe_dump(
                _manifest(slug, spec, comparison, router_config, paired_evidence), sort_keys=False
            ),
            encoding="utf-8",
        )
        bundle = BundleBuilder().build(run, output, force=force)
        bundle.validate()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    slugs = SPECS if args.model == "all" else (args.model,)
    for slug in slugs:
        print(build_one(slug, force=args.force))


if __name__ == "__main__":
    main()
