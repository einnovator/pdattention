"""Build the first evidence-bounded public PRA bundle from checked-in artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from pra_hf.bundle import BundleBuilder


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
HF_REPO = "EInnovator/pra-qwen3-0.6b"
HF_COLLECTION = "EInnovator/pra-bundles-6a971e52093232f858e660f6"
ROUTER = ROOT / "artifacts/pra_hf/routers/qwen3-0.6b-qasper-d128"


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _structural_adapter() -> dict:
    return {
        "schema_version": 1,
        "base_model": {"id": BASE_MODEL, "revision": BASE_REVISION},
        "architecture": {
            "model_type": "qwen",
            "architecture": "Qwen3ForCausalLM",
            "layers": 28,
            "hidden_size": 1024,
        },
        "mapping": {
            "layers": "model.layers",
            "attention": "self_attn",
            "q_proj": "q_proj",
            "k_proj": "k_proj",
            "v_proj": "v_proj",
            "o_proj": "o_proj",
        },
        "heads": {"query": 16, "kv": 8, "head_dim": 128},
        "position": {"type": "rope", "implementation": "native"},
        "topology": {"type": "homogeneous_global", "gqa": True},
    }


def _manifest(commit: str, router_config: dict) -> dict:
    return {
        "schema_version": 2,
        "base_model": {
            "id": BASE_MODEL,
            "revision": BASE_REVISION,
            "tokenizer_revision": BASE_REVISION,
            "architecture": "Qwen3ForCausalLM",
            "family": "qwen",
            "parameter_count": 596_049_920,
        },
        "structural_adapter": {
            "path": "structural_adapter",
            "status": "validated",
        },
        "learned_adapters": {
            "qasper-router-d128": {
                "path": "learned_adapters/qasper-router-d128",
                "type": "routing",
                "status": "validated-artifact",
                "default": True,
                "training": {
                    "dataset": "QASPER v0.3 validation",
                    "seed": router_config["training_seed"],
                    "steps": router_config["training_steps"],
                    "parameters": 262_144,
                    "objective": router_config["training_objective"],
                },
                "validation": router_config["metrics"],
            }
        },
        "profiles": {
            "quality": {
                "purpose": "Candidate maximum-quality profile; calibration is incomplete",
                "routing_adapter": None,
                "consumer_layers": "all eligible",
                "status": "CALIBRATION_PENDING",
                "recommended": False,
                "engine": "hf",
                "mode": "Selected Context",
            },
            "balanced": {
                "purpose": "Conservative training-free reference configuration",
                "routing_adapter": None,
                "consumer_layers": "all eligible",
                "status": "QUALIFIED",
                "recommended": True,
                "engine": "hf",
                "mode": "Selected Context",
            },
            "economy": {
                "purpose": "Reduced-consumer candidate; calibration is incomplete",
                "routing_adapter": None,
                "consumer_layers": "CALIBRATION_PENDING",
                "status": "CALIBRATION_PENDING",
                "recommended": False,
                "engine": "hf",
                "mode": "Selected Context",
            },
            "qasper-learned": {
                "purpose": "Research-only QASPER routing profile",
                "routing_adapter": "qasper-router-d128",
                "consumer_layers": "all eligible",
                "status": "RESEARCH",
                "recommended": False,
                "engine": "hf",
                "mode": "Selected Context",
            },
        },
        "runtime_compatibility": {
            "hf": {
                "selected_context": "validated",
                "native_memory": "controlled validation",
                "native_serving": "NOT_MEASURED",
                "recommended": "Selected Context; qualify Native Memory locally",
            },
            "mlx": {
                "selected_context": "validated",
                "native_memory": "NOT_MEASURED for this exact base revision",
                "native_serving": "NOT_MEASURED",
                "recommended": "Selected Context",
            },
            "vllm": {
                "selected_context": "validated",
                "native_memory": "NOT_MEASURED for this bundle",
                "native_serving": "NOT_MEASURED for this bundle",
                "recommended": "Selected Context",
            },
        },
        "engine_realizations": {
            "hf": {"bundle_consumed_by": "pra_runtime"},
            "remote_engines": {"bundle_consumed_by": "pra_gateway"},
        },
        "qualification": {
            "contract_version": 1,
            "status": "RESEARCH",
            "headline": [],
            "metrics": [
                {
                    "metric_class": "ROUTING_DIAGNOSTIC",
                    "model_id": BASE_MODEL,
                    "model_revision": BASE_REVISION,
                    "engine": "huggingface_eager 4.55.4",
                    "hardware": "NVIDIA GeForce GTX 950M",
                    "dataset": "allenai/qasper",
                    "workload": "QASPER v0.3 validation routing",
                    "mode": "Learned routing",
                    "quality": router_config["metrics"]["AUC0-30"],
                    "visible_tokens": None,
                    "ttft_ms": None,
                    "throughput": None,
                    "sample_count": 16,
                    "profile": "qasper-learned",
                    "evidence_tier": "CONTROLLED",
                    "date": "2026-08-11",
                    "provenance": "learned_adapters/qasper-router-d128/config.json",
                },
                {
                    "metric_class": "SEMANTIC_EQUIVALENCE",
                    "model_id": BASE_MODEL,
                    "model_revision": BASE_REVISION,
                    "engine": "huggingface_eager",
                    "hardware": "NVIDIA GeForce GTX 950M",
                    "dataset": "paper4_5_cross_model_diagnostic",
                    "workload": "three-case target-token diagnostic",
                    "mode": "Native Memory (hot)",
                    "quality": 0.436360141805,
                    "visible_tokens": None,
                    "ttft_ms": None,
                    "throughput": None,
                    "sample_count": 3,
                    "profile": "balanced",
                    "evidence_tier": "SMOKE",
                    "date": "2026-08-28",
                    "provenance": "qualification/profile_evidence.json",
                },
            ],
            "training": {
                "datasets": "QASPER",
                "seed": router_config["training_seed"],
                "method": router_config["training_objective"],
                "parameter_count": 262_144,
                "base_revision": BASE_REVISION,
            },
            "limitations": [
                "This is a research/reference bundle for mechanism reproduction, not the flagship production-economics demonstration.",
                "The routing cohort is small and QASPER-specific; transfer routing is not a production claim.",
                "Native Memory has controlled HF evidence only for this exact model revision and must be requalified per engine, quantization, and hardware.",
                "Native Serving is not qualified by this bundle.",
            ],
        },
        "provenance": {
            "pra_version": "0.2.0rc1",
            "pra_commit": commit,
            "bundle_build_commit": commit,
            "hf_repo": HF_REPO,
            "hf_collection": HF_COLLECTION,
            "license": "apache-2.0",
            "source_artifact": "artifacts/pra_hf/routers/qwen3-0.6b-qasper-d128",
            "selection_reason": "Best existing artifact with immutable base identity, weights, training provenance, QASPER validation metrics, and current runtime support.",
        },
        "trust": {
            "status": "eInnovator-qualified",
            "publisher": "EInnovator",
            "scope": "controlled QASPER routing and HF structural validation",
        },
    }


def build(output: Path, *, force: bool = False) -> Path:
    router_config = json.loads((ROUTER / "config.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="pra-qwen3-bundle-") as temporary:
        run = Path(temporary)
        structural = run / "structural_adapter"
        structural.mkdir()
        (structural / "pra_adapter.yaml").write_text(
            yaml.safe_dump(_structural_adapter(), sort_keys=False), encoding="utf-8"
        )
        shutil.copytree(ROUTER, run / "learned_adapters/qasper-router-d128")
        qualification = run / "qualification"
        qualification.mkdir()
        evidence = {
            "source": "docs/papers/shared/results/paper4_5_runtime/layer_profiles/layer_calibration_candidates.csv",
            "quality_metric": "mean_target_token_probability",
            "quality_score": 0.436360141805,
            "sample_count": 3,
            "evidence_tier": "SMOKE",
        }
        (qualification / "profile_evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        manifest = _manifest(_git_commit(), router_config)
        manifest["qualification"]["artifacts"] = ["qualification/profile_evidence.json"]
        (run / "pra.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )
        bundle = BundleBuilder().build(run, output, force=force)
        bundle.validate()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/pra_hf/bundles/pra-qwen3-0.6b")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build(args.output, force=args.force))


if __name__ == "__main__":
    main()
