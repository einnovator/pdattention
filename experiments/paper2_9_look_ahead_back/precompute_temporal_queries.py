"""Capture aligned tokenwise routing queries for the frozen Paper 2.8 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper2_5_iterative_pra.precompute_native_qk_features import (
    PARENT_TOKENS,
)
from experiments.paper2_8_qk_compression.run_gated_study import (
    MODEL_ID,
    MODEL_REVISION,
)
from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.routing.run_query_strategies import (
    _prompt_with_question_span,
    load_split_examples,
)
from pra_hf.multihop_routing_data import load_multihop_routing_examples
from pra_torch.hf import ATTENTION_INPUT_HIDDEN_STATE, PRAHFConfig, inject_pra


DATASETS = ("hotpotqa", "qasper", "2wikimultihopqa", "musique")
SPLITS = ("validation", "test")
CAPTURE_LAYERS = (8, 18, 27)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_shard_name(identity: str) -> str:
    """Map dataset identities to portable filenames without losing provenance."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".pt"


def source_feature_path(root: Path, dataset: str, split: str) -> Path:
    """Locate the inherited Paper 2.8 source-key cache for one cohort."""
    if dataset in {"hotpotqa", "qasper"}:
        return root / f"native_qk_features_{split}.pt"
    directory = "2wiki" if dataset == "2wikimultihopqa" else "musique"
    return root / "multi_dataset" / directory / f"native_qk_features_{split}.pt"


def load_source_features(root: Path) -> dict[tuple[str, str], list[dict]]:
    """Load inherited rows once and preserve their frozen order and identities."""
    output = {}
    shared = {
        split: torch.load(
            source_feature_path(root, "hotpotqa", split),
            map_location="cpu",
            weights_only=False,
        )
        for split in SPLITS
    }
    for split in SPLITS:
        for dataset in ("hotpotqa", "qasper"):
            output[(dataset, split)] = [
                row for row in shared[split] if row["dataset"] == dataset
            ]
        for dataset in ("2wikimultihopqa", "musique"):
            output[(dataset, split)] = torch.load(
                source_feature_path(root, dataset, split),
                map_location="cpu",
                weights_only=False,
            )
    return output


def load_questions(args, source_groups) -> dict[tuple[str, str], dict]:
    """Rehydrate question text and verify every inherited identity resolves once."""
    output: dict[tuple[str, str], dict] = {}
    for split, offset in (("validation", 0), ("test", 8)):
        count = max(
            len(source_groups[("hotpotqa", split)]),
            len(source_groups[("qasper", split)]),
        )
        for row in load_split_examples(args.cache_dir, count, offset, args.seed):
            output[(row["dataset"], row["id"])] = row
    for example in load_multihop_routing_examples(
        args.annotations, args.twowiki_dev, args.musique_dev
    ):
        output[(example.dataset, example.example_id)] = example.as_feature_example()
    expected = {
        (row["dataset"], row["example_id"])
        for rows in source_groups.values()
        for row in rows
    }
    missing = sorted(expected - set(output))
    if missing:
        raise RuntimeError(f"Could not rehydrate {len(missing)} identities: {missing[:3]}")
    return output


def initialize_model(args, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        attn_implementation="eager",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handle = inject_pra(
        model,
        PRAHFConfig(
            # Only the inherited Paper 2.8 routing layer is wrapped. Earlier
            # states come from the backbone's native hidden-state tuple.
            layer_ids=(max(args.capture_layers),),
            model_max_context_tokens=PARENT_TOKENS,
            max_prompt_direct_tokens=128,
            encoding_block_tokens=PARENT_TOKENS,
            routing_chunk_tokens=PARENT_TOKENS,
            routing_representation=ATTENTION_INPUT_HIDDEN_STATE,
            gist_mode="segment_mean",
            gists_per_chunk=8,
            max_materialized_memory_tokens=PARENT_TOKENS,
            top_k_references=1,
            top_k_chunks_per_reference=8,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=device.type == "cuda",
            collect_detailed_timing=False,
            collect_routing_metrics=False,
        ),
    )
    return tokenizer, model, handle


@torch.no_grad()
def capture_temporal_query(
    handle,
    tokenizer,
    question: str,
    device: torch.device,
    *,
    capture_layers=CAPTURE_LAYERS,
) -> dict:
    """Capture prompt-aligned pre-RoPE queries without changing backbone states."""
    encoded, question_span = _prompt_with_question_span(tokenizer, question, 128)
    encoded = encoded.to(device)
    positions = torch.arange(encoded.input_ids.shape[1], device=device).unsqueeze(0)
    adapter = next(iter(handle.adapters.values()))
    handle.set_memory_enabled(False)
    adapter.begin_capture(positions)
    outputs = handle.model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        position_ids=positions,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    capture = adapter.consume_capture()
    final_adapter = adapter
    decoder_layers = handle.model.model.layers
    hidden_by_layer = {
        int(layer): decoder_layers[int(layer)].input_layernorm(
            outputs.hidden_states[int(layer)]
        )[0]
        for layer in capture_layers
    }
    hidden_by_layer[0] = decoder_layers[adapter.layer_idx].input_layernorm(
        outputs.hidden_states[0]
    )[0]
    pre_query_by_layer = {}
    for layer, hidden in hidden_by_layer.items():
        projected, _, _ = final_adapter.project_qkv(hidden.unsqueeze(0))
        pre_query_by_layer[str(layer)] = projected[0].permute(1, 0, 2).to(
            "cpu", torch.float16
        )
    final_capture = capture.pre_query[0].permute(1, 0, 2)
    parity_error = float(
        (pre_query_by_layer[str(adapter.layer_idx)].float() - final_capture.cpu().float())
        .abs()
        .max()
    )
    if parity_error > 2e-3:
        raise RuntimeError(f"Layer projection parity failed: {parity_error:.6g}")
    return {
        "prompt_token_ids": encoded.input_ids[0].to("cpu", torch.int32),
        "prompt_attention_mask": encoded.attention_mask[0].to("cpu", torch.bool),
        "question_span": tuple(int(value) for value in question_span),
        "pre_query_by_layer": pre_query_by_layer,
        "projection_capture_error": parity_error,
        "capture_method": "single_adapter_native_hidden_state_tuple_v1",
    }


def paper2_8_parity_error(
    temporal: dict,
    source: dict,
    final_adapter,
) -> tuple[float, float]:
    """Compare the new B=1 state with the inherited Paper 2.8 final query."""
    current = temporal["pre_query_by_layer"][str(final_adapter.layer_idx)][-1].float()
    if "query_pre_query" in source:
        expected = source["query_pre_query"].float()
    else:
        parameter = next(final_adapter.parameters())
        hidden = source["query_hidden"].to(
            device=parameter.device, dtype=parameter.dtype
        ).view(1, 1, -1)
        projected, _, _ = final_adapter.project_qkv(hidden)
        expected = projected[0, :, 0].detach().cpu().float()
    absolute = float((current - expected).abs().max())
    relative = absolute / max(float(expected.abs().max()), 1e-8)
    return absolute, relative


def run(args: argparse.Namespace) -> dict:
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_groups = load_source_features(args.paper2_8_root)
    questions = load_questions(args, source_groups)
    device = torch.device(args.device)
    tokenizer, model, handle = initialize_model(args, device)
    final_adapter = max(handle.adapters.values(), key=lambda adapter: adapter.layer_idx)
    artifacts = {}
    parity_rows = []
    for dataset in args.datasets:
        artifacts[dataset] = {}
        for split in args.splits:
            source_rows = source_groups[(dataset, split)]
            rows = []
            shard_dir = args.output_root / ".feature_shards" / dataset / split
            shard_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(source_rows, start=1):
                identity = source["example_id"]
                shard = shard_dir / identity_shard_name(identity)
                if args.resume and shard.exists():
                    temporal = torch.load(shard, map_location="cpu", weights_only=False)
                    action = "resume"
                    if temporal.get("capture_method") != (
                        "single_adapter_native_hidden_state_tuple_v1"
                    ):
                        temporal = capture_temporal_query(
                            handle,
                            tokenizer,
                            questions[(dataset, identity)]["question"],
                            device,
                            capture_layers=tuple(args.capture_layers),
                        )
                        temporal.update(
                            {
                                "dataset": dataset,
                                "split": split,
                                "example_id": identity,
                            }
                        )
                        torch.save(temporal, shard)
                        action = "recapture"
                else:
                    temporal = capture_temporal_query(
                        handle,
                        tokenizer,
                        questions[(dataset, identity)]["question"],
                        device,
                        capture_layers=tuple(args.capture_layers),
                    )
                    temporal.update(
                        {
                            "dataset": dataset,
                            "split": split,
                            "example_id": identity,
                        }
                    )
                    torch.save(temporal, shard)
                    action = "capture"
                absolute, relative = paper2_8_parity_error(
                    temporal, source, final_adapter
                )
                if absolute > args.parity_tolerance:
                    raise RuntimeError(
                        f"B=1 parity failed for {dataset}:{identity}: {absolute:.6g}"
                    )
                temporal["paper2_8_b1_max_abs_error"] = absolute
                temporal["paper2_8_b1_max_relative_error"] = relative
                rows.append(temporal)
                parity_rows.append((dataset, split, identity, absolute, relative))
                print(
                    f"[temporal {action} {dataset} {split} {index}/{len(source_rows)}] "
                    f"tokens={len(temporal['prompt_token_ids'])} b1_error={absolute:.3g}",
                    flush=True,
                )
            dataset_dir = args.output_root / dataset
            dataset_dir.mkdir(parents=True, exist_ok=True)
            path = dataset_dir / f"temporal_query_features_{split}.pt"
            torch.save(rows, path)
            artifacts[dataset][split] = {
                "path": str(path.relative_to(ROOT)),
                "examples": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "tracked": False,
            }
    errors = [row[3] for row in parity_rows]
    manifest = {
        "schema_version": "1.0",
        "runtime": runtime_metadata(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "backbone_frozen": True,
        "memory_enabled_during_capture": False,
        "capture_layers": list(args.capture_layers),
        "embedding_ablation_layer_id": 0,
        "routing_layer": final_adapter.layer_idx,
        "representation": "prompt_tokenwise_layer27_pre_rope_query",
        "capture_method": "single_adapter_native_hidden_state_tuple_v1",
        "paper2_8_memory_side_unchanged": True,
        "paper2_8_b1_parity": {
            "examples": len(errors),
            "maximum_absolute_error": max(errors),
            "mean_absolute_error": sum(errors) / len(errors),
            "tolerance": args.parity_tolerance,
            "passed": max(errors) <= args.parity_tolerance,
        },
        "counts": dict(Counter(row[0] for row in parity_rows)),
        "feature_artifacts": artifacts,
    }
    (args.output_root / "temporal_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    inherited = Path(r"D:/git/rd/pdattention-iter-gist/data/.paper2_5_datasets")
    result_root = ROOT / "docs/papers/shared/results/paper2_9_look_ahead_back"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--capture-layers", nargs="+", type=int, default=CAPTURE_LAYERS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parity-tolerance",
        type=float,
        default=2e-2,
        help="FP16 replay tolerance; routing-selection parity is checked separately.",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/.hf_cache")
    parser.add_argument(
        "--paper2-8-root",
        type=Path,
        default=ROOT / "docs/papers/shared/results/paper2_8_qk_compression",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ROOT / "data/paper2_7_query_facets/annotations.jsonl",
    )
    parser.add_argument("--twowiki-dev", type=Path, default=inherited / "2wiki/dev.json")
    parser.add_argument(
        "--musique-dev",
        type=Path,
        default=inherited / "musique/data/musique_ans_v1.0_dev.jsonl",
    )
    parser.add_argument("--output-root", type=Path, default=result_root)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
