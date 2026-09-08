"""Qualify generic and learned routing through selected text and native MLX K/V.

The runner freezes one candidate set per example.  Generic cosine and a trained
router select from that same set, then each selection is replayed both as an
ordinary visible prefix and as native K/V.  This separates selection quality
from the representation/transport question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from statistics import fmean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.paper4_5_runtime.extract_mlx_catalog_features import (  # noqa: E402
    load_split_examples,
)
from experiments.paper4_5_runtime.run_qwen35_qualification import (  # noqa: E402
    _scores,
    select_chunk_indices,
)
from experiments.mac_scaling.run_mlx_profile_scaling import (  # noqa: E402
    runtime_metadata,
)
from experiments.paper6_2_mlx.run_answer_quality_pressure import (  # noqa: E402
    _answer_logprob,
    _metrics,
)
from experiments.paper6_2_mlx.run_matched_e0_e2 import (  # noqa: E402
    _cache_snapshot,
    _generate_timed,
    _restore_cache,
)
from pra_hf.router import PRARouter  # noqa: E402


NO_PRA = "NO_PRA"
SELECTED_GENERIC = "PRA_SELECTED_CONTEXT_NO_ADAPTOR"
NATIVE_GENERIC = "PRA_NATIVE_MEMORY_NO_ADAPTOR"
SELECTED_BUNDLE = "PRA_SELECTED_CONTEXT_BUNDLE"
NATIVE_BUNDLE = "PRA_NATIVE_MEMORY_BUNDLE"

TRANSPORT_PAIRS = {
    NATIVE_GENERIC: SELECTED_GENERIC,
    NATIVE_BUNDLE: SELECTED_BUNDLE,
}


def selected_token_ids(
    source_ids: list[int],
    spans: list[tuple[int, int]],
    selected: list[int],
) -> list[int]:
    """Concatenate selected chunks once in source order without duplicates."""

    ordered = sorted(set(map(int, selected)), key=lambda index: spans[index])
    return [
        int(token_id)
        for index in ordered
        for token_id in source_ids[spans[index][0] : spans[index][1]]
    ]


def question_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    """Build the query suffix shared by every visible/native condition."""

    content = f"Answer briefly and directly.\nQuestion: {question.strip()}\nAnswer:"
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(rendered, Mapping):
            rendered = rendered["input_ids"]
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return [int(token_id) for token_id in rendered]
    return list(tokenizer.encode(content, add_special_tokens=False))


def annotate_transport_pairs(rows: list[dict[str, Any]]) -> None:
    """Attach exact output parity only to native rows with a visible twin."""

    by_key = {
        (str(row["example_id"]), str(row["condition"])): row
        for row in rows
    }
    for row in rows:
        visible_condition = TRANSPORT_PAIRS.get(str(row["condition"]))
        if visible_condition is None:
            continue
        visible = by_key.get((str(row["example_id"]), visible_condition))
        if visible is None:
            continue
        row["transport_pair"] = visible_condition
        row["sequence_agreement_vs_selected_text"] = float(
            row["output_token_ids"] == visible["output_token_ids"]
        )


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["condition"])), []).append(row)
    output = []
    for (dataset, condition), values in sorted(groups.items()):
        parity = [
            float(row["sequence_agreement_vs_selected_text"])
            for row in values
            if "sequence_agreement_vs_selected_text" in row
        ]
        output.append(
            {
                "dataset": dataset,
                "condition": condition,
                "examples": len(values),
                "seeds": len({int(row["seed"]) for row in values}),
                "exact_match": fmean(float(row["exact_match"]) for row in values),
                "token_f1": fmean(float(row["token_f1"]) for row in values),
                "evidence_recall": fmean(
                    float(row["evidence_recall"]) for row in values
                ),
                "visible_prompt_tokens": fmean(
                    float(row["visible_prompt_tokens"]) for row in values
                ),
                "selected_native_kv_tokens": fmean(
                    float(row["selected_native_kv_tokens"]) for row in values
                ),
                "ttft_ms": fmean(float(row["ttft_ms"]) for row in values),
                "itl_ms": fmean(float(row["itl_ms"]) for row in values),
                "completion_latency_ms": fmean(
                    float(row["completion_latency_ms"]) for row in values
                ),
                "transport_sequence_agreement": fmean(parity) if parity else None,
            }
        )
    return output


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run a restartable five-condition qualification on one exact MLX model."""

    import mlx.core as mx
    import mlx_lm
    from huggingface_hub import model_info
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    from pra_mlx import (
        encode_native_memory,
        install_qwen3_segmented_attention,
        make_native_prompt_cache,
    )

    manifest = json.loads(
        (args.feature_dir / "feature_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    features = torch.load(
        args.feature_dir / "router_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )
    expected_model = str(manifest["model_id"])
    expected_revision = str(manifest["model_revision"])
    if (args.model, args.revision) != (expected_model, expected_revision):
        raise ValueError(
            "Feature/model identity mismatch: "
            f"{expected_model}@{expected_revision} != {args.model}@{args.revision}"
        )

    examples_per_dataset = int(manifest["splits"]["test"]["dataset_counts"]["hotpotqa"])
    examples = load_split_examples(
        args.cache_dir,
        examples_per_dataset,
        int(manifest["splits"]["test"]["offset"]),
        int(manifest["seed"]),
    )
    by_id = {str(example["id"]): example for example in examples}
    router = PRARouter.from_pretrained(args.router)
    model, tokenizer_wrapper = load(args.model, revision=args.revision)
    tokenizer = getattr(tokenizer_wrapper, "_tokenizer", tokenizer_wrapper)
    resolved = model_info(args.model, revision=args.revision)
    installed_layers = install_qwen3_segmented_attention(model)
    decoder = model if hasattr(model, "layers") else getattr(model, "model", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise TypeError("MLX model does not expose a decoder layer stack.")
    layer_count = len(decoder.layers)

    checkpoint = args.output.with_suffix(".jsonl")
    rows = []
    if args.resume and checkpoint.is_file():
        rows = [
            json.loads(line)
            for line in checkpoint.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    completed = {
        (str(row["example_id"]), str(row["condition"])) for row in rows
    }

    for feature_index, feature in enumerate(features[: args.max_examples or None]):
        example = by_id[str(feature["example_id"])]
        source_ids = list(
            tokenizer(example["source"], add_special_tokens=False).input_ids
        )[: args.max_source_tokens]
        usable = [
            index
            for index, span in enumerate(feature["chunk_spans"])
            if int(span[0]) < len(source_ids)
        ]
        spans = [
            (
                int(feature["chunk_spans"][index][0]),
                min(int(feature["chunk_spans"][index][1]), len(source_ids)),
            )
            for index in usable
        ]
        trimmed = {
            **feature,
            "memory_gists": feature["memory_gists"][usable],
            "positive_mask": feature["positive_mask"][usable],
            "chunk_spans": spans,
        }
        token_budget = max(1, math.ceil(len(source_ids) * args.selected_fraction))
        generic = select_chunk_indices(
            _scores(trimmed, None), spans, token_budget=token_budget
        )
        learned = select_chunk_indices(
            _scores(trimmed, router), spans, token_budget=token_budget
        )
        query = question_prompt_ids(tokenizer, str(example["question"]))
        answer = list(
            tokenizer.encode(" " + str(example["answer"]), add_special_tokens=False)
        )
        selection_specs = {
            NO_PRA: (source_ids, list(range(len(spans))), None, "full visible context"),
            SELECTED_GENERIC: (
                selected_token_ids(source_ids, spans, generic), generic, None,
                "generic cosine",
            ),
            NATIVE_GENERIC: (
                selected_token_ids(source_ids, spans, generic), generic, None,
                "generic cosine",
            ),
            SELECTED_BUNDLE: (
                selected_token_ids(source_ids, spans, learned), learned,
                args.router.name, "learned router",
            ),
            NATIVE_BUNDLE: (
                selected_token_ids(source_ids, spans, learned), learned,
                args.router.name, "learned router",
            ),
        }

        for condition, (selected_ids, selected, adaptor, routing) in selection_specs.items():
            key = (str(feature["example_id"]), condition)
            if key in completed:
                continue
            is_native = condition in TRANSPORT_PAIRS
            mx.reset_peak_memory()
            encode_started = time.perf_counter()
            memory = None
            if is_native:
                memory = encode_native_memory(model, selected_ids)

                def cache_factory():
                    return make_native_prompt_cache(model, memory)

            else:
                ordinary = make_prompt_cache(model)
                encoded = model(mx.array(selected_ids, dtype=mx.int32)[None], cache=ordinary)
                mx.eval(encoded)
                ordinary_states = _cache_snapshot(ordinary)

                def cache_factory():
                    return _restore_cache(model, ordinary_states)

            encode_ms = (time.perf_counter() - encode_started) * 1_000.0
            logprob_started = time.perf_counter()
            logprob = _answer_logprob(model, query, answer, cache_factory())
            logprob_ms = (time.perf_counter() - logprob_started) * 1_000.0
            generated = _generate_timed(
                model, tokenizer, query, cache_factory(), args.max_new_tokens
            )
            exact, token_f1 = _metrics(generated["output"], str(example["answer"]))
            positive = set(
                map(
                    int,
                    trimmed["positive_mask"].nonzero(as_tuple=False).flatten().tolist(),
                )
            )
            selected_set = set(map(int, selected))
            evidence_recall = len(positive & selected_set) / max(len(positive), 1)
            row = {
                "schema_version": "pra-mlx-router-native-qualification-v1",
                "model_id": args.model,
                "model_revision": resolved.sha,
                "quantization": args.quantization,
                "engine": f"mlx-lm {getattr(mlx_lm, '__version__', 'unknown')}",
                "profile": args.profile,
                "dataset": feature["dataset"],
                "seed": int(manifest["seed"]),
                "example_id": feature["example_id"],
                "condition": condition,
                "pra_enabled": condition != NO_PRA,
                "representation": "native_kv" if is_native else "selected_text",
                "adaptor": adaptor,
                "routing": routing,
                "selection_sha256": hashlib.sha256(
                    json.dumps(selected, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "candidate_source_tokens": len(source_ids),
                "selected_source_tokens": len(selected_ids),
                "visible_prompt_tokens": len(query) if is_native else len(selected_ids) + len(query),
                "selected_native_kv_tokens": (
                    len(selected_ids) * layer_count if is_native else 0
                ),
                "consumer_layers": list(range(layer_count)) if is_native else [],
                "active_detail_bytes": int(memory.nbytes) if memory is not None else 0,
                "selected_chunk_count": len(selected),
                "candidate_chunk_count": len(spans),
                "evidence_recall": evidence_recall,
                "answer": example["answer"],
                "output": generated["output"],
                "output_token_ids": list(map(int, generated["output_token_ids"])),
                "exact_match": exact,
                "token_f1": token_f1,
                "gold_answer_logprob": logprob,
                "encode_ms": encode_ms,
                "gold_logprob_latency_ms": logprob_ms,
                "ttft_ms": generated["ttft_ms"],
                "itl_ms": generated["itl_ms"],
                "completion_latency_ms": generated["completion_latency_ms"],
                "generated_tokens": generated["generated_tokens"],
                "peak_unified_memory_bytes": int(mx.get_peak_memory()),
                "evidence_tier": "MODEL_BACKED_NATURAL_QA_CONTROLLED",
            }
            rows.append(row)
            completed.add(key)
            _append(checkpoint, row)
            print(
                f"[{feature_index + 1}/{len(features)}] {feature['dataset']} "
                f"{condition} F1={token_f1:.3f} TTFT={generated['ttft_ms']:.1f}ms",
                flush=True,
            )
            del memory
            mx.clear_cache()

    annotate_transport_pairs(rows)
    payload = {
        "schema_version": "pra-mlx-router-native-qualification-v1",
        "protocol": "frozen-candidate generic/learned routing with paired E0/E2 replay",
        "model_id": args.model,
        "model_revision": resolved.sha,
        "quantization": args.quantization,
        "profile": args.profile,
        "feature_manifest": str(args.feature_dir / "feature_dataset_manifest.json"),
        "router": str(args.router),
        "router_config": router.artifact_config(),
        "router_weights_sha256": _file_sha256(args.router / router.WEIGHTS_NAME),
        "selected_fraction": args.selected_fraction,
        "max_source_tokens": args.max_source_tokens,
        "max_new_tokens": args.max_new_tokens,
        "layer_count": layer_count,
        "segmented_layers_patched": installed_layers,
        "runtime": runtime_metadata(),
        "rows": rows,
        "aggregate": _aggregate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quantization", default="MLX-4bit")
    parser.add_argument("--profile", default="BALANCED")
    parser.add_argument("--selected-fraction", type=float, default=0.20)
    parser.add_argument("--max-source-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.selected_fraction <= 1:
        parser.error("--selected-fraction must be in (0, 1].")
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"rows": len(result["rows"]), "model": result["model_id"]}))
