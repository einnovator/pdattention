"""Extract restartable frozen-backbone routing features from MLX checkpoints.

The extractor observes the normalized attention input at one decoder layer. It
does not replace the model's attention implementation, masks, RoPE, or sliding
window policy. References are encoded in independent model-safe blocks and
reduced to the same 32-token mean gists used by the PRA-HF router studies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.native_kv_benchmarks import load_qasper_papers
from experiments.paper2_hf.qa.run_smoke import evidence_token_spans
from pra_torch.hf import token_span_from_offsets


SPLITS = {
    "validation": (0, 8),
    "test": (8, 16),
    "train": (24, 24),
}


def _hotpot_examples(cache_dir: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rows = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation",
        cache_dir=str(cache_dir),
    )
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    examples = []
    for index in indices:
        row = rows[index]
        supporting = {
            (str(title), int(sentence_id))
            for title, sentence_id in zip(
                row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
            )
        }
        segments, evidence = [], []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            for sentence_id, sentence in enumerate(sentences):
                segment = f"{title}: {str(sentence).strip()}"
                segments.append(segment)
                if (str(title), sentence_id) in supporting:
                    evidence.append(segment)
        if evidence:
            examples.append(
                {
                    "dataset": "hotpotqa",
                    "id": str(row["id"]),
                    "question": str(row["question"]),
                    "answer": str(row["answer"]),
                    "source": "\n".join(segments),
                    "evidence": evidence,
                }
            )
        if len(examples) == count:
            return examples
    raise RuntimeError(f"HotpotQA yielded only {len(examples)} usable examples.")


def _qasper_examples(cache_dir: Path, count: int, seed: int) -> list[dict[str, Any]]:
    papers = load_qasper_papers("validation", cache_dir=cache_dir)
    candidates = []
    for paper_id, paper in papers.items():
        paragraphs = [str(paper.get("abstract", ""))]
        for section in paper.get("full_text", []):
            paragraphs.extend(str(value) for value in section.get("paragraphs", []))
        for qa in paper.get("qas", []):
            for annotation in qa.get("answers", []):
                answer = annotation.get("answer", {})
                evidence = [
                    str(value)
                    for value in answer.get("evidence", [])
                    if str(value).strip()
                ]
                if answer.get("yes_no") is None or not evidence:
                    continue
                candidates.append(
                    {
                        "dataset": "qasper",
                        "id": f"{paper_id}:{qa.get('question_id', '')}",
                        "question": str(qa["question"]),
                        "answer": "yes" if answer["yes_no"] else "no",
                        "source": "\n".join(dict.fromkeys([*evidence, *paragraphs])),
                        "evidence": evidence,
                    }
                )
                break
    random.Random(seed).shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(f"QASPER yielded only {len(candidates)} usable examples.")
    return candidates[:count]


def load_split_examples(
    cache_dir: Path, count: int, offset: int, seed: int
) -> list[dict[str, Any]]:
    """Load deterministic identity-disjoint slices from both QA datasets."""
    stop = int(offset) + int(count)
    return [
        *_hotpot_examples(cache_dir, stop, seed)[offset:stop],
        *_qasper_examples(cache_dir / "qasper", stop, seed + 1)[offset:stop],
    ]


def _prompt_with_question_span(tokenizer: Any, question: str, max_tokens: int):
    question = question.strip()
    content = f"Answer briefly and directly.\nQuestion: {question}"
    if tokenizer.chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        rendered = content
    marker = f"Question: {question}"
    marker_start = rendered.rfind(marker)
    if marker_start < 0:
        raise ValueError("Rendered prompt does not contain the exact question marker.")
    char_start = marker_start + len("Question: ")
    char_end = char_start + len(question)
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_tokens,
    )
    tokenizer.truncation_side = previous
    offsets = encoded.pop("offset_mapping")[0].tolist()
    return encoded, token_span_from_offsets(offsets, char_start, char_end)


def lexical_chunk_scores(
    tokenizer: Any,
    source: str,
    question: str,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """Return token-set Jaccard overlap for each routing chunk."""
    source_ids = tokenizer(source, add_special_tokens=False).input_ids
    question_ids = set(tokenizer(question, add_special_tokens=False).input_ids)
    scores = []
    for start, end in spans:
        chunk_ids = set(source_ids[start:end])
        union = question_ids | chunk_ids
        scores.append(len(question_ids & chunk_ids) / len(union) if union else 0.0)
    return torch.tensor(scores, dtype=torch.float32)


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def _hardware() -> str:
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True
        )
        fields = []
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith(("Chip:", "Memory:")):
                fields.append(stripped)
        return "; ".join(fields) or platform.platform()
    except (OSError, subprocess.SubprocessError):
        return platform.platform()


class MLXAttentionInputCapture:
    """Capture one layer's native normalized attention input as CPU tensors."""

    def __init__(self, model: Any, layer_id: int) -> None:
        import mlx.nn as nn

        layers = model.model.layers
        self.layer_id = layer_id if layer_id >= 0 else len(layers) + layer_id
        if not 0 <= self.layer_id < len(layers):
            raise ValueError(f"Routing layer {layer_id} is outside {len(layers)} layers.")
        self.model = model
        self.sink: list[Any] = []
        layer = layers[self.layer_id]
        self.original = layer.input_layernorm
        sink = self.sink
        original = self.original

        class CaptureNorm(nn.Module):
            def __call__(self, values):
                normalized = original(values)
                sink.append(normalized)
                return normalized

        layer.input_layernorm = CaptureNorm()

    def __call__(self, token_ids: list[int]) -> torch.Tensor:
        import mlx.core as mx

        if not token_ids:
            raise ValueError("Cannot extract routing states from an empty token sequence.")
        self.sink.clear()
        output = self.model.model(mx.array(token_ids, dtype=mx.int32)[None])
        if len(self.sink) != 1:
            raise RuntimeError(
                f"Expected one layer capture, observed {len(self.sink)} at layer {self.layer_id}."
            )
        captured = self.sink[0].astype(mx.float32)
        mx.eval(output, captured)
        return torch.from_numpy(np.asarray(captured))[0].clone()

    def close(self) -> None:
        self.model.model.layers[self.layer_id].input_layernorm = self.original


def _source_features(
    capture: MLXAttentionInputCapture,
    token_ids: list[int],
    *,
    block_tokens: int,
    chunk_tokens: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    gists: list[torch.Tensor] = []
    spans: list[tuple[int, int]] = []
    for block_start in range(0, len(token_ids), block_tokens):
        block_end = min(block_start + block_tokens, len(token_ids))
        hidden = capture(token_ids[block_start:block_end])
        for local_start in range(0, block_end - block_start, chunk_tokens):
            local_end = min(local_start + chunk_tokens, block_end - block_start)
            gists.append(hidden[local_start:local_end].mean(dim=0))
            spans.append((block_start + local_start, block_start + local_end))
    return torch.stack(gists), spans


def _feature_for_example(
    capture: MLXAttentionInputCapture,
    tokenizer: Any,
    example: dict[str, Any],
    *,
    model_id: str,
    model_revision: str,
    block_tokens: int,
    chunk_tokens: int,
) -> dict[str, Any]:
    source_ids = tokenizer(
        example["source"], add_special_tokens=False
    ).input_ids
    memory_gists, spans = _source_features(
        capture,
        source_ids,
        block_tokens=block_tokens,
        chunk_tokens=chunk_tokens,
    )
    encoded, question_span = _prompt_with_question_span(
        tokenizer, example["question"], 128
    )
    prompt_ids = encoded.input_ids[0].tolist()
    query_hidden = capture(prompt_ids)
    evidence_spans = evidence_token_spans(
        tokenizer, example["source"], example["evidence"]
    )
    positive = torch.tensor(
        [
            any(max(start, evidence_start) < min(end, evidence_end)
                for evidence_start, evidence_end in evidence_spans)
            for start, end in spans
        ],
        dtype=torch.bool,
    )
    if not bool(positive.any()):
        raise RuntimeError(
            f"No evidence-positive chunk for {example['dataset']}:{example['id']}"
        )
    return {
        "dataset": example["dataset"],
        "example_id": example["id"],
        "model_id": model_id,
        "model_revision": model_revision,
        "queries": {"last": query_hidden[-1]},
        "memory_gists": memory_gists,
        "positive_mask": positive,
        "normalized_positions": torch.tensor(
            [((start + end) / 2) / max(len(source_ids), 1) for start, end in spans],
            dtype=torch.float32,
        ),
        "lexical_scores": lexical_chunk_scores(
            tokenizer, example["source"], example["question"], spans
        ),
        "chunk_spans": spans,
        "evidence_spans": evidence_spans,
        "source_tokens": len(source_ids),
        "prompt_tokens": len(prompt_ids),
        "question_tokens": question_span[1] - question_span[0],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_dir = args.output_dir / "checkpoint_rows"
    rows_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    model, tokenizer = load(args.model_id, revision=args.model_revision)
    model.freeze()
    capture = MLXAttentionInputCapture(model, args.routing_layer)
    split_manifest: dict[str, Any] = {}
    identities: dict[str, set[tuple[str, str]]] = {}
    try:
        for split, (offset, default_count) in SPLITS.items():
            count = getattr(args, f"{split}_examples") or default_count
            examples = load_split_examples(args.cache_dir, count, offset, args.seed)
            features = []
            for index, example in enumerate(examples):
                checkpoint = rows_dir / f"{split}-{index:04d}-{_safe_key(example['id'])}.pt"
                if checkpoint.exists() and not args.force:
                    feature = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    if (
                        feature.get("model_id") != args.model_id
                        or feature.get("model_revision") != args.model_revision
                        or feature.get("example_id") != example["id"]
                    ):
                        raise RuntimeError(f"Stale checkpoint identity: {checkpoint}")
                else:
                    feature = _feature_for_example(
                        capture,
                        tokenizer,
                        example,
                        model_id=args.model_id,
                        model_revision=args.model_revision,
                        block_tokens=args.encoding_block_tokens,
                        chunk_tokens=args.routing_chunk_tokens,
                    )
                    feature["split"] = split
                    torch.save(feature, checkpoint)
                features.append(feature)
                print(
                    f"[{split} {index + 1}/{len(examples)}] "
                    f"{example['dataset']} {example['id']}",
                    flush=True,
                )
                mx.clear_cache()
            output = args.output_dir / f"router_features_{split}.pt"
            torch.save(features, output)
            identities[split] = {
                (feature["dataset"], feature["example_id"]) for feature in features
            }
            split_manifest[split] = {
                "path": output.name,
                "sha256": _sha256(output),
                "examples": len(features),
                "dataset_counts": {
                    dataset: sum(row["dataset"] == dataset for row in features)
                    for dataset in ("hotpotqa", "qasper")
                },
                "offset": offset,
                "positive_chunks": sum(
                    int(row["positive_mask"].sum().item()) for row in features
                ),
                "candidate_chunks": sum(len(row["positive_mask"]) for row in features),
            }
    finally:
        capture.close()
    leakage = {
        f"{left}_{right}": len(identities[left] & identities[right])
        for index, left in enumerate(SPLITS)
        for right in list(SPLITS)[index + 1 :]
    }
    if any(leakage.values()):
        raise RuntimeError(f"Feature split identity leakage: {leakage}")
    manifest = {
        "protocol": "MLX frozen-backbone attention-input routing features",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "routing_layer": capture.layer_id,
        "feature_source": "attention_input_hidden_state_after_native_norm",
        "feature_width": int(next(iter(features))["memory_gists"].shape[-1]),
        "encoding_block_tokens": args.encoding_block_tokens,
        "routing_chunk_tokens": args.routing_chunk_tokens,
        "gist_mode": "mean",
        "query_strategy": "last",
        "seed": args.seed,
        "splits": split_manifest,
        "identity_leakage": leakage,
        "base_parameters_trainable": 0,
        "runtime": {
            "hardware": _hardware(),
            "python": platform.python_version(),
            "mlx": getattr(mlx, "__version__", "unknown"),
            "mlx_lm": getattr(mlx_lm, "__version__", "unknown"),
            "git_sha": _git_sha(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    (args.output_dir / "feature_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--routing-layer", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--encoding-block-tokens", type=int, default=128)
    parser.add_argument("--routing-chunk-tokens", type=int, default=32)
    parser.add_argument("--train-examples", type=int)
    parser.add_argument("--validation-examples", type=int)
    parser.add_argument("--test-examples", type=int)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pra")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
