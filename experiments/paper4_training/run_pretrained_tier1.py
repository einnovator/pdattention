"""Run the Paper 4 ordinary-language pretrained consumer-learning gate.

The experiment reloads one sub-0.5B checkpoint for Frozen PRA, Consumer LoRA,
and Interface LoRA. Oracle evidence identity is held fixed so the comparison
tests native-K/V consumption rather than retrieval. Interface training rebuilds
graph-backed reference K/V every step; learned routing remains disabled until
consumer learning passes on held-out WikiText continuations.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pra_hf.pra_aware_training import (
    HFAdaptationRegime,
    hf_parameter_summary,
    install_hf_adaptation_regime,
)
from pra_torch.hf import PRAHFConfig, inject_pra


SEEDS = (11, 23, 37, 53, 71)
REGIMES: tuple[HFAdaptationRegime, ...] = (
    "frozen_pra",
    "consumer_lora",
    "interface_lora",
)


@dataclass(frozen=True)
class PretrainedMemoryExample:
    """One continuation target with oracle evidence and matched alternatives."""

    example_id: str
    prompt: str
    answer: str
    evidence: str
    whole_parent: str
    distractor: str


def pra_layers(layer_count: int, spacing: int = 4) -> tuple[int, ...]:
    """Return the preregistered one-PRA-layer-per-four-block topology."""

    if layer_count <= 0 or spacing <= 0:
        raise ValueError("Layer count and spacing must be positive.")
    selected = tuple(range(spacing - 1, layer_count, spacing))
    return selected or (layer_count - 1,)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_examples(dataset_dir: Path) -> list[PretrainedMemoryExample]:
    """Join generated WikiText questions to their source documents."""

    documents = {row["uri"]: row["text"] for row in _read_jsonl(dataset_dir / "documents.jsonl")}
    examples = []
    for row in _read_jsonl(dataset_dir / "questions.jsonl"):
        uris = list(row["reference_uris"])
        part_ids = list(row.get("part_ids", range(1, len(uris) + 1)))
        relevant = set(row.get("relevant_part_ids", part_ids[:1]))
        evidence_uris = [
            uri for uri, part_id in zip(uris, part_ids, strict=True) if part_id in relevant
        ]
        distractor_uris = [uri for uri in uris if uri not in evidence_uris]
        if not evidence_uris or any(uri not in documents for uri in uris):
            continue
        evidence = "\n\n".join(documents[uri] for uri in evidence_uris)
        whole_parent = "\n\n".join(documents[uri] for uri in uris)
        distractor = "\n\n".join(documents[uri] for uri in distractor_uris)
        if not distractor:
            distractor = documents[next(uri for uri in documents if uri not in uris)]
        prompt = str(row["prompt"])
        for marker in ("<REF_1>", "<REF_2>", "<REF_3>", "<REF_4>", "<REF_5>"):
            prompt = prompt.replace(marker, "")
        examples.append(
            PretrainedMemoryExample(
                example_id=str(row["id"]),
                prompt=" ".join(prompt.split()),
                answer=str(row["answer"]),
                evidence=evidence,
                whole_parent=whole_parent,
                distractor=distractor,
            )
        )
    if not examples:
        raise ValueError(f"No joined examples found in {dataset_dir}.")
    return examples


def split_examples(
    examples: list[PretrainedMemoryExample], seed: int, train_count: int, validation_count: int
) -> tuple[list[PretrainedMemoryExample], list[PretrainedMemoryExample]]:
    """Create one deterministic, disjoint train/validation partition."""

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    required = train_count + validation_count
    if len(shuffled) < required:
        raise ValueError(f"Need {required} examples but found {len(shuffled)}.")
    return shuffled[:train_count], shuffled[train_count:required]


def _bounded_ids(tokenizer, text: str, limit: int) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    return ids[-limit:]


def _query_batch(tokenizer, prompt: str, answer: str, limit: int, device: torch.device):
    prefix = list(tokenizer.encode(prompt.rstrip() + " ", add_special_tokens=True))
    target = list(tokenizer.encode(answer, add_special_tokens=False))
    if not target:
        raise ValueError("Continuation target tokenized to an empty sequence.")
    if len(prefix) + len(target) > limit:
        prefix = prefix[-max(1, limit - len(target)) :]
    tokens = torch.tensor([prefix + target], dtype=torch.long, device=device)
    labels = torch.tensor(
        [[-100] * len(prefix) + target], dtype=torch.long, device=device
    )
    return tokens, labels


def _forward_metrics(model, input_ids: torch.Tensor, labels: torch.Tensor):
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
    )
    logits = outputs.logits[:, :-1].float()
    targets = labels[:, 1:]
    active = targets.ne(-100)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
    )
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions[active] == targets[active]).float().mean()
    return loss, accuracy


def _condition_source(example: PretrainedMemoryExample, condition: str) -> str | None:
    if condition == "none":
        return None
    if condition == "evidence_only":
        return example.evidence
    if condition == "whole_parent":
        return example.whole_parent
    if condition == "matched_distractor":
        return example.distractor
    raise ValueError(f"Unknown condition: {condition}")


def _example_forward(
    handle,
    tokenizer,
    example: PretrainedMemoryExample,
    *,
    condition: str,
    context_limit: int,
    reference_limit: int,
    differentiable: bool,
):
    """Evaluate one answer target under visible or native evidence."""

    handle.cache.clear()
    prompt = example.prompt
    if condition == "ordinary_full_context":
        source = None
        prompt = f"Context:\n{example.whole_parent}\n\n{prompt}"
    else:
        source = _condition_source(example, condition)
    handle.set_memory_enabled(source is not None)
    if source is not None:
        reference_ids = torch.tensor(
            [_bounded_ids(tokenizer, source, reference_limit)],
            dtype=torch.long,
            device=handle.device,
        )
        handle.add_reference(
            "memory://oracle/current",
            reference_ids,
            text=source,
            differentiable=differentiable,
        )
    input_ids, labels = _query_batch(
        tokenizer, prompt, example.answer, context_limit, handle.device
    )
    return _forward_metrics(handle.model, input_ids, labels)


def evaluate(handle, tokenizer, examples, args) -> dict[str, dict[str, float]]:
    """Measure held-out native-memory utility and ordinary-LM retention."""

    rows: dict[str, list[tuple[float, float]]] = {
        condition: []
        for condition in (
            "none",
            "matched_distractor",
            "evidence_only",
            "whole_parent",
            "ordinary_full_context",
        )
    }
    handle.model.eval()
    with torch.no_grad():
        for example in examples:
            for condition in rows:
                loss, accuracy = _example_forward(
                    handle,
                    tokenizer,
                    example,
                    condition=condition,
                    context_limit=args.context_limit,
                    reference_limit=args.reference_limit,
                    differentiable=False,
                )
                rows[condition].append((float(loss.cpu()), float(accuracy.cpu())))
    handle.cache.clear()
    return {
        condition: {
            "examples": len(values),
            "answer_nll": fmean(value[0] for value in values),
            "answer_perplexity": math.exp(min(20.0, fmean(value[0] for value in values))),
            "answer_token_accuracy": fmean(value[1] for value in values),
        }
        for condition, values in rows.items()
    }


def _gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    groups = {"query_output": 0.0, "key_value": 0.0, "mlp": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm().cpu())
        if ".k_proj." in name or ".v_proj." in name:
            groups["key_value"] += value
        elif ".mlp." in name or ".feed_forward." in name:
            groups["mlp"] += value
        elif ".q_proj." in name or ".o_proj." in name:
            groups["query_output"] += value
    return groups


def train_regime(handle, tokenizer, examples, args) -> dict:
    """Optimize one LoRA scope with fixed oracle evidence and LM rehearsal."""

    parameters = [parameter for parameter in handle.model.parameters() if parameter.requires_grad]
    if not parameters:
        return {"steps": 0, "tokens": 0, "loss_history": [], "gradient_norms": {}}
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    handle.model.train()
    history = []
    gradient_max = {"query_output": 0.0, "key_value": 0.0, "mlp": 0.0}
    tokens = 0
    started = time.perf_counter()
    for step in range(args.steps):
        example = examples[step % len(examples)]
        ordinary = args.ordinary_every > 0 and (step + 1) % args.ordinary_every == 0
        condition = "ordinary_full_context" if ordinary else "evidence_only"
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy = _example_forward(
            handle,
            tokenizer,
            example,
            condition=condition,
            context_limit=args.context_limit,
            reference_limit=args.reference_limit,
            differentiable=not ordinary,
        )
        loss.backward()
        gradients = _gradient_summary(handle.model)
        for name, value in gradients.items():
            gradient_max[name] = max(gradient_max[name], value)
        torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        optimizer.step()
        tokens += args.context_limit
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            history.append(
                {
                    "step": step + 1,
                    "condition": condition,
                    "loss": float(loss.detach().cpu()),
                    "answer_token_accuracy": float(accuracy.detach().cpu()),
                }
            )
        handle.cache.clear()
    if args.device.startswith("mps"):
        torch.mps.synchronize()
    return {
        "steps": args.steps,
        "tokens_upper_bound": tokens,
        "wall_seconds": time.perf_counter() - started,
        "loss_history": history,
        "gradient_norm_max": gradient_max,
    }


def _load_model(args, regime: HFAdaptationRegime):
    dtype = torch.float32 if args.device.startswith("mps") else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    selected = pra_layers(int(model.config.num_hidden_layers), args.pra_spacing)
    config = PRAHFConfig(
        layer_ids=selected,
        encoding_block_tokens=args.reference_limit,
        routing_chunk_tokens=args.reference_limit,
        max_materialized_memory_tokens=args.reference_limit,
        top_k_references=1,
        top_k_chunks_per_reference=1,
        trigger_threshold=-1.0e9,
        kv_cache_residency="gpu",
        collect_detailed_timing=False,
    )
    handle = inject_pra(model, config)
    targets = install_hf_adaptation_regime(
        model,
        regime,
        pra_layer_ids=selected,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    return handle, tokenizer, selected, targets


def _adapter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def run(args: argparse.Namespace) -> dict:
    examples = load_examples(args.dataset_dir)
    train_examples, validation_examples = split_examples(
        examples, args.split_seed, args.train_examples, args.validation_examples
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for regime in args.regime:
        handle, tokenizer, selected, targets = _load_model(args, regime)
        before = evaluate(handle, tokenizer, validation_examples, args)
        training = train_regime(handle, tokenizer, train_examples, args)
        after = evaluate(handle, tokenizer, validation_examples, args)
        checkpoint = None
        if targets:
            checkpoint = args.output_dir / f"{regime}_adapter.pt"
            torch.save(_adapter_state(handle.model), checkpoint)
        result = {
            "regime": regime,
            "model_id": args.model,
            "revision": args.revision,
            "pra_layers": list(selected),
            "router_trained": False,
            "selection_policy": "single_oracle_evidence_entry",
            "targets": list(targets),
            "parameters": hf_parameter_summary(handle.model),
            "before": before,
            "training": training,
            "after": after,
            "evidence_nll_delta": (
                after["evidence_only"]["answer_nll"]
                - before["evidence_only"]["answer_nll"]
            ),
            "ordinary_retention_nll_delta": (
                after["ordinary_full_context"]["answer_nll"]
                - before["ordinary_full_context"]["answer_nll"]
            ),
            "checkpoint": str(checkpoint) if checkpoint else None,
        }
        results.append(result)
        (args.output_dir / f"{regime}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        del handle
        if args.device.startswith("mps"):
            torch.mps.empty_cache()
    positive = [
        row
        for row in results
        if row["regime"] != "frozen_pra"
        and row["evidence_nll_delta"] <= -args.minimum_evidence_nll_gain
        and row["ordinary_retention_nll_delta"] <= args.maximum_retention_nll_loss
    ]
    payload = {
        "schema_version": "paper4-pretrained-tier1-v1",
        "experiment": "ordinary_language_pretrained_consumer_learning",
        "evidence_tier": "PRETRAINED_MODEL_HELD_OUT_WIKITEXT_MEMORY",
        "model_id": args.model,
        "revision": args.revision,
        "device": args.device,
        "seeds": [args.split_seed],
        "split_seed": args.split_seed,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "dataset_dir"}
        },
        "results": results,
        "consumer_gate": "PASS" if positive else "FAIL",
        "routing_stage": "ENABLED_NEXT" if positive else "BLOCKED_CONSUMER_NOT_REPLICATED",
        "passing_regimes": [row["regime"] for row in positive],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/wikitext2_references_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/papers/shared/results/paper4_training/pretrained_tier1"))
    parser.add_argument("--regime", action="append", choices=REGIMES)
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--train-examples", type=int, default=128)
    parser.add_argument("--validation-examples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--ordinary-every", type=int, default=4)
    parser.add_argument("--context-limit", type=int, default=192)
    parser.add_argument("--reference-limit", type=int, default=128)
    parser.add_argument("--pra-spacing", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--minimum-evidence-nll-gain", type=float, default=0.05)
    parser.add_argument("--maximum-retention-nll-loss", type=float, default=0.10)
    args = parser.parse_args()
    args.regime = args.regime or list(REGIMES)
    return args


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"consumer_gate": result["consumer_gate"], "passing_regimes": result["passing_regimes"]}, indent=2))
