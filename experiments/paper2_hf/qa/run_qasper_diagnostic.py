"""Diagnose 32-token PRA decoding and QASPER polarity calibration.

This follow-up reuses the frozen Paper 2 identities, router, and memory-use
checkpoints. It does not modify the eight-token behavioral-judge package.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.paper2_hf.common.artifacts import runtime_metadata
from experiments.paper2_hf.qa.run_last14_combo import (
    Variant,
    _compact_gold_scores,
    _configure_variant,
    _freeze_backbone,
    _load_checkpoint,
    _prepare_records,
    _save_checkpoint,
    last_band_layers,
)
from experiments.paper2_hf.qa.run_memory_gate import _activate
from experiments.paper2_hf.qa.run_smoke import answer_metrics
from experiments.paper2_hf.qwen.run_first_night import MODEL_ID, MODEL_REVISION
from experiments.paper2_hf.routing.run_query_strategies import load_split_examples
from pra_torch.hf import MEMORY_GATE_FIXED, PRAHFConfig, inject_pra, load_hf_routing_projection


ROOT = Path(__file__).resolve().parents[3]
SEEDS = (11, 23, 37, 53, 71)
TERMINAL_PUNCTUATION = re.compile(r"[.!?][\]\)}\"']*$")
WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class DiagnosticVariant:
    """One existing memory-use checkpoint family used without retraining."""

    condition: str
    variant: Variant
    checkpoint_dir: Path | None
    checkpoint_pattern: str | None

    def checkpoint(self, seed: int) -> Path | None:
        if self.checkpoint_dir is None or self.checkpoint_pattern is None:
            return None
        return self.checkpoint_dir / self.checkpoint_pattern.format(seed=seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def _word_set(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def _starts_with_polarity(text: str) -> str | None:
    match = re.match(r"^\s*(?:[-*#]+\s*)?(?:\*\*)?(yes|no)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


def _answer_contained(prediction: str, answer: str) -> bool:
    gold = _normalize(answer)
    return bool(gold) and gold in _normalize(prediction)


def _terminal_punctuation(text: str) -> bool:
    return bool(TERMINAL_PUNCTUATION.search(text.strip()))


def _token_ids(tokenizer, forms: Iterable[str]) -> list[int]:
    ids: set[int] = set()
    for form in forms:
        encoded = tokenizer(form, add_special_tokens=False).input_ids
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(f"No single-token forms found for {tuple(forms)}")
    return sorted(ids)


def polarity_token_ids(tokenizer) -> dict[str, list[int]]:
    """Return tokenizer-native one-token forms valid at the answer boundary."""
    return {
        "yes": _token_ids(tokenizer, ("yes", "Yes", " yes", " Yes")),
        "no": _token_ids(tokenizer, ("no", "No", " no", " No")),
    }


def _logsumexp_tokens(log_probs: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    return torch.logsumexp(log_probs[token_ids], dim=0)


@torch.no_grad()
def _score_answer(
    handle,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    answer_ids: torch.Tensor,
    polarity_ids: dict[str, list[int]],
    gold_polarity: str | None,
    device: torch.device,
) -> dict[str, Any]:
    """Score the gold sequence and expose the first-token yes/no decision."""
    prompt_tokens = int(prompt_ids.shape[1])
    full_ids = torch.cat((prompt_ids, answer_ids), dim=1).to(device)
    full_mask = torch.cat((prompt_mask, torch.ones_like(answer_ids)), dim=1).to(device)
    positions = torch.arange(
        prompt_tokens - 1,
        full_ids.shape[1] - 1,
        device=device,
        dtype=torch.long,
    )
    output = handle.model(
        input_ids=full_ids,
        attention_mask=full_mask,
        use_cache=False,
        logits_to_keep=positions,
    )
    logits = output.logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    targets = answer_ids.to(device)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    first_logits = logits[0, 0]
    first_log_probs = log_probs[0, 0]
    first_target = int(targets[0, 0])
    wrong_logits = first_logits.clone()
    wrong_logits[first_target] = float("-inf")
    yes_logp = _logsumexp_tokens(first_log_probs, polarity_ids["yes"])
    no_logp = _logsumexp_tokens(first_log_probs, polarity_ids["no"])
    yes_minus_no = float((yes_logp - no_logp).cpu())
    gold_margin = None
    if gold_polarity in {"yes", "no"}:
        gold_margin = yes_minus_no if gold_polarity == "yes" else -yes_minus_no
    return {
        "gold_sequence_logprob": float(token_log_probs.sum().cpu()),
        "gold_mean_token_logprob": float(token_log_probs.mean().cpu()),
        "gold_first_token_probability": float(first_logits.softmax(dim=-1)[first_target].cpu()),
        "gold_first_token_rank": int((first_logits > first_logits[first_target]).sum().cpu()) + 1,
        "gold_first_token_margin": float((first_logits[first_target] - wrong_logits.max()).cpu()),
        "yes_logprob": float(yes_logp.cpu()),
        "no_logprob": float(no_logp.cpu()),
        "yes_minus_no_logprob": yes_minus_no,
        "gold_polarity_margin": gold_margin,
        "constrained_polarity": "yes" if yes_minus_no >= 0 else "no",
    }


@torch.no_grad()
def _generate_with_finish(
    handle,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Generate once and distinguish EOS completion from length termination."""
    encoded = {
        "input_ids": prompt_ids.to(device),
        "attention_mask": prompt_mask.to(device),
    }
    _sync(device)
    started = time.perf_counter()
    output = handle.model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    _sync(device)
    elapsed = time.perf_counter() - started
    continuation = output[0, prompt_ids.shape[1] :]
    tokens = [int(value) for value in continuation.detach().cpu().tolist()]
    eos_ids = tokenizer.eos_token_id
    eos_set = {int(eos_ids)} if isinstance(eos_ids, int) else {int(value) for value in eos_ids or []}
    eos_position = next((index for index, value in enumerate(tokens) if value in eos_set), None)
    generated_count = eos_position + 1 if eos_position is not None else len(tokens)
    emitted = eos_position is not None
    hit_limit = not emitted and len(tokens) >= max_new_tokens
    text = tokenizer.decode(tokens[:generated_count], skip_special_tokens=True).strip()
    return {
        "generated_text": text,
        "generated_token_count": generated_count,
        "decoded_token_count": len(tokenizer(text, add_special_tokens=False).input_ids),
        "finish_reason": "eos" if emitted else ("length" if hit_limit else "stopped"),
        "eos_emitted": emitted,
        "hit_max_new_tokens": hit_limit,
        "terminal_punctuation": _terminal_punctuation(text),
        "generation_seconds": elapsed,
    }


def _routing_features(record: dict[str, Any]) -> dict[str, float]:
    hits = list(record["routed"][0])
    scores = sorted((float(hit.chunk_score) for hit in hits), reverse=True)
    top1 = scores[0] if scores else float("-inf")
    top2 = scores[1] if len(scores) > 1 else top1
    probabilities = torch.tensor(scores, dtype=torch.float32).softmax(dim=0) if scores else torch.tensor([])
    entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item()) if scores else 0.0
    candidate_count = len(record["entry"].layer_memory[max(record["entry"].layer_memory)].chunks)
    return {
        "routing_top1_score": top1,
        "routing_top1_top2_margin": top1 - top2,
        "routing_selected_score_mean": statistics.fmean(scores) if scores else float("nan"),
        "routing_selected_entropy": entropy,
        "routing_selected_fraction": len(scores) / max(candidate_count, 1),
    }


def _rationale_consistent(text: str, polarity: str | None) -> bool | None:
    if polarity is None:
        return None
    remainder = re.sub(
        r"^\s*(?:[-*#]+\s*)?(?:\*\*)?(?:yes|no)(?:\*\*)?\b",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    opposite = "no" if polarity == "yes" else "yes"
    return re.search(rf"(?:^|[.!?;:]\s+)\**{opposite}\b", remainder, re.IGNORECASE) is None


def classify_error(row: dict[str, Any], no_context_text: str) -> tuple[str, list[str]]:
    """Assign one primary failure class while retaining overlapping audit flags."""
    text = row["generated_text"]
    answer = row["reference_answer"]
    polarity = _starts_with_polarity(text)
    gold = _normalize(answer)
    binary = gold in {"yes", "no"}
    flags: list[str] = []
    if row["hit_max_new_tokens"]:
        flags.append("generation_truncation")
    if _normalize(text) == _normalize(no_context_text):
        flags.append("no_behavioral_displacement")
    if binary and polarity is not None and polarity != gold:
        flags.append("polarity_inversion")
    question_words = _word_set(row["question"]) - {"what", "which", "who", "does", "did", "is", "are", "the", "a", "an"}
    text_words = _word_set(text)
    prompt_overlap = len(question_words & text_words) / max(len(text_words), 1)
    if prompt_overlap >= 0.65:
        flags.append("prompt_entity_repetition")
    evidence = " ".join(row.get("evidence", []))
    source = row.get("source", "")
    if not binary and _normalize(text) and _normalize(text) in _normalize(source) and not _answer_contained(text, answer):
        flags.append("wrong_related_entity")
    elif not binary and len(_word_set(text) & _word_set(evidence)) >= 2:
        flags.append("relation_near_miss")
    elif prompt_overlap >= 0.25:
        flags.append("topic_displacement")

    correct = _answer_contained(text, answer) or (binary and polarity == gold)
    if correct:
        primary = "correct_semantically_equivalent"
    elif "polarity_inversion" in flags:
        primary = "polarity_inversion"
    elif "no_behavioral_displacement" in flags:
        primary = "no_behavioral_displacement"
    elif "generation_truncation" in flags:
        primary = "generation_truncation"
    elif "prompt_entity_repetition" in flags:
        primary = "prompt_entity_repetition"
    elif "wrong_related_entity" in flags:
        primary = "wrong_related_entity"
    elif "relation_near_miss" in flags:
        primary = "relation_near_miss"
    elif "topic_displacement" in flags:
        primary = "topic_displacement"
    else:
        primary = "unrelated_degeneration"
    return primary, flags


def hotpot_relation_distance(row: dict[str, Any]) -> int | None:
    """Approximate progress toward Hotpot's target using reproducible lexical tiers.

    This is an audit heuristic, not a dataset-native hop annotation: supporting
    sentences containing the gold answer are treated as final evidence, other
    supporting sentences as first-hop evidence, and the remaining source as
    retrieved-memory context.
    """
    if row["dataset"] != "hotpotqa":
        return None
    text = row["generated_text"]
    answer = row["reference_answer"]
    if _answer_contained(text, answer):
        return 0
    generated = _word_set(text)
    evidence = row.get("evidence", [])
    final_evidence = [item for item in evidence if _normalize(answer) in _normalize(item)]
    first_hop = [item for item in evidence if item not in final_evidence]
    if any(len(generated & _word_set(item)) >= 2 for item in final_evidence):
        return 1
    if any(len(generated & _word_set(item)) >= 2 for item in first_hop):
        return 2
    if len(generated & _word_set(row.get("source", ""))) >= 2:
        return 3
    if len(generated & _word_set(row["question"])) >= 2:
        return 4
    return 5


def _evaluate_one(
    handle,
    tokenizer,
    record: dict[str, Any],
    condition: str,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    max_new_tokens: int,
    polarity_ids: dict[str, list[int]],
    device: torch.device,
    *,
    seed: int | None,
    variant: str,
    gate_alpha: float = 1.0,
) -> dict[str, Any]:
    gold = _normalize(record["example"]["answer"])
    gold_polarity = gold if gold in {"yes", "no"} else None
    scoring = _score_answer(
        handle,
        prompt_ids,
        prompt_mask,
        record["answer_ids"],
        polarity_ids,
        gold_polarity,
        device,
    )
    generation = _generate_with_finish(
        handle,
        tokenizer,
        prompt_ids,
        prompt_mask,
        device,
        max_new_tokens,
    )
    metrics = answer_metrics(generation["generated_text"], record["example"]["answer"])
    observed = _starts_with_polarity(generation["generated_text"])
    return {
        "dataset": record["example"]["dataset"],
        "example_id": record["example"]["id"],
        "question": record["example"]["question"],
        "reference_answer": record["example"]["answer"],
        "condition": condition,
        "variant": variant,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "gate_alpha": gate_alpha,
        **generation,
        **scoring,
        **metrics,
        "answer_contained": _answer_contained(generation["generated_text"], record["example"]["answer"]),
        "format_correct": observed is not None if gold_polarity is not None else None,
        "decoded_polarity": observed,
        "polarity_correct": observed == gold_polarity if gold_polarity is not None else None,
        "rationale_consistent": _rationale_consistent(generation["generated_text"], observed),
        **_routing_features(record),
    }


def fit_margin_calibrator(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Fit a two-parameter affine logistic calibration on identity-disjoint train rows."""
    margins = torch.tensor([float(row["yes_minus_no_logprob"]) for row in rows])
    targets = torch.tensor([1.0 if _normalize(row["reference_answer"]) == "yes" else 0.0 for row in rows])
    scale = torch.nn.Parameter(torch.tensor(1.0))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.25, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = scale * margins + bias
        loss = F.binary_cross_entropy_with_logits(logits, targets) + 1e-3 * ((scale - 1) ** 2 + bias**2)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"scale": float(scale.detach()), "bias": float(bias.detach()), "parameters": 2}


def apply_calibrator(row: dict[str, Any], calibrator: dict[str, float]) -> str:
    value = calibrator["scale"] * float(row["yes_minus_no_logprob"]) + calibrator["bias"]
    return "yes" if value >= 0 else "no"


def select_confidence_gate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Select a margin threshold on validation identities only.

    Each validation identity has a full-memory row (alpha 1) and a suppressed
    row (alpha 0). Candidate thresholds choose alpha 1 above the router's
    top1--top2 score margin and alpha 0 below it. Accuracy is primary, mean gold
    polarity margin is secondary, and broader memory use breaks exact ties.
    """
    by_example: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_example[row["example_id"]][float(row["gate_alpha"])] = row
    if any(set(values) != {0.0, 1.0} for values in by_example.values()):
        raise ValueError("Confidence-gate selection requires alpha 0 and 1 per identity.")
    margins = sorted(
        {float(values[1.0]["routing_top1_top2_margin"]) for values in by_example.values()}
    )
    thresholds = [float("-inf"), *margins, float("inf")]
    candidates = []
    for threshold in thresholds:
        chosen = []
        for values in by_example.values():
            full = values[1.0]
            alpha = 1.0 if float(full["routing_top1_top2_margin"]) >= threshold else 0.0
            chosen.append(values[alpha])
        candidates.append(
            {
                "threshold": threshold,
                "validation_accuracy": statistics.fmean(
                    float(row["constrained_polarity"] == _normalize(row["reference_answer"]))
                    for row in chosen
                ),
                "validation_mean_gold_margin": statistics.fmean(
                    float(row["gold_polarity_margin"]) for row in chosen
                ),
                "validation_full_memory_rate": statistics.fmean(float(row["gate_alpha"]) for row in chosen),
            }
        )
    return max(
        candidates,
        key=lambda row: (
            row["validation_accuracy"],
            row["validation_mean_gold_margin"],
            row["validation_full_memory_rate"],
        ),
    )


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"])].append(row)
    output = []
    for (dataset, condition), values in sorted(grouped.items()):
        binary = [row for row in values if _normalize(row["reference_answer"]) in {"yes", "no"}]
        classes = Counter(row["error_class"] for row in values)
        output.append(
            {
                "dataset": dataset,
                "condition": condition,
                "rows": len(values),
                "identities": len({row["example_id"] for row in values}),
                "seeds": len({row["seed"] for row in values if row["seed"] is not None}),
                "em": statistics.fmean(float(row["em"]) for row in values),
                "f1": statistics.fmean(float(row["f1"]) for row in values),
                "answer_containment": statistics.fmean(float(row["answer_contained"]) for row in values),
                "eos_rate": statistics.fmean(float(row["eos_emitted"]) for row in values),
                "hit_max_rate": statistics.fmean(float(row["hit_max_new_tokens"]) for row in values),
                "terminal_punctuation_rate": statistics.fmean(float(row["terminal_punctuation"]) for row in values),
                "mean_generated_tokens": statistics.fmean(float(row["generated_token_count"]) for row in values),
                "yes_no_format_rate": statistics.fmean(float(row["format_correct"]) for row in binary) if binary else None,
                "polarity_accuracy": statistics.fmean(float(row["polarity_correct"]) for row in binary) if binary else None,
                "mean_gold_polarity_margin": statistics.fmean(float(row["gold_polarity_margin"]) for row in binary) if binary else None,
                "error_counts": dict(sorted(classes.items())),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list, tuple))}
        for row in rows
    ]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _error_table(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in summary:
        counts = row["error_counts"]
        output.append(
            {
                "dataset": row["dataset"],
                "condition": row["condition"],
                "correct": counts.get("correct_semantically_equivalent", 0),
                "no_displacement": counts.get("no_behavioral_displacement", 0),
                "topic_or_relation_near_miss": counts.get("topic_displacement", 0)
                + counts.get("relation_near_miss", 0),
                "wrong_related_entity": counts.get("wrong_related_entity", 0),
                "polarity_inversion": counts.get("polarity_inversion", 0),
                "prompt_repetition": counts.get("prompt_entity_repetition", 0),
                "truncation": counts.get("generation_truncation", 0),
                "unrelated": counts.get("unrelated_degeneration", 0),
                "rows": row["rows"],
            }
        )
    return output


def _frozen_new_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("old_8_token_text") is not None:
            grouped[(row["dataset"], row["condition"])].append(row)
    for (dataset, condition), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "condition": condition,
                "rows": len(values),
                "generation_changed_rate": statistics.fmean(
                    row["generated_text"] != row["old_8_token_text"] for row in values
                ),
                "old_terminal_punctuation_rate": statistics.fmean(
                    _terminal_punctuation(row["old_8_token_text"]) for row in values
                ),
                "new_eos_rate": statistics.fmean(row["eos_emitted"] for row in values),
                "new_hit_max_rate": statistics.fmean(
                    row["hit_max_new_tokens"] for row in values
                ),
                "new_answer_containment": statistics.fmean(
                    row["answer_contained"] for row in values
                ),
            }
        )
    return output


def _plot(summary: list[dict[str, Any]], output_dir: Path) -> None:
    selected = {
        "native_no_context": "No context",
        "pra_routed_frozen": "Frozen PRA",
        "pra_oracle_frozen": "Oracle PRA",
        "pra_routed_residual_16": "Residual-16",
        "pra_routed_residual_16_qasper_trained": "Routed-trained R16",
        "pra_routed_lora_r32": "LoRA-32",
        "pra_routed_combo": "Combined",
    }
    qasper = [
        row for row in summary
        if row["dataset"] == "qasper" and row["condition"] in selected
    ]
    qasper.sort(key=lambda row: list(selected).index(row["condition"]))
    if not qasper:
        return
    labels = [selected[row["condition"]] for row in qasper]
    x = list(range(len(qasper)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].bar(x, [100 * (row["polarity_accuracy"] or 0) for row in qasper], color="#2878B5")
    axes[0].set_ylabel("Polarity accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(x, [100 * row["hit_max_rate"] for row in qasper], color="#D95F02")
    axes[1].set_ylabel("Hit 32-token limit (%)")
    axes[1].set_ylim(0, 100)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"qasper_polarity_and_finish.{suffix}", dpi=190)
    plt.close(figure)


def _write_artifacts(artifact: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generation_error_analysis.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "generation_error_rows.csv", artifact["rows"])
    _write_csv(output_dir / "generation_error_summary.csv", artifact["summary"])
    _write_csv(output_dir / "generation_error_table.csv", artifact["error_table"])
    _write_csv(
        output_dir / "frozen_8_vs_32_token_comparison.csv",
        artifact["frozen_8_vs_32_token_comparison"],
    )
    _write_csv(output_dir / "qasper_polarity_calibration.csv", artifact["calibration_rows"])
    _write_csv(
        output_dir / "qasper_confidence_gate_validation.csv",
        artifact["confidence_gate_validation_rows"],
    )
    _plot(artifact["summary"], output_dir)


def refresh_existing_artifact(path: Path) -> dict[str, Any]:
    """Rebuild derived audit fields and presentation files without model inference."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for row in artifact["rows"]:
        if row.get("variant") == "task_specific_readout":
            row["old_8_token_text"] = None
        if "hotpot_relation_distance" not in row or "source" in row:
            row["hotpot_relation_distance"] = hotpot_relation_distance(row)
        row.pop("source", None)
        row.pop("evidence", None)
    artifact["summary"] = _aggregate(artifact["rows"])
    artifact["error_table"] = _error_table(artifact["summary"])
    artifact["frozen_8_vs_32_token_comparison"] = _frozen_new_comparison(
        artifact["rows"]
    )
    _write_artifacts(artifact, path.parent)
    return artifact


def _variants(args) -> list[DiagnosticVariant]:
    last14 = args.last14_dir / "checkpoints"
    overnight = args.overnight_dir / "checkpoints"
    return [
        DiagnosticVariant("pra_routed_frozen", Variant("fixed"), None, None),
        DiagnosticVariant("pra_routed_residual_16", Variant("residual_16", residual_width=16), last14, "residual_16_seed{seed}.pt"),
        DiagnosticVariant("pra_routed_lora_r32", Variant("lora_o_r32_s64_lr1", lora_rank=32), overnight, "lora_o_r32_s64_lr1_seed{seed}.pt"),
        DiagnosticVariant(
            "pra_routed_combo",
            Variant("combo_residual_16_lora_r4", residual_width=16, lora_rank=4),
            last14,
            "combo_residual_16_lora_r4_seed{seed}.pt",
        ),
    ]


def _configure_loaded(handle, spec: DiagnosticVariant, seed: int) -> int:
    _freeze_backbone(handle)
    _configure_variant(handle, spec.variant, reset=True)
    checkpoint = spec.checkpoint(seed)
    if checkpoint is None:
        return 0
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    _, size = _load_checkpoint(checkpoint, handle, spec.variant)
    return size


def _train_routed_residual(
    handle,
    records: list[dict[str, Any]],
    layers: tuple[int, ...],
    seed: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
    checkpoint: Path,
) -> tuple[dict[str, Any], int]:
    """Train only residual-16 on the shipped router's selected QASPER memory."""
    variant = Variant("residual_16", residual_width=16)
    _freeze_backbone(handle)
    torch.manual_seed(seed)
    parameters = _configure_variant(handle, variant, reset=True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    losses = []
    started = time.perf_counter()
    handle.model.eval()
    for step in range(steps):
        record = records[order[step % len(order)]]
        _activate(handle, record, layers, "routed")
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _compact_gold_scores(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    _sync(device)
    report = {
        "variant": variant.name,
        "training_memory": "routed",
        "training_dataset": "qasper",
        "seed": seed,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "steps": steps,
        "learning_rate": learning_rate,
        "training_seconds": time.perf_counter() - started,
        "initial_loss_mean": statistics.fmean(losses[: len(records)]),
        "final_loss_mean": statistics.fmean(losses[-len(records) :]),
        "losses": losses,
    }
    return report, _save_checkpoint(checkpoint, handle, variant, seed, report)


def _old_generation_lookup(path: Path) -> dict[tuple[str, str, int | None], str]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    output: dict[tuple[str, str, int | None], str] = {}
    control_names = {"none": "native_no_context", "direct_text": "native_direct_evidence", "full_context": "native_full_context"}
    for row in artifact["test_control_rows"]:
        output[(row["example_id"], control_names[row["condition"]], None)] = row["generated_answer"]
    variant_names = {
        "fixed": "pra_routed_frozen",
        "residual_16": "pra_routed_residual_16",
        "combo_residual_16_lora_r4": "pra_routed_combo",
    }
    for row in artifact["test_rows"]:
        if row["variant"] not in variant_names:
            continue
        condition = variant_names[row["variant"]]
        if row["condition"] == "oracle" and row["variant"] == "fixed":
            condition = "pra_oracle_frozen"
        elif row["condition"] != "routed":
            continue
        output[(row["example_id"], condition, int(row["seed"]))] = row["generated_answer"]
    return output


def _load_examples(args, count: int, offset: int) -> list[dict[str, Any]]:
    return load_split_examples(args.cache_dir, count, offset, args.data_seed)


def _release_records(handle, records: list[dict[str, Any]], device: torch.device) -> None:
    """Release the large per-layer native-K/V entries before preparing another split."""
    handle.cache.clear()
    records.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _calibration_rows(
    handle,
    records: list[dict[str, Any]],
    layers: tuple[int, ...],
    polarity_ids: dict[str, list[int]],
    device: torch.device,
    split: str,
) -> list[dict[str, Any]]:
    output = []
    for record in records:
        _activate(handle, record, layers, "routed")
        scoring = _score_answer(
            handle,
            record["prompt_ids"],
            record["prompt_mask"],
            record["answer_ids"],
            polarity_ids,
            _normalize(record["example"]["answer"]),
            device,
        )
        output.append(
            {
                "split": split,
                "example_id": record["example"]["id"],
                "reference_answer": record["example"]["answer"],
                **scoring,
                **_routing_features(record),
            }
        )
    return output


def _gate_validation_rows(
    handle,
    records: list[dict[str, Any]],
    layers: tuple[int, ...],
    polarity_ids: dict[str, list[int]],
    device: torch.device,
) -> list[dict[str, Any]]:
    output = []
    for alpha in (0.0, 1.0):
        handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=alpha)
        for record in records:
            _activate(handle, record, layers, "routed")
            scoring = _score_answer(
                handle,
                record["prompt_ids"],
                record["prompt_mask"],
                record["answer_ids"],
                polarity_ids,
                _normalize(record["example"]["answer"]),
                device,
            )
            output.append(
                {
                    "example_id": record["example"]["id"],
                    "reference_answer": record["example"]["answer"],
                    "gate_alpha": alpha,
                    **scoring,
                    **_routing_features(record),
                }
            )
    handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)
    return output


def run(args) -> dict[str, Any]:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The full QASPER diagnostic requires CUDA.")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        attn_implementation="eager",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = load_hf_routing_projection(args.checkpoint, device=device)
    layers = last_band_layers(int(model.config.num_hidden_layers))
    handle = inject_pra(
        model,
        PRAHFConfig(
            layer_ids=layers,
            model_max_context_tokens=args.native_tokens,
            max_prompt_direct_tokens=args.prompt_tokens,
            encoding_block_tokens=128,
            routing_chunk_tokens=32,
            max_materialized_memory_tokens=args.memory_tokens,
            top_k_references=1,
            top_k_chunks_per_reference=3,
            trigger_threshold=float("-inf"),
            kv_cache_residency="cpu",
            kv_cache_pin_memory=True,
            kv_cache_non_blocking=True,
            collect_detailed_timing=True,
            collect_routing_metrics=True,
        ),
        routing_projection=projection,
    )
    polarity_ids = polarity_token_ids(tokenizer)
    old = _old_generation_lookup(args.last14_dir / "last14_combo.json")

    split_examples = {
        "train": _load_examples(args, args.train_examples, 0),
        "validation": _load_examples(args, args.validation_examples, args.validation_offset),
        "test": _load_examples(args, args.test_examples, args.test_offset),
    }
    identities = {name: [row["id"] for row in values] for name, values in split_examples.items()}
    if any(set(identities[left]) & set(identities[right]) for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Train, validation, and test identities must be disjoint.")

    specs = _variants(args)
    calibration_rows: list[dict[str, Any]] = []

    # Calibration needs only QASPER identities. Score each split before releasing
    # its native-K/V records so pinned host memory stays bounded.
    _configure_loaded(handle, specs[0], args.seeds[0])
    train_qasper = [row for row in split_examples["train"] if row["dataset"] == "qasper"]
    print(f"preparing QASPER calibration train references: {len(train_qasper)}", flush=True)
    train_records = _prepare_records(handle, tokenizer, train_qasper, layers, args, controls=False)
    calibration_rows.extend(
        _calibration_rows(handle, train_records, layers, polarity_ids, device, "train")
    )
    routed_training_reports = []
    routed_checkpoint_dir = args.output_dir / "checkpoints"
    for seed in args.seeds:
        report, checkpoint_bytes = _train_routed_residual(
            handle,
            train_records,
            layers,
            seed,
            args.routed_steps,
            args.routed_learning_rate,
            device,
            routed_checkpoint_dir / f"qasper_routed_residual_16_seed{seed}.pt",
        )
        report["checkpoint_bytes"] = checkpoint_bytes
        routed_training_reports.append(report)
        print(f"trained QASPER routed residual-16 seed={seed}", flush=True)
    _release_records(handle, train_records, device)

    _configure_loaded(handle, specs[0], args.seeds[0])
    validation_qasper = [
        row for row in split_examples["validation"] if row["dataset"] == "qasper"
    ]
    print(
        f"preparing QASPER calibration validation references: {len(validation_qasper)}",
        flush=True,
    )
    validation_records = _prepare_records(
        handle, tokenizer, validation_qasper, layers, args, controls=False
    )
    calibration_rows.extend(
        _calibration_rows(handle, validation_records, layers, polarity_ids, device, "validation")
    )
    calibrator = fit_margin_calibrator(
        [row for row in calibration_rows if row["split"] == "train"]
    )
    gate_validation_rows = _gate_validation_rows(
        handle, validation_records, layers, polarity_ids, device
    )
    confidence_gate = select_confidence_gate(gate_validation_rows)
    _release_records(handle, validation_records, device)

    print("preparing identity-disjoint test references", flush=True)
    test_records = _prepare_records(
        handle, tokenizer, split_examples["test"], layers, args, controls=True
    )
    print(f"prepared test: {len(test_records)}", flush=True)

    rows: list[dict[str, Any]] = []
    # Native controls are seed-independent and use exactly the same prompts as last-14.
    _configure_variant(handle, Variant("fixed"), reset=True)
    for record in test_records:
        controls = [
            ("native_no_context", record["prompt_ids"], record["prompt_mask"]),
            ("native_direct_evidence", record["direct_prompt_ids"], record["direct_prompt_mask"]),
        ]
        if record["full_context_complete"]:
            controls.append(("native_full_context", record["full_prompt_ids"], record["full_prompt_mask"]))
        for condition, prompt_ids, prompt_mask in controls:
            _activate(handle, record, layers, "none")
            row = _evaluate_one(
                handle,
                tokenizer,
                record,
                condition,
                prompt_ids,
                prompt_mask,
                args.new_tokens,
                polarity_ids,
                device,
                seed=None,
                variant="context_baseline",
            )
            row["old_8_token_text"] = old.get((row["example_id"], condition, None))
            rows.append(row)

    for spec in specs:
        seeds = (args.seeds[0],) if spec.variant.name == "fixed" else args.seeds
        for seed in seeds:
            checkpoint_bytes = _configure_loaded(handle, spec, seed)
            for record in test_records:
                for memory_condition, condition in (
                    (("routed", spec.condition)),
                    *(((("oracle", "pra_oracle_frozen")),) if spec.variant.name == "fixed" else ()),
                ):
                    _activate(handle, record, layers, memory_condition)
                    row = _evaluate_one(
                        handle,
                        tokenizer,
                        record,
                        condition,
                        record["prompt_ids"],
                        record["prompt_mask"],
                        args.new_tokens,
                        polarity_ids,
                        device,
                        seed=seed,
                        variant=spec.variant.name,
                    )
                    row["checkpoint_bytes"] = checkpoint_bytes
                    row["old_8_token_text"] = old.get((row["example_id"], condition, seed))
                    rows.append(row)
            print(f"decoded {spec.condition} seed={seed}", flush=True)

    routed_variant = Variant("residual_16", residual_width=16)
    for seed in args.seeds:
        _freeze_backbone(handle)
        _configure_variant(handle, routed_variant, reset=True)
        _, checkpoint_bytes = _load_checkpoint(
            routed_checkpoint_dir / f"qasper_routed_residual_16_seed{seed}.pt",
            handle,
            routed_variant,
        )
        for record in test_records:
            if record["example"]["dataset"] != "qasper":
                continue
            _activate(handle, record, layers, "routed")
            row = _evaluate_one(
                handle,
                tokenizer,
                record,
                "pra_routed_residual_16_qasper_trained",
                record["prompt_ids"],
                record["prompt_mask"],
                args.new_tokens,
                polarity_ids,
                device,
                seed=seed,
                variant="residual_16_qasper_routed",
            )
            row["checkpoint_bytes"] = checkpoint_bytes
            rows.append(row)
        print(f"decoded QASPER-routed residual-16 seed={seed}", flush=True)

    # Add decoding-only and calibrated decisions as explicit task-specific baselines.
    frozen_qasper = [
        row for row in rows
        if row["dataset"] == "qasper" and row["condition"] == "pra_routed_frozen"
    ]
    for source in frozen_qasper:
        for condition, decision in (
            ("pra_polarity_constrained", source["constrained_polarity"]),
            ("pra_polarity_calibrated", apply_calibrator(source, calibrator)),
        ):
            metrics = answer_metrics(decision, source["reference_answer"])
            rows.append(
                {
                    **source,
                    "condition": condition,
                    "variant": "task_specific_readout",
                    "generated_text": decision,
                    "generated_token_count": 1,
                    "decoded_token_count": 1,
                    "finish_reason": "constrained_decision",
                    "eos_emitted": False,
                    "hit_max_new_tokens": False,
                    "terminal_punctuation": False,
                    "generation_seconds": 0.0,
                    "old_8_token_text": None,
                    **metrics,
                    "answer_contained": decision == _normalize(source["reference_answer"]),
                    "format_correct": True,
                    "decoded_polarity": decision,
                    "polarity_correct": decision == _normalize(source["reference_answer"]),
                    "rationale_consistent": True,
                }
            )

    # Apply the validation-selected confidence gate to frozen routed PRA.
    _configure_loaded(handle, specs[0], args.seeds[0])
    for record in test_records:
        if record["example"]["dataset"] != "qasper":
            continue
        routing = _routing_features(record)
        alpha = (
            1.0
            if routing["routing_top1_top2_margin"] >= confidence_gate["threshold"]
            else 0.0
        )
        handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=alpha)
        _activate(handle, record, layers, "routed")
        row = _evaluate_one(
            handle,
            tokenizer,
            record,
            "pra_router_confidence_gate",
            record["prompt_ids"],
            record["prompt_mask"],
            args.new_tokens,
            polarity_ids,
            device,
            seed=args.seeds[0],
            variant="frozen_confidence_gate",
            gate_alpha=alpha,
        )
        rows.append(row)
    handle.configure_memory_gate(MEMORY_GATE_FIXED, initial_value=1.0)

    no_context = {
        row["example_id"]: row["generated_text"]
        for row in rows
        if row["condition"] == "native_no_context"
    }
    example_lookup = {
        row["id"]: row for values in split_examples.values() for row in values
    }
    for row in rows:
        example = example_lookup[row["example_id"]]
        row["source"] = example["source"]
        row["evidence"] = example["evidence"]
        primary, flags = classify_error(row, no_context.get(row["example_id"], ""))
        row["error_class"] = primary
        row["error_flags"] = flags
        row["hotpot_relation_distance"] = hotpot_relation_distance(row)
        row.pop("source", None)
        row.pop("evidence", None)

    summary = _aggregate(rows)
    error_table = _error_table(summary)
    frozen_new = _frozen_new_comparison(rows)
    artifact = {
        "runtime": runtime_metadata(),
        "protocol": "Paper 2 QASPER diagnostic follow-up; frozen eight-token judge package unchanged",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_new_tokens": args.new_tokens,
        "identities": identities,
        "identity_disjoint": True,
        "polarity_token_ids": polarity_ids,
        "calibrator": calibrator,
        "calibration_rows": calibration_rows,
        "confidence_gate": confidence_gate,
        "confidence_gate_validation_rows": gate_validation_rows,
        "routed_training_reports": routed_training_reports,
        "rows": rows,
        "summary": summary,
        "error_table": error_table,
        "frozen_8_vs_32_token_comparison": frozen_new,
        "scope_note": "HotpotQA rows are diagnostic only; iterative relation closure remains Paper 2.5 scope.",
    }
    _write_artifacts(artifact, args.output_dir)
    return artifact


def parse_args() -> argparse.Namespace:
    results = ROOT / "docs" / "papers" / "shared" / "results" / "paper2_hf"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--data-seed", type=int, default=20260811)
    parser.add_argument("--train-examples", type=int, default=12)
    parser.add_argument("--validation-examples", type=int, default=4)
    parser.add_argument("--validation-offset", type=int, default=12)
    parser.add_argument("--test-examples", type=int, default=8)
    parser.add_argument("--test-offset", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--routed-steps", type=int, default=32)
    parser.add_argument("--routed-learning-rate", type=float, default=1e-3)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--direct-text-tokens", type=int, default=640)
    parser.add_argument("--full-context-tokens", type=int, default=2048)
    parser.add_argument("--native-tokens", type=int, default=640)
    parser.add_argument("--memory-tokens", type=int, default=512)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / ".hf_cache")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=results / "routing" / "learned_adapter" / "checkpoints" / "asymmetric_linear_d128_last_joint_seed53_margin_exhaustive.pt",
    )
    parser.add_argument("--last14-dir", type=Path, default=results / "last14_combo")
    parser.add_argument("--overnight-dir", type=Path, default=results / "overnight_lora_sweep")
    parser.add_argument("--output-dir", type=Path, default=results / "error_analysis")
    parser.add_argument(
        "--refresh-existing",
        type=Path,
        help="Refresh tables and plots from an existing JSON artifact without inference.",
    )
    args = parser.parse_args()
    args.seeds = tuple(args.seeds)
    return args


if __name__ == "__main__":
    parsed = parse_args()
    result = (
        refresh_existing_artifact(parsed.refresh_existing)
        if parsed.refresh_existing
        else run(parsed)
    )
    print(json.dumps({"rows": len(result["rows"]), "summary_rows": len(result["summary"]), "calibrator": result["calibrator"]}, indent=2))
