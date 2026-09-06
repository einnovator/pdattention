"""Train and evaluate bounded cross-document native-K/V composition.

Retrieval, selected records, token order, and packed positions are frozen before
any composition arm runs. The base language model remains frozen. Only a
zero-initialized low-rank residual over request-local pre-RoPE K/V is trained.
Selective boundary re-encoding is the matched nonparametric alternative.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from experiments.paper3_2_rag.run_composition_fidelity import (
    _distribution_diagnostics,
    _execute,
    _hardware,
    _resolve_hf_revision,
    _runtime_versions,
    _token_segments,
)
from experiments.rag_vs_pra.datasets import load_multihop_rag, select_cohort
from experiments.rag_vs_pra.run_powered_decomposition import PersistentMLXBackend
from pra_hf.crossdoc_composition import CrossDocumentResidualAdapterConfig
from pra_hf.crossdoc_mlx_adapter import (
    adapted_crossdoc_memory,
    apply_mlx_crossdoc_residual_adapter,
    create_mlx_crossdoc_residual_adapter,
    mlx_adapter_parameter_count,
    normalized_kv_distillation_loss,
    selective_boundary_reencode_memory,
)
from pra_hf.rag_composition import (
    PositionPolicy,
    RAGPRAProfile,
    SelectedResource,
    compose_resources,
)
from pra_hf.rag_evaluation import (
    ChunkerConfig,
    ContextCondition,
    CrossEncoderRAGSelector,
    FirstStageBM25,
    SelectionReceipt,
    make_candidate_receipt,
    packed_context_from_ranking,
    prepare_candidate_context,
)
from pra_hf.rag_mlx_native import (
    PositionBindingMode,
    encode_native_memory,
    make_native_prompt_cache,
    native_memory_diagnostics,
    rebind_native_memories_global_packed,
    rebind_native_memories_to_receipt,
)


SCHEMA_VERSION = "paper3.2-crossdoc-residual-adapter-v1"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class PreparedExample:
    """Frozen retrieval selection and its packed/independent native tensors."""

    seed: int
    question: object
    candidate_receipt_id: str
    selection_receipt_id: str
    selected_document_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    segments: tuple[tuple[int, ...], ...]
    composition_receipt: object
    teacher_pre: object
    teacher_post: object
    independent_pre: tuple[object, ...]
    teacher_answer_logits: object
    query_tokens: tuple[int, ...]
    answer_tokens: tuple[int, ...]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def _ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return tuple(dict.fromkeys(values))


def _select_disjoint_cohort(
    questions: Sequence[object],
    *,
    count: int,
    seed: int,
    excluded_ids: set[str],
) -> tuple[object, ...]:
    """Select one deterministic cohort without calibration/evaluation leakage."""

    available = tuple(
        question
        for question in questions
        if str(getattr(question, "example_id")) not in excluded_ids
    )
    selected = tuple(select_cohort(available, max_examples=count, seed=seed))
    if len(selected) < count:
        raise ValueError(f"only {len(selected)} disjoint examples remain; requested {count}")
    excluded_ids.update(str(getattr(question, "example_id")) for question in selected)
    return selected


def _shortest_answer_tokens(backend: PersistentMLXBackend, question: object) -> tuple[int, ...]:
    answers = tuple(getattr(question, "answers"))
    encoded = tuple(
        tuple(backend.tokenizer.encode(" " + answer, add_special_tokens=False))
        for answer in answers
    )
    nonempty = tuple(tokens for tokens in encoded if tokens)
    if not nonempty:
        raise ValueError("cross-document training requires a tokenizable answer")
    return min(nonempty, key=len)


def _answer_logits(model: object, memory: object, query_tokens, answer_tokens):
    """Return answer-position logits while preserving gradients to native K/V."""

    import mlx.core as mx

    inputs = tuple(query_tokens) + tuple(answer_tokens[:-1])
    logits = model(
        mx.array(inputs, dtype=mx.int32)[None],
        cache=make_native_prompt_cache(model, memory),
    )
    start = len(query_tokens) - 1
    return logits[0, start : start + len(answer_tokens), :].astype(mx.float32)


def _prepare_example(
    *,
    seed: int,
    question: object,
    dataset_metadata: Mapping[str, object],
    by_id: Mapping[str, object],
    retriever: FirstStageBM25,
    selector: CrossEncoderRAGSelector,
    backend: PersistentMLXBackend,
    revision: str,
    chunker: ChunkerConfig,
    candidate_count: int,
    token_budget: int,
    max_resources: int,
) -> PreparedExample | None:
    candidate = make_candidate_receipt(
        dataset="multihoprag",
        dataset_revision=str(dataset_metadata["dataset_revision"]),
        corpus_revision=str(dataset_metadata["corpus_revision"]),
        corpus_sha256=str(dataset_metadata["corpus_sha256"]),
        question=question,
        retriever=retriever,
        candidate_count=candidate_count,
        chunker=chunker,
        ensure_gold=False,
        seed=seed,
    )
    prepared = prepare_candidate_context(candidate, by_id, token_count=backend.token_count)
    ranking_started = time.perf_counter()
    ranking = selector.rank(getattr(question, "question"), prepared.chunks)
    ranking_ms = (time.perf_counter() - ranking_started) * 1000.0
    context = packed_context_from_ranking(
        condition=ContextCondition.PRA_SELECTED_CONTEXT_NO_ADAPTOR,
        selector_name=selector.name,
        ranked=ranking,
        prepared=prepared,
        token_budget=token_budget,
        selector_latency_ms=ranking_ms,
    )
    selected = tuple(context.chunks[:max_resources])
    if len(selected) < 2:
        return None
    context = replace(
        context,
        chunks=selected,
        packed_tokens=sum(row.chunk.token_count for row in selected),
        candidate_chunks=prepared.chunks,
    )
    selection = SelectionReceipt.from_context(
        candidate_receipt_id=candidate.receipt_id,
        example_id=getattr(question, "example_id"),
        context=context,
        selector_revision=selector.name,
    )
    texts = tuple(row.chunk.text for row in selected)
    segments = _token_segments(backend.tokenizer, texts)
    packed_tokens = tuple(token for segment in segments for token in segment)
    record_ids = tuple(row.chunk.chunk_id for row in selected)
    document_ids = tuple(row.chunk.document_id for row in selected)
    resources = tuple(
        SelectedResource(
            resource_id=row.chunk.chunk_id,
            chunk_id=row.chunk.chunk_id,
            source_sha256=hashlib.sha256(row.chunk.text.encode("utf-8")).hexdigest(),
            source_positions=tuple(range(len(segment))),
            rank=row.rank,
            score=row.score,
        )
        for row, segment in zip(selected, segments)
    )
    composition = compose_resources(
        resources,
        selection_receipt_id=selection.receipt_id,
        profile=RAGPRAProfile.RAG_PLUS_PRA_NATIVE_REBOUND,
        position_policy=PositionPolicy.GLOBAL_PACKED,
        near_gap=0,
    )
    teacher_pre = encode_native_memory(
        backend.model,
        packed_tokens,
        position_binding_mode=PositionBindingMode.PRE_ROPE,
        model_revision=revision,
    )
    teacher_post = rebind_native_memories_global_packed(backend.model, (teacher_pre,))
    independent_pre = tuple(
        encode_native_memory(
            backend.model,
            segment,
            position_binding_mode=PositionBindingMode.PRE_ROPE,
            model_revision=revision,
        )
        for segment in segments
    )
    query_tokens = tuple(
        backend.tokenizer.encode(backend._query(question), add_special_tokens=False)
    )
    answer_tokens = _shortest_answer_tokens(backend, question)
    teacher_logits = _answer_logits(
        backend.model, teacher_post, query_tokens, answer_tokens
    )
    import mlx.core as mx

    teacher_logits = mx.stop_gradient(teacher_logits)
    mx.eval(teacher_logits)
    return PreparedExample(
        seed=seed,
        question=question,
        candidate_receipt_id=candidate.receipt_id,
        selection_receipt_id=selection.receipt_id,
        selected_document_ids=document_ids,
        record_ids=record_ids,
        segments=segments,
        composition_receipt=composition,
        teacher_pre=teacher_pre,
        teacher_post=teacher_post,
        independent_pre=independent_pre,
        teacher_answer_logits=teacher_logits,
        query_tokens=query_tokens,
        answer_tokens=answer_tokens,
    )


def _loss_components(
    adapter: object,
    example: PreparedExample,
    model: object,
    config: CrossDocumentResidualAdapterConfig,
    temperature: float,
):
    import mlx.core as mx

    corrected = apply_mlx_crossdoc_residual_adapter(example.independent_pre, adapter)
    kv_loss = normalized_kv_distillation_loss(corrected, example.teacher_pre)
    memory = rebind_native_memories_to_receipt(
        model, corrected, example.composition_receipt
    )
    logits = _answer_logits(
        model, memory, example.query_tokens, example.answer_tokens
    )
    targets = mx.array(example.answer_tokens, dtype=mx.int32)[:, None]
    selected = mx.take_along_axis(logits, targets, axis=-1).squeeze(-1)
    task_loss = mx.mean(mx.logsumexp(logits, axis=-1) - selected)
    teacher_scaled = example.teacher_answer_logits / temperature
    student_scaled = logits / temperature
    teacher_log_probability = teacher_scaled - mx.logsumexp(
        teacher_scaled, axis=-1, keepdims=True
    )
    teacher_probability = mx.exp(teacher_log_probability)
    student_log_probability = student_scaled - mx.logsumexp(
        student_scaled, axis=-1, keepdims=True
    )
    response_loss = (
        mx.mean(
            mx.sum(
                teacher_probability
                * (teacher_log_probability - student_log_probability),
                axis=-1,
            )
        )
        * temperature**2
    )
    total = (
        config.kv_distillation_weight * kv_loss
        + config.response_distillation_weight * response_loss
        + config.task_loss_weight * task_loss
    )
    return total, kv_loss, response_loss, task_loss


def _train(
    adapter: object,
    examples: Sequence[PreparedExample],
    model: object,
    config: CrossDocumentResidualAdapterConfig,
    *,
    steps: int,
    learning_rate: float,
    temperature: float,
    seed: int,
    checkpoint_every: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    if not examples:
        raise ValueError("adapter training requires calibration examples")
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    order = list(range(len(examples)))
    randomizer = random.Random(seed)
    history: list[dict[str, object]] = []

    def calibration_loss() -> float:
        values = [
            _loss_components(adapter, example, model, config, temperature)[0]
            for example in examples
        ]
        mx.eval(values)
        return statistics.fmean(float(value.item()) for value in values)

    initial_calibration_loss = calibration_loss()
    best_loss = initial_calibration_loss
    best_step = 0
    best_parameters = tree_map(
        lambda value: value + mx.zeros_like(value), adapter.trainable_parameters()
    )
    mx.eval(best_parameters)
    for step in range(steps):
        if step % len(order) == 0:
            randomizer.shuffle(order)
        example = examples[order[step % len(order)]]

        def objective():
            return _loss_components(
                adapter, example, model, config, temperature
            )[0]

        loss_and_grad = nn.value_and_grad(adapter, objective)
        started = time.perf_counter()
        loss, gradients = loss_and_grad()
        optimizer.update(adapter, gradients)
        mx.eval(adapter.parameters(), optimizer.state, loss)
        total, kv_loss, response_loss, task_loss = _loss_components(
            adapter, example, model, config, temperature
        )
        mx.eval(total, kv_loss, response_loss, task_loss)
        row = {
            "step": step + 1,
            "example_id": getattr(example.question, "example_id"),
            "seed": example.seed,
            "total_loss": float(total.item()),
            "kv_distillation_loss": float(kv_loss.item()),
            "response_distillation_loss": float(response_loss.item()),
            "task_loss": float(task_loss.item()),
            "step_ms": (time.perf_counter() - started) * 1000.0,
            "calibration_mean_loss": None,
        }
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
            current_calibration_loss = calibration_loss()
            row["calibration_mean_loss"] = current_calibration_loss
            if current_calibration_loss < best_loss:
                best_loss = current_calibration_loss
                best_step = step + 1
                best_parameters = tree_map(
                    lambda value: value + mx.zeros_like(value),
                    adapter.trainable_parameters(),
                )
                mx.eval(best_parameters)
        history.append(row)
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            print(
                f"[train {step + 1}/{steps}] total={row['total_loss']:.4f} "
                f"kv={row['kv_distillation_loss']:.4f} "
                f"response={row['response_distillation_loss']:.4f} "
                f"task={row['task_loss']:.4f}",
                flush=True,
            )
    final_calibration_loss = calibration_loss()
    adapter.update(best_parameters)
    mx.eval(adapter.parameters())
    return history, {
        "selection_metric": "mean combined calibration objective",
        "initial_calibration_loss": initial_calibration_loss,
        "final_step_calibration_loss": final_calibration_loss,
        "selected_calibration_loss": best_loss,
        "selected_step": best_step,
        "checkpoint_every": checkpoint_every,
    }


def _evaluate_condition(
    *,
    condition: str,
    example: PreparedExample,
    backend: PersistentMLXBackend,
    memory: object,
    reference_logits: object,
    transform_ms: float,
    adapter_parameters: int = 0,
    boundary_receipt: object | None = None,
) -> dict[str, object]:
    prediction, metrics, logits = _execute(backend, example.question, memory)
    distribution = _distribution_diagnostics(reference_logits, logits)
    diagnostics = native_memory_diagnostics(example.teacher_post, memory)
    support = len(
        set(example.selected_document_ids).intersection(
            getattr(example.question, "gold_document_ids")
        )
    ) / max(len(getattr(example.question, "gold_document_ids")), 1)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": example.seed,
        "example_id": getattr(example.question, "example_id"),
        "condition": condition,
        "candidate_receipt_id": example.candidate_receipt_id,
        "selection_receipt_id": example.selection_receipt_id,
        "selected_document_ids": list(example.selected_document_ids),
        "record_ids": list(example.record_ids),
        "supporting_document_coverage": support,
        "physical_native_tokens": memory.source_tokens,
        "adapter_parameters": adapter_parameters,
        "boundary_tokens": (
            boundary_receipt.boundary_tokens if boundary_receipt is not None else 0
        ),
        "boundary_context_native_tokens": (
            boundary_receipt.context_native_tokens if boundary_receipt is not None else 0
        ),
        "reencoded_tokens": (
            boundary_receipt.reencoded_tokens if boundary_receipt is not None else 0
        ),
        "request_transform_ms": transform_ms,
        "prediction": prediction,
        "gold_answers": list(getattr(example.question, "answers")),
        "output_matches_packed": None,
        "kv_rmse": statistics.fmean(
            float(layer["key_rmse"]) for layer in diagnostics["layers"]
        ),
        "value_rmse": statistics.fmean(
            float(layer["value_rmse"]) for layer in diagnostics["layers"]
        ),
        **distribution,
        **metrics,
    }


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _bootstrap_mean_interval(
    values: Sequence[float], *, seed: int, draws: int = 10_000
) -> tuple[float, float]:
    """Deterministic percentile interval over seed-level means."""

    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("bootstrap interval requires values")
    if len(values) == 1:
        return values[0], values[0]
    randomizer = random.Random(seed)
    estimates = sorted(
        statistics.fmean(randomizer.choice(values) for _ in values)
        for _ in range(draws)
    )
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]


def _exact_sign_flip_p(values: Sequence[float]) -> float:
    """Exact two-sided sign-flip test over seed-level paired effects."""

    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("sign-flip test requires values")
    observed = abs(statistics.fmean(values))
    outcomes = tuple(
        abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    return sum(value >= observed - 1e-12 for value in outcomes) / len(outcomes)


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Aggregate quality and composition cost with seeds as replication units."""

    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    conditions = []
    for condition, values in sorted(grouped.items()):
        seed_groups: dict[int, list[Mapping[str, object]]] = {}
        for row in values:
            seed_groups.setdefault(int(row["seed"]), []).append(row)
        seed_f1 = [
            statistics.fmean(float(row["token_f1"]) for row in seed_rows)
            for seed_rows in seed_groups.values()
        ]
        f1_interval = _bootstrap_mean_interval(
            seed_f1, seed=3200 + sum(ord(character) for character in condition)
        )
        conditions.append(
            {
                "condition": condition,
                "examples": len(values),
                "seeds": len(seed_groups),
                "exact_match": _mean([float(row["exact_match"]) for row in values]),
                "token_f1": _mean([float(row["token_f1"]) for row in values]),
                "seed_token_f1_std": statistics.pstdev(seed_f1) if len(seed_f1) > 1 else 0.0,
                "seed_token_f1_ci95": list(f1_interval),
                "gold_answer_mean_nll": _mean(
                    [float(row["gold_answer_mean_nll"]) for row in values]
                ),
                "first_step_js_divergence": _mean(
                    [float(row["first_step_js_divergence"]) for row in values]
                ),
                "output_match_rate": _mean(
                    [float(bool(row["output_matches_packed"])) for row in values]
                ),
                "kv_rmse": _mean([float(row["kv_rmse"]) for row in values]),
                "value_rmse": _mean([float(row["value_rmse"]) for row in values]),
                "reencoded_tokens": _mean(
                    [float(row["reencoded_tokens"]) for row in values]
                ),
                "request_transform_ms": _mean(
                    [float(row["request_transform_ms"]) for row in values]
                ),
                "adapter_parameters": max(int(row["adapter_parameters"]) for row in values),
            }
        )
    by_example: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in rows:
        by_example.setdefault(str(row["example_id"]), {})[str(row["condition"])] = row
    paired = []
    for condition in sorted(grouped):
        if condition == "C_INDEPENDENT_PRA":
            continue
        seed_deltas: dict[int, list[tuple[float, float, float]]] = {}
        for conditions_by_name in by_example.values():
            if condition not in conditions_by_name or "C_INDEPENDENT_PRA" not in conditions_by_name:
                continue
            candidate = conditions_by_name[condition]
            baseline = conditions_by_name["C_INDEPENDENT_PRA"]
            seed_deltas.setdefault(int(candidate["seed"]), []).append(
                (
                    float(candidate["token_f1"]) - float(baseline["token_f1"]),
                    float(candidate["gold_answer_mean_nll"])
                    - float(baseline["gold_answer_mean_nll"]),
                    float(candidate["first_step_js_divergence"])
                    - float(baseline["first_step_js_divergence"]),
                )
            )
        if not seed_deltas:
            continue
        seed_f1_deltas = [
            statistics.fmean(value[0] for value in values)
            for values in seed_deltas.values()
        ]
        seed_nll_deltas = [
            statistics.fmean(value[1] for value in values)
            for values in seed_deltas.values()
        ]
        seed_js_deltas = [
            statistics.fmean(value[2] for value in values)
            for values in seed_deltas.values()
        ]
        f1_interval = _bootstrap_mean_interval(
            seed_f1_deltas,
            seed=6400 + sum(ord(character) for character in condition),
        )
        paired.append(
            {
                "condition": condition,
                "baseline": "C_INDEPENDENT_PRA",
                "seeds": len(seed_deltas),
                "token_f1_delta": statistics.fmean(seed_f1_deltas),
                "token_f1_delta_ci95": list(f1_interval),
                "token_f1_sign_flip_p": _exact_sign_flip_p(seed_f1_deltas),
                "gold_answer_mean_nll_delta": statistics.fmean(seed_nll_deltas),
                "first_step_js_divergence_delta": statistics.fmean(seed_js_deltas),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "conditions": conditions,
        "paired_vs_independent": paired,
    }


def _write_outputs(
    output: Path,
    *,
    rows: Sequence[Mapping[str, object]],
    history: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / "condition_results.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output / "training_history.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = manifest["summary"]
    labels = {
        "A_FULL_CAUSAL_RAG": "Packed RAG",
        "C_INDEPENDENT_PRA": "Independent PRA",
        "Z_ZERO_INIT_RESIDUAL": "Zero-init residual",
        "R_TRAINED_RESIDUAL": "Trained residual",
        "S_BOUNDARY_REENCODE_8": "Boundary re-encode 8",
        "S_BOUNDARY_REENCODE_16": "Boundary re-encode 16",
        "S_BOUNDARY_REENCODE_32": "Boundary re-encode 32",
    }
    table = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & F1 & EM & NLL & JS & Re-enc. & Params \\",
        r"\midrule",
    ]
    for row in summary["conditions"]:
        label = labels.get(row["condition"], row["condition"].replace("_", r"\_"))
        table.append(
            f"{label} & {row['token_f1']:.3f} & {row['exact_match']:.3f} & "
            f"{row['gold_answer_mean_nll']:.3f} & {row['first_step_js_divergence']:.4f} & "
            f"{row['reencoded_tokens']:.1f} & {row['adapter_parameters']:,} \\\\"
        )
    table.extend((r"\bottomrule", r"\end{tabular}"))
    (output / "generated_adapter_table.tex").write_text(
        "\n".join(table) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    condition_rows = summary["conditions"]
    names = [labels.get(row["condition"], row["condition"]) for row in condition_rows]
    f1 = [row["token_f1"] for row in condition_rows]
    nll = [row["gold_answer_mean_nll"] for row in condition_rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(names, f1, color="#287271")
    axes[0].set_ylabel("Token F1")
    axes[0].set_ylim(bottom=0)
    axes[1].bar(names, nll, color="#d97706")
    axes[1].set_ylabel("Gold-answer mean NLL (lower is better)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "crossdoc_adapter_quality.png", dpi=180)
    figure.savefig(output / "crossdoc_adapter_quality.pdf")
    plt.close(figure)

    if history:
        figure, axis = plt.subplots(figsize=(7.5, 4.2))
        axis.plot([row["step"] for row in history], [row["total_loss"] for row in history], label="total")
        axis.plot([row["step"] for row in history], [row["kv_distillation_loss"] for row in history], label="K/V distillation")
        axis.plot([row["step"] for row in history], [row["task_loss"] for row in history], label="task NLL")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "crossdoc_adapter_training.png", dpi=180)
        figure.savefig(output / "crossdoc_adapter_training.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_eval"))
    parser.add_argument("--model", default="mlx-community/Qwen3-1.7B-4bit")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER)
    parser.add_argument("--reranker-revision", default="main")
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=1024)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=32)
    parser.add_argument("--max-resources", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--train-seeds", type=_ints, default=(11, 23, 37))
    parser.add_argument("--eval-seeds", type=_ints, default=(71, 101, 131, 151, 181))
    parser.add_argument("--train-examples-per-seed", type=int, default=4)
    parser.add_argument("--eval-examples-per-seed", type=int, default=6)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--kv-weight", type=float, default=1.0)
    parser.add_argument("--response-weight", type=float, default=0.25)
    parser.add_argument("--task-weight", type=float, default=0.25)
    parser.add_argument("--boundary-windows", type=_ints, default=(8, 16, 32))
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.learning_rate <= 0
        or args.temperature <= 0
        or args.checkpoint_every <= 0
    ):
        parser.error("steps, learning rate, temperature, and checkpoint cadence must be positive")

    started = time.time()
    documents, questions, dataset_metadata = load_multihop_rag(args.cache_dir)
    by_id = {document.document_id: document for document in documents}
    retriever = FirstStageBM25(documents)
    revision = _resolve_hf_revision(args.model, args.revision)
    reranker_revision = _resolve_hf_revision(args.reranker, args.reranker_revision)
    selector = CrossEncoderRAGSelector(
        model_id=args.reranker,
        revision=reranker_revision,
        name_prefix="crossdoc_adapter",
    )
    backend = PersistentMLXBackend(args.model, revision, args.max_new_tokens)
    chunker = ChunkerConfig(args.chunk_tokens, args.chunk_overlap)
    used_ids: set[str] = set()
    train_questions: list[tuple[int, object]] = []
    for seed in args.train_seeds:
        cohort = _select_disjoint_cohort(
            questions,
            count=args.train_examples_per_seed,
            seed=seed,
            excluded_ids=used_ids,
        )
        train_questions.extend((seed, question) for question in cohort)
    eval_questions: list[tuple[int, object]] = []
    for seed in args.eval_seeds:
        cohort = _select_disjoint_cohort(
            questions,
            count=args.eval_examples_per_seed,
            seed=seed,
            excluded_ids=used_ids,
        )
        eval_questions.extend((seed, question) for question in cohort)

    training_examples: list[PreparedExample] = []
    for index, (seed, question) in enumerate(train_questions, 1):
        print(f"[prepare train {index}/{len(train_questions)}] {question.example_id}", flush=True)
        prepared = _prepare_example(
            seed=seed,
            question=question,
            dataset_metadata=dataset_metadata,
            by_id=by_id,
            retriever=retriever,
            selector=selector,
            backend=backend,
            revision=revision,
            chunker=chunker,
            candidate_count=args.candidate_count,
            token_budget=args.token_budget,
            max_resources=args.max_resources,
        )
        if prepared is not None:
            training_examples.append(prepared)
    if not training_examples:
        raise RuntimeError("no multi-record calibration examples were prepared")
    widths = tuple(
        int(layer.keys.shape[1] * layer.keys.shape[-1])
        for layer in training_examples[0].independent_pre[0].layers
    )
    adapter_config = CrossDocumentResidualAdapterConfig(
        rank=args.rank,
        kv_distillation_weight=args.kv_weight,
        response_distillation_weight=args.response_weight,
        task_loss_weight=args.task_weight,
    )
    zero_adapter = create_mlx_crossdoc_residual_adapter(
        widths, adapter_config, seed=args.seed
    )
    adapter = create_mlx_crossdoc_residual_adapter(
        widths, adapter_config, seed=args.seed
    )
    parameter_count = mlx_adapter_parameter_count(adapter)
    history, checkpoint_selection = _train(
        adapter,
        training_examples,
        backend.model,
        adapter_config,
        steps=args.steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    adapter.save_weights(str(args.output / "crossdoc_residual_adapter.safetensors"))

    rows: list[dict[str, object]] = []
    for index, (seed, question) in enumerate(eval_questions, 1):
        print(f"[evaluate {index}/{len(eval_questions)}] {question.example_id}", flush=True)
        example = _prepare_example(
            seed=seed,
            question=question,
            dataset_metadata=dataset_metadata,
            by_id=by_id,
            retriever=retriever,
            selector=selector,
            backend=backend,
            revision=revision,
            chunker=chunker,
            candidate_count=args.candidate_count,
            token_budget=args.token_budget,
            max_resources=args.max_resources,
        )
        if example is None:
            continue
        teacher_row = _evaluate_condition(
            condition="A_FULL_CAUSAL_RAG",
            example=example,
            backend=backend,
            memory=example.teacher_post,
            reference_logits=example.teacher_answer_logits[0],
            transform_ms=0.0,
        )
        teacher_row["output_matches_packed"] = True
        rows.append(teacher_row)
        reference_logits = example.teacher_answer_logits[0]

        started_condition = time.perf_counter()
        independent = rebind_native_memories_to_receipt(
            backend.model, example.independent_pre, example.composition_receipt
        )
        independent_ms = (time.perf_counter() - started_condition) * 1000.0
        condition_memories: list[tuple[str, object, float, int, object | None]] = [
            ("C_INDEPENDENT_PRA", independent, independent_ms, 0, None)
        ]
        started_condition = time.perf_counter()
        zero_memory = adapted_crossdoc_memory(
            backend.model,
            example.independent_pre,
            example.composition_receipt,
            zero_adapter,
        )
        condition_memories.append(
            (
                "Z_ZERO_INIT_RESIDUAL",
                zero_memory,
                (time.perf_counter() - started_condition) * 1000.0,
                parameter_count,
                None,
            )
        )
        started_condition = time.perf_counter()
        trained_memory = adapted_crossdoc_memory(
            backend.model,
            example.independent_pre,
            example.composition_receipt,
            adapter,
        )
        condition_memories.append(
            (
                "R_TRAINED_RESIDUAL",
                trained_memory,
                (time.perf_counter() - started_condition) * 1000.0,
                parameter_count,
                None,
            )
        )
        for window in args.boundary_windows:
            boundary_memory, boundary_receipt = selective_boundary_reencode_memory(
                backend.model,
                example.segments,
                example.independent_pre,
                example.composition_receipt,
                record_ids=example.record_ids,
                boundary_tokens=window,
            )
            condition_memories.append(
                (
                    f"S_BOUNDARY_REENCODE_{window}",
                    boundary_memory,
                    boundary_receipt.request_reencode_ms,
                    0,
                    boundary_receipt,
                )
            )
        for condition, memory, transform_ms, parameters, boundary_receipt in condition_memories:
            row = _evaluate_condition(
                condition=condition,
                example=example,
                backend=backend,
                memory=memory,
                reference_logits=reference_logits,
                transform_ms=transform_ms,
                adapter_parameters=parameters,
                boundary_receipt=boundary_receipt,
            )
            row["output_matches_packed"] = row["prediction"] == teacher_row["prediction"]
            rows.append(row)

    summary = summarize_rows(rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "dataset": dataset_metadata,
        "model_id": args.model,
        "model_revision": revision,
        "reranker_id": args.reranker,
        "reranker_revision": reranker_revision,
        "selection_frozen_across_conditions": True,
        "base_model_frozen": True,
        "calibration_evaluation_disjoint": True,
        "train_seeds": list(args.train_seeds),
        "eval_seeds": list(args.eval_seeds),
        "train_example_ids": [
            getattr(example.question, "example_id") for example in training_examples
        ],
        "eval_example_ids": sorted(
            {str(row["example_id"]) for row in rows}
        ),
        "adapter_config": asdict(adapter_config),
        "adapter_parameters": parameter_count,
        "optimizer": {
            "name": "AdamW",
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
        },
        "checkpoint_selection": checkpoint_selection,
        "boundary_windows": list(args.boundary_windows),
        "candidate_count": args.candidate_count,
        "token_budget": args.token_budget,
        "max_resources": args.max_resources,
        "hardware": _hardware(),
        "runtime_versions": _runtime_versions(),
        "summary": summary,
    }
    _write_outputs(args.output, rows=rows, history=history, manifest=manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
