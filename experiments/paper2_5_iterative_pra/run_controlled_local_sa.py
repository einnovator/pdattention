"""Train and evaluate matched LocalSA controls for the reopened Paper 2.5.

The script is checkpointable and emits machine-readable rows after every model.
It intentionally uses small causal models as scientific instruments: the host
GPU cannot sustain the guide's aspirational 20--100M family across five windows
and five seeds, so dimensions, token count, and approximate compute are explicit
in every artifact.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from pra_torch.config import PRAConfig
from pra_torch.controlled_local_sa import (
    CONTROLLED_PROTOCOL_VERSION,
    ControlledExample,
    ControlledTokenizer,
    collate_controlled,
    controlled_examples,
    last_valid_logits,
)
from pra_torch.masks import causal_attention_mask
from pra_torch.model import PositionAwareTransformerBlock, TinyPRAModel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "docs/papers/shared/results/paper2_5_iterative_pra/controlled_local_sa"
)
WINDOWS: tuple[int | None, ...] = (16, 32, 64, 128, None)
SEEDS = (17, 29, 41, 53, 67)


def window_name(window: int | None) -> str:
    """Return the stable artifact label for one native attention window."""
    return "global" if window is None else f"w{window}"


def set_seed(seed: int) -> None:
    """Seed Python and Torch without enabling slower global determinism modes."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(
    tokenizer: ControlledTokenizer,
    *,
    window: int | None,
    device: str,
    d_model: int,
    n_layers: int,
) -> PRAConfig:
    """Build one matched RoPE decoder; only ``self_attention_window`` varies."""
    return PRAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=4 * d_model,
        max_seq_len=256,
        model_max_context_tokens=256,
        position_encoding="rope",
        self_attention_window=window,
        dropout=0.0,
        model_variant="td_sa",
        device=device,
    )


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    """Write a union-schema CSV, including an empty file for empty stages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _append_or_replace(path: Path, rows: Sequence[dict], keys: Sequence[str]) -> None:
    """Checkpoint tabular work by replacing rows with the same experiment key."""
    existing = _read_csv(path)
    replacement = {tuple(str(row[key]) for key in keys): row for row in rows}
    kept = [
        row
        for row in existing
        if tuple(str(row.get(key, "")) for key in keys) not in replacement
    ]
    _write_csv(path, [*kept, *replacement.values()])


def _batch_order(count: int, batch_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    indices = list(range(count))
    rng.shuffle(indices)
    # Similar lengths reduce padding while shuffled groups preserve composition.
    return [indices[start : start + batch_size] for start in range(0, count, batch_size)]


@torch.no_grad()
def evaluate_model(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    tokenizer: ControlledTokenizer,
    *,
    batch_size: int,
    device: str,
) -> tuple[dict, list[dict]]:
    """Measure answer loss/accuracy overall and by exact path depth."""
    model.eval()
    rows = []
    total_loss = total_correct = total = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, mask, answers = collate_controlled(
            batch, pad_token_id=tokenizer.pad_token_id, device=device
        )
        logits = last_valid_logits(model(input_ids, attention_mask=mask), mask)
        losses = F.cross_entropy(logits, answers, reduction="none")
        predictions = logits.argmax(dim=-1)
        for example, loss, predicted in zip(batch, losses, predictions):
            correct = int(int(predicted) == example.answer_id)
            rows.append(
                {
                    "example_id": example.example_id,
                    "depth": example.depth,
                    "loss": float(loss),
                    "correct": correct,
                    "evidence_distance": example.evidence_distance,
                    "distractor_count": example.distractor_count,
                    "lexical_overlap": example.lexical_overlap,
                    "relation_types": example.relation_types,
                    "branching": example.branching,
                }
            )
            total_loss += float(loss)
            total_correct += correct
            total += 1
    metrics = {
        "eval_loss": total_loss / max(total, 1),
        "eval_perplexity": math.exp(min(total_loss / max(total, 1), 20.0)),
        "eval_accuracy": total_correct / max(total, 1),
        "eval_examples": total,
    }
    for depth in sorted({row["depth"] for row in rows}):
        depth_rows = [row for row in rows if row["depth"] == depth]
        metrics[f"accuracy_depth_{depth}"] = statistics.fmean(
            row["correct"] for row in depth_rows
        )
    return metrics, rows


def train_one(
    *,
    cfg: PRAConfig,
    tokenizer: ControlledTokenizer,
    train_examples: Sequence[ControlledExample],
    validation_examples: Sequence[ControlledExample],
    test_examples: Sequence[ControlledExample],
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    checkpoint: Path,
) -> tuple[TinyPRAModel, dict]:
    """Train one answer-supervised control and persist a resumable checkpoint."""
    set_seed(seed)
    model = TinyPRAModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    completed = 0
    train_tokens = 0
    elapsed_before = 0.0
    losses: list[float] = []
    validation_history: list[dict] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_step = 0
    best_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_best.pt")
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("protocol_version") not in {None, CONTROLLED_PROTOCOL_VERSION}:
            raise ValueError(
                f"Checkpoint protocol {state.get('protocol_version')!r} does not match "
                f"{CONTROLLED_PROTOCOL_VERSION!r}."
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        completed = int(state["step"])
        train_tokens = int(state.get("train_tokens", 0))
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        losses = list(state.get("losses", []))
        validation_history = list(state.get("validation_history", []))
        best_accuracy = float(state.get("best_accuracy", -1.0))
        best_loss = float(state.get("best_loss", float("inf")))
        best_step = int(state.get("best_step", 0))
    invocation_start_tokens = train_tokens
    started = time.perf_counter()
    model.train()
    epoch = 0
    while completed < steps:
        batches = _batch_order(len(train_examples), batch_size, seed + epoch * 7919)
        for indices in batches:
            if completed >= steps:
                break
            batch = [train_examples[index] for index in indices]
            input_ids, mask, answers = collate_controlled(
                batch, pad_token_id=tokenizer.pad_token_id, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = last_valid_logits(model(input_ids, attention_mask=mask), mask)
            loss = F.cross_entropy(logits, answers)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Linear warmup followed by cosine decay keeps all matched runs stable.
            completed += 1
            warmup = min(50, max(steps // 10, 1))
            if completed <= warmup:
                scale = completed / warmup
            else:
                progress = (completed - warmup) / max(steps - warmup, 1)
                scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * scale
            optimizer.step()
            train_tokens += int(mask.sum())
            losses.append(float(loss.detach()))
            if completed % 100 == 0 or completed == steps:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                validation, _ = evaluate_model(
                    model,
                    validation_examples,
                    tokenizer,
                    batch_size=batch_size,
                    device=device,
                )
                validation_row = {"step": completed, **validation}
                validation_history = [
                    row for row in validation_history if int(row["step"]) != completed
                ]
                validation_history.append(validation_row)
                candidate_accuracy = float(validation["eval_accuracy"])
                candidate_loss = float(validation["eval_loss"])
                if candidate_accuracy > best_accuracy or (
                    candidate_accuracy == best_accuracy and candidate_loss < best_loss
                ):
                    best_accuracy = candidate_accuracy
                    best_loss = candidate_loss
                    best_step = completed
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "step": completed,
                            "validation": validation,
                            "config": asdict(cfg),
                            "protocol_version": CONTROLLED_PROTOCOL_VERSION,
                        },
                        best_checkpoint,
                    )
                model.train()
                elapsed_total = elapsed_before + time.perf_counter() - started
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": completed,
                        "train_tokens": train_tokens,
                        "elapsed_seconds": elapsed_total,
                        "losses": losses[-200:],
                        "validation_history": validation_history,
                        "best_accuracy": best_accuracy,
                        "best_loss": best_loss,
                        "best_step": best_step,
                        "config": asdict(cfg),
                        "protocol_version": CONTROLLED_PROTOCOL_VERSION,
                    },
                    checkpoint,
                )
        epoch += 1
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    elapsed_total = elapsed_before + elapsed
    invocation_tokens = train_tokens - invocation_start_tokens
    if not best_checkpoint.exists():
        raise RuntimeError("Training completed without a validation-selected checkpoint.")
    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    test, _ = evaluate_model(
        model,
        test_examples,
        tokenizer,
        batch_size=batch_size,
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    mean_tokens = train_tokens / max(completed, 1)
    # Dense executed-FLOP estimate is reported, not presented as profiler output.
    approximate_flops_per_step = 6 * parameter_count * mean_tokens
    metrics = {
        **{key.replace("eval_", "test_"): value for key, value in test.items()},
        "step": completed,
        "selected_step": best_step,
        "selected_validation_accuracy": best_accuracy,
        "selected_validation_loss": best_loss,
        "validation_history_json": json.dumps(validation_history, sort_keys=True),
        "train_tokens": train_tokens,
        "train_loss_last_50": statistics.fmean(losses[-50:]),
        "parameter_count": parameter_count,
        "elapsed_seconds_this_invocation": elapsed,
        "elapsed_seconds_total": elapsed_total,
        "tokens_per_second_this_invocation": invocation_tokens / max(elapsed, 1e-9),
        "tokens_per_second_total": train_tokens / max(elapsed_total, 1e-9),
        "approximate_flops_per_step": approximate_flops_per_step,
        "approximate_total_flops": approximate_flops_per_step * completed,
    }
    return model.eval(), metrics


def _reference_spans(example: ControlledExample) -> dict[str, tuple[int, int]]:
    """Recover fact spans from the benchmark's exact source serialization."""
    cursor = 1
    spans = {}
    for ref in example.references:
        spans[ref.uri] = (cursor, cursor + len(ref.token_ids))
        cursor += len(ref.token_ids) + example.evidence_gap
    return spans


@torch.no_grad()
def _block_output_trace(
    model: TinyPRAModel,
    ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Capture final-token block outputs for restricted-context dependence."""
    outputs: list[torch.Tensor] = []
    hooks = [
        block.register_forward_hook(
            lambda _module, _inputs, output: outputs.append(output[:, -1].detach().clone())
        )
        for block in model.blocks
    ]
    try:
        model(ids, use_pra_memory=False, attention_mask=attention_mask)
    finally:
        for hook in hooks:
            hook.remove()
    return outputs


@torch.no_grad()
def topology_rows(
    model: TinyPRAModel,
    examples: Sequence[ControlledExample],
    *,
    window: int | None,
    seed: int,
    device: str,
) -> tuple[list[dict], list[dict]]:
    """Measure native Q/K edge ranks and layerwise state transformation."""
    model.eval()
    edges: list[dict] = []
    context_rows: list[dict] = []
    for example in examples:
        ids = torch.tensor([example.full_input_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(ids)
        restriction = window if window is not None else min(32, ids.shape[1])
        restricted_mask = mask.clone()
        restricted_mask[:, : max(ids.shape[1] - restriction, 0)] = 0
        restricted_trace = _block_output_trace(model, ids, restricted_mask)
        positions = torch.arange(ids.shape[1], device=device)
        x = model.position_encoding.apply_embeddings(model.token_emb(ids), positions, model.pos_emb)
        spans = _reference_spans(example)
        by_hop = {ref.hop: ref for ref in example.references if ref.is_evidence}
        for layer_id, block in enumerate(model.blocks):
            if not isinstance(block, PositionAwareTransformerBlock):
                raise TypeError("Controlled topology tracing requires RoPE position-aware blocks.")
            before = x
            norm = block.ln1(x)
            attention = block.attn
            query = attention._split_heads(attention.q_proj(norm))
            key = attention._split_heads(attention.k_proj(norm))
            value = attention._split_heads(attention.v_proj(norm))
            query, key = attention.position_encoding.transform_qk(query, key, positions)
            scores = query @ key.transpose(-2, -1) / math.sqrt(attention.head_dim)
            hidden = causal_attention_mask(ids.shape[1], device, window=window)
            scores = scores.masked_fill(hidden[None, None], float("-inf"))
            weights = F.softmax(scores, dim=-1)
            attention_output = attention.o_proj(attention._merge_heads(weights @ value))
            after_attention = before + block.residual_dropout(attention_output)
            ffn_output = block.residual_dropout(block.ff(block.ln2(after_attention)))
            x = after_attention + ffn_output

            final_weights = weights[0, :, -1, :]
            entropy = -(final_weights.clamp_min(1e-12).log() * final_weights).sum(dim=-1).mean()
            cosine = F.cosine_similarity(before[:, -1], after_attention[:, -1]).mean()
            context_rows.append(
                {
                    "window": window_name(window),
                    "seed": seed,
                    "example_id": example.example_id,
                    "depth": example.depth,
                    "layer_id": layer_id,
                    "attention_contribution_ratio": float(
                        attention_output[:, -1].norm() / before[:, -1].norm().clamp_min(1e-12)
                    ),
                    "post_attention_cosine": float(cosine),
                    "post_attention_displacement": float(1.0 - cosine),
                    "attention_entropy": float(entropy),
                    "effective_support": float(entropy.exp()),
                    "ffn_magnitude_ratio": float(
                        ffn_output[:, -1].norm() / after_attention[:, -1].norm().clamp_min(1e-12)
                    ),
                    "restricted_context_window": restriction,
                    "restricted_context_dependence": float(
                        1.0
                        - F.cosine_similarity(
                            x[:, -1], restricted_trace[layer_id], dim=-1
                        ).mean()
                    ),
                    "theoretical_receptive_field": (
                        ids.shape[1]
                        if window is None
                        else min(ids.shape[1], 1 + (layer_id + 1) * (window - 1))
                    ),
                }
            )

            for hop in range(example.depth - 1):
                source_ref = by_hop[hop]
                target_ref = by_hop[hop + 1]
                source_span = spans[source_ref.uri]
                source_position = source_span[1] - 1
                source_query = query[0, :, source_position, :].reshape(-1)
                ranked = []
                for candidate in example.references:
                    if candidate.uri == source_ref.uri:
                        continue
                    start, end = spans[candidate.uri]
                    candidate_key = key[0, :, start:end, :].mean(dim=1).reshape(-1)
                    score = F.cosine_similarity(source_query, candidate_key, dim=0)
                    ranked.append((float(score), candidate.uri))
                ranked.sort(key=lambda item: (-item[0], item[1]))
                rank = next(
                    (
                        index
                        for index, (_score, uri) in enumerate(ranked, start=1)
                        if uri == target_ref.uri
                    ),
                    None,
                )
                shortcut = any(
                    uri == by_hop[later].uri
                    for _score, uri in ranked[: max((rank or 1) - 1, 0)]
                    for later in range(hop + 2, example.depth)
                )
                edges.append(
                    {
                        "window": window_name(window),
                        "seed": seed,
                        "example_id": example.example_id,
                        "depth": example.depth,
                        "layer_id": layer_id,
                        "source_hop": hop,
                        "target_hop": hop + 1,
                        "target_rank": rank if rank is not None else 0,
                        "target_reachable": int(rank is not None),
                        "target_within_direct_window": int(
                            target_ref.uri in spans
                            and spans[target_ref.uri][1] <= source_position + 1
                            and (
                                window is None
                                or spans[target_ref.uri][1]
                                > source_position - window + 1
                            )
                        ),
                        "target_within_effective_receptive_field": int(
                            window is None
                            or source_position - spans[target_ref.uri][1] + 1
                            <= (layer_id + 1) * (window - 1)
                        ),
                        "reciprocal_rank": 1.0 / rank if rank is not None else 0.0,
                        **{
                            f"recall_at_{cutoff}": int(
                                rank is not None and rank <= cutoff
                            )
                            for cutoff in (1, 2, 4, 6, 8)
                        },
                        "shortcut": int(shortcut),
                        "candidate_count": len(ranked),
                        "graph_density_at_4": min(4, len(ranked)) / max(len(ranked), 1),
                    }
                )
    return edges, context_rows


def aggregate_topology(rows: Sequence[dict]) -> list[dict]:
    """Aggregate edge metrics and path survival by window, seed, and layer."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["window"], row["seed"], row["layer_id"]), []).append(row)
    output = []
    for (window, seed, layer_id), group in sorted(groups.items(), key=str):
        by_example: dict[str, list[dict]] = {}
        for row in group:
            by_example.setdefault(row["example_id"], []).append(row)
        output.append(
            {
                "window": window,
                "seed": seed,
                "layer_id": layer_id,
                "edge_count": len(group),
                "mrr": statistics.fmean(row["reciprocal_rank"] for row in group),
                **{
                    f"edge_recall_at_{cutoff}": statistics.fmean(
                        row[f"recall_at_{cutoff}"] for row in group
                    )
                    for cutoff in (1, 2, 4, 6, 8)
                },
                "complete_path_survival_at_4": statistics.fmean(
                    all(edge["recall_at_4"] for edge in example_rows)
                    for example_rows in by_example.values()
                ),
                "shortcut_rate": statistics.fmean(row["shortcut"] for row in group),
                "unreachable_at_4": statistics.fmean(
                    not all(edge["recall_at_4"] for edge in example_rows)
                    for example_rows in by_example.values()
                ),
                "graph_density_at_4": statistics.fmean(row["graph_density_at_4"] for row in group),
            }
        )
    return output


def parse_windows(values: str) -> list[int | None]:
    """Parse comma-delimited finite windows plus the literal ``global``."""
    output = []
    for value in values.split(","):
        value = value.strip().lower()
        output.append(None if value == "global" else int(value))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--windows", default="16,32,64,128,global")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--test-examples", type=int, default=512)
    parser.add_argument("--topology-examples", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ControlledTokenizer()
    windows = parse_windows(args.windows)
    seeds = [int(value) for value in args.seeds.split(",")]
    configs = []
    for window in windows:
        cfg = model_config(
            tokenizer,
            window=window,
            device=args.device,
            d_model=args.d_model,
            n_layers=args.layers,
        )
        configs.append({"window": window_name(window), **asdict(cfg)})
    (args.output_dir / "controlled_model_configs.json").write_text(
        json.dumps(
            {
                "protocol": "matched answer-supervised randomized associative chains",
                "protocol_version": CONTROLLED_PROTOCOL_VERSION,
                "materialization_policy": {
                    "selected_chunks": "one five-token fact per selected URI",
                    "whole_parent_control": "all tokens of the selected fact URI",
                    "fixed_native_kv_budget_tokens": 20,
                    "external_materialization_dependency": False,
                },
                "seeds": seeds,
                "corpus_seeds": {
                    "train": 100001,
                    "validation": 100002,
                    "topology": 100003,
                    "pra": 100004,
                    "test": 100005,
                },
                "configs": configs,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for window in windows:
        for seed in seeds:
            cfg = model_config(
                tokenizer,
                window=window,
                device=args.device,
                d_model=args.d_model,
                n_layers=args.layers,
            )
            train_data = controlled_examples(tokenizer, count=args.train_examples, seed=100_001)
            validation = controlled_examples(
                tokenizer, count=args.validation_examples, seed=100_002
            )
            test_data = controlled_examples(
                tokenizer, count=args.test_examples, seed=100_005
            )
            checkpoint = args.output_dir / "checkpoints" / f"{window_name(window)}_seed{seed}.pt"
            if args.skip_training:
                state = torch.load(checkpoint, map_location=args.device, weights_only=False)
                model = TinyPRAModel(cfg).to(args.device)
                best_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_best.pt")
                selected = torch.load(
                    best_checkpoint if best_checkpoint.exists() else checkpoint,
                    map_location=args.device,
                    weights_only=False,
                )
                model.load_state_dict(selected["model"])
                metrics, _ = evaluate_model(
                    model,
                    test_data,
                    tokenizer,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                metrics.update(
                    step=state["step"],
                    selected_step=selected.get("step", state["step"]),
                    train_tokens=state.get("train_tokens", 0),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                )
                metrics = {
                    key.replace("eval_", "test_"): value
                    for key, value in metrics.items()
                }
            else:
                model, metrics = train_one(
                    cfg=cfg,
                    tokenizer=tokenizer,
                    train_examples=train_data,
                    validation_examples=validation,
                    test_examples=test_data,
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    checkpoint=checkpoint,
                )
            training_row = {
                "window": window_name(window),
                "seed": seed,
                "device": args.device,
                "steps_requested": args.steps,
                **metrics,
            }
            _append_or_replace(
                args.output_dir / "training_runs.csv",
                [training_row],
                ("window", "seed"),
            )
            topology_data = controlled_examples(
                tokenizer,
                count=args.topology_examples,
                seed=100_003,
                depths=(2, 3, 4, 8),
                distractors=(4, 8),
                evidence_gaps=(0, 2, 6),
            )
            edge_rows, context_rows = topology_rows(
                model,
                topology_data,
                window=window,
                seed=seed,
                device=args.device,
            )
            _append_or_replace(
                args.output_dir / "receptive_field_topology_rows.csv",
                edge_rows,
                ("window", "seed", "example_id", "layer_id", "source_hop"),
            )
            _append_or_replace(
                args.output_dir / "layer_contextualization_by_window.csv",
                context_rows,
                ("window", "seed", "example_id", "layer_id"),
            )
            aggregate = aggregate_topology(edge_rows)
            _append_or_replace(
                args.output_dir / "receptive_field_topology.csv",
                aggregate,
                ("window", "seed", "layer_id"),
            )
            print(
                f"completed {window_name(window)} seed={seed}: "
                f"accuracy={float(metrics['test_accuracy']):.4f}"
            )


if __name__ == "__main__":
    main()
